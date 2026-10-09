# # MantisWrappers
#
# Wraps the Mantis FEM library for particle-to-B-spline projection, entropy
# computation, and Landau collision operators on a 2D Cartesian velocity
# domain.
#
# Refactored to hold all FEM state in a single `Workspace` struct constructed
# from a `SimParameters` instance. No module-level globals depend on
# parameters.

module MantisWrappers

using Mantis
using LinearAlgebra
using LinearAlgebra: mul!, lu

# `SimParameters` is defined in Parameters.jl, included by main.jl
# *before* this module is loaded. We forward-declare via a parameter alias so
# Workspace can reference the type without re-including the file.
include("Parameters.jl")

# ## Bézier-extraction scratch buffers (per dimension, sized by P_DEG)
struct ParticleBuf
    B::Vector{Float64}
    dB::Vector{Float64}
    phi::Vector{Float64}
    dphi::Vector{Float64}
end
function ParticleBuf(p_deg::Int)
    ParticleBuf(zeros(p_deg+1), zeros(p_deg+1),
        zeros(p_deg+1), zeros(p_deg+1))
end

# ## Workspace
#
# Holds geometry, function spaces, mass-matrix LU, quadrature rule, Bézier
# extraction caches, and per-particle / per-DOF scratch arrays. One instance
# is built per simulation run.
struct Workspace{G, X, L, Q, E1, E2}
    # Parameters echoed for convenience (so functions don't need both args)
    p::SimParameters

    # Anisotropic breakpoint vectors (length n_elem_d + 1)
    bp1::Vector{Float64}
    bp2::Vector{Float64}

    # Function-space objects (parametric to preserve concrete types)
    geo_2d::G
    X⁰::X
    n_dofs::Int
    n_elements::Int
    lin_indices::LinearIndices{2, Tuple{Base.OneTo{Int}, Base.OneTo{Int}}}

    # Mass-matrix LU factorization
    M_lu::L

    # Quadrature rule for entropy / r-vector integration
    qrule_integrate::Q

    # Bézier extraction cache, per dimension (concrete element type via E1/E2)
    ext1d_1::Vector{E1}
    ext1d_2::Vector{E2}
    basis_start_1d_1::Vector{Int}
    basis_start_1d_2::Vector{Int}
    n_dofs_1d_1::Int
    n_dofs_1d_2::Int
    lin_dofs_2d::LinearIndices{2, Tuple{Base.OneTo{Int}, Base.OneTo{Int}}}

    # Per-particle scratch (single-threaded — locate_particle / l2_project /
    # compute_G! all run serial; compute_collision! threads but doesn't touch
    # the spline machinery)
    pbuf1::ParticleBuf
    pbuf2::ParticleBuf
    lp_vals::Vector{Float64}
    lp_gids::Vector{Int}
    G_vals::Vector{Float64}
    G_dxi1::Vector{Float64}
    G_dxi2::Vector{Float64}
    G_gids::Vector{Int}
end

"""
    build_workspace(p::SimParameters)

Construct geometry, function space, mass matrix, quadrature, and all
preallocated scratch from the configuration `p`. Each velocity dimension uses
its own breakpoint vector supplied by `p.bp1`, `p.bp2` (anisotropic, possibly
non-uniform). The breakpoints are aliased into the workspace without copying.
"""
function build_workspace(p::SimParameters)
    bp1 = p.bp1
    bp2 = p.bp2

    geo_1d_1 = Geometry.CartesianGeometry((bp1,))
    geo_1d_2 = Geometry.CartesianGeometry((bp2,))
    B_1d_1 = FunctionSpaces.BSplineSpace(geo_1d_1, p.P_DEG, p.K_REG)
    B_1d_2 = FunctionSpaces.BSplineSpace(geo_1d_2, p.P_DEG, p.K_REG)
    TP = FunctionSpaces.TensorProductSpace((B_1d_1, B_1d_2),
        Geometry.CartesianGeometry)
    X⁰ = Forms.FormSpace(0, TP, "f")

    n_dofs = Forms.get_num_basis(X⁰)
    geo_2d = Forms.get_geometry(X⁰)
    n_elements = Geometry.get_num_elements(geo_2d)
    lin_indices = LinearIndices((length(bp1)-1, length(bp2)-1))

    # Mass matrix M_{ij} = ∫ φᵢ φⱼ dv
    M_lu = let
        qrule = Quadrature.tensor_product_rule((p.P_DEG + 1, p.P_DEG + 1),
            Quadrature.gauss_legendre)
        dΩ = Quadrature.StandardQuadrature(qrule, n_elements)
        f_zero = Forms.AnalyticalFormField(0, x -> [zeros(size(x, 1))], geo_2d, "0")
        wfi = Assemblers.WeakFormInputs(X⁰, f_zero)
        v⁰ = Assemblers.get_test_form(wfi)
        u⁰ = Assemblers.get_trial_form(wfi)
        M_expr = ∫(v⁰ ∧ ★(u⁰), dΩ)
        M_wf = Assemblers.WeakForm(((M_expr,),), ((0,),), wfi)
        M, _ = Assemblers.assemble(M_wf)
        lu(M)
    end

    qrule_integrate = Quadrature.tensor_product_rule((p.N_QUAD, p.N_QUAD),
        Quadrature.gauss_legendre)

    # Bézier extraction per 1D element, per dimension
    n_elem_1 = length(bp1) - 1
    n_elem_2 = length(bp2) - 1
    ext1d_1 = [FunctionSpaces.get_extraction(B_1d_1, e, 1)[1] for e in 1:n_elem_1]
    ext1d_2 = [FunctionSpaces.get_extraction(B_1d_2, e, 1)[1] for e in 1:n_elem_2]
    basis_start_1d_1 = [first(FunctionSpaces.get_basis_indices(B_1d_1, e))
                        for e in 1:n_elem_1]
    basis_start_1d_2 = [first(FunctionSpaces.get_basis_indices(B_1d_2, e))
                        for e in 1:n_elem_2]
    n_dofs_1d_1 = FunctionSpaces.get_num_basis(B_1d_1)
    n_dofs_1d_2 = FunctionSpaces.get_num_basis(B_1d_2)
    lin_dofs_2d = LinearIndices((n_dofs_1d_1, n_dofs_1d_2))

    # Scratch
    pbuf1 = ParticleBuf(p.P_DEG)
    pbuf2 = ParticleBuf(p.P_DEG)
    nloc = (p.P_DEG + 1)^2
    lp_vals = zeros(nloc)
    lp_gids = zeros(Int, nloc)
    G_vals = zeros(nloc)
    G_dxi1 = zeros(nloc)
    G_dxi2 = zeros(nloc)
    G_gids = zeros(Int, nloc)

    return Workspace(p, bp1, bp2,
        geo_2d, X⁰, n_dofs, n_elements, lin_indices,
        M_lu, qrule_integrate,
        ext1d_1, ext1d_2,
        basis_start_1d_1, basis_start_1d_2,
        n_dofs_1d_1, n_dofs_1d_2, lin_dofs_2d,
        pbuf1, pbuf2,
        lp_vals, lp_gids,
        G_vals, G_dxi1, G_dxi2, G_gids)
end

# ## Particle location
struct ParticleLocation
    elem_id::Int
    xi::Points.CartesianPoints
    h1::Float64
    h2::Float64
end

function locate_particle(ws::Workspace, v1, v2)
    (v1 <= ws.bp1[1] || v1 >= ws.bp1[end] ||
     v2 <= ws.bp2[1] || v2 >= ws.bp2[end]) && return nothing
    i = searchsortedlast(ws.bp1, v1)
    j = searchsortedlast(ws.bp2, v2)
    h1 = ws.bp1[i + 1] - ws.bp1[i]
    h2 = ws.bp2[j + 1] - ws.bp2[j]
    ξ1 = (v1 - ws.bp1[i]) / h1
    ξ2 = (v2 - ws.bp2[j]) / h2
    return ParticleLocation(ws.lin_indices[i, j],
        Points.CartesianPoints(([ξ1], [ξ2])), h1, h2)
end

# ## Evaluation wrappers
function evaluate(ws::Workspace, ff::Forms.AbstractFormField, e::Int)
    Forms.evaluate(ff, e, ws.qrule_integrate.nodes)
end
function evaluate(ws::Workspace, fs::Forms.AbstractFormSpace, elem_id::Int)
    Forms.evaluate(fs, elem_id, ws.qrule_integrate.nodes)
end
evaluate(ws::Workspace, e::Int) = evaluate(ws, ws.X⁰, e)
function evaluate(ws::Workspace, fs::Forms.AbstractFormSpace, loc::ParticleLocation)
    Forms.evaluate(fs, loc.elem_id, loc.xi)
end
function evaluate(ws::Workspace, ff::Forms.AbstractFormField, loc::ParticleLocation)
    Forms.evaluate(ff, loc.elem_id, loc.xi)
end
evaluate(ws::Workspace, loc::ParticleLocation) = evaluate(ws, ws.X⁰, loc)

build_field(ws::Workspace, coeffs) = Forms.build_form_field(ws.X⁰, coeffs)
element_measure(ws::Workspace, e) = Geometry.get_element_measure(ws.geo_2d, e)

# ## Bernstein basis evaluation (degree p, point ξ ∈ [0,1])
function _bernstein_eval!(B::AbstractVector, dB::AbstractVector, p::Int, ξ::Float64)
    one_minus = 1.0 - ξ
    @inbounds for i in 0:p
        B[i + 1] = binomial(p, i) * ξ^i * one_minus^(p-i)
    end
    if p == 0
        fill!(dB, 0.0)
    else
        @inbounds for i in 0:p
            bm1_left = i > 0 ? binomial(p-1, i-1) * ξ^(i-1) * one_minus^(p-i) : 0.0
            bm1_right = i < p ? binomial(p-1, i) * ξ^i * one_minus^(p-1-i) : 0.0
            dB[i + 1] = p * (bm1_left - bm1_right)
        end
    end
    return nothing
end

# Fill `vals` and `gids` with φ_{j1,j2}(ξ) and global DOF indices on `loc`.
function fast_eval_particle!(ws::Workspace,
        vals::AbstractVector, gids::AbstractVector,
        loc::ParticleLocation)
    p_deg = ws.p.P_DEG
    ξ = loc.xi[1]
    ξ1 = ξ[1]
    ξ2 = ξ[2]
    ci = CartesianIndices(ws.lin_indices)[loc.elem_id]
    i = ci[1]
    j = ci[2]

    C1 = ws.ext1d_1[i]
    C2 = ws.ext1d_2[j]
    s1 = ws.basis_start_1d_1[i]
    s2 = ws.basis_start_1d_2[j]

    _bernstein_eval!(ws.pbuf1.B, ws.pbuf1.dB, p_deg, ξ1)
    _bernstein_eval!(ws.pbuf2.B, ws.pbuf2.dB, p_deg, ξ2)
    mul!(ws.pbuf1.phi, transpose(C1), ws.pbuf1.B)
    mul!(ws.pbuf2.phi, transpose(C2), ws.pbuf2.B)

    p1 = p_deg + 1
    k = 0
    @inbounds for j2 in 1:p1, j1 in 1:p1

        k += 1
        vals[k] = ws.pbuf1.phi[j1] * ws.pbuf2.phi[j2]
        gids[k] = ws.lin_dofs_2d[s1 + j1 - 1, s2 + j2 - 1]
    end
    return k
end

# Values + canonical-coord gradients ∂φ/∂ξ_d. Physical: ∂/∂v_d = (∂/∂ξ_d) / h_d.
function fast_eval_particle_grad!(ws::Workspace,
        vals::AbstractVector,
        dvals_dxi1::AbstractVector,
        dvals_dxi2::AbstractVector,
        gids::AbstractVector,
        loc::ParticleLocation)
    p_deg = ws.p.P_DEG
    ξ = loc.xi[1]
    ξ1 = ξ[1]
    ξ2 = ξ[2]
    ci = CartesianIndices(ws.lin_indices)[loc.elem_id]
    i = ci[1]
    j = ci[2]

    C1 = ws.ext1d_1[i]
    C2 = ws.ext1d_2[j]
    s1 = ws.basis_start_1d_1[i]
    s2 = ws.basis_start_1d_2[j]

    _bernstein_eval!(ws.pbuf1.B, ws.pbuf1.dB, p_deg, ξ1)
    _bernstein_eval!(ws.pbuf2.B, ws.pbuf2.dB, p_deg, ξ2)
    mul!(ws.pbuf1.phi, transpose(C1), ws.pbuf1.B)
    mul!(ws.pbuf1.dphi, transpose(C1), ws.pbuf1.dB)
    mul!(ws.pbuf2.phi, transpose(C2), ws.pbuf2.B)
    mul!(ws.pbuf2.dphi, transpose(C2), ws.pbuf2.dB)

    p1 = p_deg + 1
    k = 0
    @inbounds for j2 in 1:p1, j1 in 1:p1

        k += 1
        vals[k] = ws.pbuf1.phi[j1] * ws.pbuf2.phi[j2]
        dvals_dxi1[k] = ws.pbuf1.dphi[j1] * ws.pbuf2.phi[j2]
        dvals_dxi2[k] = ws.pbuf1.phi[j1] * ws.pbuf2.dphi[j2]
        gids[k] = ws.lin_dofs_2d[s1 + j1 - 1, s2 + j2 - 1]
    end
    return k
end

# Sample f_s on a 2D Cartesian grid (e.g. for heatmap rendering). Points
# outside the domain → 0.
function evaluate_on_grid(ws::Workspace, field::Forms.FormField,
        v1_grid::AbstractVector, v2_grid::AbstractVector)
    nv1 = length(v1_grid)
    nv2 = length(v2_grid)
    F = zeros(nv1, nv2)
    for (j, v2) in enumerate(v2_grid)
        for (i, v1) in enumerate(v1_grid)
            loc = locate_particle(ws, v1, v2)
            isnothing(loc) && continue
            fv, _ = evaluate(ws, field, loc)
            F[i, j] = fv[1][1]
        end
    end
    return F
end

export SimParameters, parse_overrides, print_summary
export Workspace, build_workspace
export ParticleLocation, locate_particle, evaluate, build_field, element_measure
export fast_eval_particle!, fast_eval_particle_grad!
export evaluate_on_grid

# Physics routines (l2_project!, compute_entropy, compute_r!, compute_G!,
# compute_collision!) plus diagnostics (compute_negative_part,
# compute_fp_minus_fs_l2, compute_momentum, compute_energy). All take
# `ws::Workspace` as first argument except the particle moments, which are
# mesh-independent.
include("functions.jl")

export compute_entropy, compute_r!, compute_G!, compute_collision!, compute_collision_metric!, l2_project!
export eval_loggrad_at_particles!, compute_moments, compute_drift_multipliers,
       compute_LB_velocity!
export compute_negative_part_l1, compute_fs_minus_fp_l2
export compute_momentum, compute_energy
export USE_LOGSQ

end # module
