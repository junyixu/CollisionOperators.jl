# Physics + diagnostics routines. All take `ws::Workspace` as first argument.

# Toggle for the log-square identity  log f = ½ log f².  When true, the entropy
# and entropy-gradient-seed integrands use ½·log(f²), so quadrature points where
# the projected f_s is slightly NEGATIVE (Gibbs undershoot) still contribute
# (guard on |f|, i.e. f² > floor) instead of being clamped to zero by the
# `f_val > 1e-30` positivity test. Default false = original clamped behavior.
# Set once per run from `PARAMS.use_logsq` in `run_simulation`.
const USE_LOGSQ = Ref(false)

# ##############################################################################
# Lenard-Bernstein (LB) collision operator — 2D conservative form.
# Ported from the LB branch (Jeyakumar et al. 2024, generalised to 2D velocity
# space). Selected at runtime by `collision_model == :lb`; shares the identical
# FEM scaffolding as the Landau path (same mesh, projection, Anderson solver),
# differing only in the per-iteration RHS assembled here:
#   drift  v̇_α = -ν ( ∇f_s/f_s + A + B v_α )
# with A ∈ ℝ² (momentum multipliers) and B ∈ ℝ (energy multiplier) solved so the
# discrete momentum Σ w_α v̇_α and energy Σ w_α v_α·v̇_α are conserved exactly.
# ##############################################################################

# Floor on f_s used in 1/f_s evaluation; f_s = 0 is a hard singularity.
const FS_FLOOR = 1e-30

# Limiter on the per-component log-density gradient g = ∇f_s/f_s. The physical
# |∇log f| for a Gaussian is bounded, but an L²-spline Gibbs undershoot can drive
# f_s(v_α) → 0⁺ and make the raw ratio explode (~1e30), blowing up the implicit
# solve. Clamping g to ±G_MAX removes that singularity without distorting the
# bulk; conservation is still exact because the multipliers are solved from the
# (clamped) g.
const G_MAX = 100.0

# Direct log-density gradient g_α = ∇f_s(v_α)/f_s(v_α) at each particle from FE
# coefficients `f_coeffs` (LB plain-midpoint path — no mass-matrix projection).
# Each component clamped to ±G_MAX; out-of-domain particles → g = 0 (drift
# reduces to A + B v, keeping the moment-conservation algebra consistent).
function eval_loggrad_at_particles!(ws::Workspace, g::AbstractMatrix,
        v_parts, f_coeffs)
    nloc = (ws.p.P_DEG + 1)^2
    @inbounds for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        if isnothing(loc)
            g[α, 1] = 0.0
            g[α, 2] = 0.0
            continue
        end
        fast_eval_particle_grad!(ws, ws.G_vals, ws.G_dxi1, ws.G_dxi2, ws.G_gids, loc)
        inv_h1 = 1.0 / loc.h1
        inv_h2 = 1.0 / loc.h2
        f = 0.0
        d1 = 0.0
        d2 = 0.0
        for j in 1:nloc
            c = f_coeffs[ws.G_gids[j]]
            f += c * ws.G_vals[j]
            d1 += c * ws.G_dxi1[j] * inv_h1
            d2 += c * ws.G_dxi2[j] * inv_h2
        end
        invf = 1.0 / max(abs(f), FS_FLOOR)
        g[α, 1] = clamp(d1 * invf, -G_MAX, G_MAX)
        g[α, 2] = clamp(d2 * invf, -G_MAX, G_MAX)
    end
    return nothing
end

# Raw weighted particle moments: n = Σw, U = Σw v, Q = Σw|v|². Feeds the 3×3
# drift-multiplier solve.
function compute_moments(v_parts, w_parts)
    n = 0.0
    U1 = 0.0
    U2 = 0.0
    Q = 0.0
    @inbounds for α in axes(v_parts, 1)
        w = w_parts[α]
        a = v_parts[α, 1]
        b = v_parts[α, 2]
        n += w
        U1 += w * a
        U2 += w * b
        Q += w * (a^2 + b^2)
    end
    return n, U1, U2, Q
end

# Drift multipliers A = (A1, A2) and B: solve the 3×3 symmetric system enforcing
# exact momentum + energy conservation
#   [ n  0  U1 ][A1]   [ Sg1 ]
#   [ 0  n  U2 ][A2] =-[ Sg2 ]
#   [ U1 U2 Q  ][B ]   [ P   ]
# with Sg = Σ w_α g_α, P = Σ w_α v_α·g_α.
function compute_drift_multipliers(v_parts, w_parts, g, n, U1, U2, Q)
    Sg1 = 0.0
    Sg2 = 0.0
    P = 0.0
    @inbounds for α in axes(v_parts, 1)
        g1 = g[α, 1]
        g2 = g[α, 2]
        w = w_parts[α]
        Sg1 += w * g1
        Sg2 += w * g2
        P += w * (v_parts[α, 1] * g1 + v_parts[α, 2] * g2)
    end
    Mmat = [n 0.0 U1; 0.0 n U2; U1 U2 Q]
    rhs = [-Sg1, -Sg2, -P]
    sol = Mmat \ rhs
    return sol[1], sol[2], sol[3]   # A1, A2, B
end

# LB velocity update v̇_α = -ν ( g_α + A + B v_α ), writing into `dot_v`.
function compute_LB_velocity!(dot_v, v_parts, g, A1, A2, B, ν)
    @inbounds for α in axes(v_parts, 1)
        dot_v[α, 1] = -ν * (g[α, 1] + A1 + B * v_parts[α, 1])
        dot_v[α, 2] = -ν * (g[α, 2] + A2 + B * v_parts[α, 2])
    end
    return nothing
end

# ## L² projection of weighted Dirac measure onto X⁰
#
# Solves M c = b where b_k = Σ_α w_α φ_k(v_α). The result f_s(v) = Σ c_k φ_k(v).
function l2_project!(ws::Workspace, f_coeffs, v_parts, w_parts)
    rhs = zeros(ws.n_dofs)
    nloc = (ws.p.P_DEG + 1)^2
    for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        fast_eval_particle!(ws, ws.lp_vals, ws.lp_gids, loc)
        @inbounds for j in 1:nloc
            rhs[ws.lp_gids[j]] += w_parts[α] * ws.lp_vals[j]
        end
    end
    f_coeffs .= ws.M_lu \ rhs
    return nothing
end

# ## Entropy S_h = -∫ f_s log f_s dv (so dS/dt ≥ 0 with our sign convention)
function compute_entropy(ws::Workspace, field::Forms.FormField)
    S = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            if USE_LOGSQ[]
                # log f = ½ log f²: keep undershoot (f<0) points via |f| guard.
                if f_val^2 > 1e-60
                    S -= f_val * 0.5 * log(f_val^2) * ws.qrule_integrate.weights[q] * jac
                end
            elseif f_val > 1e-30
                S -= f_val * log(f_val) * ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return S
end

# ## r_i = ∫ φ_i (1 + log f_s) dv  → entropy gradient seed
function compute_r!(ws::Workspace, r, field::Forms.FormField)
    fill!(r, 0.0)
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        evals, indices = evaluate(ws, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            integrand = if USE_LOGSQ[]
                # 1 + log f = 1 + ½ log f²: undershoot points contribute via |f| guard.
                f_val^2 > 1e-60 ? (1 + 0.5 * log(f_val^2)) : 0.0
            else
                f_val > 1e-30 ? (1 + log(f_val)) : 0.0
            end
            for (j, gidx) in enumerate(indices[1])
                r[gidx] += integrand * evals[1][q, j] *
                           ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return nothing
end

# ## G_α = ∇L(v_α) where L = M⁻¹ r — particle-side entropy gradient
function compute_G!(ws::Workspace, G, v_parts, L_vec)
    fill!(G, 0.0)
    nloc = (ws.p.P_DEG + 1)^2
    for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        fast_eval_particle_grad!(ws, ws.G_vals, ws.G_dxi1, ws.G_dxi2, ws.G_gids, loc)
        inv_h1 = 1.0 / loc.h1
        inv_h2 = 1.0 / loc.h2
        acc1 = 0.0
        acc2 = 0.0
        @inbounds for j in 1:nloc
            L = L_vec[ws.G_gids[j]]
            acc1 += L * ws.G_dxi1[j] * inv_h1
            acc2 += L * ws.G_dxi2[j] * inv_h2
        end
        G[α, 1] = acc1
        G[α, 2] = acc2
    end
    return nothing
end

# ## Landau collision-operator velocity update
function compute_collision!(ws::Workspace, dot_v, v_parts, w_parts, G)
    v1_lo, v1_hi = ws.bp1[1], ws.bp1[end]
    v2_lo, v2_hi = ws.bp2[1], ws.bp2[end]
    fill!(dot_v, 0.0)
    N = size(v_parts, 1)
    Threads.@threads for γ in 1:N
        vγ1, vγ2 = v_parts[γ, 1], v_parts[γ, 2]
        (vγ1 <= v1_lo || vγ1 >= v1_hi ||
         vγ2 <= v2_lo || vγ2 >= v2_hi) && continue
        Gγ1, Gγ2 = G[γ, 1], G[γ, 2]
        acc1, acc2 = 0.0, 0.0
        for α in 1:N
            γ == α && continue
            vα1, vα2 = v_parts[α, 1], v_parts[α, 2]
            (vα1 <= v1_lo || vα1 >= v1_hi ||
             vα2 <= v2_lo || vα2 >= v2_hi) && continue
            d1 = vγ1 - vα1
            d2 = vγ2 - vα2
            dist2 = d1^2 + d2^2
            dist2 < 1e-24 && continue
            dist = sqrt(dist2)
            g1 = G[α, 1] - Gγ1
            g2 = G[α, 2] - Gγ2
            dv_dot_g = (d1 * g1 + d2 * g2) / dist2
            inv_dist = 1.0 / dist
            acc1 += w_parts[α] * (g1 - d1 * dv_dot_g) * inv_dist
            acc2 += w_parts[α] * (g2 - d2 * dv_dot_g) * inv_dist
        end
        dot_v[γ, 1] = acc1
        dot_v[γ, 2] = acc2
    end
    return nothing
end

"""
    compute_collision_metric!(ws, dot_v, A, v_parts, w_parts, G)

`compute_collision!` plus, in the same pair loop, the metric
``A_\\gamma = \\sum_\\alpha w_\\alpha U(v_\\gamma - v_\\alpha)`` stored as `A[γ, :] =
(A₁₁, A₁₂, A₂₂)`. Then ``\\dot v_\\gamma = B_\\gamma - A_\\gamma G_\\gamma`` with ``B`` the
``U``-weighted sum of the other particles' ``G``: what `step_defect!` freezes. `dot_v`
is accumulated in the same antisymmetric form as `compute_collision!`, so it is
bit-identical to it.
"""
function compute_collision_metric!(ws::Workspace, dot_v, A, v_parts, w_parts, G)
    v1_lo, v1_hi = ws.bp1[1], ws.bp1[end]
    v2_lo, v2_hi = ws.bp2[1], ws.bp2[end]
    fill!(dot_v, 0.0)
    fill!(A, 0.0)
    N = size(v_parts, 1)
    Threads.@threads for γ in 1:N
        vγ1, vγ2 = v_parts[γ, 1], v_parts[γ, 2]
        (vγ1 <= v1_lo || vγ1 >= v1_hi ||
         vγ2 <= v2_lo || vγ2 >= v2_hi) && continue
        Gγ1, Gγ2 = G[γ, 1], G[γ, 2]
        acc1, acc2 = 0.0, 0.0
        a11, a12, a22 = 0.0, 0.0, 0.0
        for α in 1:N
            γ == α && continue
            vα1, vα2 = v_parts[α, 1], v_parts[α, 2]
            (vα1 <= v1_lo || vα1 >= v1_hi ||
             vα2 <= v2_lo || vα2 >= v2_hi) && continue
            d1 = vγ1 - vα1
            d2 = vγ2 - vα2
            dist2 = d1^2 + d2^2
            dist2 < 1e-24 && continue
            dist = sqrt(dist2)
            g1 = G[α, 1] - Gγ1
            g2 = G[α, 2] - Gγ2
            dv_dot_g = (d1 * g1 + d2 * g2) / dist2
            inv_dist = 1.0 / dist
            acc1 += w_parts[α] * (g1 - d1 * dv_dot_g) * inv_dist
            acc2 += w_parts[α] * (g2 - d2 * dv_dot_g) * inv_dist
            s = w_parts[α] * inv_dist
            a11 += s * (1 - d1 * d1 / dist2)
            a12 -= s * d1 * d2 / dist2
            a22 += s * (1 - d2 * d2 / dist2)
        end
        dot_v[γ, 1] = acc1
        dot_v[γ, 2] = acc2
        A[γ, 1] = a11
        A[γ, 2] = a12
        A[γ, 3] = a22
    end
    return nothing
end

# ##############################################################################
# Diagnostics
# ##############################################################################
#
# (1) Negative-part L¹ norm of f_s:    ∫ max(-f_s, 0) dv
#     A direct probe of L²-projection Gibbs oscillations: the empirical density
#     is non-negative everywhere, so any negative lobe in f_s is a projection
#     artifact. Computed on the same Gauss–Legendre grid used elsewhere.
function compute_negative_part_l1(ws::Workspace, field::Forms.FormField)
    neg = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            if f_val < 0.0
                neg += (-f_val) * ws.qrule_integrate.weights[q] * jac
            end
        end
    end
    return neg
end

# (2) ‖f_s − f_p‖₂  with f_p the *element-constant histogram density*:
#       f_p(v) = (Σ_{α: v_α ∈ e(v)} w_α) / |e(v)|
#     where e(v) is the element containing v. Discrepancy between the smooth
#     B-spline f_s and the piecewise-constant histogram f_p captures both the
#     Gibbs oscillation amplitude *and* the cell-to-cell mass-distribution
#     mismatch that drives spurious gradients via L = M⁻¹ r.
function compute_fs_minus_fp_l2(ws::Workspace, field::Forms.FormField,
        v_parts::AbstractMatrix, w_parts::AbstractVector)
    # Per-element particle mass:  m_e = Σ_{α ∈ e} w_α
    elem_mass = zeros(ws.n_elements)
    for α in axes(v_parts, 1)
        loc = locate_particle(ws, v_parts[α, 1], v_parts[α, 2])
        isnothing(loc) && continue
        elem_mass[loc.elem_id] += w_parts[α]
    end

    sumsq = 0.0
    for e in 1:ws.n_elements
        jac = element_measure(ws, e)            # |element|
        fp_e = elem_mass[e] / jac                # histogram density on element e
        fv, _ = evaluate(ws, field, e)
        for q in eachindex(ws.qrule_integrate.weights)
            f_val = fv[1][q]
            d = f_val - fp_e
            sumsq += d^2 * ws.qrule_integrate.weights[q] * jac
        end
    end
    return sqrt(sumsq)
end

# (3) Conserved particle moments: total momentum P = Σ w_α v_α and kinetic
#     energy E = ½ Σ w_α |v_α|². Both are exactly conserved by the Landau sum
#     and by the LB drift multipliers, so their drift measures solver error
#     rather than physics. Mesh-independent — no `Workspace` needed.
function compute_momentum(v_parts, w_parts)
    p1 = sum(w_parts[α] * v_parts[α, 1] for α in axes(v_parts, 1))
    p2 = sum(w_parts[α] * v_parts[α, 2] for α in axes(v_parts, 1))
    return (p1, p2)
end

function compute_energy(v_parts, w_parts)
    return 0.5 * sum(w_parts[α] * (v_parts[α, 1]^2 + v_parts[α, 2]^2)
    for α in axes(v_parts, 1))
end
