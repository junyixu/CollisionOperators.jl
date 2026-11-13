# gs_solvers.jl — the implicit step solved with components of GeometricSolvers.jl
# (https://github.com/JuliaGNI/GeometricSolvers.jl). main.jl includes this file only
# for `--solver=gs_nk|gs_anderson|gs_npc`, so the default solvers keep no dependency
# on the package (which needs Julia 1.13).
#
# All three solve F(v) = v − 𝒢(v) = 0 for the Picard map 𝒢 of solver.jl, with the
# stopping rule and the conservative exit 𝒢(v) of `step_anderson!`:
#
#   :gs_nk        Newton(linsolve = GMRES, jacobian = JVPFunction): the product is
#                 `nk_jvp`, the forward difference of the FP64 map or `picard_jvp!`
#                 (FP32 or FP64 tangent of the pair sum); the residual stays in FP64.
#   :gs_anderson  Anderson(Picard(damping), gs_m) on F.
#   :gs_npc       LeftPreconditioned(outer, FrozenMetric()): the frozen-metric solve
#                 of `step_defect!` as a nonlinear left preconditioner. The outer is
#                 Anderson(Picard(), gs_m), or Picard() (= `step_defect!`) for gs_m = 0.

using GeometricSolvers
using GeometricSolvers: NonlinearProblem, Newton, GMRES, JVPFunction, Picard, Anderson,
                        LeftPreconditioned, FixedForcing, EisenstatWalkerChoice2,
                        Options, StepInfo, SUCCESS
using LinearAlgebra: norm

const GS = GeometricSolvers

"""
    GSContext

Everything a GeometricSolvers solve of one time step needs, passed to it as the
parameters `p`: the workspace and scratch arrays of [`picard_map!`](@ref), the step
data (`v0`, `w`, `S0`, `dt`), single-entry caches of the last FP64 map and
frozen-metric solve, and the evaluation counters.

The caches matter for `LeftPreconditioned`, which asks for the global residual and
the inner step at the same point more than once per iteration.
"""
mutable struct GSContext{W, B, D, J}
    ws::W
    N::Int
    buf::B                    # scratch arrays of main.jl's time loop
    dc::D                     # A, Gk, Φu for the frozen metric (or nothing)
    v0::Matrix{Float64}
    w::Vector{Float64}
    S0::Float64
    dt::Float64
    use_gonzalez::Bool
    with_metric::Bool         # FP64 maps also return the metric A (:gs_npc)
    # last FP64 map: 𝒢(x64) in G64
    x64::Vector{Float64}
    G64::Matrix{Float64}
    has64::Bool
    xp::Vector{Float64}       # difference probe and its map (:fd)
    Gp::Matrix{Float64}
    jw::J                     # workspace of picard_jvp! (or nothing)
    # last frozen-metric solve: N(xN) in uN
    xN::Vector{Float64}
    uN::Matrix{Float64}
    hasN::Bool
    # counters of the current step
    n64::Int
    n_jvp::Int
    n_inner::Int
    nrm0::Float64
    # settings
    jvp::Symbol
    fd_h::Float64
    eta::Float64
    max_inner::Int
    m_inner::Int
    damping::Float64
    abs_floor::Float64
end

# 𝒢(x) with the FP64 pair sum, uncached, into `out`; with `metric` also A and G_eff.
function gs_map64!(c::GSContext, out::Matrix{Float64}, x; metric::Bool = false)
    b = c.buf
    X = reshape(x, c.N, 2)
    if metric
        landau_geff!(c.ws, b.G_eff, X, c.v0, c.w, c.S0, b.v_mid, b.dv, b.dS_mid,
            b.f_buf, b.r_vec, b.L_vec, b.G; use_gonzalez = c.use_gonzalez)
        Base.invokelatest(COLLMETRIC_FN[], c.ws, b.dot_v, c.dc.A, b.v_mid, c.w, b.G_eff)
        @. out = c.v0 + c.dt * b.dot_v
        c.dc.Gk .= b.G_eff
    else
        picard_map!(c.ws, out, X, c.v0, c.w, c.S0, c.dt, b.v_mid, b.dv, b.dS_mid,
            b.G_eff, b.dot_v, b.f_buf, b.r_vec, b.L_vec, b.G;
            use_gonzalez = c.use_gonzalez)
    end
    c.n64 += 1
    return out
end

# Make the FP64 cache hold x.
function gs_full64!(c::GSContext, x)
    c.has64 && x == c.x64 && return nothing
    gs_map64!(c, c.G64, x; metric = c.with_metric)
    copyto!(c.x64, x)
    c.has64 = true
    return nothing
end

# The residual of the problem: F(x) = x − 𝒢(x), FP64.
function gs_residual!(r, x, c::GSContext)
    gs_full64!(c, x)
    r .= x .- vec(c.G64)
    c.n64 == 1 && (c.nrm0 = norm(r))
    return r
end

# J v for J = I − ∂𝒢/∂x: `picard_jvp!`, or for `:fd` the forward difference of the FP64
# map that moves the probe by `fd_h`, as in `newton_krylov!`.
function gs_jvp!(Jv, x, v, c::GSContext)
    c.n_jvp += 1
    b = c.buf
    if c.jvp !== :fd
        return picard_jvp!(c.ws, Jv, x, v, c.jw, c.jvp, c.v0, c.w, c.S0, c.dt, b.v_mid,
            b.dv, b.dS_mid, b.G_eff, b.dot_v, b.f_buf, b.r_vec, b.L_vec, b.G;
            fd_h = c.fd_h, use_gonzalez = c.use_gonzalez)
    end
    nv = norm(v)
    if nv == 0
        fill!(Jv, 0.0)
        return Jv
    end
    h = c.fd_h / nv
    @. c.xp = x + h * v
    gs_full64!(c, x)
    gs_map64!(c, c.Gp, c.xp)
    Jv .= v .- (vec(c.Gp) .- vec(c.G64)) ./ h
    return Jv
end

# ---- the frozen-metric solve as an inner method -------------------------------

"""
    FrozenMetric()

The frozen-metric solve of [`step_defect!`](@ref) as a nonlinear method of
GeometricSolvers: one step from `x` returns ``N(x)``, the fixed point of the cheap map
``\\Phi_x(u) = \\mathcal{G}(x) - \\Delta t\\, A_x (G_\\text{eff}(u) - G_\\text{eff}(x))``
found by Anderson to the tolerance `eta` relative to ``\\lVert\\mathcal{G}(x) - x\\rVert``.
It is meant as the `inner` of `LeftPreconditioned`, which reads only the iterate the
step returns, so the step does not evaluate the global residual at ``N(x)``.
"""
struct FrozenMetric end

struct FrozenMetricState{V}
    Fx::V
    x₊::V
end

GS.init_state(::FrozenMetric, ::NonlinearProblem, x, p) =
    FrozenMetricState(similar(x), similar(x))
GS.residual_buffer(s::FrozenMetricState) = s.Fx
GS.start!!(s::FrozenMetricState, ::FrozenMetric, prob, x, Fx, p, fnorm₀) = s
GS.finish!!(s::FrozenMetricState, ::FrozenMetric, prob, x, Fx, status) = (x, status, s)

function GS.step!!(s::FrozenMetricState, ::FrozenMetric, prob, x, Fx, c::GSContext)
    gs_frozen_solve!(c, s.x₊, x)
    stepnorm = sqrt(sum(abs2(a - b) for (a, b) in zip(s.x₊, x)))
    return s.x₊, Fx, s,
    StepInfo{Float64}(SUCCESS, norm(Fx), stepnorm, Int32(0), false, false)
end

# N(x) into `out`, cached.
function gs_frozen_solve!(c::GSContext, out, x)
    if c.hasN && x == c.xN
        copyto!(out, c.uN)
        return out
    end
    gs_full64!(c, x)          # 𝒢(x), A and G_eff(x), unless x is the cached point
    b, dc, N, dt = c.buf, c.dc, c.N, c.dt
    Gv, A, Gk = c.G64, dc.A, dc.Gk
    nrm = sqrt(sum(abs2(a - g) for (a, g) in zip(x, Gv)))

    function frozen_map!(o, u)
        landau_geff!(c.ws, b.G_eff, u, c.v0, c.w, c.S0, b.v_mid, b.dv, b.dS_mid,
            b.f_buf, b.r_vec, b.L_vec, b.G; use_gonzalez = c.use_gonzalez)
        @inbounds for a in 1:N
            d1 = b.G_eff[a, 1] - Gk[a, 1]
            d2 = b.G_eff[a, 2] - Gk[a, 2]
            o[a, 1] = Gv[a, 1] - dt * (A[a, 1] * d1 + A[a, 2] * d2)
            o[a, 2] = Gv[a, 2] - dt * (A[a, 2] * d1 + A[a, 3] * d2)
        end
        return o
    end

    u = c.uN
    copyto!(u, x)
    c.n_inner += _anderson_inner!(frozen_map!, u, dc.Φu, b.r_curr, b.r_prev, b.Gv_prev,
        b.v_old_buf, b.ΔF, b.ΔG, max(c.eta * nrm, 0.5 * c.abs_floor), c.max_inner,
        c.m_inner, c.damping)
    copyto!(c.xN, x)
    c.hasN = true
    copyto!(out, u)
    return out
end

# ---- setup and the per-step driver --------------------------------------------

"""
    gs_setup(ws, p, fd_h, buf, dc, jw)

Build the GeometricSolvers problem, method and solver state for `p.solver`, once per
run. `buf` holds the scratch arrays of the time loop and `dc` the frozen-metric arrays
(`nothing` unless `:gs_npc`). Returns what [`gs_step!`](@ref) takes.
"""
function gs_setup(ws, p, fd_h, buf, dc, jw)
    N = p.N_PARTICLES
    z() = zeros(2N)
    Z() = zeros(N, 2)
    c = GSContext(ws, N, buf, dc, Z(), zeros(N), 0.0, p.DT, p.use_gonzalez,
        p.solver === :gs_npc,
        z(), Z(), false, z(), Z(), jw, z(), Z(), false,
        0, 0, 0, 0.0,
        p.nk_jvp, fd_h, p.dc_eta, p.dc_max_inner, p.m_anderson, p.damping, p.abs_floor)

    # Anderson and Picard prepare a derivative of a problem without a Jacobian by AD,
    # which the GPU kernels cannot take; none of the methods here reads it.
    nojac(J, x, q) = error("the Jacobian of the Picard map is not available")
    method, prob = if p.solver === :gs_nk
        forcing = p.gs_eta > 0 ? FixedForcing(p.gs_eta) :
                  EisenstatWalkerChoice2(; η_max = p.nk_eta_max)
        Newton(; linsolve = GMRES(; maxiter = p.nk_krylov_max, forcing),
            jacobian = JVPFunction(gs_jvp!)), NonlinearProblem(gs_residual!)
    elseif p.solver === :gs_anderson
        Anderson(Picard(; damping = p.damping), p.gs_m),
        NonlinearProblem(gs_residual!, nojac)
    else
        outer = p.gs_m > 0 ? Anderson(Picard(), p.gs_m) : Picard()
        LeftPreconditioned(outer, FrozenMetric()), NonlinearProblem(gs_residual!, nojac)
    end
    println("GeometricSolvers $(pkgversion(GeometricSolvers)): ", method)
    solver = GS.init(prob, z(), method, c; f_abstol = p.abs_floor, f_reltol = 0.0)
    return (; c, solver, tol = p.tol, abs_floor = p.abs_floor, max_iter = p.max_iter,
        exit_picard_step = p.exit_picard_step, npc = p.solver === :gs_npc)
end

"""
    gs_step!(gs, v1, v0, w, S0, dt; verbose = false)
        -> (fp64_maps, residual, inner, initial_residual)

Solve one implicit step with the GeometricSolvers method of [`gs_setup`](@ref), from the
predictor in `v1`, to `step_anderson!`'s target ``\\max(\\texttt{tol}\\,\\lVert v\\rVert,
\\texttt{abs\\_floor})`` on ``\\lVert\\mathcal{G}(v) - v\\rVert``. `v1` leaves holding
``\\mathcal{G}(v)`` (the conservative exit) unless `exit_picard_step` is off. `inner`
counts the frozen-metric maps (`:gs_npc`) or the Jacobian products (`:gs_nk`).
"""
function gs_step!(gs, v1, v0, w, S0, dt; verbose::Bool = false)
    c = gs.c
    c.v0 .= v0
    c.w .= w
    c.S0 = S0
    c.dt = dt
    c.has64 = c.hasN = false
    c.jw !== nothing && (c.jw.has[] = false)
    c.n64 = c.n_jvp = c.n_inner = 0
    c.nrm0 = 0.0

    x = vec(v1)
    eff_tol = max(gs.tol * (norm(x) + 1e-30), gs.abs_floor)
    opts = Options(Float64; f_abstol = eff_tol, f_reltol = 0.0, x_abstol = 0.0,
        x_reltol = 0.0, max_iterations = min(gs.max_iter, 10_000), max_stalls = 5)
    s = gs.solver
    x₊, status, state = GS.solve!!(s.state, s.prob, x, s.method, c, opts)
    s.state === state || (s.state = state)
    x₊ === x || copyto!(x, x₊)

    # ‖F(x)‖ of the returned iterate and the conservative exit 𝒢(x); both come from
    # the cache unless the solve ended away from its last evaluation.
    gs_full64!(c, x)
    nrm = sqrt(sum(abs2(a - g) for (a, g) in zip(x, c.G64)))
    gs.exit_picard_step && copyto!(x, c.G64)
    inner = gs.npc ? c.n_inner : c.n_jvp
    (verbose || status.code != SUCCESS) &&
        println("    [gs $(status.code)]  it=$(status.iterations)  fp64=$(c.n64)  " *
                "inner=$inner  ‖F‖=$nrm  (status $(status.fnorm))")
    return c.n64, nrm, inner, c.nrm0
end
