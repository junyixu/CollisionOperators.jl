#! /usr/bin/env -S julia --startup-file=no
# Predictor / preconditioner probe for the implicit Landau step. From a
# checkpoint it takes `steps` steps; on each it re-solves the step from several
# starts, and once with a per-particle preconditioner, counting Anderson
# iterations against the Euler start. The trajectory always advances with the
# Euler-start root, so every variant solves the same steps.
#
#   julia --startup-file=no scripts/predictor_probe.jl <preset> <checkpoint.jls> \
#         [--steps=5] [--probes=8] [--fd_h=1e-5] [--key=value …]   (SimParameters overrides)
#
# Starts (Landau, F_γ = Σ_α w_α U(v_γ − v_α)(G_α − G_γ) = B_γ − A_γ G_γ):
#   euler    vⁿ + Δt·F(vⁿ)
#   picard   one undamped Picard step from euler: 1 map evaluation, the ceiling for
#            any predictor that approximates that step
#   fm_self  vⁿ + Δt·(Fⁿ − A_γ (G_eff,γ − G_γ)): metric A, B frozen at vⁿ, only the
#            particle's own entropy gradient moved to the Euler midpoint; O(N) once A
#            is known (A changes slowly, so it can be refreshed every few steps)
#   fm_full  vⁿ + Δt·F(vⁿ; G_eff): every particle's G moved, positions frozen
#            (one pair sum without a projection)
# Solver variants (both start from euler):
#   pc_id    this file's Anderson with P = I; must reproduce `euler` (sanity check)
#   pc_diag  Anderson on v ↦ v + P(𝒢(v) − v), P_γ = (I − J_γγ)⁻¹, with the 2×2
#            diagonal blocks J_γγ of the Picard map estimated by random-sign probing
#            (2·probes map evaluations): what a local preconditioner could do at best
# The converged root and the conservative exit (return 𝒢(v)) are the same for all.
#
# Output: probe_<suffix>.csv, one row per step and variant.

include(joinpath(@__DIR__, "..", "main.jl"))
using Printf, Random, Serialization, Statistics

# ---- setup -------------------------------------------------------------------------
length(ARGS) >= 2 || error("usage: predictor_probe.jl <preset> <checkpoint.jls> [--key=value …]")
const PROBE_KEYS = ("steps", "probes", "fd_h")
probe_opts = Dict(String(m[1]) => String(m[2])
                  for m in (match(r"^--(\w+)=(.*)$", a) for a in ARGS[3:end])
                  if m !== nothing && m[1] in PROBE_KEYS)
const P = parse_overrides(include(abspath(ARGS[1]))::SimParameters,
    filter(a -> !any(k -> startswith(a, "--$k="), PROBE_KEYS), ARGS[3:end]))
const NSTEPS = parse(Int, get(probe_opts, "steps", "5"))
const NPROBES = parse(Int, get(probe_opts, "probes", "8"))
const FD_H = parse(Float64, get(probe_opts, "fd_h", P.gpu_fp32 ? "1e-4" : "1e-5"))
P.collision_model == :landau || error("probe is Landau-only")
USE_LOGSQ[] = P.use_logsq
P.use_gpu && enable_gpu!(P)

# ---- the metric part A_γ = Σ_α w_α U(v_γ − v_α), and B_γ = Σ_α w_α U G_α ------------
# Same pair loop and domain cut as compute_collision!, so F = B − A G holds exactly.
function collision_tensor!(A, B, ws, v, w, G)
    v1_lo, v1_hi = ws.bp1[1], ws.bp1[end]
    v2_lo, v2_hi = ws.bp2[1], ws.bp2[end]
    inside(a) = v1_lo < v[a, 1] < v1_hi && v2_lo < v[a, 2] < v2_hi
    fill!(A, 0.0); fill!(B, 0.0)
    N = size(v, 1)
    Threads.@threads for γ in 1:N
        inside(γ) || continue
        a11 = a12 = a22 = b1 = b2 = 0.0
        for α in 1:N
            (α == γ || !inside(α)) && continue
            d1 = v[γ, 1] - v[α, 1]; d2 = v[γ, 2] - v[α, 2]
            r2 = d1^2 + d2^2
            r2 < 1e-24 && continue
            s = w[α] / sqrt(r2)
            u11 = s * (1 - d1^2 / r2); u12 = -s * d1 * d2 / r2; u22 = s * (1 - d2^2 / r2)
            a11 += u11; a12 += u12; a22 += u22
            b1 += u11 * G[α, 1] + u12 * G[α, 2]
            b2 += u12 * G[α, 1] + u22 * G[α, 2]
        end
        A[γ, 1] = a11; A[γ, 2] = a12; A[γ, 3] = a22
        B[γ, 1] = b1; B[γ, 2] = b2
    end
    return nothing
end

# ---- workspace, state, buffers --------------------------------------------------------
ws = build_workspace(P)
ck = deserialize(ARGS[2])
v = copy(ck.v_particles); w = copy(ck.w_particles)
N = size(v, 1)
N == P.N_PARTICLES || error("checkpoint has N=$N, preset N=$(P.N_PARTICLES)")
z() = zeros(N, 2)
f_coeffs = zeros(ws.n_dofs); f_buf = zeros(ws.n_dofs)
r_vec = zeros(ws.n_dofs); L_vec = zeros(ws.n_dofs)
G = z(); F = z(); v_mid = z(); dv = z(); dS_mid = z(); G_eff = z(); dot_v_buf = z()
Gv = z(); r_curr = z(); r_prev = z(); Gv_prev = z(); v_old = z()
ΔF = zeros(2N, P.m_anderson); ΔG = zeros(2N, P.m_anderson)
A = zeros(N, 3); Bt = z()

pmap!(out, vin, v0, S0) = picard_map!(ws, out, vin, v0, w, S0, P.DT,
    v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf, r_vec, L_vec, G; use_gonzalez = P.use_gonzalez)

solve!(v1, v0, S0) = step_anderson!(ws, v1, v0, w, S0, P.DT,
    v_mid, dv, dS_mid, G_eff, dot_v_buf, f_buf, r_vec, L_vec, G,
    Gv, r_curr, r_prev, Gv_prev, v_old, ΔF, ΔG;
    m = P.m_anderson, max_iter = P.max_iter, tol = P.tol, abs_floor = P.abs_floor,
    stag_window = P.stag_window, stag_rel_tol = P.stag_rel_tol,
    damp_decay_start = P.damp_decay_start, damp_decay_factor = P.damp_decay_factor,
    damping = P.damping, use_anderson = P.use_anderson, use_gonzalez = P.use_gonzalez,
    exit_picard_step = P.exit_picard_step)

# Anderson exactly as step_anderson! (no restarts), but on the preconditioned map
# v ↦ v + Pc(𝒢(v) − v). Convergence, stagnation and the exit use the true residual
# 𝒢(v) − v, and the exit returns 𝒢(v) as step_anderson! does.
function solve_pc!(v1, v0, S0, Pc)
    m = P.m_anderson; β0 = P.damping
    Gt = z(); rt = z(); rh = z(); Gh = z(); rh_prev = z(); Gh_prev = z(); vold = z()
    v_best = copy(v1); nrm_best = Inf; nrm_win = Inf; nrm0 = 0.0
    hist = 0; slot = 0
    for k in 1:P.max_iter
        vold .= v1
        pmap!(Gt, v1, v0, S0)
        @. rt = Gt - v1
        nrm = norm(rt); k == 1 && (nrm0 = nrm)
        if nrm < nrm_best
            nrm_best = nrm; v_best .= Gt
        end
        if nrm < max(P.tol * (norm(v1) + 1e-30), P.abs_floor)
            v1 .= Gt
            return k, nrm, nrm0
        end
        if k > P.stag_window && k % P.stag_window == 0
            (nrm_win - nrm_best) / (nrm_win + 1e-30) < P.stag_rel_tol &&
                (v1 .= v_best; return k, nrm_best, nrm0)
            nrm_win = nrm_best
        end
        @inbounds for a in 1:N   # preconditioned residual and map value
            rh[a, 1] = Pc[a, 1] * rt[a, 1] + Pc[a, 2] * rt[a, 2]
            rh[a, 2] = Pc[a, 3] * rt[a, 1] + Pc[a, 4] * rt[a, 2]
        end
        @. Gh = v1 + rh
        β = k > P.damp_decay_start ? β0 * P.damp_decay_factor : β0
        if k == 1 || !P.use_anderson
            @. v1 = β * Gh + (1 - β) * vold
        else
            slot = mod1(slot + 1, m); hist = min(hist + 1, m)
            @views ΔF[:, slot] .= vec(rh) .- vec(rh_prev)
            @views ΔG[:, slot] .= vec(Gh) .- vec(Gh_prev)
            Fv = @view ΔF[:, 1:hist]; Gvw = @view ΔG[:, 1:hist]
            AtA = Fv' * Fv
            λ2 = 1e-10 * sum(AtA[j, j] for j in 1:hist) / hist + 1e-30
            for j in 1:hist
                AtA[j, j] += λ2
            end
            γc = AtA \ (Fv' * vec(rh))
            v1 .= Gh
            mul!(vec(v1), Gvw, γc, -1.0, 1.0)
            @. v1 = β * v1 + (1 - β) * vold
        end
        rh_prev .= rh; Gh_prev .= Gh
    end
    v1 .= v_best
    return P.max_iter, nrm_best, nrm0
end

# 2×2 diagonal blocks of the Picard-map Jacobian at `vE`, by random-sign probing:
# perturb every particle by ±h along e_k; particle γ's own response, sign-corrected,
# averages to J_γγ e_k while the cross terms average out.
function diag_jacobian(vE, v0, S0)
    J = zeros(N, 4)   # [J11 J12 J21 J22]
    base = z(); pert = z(); vp = z()
    pmap!(base, vE, v0, S0)
    rng = Xoshiro(1)
    for k in 1:2, _ in 1:NPROBES
        s = rand(rng, (-1.0, 1.0), N)
        vp .= vE; @views vp[:, k] .+= FD_H .* s
        pmap!(pert, vp, v0, S0)
        @inbounds for a in 1:N
            J[a, k] += s[a] * (pert[a, 1] - base[a, 1]) / FD_H / NPROBES      # J_{1k}
            J[a, 2 + k] += s[a] * (pert[a, 2] - base[a, 2]) / FD_H / NPROBES  # J_{2k}
        end
    end
    return J
end

# Pc = (I − J)⁻¹ per particle; fall back to I where I − J is near-singular.
function local_preconditioner(J)
    Pc = zeros(N, 4)
    nfix = 0
    for a in 1:N
        m11 = 1 - J[a, 1]; m12 = -J[a, 2]; m21 = -J[a, 3]; m22 = 1 - J[a, 4]
        d = m11 * m22 - m12 * m21
        if abs(d) < 0.05
            Pc[a, 1] = 1; Pc[a, 4] = 1; nfix += 1
        else
            Pc[a, 1] = m22 / d; Pc[a, 2] = -m12 / d; Pc[a, 3] = -m21 / d; Pc[a, 4] = m11 / d
        end
    end
    return Pc, nfix
end

dist(a, b) = sqrt(sum(abs2, a .- b))

# ---- main loop ----------------------------------------------------------------------------
out = "probe_$(P.suffix).csv"
io = open(out, "w")
println(io, "step,variant,iter,r0,residual,decades,extra_map_evals")
Pid = zeros(N, 4); Pid[:, 1] .= 1; Pid[:, 4] .= 1
step0 = ck.step
for n in 1:NSTEPS
    step = step0 + n
    Base.invokelatest(L2PROJ_FN[], ws, f_coeffs, v, w)
    S0 = compute_entropy(ws, build_field(ws, f_coeffs))
    compute_r!(ws, r_vec, build_field(ws, f_coeffs))
    ldiv!(L_vec, ws.M_lu, r_vec)
    Base.invokelatest(COMPG_FN[], ws, G, v, L_vec)
    Base.invokelatest(COLLISION_FN[], ws, F, v, w, G)
    Gn = copy(G); Fn = copy(F)
    vE = v .+ P.DT .* Fn

    # one Picard map from euler; G_eff then holds the midpoint entropy gradient
    vP = z(); pmap!(vP, vE, v, S0)
    Geff = copy(G_eff)
    t_A = @elapsed collision_tensor!(A, Bt, ws, v, w, Gn)
    cons = sqrt(sum(abs2, Bt .- [A[:, 1] .* Gn[:, 1] .+ A[:, 2] .* Gn[:, 2] A[:, 2] .* Gn[:, 1] .+ A[:, 3] .* Gn[:, 2]] .- Fn)) /
           sqrt(sum(abs2, Fn))
    vS = similar(v)
    @inbounds for a in 1:N
        d1 = Geff[a, 1] - Gn[a, 1]; d2 = Geff[a, 2] - Gn[a, 2]
        vS[a, 1] = v[a, 1] + P.DT * (Fn[a, 1] - (A[a, 1] * d1 + A[a, 2] * d2))
        vS[a, 2] = v[a, 2] + P.DT * (Fn[a, 2] - (A[a, 2] * d1 + A[a, 3] * d2))
    end
    Ff = z(); Base.invokelatest(COLLISION_FN[], ws, Ff, v, w, Geff)
    vF = v .+ P.DT .* Ff

    # reference root from euler (also advances the trajectory)
    vstar = copy(vE); itE, resE, _, r0E = solve!(vstar, v, S0)
    gap = dist(vE, vstar)
    rows = Tuple{String, Int, Float64, Float64, Float64, Int}[("euler", itE, r0E, resE, 0.0, 0)]
    for (name, v0x, extra) in (("picard", vP, 1), ("fm_self", vS, 0), ("fm_full", vF, 0))
        x = copy(v0x); it, res, _, r0 = solve!(x, v, S0)
        push!(rows, (name, it, r0, res, log10(gap / dist(v0x, vstar)), extra))
    end
    x = copy(vE); it, res, r0 = solve_pc!(x, v, S0, Pid)
    push!(rows, ("pc_id", it, r0, res, 0.0, 0))
    J = diag_jacobian(vE, v, S0)
    Pc, nfix = local_preconditioner(J)
    x = copy(vE); it, res, r0 = solve_pc!(x, v, S0, Pc)
    push!(rows, ("pc_diag", it, r0, res, 0.0, 2NPROBES + 1))
    for r in rows
        println(io, join((step, r...), ','))
    end
    flush(io)

    nJ = [sqrt(J[a, 1]^2 + J[a, 2]^2 + J[a, 3]^2 + J[a, 4]^2) for a in 1:N]
    sp = hypot.(v[:, 1], v[:, 2])
    top = sortperm(nJ; rev = true)[1:max(1, N ÷ 100)]
    @printf("step %d  F=B−AG rel.err %.1e  A in %.1f s | iter: %s | ‖J_γγ‖ median %.3f, p99 %.3f, max %.2f, top-1%% median |v| %.2f (all %.2f), Pc fallbacks %d\n",
        step, cons, t_A, join(("$(r[1])=$(r[2])" for r in rows), " "),
        median(nJ), quantile(nJ, 0.99), maximum(nJ), median(sp[top]), median(sp), nfix)
    v .= vstar
end
close(io)
println("wrote $out")
