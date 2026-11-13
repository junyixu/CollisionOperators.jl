#! /usr/bin/env -S julia --startup-file=no
# Accuracy and cost of the Jacobian products of `picard_jvp!` (nk_jvp = :tan32, :tan64,
# :fd32) against a central difference of the FP64 Picard map, at the explicit Euler
# predictor of step 1.
#
#   julia --startup-file=no scripts/check_jvp.jl <preset> --use_gpu=true --gpu_fp32=false

include(joinpath(@__DIR__, "..", "main.jl"))
using Printf, Random, LinearAlgebra

const P = parse_overrides(include(abspath(ARGS[1]))::SimParameters, ARGS[2:end])
USE_LOGSQ[] = P.use_logsq
P.use_gpu && enable_gpu!(P)

ws = build_workspace(P)
Random.seed!(P.seed)
N = P.N_PARTICLES
v0 = zeros(N, 2)
if P.v1_peak != 0
    v0[:, 1] .= [fill(-P.v1_peak, N ÷ 2); fill(P.v1_peak, N - N ÷ 2)] .+ P.σ1 .* randn(N)
else
    v0[:, 1] .= P.σ1 .* randn(N)
end
v0[:, 2] .= P.σ2 .* randn(N)
w = fill(1 / N, N)
dt = P.DT

f = zeros(ws.n_dofs); r = zeros(ws.n_dofs); L = zeros(ws.n_dofs); fb = zeros(ws.n_dofs)
G = zeros(N, 2); dotv = zeros(N, 2); Gv = zeros(N, 2)
vm = zeros(N, 2); dv = zeros(N, 2); dS = zeros(N, 2); Ge = zeros(N, 2)
Base.invokelatest(L2PROJ_FN[], ws, f, v0, w)
fs = build_field(ws, f)
S0 = compute_entropy(ws, fs)
compute_r!(ws, r, fs); ldiv!(L, ws.M_lu, r)
Base.invokelatest(COMPG_FN[], ws, G, v0, L)
Base.invokelatest(COLLISION_FN[], ws, dotv, v0, w, G)
x = vec(v0 .+ dt .* dotv)

function F!(out, v)
    picard_map!(ws, Gv, reshape(v, N, 2), v0, w, S0, dt, vm, dv, dS, Ge, dotv, fb, r, L, G;
        use_gonzalez = P.use_gonzalez)
    out .= v .- vec(Gv)
    return out
end

Fx = F!(zeros(2N), x)
@printf("N = %d, DT = %g, ‖F(x)‖ = %.3e at the Euler predictor\n", N, dt, norm(Fx))
u_rand = normalize!(randn(2N))
u_res = normalize(Fx)

# Reference: central difference of the FP64 map, probe moved by h.
function jref(u, h)
    a = F!(zeros(2N), x .+ h .* u)
    b = F!(zeros(2N), x .- h .* u)
    return (a .- b) ./ 2h
end

jw = jvp_workspace(N)
function prod(mode, u, h)
    y = zeros(2N)
    picard_jvp!(ws, y, x, u, jw, mode, v0, w, S0, dt, vm, dv, dS, Ge, dotv, fb, r, L, G;
        fd_h = h, use_gonzalez = P.use_gonzalez)
    return y
end

for (name, u) in (("random direction", u_rand), ("residual direction", u_res))
    println("\n== $name: relative error of (J − I)u against the FP64 central difference ==")
    ref6 = jref(u, 1e-6) .- u
    ref5 = jref(u, 1e-5) .- u
    @printf("reference: ‖(J−I)u‖ = %.3e, central h=1e-5 vs h=1e-6 differ by %.2e\n",
        norm(ref6), norm(ref5 .- ref6) / norm(ref6))
    for (mode, hs) in ((:tan64, (1e-6, 1e-5)), (:tan32, (1e-6, 1e-5)),
        (:fd32, (1e-5, 1e-4, 1e-3, 1e-2)))
        for h in hs
            jw.has[] = false
            y = prod(mode, u, h) .- u
            @printf("  %-6s h = %.0e : %.2e\n", mode, h, norm(y .- ref6) / norm(ref6))
        end
    end
end

println("\n== cost per product (min of 5), ms ==")
tmin(f) = 1e3 * minimum(@elapsed(f()) for _ in 1:5)
@printf("FP64 map            %.1f\n", tmin(() -> F!(zeros(2N), x)))
for mode in (:tan64, :tan32, :fd32)
    jw.has[] = false
    prod(mode, u_rand, 1e-5)
    @printf("%-6s product      %.1f\n", mode, tmin(() -> prod(mode, u_rand, 1e-5)))
end
tgeff = tmin(() -> landau_geff!(ws, Ge, reshape(x, N, 2), v0, w, S0, vm, dv, dS, fb, r, L, G;
    use_gonzalez = P.use_gonzalez))
@printf("landau_geff! alone  %.1f\n", tgeff)
