#! /usr/bin/env -S julia --startup-file=no
# Check and time the FP64 pair kernel that also returns the metric A (solver = :defect)
# against the plain kernel: F must be bit-identical, A must match the CPU loop.
#
#   julia --startup-file=no scripts/check_metric_gpu.jl <preset> [--N_PARTICLES=…]

include(joinpath(@__DIR__, "..", "main.jl"))
using Printf, Random, Statistics

const P = parse_overrides(include(abspath(ARGS[1]))::SimParameters, ARGS[2:end])
P.use_gpu && !P.gpu_fp32 || error("run with --use_gpu=true --gpu_fp32=false")
USE_LOGSQ[] = P.use_logsq
enable_gpu!(P)

ws = build_workspace(P)
Random.seed!(P.seed)
N = P.N_PARTICLES
v = zeros(N, 2)
if P.v1_peak != 0
    v[:, 1] .= [fill(-P.v1_peak, N ÷ 2); fill(P.v1_peak, N - N ÷ 2)] .+ P.σ1 .* randn(N)
else
    v[:, 1] .= P.σ1 .* randn(N)
end
v[:, 2] .= P.σ2 .* randn(N)
w = fill(1 / N, N)
f = zeros(ws.n_dofs); r = zeros(ws.n_dofs); L = zeros(ws.n_dofs)
G = zeros(N, 2)
Base.invokelatest(L2PROJ_FN[], ws, f, v, w)
compute_r!(ws, r, build_field(ws, f)); ldiv!(L, ws.M_lu, r)
Base.invokelatest(COMPG_FN[], ws, G, v, L)

F0 = zeros(N, 2); F1 = zeros(N, 2); A1 = zeros(N, 3)
gpuF = getglobal(Main, :compute_collision_gpu!)
gpuFA = getglobal(Main, :compute_collision_metric_gpu!)
Base.invokelatest(gpuF, ws, F0, v, w, G)
Base.invokelatest(gpuFA, ws, F1, A1, v, w, G)
@printf("F (metric kernel) vs F (plain kernel): max |Δ| = %.3e (must be 0)\n", maximum(abs, F1 .- F0))

t(f!, args...) = minimum(@elapsed(Base.invokelatest(f!, args...)) for _ in 1:5)
tF = t(gpuF, ws, F0, v, w, G)
tFA = t(gpuFA, ws, F1, A1, v, w, G)
@printf("N = %d: plain kernel %.1f ms, F + A kernel %.1f ms (%.2fx)\n", N, 1e3tF, 1e3tFA, tFA / tF)

# A against the CPU loop on a subsample of γ (the CPU loop is O(N²))
FA = zeros(N, 2); AA = zeros(N, 3)
idx = 1:min(N, 4000)
compute_collision_metric!(ws, FA, AA, v, w, G)
@printf("A (GPU) vs A (CPU): max rel. error %.2e\n",
    maximum(abs, A1[idx, :] .- AA[idx, :]) / maximum(abs, AA[idx, :]))
@printf("F (GPU) vs F (CPU): max rel. error %.2e\n",
    maximum(abs, F1[idx, :] .- FA[idx, :]) / maximum(abs, FA[idx, :]))
