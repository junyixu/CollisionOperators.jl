# warmstart.jl — initial guesses for the implicit solve beyond the Euler predictor.
#
# Each step solves v = 𝒢(v) by Anderson iteration from a guess v⁽⁰⁾. The Euler
# predictor v_E = vⁿ + Δt·v̇ⁿ misses the root v* by the gap g = v* − v_E = O(Δt²).
# Nothing here touches the solver or its root, only where it starts:
#
#   1. NN warm start (`warmstart = :nn`): a per-particle MLP predicts
#      δ = g / Δt² from the current state, and v⁽⁰⁾ = v_E + Δt²·δ̂.
#   2. Training dump (`dump_training = true`): per step, the pre-solve state and
#      the label δ, formed in Float64 from the converged root before storing.
#   3. Oracle sweep (`oracle_every > 0`): restart the solve from v* + ε·e with
#      ‖e‖ = ‖g‖, which shows how many iterations a predictor that removes all
#      but a fraction ε of the gap would save. No predictor can beat it.
#
# Stdlib only, so train_warmstart.jl can include this file without Mantis/CUDA.

using LinearAlgebra
using Random
using Serialization

# ---- features -----------------------------------------------------------------
#
# Per particle, all from the state the predictor block already has:
#    1-2   v          velocity
#    3-4   v̇          collision RHS at vⁿ (the Euler drift)
#    5-6   G          entropy-gradient field at the particle
#    7-8   ξ          position inside its cell, in [0, 1]
#    9-10  warn       distance to the next knot along v̇, in Euler steps Δt·|v̇|
#                     (clamped): the trajectory is only C¹ where it crosses knots
#   11-12  Δcell      cell sizes
#   13-14  T₁, T₂     ensemble variances (where the relaxation is)
#   15     |v|        speed

const FEATURE_VERSION = 2
const N_FEAT = 15
const WARN_CLAMP = 10.0

function ensemble_temperatures(v::AbstractMatrix, w::AbstractVector)
    s = sum(w)
    m1 = sum(w[α] * v[α, 1] for α in axes(v, 1)) / s
    m2 = sum(w[α] * v[α, 2] for α in axes(v, 1)) / s
    T1 = sum(w[α] * (v[α, 1] - m1)^2 for α in axes(v, 1)) / s
    T2 = sum(w[α] * (v[α, 2] - m2)^2 for α in axes(v, 1)) / s
    return T1, T2
end

# Cell-relative coordinate, cell size and distance to the next knot in the
# direction of motion, for one coordinate on breakpoints `bp`.
@inline function cell_geometry(x::Float64, xdot::Float64, bp::AbstractVector{Float64})
    j = clamp(searchsortedlast(bp, x), 1, length(bp) - 1)
    Δ = bp[j + 1] - bp[j]
    ξ = clamp((x - bp[j]) / Δ, 0.0, 1.0)
    dist = xdot >= 0 ? bp[j + 1] - x : x - bp[j]
    return ξ, Δ, max(dist, 0.0)
end

"""
    build_features!(X, v, dot_v, G, w, bp1, bp2, dt, idx = axes(v, 1)) -> X

Write the features of particles `idx` into the columns of `X` (`N_FEAT × length(idx)`).
"""
function build_features!(X::AbstractMatrix{Float32}, v::AbstractMatrix,
        dot_v::AbstractMatrix, G::AbstractMatrix, w::AbstractVector,
        bp1::AbstractVector{Float64}, bp2::AbstractVector{Float64}, dt::Float64,
        idx = axes(v, 1))
    size(X) == (N_FEAT, length(idx)) || error("X must be ($N_FEAT, $(length(idx)))")
    T1, T2 = ensemble_temperatures(v, w)
    @inbounds for (c, α) in enumerate(idx)
        v1, v2 = Float64(v[α, 1]), Float64(v[α, 2])
        vd1, vd2 = Float64(dot_v[α, 1]), Float64(dot_v[α, 2])
        ξ1, Δ1, d1 = cell_geometry(v1, vd1, bp1)
        ξ2, Δ2, d2 = cell_geometry(v2, vd2, bp2)
        X[1, c] = v1
        X[2, c] = v2
        X[3, c] = vd1
        X[4, c] = vd2
        X[5, c] = G[α, 1]
        X[6, c] = G[α, 2]
        X[7, c] = ξ1
        X[8, c] = ξ2
        X[9, c] = min(d1 / (dt * abs(vd1) + 1e-300), WARN_CLAMP)
        X[10, c] = min(d2 / (dt * abs(vd2) + 1e-300), WARN_CLAMP)
        X[11, c] = Δ1
        X[12, c] = Δ2
        X[13, c] = T1
        X[14, c] = T2
        X[15, c] = hypot(v1, v2)
    end
    return X
end

# ---- MLP: tanh hidden layers, linear output, hand-rolled backprop ---------------

const Layer = Tuple{Matrix{Float32}, Vector{Float32}}

function init_mlp(rng::AbstractRNG; n_in::Int = N_FEAT, hidden::Int = 128,
        depth::Int = 3, n_out::Int = 2)
    sizes = [n_in; fill(hidden, depth); n_out]
    return Layer[(randn(rng, Float32, sizes[l + 1], sizes[l]) .*
                  sqrt(2.0f0 / (sizes[l] + sizes[l + 1])),
                     zeros(Float32, sizes[l + 1])) for l in 1:(length(sizes) - 1)]
end

# Generic over the array type, so the same code runs on CuArrays.
function mlp_forward(layers, X)
    H = X
    for (l, (W, b)) in enumerate(layers)
        Z = W * H .+ b
        H = l == length(layers) ? Z : tanh.(Z)
    end
    return H
end

# Mean squared error ½‖Ŷ − Y‖² / B and its gradient with respect to every layer.
function mlp_loss_grad(layers::Vector{Layer}, X::AbstractMatrix{Float32},
        Y::AbstractMatrix{Float32})
    B = size(X, 2)
    Hs = Matrix{Float32}[X]               # Hs[l] is the input to layer l
    for (l, (W, b)) in enumerate(layers)
        Z = W * Hs[l] .+ b
        push!(Hs, l == length(layers) ? Z : tanh.(Z))
    end
    E = Hs[end] .- Y
    loss = sum(abs2, E) / (2B)
    D = E ./ Float32(B)
    grads = Vector{Layer}(undef, length(layers))
    for l in length(layers):-1:1
        W, _ = layers[l]
        grads[l] = (D * Hs[l]', vec(sum(D; dims = 2)))
        l > 1 && (D = (W' * D) .* (1 .- Hs[l] .^ 2))
    end
    return loss, grads
end

# ---- model: weights + normalisation + training record -----------------------------

struct WarmstartModel
    layers::Vector{Layer}
    μx::Vector{Float32}       # feature normalisation
    σx::Vector{Float32}
    σy::Vector{Float32}       # target scale: the MLP regresses δ ./ σy
    feature_version::Int
    info::Dict{String, Any}   # how it was trained and how well it validated
end

save_warmstart_model(fname::String, m::WarmstartModel) = serialize(fname, m)

function load_warmstart_model(fname::String)
    isfile(fname) || error("NN weights not found: $fname")
    m = deserialize(fname)
    m isa WarmstartModel || error("$fname does not hold a WarmstartModel")
    m.feature_version == FEATURE_VERSION ||
        error("$fname uses features v$(m.feature_version), code has v$FEATURE_VERSION")
    return m
end

# Prediction δ̂ (2 × B, physical units) for normalised features already in X.
predict_delta(m::WarmstartModel, layers, Xn) = mlp_forward(layers, Xn) .* m.σy

# ---- runtime warm start -------------------------------------------------------------

# Where the MLP runs: main() sets this to CuArray when the GPU is enabled.
const NN_TO_DEVICE = Ref{Any}(identity)

"""
    NNWarmstart(model, N; cap = 3.0, to_device = identity)

Inference state for `warmstart = :nn`: the model, its layers on the device that
`to_device` maps to, and a feature buffer for `N` particles. The correction
added to particle α is clipped to `cap` times the mean Euler step ‖Δt·v̇‖.
"""
mutable struct NNWarmstart{F}
    model::WarmstartModel
    layers_dev::Vector{Any}
    to_device::F
    X::Matrix{Float32}
    cap::Float64
end

function NNWarmstart(model::WarmstartModel, N::Int; cap::Float64 = 3.0,
        to_device = identity)
    # invokelatest: CUDA (and so CuArray) is loaded after main() starts.
    layers_dev = Base.invokelatest() do
        Any[(to_device(W), to_device(b)) for (W, b) in model.layers]
    end
    return NNWarmstart(model, layers_dev, to_device, Matrix{Float32}(undef, N_FEAT, N), cap)
end

# Scale on the host: σy is a host vector, which a GPU broadcast cannot read.
_nn_predict(nn::NNWarmstart, Xn) =
    Array(mlp_forward(nn.layers_dev, nn.to_device(Xn))) .* nn.model.σy

"""
    nn_correct!(v1, nn, v, dot_v, G, w, bp1, bp2, dt) -> n_clipped

Add Δt²·δ̂ to the Euler prediction in `v1` and return how many particles had
their correction clipped.
"""
function nn_correct!(v1::AbstractMatrix, nn::NNWarmstart, v::AbstractMatrix,
        dot_v::AbstractMatrix, G::AbstractMatrix, w::AbstractVector,
        bp1::AbstractVector{Float64}, bp2::AbstractVector{Float64}, dt::Float64)
    m = nn.model
    X = build_features!(nn.X, v, dot_v, G, w, bp1, bp2, dt)
    X .= (X .- m.μx) ./ m.σx
    # invokelatest again, for the CUDA matmul methods.
    δ̂ = Base.invokelatest(_nn_predict, nn, X)

    N = size(v, 1)
    cap = nn.cap * dt * sum(hypot(dot_v[α, 1], dot_v[α, 2]) for α in 1:N) / N
    n_clipped = 0
    @inbounds for α in 1:N
        c1, c2 = dt^2 * δ̂[1, α], dt^2 * δ̂[2, α]
        nrm = hypot(c1, c2)
        s = nrm > cap ? cap / nrm : 1.0
        n_clipped += nrm > cap
        v1[α, 1] += s * c1
        v1[α, 2] += s * c2
    end
    return n_clipped
end

# ---- training dump ------------------------------------------------------------------
#
# Header: magic "NNWSDMP2", N, length(bp1), length(bp2) (Int64), bp1, bp2, w
#         (Float64).
# Record: step (Int64), dt (Float64), iter (Int64), r0 (Float64), then vⁿ, v̇ⁿ, Gⁿ
#         and δ = (v* − vⁿ − Δt·v̇ⁿ)/Δt², each an N×2 Float32 matrix stored
#         column-major.
#
# δ is formed in Float64 before it is rounded. Rebuilding it from Float32 copies
# of vⁿ and v^{n+1}, as the first dump format did, leaves rounding noise of
# ulp(v)/Δt² ≈ 0.1–0.5 at Δt = 1e-3, about the size of δ itself. Records are
# fixed-size and self-contained (no pairing of consecutive steps), so a resume
# truncates the dump to the checkpoint and appends.

const DUMP_MAGIC = b"NNWSDMP2"

dump_record_bytes(N::Int) = 4 * 8 + 4 * (2N * 4)

struct DumpHeader
    N::Int
    bp1::Vector{Float64}
    bp2::Vector{Float64}
    w::Vector{Float64}
end

dump_header_bytes(h::DumpHeader) = 8 + 3 * 8 + 8 * (length(h.bp1) + length(h.bp2) + h.N)

function read_dump_header(io::IO)
    magic = read(io, 8)
    magic == DUMP_MAGIC ||
        error("bad dump magic $(String(copy(magic))); this reader needs $(String(copy(DUMP_MAGIC)))")
    N, n1, n2 = (Int(read(io, Int64)) for _ in 1:3)
    rd(n) = read!(io, Vector{Float64}(undef, n))
    return DumpHeader(N, rd(n1), rd(n2), rd(N))
end

"""
    open_training_dump(fname, bp1, bp2, w; resume_step = nothing) -> IO

Start a new dump, or, with `resume_step`, keep the records of an existing one up to
that step and append after them.
"""
function open_training_dump(fname::String, bp1, bp2, w; resume_step = nothing)
    h = DumpHeader(length(w), bp1, bp2, w)
    if resume_step !== nothing && isfile(fname)
        old = open(read_dump_header, fname)
        (old.N == h.N && old.bp1 == h.bp1 && old.bp2 == h.bp2) ||
            error("$fname was written for a different N or mesh")
        nrec = (filesize(fname) - dump_header_bytes(h)) ÷ dump_record_bytes(h.N)
        keep = 0
        open(fname) do io
            for r in 0:(nrec - 1)
                seek(io, dump_header_bytes(h) + r * dump_record_bytes(h.N))
                read(io, Int64) <= resume_step || break
                keep = r + 1
            end
        end
        truncate_to = dump_header_bytes(h) + keep * dump_record_bytes(h.N)
        open(io -> truncate(io, truncate_to), fname, "r+")
        return open(fname, "a")
    end
    io = open(fname, "w")
    write(io, DUMP_MAGIC, Int64(h.N), Int64(length(bp1)), Int64(length(bp2)))
    write(io, Float64.(bp1), Float64.(bp2), Float64.(w))
    flush(io)
    return io
end

"""
    write_dump_record(io, step, dt, iter, r0, v, dot_v, G, v_star, v_euler)

Append one step: the pre-solve state and the label δ = (v_star − v_euler)/Δt².
"""
function write_dump_record(io::IO, step::Int, dt::Float64, iter::Int, r0::Float64,
        v::AbstractMatrix, dot_v::AbstractMatrix, G::AbstractMatrix,
        v_star::AbstractMatrix, v_euler::AbstractMatrix)
    write(io, Int64(step), Float64(dt), Int64(iter), Float64(r0))
    write(io, Float32.(vec(v)), Float32.(vec(dot_v)), Float32.(vec(G)))
    write(io, Float32.(vec((v_star .- v_euler) ./ dt^2)))
    flush(io)
    return nothing
end

"""
    foreach_dump_record(f, fname) -> DumpHeader

Stream the complete records of a dump, calling `f(header, rec)` with
`rec = (; step, dt, iter, r0, v, dot_v, G, δ)` (N×2 Float32 matrices, reused
between calls).
"""
function foreach_dump_record(f, fname::String)
    open(fname) do io
        h = read_dump_header(io)
        bufs = [Matrix{Float32}(undef, h.N, 2) for _ in 1:4]
        # complete records only: a run killed mid-write can leave a partial tail
        nrec = (filesize(fname) - dump_header_bytes(h)) ÷ dump_record_bytes(h.N)
        for _ in 1:nrec
            step = Int(read(io, Int64))
            dt = read(io, Float64)
            iter = Int(read(io, Int64))
            r0 = read(io, Float64)
            foreach(b -> read!(io, b), bufs)
            f(h, (; step, dt, iter, r0, v = bufs[1], dot_v = bufs[2], G = bufs[3],
                δ = bufs[4]))
        end
        return h
    end
end

# ---- oracle sweep ---------------------------------------------------------------------

parse_oracle_eps(s::AbstractString) = parse.(Float64, split(s, ','; keepempty = false))

"""
    oracle_start!(v0, v_star, v_euler, ε, mode, rng) -> v0

A start that leaves a fraction `ε` of the Euler gap g = v_euler − v_star:
`mode = :scale` uses v_star + ε·g (the gap shrunk in place);
`mode = :noise` gives each particle an error of size ε·‖g_α‖ in a random
direction, closer to what an imperfect predictor leaves.
"""
function oracle_start!(v0::AbstractMatrix, v_star::AbstractMatrix,
        v_euler::AbstractMatrix, ε::Float64, mode::Symbol, rng::AbstractRNG)
    if mode === :scale
        @. v0 = v_star + ε * (v_euler - v_star)
    elseif mode === :noise
        @inbounds for α in axes(v0, 1)
            r = ε * hypot(v_euler[α, 1] - v_star[α, 1], v_euler[α, 2] - v_star[α, 2])
            s, c = sincos(2π * rand(rng))
            v0[α, 1] = v_star[α, 1] + r * c
            v0[α, 2] = v_star[α, 2] + r * s
        end
    else
        error("unknown oracle mode $mode")
    end
    return v0
end
