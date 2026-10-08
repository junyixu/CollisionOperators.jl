# warmstart_nn.jl — NN correction on top of the explicit Euler predictor.
#
# The implicit solve per step is v = Φ(v); the Euler predictor
# v⁽⁰⁾ = vⁿ + Δt·v̇ⁿ leaves a gap δ = (v^{n+1,*} − vⁿ − Δt·v̇ⁿ)/Δt² of size
# O(‖v̈‖) dominated by particles crossing spline knots (∇²φ jumps there —
# measured τ_eff ≈ Δt, which is why extrapolation loses to Euler). A small
# per-particle MLP predicts δ from the *current state only* (no history):
#     v⁽⁰⁾ = vⁿ + Δt·v̇ⁿ + Δt²·δ̂.
# The solver and its converged root are untouched; a bad δ̂ only costs
# iterations (and is clamped below to a few Euler-step norms).
#
# Zero non-stdlib deps (LinearAlgebra + Serialization) so this file can be
# included from the hot loop, the trainer, and the probe alike.
#
# Feature vector (Float32, per particle) — FEATURE_VERSION 1:
#    1-2   v          particle velocity
#    3-4   v̇          Euler drift (collision RHS at vⁿ)
#    5-6   G          entropy-gradient field at particle
#    7-8   ξ          cell-relative coords in [0,1]²
#    9-10  warn       knots-until-crossing: dist to next knot along v̇ᵢ,
#                     in units of Δt·|v̇ᵢ| (clamped to WARN_CLAMP)
#   11-12  Δcell      local cell sizes
#   13-14  T₁,T₂      global variances (broadcast; relaxation phase)
#   15     |v|        speed (isotropy coordinate)

using LinearAlgebra
using Random
using Serialization

const FEATURE_VERSION = 1
const N_FEAT = 15
const WARN_CLAMP = 10.0f0

# ---- features ----------------------------------------------------------------

# Weighted variances of the particle ensemble = anisotropic temperatures.
function ensemble_temperatures(v::AbstractMatrix, w::AbstractVector)
    s = sum(w)
    m1 = sum(@inbounds w[α] * v[α, 1] for α in axes(v, 1)) / s
    m2 = sum(@inbounds w[α] * v[α, 2] for α in axes(v, 1)) / s
    T1 = sum(@inbounds w[α] * (v[α, 1] - m1)^2 for α in axes(v, 1)) / s
    T2 = sum(@inbounds w[α] * (v[α, 2] - m2)^2 for α in axes(v, 1)) / s
    return T1, T2
end

# Cell index, relative coordinate, cell size, and distance to the next knot
# in the direction of motion, for one scalar coordinate on breakpoints bp.
@inline function cell_geometry(x::Float64, xdot::Float64, bp::Vector{Float64})
    j = clamp(searchsortedlast(bp, x), 1, length(bp) - 1)
    Δ = bp[j+1] - bp[j]
    ξ = clamp((x - bp[j]) / Δ, 0.0, 1.0)
    dist = xdot >= 0.0 ? bp[j+1] - x : x - bp[j]
    return ξ, Δ, dist
end

"""
    build_features!(X, v, dot_v, G, w, bp1, bp2, dt) -> X

Fill `X :: Matrix{Float32}` of size (N_FEAT, N). All inputs are the per-step
state already available in the main loop's predictor block.
"""
function build_features!(X::Matrix{Float32}, v::AbstractMatrix,
                         dot_v::AbstractMatrix, G::AbstractMatrix,
                         w::AbstractVector,
                         bp1::Vector{Float64}, bp2::Vector{Float64},
                         dt::Float64)
    N = size(v, 1)
    size(X) == (N_FEAT, N) || error("X must be ($N_FEAT, $N)")
    T1, T2 = ensemble_temperatures(v, w)
    @inbounds for α in 1:N
        v1, v2  = v[α, 1], v[α, 2]
        vd1, vd2 = dot_v[α, 1], dot_v[α, 2]
        ξ1, Δ1, d1 = cell_geometry(v1, vd1, bp1)
        ξ2, Δ2, d2 = cell_geometry(v2, vd2, bp2)
        X[1, α]  = v1
        X[2, α]  = v2
        X[3, α]  = vd1
        X[4, α]  = vd2
        X[5, α]  = G[α, 1]
        X[6, α]  = G[α, 2]
        X[7, α]  = ξ1
        X[8, α]  = ξ2
        X[9, α]  = min(Float32(d1 / (dt * abs(vd1) + 1e-30)), WARN_CLAMP)
        X[10, α] = min(Float32(d2 / (dt * abs(vd2) + 1e-30)), WARN_CLAMP)
        X[11, α] = Δ1
        X[12, α] = Δ2
        X[13, α] = T1
        X[14, α] = T2
        X[15, α] = sqrt(v1^2 + v2^2)
    end
    return X
end

build_features(v, dot_v, G, w, bp1, bp2, dt) =
    build_features!(Matrix{Float32}(undef, N_FEAT, size(v, 1)),
                    v, dot_v, G, w, bp1, bp2, dt)

# ---- MLP (tanh, hand-rolled; layers N_FEAT → h → h → h → 2) -------------------

function init_mlp(; hidden::Int=128, seed::Int=0, n_in::Int=N_FEAT)
    rng = Random.Xoshiro(seed)
    glorot(nout, nin) = randn(rng, Float32, nout, nin) .*
                        sqrt(2.0f0 / Float32(nin + nout))
    return (W1=glorot(hidden, n_in), b1=zeros(Float32, hidden),
            W2=glorot(hidden, hidden), b2=zeros(Float32, hidden),
            W3=glorot(hidden, hidden), b3=zeros(Float32, hidden),
            W4=glorot(2, hidden),      b4=zeros(Float32, 2))
end

# Forward pass on a batch X (N_FEAT × B), returns Ŷ (2 × B) in normalized
# target units. Also returns hidden activations for backprop when `cache=true`.
function mlp_forward(θ, X::Matrix{Float32}; cache::Bool=false)
    H1 = tanh.(θ.W1 * X .+ θ.b1)
    H2 = tanh.(θ.W2 * H1 .+ θ.b2)
    H3 = tanh.(θ.W3 * H2 .+ θ.b3)
    Y  = θ.W4 * H3 .+ θ.b4
    return cache ? (Y, H1, H2, H3) : Y
end

# MSE loss gradient. Returns (loss, grads) with grads matching θ's fields.
function mlp_loss_grad(θ, X::Matrix{Float32}, T::Matrix{Float32})
    B = size(X, 2)
    Y, H1, H2, H3 = mlp_forward(θ, X; cache=true)
    E = Y .- T
    loss = sum(abs2, E) / (2 * B)
    dY  = E ./ Float32(B)
    dW4 = dY * H3';                dB4 = vec(sum(dY; dims=2))
    dH3 = θ.W4' * dY;  dZ3 = dH3 .* (1 .- H3 .^ 2)
    dW3 = dZ3 * H2';               dB3 = vec(sum(dZ3; dims=2))
    dH2 = θ.W3' * dZ3; dZ2 = dH2 .* (1 .- H2 .^ 2)
    dW2 = dZ2 * H1';               dB2 = vec(sum(dZ2; dims=2))
    dH1 = θ.W2' * dZ2; dZ1 = dH1 .* (1 .- H1 .^ 2)
    dW1 = dZ1 * X';                dB1 = vec(sum(dZ1; dims=2))
    return loss, (W1=dW1, b1=dB1, W2=dW2, b2=dB2,
                  W3=dW3, b3=dB3, W4=dW4, b4=dB4)
end

# ---- model container (weights + normalization stats) --------------------------

# μx/σx normalize features; σy scales targets so the MLP regresses O(1) values.
struct WarmstartModel
    θ::NamedTuple
    μx::Vector{Float32}
    σx::Vector{Float32}
    σy::Vector{Float32}
    feature_version::Int
end

save_warmstart_model(fname::String, m::WarmstartModel) =
    open(io -> serialize(io, m), fname, "w")

function load_warmstart_model(fname::String)
    isfile(fname) || error("NN weights not found: $fname")
    m = open(deserialize, fname)
    m.feature_version == FEATURE_VERSION ||
        error("weights use feature v$(m.feature_version), code has v$FEATURE_VERSION")
    return m
end

"""
    nn_warmstart_correct!(v1, m, v, dot_v, G, w, bp1, bp2, dt)

Add Δt²·δ̂ to the Euler prediction already stored in `v1`. Per-particle clamp:
‖Δt²δ̂_α‖ ≤ 3 × mean‖Δt·v̇‖, so a bad prediction can cost at most a few
Euler steps' worth of displacement (solver correctness unaffected either way).
"""
function nn_warmstart_correct!(v1::AbstractMatrix, m::WarmstartModel,
                               v::AbstractMatrix, dot_v::AbstractMatrix,
                               G::AbstractMatrix, w::AbstractVector,
                               bp1::Vector{Float64}, bp2::Vector{Float64},
                               dt::Float64)
    N = size(v, 1)
    X = build_features(v, dot_v, G, w, bp1, bp2, dt)
    @inbounds for j in 1:N, i in 1:N_FEAT
        X[i, j] = (X[i, j] - m.μx[i]) / m.σx[i]
    end
    Ŷ = mlp_forward(m.θ, X)

    euler_mean = sum(@inbounds sqrt(dot_v[α, 1]^2 + dot_v[α, 2]^2)
                     for α in 1:N) * dt / N
    cap = 3.0 * euler_mean + 1e-30
    dt2 = dt^2
    @inbounds for α in 1:N
        δ1 = Float64(Ŷ[1, α] * m.σy[1]) * dt2
        δ2 = Float64(Ŷ[2, α] * m.σy[2]) * dt2
        nrm = sqrt(δ1^2 + δ2^2)
        s = nrm > cap ? cap / nrm : 1.0
        v1[α, 1] += s * δ1
        v1[α, 2] += s * δ2
    end
    return v1
end

# ---- training-dump binary format ----------------------------------------------
# Header:  magic "NNWSDMP1" (8 bytes), N::Int64, n_dofs::Int64, dt::Float64.
# Record:  step::Int64, v::2N Float32, dot_v::2N Float32, G::2N Float32,
#          L_vec::n_dofs Float64  (column-major vec of the N×2 matrices).
# Fixed record size ⇒ seekable; appended + flushed per step (crash-safe).

const DUMP_MAGIC = b"NNWSDMP1"

dump_record_bytes(N::Int, n_dofs::Int) = 8 + 3 * (2N * 4) + n_dofs * 8

function open_training_dump(fname::String, N::Int, n_dofs::Int, dt::Float64;
                            append::Bool=false)
    if append && isfile(fname)
        return open(fname, "a")
    end
    io = open(fname, "w")
    write(io, DUMP_MAGIC)
    write(io, Int64(N), Int64(n_dofs), Float64(dt))
    flush(io)
    return io
end

function write_dump_record(io::IO, step::Int, v::AbstractMatrix,
                           dot_v::AbstractMatrix, G::AbstractMatrix,
                           L_vec::AbstractVector)
    write(io, Int64(step))
    write(io, Float32.(vec(v)))
    write(io, Float32.(vec(dot_v)))
    write(io, Float32.(vec(G)))
    write(io, Float64.(L_vec))
    flush(io)
    return nothing
end

function read_dump_header(io::IO)
    magic = read(io, 8)
    magic == DUMP_MAGIC || error("bad dump magic: $magic")
    N      = Int(read(io, Int64))
    n_dofs = Int(read(io, Int64))
    dt     = read(io, Float64)
    return N, n_dofs, dt
end

"""
    foreach_dump_record(f, fname)

Stream records without loading the file: calls
`f(step, v, dot_v, G, L_vec)` with N×2 Float32 matrices (and Float64 L_vec)
for every record. Returns (N, n_dofs, dt).
"""
function foreach_dump_record(f, fname::String)
    open(fname) do io
        N, n_dofs, dt = read_dump_header(io)
        while !eof(io)
            step  = Int(read(io, Int64))
            v     = reshape(reinterpret(Float32, read(io, 2N * 4)) |> collect, N, 2)
            dot_v = reshape(reinterpret(Float32, read(io, 2N * 4)) |> collect, N, 2)
            G     = reshape(reinterpret(Float32, read(io, 2N * 4)) |> collect, N, 2)
            L     = reinterpret(Float64, read(io, n_dofs * 8)) |> collect
            f(step, v, dot_v, G, L)
        end
        return N, n_dofs, dt
    end
end

count_dump_records(fname::String, N::Int, n_dofs::Int) =
    (filesize(fname) - (8 + 8 + 8 + 8)) ÷ dump_record_bytes(N, n_dofs)
