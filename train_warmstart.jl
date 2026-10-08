# Offline trainer for the NN warm start (see warmstart_nn.jl for the model
# and PLAN_nn_warmstart.md for the pipeline). Hand-rolled Adam + backprop —
# zero non-stdlib deps.
#
# Data: training_dump_<suffix>.bin from a run with dump_training=true.
# Labels δ = (v^{n+1,*} − vⁿ − Δt·v̇ⁿ)/Δt² from consecutive records.
# Split: by time blocks (last val_frac of step pairs = validation) — random
# splits would leak across adjacent, highly correlated steps.
#
# Run as (systemd unit for real training, foreground ok for short tests):
#   julia --project=. train_warmstart.jl training_dump_<suffix>.bin \
#         nn_warmstart_<suffix>.jls [--epochs=30] [--hidden=128] [--lr=1e-3] \
#         [--batch=4096] [--max_samples=2000000] [--val_frac=0.2] [--seed=0]
include("warmstart_nn.jl")

length(ARGS) >= 2 || error("usage: train_warmstart.jl <dump.bin> <out.jls> [--key=val …]")
dump_file, out_file = ARGS[1], ARGS[2]
opts = Dict{String,String}()
for tok in ARGS[3:end]
    m = match(r"^--(\w+)=(.+)$", tok)
    m === nothing && error("bad option $tok")
    opts[m.captures[1]] = m.captures[2]
end
epochs      = parse(Int,     get(opts, "epochs",      "30"))
hidden      = parse(Int,     get(opts, "hidden",      "128"))
lr          = parse(Float32, get(opts, "lr",          "1e-3"))
batch       = parse(Int,     get(opts, "batch",       "4096"))
max_samples = parse(Int,     get(opts, "max_samples", "2000000"))
val_frac    = parse(Float64, get(opts, "val_frac",    "0.2"))
seed        = parse(Int,     get(opts, "seed",        "0"))

N_dump, n_dofs, dt = open(read_dump_header, dump_file)
n_records = count_dump_records(dump_file, N_dump, n_dofs)
n_pairs   = n_records - 1
n_pairs >= 10 || error("dump too short: $n_records records")
per_pair  = clamp(ceil(Int, max_samples / n_pairs), 1, N_dump)

# Breakpoints from the step-0 fs snapshot next to the dump (same logic as probe).
function load_breakpoints(dump_file)
    m = match(r"training_dump_(.+)\.bin", basename(dump_file))
    if m !== nothing
        snap = "fs_snapshot_$(m.captures[1])_step0000.csv"
        if isfile(snap)
            bp1 = Float64[]; bp2 = Float64[]
            for ln in eachline(snap)
                startswith(ln, "# bp1=") && (bp1 = parse.(Float64, split(ln[8:end], ',')))
                startswith(ln, "# bp2=") && (bp2 = parse.(Float64, split(ln[8:end], ',')))
                isempty(bp1) || isempty(bp2) || return bp1, bp2
            end
        end
    end
    @warn "breakpoints not found next to dump; using sq_d04 preset mesh"
    bp = [-6.0; -5.0; collect(LinRange(-4.0, 4.0, 21)); 5.0; 6.0]
    return bp, copy(bp)
end
const BP1, BP2 = load_breakpoints(dump_file)

println("dump: N=$N_dump n_dofs=$n_dofs dt=$dt records=$n_records " *
        "→ $per_pair samples/pair, ≤$(per_pair * n_pairs) total")

# ---- collect samples (streaming) ----------------------------------------------
rng = Random.Xoshiro(seed)
S_cap = per_pair * n_pairs
X = Matrix{Float32}(undef, N_FEAT, S_cap)
Y = Matrix{Float32}(undef, 2, S_cap)
pair_of = Vector{Int}(undef, S_cap)   # pair index per sample (for time split)

s_used = 0
pair_i = 0
prev = nothing
foreach_dump_record(dump_file) do step, v, dot_v, G, L
    global prev, s_used, pair_i
    if prev !== nothing && step == prev.step + 1
        pair_i += 1
        N = size(v, 1)
        w = fill(1.0 / N, N)
        vp  = Float64.(prev.v);  dvp = Float64.(prev.dot_v)
        Xfull = build_features(vp, dvp, Float64.(prev.G), w, BP1, BP2, dt)
        idx = rand(rng, 1:N, per_pair)
        @inbounds for α in idx
            s_used += 1
            for i in 1:N_FEAT
                X[i, s_used] = Xfull[i, α]
            end
            Y[1, s_used] = (Float64(v[α, 1]) - vp[α, 1] - dt * dvp[α, 1]) / dt^2
            Y[2, s_used] = (Float64(v[α, 2]) - vp[α, 2] - dt * dvp[α, 2]) / dt^2
            pair_of[s_used] = pair_i
        end
    end
    prev = (; step, v, dot_v, G)
end
X = X[:, 1:s_used]; Y = Y[:, 1:s_used]; pair_of = pair_of[1:s_used]
println("collected $s_used samples from $pair_i pairs")

# ---- normalize ------------------------------------------------------------------
μx = vec(sum(X; dims=2)) ./ s_used
σx = sqrt.(vec(sum(abs2, X .- μx; dims=2)) ./ s_used) .+ 1.0f-8
σy = sqrt.(vec(sum(abs2, Y; dims=2)) ./ s_used) .+ 1.0f-30
Xn = (X .- μx) ./ σx
Yn = Y ./ σy

# ---- time-block split ------------------------------------------------------------
val_start_pair = ceil(Int, (1 - val_frac) * pair_i)
train_idx = findall(<(val_start_pair), pair_of)
val_idx   = findall(>=(val_start_pair), pair_of)
println("train=$(length(train_idx))  val=$(length(val_idx))  " *
        "(val = pairs ≥ $val_start_pair of $pair_i)")

# ---- Adam ------------------------------------------------------------------------
θ = init_mlp(; hidden, seed)
adam_m = map(zero, θ); adam_v = map(zero, θ)
β1, β2, ϵ = 0.9f0, 0.999f0, 1.0f-8
t_adam = 0

function adam_step!(θ, g)
    global t_adam += 1
    bc1 = 1 - β1^t_adam; bc2 = 1 - β2^t_adam
    for k in keys(θ)
        @. adam_m[k] = β1 * adam_m[k] + (1 - β1) * g[k]
        @. adam_v[k] = β2 * adam_v[k] + (1 - β2) * g[k]^2
        @. θ[k] -= lr * (adam_m[k] / bc1) / (sqrt(adam_v[k] / bc2) + ϵ)
    end
end

val_loss(θ) = begin
    Ŷ = mlp_forward(θ, Xn[:, val_idx])
    sum(abs2, Ŷ .- Yn[:, val_idx]) / (2 * length(val_idx))
end
# decades on the *unnormalized* residual, the deployment-relevant number
val_decades(θ) = begin
    Ŷ = mlp_forward(θ, Xn[:, val_idx]) .* σy
    rms_y = sqrt(sum(abs2, Y[:, val_idx]) / length(val_idx))
    rms_r = sqrt(sum(abs2, Ŷ .- Y[:, val_idx]) / length(val_idx))
    log10(rms_y / rms_r)
end

best_val = Inf
for ep in 1:epochs
    perm = Random.shuffle(rng, train_idx)
    tr_loss = 0.0; nb = 0
    for lo in 1:batch:length(perm)
        cols = perm[lo:min(lo + batch - 1, length(perm))]
        loss, g = mlp_loss_grad(θ, Xn[:, cols], Yn[:, cols])
        adam_step!(θ, g)
        tr_loss += loss; nb += 1
    end
    vl = val_loss(θ)
    vd = val_decades(θ)
    marker = ""
    if vl < best_val
        global best_val = vl
        save_warmstart_model(out_file,
            WarmstartModel(map(copy, θ), Float32.(μx), Float32.(σx),
                           Float32.(σy), FEATURE_VERSION))
        marker = "  [saved]"
    end
    println("epoch $ep/$epochs  train=$(round(tr_loss/nb; sigdigits=4))  " *
            "val=$(round(vl; sigdigits=4))  val_decades=$(round(vd; digits=2))$marker")
end

println("\nbest model → $out_file")
println("deploy: --warmstart=nn --nn_weights=$out_file")
