#! /usr/bin/env -S julia --startup-file=no
# Offline trainer for the NN warm start (model and dump format: warmstart.jl).
#
#   julia --startup-file=no train_warmstart.jl --train=a.bin[,b.bin…] --out=nn.jls \
#         [--val=c.bin,…] [--val_frac=0.2] [--epochs=30] [--hidden=128] [--depth=3] \
#         [--lr=1e-3] [--batch=4096] [--per_step=1000] [--seed=0] [--stag_window=30]
#   julia --startup-file=no train_warmstart.jl --eval=nn.jls --val=c.bin[,…] [--stag_window=30]
#
# Samples `per_step` random particles from every record. Validation is either the
# `--val` dumps (a later time window, another seed) or, without them, the last
# `val_frac` of each training dump's steps: a random split would leak between
# adjacent steps, which are nearly identical. Steps whose solve left through the
# stagnation exit (iter a multiple of the run's `stag_window`; 0 keeps all) are
# skipped: their v* is the best iterate, not the root, so their label is off.
#
# The score is decades = log10(‖δ‖ / ‖δ − δ̂‖) over the validation samples: how many
# orders of magnitude the predictor takes off the Euler gap. Compare it with the
# oracle sweep (oracle_<suffix>.csv) to read off the iterations it would save.
# A ridge regression on the same features is printed as a baseline.

using LinearAlgebra
using Printf
using Random
include(joinpath(@__DIR__, "warmstart.jl"))

function parse_cli(args)
    opts = Dict{String, String}()
    for tok in args
        m = match(r"^--(\w+)=(.*)$", tok)
        m === nothing && error("bad option $tok (want --key=value)")
        opts[m.captures[1]] = m.captures[2]
    end
    return opts
end

files(s) = String.(split(s, ','; keepempty = false))

# Features and labels of `per_step` particles from every converged record of each
# dump, with the record's position in its dump as a fraction in [0, 1).
function collect_samples(dumps; per_step::Int, rng::AbstractRNG, stag::Int)
    Xs, Ys, pos = Matrix{Float32}[], Matrix{Float32}[], Float64[]
    for f in dumps
        h = open(read_dump_header, f)
        nrec = (filesize(f) - dump_header_bytes(h)) ÷ dump_record_bytes(h.N)
        nrec > 0 || error("$f has no records")
        k = min(per_step, h.N)
        r = 0; skipped = 0
        foreach_dump_record(f) do h, rec
            r += 1
            if stag > 0 && rec.iter % stag == 0
                skipped += 1
                return
            end
            idx = randperm(rng, h.N)[1:k]
            X = Matrix{Float32}(undef, N_FEAT, k)
            build_features!(X, rec.v, rec.dot_v, rec.G, h.w, h.bp1, h.bp2, rec.dt, idx)
            push!(Xs, X)
            push!(Ys, permutedims(rec.δ[idx, :]))
            append!(pos, fill((r - 1) / nrec, k))
        end
        @printf("%s: N=%d, %d records (%d stalled, skipped), %d samples\n",
            f, h.N, nrec, skipped, k * (nrec - skipped))
    end
    return reduce(hcat, Xs), reduce(hcat, Ys), pos
end

decades(Y, Ŷ) = log10(sqrt(sum(abs2, Y)) / sqrt(sum(abs2, Y .- Ŷ)))

function ridge_decades(Xtr, Ytr, Xva, Yva; λ = 1e-6)
    A(X) = [Float64.(X); ones(1, size(X, 2))]
    At = A(Xtr)
    B = (At * At' + λ * size(At, 2) * I) \ (At * Float64.(Ytr)')
    return decades(Float64.(Yva), (A(Xva)' * B)')
end

function train(opts)
    rng = Xoshiro(parse(Int, get(opts, "seed", "0")))
    epochs = parse(Int, get(opts, "epochs", "30"))
    hidden = parse(Int, get(opts, "hidden", "128"))
    depth = parse(Int, get(opts, "depth", "3"))
    lr0 = parse(Float32, get(opts, "lr", "1e-3"))
    batch = parse(Int, get(opts, "batch", "4096"))
    per_step = parse(Int, get(opts, "per_step", "1000"))
    stag = parse(Int, get(opts, "stag_window", "30"))
    out = opts["out"]

    X, Y, pos = collect_samples(files(opts["train"]); per_step, rng, stag)
    if haskey(opts, "val")
        Xva, Yva, _ = collect_samples(files(opts["val"]); per_step, rng, stag)
        Xtr, Ytr = X, Y
    else
        cut = 1 - parse(Float64, get(opts, "val_frac", "0.2"))
        tr, va = pos .< cut, pos .>= cut
        Xtr, Ytr, Xva, Yva = X[:, tr], Y[:, tr], X[:, va], Y[:, va]
    end
    @printf("train %d samples, val %d samples\n", size(Xtr, 2), size(Xva, 2))
    @printf("ridge baseline: val decades %.3f\n", ridge_decades(Xtr, Ytr, Xva, Yva))

    μx = vec(sum(Xtr; dims = 2)) ./ size(Xtr, 2)
    σx = sqrt.(vec(sum(abs2, Xtr .- μx; dims = 2)) ./ size(Xtr, 2)) .+ 1.0f-8
    σy = sqrt.(vec(sum(abs2, Ytr; dims = 2)) ./ size(Ytr, 2)) .+ 1.0f-30
    Xn, Yn, Xvn = (Xtr .- μx) ./ σx, Ytr ./ σy, (Xva .- μx) ./ σx

    layers = init_mlp(rng; hidden, depth)
    mom = [(zero(W), zero(b)) for (W, b) in layers]
    vel = [(zero(W), zero(b)) for (W, b) in layers]
    β1, β2, ϵ = 0.9f0, 0.999f0, 1.0f-8
    t = 0
    nb = cld(size(Xn, 2), batch)
    best = -Inf
    for ep in 1:epochs
        perm = randperm(rng, size(Xn, 2))
        loss_sum = 0.0
        for k in 1:nb
            cols = perm[((k - 1) * batch + 1):min(k * batch, end)]
            loss, g = mlp_loss_grad(layers, Xn[:, cols], Yn[:, cols])
            loss_sum += loss
            t += 1
            lr = lr0 * Float32(0.5 * (1 + cospi(t / (epochs * nb))))   # cosine decay
            for l in eachindex(layers), j in 1:2
                θ, m, v, gj = layers[l][j], mom[l][j], vel[l][j], g[l][j]
                @. m = β1 * m + (1 - β1) * gj
                @. v = β2 * v + (1 - β2) * gj^2
                @. θ -= lr * (m / (1 - β1^t)) / (sqrt(v / (1 - β2^t)) + ϵ)
            end
        end
        dva = decades(Yva, mlp_forward(layers, Xvn) .* σy)
        saved = dva > best
        if saved
            best = dva
            info = Dict{String, Any}("val_decades" => dva, "epoch" => ep,
                "train" => opts["train"], "val" => get(opts, "val", "time split"),
                "hidden" => hidden, "depth" => depth, "per_step" => per_step)
            save_warmstart_model(out,
                WarmstartModel(deepcopy(layers), μx, σx, σy, FEATURE_VERSION, info))
        end
        @printf("epoch %3d  train loss %.4g  val decades %.3f%s\n",
            ep, loss_sum / nb, dva, saved ? "  [saved]" : "")
    end
    @printf("\nbest val decades %.3f → %s\ndeploy: --warmstart=nn --nn_weights=%s\n",
        best, out, out)
end

function evaluate(opts)
    m = load_warmstart_model(opts["eval"])
    println("model: ", m.info)
    for f in files(opts["val"])
        X, Y, pos = collect_samples([f]; per_step = parse(Int, get(opts, "per_step", "1000")),
            rng = Xoshiro(0), stag = parse(Int, get(opts, "stag_window", "30")))
        Ŷ = predict_delta(m, m.layers, (X .- m.μx) ./ m.σx)
        thirds = [findall(p -> (k - 1) / 3 <= p < k / 3, pos) for k in 1:3]
        @printf("%s: decades %.3f  (by thirds of the run: %s)\n", f, decades(Y, Ŷ),
            join((@sprintf("%.3f", decades(Y[:, i], Ŷ[:, i])) for i in thirds), ", "))
    end
end

opts = parse_cli(ARGS)
haskey(opts, "eval") ? evaluate(opts) : train(opts)
