# Unit tests for warmstart.jl: MLP backprop, the training-dump format (including
# the truncation a resume does), features and the oracle starts. The file is
# stdlib-only, so this runs in the core group.
using Test
using Random

include(joinpath(@__DIR__, "..", "..", "warmstart.jl"))

@testset "mlp_loss_grad matches finite differences" begin
    rng = Xoshiro(1)
    layers = init_mlp(rng; n_in = 4, hidden = 5, depth = 2)
    X = randn(rng, Float32, 4, 7)
    Y = randn(rng, Float32, 2, 7)
    _, g = mlp_loss_grad(layers, X, Y)
    loss64(ls) = sum(abs2, mlp_forward([(Float64.(W), Float64.(b)) for (W, b) in ls],
        Float64.(X)) .- Y) / (2 * size(X, 2))
    for (l, j, i) in ((1, 1, 3), (2, 2, 4), (3, 1, 7), (3, 2, 1))
        h = 1.0f-2
        lp, lm = deepcopy(layers), deepcopy(layers)
        lp[l][j][i] += h
        lm[l][j][i] -= h
        @test g[l][j][i] ≈ (loss64(lp) - loss64(lm)) / (2h) rtol = 1e-2
    end
end

@testset "training dump round trip and resume truncation" begin
    N, dt = 6, 1e-3
    bp = collect(-3.0:1.0:3.0)
    w = fill(1 / N, N)
    rng = Xoshiro(2)
    mats(s) = (randn(rng, N, 2), randn(rng, N, 2), randn(rng, N, 2))
    fname = tempname()
    io = open_training_dump(fname, bp, bp, w)
    written = Dict{Int, Any}()
    for step in 1:5
        v, dv, G = mats(step)
        v_star = v .+ dt .* dv .+ dt^2 .* randn(rng, N, 2)
        v_eul = v .+ dt .* dv
        write_dump_record(io, step, dt, 10 + step, 0.1 * step, v, dv, G, v_star, v_eul)
        written[step] = (v, (v_star .- v_eul) ./ dt^2)
    end
    close(io)
    @test filesize(fname) == dump_header_bytes(DumpHeader(N, bp, bp, w)) +
                             5 * dump_record_bytes(N)

    seen = Int[]
    h = foreach_dump_record(fname) do h, rec
        push!(seen, rec.step)
        v, δ = written[rec.step]
        @test rec.v ≈ Float32.(v)
        @test rec.δ ≈ Float32.(δ)
        @test rec.iter == 10 + rec.step
    end
    @test seen == 1:5
    @test h.bp1 == bp && h.w == w

    # A resume from step 3 keeps records 1–3 and appends after them.
    io = open_training_dump(fname, bp, bp, w; resume_step = 3)
    v, dv, G = mats(9)
    write_dump_record(io, 4, dt, 0, 0.0, v, dv, G, v, v)
    close(io)
    seen = Int[]
    foreach_dump_record((h, rec) -> push!(seen, rec.step), fname)
    @test seen == [1, 2, 3, 4]
    # a run killed mid-record leaves a partial tail; readers stop before it
    open(io -> write(io, zeros(UInt8, 100)), fname, "a")
    seen = Int[]
    foreach_dump_record((h, rec) -> push!(seen, rec.step), fname)
    @test seen == [1, 2, 3, 4]
    @test_throws ErrorException open_training_dump(fname, bp, bp[1:(end - 1)], w;
        resume_step = 1)
    rm(fname)
end

@testset "features" begin
    bp = [-2.0, -1.0, 0.0, 1.0, 2.0]
    v = [0.25 -0.5; -1.5 1.9]
    dv = [1.0 -1.0; -2.0 0.0]
    X = Matrix{Float32}(undef, N_FEAT, 2)
    build_features!(X, v, dv, zero(v), [0.5, 0.5], bp, bp, 0.1)
    @test X[7, 1] ≈ 0.25 && X[8, 1] ≈ 0.5            # ξ inside the cell
    @test X[9, 1] ≈ 0.75 / 0.1                       # 0.75 to the knot at 1, Euler step 0.1
    @test X[10, 1] ≈ 0.5 / 0.1                       # moving down: 0.5 to the knot at -1
    @test X[10, 2] == WARN_CLAMP                     # not moving
    @test X[11, 1] == 1.0
    @test X[15, 2] ≈ hypot(1.5, 1.9)
    @test all(isfinite, X)
end

@testset "oracle starts leave a fraction ε of the gap" begin
    rng = Xoshiro(3)
    v_star = randn(rng, 50, 2)
    v_eul = v_star .+ 1e-3 .* randn(rng, 50, 2)
    v0 = similar(v_star)
    gap = sqrt(sum(abs2, v_eul .- v_star))
    for mode in (:scale, :noise), ε in (1.0, 0.1, 0.0)
        oracle_start!(v0, v_star, v_eul, ε, mode, rng)
        @test sqrt(sum(abs2, v0 .- v_star)) ≈ ε * gap atol = 1e-15
    end
    @test parse_oracle_eps("1,0.1,0") == [1.0, 0.1, 0.0]
end
