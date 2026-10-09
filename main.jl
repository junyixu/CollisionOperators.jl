#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
#
# Gonzalez discrete-gradient time integration of the Landau collision
# operator. Refactored entry point with CLI configuration:
#
#   julia --project=. main.jl parameters_default.jl
#   julia --project=. main.jl parameters_picard.jl
#   julia --project=. main.jl parameters_default.jl \
#         --N_STEPS=200 --use_anderson=false --suffix=picard_short
#
# ARGS[1]   : path to a preset file that defines `PARAMS::SimParameters`
# ARGS[2:]  : zero or more `--key=value` overrides applied on top of PARAMS
#
# Diagnostics pushed to CSV every step:
#   iter        : Picard-map evaluations used (Anderson or Newton–Krylov)
#   residual    : ‖G(v) − v‖₂ at the converged iterate
#   fp_minus_fs : ‖f_s − f_p‖₂  (histogram-based projection-error norm)
#   neg_part    : ∫ max(−f_s, 0) dv  (Gibbs negative-part L¹)

include("MantisWrappers.jl")
using .MantisWrappers

using Random
using LinearAlgebra: ldiv!

# The rest of the driver, included into `Main` so the `julia main.jl <preset>`
# entry point and the runtime GPU swap below keep working unchanged.
include("io.jl")       # rclone mirror, params record, checkpoints, cons CSV
include("solver.jl")   # Picard map + Anderson solve, CPU/GPU hot-loop hooks
include("plots.jl")    # fs_snapshot CSV + diagnostics PNGs
include("warmstart.jl")   # NN warm start, training dump, oracle sweep

function run_simulation(p::SimParameters; resume = nothing)
    print_summary(p)
    Random.seed!(p.seed)

    # Select entropy/seed integrand variant for compute_entropy / compute_r!.
    USE_LOGSQ[] = p.use_logsq

    ws = build_workspace(p)
    println("Workspace: n_dofs=$(ws.n_dofs)  n_elements=$(ws.n_elements)")

    # ---- State init: either fresh sample or resume from checkpoint ----
    cons_csv = "conservation_history_$(p.suffix).csv"
    snap_csv = "particle_snapshots_$(p.suffix).csv"
    start_step = 0
    t_start = 0.0          # physical time at start_step
    local v_particles, w_particles, f_coeffs
    local entropy_history, energy_history, momentum_history
    local iter_history, res_history, fp_l2_history, neg_history

    if resume !== nothing
        ckpt = load_checkpoint(p.suffix, resume)
        start_step = ckpt.step
        v_particles = ckpt.v_particles
        w_particles = ckpt.w_particles
        f_coeffs = ckpt.f_coeffs
        entropy_history = ckpt.entropy_history
        energy_history = ckpt.energy_history
        momentum_history = ckpt.momentum_history
        iter_history = ckpt.iter_history
        res_history = ckpt.res_history
        fp_l2_history = ckpt.fp_l2_history
        neg_history = ckpt.neg_history
        copy!(Random.default_rng(), ckpt.rng_state)

        size(v_particles, 1) == p.N_PARTICLES || error(
            "Checkpoint N_PARTICLES=$(size(v_particles,1)) ≠ preset N_PARTICLES=$(p.N_PARTICLES)")
        length(f_coeffs) == ws.n_dofs || error(
            "Checkpoint n_dofs=$(length(f_coeffs)) ≠ workspace n_dofs=$(ws.n_dofs); mesh changed?")
        start_step < p.N_STEPS || error(
            "Checkpoint step=$start_step ≥ N_STEPS=$(p.N_STEPS); nothing to do")

        # The CSV may predate the `dt` column; normalise it before reading
        # t_start out of it or appending rows in the new layout.
        migrate_cons_csv!(cons_csv)

        # Physical time across the resume: the checkpoint if it has one, else
        # the migrated CSV, else assume DT never changed up to here.
        t_start = hasproperty(ckpt, :t) ? ckpt.t :
                  something(cons_time_at(cons_csv, start_step),
                      start_step * p.DT)

        println("Resuming from step $start_step, t=$t_start " *
                "(running through $(p.N_STEPS) at DT=$(p.DT))")
    else
        v_particles = zeros(p.N_PARTICLES, 2)
        if p.v1_peak != 0.0
            # Bimodal in v₁ (ported from the bimodal-v1 branch): balanced 50/50
            # split centered at ±v1_peak so the net v₁ momentum is exactly zero
            # regardless of N. v₂ stays N(0,σ2²).
            half = p.N_PARTICLES ÷ 2
            centers = [fill(-p.v1_peak, half); fill(p.v1_peak, p.N_PARTICLES - half)]
            v_particles[:, 1] .= centers .+ p.σ1 .* randn(p.N_PARTICLES)
        else
            v_particles[:, 1] .= p.σ1 * randn(p.N_PARTICLES)
        end
        v_particles[:, 2] .= p.σ2 * randn(p.N_PARTICLES)
        w_particles = fill(1.0 / p.N_PARTICLES, p.N_PARTICLES)
        f_coeffs = zeros(ws.n_dofs)

        l2_project!(ws, f_coeffs, v_particles, w_particles)

        entropy_history = Float64[]
        energy_history = Float64[]
        momentum_history = NTuple{2, Float64}[]
        iter_history = Int[]
        res_history = Float64[]
        fp_l2_history = Float64[]
        neg_history = Float64[]

        f_s0 = build_field(ws, f_coeffs)
        push!(entropy_history, compute_entropy(ws, f_s0))
        push!(energy_history, compute_energy(v_particles, w_particles))
        push!(momentum_history, compute_momentum(v_particles, w_particles))
        println("Initial  S_h = $(entropy_history[end])")
        println("Initial  E   = $(energy_history[end])")
        println("Initial  P   = $(momentum_history[end])")
        println("Initial  ‖f_s − f_p‖₂ = " *
                string(compute_fs_minus_fp_l2(ws, f_s0, v_particles, w_particles)))
        println("Initial  ∫max(−f_s,0) = " *
                string(compute_negative_part_l1(ws, f_s0)))
    end

    f_s = build_field(ws, f_coeffs)

    r_vec = zeros(ws.n_dofs)
    L_vec = zeros(ws.n_dofs)
    G = zeros(p.N_PARTICLES, 2)
    dot_v = zeros(p.N_PARTICLES, 2)
    v1 = copy(v_particles)

    v_mid = similar(v_particles)
    dv = similar(v_particles)
    dS_mid = zeros(p.N_PARTICLES, 2)
    G_eff = zeros(p.N_PARTICLES, 2)
    f_buf = zeros(ws.n_dofs)

    Gv = zeros(p.N_PARTICLES, 2)
    r_curr = zeros(p.N_PARTICLES, 2)
    r_prev = zeros(p.N_PARTICLES, 2)
    Gv_prev = zeros(p.N_PARTICLES, 2)
    v_old_buf = zeros(p.N_PARTICLES, 2)
    ΔF = zeros(2 * p.N_PARTICLES, p.m_anderson)
    ΔG = zeros(2 * p.N_PARTICLES, p.m_anderson)

    p.solver in (:anderson, :newton) || error("unknown solver=$(p.solver)")
    nk = p.solver === :newton ? NKWorkspace(2 * p.N_PARTICLES, p.nk_krylov_max) :
         nothing
    nk_fd_h = p.nk_fd_h > 0 ? p.nk_fd_h : (p.use_gpu && p.gpu_fp32 ? 1e-5 : 1e-6)

    # Warm start (stateless, so a resume needs nothing from the checkpoint).
    p.warmstart in (:euler, :nn) || error("unknown warmstart=$(p.warmstart)")
    nn = p.warmstart === :nn ?
         NNWarmstart(load_warmstart_model(p.nn_weights), p.N_PARTICLES;
        cap = p.nn_cap, to_device = NN_TO_DEVICE[]) : nothing
    nn === nothing || println("NN warm start: $(p.nn_weights)  " *
                              "$(nn.model.info)")
    v_euler = similar(v_particles)   # the Euler predictor, kept for the label / oracle

    # Training dump: v̇ and G are scratch inside the solve, so stage them first.
    dump_file = "training_dump_$(p.suffix).bin"
    dump_io = p.dump_training ?
              open_training_dump(dump_file, ws.bp1, ws.bp2, w_particles;
        resume_step = start_step > 0 ? start_step : nothing) : nothing
    dump_dotv = p.dump_training ? similar(dot_v) : dot_v
    dump_G = p.dump_training ? similar(G) : G

    oracle_eps = parse_oracle_eps(p.oracle_eps)
    oracle_csv = "oracle_$(p.suffix).csv"
    oracle_io = if p.oracle_every <= 0
        nothing
    elseif start_step > 0 && isfile(oracle_csv)
        truncate_csv_after(oracle_csv, start_step)
        open(oracle_csv, "a")
    else
        io = open(oracle_csv, "w")
        println(io, "step,mode,eps,iter,r0,residual")
        io
    end
    v_star = p.oracle_every > 0 ? similar(v_particles) : v_particles
    v_try = p.oracle_every > 0 ? similar(v_particles) : v_particles

    # Snapshot every 25 steps (plus final step if not already a multiple of 25).
    # Crash-safe: conservation + particles appended per-step / per-snapshot so a
    # killed run still leaves usable data through the last completed step.
    snapshot_steps = Set(0:p.snap_every:p.N_STEPS)
    push!(snapshot_steps, p.N_STEPS)
    snapshots_v = Dict{Int, Matrix{Float64}}()

    if start_step == 0
        snapshots_v[0] = copy(v_particles)
        save_fs_snapshot(ws, p.suffix, 0, f_coeffs)
        plot_fs_diagnostics(ws, f_coeffs, p.suffix, 0)

        cons_io = open(cons_csv, "w")
        println(cons_io, join(CONS_COLS, ','))
        println(cons_io,
            "0,0.0,$(entropy_history[1]),$(energy_history[1])," *
            "$(momentum_history[1][1]),$(momentum_history[1][2])," *
            "0,0.0,0.0,0.0,0.0,0.0")
        flush(cons_io)
        rclone_upload(p.suffix, cons_csv)

        snap_io = open(snap_csv, "w")
        println(snap_io, "step,time,particle_idx,v1,v2")
        for i in axes(v_particles, 1)
            println(snap_io, "0,0.0,$i,$(v_particles[i, 1]),$(v_particles[i, 2])")
        end
        flush(snap_io)

        # Write a step-0 checkpoint so future runs can resume even before any
        # time-stepping completed (cheap and uniform).
        save_checkpoint(p.suffix, 0, 0.0, v_particles, w_particles, f_coeffs,
            entropy_history, energy_history, momentum_history,
            iter_history, res_history, fp_l2_history, neg_history,
            copy(Random.default_rng()))
    else
        # Resume: keep existing CSV rows ≤ start_step, append from now on.
        truncate_csv_after(cons_csv, start_step)
        truncate_csv_after(snap_csv, start_step)
        cons_io = open(cons_csv, "a")
        snap_io = open(snap_csv, "a")
    end

    for step in (start_step + 1):p.N_STEPS
        S0 = entropy_history[end]

        # Explicit predictor for the Anderson initial guess. Landau uses the
        # FEM-projected log-gradient G = ∇(M⁻¹r); LB uses the direct clamped
        # ∇f_s/f_s (consistent with its picard map — FEM seed destabilises LB).
        if p.collision_model == :lb
            Base.invokelatest(LOGGRAD_FN[], ws, G, v_particles, f_coeffs)
            n_lb, U1_lb, U2_lb, Q_lb = compute_moments(v_particles, w_particles)
            A1_lb, A2_lb, B_lb = compute_drift_multipliers(v_particles, w_particles,
                G, n_lb, U1_lb, U2_lb, Q_lb)
            compute_LB_velocity!(dot_v, v_particles, G, A1_lb, A2_lb, B_lb, p.nu)
        else
            compute_r!(ws, r_vec, f_s)
            ldiv!(L_vec, ws.M_lu, r_vec)
            Base.invokelatest(COMPG_FN[], ws, G, v_particles, L_vec)
            Base.invokelatest(COLLISION_FN[], ws, dot_v, v_particles, w_particles, G)
        end
        @. v_euler = v_particles + p.DT * dot_v
        v1 .= v_euler
        if dump_io !== nothing
            dump_dotv .= dot_v
            dump_G .= G
        end
        if nn !== nothing
            t_nn = @elapsed n_clip = nn_correct!(v1, nn, v_particles, dot_v, G,
                w_particles, ws.bp1, ws.bp2, p.DT)
            (step <= start_step + 3 || step % 25 == 0) &&
                println("  nn: $(round(1e3 * t_nn; digits = 1)) ms, clipped $n_clip")
        end

        # One implicit step from the guess in `v`, which leaves holding the root.
        implicit_solve!(v; verbose = false) =
            if nk === nothing
                step_anderson!(ws,
                    v, v_particles, w_particles, S0, p.DT,
                    v_mid, dv, dS_mid, G_eff, dot_v, f_buf,
                    r_vec, L_vec, G,
                    Gv, r_curr, r_prev, Gv_prev, v_old_buf, ΔF, ΔG;
                    m = p.m_anderson, max_iter = p.max_iter, tol = p.tol,
                    abs_floor = p.abs_floor,
                    stag_window = p.stag_window,
                    stag_rel_tol = p.stag_rel_tol,
                    damp_decay_start = p.damp_decay_start,
                    damp_decay_factor = p.damp_decay_factor,
                    damping = p.damping, use_anderson = p.use_anderson,
                    use_gonzalez = p.use_gonzalez,
                    exit_picard_step = p.exit_picard_step,
                    verbose)
            else
                step_newton!(ws,
                    v, v_particles, w_particles, S0, p.DT,
                    v_mid, dv, dS_mid, G_eff, dot_v, f_buf,
                    r_vec, L_vec, G, Gv, nk;
                    max_iter = p.max_iter, tol = p.tol, abs_floor = p.abs_floor,
                    fd_h = nk_fd_h, fd_rel = p.nk_fd_rel, eta_max = p.nk_eta_max,
                    use_gonzalez = p.use_gonzalez,
                    exit_picard_step = p.exit_picard_step,
                    verbose)
            end
        iter, res_final, n_rs, r0_init = implicit_solve!(v1;
            verbose = (step <= start_step + 3))

        dump_io !== nothing &&
            write_dump_record(dump_io, step, p.DT, iter, r0_init,
                v_particles, dump_dotv, dump_G, v1, v_euler)

        # Oracle: restart from v* plus a fraction ε of the Euler gap. Scale ε = 1
        # is the Euler start again, so its row should repeat `iter`.
        if oracle_io !== nothing && step % p.oracle_every == 0
            v_star .= v1
            rng = Random.Xoshiro(step)   # not the global RNG: that one is checkpointed
            for mode in (:scale, :noise), ε in oracle_eps
                mode === :noise && ε == 0 && continue
                oracle_start!(v_try, v_star, v_euler, ε, mode, rng)
                it, res, _, r0 = implicit_solve!(v_try)
                println(oracle_io, "$step,$mode,$ε,$it,$r0,$res")
            end
            flush(oracle_io)
        end
        v_particles .= v1

        l2_project!(ws, f_coeffs, v_particles, w_particles)
        f_s = build_field(ws, f_coeffs)

        push!(entropy_history, compute_entropy(ws, f_s))
        push!(energy_history, compute_energy(v_particles, w_particles))
        push!(momentum_history, compute_momentum(v_particles, w_particles))
        push!(iter_history, iter)
        push!(res_history, res_final)
        push!(fp_l2_history, compute_fs_minus_fp_l2(ws, f_s, v_particles, w_particles))
        push!(neg_history, compute_negative_part_l1(ws, f_s))

        # Accumulated from t_start, so a resume at a different DT extends the
        # history instead of rescaling it.
        t = t_start + (step - start_step) * p.DT

        # Append this step's conservation row (crash-safe).
        let P = momentum_history[end]
            println(cons_io,
                "$step,$t,$(entropy_history[end]),$(energy_history[end])," *
                "$(P[1]),$(P[2]),$iter,$res_final," *
                "$(fp_l2_history[end]),$(neg_history[end]),$r0_init,$(p.DT)")
            flush(cons_io)
        end

        if step in snapshot_steps
            snapshots_v[step] = copy(v_particles)
            save_fs_snapshot(ws, p.suffix, step, f_coeffs)
            plot_fs_diagnostics(ws, f_coeffs, p.suffix, step)
            # `time` here is derived from `step`, and these files run to GB, so
            # a resume does not rewrite them: rows a legacy run already wrote
            # keep their `step * DT`. Take physical time from the conservation
            # CSV, which the resume does migrate.
            for i in axes(v_particles, 1)
                println(snap_io,
                    "$step,$t,$i,$(v_particles[i, 1]),$(v_particles[i, 2])")
            end
            flush(snap_io)
            save_checkpoint(p.suffix, step, t, v_particles, w_particles, f_coeffs,
                entropy_history, energy_history, momentum_history,
                iter_history, res_history, fp_l2_history, neg_history,
                copy(Random.default_rng()))
            # Mirror the growing conservation CSV at snapshot cadence (not every
            # step — that would spawn an rclone process per timestep).
            rclone_upload(p.suffix, cons_csv)
            oracle_io === nothing || rclone_upload(p.suffix, oracle_csv)
        end

        step % 25 == 0 &&
            println("Step $step/$(p.N_STEPS)  iter=$iter  " *
                    (nk === nothing ? "rs=" : "newton=") * "$n_rs" *
                    "  ‖r‖=$(round(res_final; sigdigits=3))" *
                    "  ‖f_s−f_p‖=$(round(fp_l2_history[end]; sigdigits=4))" *
                    "  neg=$(round(neg_history[end]; sigdigits=4))" *
                    "  S=$(round(entropy_history[end]; digits=6))" *
                    "  E=$(round(energy_history[end]; digits=8))")
    end

    # CSVs already streamed per-step / per-snapshot above. Just close.
    close(cons_io)
    close(snap_io)
    # Final mirror so the last steps (if not a multiple of 25) reach S3 too.
    rclone_upload(p.suffix, cons_csv; final = true)
    if oracle_io !== nothing
        close(oracle_io)
        rclone_upload(p.suffix, oracle_csv; final = true)
    end
    if dump_io !== nothing   # GBs: uploaded once, at the end, in the background
        close(dump_io)
        rclone_upload(p.suffix, dump_file; final = true)
    end
    println("Saved $cons_csv")
    println("Saved $snap_csv")

    plot_run_dashboard(ws,
        entropy_history, energy_history, momentum_history,
        iter_history, res_history, fp_l2_history,
        neg_history, p.suffix)

    return (; entropy_history, energy_history, momentum_history,
        iter_history, res_history, fp_l2_history, neg_history,
        snapshots_v,
        label = (p.solver === :newton ? "Newton–Krylov" :
                 p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard"))
end

function main(args = ARGS)
    if isempty(args)
        preset = "parameters_default.jl"
        overrides = String[]
    else
        preset = args[1]
        overrides = collect(String, args[2:end])
    end
    isfile(preset) || error("Preset file not found: $preset")

    # Strip --resume=<step|auto> out of overrides before parse_overrides sees it
    # (it's not a SimParameters field). Accepts an integer step number or the
    # literal `auto` to pick the highest-step checkpoint for this suffix.
    resume = nothing
    overrides = filter(overrides) do tok
        if startswith(tok, "--resume=")
            val = tok[(length("--resume=") + 1):end]
            resume = (val == "auto") ? :auto : parse(Int, val)
            return false
        end
        return true
    end

    println("Loading preset: $preset")
    # The preset file ends by binding PARAMS = SimParameters(...). `include`
    # returns the last expression's value, so we capture PARAMS without
    # relying on module globals.
    params_loaded = include(joinpath(@__DIR__, preset))
    p = parse_overrides(params_loaded::SimParameters, overrides)

    if p.use_gpu
        println("GPU enabled — loading CUDA…")
        # collision_gpu.jl provides _upload_col! (shared staging helper) plus the
        # O(N²) Landau kernels; projection_gpu.jl provides the P_DEG=2 particle↔
        # spline kernels used by BOTH operators.
        include(joinpath(@__DIR__, "collision_gpu.jl"))
        if p.P_DEG == 2
            include(joinpath(@__DIR__, "projection_gpu.jl"))
            L2PROJ_FN[] = getglobal(Main, :l2_project_gpu!)
            COMPG_FN[] = getglobal(Main, :compute_G_gpu!)
            println("GPU projection chain enabled (P_DEG=2 kernels)")
        else
            @warn "use_gpu: projection kernels are P_DEG=2-specialized; " *
                  "projection stays on CPU for P_DEG=$(p.P_DEG)"
        end

        if p.collision_model == :lb
            # LB has no O(N²) sum: GPU accelerates the projection + log-gradient
            # gather (the ~40 ms/iter bulk); the O(N) drift stays on CPU.
            if p.P_DEG == 2
                LOGGRAD_FN[] = getglobal(Main, :eval_loggrad_gpu!)
                println("GPU LB log-gradient enabled")
            end
        else
            # The NN warm start runs its MLP on the device too. `using CUDA` just
            # happened in a newer world, so plain getglobal cannot see CuArray yet.
            NN_TO_DEVICE[] = Base.invokelatest(getglobal, Main, :CuArray)
            if p.gpu_fp32
                println("  ⚠ FP32 collision kernel (conservation experiment)")
                COLLISION_FN[] = getglobal(Main, :compute_collision_gpu32!)
            else
                COLLISION_FN[] = getglobal(Main, :compute_collision_gpu!)
            end
        end
    end

    params_file = "params_$(p.suffix).jl"
    save_params(params_file, p)
    rclone_upload(p.suffix, params_file)
    res = run_simulation(p; resume = resume)
    if isempty(res.iter_history)
        println("\n--- No new steps run (already at N_STEPS) ---")
    else
        a = sum(res.iter_history) / length(res.iter_history)
        println("\n--- Inner-iter summary ---")
        println("$(res.label):  avg=$(round(a; digits=2))  max=$(maximum(res.iter_history))" *
                "  steps=$(length(res.iter_history))")
    end
end

# Run only when executed as a script (`julia main.jl ...`), so the test suite and
# the REPL can `include` this file to reach the functions above without starting
# a simulation.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
