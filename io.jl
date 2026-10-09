# io.jl — run-record I/O for the driver in main.jl: S3 mirroring, the resolved
# parameter preset, checkpoints, and the conservation-history CSV.
#
# Deliberately free of `Workspace` (and so of Mantis) and of CairoMakie, so the
# CSV / checkpoint / rclone helpers can be unit-tested on their own:
#
#   include("Parameters.jl"); include("io.jl")
#
# `save_params` is the only function here that needs a type from Parameters.jl.
# Output that needs the FEM workspace (fs snapshots, PNGs) lives in plots.jl.

using Serialization

# ---- rclone live upload -----------------------------------------------------
# Mirror conservation CSV + fs snapshots to S3 as they are written, so a remote
# machine can plot mid-run. Best-effort: failures warn, never abort the sim.
#   RCLONE_UPLOAD=0  → disable entirely (default on)
#   RCLONE_REMOTE=…  → override bucket dir (default: suffix with '_'→'-', i.e.
#                      suffix `bimodal_v1` → mpcdf-s3://…/bimodal-v1)
rclone_enabled() = get(ENV, "RCLONE_UPLOAD", "1") != "0"

function rclone_remote(suffix::String)
    get(ENV, "RCLONE_REMOTE",
        "mpcdf-s3://collision-operators/" * replace(suffix, '_' => '-'))
end

# Uploads run in the background so the time stepping never waits on the network
# (a 1.9 GB dump took 13+ min from a US pod to MPCDF). `main` waits for them
# before it returns (`wait_uploads`), so a pod deleted after the run loses nothing.
#   RCLONE_WAIT=0    → return without waiting, so a launch script can start the
#                      next run while these still upload; it must then wait for
#                      `pgrep -x rclone` to be empty before it finishes
const UPLOADS = Dict{Tuple{String, String}, Base.Process}()   # (remote, file) → latest upload

"""
    rclone_upload(suffix, fname; final = false)

Start uploading `fname` to the run's S3 directory without waiting. While an
earlier upload of the same file to the same place is still running, a mid-run
call is skipped (the next one catches up) and a `final` call waits for it and then
uploads again, so the last version always reaches S3.
"""
function rclone_upload(suffix::String, fname::String; final::Bool = false)
    rclone_enabled() || return nothing
    remote = rclone_remote(suffix)
    key = (remote, abspath(fname))
    prev = get(UPLOADS, key, nothing)
    if prev !== nothing && process_running(prev)
        final || return nothing
        wait(prev)
    end
    prev === nothing || success(prev) ||
        @warn "rclone upload failed" fname remote exitcode = prev.exitcode
    try
        # The conservation CSV grows while it uploads; send what is there.
        UPLOADS[key] = run(`rclone copyto --local-no-check-updated $fname $remote/$fname`;
            wait = false)
    catch e
        @warn "rclone upload failed" fname remote exception = e
    end
    return nothing
end

"""
    wait_uploads()

Block until every background upload has finished, warning about any that failed.
"""
function wait_uploads()
    for ((remote, fname), proc) in UPLOADS
        wait(proc)
        success(proc) || @warn "rclone upload failed" fname remote exitcode = proc.exitcode
    end
    empty!(UPLOADS)
    return nothing
end

# ---- Run record ---------------------------------------------------------------
# Writes the resolved parameters (preset + CLI overrides) as a preset file:
# `julia main.jl <fname>` reruns with identical parameters. A plain function
# rather than a method of FileIO's `save` (re-exported by CairoMakie), so
# recording the configuration does not pull a plotting stack into that path.
function save_params(fname::AbstractString, p::SimParameters)
    open(fname, "w") do io
        print(io, "PARAMS = ")
        show(io, MIME("text/plain"), p)
        println(io)
    end
    println("Saved $fname")
    return fname
end

# ---- Checkpoint / resume ----------------------------------------------------
# Full simulation state serialized via stdlib `Serialization` (no extra deps).
# Written at every snapshot step (≡ every 25 steps + final). On resume, the
# main loop reloads state and appends to the existing CSVs.
#
# State captured:
#   step          : last completed step (resume continues at step+1)
#   v_particles   : N×2 velocity matrix
#   w_particles   : N weights (constant 1/N, saved for sanity)
#   f_coeffs      : spline coefficient vector (size = ws.n_dofs)
#   *_history     : seven diagnostic vectors (length = step+1 or step)
#   rng_state     : copy of Random.default_rng() so any post-resume sampling
#                   (none currently — kept for forward compat) is reproducible
function checkpoint_path(suffix::String, step::Int)
    "checkpoint_$(suffix)_step$(lpad(step, 4, '0')).jls"
end

function save_checkpoint(suffix::String, step::Int, t::Float64,
        v_particles, w_particles,
        f_coeffs, entropy_history, energy_history,
        momentum_history, iter_history, res_history,
        fp_l2_history, neg_history, rng_state)
    fname = checkpoint_path(suffix, step)
    open(fname, "w") do io
        serialize(io,
            (; step, t, v_particles, w_particles, f_coeffs,
                entropy_history, energy_history, momentum_history,
                iter_history, res_history, fp_l2_history, neg_history,
                rng_state))
    end
    println("Saved $fname")
    rclone_upload(suffix, fname)
    return fname
end

# Drop rows past `step`, left by a run that got further than the checkpoint
# being resumed from, so the appended rows don't repeat steps. Column 1 is the
# step; the header line is kept.
function truncate_csv_after(fname::String, step::Int)
    isfile(fname) || return nothing
    lines = readlines(fname)
    keep = filter(l -> something(tryparse(Int, first(split(l, ','))), -1) <= step, lines)
    length(keep) == length(lines) || write(fname, join(keep, '\n') * '\n')
    return nothing
end

# Conservation-CSV columns. `time` is the accumulated physical time and `dt` the
# step size that produced the row, so `cumsum(dt) == time` holds even when a
# resume changes DT. Older files stored `time = step * DT`, which silently
# rescaled the whole history whenever a resume used a different DT.
# Columns of solver_stats_<suffix>.csv (see main.jl).
const STATS_COLS = ["step", "iter", "inner", "t_solve", "t_step"]

const CONS_COLS = ["step", "time", "entropy", "energy", "momentum_1",
    "momentum_2", "iter", "residual", "fp_minus_fs", "neg_part", "r0", "dt"]

# Bring a conservation CSV written by an older version up to `CONS_COLS`: fill
# any column it lacks with 0.0, recover each step's DT from the legacy
# `time = step * DT` and replace `time` with the accumulated physical time.
# A no-op once `dt` is there, so resuming the same run twice is safe.
function migrate_cons_csv!(fname::String)
    isfile(fname) || return nothing
    lines = readlines(fname)
    isempty(lines) && return nothing
    old = String.(split(first(lines), ','))
    "dt" in old && return nothing
    rows = [Dict(zip(old, String.(split(l, ','))))
            for l in Iterators.drop(lines, 1)]
    # time / step is that row's DT. The last row of a repeated step wins, since
    # an older resume appended instead of truncating; a step absent from the
    # file contributes nothing to the sum.
    dt_at = Dict{Int, Float64}()
    for r in rows
        st = tryparse(Int, get(r, "step", ""))
        t = tryparse(Float64, get(r, "time", ""))
        (st === nothing || t === nothing || st < 0) && continue
        dt_at[st] = st == 0 ? 0.0 : t / st
    end
    isempty(dt_at) && return nothing
    t_at, t_acc = Dict{Int, Float64}(), 0.0
    for st in 0:maximum(keys(dt_at))
        t_acc += get(dt_at, st, 0.0)
        t_at[st] = t_acc
    end
    tmp = fname * ".migrating"
    open(tmp, "w") do io
        println(io, join(CONS_COLS, ','))
        for r in rows
            st = tryparse(Int, get(r, "step", ""))
            (st === nothing || !haskey(t_at, st)) && continue
            r["time"] = string(t_at[st])
            r["dt"] = string(dt_at[st])
            println(io, join((get(r, c, "0.0") for c in CONS_COLS), ','))
        end
    end
    mv(tmp, fname; force = true)
    println("Migrated $fname to accumulated `time` + `dt` " *
            "($(length(old)) → $(length(CONS_COLS)) columns)")
    return nothing
end

# Accumulated physical time recorded for `step` in a migrated conservation CSV
# (last row wins if the step repeats); `nothing` if the step is not in the file.
function cons_time_at(fname::String, step::Int)
    isfile(fname) || return nothing
    t = nothing
    for l in Iterators.drop(readlines(fname), 1)
        f = split(l, ',')
        length(f) >= 2 && tryparse(Int, f[1]) == step &&
            (t = tryparse(Float64, f[2]))
    end
    return t
end

# `step=:auto` (or any non-positive Int) → pick the highest-step checkpoint
# matching the suffix from the cwd.
function load_checkpoint(suffix::String, step)
    fname = if step === :auto || (step isa Integer && step <= 0)
        files = filter(readdir()) do f
            startswith(f, "checkpoint_$(suffix)_step") && endswith(f, ".jls")
        end
        isempty(files) && error("No checkpoint files for suffix=$suffix")
        sort(files)[end]
    else
        checkpoint_path(suffix, Int(step))
    end
    isfile(fname) || error("Checkpoint not found: $fname")
    println("Loading checkpoint: $fname")
    return open(deserialize, fname)
end
