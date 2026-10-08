# Parameters.jl — simulation configuration carried via a single immutable struct.
#
# Workflow:
#   1. Concrete preset files (parameters_default.jl, parameters_finemesh.jl, …)
#      construct a `SimParameters` value bound to the global symbol `PARAMS`.
#      Each preset directly provides the breakpoint vectors `bp1`, `bp2`
#      (anisotropic, possibly non-uniform).
#   2. The CLI entry point in main.jl picks the preset file from
#      ARGS[1] (defaulting to "parameters_default.jl") and `include`s it.
#   3. Remaining ARGS of the form --key=val override individual *scalar* fields
#      via `parse_overrides` (vector fields like `bp1`/`bp2` are not CLI-
#      overridable — re-edit the preset file instead).
#
# Every downstream module (MantisWrappers, functions, main loop) accepts an
# instance of `SimParameters` rather than reading globals; this is the only
# state shared across the call chain besides the preallocated `Workspace`.

Base.@kwdef struct SimParameters
    # Velocity-space breakpoints (anisotropic, possibly non-uniform).
    # `bp1[1]`/`bp1[end]` define the v₁ domain; cell count = `length(bp1) - 1`.
    bp1::Vector{Float64} = [-6.0; -5.0; LinRange(-4.0, 4.0, 17); 5.0; 6.0]
    bp2::Vector{Float64} = [-6.0; LinRange(-2.5, 2.5, 13); 6.0]

    # B-spline space
    P_DEG::Int = 2
    K_REG::Int = 1
    N_QUAD::Int = 6              # 1D Gauss–Legendre points per element

    # Particle initial condition (anisotropic Gaussian, untruncated)
    N_PARTICLES::Int = 10_000
    σ1::Float64 = 4/3
    σ2::Float64 = 0.5

    # Time integration
    DT::Float64 = 0.001
    N_STEPS::Int = 400

    # Snapshot cadence: every `snap_every` steps write fs_snapshot CSV, render the
    # fs_diag PNG, dump particle rows, checkpoint, and mirror the CSV to S3. Larger
    # values cut plotting/I-O overhead on long runs (the conservation CSV is always
    # per-step regardless).
    snap_every::Int = 25

    # Collision operator: :landau = O(N²) perpendicular-projection Landau sum
    # (GPU-capable via use_gpu/gpu_fp32/gpu_fp16); :lb = O(N) conservative
    # Lenard-Bernstein drift v̇ = -ν(∇f/f + A + B v). Both share the identical
    # mesh / projection / Anderson scaffolding and the use_gonzalez toggle, so an
    # LB run on the same preset isolates operator-specific behavior. GPU applies
    # to :landau only — :lb + use_gpu runs on CPU (warned at startup).
    collision_model::Symbol = :landau

    # LB collision frequency ν (only used when collision_model == :lb).
    nu::Float64 = 1.0

    # Discrete-gradient variant: true = Gonzalez (entropy-exact, has |Δv|²
    # denominator that blows up near equilibrium); false = plain implicit
    # midpoint (drop the Gonzalez correction term).
    use_gonzalez::Bool = true

    # Entropy/seed integrand variant: false = positivity-clamped log f (drop
    # f_s<0 quadrature points); true = log f = ½ log f² identity, so Gibbs
    # undershoot points contribute via |f| guard. Probe for spike sensitivity.
    use_logsq::Bool = false

    # Implicit-solver knobs
    use_anderson::Bool = true
    damping::Float64 = 0.7
    m_anderson::Int = 8
    tol::Float64 = 1e-12   # relative tol on ‖r‖ / ‖v‖
    max_iter::Int = 2000
    # Anderson convergence safety net (see `step_anderson!` doc-comment):
    abs_floor::Float64 = 1e-7   # cap on effective tol — past this, asking
    # for less is pointless (Picard noise floor)
    stag_window::Int = 50     # iters between stagnation checks
    stag_rel_tol::Float64 = 0.01   # < 1% drop in `nrm_best` over window ⇒ exit
    damp_decay_start::Int = 200    # iter index after which damping is decayed
    damp_decay_factor::Float64 = 0.5  # damping multiplier once decay starts

    # Implicit solver: :anderson = Anderson-accelerated Picard (`step_anderson!`);
    # :newton = Jacobian-free Newton–Krylov (`step_newton!`). Both share `tol`,
    # `abs_floor` and `max_iter` (a cap on Picard-map evaluations), and both log
    # evaluations in the `iter` column, so runs compare one-to-one.
    solver::Symbol = :anderson
    nk_krylov_max::Int = 30        # GMRES basis size per Newton step
    nk_eta_max::Float64 = 0.9      # Eisenstat–Walker forcing-term cap
    # Absolute finite-difference step for J·u (the probe moves by exactly this).
    # 0 = auto: 1e-5 with the FP32 collision kernel, 1e-6 otherwise. A Taylor
    # test puts the Landau map's linear range near 1e-5 along its stiff
    # direction (docs/src/newton_krylov.md).
    nk_fd_h::Float64 = 0.0
    # Legacy relative step h = nk_fd_rel·‖v‖; > 0 overrides nk_fd_h. Only for
    # reproducing the first runs (√eps·‖v‖ ≈ 0.1 in FP32, far too large).
    nk_fd_rel::Float64 = 0.0
    # true = both solvers return G(v), one final Picard update of the accepted
    # iterate v: momentum is then conserved exactly and energy to O(Δt‖F‖).
    # false = return v itself (smaller residual, but conservation errors first
    # order in ‖F‖; 3–10x worse energy drift in the FP32 A/B). Keep true.
    exit_picard_step::Bool = true

    # Initial guess of the implicit solve (warmstart.jl): :euler = explicit Euler
    # predictor; :nn = Euler + Δt²·δ̂ from the MLP in `nn_weights`, each particle's
    # correction clipped to `nn_cap` × the mean Euler step ‖Δt·v̇‖.
    warmstart::Symbol = :euler
    nn_weights::String = ""
    nn_cap::Float64 = 3.0

    # Append each step's pre-solve state and NN label to
    # training_dump_<suffix>.bin (1.3 MB/step at N=40k) for train_warmstart.jl.
    dump_training::Bool = false

    # Oracle sweep: every `oracle_every` steps (0 = off), re-solve the step from
    # starts that leave a fraction ε ∈ `oracle_eps` of the Euler gap, logging the
    # iterations to oracle_<suffix>.csv. Costs one extra solve per ε and mode.
    oracle_every::Int = 0
    oracle_eps::String = "1,0.3,0.1,0.03,0.01,0.001,0"

    # `v1_peak == 0` → single anisotropic Gaussian N(0, diag(σ1², σ2²)).
    # `v1_peak  > 0` → bimodal in v₁: balanced 50/50 mixture of
    #                  N(±v1_peak, σ1²) in v₁, with v₂ ~ N(0, σ2²).
    # (Ported from the bimodal-v1 branch.)
    v1_peak::Float64 = 0.0

    # Run the O(N²) Landau collision sum on the GPU (CUDA.jl, Float64).
    # Loads collision_gpu.jl on demand; CPU runs never touch CUDA.
    use_gpu::Bool = false

    # EXPERIMENT: Float32 pair math in the GPU collision kernel (accumulation
    # per slice FP32, reduction FP64). Degrades U(d)·d=0 to FP32 roundoff ⇒
    # measurable energy drift; exists to quantify the speed/conservation
    # trade. Requires use_gpu=true. Projection stays FP64.
    gpu_fp32::Bool = false

    # Run identifier (used in output file names)
    suffix::String = "anderson"

    # Random seed (so independent runs use the same particle IC)
    seed::Int = 42
end

# Parse a single CLI token of the form "--key=val". The space-separated form
# would require sequential scanning; we only support the equals form to keep
# the parser stateless and easy to read.
function _parse_override_token(tok::AbstractString)
    startswith(tok, "--") || return nothing
    eq = findfirst('=', tok)
    eq === nothing && error("CLI override $tok must use --key=value form")
    key = Symbol(tok[3:(eq - 1)])
    val_str = tok[(eq + 1):end]
    return key, val_str
end

# Coerce a string into the field's declared type. Bool accepts true/false/1/0;
# strings pass through; numeric types use parse(T, s). Vector fields are
# rejected at the call site — too clumsy to express on the CLI.
function _coerce(::Type{T}, s::AbstractString) where {T}
    if T === Bool
        s in ("true", "1") && return true
        s in ("false", "0") && return false
        error("Cannot parse $s as Bool")
    elseif T === Symbol
        return Symbol(s)
    elseif T <: AbstractString
        return String(s)
    elseif T <: AbstractVector
        error("Vector parameters (`bp1`, `bp2`) are not CLI-overridable; edit the preset file")
    else
        return parse(T, s)
    end
end

"""
    parse_overrides(p::SimParameters, args)

Apply every `--key=value` token in `args` on top of `p`, returning a new
`SimParameters`. Unknown keys raise. Empty `args` returns `p` unchanged.
"""
function parse_overrides(p::SimParameters, args)
    isempty(args) && return p
    fields = fieldnames(SimParameters)
    field_type = Dict(f => fieldtype(SimParameters, f) for f in fields)
    overrides = Dict{Symbol, Any}()
    for tok in args
        parsed = _parse_override_token(tok)
        parsed === nothing && continue
        key, val_str = parsed
        haskey(field_type, key) || error("Unknown parameter: $key")
        overrides[key] = _coerce(field_type[key], val_str)
    end
    return SimParameters(; (f => get(overrides, f, getfield(p, f)) for f in fields)...)
end

"""
    show(io, ::MIME"text/plain", p::SimParameters)

Print `p` in keyword-constructor form, one field per line. The output is valid
Julia, so `PARAMS = <output>` is itself a preset that reproduces the run.
"""
function Base.show(io::IO, ::MIME"text/plain", p::SimParameters)
    println(io, "SimParameters(")
    for f in fieldnames(SimParameters)
        println(io, "    ", f, " = ", repr(getfield(p, f)), ",")
    end
    print(io, ")")
end

"""
    print_summary(p::SimParameters)

One-shot human-readable dump of the active configuration. Used at the top of
each run so the .log captures exactly what was run.
"""
function print_summary(p::SimParameters)
    n1, n2 = length(p.bp1) - 1, length(p.bp2) - 1
    dv1_min, dv1_max = extrema(diff(p.bp1))
    dv2_min, dv2_max = extrema(diff(p.bp2))
    println("==== SimParameters ====")
    println("v₁ ∈ [$(p.bp1[1]), $(p.bp1[end])]  cells=$n1  Δv₁ ∈ [$dv1_min, $dv1_max]")
    println("v₂ ∈ [$(p.bp2[1]), $(p.bp2[end])]  cells=$n2  Δv₂ ∈ [$dv2_min, $dv2_max]")
    println("P_DEG=$(p.P_DEG)  K_REG=$(p.K_REG)  N_QUAD=$(p.N_QUAD)")
    println("N_PARTICLES=$(p.N_PARTICLES)  σ=($(p.σ1), $(p.σ2))  seed=$(p.seed)")
    println("DT=$(p.DT)  N_STEPS=$(p.N_STEPS)")
    if p.solver === :newton
        println("solver=Newton–Krylov(krylov_max=$(p.nk_krylov_max), " *
                "η_max=$(p.nk_eta_max), " *
                (p.nk_fd_rel > 0 ? "fd_rel=$(p.nk_fd_rel)" :
                 "fd_h=$(p.nk_fd_h == 0 ? "auto" : p.nk_fd_h)") * ")" *
                "  tol=$(p.tol)  abs_floor=$(p.abs_floor)  max_iter=$(p.max_iter)")
    else
        println("solver=$(p.use_anderson ? "Anderson(m=$(p.m_anderson))" : "Picard")" *
                "  damping=$(p.damping)  tol=$(p.tol)  abs_floor=$(p.abs_floor)" *
                "  max_iter=$(p.max_iter)")
    end
    println("collision=$(p.collision_model)$(p.collision_model == :lb ? "  ν=$(p.nu)" : "")" *
            "  disc_grad=$(p.use_gonzalez ? "Gonzalez" : "plain-midpoint")" *
            "  entropy_integrand=$(p.use_logsq ? "½log f²" : "clamped log f")")
    println("suffix=$(p.suffix)")
    println("=======================")
end
