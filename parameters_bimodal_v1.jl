# bimodal_v1 — 2D Landau, the run behind the "~15 iterations per step" figure
# (S3 `bimodal-v1/`, 15.8 mean iter over steps 1–1500 in FP64). Rebuilt from that
# run's fs_snapshot header: with seed 42 this reproduces its step-0 S, E and P to
# the last digit, which also pins use_logsq = true. IC: bimodal in v₁ (balanced
# 50/50 mixture of N(±2, 1)), v₂ ~ N(0, 1). FP32 runs must override
# abs_floor (--abs_floor=1e-8).
PARAMS = SimParameters(
    bp1 = [-6.0; LinRange(-5.0, 5.0, 26); 6.0],               # Δ=0.4 over [-5,5]
    bp2 = [-6.0; -4.5; LinRange(-3.0, 3.0, 16); 4.5; 6.0],    # Δ=0.4 over [-3,3]
    P_DEG = 2, K_REG = 1, N_QUAD = 6,
    N_PARTICLES = 40_000,
    σ1 = 1.0, σ2 = 1.0,
    v1_peak = 2.0,
    collision_model = :landau,
    DT = 0.001, N_STEPS = 1500,
    use_anderson = true,
    use_gonzalez = true,
    use_logsq = true,
    damping = 0.7, m_anderson = 8,
    tol = 1e-12, max_iter = 2000,
    abs_floor = 1e-10,
    stag_window = 30, stag_rel_tol = 0.1,
    damp_decay_start = 200, damp_decay_factor = 0.5,
    suffix = "bimodal_v1",
    seed = 42
)
