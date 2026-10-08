# sq_d04 — 2D Landau on the square mesh (bp1 = bp2, inner Δ=0.4), anisotropic IC
# σ1=4/3, σ2=0.5 relaxing to an isotropic Maxwellian. Same settings as
# landau_collision_private's parameters_sq_d04.jl. FP32 runs must override
# abs_floor (--abs_floor=1e-8).
PARAMS = SimParameters(
    bp1 = [-6.0; -5.0; LinRange(-4.0, 4.0, 21); 5.0; 6.0],    # Δinner=0.4
    bp2 = [-6.0; -5.0; LinRange(-4.0, 4.0, 21); 5.0; 6.0],    # = bp1
    P_DEG = 2, K_REG = 1, N_QUAD = 6,
    N_PARTICLES = 40_000,
    σ1 = 4/3, σ2 = 0.5,
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
    suffix = "sq_d04",
    seed = 42
)
