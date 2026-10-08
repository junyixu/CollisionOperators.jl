using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    # io.jl needs only Parameters.jl, so its helpers run on every core pass.
    @safetestset "io.jl helpers" include("integration/io_helpers.jl")
    # newton_krylov.jl is generic over plain vectors, so it runs here too.
    @safetestset "GMRES / Newton–Krylov" include("integration/newton_krylov.jl")
    # warmstart.jl is stdlib-only as well.
    @safetestset "NN warm start" include("integration/warmstart.jl")
end

# main.jl is a script that pulls CairoMakie and Mantis, so the kernels reachable
# only through it are exercised in the slow group rather than on every core run.
if "slow" in GROUPS
    @safetestset "main.jl helpers" include("integration/main_helpers.jl")
end
