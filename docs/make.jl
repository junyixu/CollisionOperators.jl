using CollisionOperators
using Documenter

# Defines the `Driver` module and the doctest setup. Kept in its own file because
# the Doctests CI job includes it too, without running this script.
include(joinpath(@__DIR__, "doctestsetup.jl"))

makedocs(;
    modules = [CollisionOperators, Driver],
    authors = "Michael Kraus, Junyi Xu <junyixu0@gmail.com>",
    sitename = "CollisionOperators.jl",
    format = Documenter.HTML(;
        canonical = "https://JuliaPlasma.github.io/CollisionOperators.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Operators" => [
            "Lenard–Bernstein (2D)" => "lenard_bernstein.md"
        ],
        "Gonzalez vs. plain midpoint" => "gonzalez_vs_midpoint.md",
        "Cost of the implicit solve" => "implicit_solve_cost.md",
        "Anderson window update" => "anderson_window.md",
        "Newton–Krylov vs. Anderson" => "newton_krylov.md",
        "Defect correction vs. Anderson" => "defect_correction.md",
        "Driver API" => "solver.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaPlasma/CollisionOperators.jl",
    devbranch = "main"
)
