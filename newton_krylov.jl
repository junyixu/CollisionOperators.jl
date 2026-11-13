# newton_krylov.jl — matrix-free GMRES and an inexact Jacobian-free Newton–Krylov
# (JFNK) solver for F(x) = 0. Generic over a residual closure and plain vectors:
# no Mantis, no GPU, so it is unit-testable on its own. solver.jl wires it to
# `picard_map!` through `step_newton!`.
#
# The GMRES core follows Krylov.jl's `gmres.jl` (Arnoldi with modified
# Gram–Schmidt, QR of the Hessenberg matrix by Givens reflections updated one
# column at a time, residual norm read off the rotated right-hand side) but is
# written out here so the driver keeps no Krylov.jl dependency.

using LinearAlgebra: norm, dot, axpy!

"""
    GMRESWorkspace(n, mem)

Storage for [`gmres!`](@ref) on vectors of length `n` with at most `mem`
Arnoldi vectors: the basis `V` (`n × (mem+1)`), the Givens-rotated Hessenberg
matrix `R` (upper triangular after rotation), the rotations `c`/`s`, and the
rotated right-hand side `z`.
"""
struct GMRESWorkspace
    V::Matrix{Float64}
    R::Matrix{Float64}
    c::Vector{Float64}
    s::Vector{Float64}
    z::Vector{Float64}
    w::Vector{Float64}
end

function GMRESWorkspace(n::Integer, mem::Integer)
    return GMRESWorkspace(zeros(n, mem + 1), zeros(mem, mem),
        zeros(mem), zeros(mem), zeros(mem + 1), zeros(n))
end

# Symmetric Givens reflection [c s; s -c] with [c s; s -c] * [a; b] = [ρ; 0].
# Real branch of Krylov.jl's `sym_givens` (after Saunders and Choi).
function _sym_givens(a::Float64, b::Float64)
    if iszero(b)
        c = sign(a) + iszero(a)
        s = 0.0
        ρ = abs(a)
    elseif iszero(a)
        c = 0.0
        s = sign(b)
        ρ = abs(b)
    elseif abs(b) > abs(a)
        t = a / b
        s = sign(b) / sqrt(1.0 + t * t)
        c = s * t
        ρ = b / s
    else
        t = b / a
        c = sign(a) / sqrt(1.0 + t * t)
        s = c * t
        ρ = a / c
    end
    return c, s, ρ
end

@doc raw"""
    gmres!(x, A!, b, gw::GMRESWorkspace; rtol, atol = 0.0, itmax = size(gw.R, 1))

Solve ``A x = b`` from ``x_0 = 0`` by unrestarted GMRES, with ``A`` applied
matrix-free as `A!(y, u)` (``y \leftarrow A u``). Stops once
``\lVert b - A x_k \rVert \le \texttt{atol} + \texttt{rtol}\,\lVert b \rVert``, on
Arnoldi breakdown, or after `itmax` products (capped by the workspace size).

Returns `(k, rnorm)`: the number of products with ``A`` and the final residual
norm, which GMRES knows without forming ``b - A x_k``.

Every product is one call to `A!`, which for [`newton_krylov!`](@ref) is one
residual evaluation, so `k` is the cost that matters.
"""
function gmres!(x::AbstractVector, A!, b::AbstractVector, gw::GMRESWorkspace;
        rtol::Real, atol::Real = 0.0, itmax::Integer = size(gw.R, 1))
    (; V, R, c, s, z, w) = gw
    mem = min(itmax, size(R, 1))
    fill!(x, 0.0)
    β = norm(b)
    β == 0 && return 0, 0.0

    ε = atol + rtol * β
    btol = eps(Float64)^(3 / 4)   # breakdown threshold, as in Krylov.jl
    fill!(z, 0.0)
    z[1] = β
    @views V[:, 1] .= b ./ β

    k = 0
    rnorm = β
    while k < mem
        k += 1
        A!(w, view(V, :, k))
        # Modified Gram–Schmidt against the basis built so far.
        @inbounds for i in 1:k
            vi = view(V, :, i)
            h = dot(vi, w)
            R[i, k] = h
            axpy!(-h, vi, w)
        end
        hnext = norm(w)

        # Apply the previous reflections to the new column, then the new one.
        @inbounds for i in 1:(k - 1)
            tmp = c[i] * R[i, k] + s[i] * R[i + 1, k]
            R[i + 1, k] = s[i] * R[i, k] - c[i] * R[i + 1, k]
            R[i, k] = tmp
        end
        c[k], s[k], R[k, k] = _sym_givens(R[k, k], hnext)
        z[k + 1] = s[k] * z[k]
        z[k] = c[k] * z[k]
        rnorm = abs(z[k + 1])

        (rnorm ≤ ε || hnext ≤ btol) && break
        k < mem && (@views V[:, k + 1] .= w ./ hnext)
    end

    # Back substitution R[1:k,1:k] y = z[1:k], overwriting z, then x = V y.
    @inbounds for i in k:-1:1
        acc = z[i]
        for j in (i + 1):k
            acc -= R[i, j] * z[j]
        end
        z[i] = abs(R[i, i]) ≤ btol ? 0.0 : acc / R[i, i]
    end
    @inbounds for i in 1:k
        axpy!(z[i], view(V, :, i), x)
    end
    return k, rnorm
end

"""
    NKWorkspace(n, krylov_max)

Buffers for [`newton_krylov!`](@ref) on unknowns of length `n`: the residual at
the current and the trial iterate, the Newton step, the finite-difference probe,
the best iterate seen, and a [`GMRESWorkspace`](@ref) of `krylov_max` vectors.
"""
struct NKWorkspace
    F::Vector{Float64}       # F(x) at the current iterate
    Ft::Vector{Float64}      # F at the line-search trial point
    xt::Vector{Float64}      # line-search trial point
    xp::Vector{Float64}      # finite-difference probe x + h u
    Fp::Vector{Float64}      # F(x + h u)
    δ::Vector{Float64}       # Newton step
    x_best::Vector{Float64}
    F_best::Vector{Float64}
    gmres::GMRESWorkspace
end

function NKWorkspace(n::Integer, krylov_max::Integer)
    return NKWorkspace(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
        zeros(n), zeros(n), GMRESWorkspace(n, krylov_max))
end

@doc raw"""
    newton_krylov!(x, F!, nk::NKWorkspace; tol, max_evals, fd_h,
                   eta_max = 0.9, eta_gamma = 0.9, armijo = 1e-4,
                   max_backtrack = 8, verbose = false)

Drive ``\lVert F(x) \rVert`` below `tol` by inexact Newton, starting from `x` and
overwriting it. `F!(y, x)` writes ``F(x)`` into `y`.

# The Newton step

Each step solves ``J(x_k)\,\delta = -F(x_k)`` with [`gmres!`](@ref), never
forming ``J``: a Jacobian–vector product is the forward difference

```math
J u \approx \frac{F(x + h u) - F(x)}{h},
\qquad
h = \texttt{fd\_h}\,/\,\lVert u \rVert ,
```

so each GMRES iteration costs one residual evaluation and the probe
``x + h u`` always moves by exactly `fd_h`. The step is absolute on purpose. It
must be small against the scale on which ``J`` changes, which can be far
smaller than ``\lVert x \rVert``, and large against the noise in ``F``. For the
Landau step a Taylor test puts the first limit near ``10^{-5}`` along its stiff
direction, while a step scaled by ``\sqrt{\varepsilon}\,\lVert x \rVert`` comes out at
``\approx 0.1`` in FP32 (see `docs/src/newton_krylov.md`).

# Forcing term

The linear solve is only as tight as the Newton step needs. GMRES stops at
relative residual ``\eta_k``, chosen by Eisenstat–Walker (choice 2),

```math
\eta_k = \gamma \left(\frac{\lVert F_k \rVert}{\lVert F_{k-1} \rVert}\right)^2 ,
```

safeguarded from dropping abruptly (``\eta_k \ge \gamma\,\eta_{k-1}^2`` while
that exceeds ``0.1``), capped at `eta_max`, and kept above
``\tfrac12\,\texttt{tol}/\lVert F_k \rVert`` so the last step does not over-solve
past the target. The first step uses ``\eta_0 = \texttt{eta\_max}``.

# Globalisation

The step is accepted along ``x_k + \lambda\delta`` with ``\lambda`` halved from 1
until the Armijo condition
``\lVert F(x_k + \lambda\delta) \rVert \le (1 - \alpha\lambda(1 - \eta_k))\lVert F_k \rVert``
holds. If `max_backtrack` halvings do not get there, the residual has hit the
noise floor of ``F`` and the solve stops.

# A cheaper Jacobian product

`Jv!(y, x, u)`, if given, replaces the finite difference of `F!` as the product
``y \leftarrow J(x)\,u``: a product through a cheaper evaluation of the same map
(the FP32 pair sum) or an analytic tangent. The residual, the line search and the
stopping test stay on `F!`, so the root is that of `F!`; with a low-precision
product the solve is a defect correction in precision. `n_jvp` counts the calls.

# Returns

`(n_evals, rnorm, n_newton, rnorm0, status, n_jvp)` where `n_evals` counts every call to
`F!` (the solver's cost), `status` is `:converged`, `:stalled` (line search
failed) or `:budget` (`max_evals` reached). On `:stalled`/`:budget`, `x` and
`nk.F` hold the best iterate seen. In every case `nk.F` holds ``F(x)``.
"""
function newton_krylov!(x::AbstractVector, F!, nk::NKWorkspace;
        tol::Real, max_evals::Integer, fd_h::Real,
        eta_max::Real = 0.9, eta_gamma::Real = 0.9, armijo::Real = 1e-4,
        max_backtrack::Integer = 8, verbose::Bool = false, Jv! = nothing)
    (; F, Ft, xt, xp, Fp, δ, x_best, F_best) = nk
    n_jvp = 0
    krylov_max = size(nk.gmres.R, 1)

    F!(F, x)
    n_evals = 1
    rnorm = norm(F)
    rnorm0 = rnorm
    rnorm_best = rnorm
    x_best .= x
    F_best .= F
    verbose && println("    nk=0  evals=1  ‖F‖=$rnorm")

    # Matrix-free J u by forward difference about the current x (F holds F(x)).
    function jvp!(y, u)
        if Jv! !== nothing
            n_jvp += 1
            return Jv!(y, x, u)
        end
        h = fd_h / norm(u)
        @. xp = x + h * u
        F!(Fp, xp)
        n_evals += 1
        @. y = (Fp - F) / h
        return y
    end

    η = Float64(eta_max)
    rnorm_prev = rnorm
    n_newton = 0
    status = :budget
    while true
        if rnorm < tol
            status = :converged
            break
        end
        # Need at least one product and one trial evaluation.
        budget = max_evals - n_evals
        budget < 2 && break

        if n_newton > 0
            η_ew = eta_gamma * (rnorm / rnorm_prev)^2
            η_safe = eta_gamma * η^2
            η_safe > 0.1 && (η_ew = max(η_ew, η_safe))
            η = clamp(η_ew, 0.5 * tol / rnorm, Float64(eta_max))
        end

        # Solve J δ = F, then step along -δ.
        k_lin, _ = gmres!(δ, jvp!, F, nk.gmres; rtol = η,
            itmax = Jv! === nothing ? min(krylov_max, budget - 1) : krylov_max)
        n_newton += 1

        # Backtracking line search on ‖F‖ (Armijo).
        λ = 1.0
        rnorm_t = Inf
        accepted = false
        for _ in 0:max_backtrack
            @. xt = x - λ * δ
            F!(Ft, xt)
            n_evals += 1
            rnorm_t = norm(Ft)
            if rnorm_t ≤ (1 - armijo * λ * (1 - η)) * rnorm
                accepted = true
                break
            end
            n_evals ≥ max_evals && break
            λ /= 2
        end
        if rnorm_t < rnorm_best
            rnorm_best = rnorm_t
            x_best .= xt
            F_best .= Ft
        end
        verbose && println("    nk=$n_newton  evals=$n_evals  gmres=$k_lin  η=" *
                "$(round(η; sigdigits=3))  λ=$λ  ‖F‖=$rnorm_t" *
                (accepted ? "" : "  [line search failed]"))
        if !accepted
            status = n_evals ≥ max_evals ? :budget : :stalled
            break
        end

        x .= xt
        F .= Ft
        rnorm_prev = rnorm
        rnorm = rnorm_t
    end

    if status !== :converged && rnorm_best < rnorm
        x .= x_best
        F .= F_best
        rnorm = rnorm_best
    end
    return n_evals, rnorm, n_newton, rnorm0, status, n_jvp
end
