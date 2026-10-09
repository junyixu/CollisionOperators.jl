# GPU (CUDA.jl) implementation of the O(N²) Landau collision sum — same
# semantics as `compute_collision!` in functions.jl: perpendicular-projection
# kernel U(d) = (I − d̂d̂ᵀ)/|d| applied to (G_α − G_γ), w-weighted, identical
# domain-boundary masks and coincident-pair guard (γ==α falls under the
# dist² < 1e-24 guard, matching the CPU skip). Float64 accumulation — the
# scheme's conservation rests on the antisymmetric pair structure, so no FP32
# shortcuts in the accumulator.
#
# Two performance-critical choices (measured on the RTX 4090, N=40k):
#
# 1. _rsqrt64: f32 SFU seed + two FP64 Newton steps (~1 ulp) replaces the
#    FP64 sqrt and both divisions in the pair loop — those are long software
#    sequences at 1/64 rate on GeForce (168 → matters combined with 2.).
#    Deterministic ⇒ the (γ,α)/(α,γ) kernel values stay bitwise symmetric ⇒
#    momentum conservation untouched.
#
# 2. Sliced inner loop: one thread per γ gives only N threads ≈ one wave on
#    128 SMs — unbalanced (156 blocks at tpb=256 ⇒ a 2× tail) and too little
#    latency hiding for the FP64 dependency chain. Thread (γ,s) sums the s-th
#    of _SLICES contiguous α-chunks into partials; a trivial second kernel
#    reduces over s. Summation stays deterministic (fixed chunk grouping), so
#    results are reproducible run to run; agreement with the CPU loop is at
#    roundoff (~1e-14 rel), not bitwise.
#
# Measured N=40k: 198 ms (naive) → 104 ms, flat across S∈[4,16], tpb∈[64,256]
# ⇒ pinned at the FP64-pipe ceiling (~15.4 G pairs/s ≈ 55% of instruction
# peak with the compare/convert mix). Further wins need either dropping the
# per-α boundary compares (particle compaction, ~15%) or leaving exact FP64
# pair math — rejected: conservation rests on it.

using CUDA

const _SLICES = 16          # α-chunks per γ (bench: see bench_collision_gpu.jl)
const _TPB = 128         # threads per block for the pair kernel

# 1/sqrt(x) to ~1 ulp without FP64 division or sqrt: f32 special-function-unit
# seed (~22 bits) + two FP64 Newton steps (→44 → >53 bits), all FMA-rate ops.
@inline function _rsqrt64(x::Float64)
    y = Float64(CUDA.rsqrt(Float32(x)))
    h = 0.5 * x
    y = y * (1.5 - h * y * y)
    y = y * (1.5 - h * y * y)
    return y
end

# Partial sums: thread id ↦ (γ = row-major fastest, slice s). Warp-adjacent
# threads share s and read the same α entry ⇒ hardware broadcast, and their
# partial writes at (s-1)N+γ coalesce.
function _landau_partial!(p1, p2, v1, v2, g1, g2, w, N, S,
        lo1, hi1, lo2, hi2)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > N * S && return nothing
    γ = (tid - 1) % N + 1
    s = (tid - 1) ÷ N + 1
    @inbounds begin
        vγ1 = v1[γ]
        vγ2 = v2[γ]
        if vγ1 <= lo1 || vγ1 >= hi1 || vγ2 <= lo2 || vγ2 >= hi2
            p1[tid] = 0.0
            p2[tid] = 0.0
            return nothing
        end
        chunk = cld(N, S)
        αlo = (s - 1) * chunk + 1
        αhi = min(s * chunk, N)
        Gγ1 = g1[γ]
        Gγ2 = g2[γ]
        acc1 = 0.0
        acc2 = 0.0
        for α in αlo:αhi
            vα1 = v1[α]
            vα2 = v2[α]
            if vα1 <= lo1 || vα1 >= hi1 || vα2 <= lo2 || vα2 >= hi2
                continue
            end
            d1 = vγ1 - vα1
            d2 = vγ2 - vα2
            dist2 = d1 * d1 + d2 * d2
            dist2 < 1e-24 && continue
            gg1 = g1[α] - Gγ1
            gg2 = g2[α] - Gγ2
            inv_dist = _rsqrt64(dist2)
            dv_dot_g = (d1 * gg1 + d2 * gg2) * (inv_dist * inv_dist)
            acc1 += w[α] * (gg1 - d1 * dv_dot_g) * inv_dist
            acc2 += w[α] * (gg2 - d2 * dv_dot_g) * inv_dist
        end
        p1[tid] = acc1
        p2[tid] = acc2
    end
    return nothing
end

function _reduce_partials!(o1, o2, p1, p2, N, S)
    γ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    γ > N && return nothing
    @inbounds begin
        a1 = 0.0
        a2 = 0.0
        for s in 1:S
            a1 += p1[(s - 1) * N + γ]
            a2 += p2[(s - 1) * N + γ]
        end
        o1[γ] = a1
        o2[γ] = a2
    end
    return nothing
end

# Persistent device + host staging buffers, reallocated only if N changes.
mutable struct GpuCollisionBuf
    N::Int
    v1::CuVector{Float64}
    v2::CuVector{Float64}
    g1::CuVector{Float64}
    g2::CuVector{Float64}
    w::CuVector{Float64}
    o1::CuVector{Float64}
    o2::CuVector{Float64}
    p1::CuVector{Float64}
    p2::CuVector{Float64}   # N×_SLICES partials
    h::Vector{Float64}                              # host staging, length N
end

const _GPU_BUF = Ref{Union{Nothing, GpuCollisionBuf}}(nothing)

function _gpu_buf(N::Int)
    b = _GPU_BUF[]
    if b === nothing || b.N != N
        b = GpuCollisionBuf(N,
            CUDA.zeros(Float64, N), CUDA.zeros(Float64, N),
            CUDA.zeros(Float64, N), CUDA.zeros(Float64, N),
            CUDA.zeros(Float64, N),
            CUDA.zeros(Float64, N), CUDA.zeros(Float64, N),
            CUDA.zeros(Float64, N * _SLICES), CUDA.zeros(Float64, N * _SLICES),
            zeros(N))
        _GPU_BUF[] = b
    end
    return b
end

@inline function _upload_col!(dst::CuVector{T}, src::AbstractMatrix, col::Int, h::Vector{T}) where {T}
    @inbounds for i in eachindex(h)
        h[i] = T(src[i, col])
    end
    copyto!(dst, h)
    return nothing
end

# ---- Float32 experiment (--gpu_fp32=true) -------------------------------------
# Pair math entirely in Float32 (native SFU rsqrt, no Newton); per-slice
# accumulation in Float32 (≤ ~2.5k terms/slice ⇒ √n·ε₃₂ ≈ 3e-6 rel), slice
# reduction in Float64. Pair antisymmetry is still bitwise (deterministic
# FP32), but U(d)·d = 0 only holds to FP32 roundoff ⇒ energy conservation is
# expected to degrade to ~1e-7-level drift. This variant exists to MEASURE
# that trade, not because it is recommended. Projection stays FP64.

function _landau_partial32!(p1, p2, v1, v2, g1, g2, w, N, S,
        lo1, hi1, lo2, hi2)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > N * S && return nothing
    γ = (tid - 1) % N + 1
    s = (tid - 1) ÷ N + 1
    @inbounds begin
        vγ1 = v1[γ]
        vγ2 = v2[γ]
        if vγ1 <= lo1 || vγ1 >= hi1 || vγ2 <= lo2 || vγ2 >= hi2
            p1[tid] = 0.0f0
            p2[tid] = 0.0f0
            return nothing
        end
        chunk = cld(N, S)
        αlo = (s - 1) * chunk + 1
        αhi = min(s * chunk, N)
        Gγ1 = g1[γ]
        Gγ2 = g2[γ]
        acc1 = 0.0f0
        acc2 = 0.0f0
        for α in αlo:αhi
            vα1 = v1[α]
            vα2 = v2[α]
            if vα1 <= lo1 || vα1 >= hi1 || vα2 <= lo2 || vα2 >= hi2
                continue
            end
            d1 = vγ1 - vα1
            d2 = vγ2 - vα2
            dist2 = d1 * d1 + d2 * d2
            dist2 < 1.0f-12 && continue      # FP32 analogue of the 1e-24 guard
            gg1 = g1[α] - Gγ1
            gg2 = g2[α] - Gγ2
            inv_dist = CUDA.rsqrt(dist2)
            dv_dot_g = (d1 * gg1 + d2 * gg2) * (inv_dist * inv_dist)
            acc1 += w[α] * (gg1 - d1 * dv_dot_g) * inv_dist
            acc2 += w[α] * (gg2 - d2 * dv_dot_g) * inv_dist
        end
        p1[tid] = acc1
        p2[tid] = acc2
    end
    return nothing
end

mutable struct GpuCollisionBuf32
    N::Int
    v1::CuVector{Float32}
    v2::CuVector{Float32}
    g1::CuVector{Float32}
    g2::CuVector{Float32}
    w::CuVector{Float32}
    o1::CuVector{Float64}
    o2::CuVector{Float64}
    p1::CuVector{Float32}
    p2::CuVector{Float32}
    hf::Vector{Float32}
    h::Vector{Float64}
end

const _GPU_BUF32 = Ref{Union{Nothing, GpuCollisionBuf32}}(nothing)

function _gpu_buf32(N::Int)
    b = _GPU_BUF32[]
    if b === nothing || b.N != N
        b = GpuCollisionBuf32(N,
            CUDA.zeros(Float32, N), CUDA.zeros(Float32, N),
            CUDA.zeros(Float32, N), CUDA.zeros(Float32, N),
            CUDA.zeros(Float32, N),
            CUDA.zeros(Float64, N), CUDA.zeros(Float64, N),
            CUDA.zeros(Float32, N * _SLICES), CUDA.zeros(Float32, N * _SLICES),
            zeros(Float32, N), zeros(N))
        _GPU_BUF32[] = b
    end
    return b
end

function compute_collision_gpu32!(ws::Workspace, dot_v, v_parts, w_parts, G)
    N = size(v_parts, 1)
    b = _gpu_buf32(N)
    _upload_col!(b.v1, v_parts, 1, b.hf)
    _upload_col!(b.v2, v_parts, 2, b.hf)
    _upload_col!(b.g1, G, 1, b.hf)
    _upload_col!(b.g2, G, 2, b.hf)
    copyto!(b.w, Float32.(w_parts))
    @cuda threads=_TPB blocks=cld(N*_SLICES, _TPB) _landau_partial32!(
        b.p1, b.p2, b.v1, b.v2, b.g1, b.g2, b.w, N, _SLICES,
        Float32(ws.bp1[1]), Float32(ws.bp1[end]),
        Float32(ws.bp2[1]), Float32(ws.bp2[end]))
    @cuda threads=256 blocks=cld(N, 256) _reduce_partials!(
        b.o1, b.o2, b.p1, b.p2, N, _SLICES)
    copyto!(b.h, b.o1)
    @inbounds for i in 1:N
        dot_v[i, 1] = b.h[i]
    end
    copyto!(b.h, b.o2)
    @inbounds for i in 1:N
        dot_v[i, 2] = b.h[i]
    end
    return nothing
end

function compute_collision_gpu!(ws::Workspace, dot_v, v_parts, w_parts, G)
    N = size(v_parts, 1)
    b = _gpu_buf(N)
    _upload_col!(b.v1, v_parts, 1, b.h)
    _upload_col!(b.v2, v_parts, 2, b.h)
    _upload_col!(b.g1, G, 1, b.h)
    _upload_col!(b.g2, G, 2, b.h)
    copyto!(b.w, w_parts)
    @cuda threads=_TPB blocks=cld(N*_SLICES, _TPB) _landau_partial!(
        b.p1, b.p2, b.v1, b.v2, b.g1, b.g2, b.w, N, _SLICES,
        ws.bp1[1], ws.bp1[end], ws.bp2[1], ws.bp2[end])
    @cuda threads=256 blocks=cld(N, 256) _reduce_partials!(
        b.o1, b.o2, b.p1, b.p2, N, _SLICES)
    copyto!(b.h, b.o1)
    @inbounds for i in 1:N
        dot_v[i, 1] = b.h[i]
    end
    copyto!(b.h, b.o2)
    @inbounds for i in 1:N
        dot_v[i, 2] = b.h[i]
    end
    return nothing
end

# ---- F and the metric A in one pass (solver = :defect) ------------------------
# `_landau_partial!` plus A_γ = Σ_α w_α U(v_γ − v_α) (3 entries). F is accumulated
# with the same expressions, but the compiler may fuse multiply-adds differently in
# the longer loop, so F agrees with compute_collision_gpu! to about one ulp, not bit
# for bit (max |ΔF| 2–4e-16 on an H100 and an RTX 4090). Measured cost at N = 40k:
# 1.25x (H100) and 1.34x (RTX 4090) that of the plain kernel.

function _landau_metric_partial!(p1, p2, q11, q12, q22, v1, v2, g1, g2, w, N, S,
        lo1, hi1, lo2, hi2)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > N * S && return nothing
    γ = (tid - 1) % N + 1
    s = (tid - 1) ÷ N + 1
    @inbounds begin
        vγ1 = v1[γ]
        vγ2 = v2[γ]
        if vγ1 <= lo1 || vγ1 >= hi1 || vγ2 <= lo2 || vγ2 >= hi2
            p1[tid] = 0.0; p2[tid] = 0.0
            q11[tid] = 0.0; q12[tid] = 0.0; q22[tid] = 0.0
            return nothing
        end
        chunk = cld(N, S)
        αlo = (s - 1) * chunk + 1
        αhi = min(s * chunk, N)
        Gγ1 = g1[γ]
        Gγ2 = g2[γ]
        acc1 = 0.0
        acc2 = 0.0
        a11 = 0.0
        a12 = 0.0
        a22 = 0.0
        for α in αlo:αhi
            vα1 = v1[α]
            vα2 = v2[α]
            if vα1 <= lo1 || vα1 >= hi1 || vα2 <= lo2 || vα2 >= hi2
                continue
            end
            d1 = vγ1 - vα1
            d2 = vγ2 - vα2
            dist2 = d1 * d1 + d2 * d2
            dist2 < 1e-24 && continue
            gg1 = g1[α] - Gγ1
            gg2 = g2[α] - Gγ2
            inv_dist = _rsqrt64(dist2)
            inv_d2 = inv_dist * inv_dist
            dv_dot_g = (d1 * gg1 + d2 * gg2) * inv_d2
            acc1 += w[α] * (gg1 - d1 * dv_dot_g) * inv_dist
            acc2 += w[α] * (gg2 - d2 * dv_dot_g) * inv_dist
            sw = w[α] * inv_dist
            a11 += sw * (1.0 - d1 * d1 * inv_d2)
            a12 -= sw * d1 * d2 * inv_d2
            a22 += sw * (1.0 - d2 * d2 * inv_d2)
        end
        p1[tid] = acc1; p2[tid] = acc2
        q11[tid] = a11; q12[tid] = a12; q22[tid] = a22
    end
    return nothing
end

function _reduce_partial1!(o, p, N, S)
    γ = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    γ > N && return nothing
    @inbounds begin
        a = 0.0
        for s in 1:S
            a += p[(s - 1) * N + γ]
        end
        o[γ] = a
    end
    return nothing
end

mutable struct GpuMetricBuf
    N::Int
    q11::CuVector{Float64}
    q12::CuVector{Float64}
    q22::CuVector{Float64}   # N×_SLICES partials of A
    a11::CuVector{Float64}
    a12::CuVector{Float64}
    a22::CuVector{Float64}
end

const _GPU_MBUF = Ref{Union{Nothing, GpuMetricBuf}}(nothing)

function _gpu_mbuf(N::Int)
    b = _GPU_MBUF[]
    if b === nothing || b.N != N
        b = GpuMetricBuf(N, CUDA.zeros(Float64, N * _SLICES), CUDA.zeros(Float64, N * _SLICES),
            CUDA.zeros(Float64, N * _SLICES),
            CUDA.zeros(Float64, N), CUDA.zeros(Float64, N), CUDA.zeros(Float64, N))
        _GPU_MBUF[] = b
    end
    return b
end

function compute_collision_metric_gpu!(ws::Workspace, dot_v, A, v_parts, w_parts, G)
    N = size(v_parts, 1)
    b = _gpu_buf(N)
    m = _gpu_mbuf(N)
    _upload_col!(b.v1, v_parts, 1, b.h)
    _upload_col!(b.v2, v_parts, 2, b.h)
    _upload_col!(b.g1, G, 1, b.h)
    _upload_col!(b.g2, G, 2, b.h)
    copyto!(b.w, w_parts)
    @cuda threads=_TPB blocks=cld(N*_SLICES, _TPB) _landau_metric_partial!(
        b.p1, b.p2, m.q11, m.q12, m.q22, b.v1, b.v2, b.g1, b.g2, b.w, N, _SLICES,
        ws.bp1[1], ws.bp1[end], ws.bp2[1], ws.bp2[end])
    @cuda threads=256 blocks=cld(N, 256) _reduce_partials!(
        b.o1, b.o2, b.p1, b.p2, N, _SLICES)
    @cuda threads=256 blocks=cld(N, 256) _reduce_partials!(
        m.a11, m.a12, m.q11, m.q12, N, _SLICES)
    @cuda threads=256 blocks=cld(N, 256) _reduce_partial1!(m.a22, m.q22, N, _SLICES)
    for (dst, col, src) in ((dot_v, 1, b.o1), (dot_v, 2, b.o2),
                            (A, 1, m.a11), (A, 2, m.a12), (A, 3, m.a22))
        copyto!(b.h, src)
        @inbounds for i in 1:N
            dst[i, col] = b.h[i]
        end
    end
    return nothing
end
