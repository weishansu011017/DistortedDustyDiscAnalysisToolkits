# ──────────────────────────────────────────────────────────────────────────── #
#  Benchmark: tiny_eigvals — CPU (serial / threaded) vs CUDA GPU
# ──────────────────────────────────────────────────────────────────────────── #
#
#  This script tests whether `tiny_eigvals` can run inside a CUDA kernel
#  and compares performance (throughput) between CPU and GPU for batch
#  eigenvalue computation of 8×8 complex matrices (the size used by the
#  Streaming Instability growth-rate solver).
#
#  Usage:
#    julia --project=. -t 16 test/benchmark_tiny_eigvals_gpu.jl        # 16 threads
#    julia --project=. -t auto test/benchmark_tiny_eigvals_gpu.jl      # all cores
#
#  Requirements: CUDA.jl must be installed and a CUDA-capable GPU available.
#
# ──────────────────────────────────────────────────────────────────────────── #

using Partia
using CUDA
using StaticArrays
using BenchmarkTools
using LinearAlgebra
using Test

# ========================== Setup =========================================== #

println("=" ^ 78)
println("  tiny_eigvals — CPU (serial / threaded) vs CUDA GPU Benchmark")
println("=" ^ 78)
println()

# System info
nthreads_avail = Threads.nthreads()
println("CPU Threads: ", nthreads_avail)
println("GPU Device : ", CUDA.name(CUDA.device()))
println("CUDA       : ", CUDA.runtime_version())
println("Julia      : ", VERSION)
println()

# ========================== Helper: build test matrices ===================== #

# Reuse the SI growth-rate problem (8×8) to generate realistic test matrices.
# Physical parameters (same as growthrate_streaminginstability.jl)
const Hlr      = 0.05
const η_param  = 0.0025
const ηvₖlcₛ   = η_param / Hlr
const invηvₖlcₛ = inv(ηvₖlcₛ)

@inline ΚH(kηr::Float64) = invηvₖlcₛ * kηr

Δ(ε, τ)       = (1 + ε)^2 + τ^2
uxlcₛ(ε, τ)   =  (ηvₖlcₛ) * (2ε * τ)            / Δ(ε, τ)
uylcₛ(ε, τ)   = -(ηvₖlcₛ) * (1 + ε*τ^2/Δ(ε,τ))   / (1 + ε)
wxlcₛ(ε, τ)   = -(ηvₖlcₛ) * 2τ                   / Δ(ε, τ)
wylcₛ(ε, τ)   = -(ηvₖlcₛ) * (1 - τ^2/Δ(ε,τ))     / (1 + ε)

"""Build an 8×8 SI perturbation matrix for given (St, ε, Κx, Κz)."""
function build_SI_matrix(St::Float64, ε::Float64, Κx::Float64, Κz::Float64)
    vx = uxlcₛ(ε, St)
    vy = uylcₛ(ε, St)
    ωx = wxlcₛ(ε, St)
    ωy = wylcₛ(ε, St)
    ρg = 1.0
    ρd = ρg * ε
    invSt  = 1.0 / St
    εinvSt = ε * invSt
    Rx = εinvSt * (ωx - vx)
    Ry = εinvSt * (ωy - vy)
    A  = -im * Κx * ωx
    B  = -im * Κx * vx

    M = zero(MMatrix{8, 8, ComplexF64})
    @inbounds begin
        M[1,1] = A;                M[6,1] = Rx;              M[7,1] = Ry
        M[1,2] = -im * Κx;        M[2,2] = A - invSt;       M[3,2] = -0.5;       M[6,2] = εinvSt
        M[2,3] = 2;               M[3,3] = A - invSt;       M[7,3] = εinvSt
        M[1,4] = -im * Κz;        M[4,4] = A - invSt;       M[8,4] = εinvSt
        M[5,5] = B;               M[6,5] = (-im*Κx) - Rx;   M[7,5] = -Ry;        M[8,5] = -im * Κz
        M[2,6] = invSt;           M[5,6] = -im * Κx;        M[6,6] = B - εinvSt; M[7,6] = -0.5
        M[3,7] = invSt;           M[6,7] = 2;               M[7,7] = B - εinvSt
        M[4,8] = invSt;           M[5,8] = -im * Κz;        M[8,8] = B - εinvSt
    end
    return SMatrix{8,8,ComplexF64}(M)
end

# ========================== Test 1: Correctness on GPU ====================== #

println("-" ^ 72)
println("  Test 1: Verify tiny_eigvals runs correctly inside a CUDA kernel")
println("-" ^ 72)

# Build a single test matrix (linA parameters)
test_mat = build_SI_matrix(0.1, 3.0, ΚH(30.0), ΚH(30.0))

# CPU result
cpu_eigvals = tiny_eigvals(test_mat)
cpu_maxreal = maximum(real, cpu_eigvals)
println("  CPU max Re(λ) = ", cpu_maxreal)
println("  Expected ≈ 0.4190204 (linA growth rate)")

# GPU kernel: compute tiny_eigvals for a single matrix
function gpu_eigvals_kernel!(output, mat)
    eigs = tiny_eigvals(mat)
    λmax = -Inf
    for k in SOneTo(8)
        r = real(eigs[k])
        λmax = ifelse(r > λmax, r, λmax)
    end
    @inbounds output[1] = λmax
    return nothing
end

gpu_output = CUDA.zeros(Float64, 1)
try
    @cuda threads=1 gpu_eigvals_kernel!(gpu_output, test_mat)
    synchronize()
    gpu_maxreal = Array(gpu_output)[1]
    println("  GPU max Re(λ) = ", gpu_maxreal)
    
    @testset "GPU tiny_eigvals correctness" begin
        @test gpu_maxreal ≈ cpu_maxreal rtol=1e-10
        @test gpu_maxreal ≈ 0.4190204 rtol=1e-3
    end
    println("  ✓ GPU result matches CPU!\n")
catch e
    println("  ✗ GPU kernel failed: ", e)
    println("  This likely means tiny_eigvals uses constructs not supported on GPU.\n")
    rethrow(e)
end

# ========================== Test 2: Batch benchmark ========================= #

println("-" ^ 78)
println("  Test 2: Batch performance — CPU (serial / threaded) vs GPU")
println("-" ^ 78)

# ── Problem sizes to benchmark ──────────────────────────────────────── #
problem_dims = [128, 256, 512, 1024]

St_val = 0.1
ε_val  = 3.0

# ── Kernel functions ────────────────────────────────────────────────── #

function cpu_batch_serial!(results::Vector{Float64}, mats::Vector{SMatrix{8,8,ComplexF64,64}})
    @inbounds for i in eachindex(mats)
        eigs = tiny_eigvals(mats[i])
        λmax = -Inf
        for k in SOneTo(8)
            r = real(eigs[k])
            λmax = ifelse(r > λmax, r, λmax)
        end
        results[i] = λmax
    end
    return nothing
end

function cpu_batch_threaded!(results::Vector{Float64}, mats::Vector{SMatrix{8,8,ComplexF64,64}})
    Threads.@threads for i in eachindex(mats)
        eigs = tiny_eigvals(mats[i])
        λmax = -Inf
        for k in SOneTo(8)
            r = real(eigs[k])
            λmax = ifelse(r > λmax, r, λmax)
        end
        @inbounds results[i] = λmax
    end
    return nothing
end

function gpu_batch_kernel!(results, mats, N)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N
        @inbounds begin
            eigs = tiny_eigvals(mats[idx])
            λmax = -Inf
            for k in SOneTo(8)
                r = real(eigs[k])
                λmax = ifelse(r > λmax, r, λmax)
            end
            results[idx] = λmax
        end
    end
    return nothing
end

# ── Storage for summary table ───────────────────────────────────────── #
nthreads = Threads.nthreads()

summary_header  = "  " * rpad("N×N", 12) * rpad("Nproblems", 12) *
                  rpad("CPU serial", 16) * rpad("CPU $(nthreads)T", 16) *
                  rpad("GPU", 16) *
                  rpad("GPU/serial", 14) * rpad("GPU/thread", 14)
summary_lines   = String[]

for N_per_dim in problem_dims
    Nproblems = N_per_dim * N_per_dim

    Κxs = range(ΚH(1.0), ΚH(100.0), length=N_per_dim)
    Κzs = range(ΚH(1.0), ΚH(100.0), length=N_per_dim)

    # Build all matrices
    all_mats = [build_SI_matrix(St_val, ε_val, Float64(kx), Float64(kz)) for kx in Κxs, kz in Κzs]
    all_mats_flat = vec(all_mats)

    println()
    println("  ── $N_per_dim × $N_per_dim = $Nproblems problems ──")

    # ── CPU serial ────────────────────────────────────────── #
    cpu_results = Vector{Float64}(undef, Nproblems)
    cpu_batch_serial!(cpu_results, all_mats_flat)  # warmup

    print("    CPU (serial)    : ")
    cpu_bench = @benchmark cpu_batch_serial!($cpu_results, $all_mats_flat) samples=20 evals=1
    display(cpu_bench)
    println()

    # ── CPU threaded ──────────────────────────────────────── #
    cpu_t_results = Vector{Float64}(undef, Nproblems)
    cpu_batch_threaded!(cpu_t_results, all_mats_flat)  # warmup

    print("    CPU ($nthreads threads) : ")
    cpu_t_bench = @benchmark cpu_batch_threaded!($cpu_t_results, $all_mats_flat) samples=20 evals=1
    display(cpu_t_bench)
    println()

    # ── GPU ───────────────────────────────────────────────── #
    gpu_mats    = CuArray(all_mats_flat)
    gpu_results = CUDA.zeros(Float64, Nproblems)

    threads_per_block = 256
    nblocks = cld(Nproblems, threads_per_block)

    # Warmup
    @cuda threads=threads_per_block blocks=nblocks gpu_batch_kernel!(gpu_results, gpu_mats, Nproblems)
    synchronize()

    # Verify correctness
    gpu_results_host = Array(gpu_results)
    @testset "GPU correctness N=$N_per_dim" begin
        @test all(isapprox.(gpu_results_host, cpu_results, rtol=1e-6))
    end

    print("    GPU (CUDA)      : ")
    gpu_bench = @benchmark begin
        @cuda threads=$threads_per_block blocks=$nblocks gpu_batch_kernel!($gpu_results, $gpu_mats, $Nproblems)
        synchronize()
    end samples=20 evals=1
    display(gpu_bench)
    println()

    # ── Collect summary ───────────────────────────────────── #
    t_serial  = median(cpu_bench.times) / 1e6
    t_thread  = median(cpu_t_bench.times) / 1e6
    t_gpu     = median(gpu_bench.times) / 1e6

    line = "  " * rpad("$(N_per_dim)×$(N_per_dim)", 12) *
           rpad("$Nproblems", 12) *
           rpad("$(round(t_serial, digits=2)) ms", 16) *
           rpad("$(round(t_thread, digits=2)) ms", 16) *
           rpad("$(round(t_gpu, digits=3)) ms", 16) *
           rpad("$(round(t_serial / t_gpu, digits=1))×", 14) *
           rpad("$(round(t_thread / t_gpu, digits=1))×", 14)
    push!(summary_lines, line)

    # Free GPU memory
    CUDA.unsafe_free!(gpu_mats)
    CUDA.unsafe_free!(gpu_results)
end

# ========================== Summary ========================================= #

println()
println("=" ^ 100)
println("  Summary  (CPU threads = $nthreads)")
println("=" ^ 100)
println(summary_header)
println("  " * "-" ^ 96)
for l in summary_lines
    println(l)
end
println("=" ^ 100)
