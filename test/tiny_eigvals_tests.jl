# ──────────────────────────────────────────────────────────────────────────── #
#  Test: TinyEigvals — Fixed-Size Eigenvalue Solver
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ─────────────────────
#  Validates the custom Hessenberg-QR eigenvalue solver `tiny_eigvals!`
#  against LAPACK's `eigvals` for all supported matrix sizes N = 1…15.
#
#  1. Correctness (tiny_eigvals! vs LAPACK eigvals)
#     • For every N ∈ {1,…,15} and T ∈ {ComplexF64, ComplexF32}, generates
#       a random matrix scaled by 1e8 and compares sorted eigenvalue tuples.
#     • Tolerances: rtol = 1e-12 / atol = 1e-10 for Float64,
#                   rtol = 1e-5  / atol = 1e-3  for Float32.
#
#  2. Scaling robustness (_scale! / _unscale! round-trip)
#     • Verifies that extreme magnitudes (1e±300 for Float64, 1e±30 for
#       Float32) are correctly handled by the pre-/post-scaling routines.
#     • Eigenvalues after scale → LAPACK → unscale must match direct LAPACK
#       on the original matrix.
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using LinearAlgebra
using StaticArrays
using Random
using PhantomRevealer.TinyEigvals
import PhantomRevealer.TinyEigvals: _scale!, _unscale!

# ============================== Test body =================================== #

# ── 1. Correctness — tiny_eigvals! vs LAPACK eigvals ─────────────────── #

@testset "tiny_eigvals! vs LAPACK eigvals" begin
    Random.seed!(1234)
    for T in (ComplexF64, ComplexF32)
        rtol = T === ComplexF64 ? 1e-12 : 1e-5
        atol = T === ComplexF64 ? 1e-10 : 1e-3
        for N in 1:15
            A0 = 1e8 .* (randn(T, N, N) .- 1.0)
            M = MMatrix{N,N,T}(A0)
            W = tiny_eigvals!(M)
            Wref = eigvals(Matrix(A0))

            Wvec = collect(W)
            sort!(Wvec, by = x -> (real(x), imag(x)))
            sort!(Wref, by = x -> (real(x), imag(x)))

            @test isapprox(Wvec, Wref; rtol=rtol, atol=atol)
        end
    end
end

# ── 2. Scaling robustness — extreme magnitudes ───────────────────────── #

@testset "scale/unscale — extreme magnitudes" begin
    Random.seed!(5678)

    extremes = [
        (ComplexF64, 1e300),
        (ComplexF64, 1e-300),
        (ComplexF32, 1f30),
        (ComplexF32, 1f-30),
    ]

    for (T, fac) in extremes
        A0 = randn(T, 4, 4) .* fac
        M = MMatrix{4,4,T}(A0)

        # Scale → LAPACK → Unscale
        scaled = copy(M)
        α = _scale!(scaled)
        Wscaled = eigvals(Matrix(scaled))
        W = MVector{4,T}(Wscaled)
        _unscale!(W, α)

        # Direct LAPACK reference
        Wref = eigvals(Matrix(A0))

        sort!(W,    by = x -> (real(x), imag(x)))
        sort!(Wref, by = x -> (real(x), imag(x)))

        atol_val = T === ComplexF64 ? 1e-12 : 1e-5
        @test maximum(abs.(W .- Wref)) ≤ atol_val
    end
end
