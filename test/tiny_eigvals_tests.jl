using Test
using LinearAlgebra, StaticArrays, Random
using PhantomRevealer.TinyEigvals
import PhantomRevealer.TinyEigvals: _scale!, _unscale!

@testset "tiny_eigvals vs eigvals" begin
	Random.seed!(1234)
	for T in (ComplexF64, ComplexF32)
		rtol = T === ComplexF64 ? 1e-12 : 1e-5
        atol = T === ComplexF64 ? 1e-10 : 1e-3
		for N in 1:15
			A0 = 1e8 .* (randn(T, N, N) .- 1.0)
			M = MMatrix{N,N,T}(A0)
			W = tiny_eigvals!(M)
			Wref = eigvals(Matrix(A0))
            Wvector = collect(W)
			sort!(Wvector, by = x -> (real(x), imag(x)))
			sort!(Wref, by = x -> (real(x), imag(x)))
			@test isapprox(Wvector, Wref; rtol=rtol, atol=atol)
		end
	end
end

@testset "scale/unscale extremes" begin
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

        scaled = copy(M)
        α = _scale!(scaled)

        Wscaled = eigvals(Matrix(scaled))
        W = MVector{4,T}(Wscaled)

        _unscale!(W, α)

        Wref = eigvals(Matrix(A0))

        sort!(W, by = x -> (real(x), imag(x)))
        sort!(Wref, by = x -> (real(x), imag(x)))

        atol = T === ComplexF64 ? 1e-12 : 1e-5
        @test maximum(abs.(W .- Wref)) ≤ atol
    end
end
