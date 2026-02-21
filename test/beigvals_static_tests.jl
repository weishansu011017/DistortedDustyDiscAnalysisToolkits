using Test
using LinearAlgebra, StaticArrays, Random
using PhantomRevealer.BatchEigvals
import PhantomRevealer.BatchEigvals: _scale!, _unscale!

@testset "beigvals vs eigvals" begin
	Random.seed!(1234)
	for T in (ComplexF64, ComplexF32)
		atol = T === ComplexF64 ? 1e-12 : 1e-5
		for N in 1:15
			A0 = randn(T, N, N)
			M = MMatrix{N,N,T}(A0)
			W = beigvals!(M)
			Wref = eigvals(Matrix(A0))
			sort!(W, by = x -> (real(x), imag(x)))
			sort!(Wref, by = x -> (real(x), imag(x)))
			@test maximum(abs.(W .- Wref)) ≤ atol
		end
	end
end

@testset "scale/unscale extremes" begin
	Random.seed!(5678)
	extremes = [(ComplexF64, 1e300), (ComplexF64, 1e-300), (ComplexF32, 1f30), (ComplexF32, 1f-30)]
	for (T, fac) in extremes
		A0 = randn(T, 4, 4) .* fac
		M = MMatrix{4,4,T}(A0)
		scaled, α = PhantomRevealer.BatchEigvals._scale!(M)
		Wscaled = eigvals(Matrix(M))
		W = MVector{4,T}(undef)
		@inbounds for i in 1:4
			W[i] = Wscaled[i]
		end
		PhantomRevealer.BatchEigvals._unscale!(W, α)
		Wref = eigvals(Matrix(A0))
		sort!(W, by = x -> (real(x), imag(x)))
		sort!(Wref, by = x -> (real(x), imag(x)))
		atol = T === ComplexF64 ? 1e-12 : 1e-5
		@test scaled == true
		@test maximum(abs.(W .- Wref)) ≤ atol
	end
end
