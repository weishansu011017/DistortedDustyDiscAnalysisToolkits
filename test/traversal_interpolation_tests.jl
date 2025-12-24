using Test
using Random
using PhantomRevealer

const KERN = M4_spline()

function _brute_density(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    xs, ys, zs = input.x, input.y, input.z
    ms, hs = input.m, input.h
    ρ = zero(T)
    @inbounds for i in 1:input.Npart
        hb = hs[i]
        rb = (xs[i], ys[i], zs[i])
        hsel = strategy === itpGather ? ha : hb
        radius = strategy === itpSymmetric ? Kvalid * max(ha, hb) : Kvalid * hsel
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if d2 <= radius * radius
            if strategy === itpSymmetric
                W = T(0.5) * (Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha) + Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb))
            else
                W = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            ρ += ms[i] * W
        end
    end
    return ρ
end

function _brute_number_density(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    xs, ys, zs = input.x, input.y, input.z
    hs = input.h
    n = zero(T)
    @inbounds for i in 1:input.Npart
        hb = hs[i]
        rb = (xs[i], ys[i], zs[i])
        hsel = strategy === itpGather ? ha : hb
        radius = strategy === itpSymmetric ? Kvalid * max(ha, hb) : Kvalid * hsel
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if d2 <= radius * radius
            if strategy === itpSymmetric
                W = T(0.5) * (Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha) + Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb))
            else
                W = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            n += W
        end
    end
    return n
end

function _brute_quantity(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, column::Int, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    xs, ys, zs = input.x, input.y, input.z
    ms, hs, ρs = input.m, input.h, input.ρ
    As = input.quant[column]
    numer = denom = zero(T)
    @inbounds for i in 1:input.Npart
        hb = hs[i]
        rb = (xs[i], ys[i], zs[i])
        hsel = strategy === itpGather ? ha : hb
        radius = strategy === itpSymmetric ? Kvalid * max(ha, hb) : Kvalid * hsel
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if d2 <= radius * radius
            if strategy === itpSymmetric
                W = T(0.5) * (Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha) + Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb))
            else
                W = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            weight = ms[i] * W / ρs[i]
            numer += As[i] * weight
            denom += weight
        end
    end
    return numer / denom
end

@testset "Traversal interpolation matches brute force" begin
    rng = MersenneTwister(0xBADA55)
    n = 200
    x = rand(rng, n)
    y = rand(rng, n)
    z = rand(rng, n)
    m = rand(rng, n) .+ 0.1
    h = rand(rng, n) .* 0.15 .+ 0.05
    ρ = rand(rng, n) .+ 0.5
    q1 = rand(rng, n)
    q2 = rand(rng, n)
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 2}(n, KERN, x, y, z, m, h, ρ, (q1, q2))
    LBVH = LinearBVH!(input, Val(3))

    reference_point = (0.4, 0.35, 0.25)
    ha = 0.12

    for strategy in (itpGather, itpScatter, itpSymmetric)
        @test density(input, reference_point, ha, LBVH, strategy) ≈ _brute_density(input, reference_point, ha, strategy) atol=1e-10 rtol=1e-8
        @test number_density(input, reference_point, ha, LBVH, strategy) ≈ _brute_number_density(input, reference_point, ha, strategy) atol=1e-10 rtol=1e-8
        @test quantity_interpolate(input, reference_point, ha, LBVH, 1, true, strategy) ≈ _brute_quantity(input, reference_point, ha, 1, strategy) atol=1e-10 rtol=1e-8
    end
end

function _brute_LOS_density(input::InterpolationInput{T}, reference_point::NTuple{2, T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    xs, ys, hs, ms = input.x, input.y, input.h, input.m
    Σ = zero(T)
    @inbounds for i in 1:input.Npart
        hb = hs[i]
        rb = (xs[i], ys[i])
        hsel = strategy === itpGather ? ha : hb
        radius = strategy === itpSymmetric ? Kvalid * max(ha, hb) : Kvalid * hsel
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2
        if d2 <= radius * radius
            if strategy === itpSymmetric
                W = T(0.5) * (Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha) + Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb))
            else
                W = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            Σ += ms[i] * W
        end
    end
    return Σ
end

@testset "LOS traversal matches brute force" begin
    rng = MersenneTwister(0xF00D)
    n = 150
    x = rand(rng, n)
    y = rand(rng, n)
    z = zeros(Float64, n)
    m = rand(rng, n) .+ 0.1
    h = rand(rng, n) .* 0.1 .+ 0.04
    ρ = rand(rng, n) .+ 0.3
    q1 = rand(rng, n)
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 1}(n, KERN, x, y, z, m, h, ρ, (q1,))
    LBVH = LinearBVH!(input, Val(2))

    reference_point = (0.2, 0.8)
    ha = 0.08

    for strategy in (itpGather, itpScatter, itpSymmetric)
        @test LOS_density(input, reference_point, ha, LBVH, strategy) ≈ _brute_LOS_density(input, reference_point, ha, strategy) atol=1e-10 rtol=1e-8
    end
end