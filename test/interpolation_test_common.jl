using Random
using PhantomRevealer

const KERN = M4_spline()
const KI_COMMON = PhantomRevealer.KernelInterpolation

@inline function support_radius(strategy, ha, hb, Kvalid)
    return strategy === itpSymmetric ?
        Kvalid * max(ha, hb) :
        Kvalid * (strategy === itpGather ? ha : hb)
end

@inline within_radius(d2, radius) = d2 <= radius * radius

@inline function squared_distance_point_line(point::NTuple{3,T}, origin::NTuple{3,T}, direction::NTuple{3,T}) where {T<:AbstractFloat}
    Δ2 = zero(T)
    Δm = zero(T)
    @inbounds for d in 1:3
        Δ = point[d] - origin[d]
        Δ2 += Δ * Δ
        Δm += Δ * direction[d]
    end
    return Δ2 - Δm * Δm
end

function brute_density(input::InterpolationInput{T}, ref::NTuple{3,T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    ρ = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            if strategy === itpSymmetric
                W = T(0.5) * (Smoothed_kernel_function(typeof(KERN), ref, rb, ha) +
                              Smoothed_kernel_function(typeof(KERN), ref, rb, hb))
            else
                hsel = strategy === itpGather ? ha : hb
                W = Smoothed_kernel_function(typeof(KERN), ref, rb, hsel)
            end
            ρ += input.m[i] * W
        end
    end
    return ρ
end

function brute_number_density(input::InterpolationInput{T}, ref::NTuple{3,T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    n = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            if strategy === itpSymmetric
                W = T(0.5) * (Smoothed_kernel_function(typeof(KERN), ref, rb, ha) +
                              Smoothed_kernel_function(typeof(KERN), ref, rb, hb))
            else
                hsel = strategy === itpGather ? ha : hb
                W = Smoothed_kernel_function(typeof(KERN), ref, rb, hsel)
            end
            n += W
        end
    end
    return n
end

function brute_quantity(input::InterpolationInput{T}, ref::NTuple{3,T}, ha::T, col::Int, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    numer = zero(T)
    denom = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            if strategy === itpSymmetric
                W = T(0.5) * (Smoothed_kernel_function(typeof(KERN), ref, rb, ha) +
                              Smoothed_kernel_function(typeof(KERN), ref, rb, hb))
            else
                hsel = strategy === itpGather ? ha : hb
                W = Smoothed_kernel_function(typeof(KERN), ref, rb, hsel)
            end
            weight = input.m[i] * W / input.ρ[i]
            numer += input.quant[col][i] * weight
            denom += weight
        end
    end
    return iszero(denom) ? T(NaN) : numer / denom
end

function brute_line_integrated_density(input::InterpolationInput{T}, origin::NTuple{3,T}, direction::NTuple{3,T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    Sigma = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = squared_distance_point_line(rb, origin, direction)
        if within_radius(d2, radius)
            Δr = sqrt(d2)
            if strategy === itpSymmetric
                W = T(0.5) * (line_integrated_kernel_function(typeof(KERN), Δr, ha) +
                              line_integrated_kernel_function(typeof(KERN), Δr, hb))
            else
                hsel = strategy === itpGather ? ha : hb
                W = line_integrated_kernel_function(typeof(KERN), Δr, hsel)
            end
            Sigma += input.m[i] * W
        end
    end
    return Sigma
end

function random_input_3d(rng::AbstractRNG, n::Int)
    x = rand(rng, n)
    y = rand(rng, n)
    z = rand(rng, n)
    m = rand(rng, n) .+ 0.05
    h = rand(rng, n) .* 0.12 .+ 0.04
    ρ = rand(rng, n) .+ 0.3
    q1 = rand(rng, n)
    q2 = rand(rng, n)
    q3 = rand(rng, n)
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 3}(
        n, KERN, x, y, z, m, h, ρ, (q1, q2, q3))
    LBVH = LinearBVH!(input, Val(3))
    return input, LBVH
end

function random_input_line_integrated(rng::AbstractRNG, n::Int)
    x = rand(rng, n)
    y = rand(rng, n)
    z = rand(rng, n)
    m = rand(rng, n) .+ 0.05
    h = rand(rng, n) .* 0.08 .+ 0.02
    ρ = rand(rng, n) .+ 0.2
    q1 = rand(rng, n)
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 1}(
        n, KERN, x, y, z, m, h, ρ, (q1,))
    LBVH = LinearBVH!(input, Val(3))
    return input, LBVH
end

function make_empty_input()
    x = [10.0]
    y = [10.0]
    z = [10.0]
    m = [1.0]
    h = [0.01]
    ρ = [1.0]
    q1 = [0.0]
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 1}(
        1, KERN, x, y, z, m, h, ρ, (q1,))
    return input, LinearBVH!(input, Val(3))
end
