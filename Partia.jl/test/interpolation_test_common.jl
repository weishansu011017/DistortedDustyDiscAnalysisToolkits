using Random
using Partia

"""Shared test helpers for interpolation-related testsets."""

kern = M4_spline()

# Small comparison utilities

approx_with_nan(a, b; atol, rtol) = isequal(a, b) || isapprox(a, b; atol = atol, rtol = rtol)
approx_with_nan(a::Tuple, b::Tuple; atol, rtol) = all(approx_with_nan(ai, bi; atol = atol, rtol = rtol) for (ai, bi) in zip(a, b))

# Kernel support and geometric helpers

@inline function support_radius(strategy, ha, hb, Kvalid)
    return strategy === itpSymmetric ?
        Kvalid * max(ha, hb) :
        Kvalid * (strategy === itpGather ? ha : hb)
end

@inline within_radius(d2, radius) = d2 <= radius * radius

@inline function kernel_weight(ref::NTuple{3,T}, rb::NTuple{3,T}, ha::T, hb::T, strategy) where {T<:AbstractFloat}
    if strategy === itpSymmetric
        return T(0.5) * (
            Smoothed_kernel_function(typeof(kern), ref, rb, ha) +
            Smoothed_kernel_function(typeof(kern), ref, rb, hb)
        )
    end
    hsel = strategy === itpGather ? ha : hb
    return Smoothed_kernel_function(typeof(kern), ref, rb, hsel)
end

@inline function kernel_gradient(ref::NTuple{3,T}, rb::NTuple{3,T}, ha::T, hb::T, strategy) where {T<:AbstractFloat}
    if strategy === itpSymmetric
        ∇Wa = Smoothed_gradient_kernel_function(typeof(kern), ref, rb, ha)
        ∇Wb = Smoothed_gradient_kernel_function(typeof(kern), ref, rb, hb)
        return (
            T(0.5) * (∇Wa[1] + ∇Wb[1]),
            T(0.5) * (∇Wa[2] + ∇Wb[2]),
            T(0.5) * (∇Wa[3] + ∇Wb[3]),
        )
    end
    hsel = strategy === itpGather ? ha : hb
    return Smoothed_gradient_kernel_function(typeof(kern), ref, rb, hsel)
end

@inline function line_integrated_weight(Δr::T, ha::T, hb::T, strategy) where {T<:AbstractFloat}
    if strategy === itpSymmetric
        return T(0.5) * (
            line_integrated_kernel_function(typeof(kern), Δr, ha) +
            line_integrated_kernel_function(typeof(kern), Δr, hb)
        )
    end
    hsel = strategy === itpGather ? ha : hb
    return line_integrated_kernel_function(typeof(kern), Δr, hsel)
end

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

# Brute-force reference evaluators for single-point interpolation

function brute_density(input::InterpolationInput{3,T}, ref::NTuple{3,T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    ρ = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            W = kernel_weight(ref, rb, ha, hb, strategy)
            ρ += input.m[i] * W
        end
    end
    return ρ
end

function brute_number_density(input::InterpolationInput{3,T}, ref::NTuple{3,T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    n = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            W = kernel_weight(ref, rb, ha, hb, strategy)
            n += W
        end
    end
    return n
end

function brute_quantity(input::InterpolationInput{3,T}, ref::NTuple{3,T}, ha::T, col::Int, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    numer = zero(T)
    denom = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            W = kernel_weight(ref, rb, ha, hb, strategy)
            weight = input.m[i] * W / input.ρ[i]
            numer += input.quant[col][i] * weight
            denom += weight
        end
    end
    return iszero(denom) ? T(NaN) : numer / denom
end

function brute_line_integrated_density(input::InterpolationInput{3,T}, origin::NTuple{3,T}, direction::NTuple{3,T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    Sigma = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = squared_distance_point_line(rb, origin, direction)
        if within_radius(d2, radius)
            Δr = sqrt(d2)
            W = line_integrated_weight(Δr, ha, hb, strategy)
            Sigma += input.m[i] * W
        end
    end
    return Sigma
end

function brute_gradient_density(input::InterpolationInput{3,T}, ref::NTuple{3,T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    ∇ρf = zeros(T, 3)
    ∇ρb = zeros(T, 3)
    ρ = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            W = kernel_weight(ref, rb, ha, hb, strategy)
            ∇W = kernel_gradient(ref, rb, ha, hb, strategy)
            mb = input.m[i]
            ρb = input.ρ[i]
            invρb = inv(ρb)
            ρ += mb * W
            ∇ρf[1] += mb * ∇W[1]
            ∇ρf[2] += mb * ∇W[2]
            ∇ρf[3] += mb * ∇W[3]
            ∇ρb[1] += mb * invρb * ∇W[1]
            ∇ρb[2] += mb * invρb * ∇W[2]
            ∇ρb[3] += mb * invρb * ∇W[3]
        end
    end
    iszero(ρ) && return (T(NaN), T(NaN), T(NaN))
    return (
        ∇ρf[1] - ρ * ∇ρb[1],
        ∇ρf[2] - ρ * ∇ρb[2],
        ∇ρf[3] - ρ * ∇ρb[3],
    )
end

function brute_gradient_quantity(input::InterpolationInput{3,T}, ref::NTuple{3,T}, ha::T, col::Int, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    ∇Af = zeros(T, 3)
    ∇Ab = zeros(T, 3)
    A = zero(T)
    S1 = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            W = kernel_weight(ref, rb, ha, hb, strategy)
            ∇W = kernel_gradient(ref, rb, ha, hb, strategy)
            mb = input.m[i]
            ρb = input.ρ[i]
            Ab = input.quant[col][i]
            invρb = inv(ρb)
            weight = mb * W * invρb
            grad_weight_x = mb * invρb * ∇W[1]
            grad_weight_y = mb * invρb * ∇W[2]
            grad_weight_z = mb * invρb * ∇W[3]
            A += Ab * weight
            S1 += weight
            ∇Af[1] += Ab * grad_weight_x
            ∇Af[2] += Ab * grad_weight_y
            ∇Af[3] += Ab * grad_weight_z
            ∇Ab[1] += grad_weight_x
            ∇Ab[2] += grad_weight_y
            ∇Ab[3] += grad_weight_z
        end
    end
    iszero(S1) && return (T(NaN), T(NaN), T(NaN))
    A /= S1
    return (
        ∇Af[1] - A * ∇Ab[1],
        ∇Af[2] - A * ∇Ab[2],
        ∇Af[3] - A * ∇Ab[3],
    )
end

function brute_divergence(input::InterpolationInput{3,T}, ref::NTuple{3,T}, ha::T, cols::NTuple{3,Int}, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    ∇Af = zero(T)
    ∇Axb = zero(T)
    ∇Ayb = zero(T)
    ∇Azb = zero(T)
    Ax = zero(T)
    Ay = zero(T)
    Az = zero(T)
    S1 = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            W = kernel_weight(ref, rb, ha, hb, strategy)
            ∇W = kernel_gradient(ref, rb, ha, hb, strategy)
            mb = input.m[i]
            ρb = input.ρ[i]
            Axb = input.quant[cols[1]][i]
            Ayb = input.quant[cols[2]][i]
            Azb = input.quant[cols[3]][i]
            invρb = inv(ρb)
            weight = mb * W * invρb
            grad_weight_x = mb * invρb * ∇W[1]
            grad_weight_y = mb * invρb * ∇W[2]
            grad_weight_z = mb * invρb * ∇W[3]
            Ax += Axb * weight
            Ay += Ayb * weight
            Az += Azb * weight
            ∇Af += Axb * grad_weight_x + Ayb * grad_weight_y + Azb * grad_weight_z
            ∇Axb += grad_weight_x
            ∇Ayb += grad_weight_y
            ∇Azb += grad_weight_z
            S1 += weight
        end
    end
    iszero(S1) && return T(NaN)
    Ax /= S1
    Ay /= S1
    Az /= S1
    return ∇Af - (Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb)
end

function brute_curl(input::InterpolationInput{3,T}, ref::NTuple{3,T}, ha::T, cols::NTuple{3,Int}, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    ∇Axf = zero(T)
    ∇Ayf = zero(T)
    ∇Azf = zero(T)
    mlρ∂xW = zero(T)
    mlρ∂yW = zero(T)
    mlρ∂zW = zero(T)
    Ax = zero(T)
    Ay = zero(T)
    Az = zero(T)
    S1 = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (ref[1] - rb[1])^2 + (ref[2] - rb[2])^2 + (ref[3] - rb[3])^2
        if within_radius(d2, radius)
            W = kernel_weight(ref, rb, ha, hb, strategy)
            ∇W = kernel_gradient(ref, rb, ha, hb, strategy)
            mb = input.m[i]
            ρb = input.ρ[i]
            Axb = input.quant[cols[1]][i]
            Ayb = input.quant[cols[2]][i]
            Azb = input.quant[cols[3]][i]
            invρb = inv(ρb)
            weight = mb * W * invρb
            grad_weight_x = mb * invρb * ∇W[1]
            grad_weight_y = mb * invρb * ∇W[2]
            grad_weight_z = mb * invρb * ∇W[3]
            Ax += Axb * weight
            Ay += Ayb * weight
            Az += Azb * weight
            ∇Axf += Ayb * grad_weight_z - Azb * grad_weight_y
            ∇Ayf += Azb * grad_weight_x - Axb * grad_weight_z
            ∇Azf += Axb * grad_weight_y - Ayb * grad_weight_x
            mlρ∂xW += grad_weight_x
            mlρ∂yW += grad_weight_y
            mlρ∂zW += grad_weight_z
            S1 += weight
        end
    end
    iszero(S1) && return (T(NaN), T(NaN), T(NaN))
    Ax /= S1
    Ay /= S1
    Az /= S1
    ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
    ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
    ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW
    return (
        -(∇Axf - ∇Axb),
        -(∇Ayf - ∇Ayb),
        -(∇Azf - ∇Azb),
    )
end

# Brute-force reference evaluators for line-integrated interpolation

function brute_line_integrated_quantity(input::InterpolationInput{3,T}, origin::NTuple{3,T}, direction::NTuple{3,T}, ha::T, col::Int, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(kern), T)
    numer = zero(T)
    denom = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (get_xcoord(input)[i], get_ycoord(input)[i], get_zcoord(input)[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = squared_distance_point_line(rb, origin, direction)
        if within_radius(d2, radius)
            Δr = sqrt(d2)
            W = line_integrated_weight(Δr, ha, hb, strategy)
            weight = input.m[i] * W / input.ρ[i]
            numer += input.quant[col][i] * weight
            denom += weight
        end
    end
    return iszero(denom) ? T(NaN) : numer / denom
end

# Synthetic fixtures

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
    input = InterpolationInput((x, y, z), m, h, ρ, (q1, q2, q3); smoothed_kernel = typeof(kern))
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
    input = InterpolationInput((x, y, z), m, h, ρ, (q1,); smoothed_kernel = typeof(kern))
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
    input = InterpolationInput((x, y, z), m, h, ρ, (q1,); smoothed_kernel = typeof(kern))
    return input, LinearBVH!(input, Val(3))
end
