using Random
using PhantomRevealer

const KERN = M4_spline()
const KI = PhantomRevealer.KernelInterpolation

# Support radius helpers
@inline function support_radius(strategy, ha, hb, Kvalid)
    return strategy === itpSymmetric ? Kvalid * max(ha, hb) : Kvalid * (strategy === itpGather ? ha : hb)
end

@inline function within_radius(d2, radius)
    return d2 <= radius * radius
end

# ----------------------
# Brute-force baselines
# ----------------------

function brute_density(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    ρ = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if within_radius(d2, radius)
            if strategy === itpSymmetric
                Wah = Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha)
                Wbh = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb)
                W = T(0.5) * (Wah + Wbh)
            else
                hsel = strategy === itpGather ? ha : hb
                W = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            ρ += input.m[i] * W
        end
    end
    return ρ
end

function brute_number_density(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    n = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if within_radius(d2, radius)
            if strategy === itpSymmetric
                Wah = Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha)
                Wbh = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb)
                W = T(0.5) * (Wah + Wbh)
            else
                hsel = strategy === itpGather ? ha : hb
                W = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            n += W
        end
    end
    return n
end

function brute_quantity(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, column::Int, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    numer = zero(T)
    denom = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if within_radius(d2, radius)
            if strategy === itpSymmetric
                Wah = Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha)
                Wbh = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb)
                W = T(0.5) * (Wah + Wbh)
            else
                hsel = strategy === itpGather ? ha : hb
                W = Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            weight = input.m[i] * W / input.ρ[i]
            numer += input.quant[column][i] * weight
            denom += weight
        end
    end
    return iszero(denom) ? T(NaN) : numer / denom
end

function brute_gradient_density(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    ∇ρxf = zero(T)
    ∇ρyf = zero(T)
    ∇ρzf = zero(T)
    ∇ρxb = zero(T)
    ∇ρyb = zero(T)
    ∇ρzb = zero(T)
    ρ = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if within_radius(d2, radius)
            mb = input.m[i]
            ρb = input.ρ[i]
            if strategy === itpSymmetric
                Δf = KI._gradient_density_accumulation(reference_point, rb, mb, ρb, ha, hb, KERN)
                ρ += KI._density_accumulation(reference_point, rb, mb, ha, hb, KERN)
            else
                hsel = strategy === itpGather ? ha : hb
                Δf = KI._gradient_density_accumulation(reference_point, rb, mb, ρb, hsel, KERN)
                ρ += KI._density_accumulation(reference_point, rb, mb, hsel, KERN)
            end
            ∇ρxf += Δf[1]
            ∇ρyf += Δf[2]
            ∇ρzf += Δf[3]
            ∇ρxb += Δf[4]
            ∇ρyb += Δf[5]
            ∇ρzb += Δf[6]
        end
    end
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end
    ∇ρxb *= ρ
    ∇ρyb *= ρ
    ∇ρzb *= ρ
    return (∇ρxf - ∇ρxb, ∇ρyf - ∇ρyb, ∇ρzf - ∇ρzb)
end

function brute_gradient_quantity(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, column::Int, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    ∇Axf = zero(T)
    ∇Ayf = zero(T)
    ∇Azf = zero(T)
    ∇Axb = zero(T)
    ∇Ayb = zero(T)
    ∇Azb = zero(T)
    mWlρ = zero(T)
    A = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if within_radius(d2, radius)
            mb = input.m[i]
            ρb = input.ρ[i]
            Ab = input.quant[column][i]
            if strategy === itpSymmetric
                Δf = KI._gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, KERN)
                A += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, KERN)
                mWlρ += KI._ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, KERN)
            else
                hsel = strategy === itpGather ? ha : hb
                Δf = KI._gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, hsel, KERN)
                A += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hsel, KERN)
                mWlρ += KI._ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hsel, KERN)
            end
            ∇Axf += Δf[1]
            ∇Ayf += Δf[2]
            ∇Azf += Δf[3]
            ∇Axb += Δf[4]
            ∇Ayb += Δf[5]
            ∇Azb += Δf[6]
        end
    end
    if iszero(mWlρ)
        return (T(NaN), T(NaN), T(NaN))
    end
    A /= mWlρ
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A
    return (∇Axf - ∇Axb, ∇Ayf - ∇Ayb, ∇Azf - ∇Azb)
end

function brute_divergence(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, cols::NTuple{3, Int}, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    Ax = zero(T)
    Ay = zero(T)
    Az = zero(T)
    ∇Af = zero(T)
    ∇Axb = zero(T)
    ∇Ayb = zero(T)
    ∇Azb = zero(T)
    mWlρ = zero(T)
    ax_idx, ay_idx, az_idx = cols
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if within_radius(d2, radius)
            mb = input.m[i]
            ρb = input.ρ[i]
            Axb = input.quant[ax_idx][i]
            Ayb = input.quant[ay_idx][i]
            Azb = input.quant[az_idx][i]
            if strategy === itpSymmetric
                Δf = KI._divergence_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, hb, KERN)
                Ax += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, hb, KERN)
                Ay += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, hb, KERN)
                Az += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, hb, KERN)
                mWlρ += KI._ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, KERN)
            else
                hsel = strategy === itpGather ? ha : hb
                Δf = KI._divergence_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, hsel, KERN)
                Ax += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, hsel, KERN)
                Ay += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, hsel, KERN)
                Az += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, hsel, KERN)
                mWlρ += KI._ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hsel, KERN)
            end
            ∇Af += Δf[1]
            ∇Axb += Δf[2]
            ∇Ayb += Δf[3]
            ∇Azb += Δf[4]
        end
    end
    if iszero(mWlρ)
        return T(NaN)
    end
    Ax /= mWlρ
    Ay /= mWlρ
    Az /= mWlρ
    ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb
    return ∇Af - ∇Ab
end

function brute_curl(input::InterpolationInput{T}, reference_point::NTuple{3, T}, ha::T, cols::NTuple{3, Int}, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    Ax = zero(T)
    Ay = zero(T)
    Az = zero(T)
    ∇Axf = zero(T)
    ∇Ayf = zero(T)
    ∇Azf = zero(T)
    mlρ∂xW = zero(T)
    mlρ∂yW = zero(T)
    mlρ∂zW = zero(T)
    mWlρ = zero(T)
    ax_idx, ay_idx, az_idx = cols
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i], input.z[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2 + (reference_point[3] - rb[3])^2
        if within_radius(d2, radius)
            mb = input.m[i]
            ρb = input.ρ[i]
            Axb = input.quant[ax_idx][i]
            Ayb = input.quant[ay_idx][i]
            Azb = input.quant[az_idx][i]
            if strategy === itpSymmetric
                Δf = KI._curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, hb, KERN)
                Ax += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, hb, KERN)
                Ay += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, hb, KERN)
                Az += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, hb, KERN)
                mWlρ += KI._ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, KERN)
            else
                hsel = strategy === itpGather ? ha : hb
                Δf = KI._curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, hsel, KERN)
                Ax += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, hsel, KERN)
                Ay += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, hsel, KERN)
                Az += KI._quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, hsel, KERN)
                mWlρ += KI._ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hsel, KERN)
            end
            ∇Axf += Δf[1]
            ∇Ayf += Δf[2]
            ∇Azf += Δf[3]
            mlρ∂xW += Δf[4]
            mlρ∂yW += Δf[5]
            mlρ∂zW += Δf[6]
        end
    end
    if iszero(mWlρ)
        return (T(NaN), T(NaN), T(NaN))
    end
    Ax /= mWlρ
    Ay /= mWlρ
    Az /= mWlρ
    ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
    ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
    ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW
    return (-(∇Axf - ∇Axb), -(∇Ayf - ∇Ayb), -(∇Azf - ∇Azb))
end

function brute_LOS_density(input::InterpolationInput{T}, reference_point::NTuple{2, T}, ha::T, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    Σ = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2
        if within_radius(d2, radius)
            if strategy === itpSymmetric
                Wah = LOSint_Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha)
                Wbh = LOSint_Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb)
                W = T(0.5) * (Wah + Wbh)
            else
                hsel = strategy === itpGather ? ha : hb
                W = LOSint_Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            Σ += input.m[i] * W
        end
    end
    return Σ
end

function brute_LOS_quantity(input::InterpolationInput{T}, reference_point::NTuple{2, T}, ha::T, column::Int, strategy) where {T}
    Kvalid = KernelFunctionValid(typeof(KERN), T)
    numer = zero(T)
    denom = zero(T)
    @inbounds for i in 1:input.Npart
        hb = input.h[i]
        rb = (input.x[i], input.y[i])
        radius = support_radius(strategy, ha, hb, Kvalid)
        d2 = (reference_point[1] - rb[1])^2 + (reference_point[2] - rb[2])^2
        if within_radius(d2, radius)
            if strategy === itpSymmetric
                Wah = LOSint_Smoothed_kernel_function(typeof(KERN), reference_point, rb, ha)
                Wbh = LOSint_Smoothed_kernel_function(typeof(KERN), reference_point, rb, hb)
                W = T(0.5) * (Wah + Wbh)
            else
                hsel = strategy === itpGather ? ha : hb
                W = LOSint_Smoothed_kernel_function(typeof(KERN), reference_point, rb, hsel)
            end
            weight = input.m[i] * W / input.ρ[i]
            numer += input.quant[column][i] * weight
            denom += weight
        end
    end
    return iszero(denom) ? T(NaN) : numer / denom
end

# ----------------------
# Random input helpers
# ----------------------

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
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 3}(n, KERN, x, y, z, m, h, ρ, (q1, q2, q3))
    LBVH = LinearBVH!(input, Val(3))
    return input, LBVH
end

function random_input_LOS(rng::AbstractRNG, n::Int)
    x = rand(rng, n)
    y = rand(rng, n)
    z = zeros(Float64, n)
    m = rand(rng, n) .+ 0.05
    h = rand(rng, n) .* 0.08 .+ 0.02
    ρ = rand(rng, n) .+ 0.2
    q1 = rand(rng, n)
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 1}(n, KERN, x, y, z, m, h, ρ, (q1,))
    LBVH = LinearBVH!(input, Val(2))
    return input, LBVH
end

function make_empty_input()
    # Use a single particle placed far away so every test reference sees zero neighbors
    x = [10.0]
    y = [10.0]
    z = [10.0]
    m = [1.0]
    h = [0.01]
    ρ = [1.0]
    q1 = [0.0]
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 1}(1, KERN, x, y, z, m, h, ρ, (q1,))
    return input, LinearBVH!(input, Val(3))
end
