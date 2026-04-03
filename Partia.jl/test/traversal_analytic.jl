# ──────────────────────────────────────────────────────────────────────────── #
#  Test: Traversal Kernels - Analytic Linear-Field Regression
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ────────────────────
#  Verifies the public traversal interpolation routines against manufactured
#  linear fields on a uniform particle cloud.
#
#  1. Scalar interpolation
#  2. Gradient interpolation
#  3. Divergence interpolation
#  4. Curl interpolation
#  5. Density consistency
#  6. Vanishing density-gradient error for a uniform-density cloud
#
#  The goal here is not brute-force equality, but regression-level accuracy
#  against analytic expectations for smooth fields sampled away from boundaries.
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using Random
using Partia
using Partia.KernelInterpolation: _density_kernel,
    _gradient_density_kernel,
    _quantity_interpolate_kernel,
    _gradient_quantity_interpolate_kernel,
    _divergence_quantity_interpolate_kernel,
    _curl_quantity_interpolate_kernel

kern = M4_spline()
strategies = (itpGather, itpScatter, itpSymmetric)
kvalid = KernelFunctionValid(typeof(kern), Float64)

# Analytic manufactured solutions (linear fields)
@inline analytic_scalar(x, y, z) = x + y + z
@inline analytic_grad_scalar(::Float64, ::Float64, ::Float64) = (1.0, 1.0, 1.0)
@inline analytic_vecA(x, y, z) = (x, y, z)
@inline analytic_divA(::Float64, ::Float64, ::Float64) = 3.0
@inline analytic_curlA(::Float64, ::Float64, ::Float64) = (0.0, 0.0, 0.0)

# Structured particle generation
function make_uniform_cloud_3d(nx::Int; eta::Float64)
    dx = 1.0 / nx
    coords = collect(range(dx / 2, stop = 1.0 - dx / 2, step = dx))
    x = Float64[]
    y = Float64[]
    z = Float64[]
    @inbounds for xi in coords, yi in coords, zi in coords
        push!(x, xi)
        push!(y, yi)
        push!(z, zi)
    end

    n = length(x)
    m = fill(dx^3, n)
    h = fill(eta * dx, n)
    q1 = similar(x)
    q2 = similar(x)
    q3 = similar(x)
    q4 = similar(x)
    @inbounds for i in 1:n
        q1[i] = analytic_scalar(x[i], y[i], z[i])
        q2[i], q3[i], q4[i] = analytic_vecA(x[i], y[i], z[i])
    end

    rho = ones(Float64, n)
    input = InterpolationInput((x, y, z), m, h, rho, (q1, q2, q3, q4); smoothed_kernel = typeof(kern))
    return input, dx, h[1]
end

sample_reference_points(rng::AbstractRNG, n::Int, h::Float64) = begin
    margin = 1.5 * kvalid * h
    lo = margin
    hi = 1.0 - margin
    refs = NTuple{3, Float64}[]
    @inbounds for _ in 1:n
        x = rand(rng) * (hi - lo) + lo
        y = rand(rng) * (hi - lo) + lo
        z = rand(rng) * (hi - lo) + lo
        push!(refs, (x, y, z))
    end
    refs
end

mean_abs(v) = isempty(v) ? 0.0 : sum(abs, v) / length(v)

@testset "Analytic traversal kernels (linear field)" begin
    rng = Xoshiro(0xA11CE)
    configs = ((nx = 8, eta = 1.2), (nx = 12, eta = 1.2))

    for cfg in configs
        input, _, h = make_uniform_cloud_3d(cfg.nx; eta = cfg.eta)
        LBVH = LinearBVH!(input, Val(3))
        refs = sample_reference_points(rng, 12, h)

        scalar_err = Float64[]
        grad_err = Float64[]
        div_err = Float64[]
        curl_err = Float64[]
        rho_err = Float64[]
        grad_rho_err = Float64[]

        for strategy in strategies, ref in refs
            s_ref = analytic_scalar(ref...)
            g_ref = analytic_grad_scalar(ref...)
            div_ref = analytic_divA(ref...)
            curl_ref = analytic_curlA(ref...)

            if strategy === itpScatter
                s_val = _quantity_interpolate_kernel(input, ref, LBVH, 1, true, strategy)
                g_val = _gradient_quantity_interpolate_kernel(input, ref, LBVH, 1, strategy)
                div_val = _divergence_quantity_interpolate_kernel(input, ref, LBVH, 2, 3, 4, strategy)
                curl_val = _curl_quantity_interpolate_kernel(input, ref, LBVH, 2, 3, 4, strategy)
                rho_val = _density_kernel(input, ref, LBVH, strategy)
                grad_rho_val = _gradient_density_kernel(input, ref, LBVH, strategy)
            else
                s_val = _quantity_interpolate_kernel(input, ref, h, LBVH, 1, true, strategy)
                g_val = _gradient_quantity_interpolate_kernel(input, ref, h, LBVH, 1, strategy)
                div_val = _divergence_quantity_interpolate_kernel(input, ref, h, LBVH, 2, 3, 4, strategy)
                curl_val = _curl_quantity_interpolate_kernel(input, ref, h, LBVH, 2, 3, 4, strategy)
                rho_val = _density_kernel(input, ref, h, LBVH, strategy)
                grad_rho_val = _gradient_density_kernel(input, ref, h, LBVH, strategy)
            end

            push!(scalar_err, abs(s_val - s_ref))
            push!(grad_err, sqrt(sum((g_val .- g_ref) .^ 2)))
            push!(div_err, abs(div_val - div_ref))
            push!(curl_err, sqrt(sum((curl_val .- curl_ref) .^ 2)))
            push!(rho_err, abs(rho_val - 1.0))
            push!(grad_rho_err, sqrt(sum(grad_rho_val .^ 2)))
        end

        @test mean_abs(scalar_err) <= 2e-2
        @test mean_abs(grad_err) <= 6e-2
        @test mean_abs(div_err) <= 1.0e-1
        @test mean_abs(curl_err) <= 5e-2
        @test mean_abs(rho_err) <= 5e-2
        @test mean_abs(grad_rho_err) <= 5e-2
    end
end
