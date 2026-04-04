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

@static if !isdefined(@__MODULE__, :analytic_scalar)
    include("interpolation_analytic_test_common.jl")
end

kern = M4_spline()
strategies = (itpGather, itpScatter, itpSymmetric)
kvalid = KernelFunctionValid(typeof(kern), Float64)

@testset "Analytic traversal kernels (linear field)" begin
    rng = Xoshiro(0xA11CE)
    configs = ((nx = 8, eta = 1.2), (nx = 12, eta = 1.2))

    for cfg in configs
        input, _, h = make_uniform_cloud_3d(cfg.nx; eta = cfg.eta, kernel = typeof(kern), variable_h = false)
        LBVH = LinearBVH!(input, Val(3))
        refs = sample_reference_points(rng, 12, h; kernel = typeof(kern))

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
