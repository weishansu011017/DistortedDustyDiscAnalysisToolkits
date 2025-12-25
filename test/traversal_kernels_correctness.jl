using Test
using Random
using PhantomRevealer

@static if !isdefined(@__MODULE__, :support_radius)
    include("traversal_test_common.jl")
end

approx_with_nan(a, b; atol, rtol) = isequal(a, b) || isapprox(a, b; atol=atol, rtol=rtol)
approx_with_nan(a::Tuple, b::Tuple; atol, rtol) = all(approx_with_nan(ai, bi; atol=atol, rtol=rtol) for (ai, bi) in zip(a, b))

@testset "Traversal kernels match brute force" begin
    rng = MersenneTwister(0xC0FFEE)

    # Setup → Data
    input, LBVH = random_input_3d(rng, 80)
    reference_points = ((0.2, 0.3, 0.4), (0.7, 0.2, 0.1))
    ha_values = (0.05, 0.12)

    # Traversal → Baseline → Assertions
    for ref in reference_points, ha in ha_values, strategy in (itpGather, itpScatter, itpSymmetric)
        @test density(input, ref, ha, LBVH, strategy) ≈ brute_density(input, ref, ha, strategy) atol=1e-10 rtol=1e-8
        @test approx_with_nan(quantity_interpolate(input, ref, ha, LBVH, 1, true, strategy), brute_quantity(input, ref, ha, 1, strategy); atol=1e-10, rtol=1e-8)
        @test approx_with_nan(gradient_density(input, ref, ha, LBVH, strategy), brute_gradient_density(input, ref, ha, strategy); atol=5e-9, rtol=1e-7)
        @test approx_with_nan(gradient_quantity_interpolate(input, ref, ha, LBVH, 2, strategy), brute_gradient_quantity(input, ref, ha, 2, strategy); atol=5e-9, rtol=1e-7)
        @test approx_with_nan(divergence_quantity_interpolate(input, ref, ha, LBVH, 1, 2, 3, strategy), brute_divergence(input, ref, ha, (1, 2, 3), strategy); atol=5e-9, rtol=1e-7)
        @test approx_with_nan(curl_quantity_interpolate(input, ref, ha, LBVH, 1, 2, 3, strategy), brute_curl(input, ref, ha, (1, 2, 3), strategy); atol=5e-9, rtol=1e-7)
    end
end

@testset "LOS kernels match brute force" begin
    rng = MersenneTwister(0x1EE7)
    input, LBVH = random_input_LOS(rng, 60)
    reference_points = ((0.1, 0.9), (0.6, 0.4))
    ha_values = (0.04, 0.09)
    for ref in reference_points, ha in ha_values, strategy in (itpGather, itpScatter, itpSymmetric)
        @test LOS_density(input, ref, ha, LBVH, strategy) ≈ brute_LOS_density(input, ref, ha, strategy) atol=1e-10 rtol=1e-8
        @test approx_with_nan(LOS_quantities_interpolate(input, ref, ha, LBVH, (1,), (true,), strategy)[1], brute_LOS_quantity(input, ref, ha, 1, strategy); atol=1e-10, rtol=1e-8)
    end
end

@testset "Empty and no-neighbor cases" begin
    empty_input, empty_bvh = make_empty_input()
    ref3d = (0.5, 0.5, 0.5)
    ha = 0.05
    for strategy in (itpGather, itpScatter, itpSymmetric)
        @test density(empty_input, ref3d, ha, empty_bvh, strategy) == 0.0
        @test isnan(quantity_interpolate(empty_input, ref3d, ha, empty_bvh, 1, true, strategy))
        @test all(isnan, gradient_density(empty_input, ref3d, ha, empty_bvh, strategy))
        @test all(isnan, gradient_quantity_interpolate(empty_input, ref3d, ha, empty_bvh, 1, strategy))
        @test isnan(divergence_quantity_interpolate(empty_input, ref3d, ha, empty_bvh, 1, 1, 1, strategy))
        @test all(isnan, curl_quantity_interpolate(empty_input, ref3d, ha, empty_bvh, 1, 1, 1, strategy))
    end

    rng = MersenneTwister(0xDEAD)
    input, LBVH = random_input_3d(rng, 20)
    far_point = (2.0, 2.0, 2.0)
    tiny_ha = 1e-4
    for strategy in (itpGather, itpScatter, itpSymmetric)
        @test density(input, far_point, tiny_ha, LBVH, strategy) == 0.0
        @test isnan(quantity_interpolate(input, far_point, tiny_ha, LBVH, 1, true, strategy))
        @test all(isnan, gradient_density(input, far_point, tiny_ha, LBVH, strategy))
        @test all(isnan, gradient_quantity_interpolate(input, far_point, tiny_ha, LBVH, 1, strategy))
        @test isnan(divergence_quantity_interpolate(input, far_point, tiny_ha, LBVH, 1, 2, 3, strategy))
        @test all(isnan, curl_quantity_interpolate(input, far_point, tiny_ha, LBVH, 1, 2, 3, strategy))
    end
end
