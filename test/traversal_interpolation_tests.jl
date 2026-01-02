using Test
using Random
using PhantomRevealer
using PhantomRevealer.KernelInterpolation: _density_kernel, _number_density_kernel, _quantity_interpolate_kernel, _divergence_quantity_interpolate_kernel, _curl_quantity_interpolate_kernel, _LOS_density_kernel

@static if !isdefined(@__MODULE__, :support_radius)
    include("traversal_test_common.jl")
end

@testset "Traversal interpolation matches brute force" begin
    rng = MersenneTwister(0xBADA55)

    # Setup → Data
    input, LBVH = random_input_3d(rng, 200)
    reference_point = (0.4, 0.35, 0.25)
    ha = 0.12

    # Traversal → Baseline → Assertions
    for strategy in (itpGather, itpScatter, itpSymmetric)
        dens = strategy === itpScatter ? _density_kernel(input, reference_point, LBVH, strategy) : _density_kernel(input, reference_point, ha, LBVH, strategy)
        n_dens = strategy === itpScatter ? _number_density_kernel(input, reference_point, LBVH, strategy) : _number_density_kernel(input, reference_point, ha, LBVH, strategy)
        qty = strategy === itpScatter ? _quantity_interpolate_kernel(input, reference_point, LBVH, 1, true, strategy) : _quantity_interpolate_kernel(input, reference_point, ha, LBVH, 1, true, strategy)

        @test dens ≈ brute_density(input, reference_point, ha, strategy) atol=1e-10 rtol=1e-8
        @test n_dens ≈ brute_number_density(input, reference_point, ha, strategy) atol=1e-10 rtol=1e-8
        @test qty ≈ brute_quantity(input, reference_point, ha, 1, strategy) atol=1e-10 rtol=1e-8
    end
end

@testset "Divergence and curl vanish for uniform field" begin
    # Setup → Data
    n = 4
    x = [0.0, 0.05, 0.11, -0.08]
    y = [0.02, -0.03, 0.04, 0.01]
    z = [0.0, 0.01, -0.02, 0.03]
    m = fill(1.0, n)
    h = fill(0.12, n)
    ρ = fill(1.0, n)
    vx = fill(1.0, n)
    vy = fill(-2.0, n)
    vz = fill(0.5, n)
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 3}(n, KERN, x, y, z, m, h, ρ, (vx, vy, vz))

    # Build LBVH
    LBVH = LinearBVH!(input, Val(3))

    # Traversal → Assertions
    reference_point = (x[1], y[1], z[1])
    ha = h[1]
    for strategy in (itpGather, itpScatter, itpSymmetric)
        divv = strategy === itpScatter ? _divergence_quantity_interpolate_kernel(input, reference_point, LBVH, 1, 2, 3, strategy) : _divergence_quantity_interpolate_kernel(input, reference_point, ha, LBVH, 1, 2, 3, strategy)
        curlv = strategy === itpScatter ? _curl_quantity_interpolate_kernel(input, reference_point, LBVH, 1, 2, 3, strategy) : _curl_quantity_interpolate_kernel(input, reference_point, ha, LBVH, 1, 2, 3, strategy)
        @test divv ≈ 0.0 atol = 1e-12 rtol = 1e-10
        @test curlv[1] ≈ 0.0 atol = 1e-12 rtol = 1e-10
        @test curlv[2] ≈ 0.0 atol = 1e-12 rtol = 1e-10
        @test curlv[3] ≈ 0.0 atol = 1e-12 rtol = 1e-10
    end
end

@testset "LOS traversal matches brute force" begin
    rng = MersenneTwister(0xF00D)

    # Setup → Data
    input, LBVH = random_input_LOS(rng, 150)
    reference_point = (0.2, 0.8)
    ha = 0.08

    # Traversal → Baseline → Assertions
    for strategy in (itpGather, itpScatter, itpSymmetric)
        ρ_LOS = strategy === itpScatter ? _LOS_density_kernel(input, reference_point, LBVH, strategy) : _LOS_density_kernel(input, reference_point, ha, LBVH, strategy)
        @test ρ_LOS ≈ brute_LOS_density(input, reference_point, ha, strategy) atol=1e-10 rtol=1e-8
    end
end