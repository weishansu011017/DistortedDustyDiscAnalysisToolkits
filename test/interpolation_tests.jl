# ──────────────────────────────────────────────────────────────────────────── #
#  Test: SPH Interpolation — Constructors, Traversal, and Physical Checks
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ─────────────────────
#  End-to-end validation of the KernelInterpolation pipeline:
#
#  1. InterpolationInput constructors
#     • `MassFromColumn` — reads per-particle mass from DataFrame column.
#     • `MassFromParams` — reads scalar mass from parameter dict.
#     • Verifies particle count, element-type promotion, quantity catalog
#       (scalar / gradient / divergence / curl slot indices), data integrity,
#       and error paths for missing columns.
#
#  2. BVH traversal interpolation — brute-force comparison
#     • Density, number density, quantity, gradient, divergence, and curl
#       interpolation (3D) against O(N²) brute-force baselines for all three
#       strategies (Gather, Scatter, Symmetric).
#     • Line-integrated column-density and quantity interpolation.
#
#  3. Physical sanity checks
#     • Divergence and curl of a uniform vector field must vanish to
#       machine epsilon.
#
#  Brute-force reference implementations live in the helper file
#  `interpolation_test_common.jl`, which is included by this file.
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using Random
using DataFrames
using PhantomRevealer
using PhantomRevealer.KernelInterpolation: _density_kernel, _number_density_kernel,
    _gradient_density_kernel,
    _gradient_quantity_interpolate_kernel,
    _quantity_interpolate_kernel,
    _divergence_quantity_interpolate_kernel,
    _curl_quantity_interpolate_kernel,
    _line_integrated_density_kernel,
    _line_integrated_quantities_interpolate_kernel

# ========================== Module aliases ================================== #

io_mod = PhantomRevealer.IO
ki_mod = PhantomRevealer.KernelInterpolation

# ========================== Common infrastructure =========================== #

@static if !isdefined(@__MODULE__, :support_radius)
    include("interpolation_test_common.jl")
end

# ── 4. Empty and no-neighbor behavior ───────────────────────────────── #

@testset "Traversal interpolation — empty and no-neighbor behavior" begin
    empty_input, empty_bvh = make_empty_input()
    ref3d = (0.5, 0.5, 0.5)
    ha = 0.05

    for strategy in (itpGather, itpScatter, itpSymmetric)
        dens = strategy === itpScatter ?
            _density_kernel(empty_input, ref3d, empty_bvh, strategy) :
            _density_kernel(empty_input, ref3d, ha, empty_bvh, strategy)
        grad_dens = strategy === itpScatter ?
            _gradient_density_kernel(empty_input, ref3d, empty_bvh, strategy) :
            _gradient_density_kernel(empty_input, ref3d, ha, empty_bvh, strategy)
        qty = strategy === itpScatter ?
            _quantity_interpolate_kernel(empty_input, ref3d, empty_bvh, 1, true, strategy) :
            _quantity_interpolate_kernel(empty_input, ref3d, ha, empty_bvh, 1, true, strategy)
        grad_qty = strategy === itpScatter ?
            _gradient_quantity_interpolate_kernel(empty_input, ref3d, empty_bvh, 1, strategy) :
            _gradient_quantity_interpolate_kernel(empty_input, ref3d, ha, empty_bvh, 1, strategy)
        divv = strategy === itpScatter ?
            _divergence_quantity_interpolate_kernel(empty_input, ref3d, empty_bvh, 1, 1, 1, strategy) :
            _divergence_quantity_interpolate_kernel(empty_input, ref3d, ha, empty_bvh, 1, 1, 1, strategy)
        curlv = strategy === itpScatter ?
            _curl_quantity_interpolate_kernel(empty_input, ref3d, empty_bvh, 1, 1, 1, strategy) :
            _curl_quantity_interpolate_kernel(empty_input, ref3d, ha, empty_bvh, 1, 1, 1, strategy)

        @test dens == 0.0
        @test all(isnan, grad_dens)
        @test isnan(qty)
        @test all(isnan, grad_qty)
        @test isnan(divv)
        @test all(isnan, curlv)
    end

    rng = Xoshiro(0xDEAD)
    input, LBVH = random_input_3d(rng, 20)
    far_point = (2.0, 2.0, 2.0)
    tiny_ha = 1.0e-4

    for strategy in (itpGather, itpScatter, itpSymmetric)
        dens = strategy === itpScatter ?
            _density_kernel(input, far_point, LBVH, strategy) :
            _density_kernel(input, far_point, tiny_ha, LBVH, strategy)
        grad_dens = strategy === itpScatter ?
            _gradient_density_kernel(input, far_point, LBVH, strategy) :
            _gradient_density_kernel(input, far_point, tiny_ha, LBVH, strategy)
        qty = strategy === itpScatter ?
            _quantity_interpolate_kernel(input, far_point, LBVH, 1, true, strategy) :
            _quantity_interpolate_kernel(input, far_point, tiny_ha, LBVH, 1, true, strategy)
        grad_qty = strategy === itpScatter ?
            _gradient_quantity_interpolate_kernel(input, far_point, LBVH, 1, strategy) :
            _gradient_quantity_interpolate_kernel(input, far_point, tiny_ha, LBVH, 1, strategy)
        divv = strategy === itpScatter ?
            _divergence_quantity_interpolate_kernel(input, far_point, LBVH, 1, 2, 3, strategy) :
            _divergence_quantity_interpolate_kernel(input, far_point, tiny_ha, LBVH, 1, 2, 3, strategy)
        curlv = strategy === itpScatter ?
            _curl_quantity_interpolate_kernel(input, far_point, LBVH, 1, 2, 3, strategy) :
            _curl_quantity_interpolate_kernel(input, far_point, tiny_ha, LBVH, 1, 2, 3, strategy)

        @test dens == 0.0
        @test all(isnan, grad_dens)
        @test isnan(qty)
        @test all(isnan, grad_qty)
        @test isnan(divv)
        @test all(isnan, curlv)
    end
end

# ============================== Test body =================================== #

# ── 1a. InterpolationInput — MassFromColumn constructor ──────────────── #

@testset "InterpolationInput — MassFromColumn constructor" begin
    df = DataFrame(
        x    = Float32[0.0, 1.0, 2.0],
        y    = Float32[1.0, 2.0, 3.0],
        z    = Float32[2.0, 3.0, 4.0],
        h    = Float32[0.2, 0.25, 0.3],
        rho  = Float64[1.0, 1.1, 0.9],
        mass = Float32[0.5, 0.6, 0.7],
        P    = Float32[10.0, 11.0, 12.0],
        vx   = Float32[0.1, 0.0, -0.1],
        vy   = Float32[0.0, 0.1, 0.0],
        vz   = Float32[-0.1, 0.0, 0.1],
        Bx   = Float32[1.0, 1.1, 1.2],
        By   = Float32[1.2, 1.3, 1.4],
        Bz   = Float32[1.4, 1.5, 1.6],
    )
    params = Dict{Symbol,Any}()
    prdf = io_mod.ParticleDataFrame(df, params)
    mass_source = MassFromColumn(:mass)

    input, catalog = build_input(
        prdf, mass_source;
        scalars     = [:P],
        gradients   = [:P],
        divergences = [:v],
        curls       = [:B],
    )

    # Particle count and type promotion
    @test input.Npart == 3
    @test eltype(input.x) === Float64
    @test length(input.quant) == 7

    # Catalog slot indices
    @test ki_mod.scalar_index(catalog, :P) == 1
    @test ki_mod.grad_slot(catalog, :P) == 1
    @test ki_mod.div_slots(catalog, :v) == (2, 3, 4)
    @test ki_mod.curl_slots(catalog, :B) == (5, 6, 7)

    # Ordered names
    @test catalog.ordered_names == (
        :P,
        Symbol("∇", "P", "ˣ"),  Symbol("∇", "P", "ʸ"),  Symbol("∇", "P", "ᶻ"),
        Symbol("∇⋅", "v"),
        Symbol("∇×", "B", "ˣ"), Symbol("∇×", "B", "ʸ"), Symbol("∇×", "B", "ᶻ"),
    )

    # Data integrity
    @test all(input.quant[1] .== Float64.(df.P))
    @test all(input.quant[2] .== Float64.(df.vx))
    @test all(input.quant[7] .== Float64.(df.Bz))

    # Missing column → ArgumentError
    df_missing = select(df, Not(:Bz))
    prdf_missing = io_mod.ParticleDataFrame(df_missing, params)
    @test_throws ArgumentError build_input(
        prdf_missing, mass_source;
        scalars = Symbol[], curls = [:B],
    )
end

# ── 1b. InterpolationInput — MassFromParams constructor ──────────────── #

@testset "InterpolationInput — MassFromParams constructor" begin
    df = DataFrame(
        x   = Float32[0.0, 1.0, 2.0],
        y   = Float32[1.0, 2.0, 3.0],
        z   = Float32[2.0, 3.0, 4.0],
        h   = Float32[0.2, 0.25, 0.3],
        rho = Float64[1.0, 1.1, 0.9],
        P   = Float32[10.0, 11.0, 12.0],
        vx  = Float32[0.1, 0.0, -0.1],
        vy  = Float32[0.0, 0.1, 0.0],
        vz  = Float32[-0.1, 0.0, 0.1],
        Bx  = Float32[1.0, 1.1, 1.2],
        By  = Float32[1.2, 1.3, 1.4],
        Bz  = Float32[1.4, 1.5, 1.6],
    )
    params = Dict{Symbol,Any}(:mass => 0.42f0)
    prdf = io_mod.ParticleDataFrame(df, params)
    mass_source = MassFromParams(:mass)

    input, catalog = build_input(
        prdf, mass_source;
        scalars     = [:P],
        gradients   = [:P],
        divergences = [:v],
        curls       = [:B],
    )

    @test input.Npart == 3
    @test eltype(input.x) === Float64
    @test length(input.quant) == 7
    @test all(input.m .== fill(Float64(0.42f0), 3))
    @test !(:mass in propertynames(df))

    @test ki_mod.scalar_index(catalog, :P) == 1
    @test ki_mod.grad_slot(catalog, :P) == 1
    @test ki_mod.div_slots(catalog, :v) == (2, 3, 4)
    @test ki_mod.curl_slots(catalog, :B) == (5, 6, 7)

    # Missing required column → error
    df_missing = select(df, Not(:h))
    prdf_missing = io_mod.ParticleDataFrame(df_missing, params)
    @test_throws ArgumentError build_input(
        prdf_missing, mass_source;
        scalars = Symbol[],
    )
end

# ── 2a. Traversal — density / number density / quantity ──────────────── #

@testset "Traversal interpolation — density & quantity (3D)" begin
    rng = MersenneTwister(0xBADA55)
    input, LBVH = random_input_3d(rng, 200)
    reference_point = (0.4, 0.35, 0.25)
    ha = 0.12

    for strategy in (itpGather, itpScatter, itpSymmetric)
        dens = strategy === itpScatter ?
            _density_kernel(input, reference_point, LBVH, strategy) :
            _density_kernel(input, reference_point, ha, LBVH, strategy)
        n_dens = strategy === itpScatter ?
            _number_density_kernel(input, reference_point, LBVH, strategy) :
            _number_density_kernel(input, reference_point, ha, LBVH, strategy)
        qty = strategy === itpScatter ?
            _quantity_interpolate_kernel(input, reference_point, LBVH, 1, true, strategy) :
            _quantity_interpolate_kernel(input, reference_point, ha, LBVH, 1, true, strategy)

        @test dens   ≈ brute_density(input, reference_point, ha, strategy)         atol=1e-10 rtol=1e-8
        @test n_dens ≈ brute_number_density(input, reference_point, ha, strategy)  atol=1e-10 rtol=1e-8
        @test qty    ≈ brute_quantity(input, reference_point, ha, 1, strategy)      atol=1e-10 rtol=1e-8
    end
end

# ── 2b. Traversal — gradients and vector operators (3D) ─────────────── #

@testset "Traversal interpolation — gradients and vector operators (3D)" begin
    rng = MersenneTwister(0xC0FFEE)
    input, LBVH = random_input_3d(rng, 80)
    reference_points = ((0.2, 0.3, 0.4), (0.7, 0.2, 0.1))
    ha_values = (0.05, 0.12)

    for reference_point in reference_points, ha in ha_values, strategy in (itpGather, itpScatter, itpSymmetric)
        grad_ρ = strategy === itpScatter ?
            _gradient_density_kernel(input, reference_point, LBVH, strategy) :
            _gradient_density_kernel(input, reference_point, ha, LBVH, strategy)
        grad_A = strategy === itpScatter ?
            _gradient_quantity_interpolate_kernel(input, reference_point, LBVH, 2, strategy) :
            _gradient_quantity_interpolate_kernel(input, reference_point, ha, LBVH, 2, strategy)
        div_A = strategy === itpScatter ?
            _divergence_quantity_interpolate_kernel(input, reference_point, LBVH, 1, 2, 3, strategy) :
            _divergence_quantity_interpolate_kernel(input, reference_point, ha, LBVH, 1, 2, 3, strategy)
        curl_A = strategy === itpScatter ?
            _curl_quantity_interpolate_kernel(input, reference_point, LBVH, 1, 2, 3, strategy) :
            _curl_quantity_interpolate_kernel(input, reference_point, ha, LBVH, 1, 2, 3, strategy)

        @test approx_with_nan(grad_ρ, brute_gradient_density(input, reference_point, ha, strategy); atol = 5e-10, rtol = 1e-8)
        @test approx_with_nan(grad_A, brute_gradient_quantity(input, reference_point, ha, 2, strategy); atol = 5e-10, rtol = 1e-8)
        @test approx_with_nan(div_A, brute_divergence(input, reference_point, ha, (1, 2, 3), strategy); atol = 5e-10, rtol = 1e-8)
        @test approx_with_nan(curl_A, brute_curl(input, reference_point, ha, (1, 2, 3), strategy); atol = 5e-10, rtol = 1e-8)
    end
end

# ── 2c. Traversal — line-integrated column density and quantity ─────── #

@testset "Traversal interpolation — line-integrated column density and quantity" begin
    rng = MersenneTwister(0xF00D)
    input, LBVH = random_input_line_integrated(rng, 150)
    origin = (0.2, 0.8, 0.0)
    direction = (0.0, 0.0, 1.0)
    ha = 0.08

    for strategy in (itpGather, itpScatter, itpSymmetric)
        Sigma = strategy === itpScatter ?
            _line_integrated_density_kernel(input, origin, direction, LBVH, strategy) :
            _line_integrated_density_kernel(input, origin, direction, ha, LBVH, strategy)
        qty = strategy === itpScatter ?
            _line_integrated_quantities_interpolate_kernel(input, origin, direction, LBVH, (1,), (true,), strategy)[1] :
            _line_integrated_quantities_interpolate_kernel(input, origin, direction, ha, LBVH, (1,), (true,), strategy)[1]

        @test Sigma ≈ brute_line_integrated_density(input, origin, direction, ha, strategy) atol = 1e-10 rtol = 1e-8
        @test approx_with_nan(qty, brute_line_integrated_quantity(input, origin, direction, ha, 1, strategy); atol = 1e-10, rtol = 1e-8)
    end
end

# ── 3. Divergence & curl vanish for uniform field ────────────────────── #

@testset "Uniform field — divergence = 0, curl = 0" begin
    n = 4
    x  = [0.0, 0.05, 0.11, -0.08]
    y  = [0.02, -0.03, 0.04, 0.01]
    z  = [0.0, 0.01, -0.02, 0.03]
    m  = fill(1.0, n)
    h  = fill(0.12, n)
    ρ  = fill(1.0, n)
    vx = fill(1.0, n)
    vy = fill(-2.0, n)
    vz = fill(0.5, n)
    input = InterpolationInput{Float64, Vector{Float64}, typeof(kern), 3}(
        n, kern, x, y, z, m, h, ρ, (vx, vy, vz),
    )
    LBVH_local = LinearBVH!(input, Val(3))

    reference_point = (x[1], y[1], z[1])
    ha = h[1]

    for strategy in (itpGather, itpScatter, itpSymmetric)
        divv = strategy === itpScatter ?
            _divergence_quantity_interpolate_kernel(input, reference_point, LBVH_local, 1, 2, 3, strategy) :
            _divergence_quantity_interpolate_kernel(input, reference_point, ha, LBVH_local, 1, 2, 3, strategy)
        curlv = strategy === itpScatter ?
            _curl_quantity_interpolate_kernel(input, reference_point, LBVH_local, 1, 2, 3, strategy) :
            _curl_quantity_interpolate_kernel(input, reference_point, ha, LBVH_local, 1, 2, 3, strategy)

        @test divv     ≈ 0.0  atol=1e-12 rtol=1e-10
        @test curlv[1] ≈ 0.0  atol=1e-12 rtol=1e-10
        @test curlv[2] ≈ 0.0  atol=1e-12 rtol=1e-10
        @test curlv[3] ≈ 0.0  atol=1e-12 rtol=1e-10
    end
end
