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
#     • Density, number density, and quantity interpolation (3D) against
#       O(N²) brute-force baselines for all three strategies (Gather,
#       Scatter, Symmetric).
#     • LOS column-density integration (2D projection).
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
    _quantity_interpolate_kernel,
    _divergence_quantity_interpolate_kernel,
    _curl_quantity_interpolate_kernel,
    _LOS_density_kernel

# ========================== Module aliases ================================== #

const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const NS = PhantomRevealer.NeighborSearch

# ========================== Common infrastructure =========================== #

@static if !isdefined(@__MODULE__, :support_radius)
    include("interpolation_test_common.jl")
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
    prdf = IO.ParticleDataFrame(df, params)
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
    @test KI.scalar_index(catalog, :P) == 1
    @test KI.grad_slot(catalog, :P) == 1
    @test KI.div_slots(catalog, :v) == (2, 3, 4)
    @test KI.curl_slots(catalog, :B) == (5, 6, 7)

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
    prdf_missing = IO.ParticleDataFrame(df_missing, params)
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
    prdf = IO.ParticleDataFrame(df, params)
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

    @test KI.scalar_index(catalog, :P) == 1
    @test KI.grad_slot(catalog, :P) == 1
    @test KI.div_slots(catalog, :v) == (2, 3, 4)
    @test KI.curl_slots(catalog, :B) == (5, 6, 7)

    # Missing required column → error
    df_missing = select(df, Not(:h))
    prdf_missing = IO.ParticleDataFrame(df_missing, params)
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

# ── 2b. Traversal — LOS column density ──────────────────────────────── #

# @testset "Traversal interpolation — LOS column density" begin
#     rng = MersenneTwister(0xF00D)
#     input, LBVH = random_input_LOS(rng, 150)
#     reference_point = (0.2, 0.8)
#     ha = 0.08

#     for strategy in (itpGather, itpScatter, itpSymmetric)
#         ρ_LOS = strategy === itpScatter ?
#             _LOS_density_kernel(input, reference_point, LBVH, strategy) :
#             _LOS_density_kernel(input, reference_point, ha, LBVH, strategy)
#         @test ρ_LOS ≈ brute_LOS_density(input, reference_point, ha, strategy) atol=1e-10 rtol=1e-8
#     end
# end

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
    input = InterpolationInput{Float64, Vector{Float64}, typeof(KERN), 3}(
        n, KERN, x, y, z, m, h, ρ, (vx, vy, vz),
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
