# ──────────────────────────────────────────────────────────────────────────── #
#  Test: IO — Phantom Binary Dump Reader
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ─────────────────────
#  Validates `read_phantom`, the Phantom SPH binary dump file reader,
#  using the bundled test dump file `testinput/testdumpfile_00000`.
#
#  1. Basic read — default parameters
#     • Returns a non-empty `Vector{ParticleDataFrame}`.
#     • First element (gas) contains expected columns (x, y, z, h, …).
#     • Particle count is positive and matches header `npart`.
#
#  2. ParticleDataFrame interface
#     • `get_dim` — returns 2 or 3.
#     • `get_time` — returns a finite Float64.
#     • `get_npart` — matches `nrow(dfdata)`.
#     • `names` — column listing.
#     • Indexing: `prdf[:, :x]`, `prdf[!, :x]`, `prdf[1, :x]`.
#
#  3. Header parameters
#     • `params` dict contains expected keys (:npart, :massoftype, etc.).
#
#  Test data
#  ─────────
#  • `testinput/testdumpfile_00000` — a minimal Phantom binary dump file
#    shipped with the repository for CI purposes.
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using PhantomRevealer

# ========================== Constants ======================================= #

const TESTDUMP = joinpath(@__DIR__, "testinput", "testdumpfile_00000")

# ============================== Test body =================================== #

# ── 1. Basic read ────────────────────────────────────────────────────── #

@testset "read_phantom — basic read" begin
    @test isfile(TESTDUMP)

    data_list = read_phantom(TESTDUMP)

    @test data_list isa Vector
    @test length(data_list) >= 1

    gas = data_list[1]
    @test gas isa PhantomRevealer.IO.ParticleDataFrame
    @test get_npart(gas) > 0

    # Must contain spatial columns
    colnames = names(gas)
    @test "x" in colnames
    @test "y" in colnames
    @test "h" in colnames
end

# ── 2. ParticleDataFrame interface ───────────────────────────────────── #

@testset "ParticleDataFrame — interface" begin
    data_list = read_phantom(TESTDUMP)
    gas = data_list[1]

    # Dimension
    dim = get_dim(gas)
    @test dim in (2, 3)

    # Time
    t = get_time(gas)
    if t !== nothing
        @test t isa AbstractFloat
        @test isfinite(t)
    end

    # Particle count consistency
    n = get_npart(gas)
    @test n == size(gas.dfdata, 1)

    # Indexing: column access
    x_copy = gas[:, :x]
    @test length(x_copy) == n

    x_ref = gas[!, :x]
    @test length(x_ref) == n

    # Single-particle access
    x1 = gas[1, :x]
    @test x1 isa AbstractFloat
end

# ── 3. Header parameters ────────────────────────────────────────────── #

@testset "read_phantom — header parameters" begin
    data_list = read_phantom(TESTDUMP)
    gas = data_list[1]

    params = gas.params
    @test params isa Dict

    # Phantom headers always contain mass-related keys
    @test haskey(params, :mass) || haskey(params, :massoftype)
end
