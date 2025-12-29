using Test
using DataFrames

using PhantomRevealer

const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const MassFromColumn = PhantomRevealer.MassFromColumn
const MassFromParams = PhantomRevealer.MassFromParams
const NS = PhantomRevealer.NeighborSearch

@testset "InterpolationInput mass-from-column constructor" begin
    df = DataFrame(
        x = Float32[0.0, 1.0, 2.0],
        y = Float32[1.0, 2.0, 3.0],
        z = Float32[2.0, 3.0, 4.0],
        h = Float32[0.2, 0.25, 0.3],
        rho = Float64[1.0, 1.1, 0.9],
        mass = Float32[0.5, 0.6, 0.7],
        P = Float32[10.0, 11.0, 12.0],
        vx = Float32[0.1, 0.0, -0.1],
        vy = Float32[0.0, 0.1, 0.0],
        vz = Float32[-0.1, 0.0, 0.1],
        Bx = Float32[1.0, 1.1, 1.2],
        By = Float32[1.2, 1.3, 1.4],
        Bz = Float32[1.4, 1.5, 1.6],
    )
    params = Dict{Symbol, Any}()
    prdf = IO.ParticleDataFrame(df, params)
    mass_source = MassFromColumn(:mass)

    input, catalog = build_input(
        prdf,
        mass_source;
        scalars = [:P],
        gradients = [:P],
        divergences = [:v],
        curls = [:B]
    )

    @test input.Npart == 3
    @test eltype(input.x) === Float64
    @test length(input.quant) == 7

    @test KI.scalar_index(catalog, :P) == 1
    @test KI.grad_slot(catalog, :P) == 1
    @test KI.div_slots(catalog, :v) == (2, 3, 4)
    @test KI.curl_slots(catalog, :B) == (5, 6, 7)
    @test catalog.ordered_names == (
        :P,
        Symbol("∇", "P", "ˣ"),
        Symbol("∇", "P", "ʸ"),
        Symbol("∇", "P", "ᶻ"),
        Symbol("∇⋅", "v"),
        Symbol("∇×", "B", "ˣ"),
        Symbol("∇×", "B", "ʸ"),
        Symbol("∇×", "B", "ᶻ"),
    )

    @test all(input.quant[1] .== Float64.(df.P))
    @test all(input.quant[2] .== Float64.(df.vx))
    @test all(input.quant[7] .== Float64.(df.Bz))

    df_missing = select(df, Not(:Bz))
    prdf_missing = IO.ParticleDataFrame(df_missing, params)
    @test_throws ArgumentError build_input(
        prdf_missing,
        mass_source;
        scalars = Symbol[],
        curls = [:B]
    )
end

@testset "InterpolationInput mass-from-params constructor" begin
    df = DataFrame(
        x = Float32[0.0, 1.0, 2.0],
        y = Float32[1.0, 2.0, 3.0],
        z = Float32[2.0, 3.0, 4.0],
        h = Float32[0.2, 0.25, 0.3],
        rho = Float64[1.0, 1.1, 0.9],
        P = Float32[10.0, 11.0, 12.0],
        vx = Float32[0.1, 0.0, -0.1],
        vy = Float32[0.0, 0.1, 0.0],
        vz = Float32[-0.1, 0.0, 0.1],
        Bx = Float32[1.0, 1.1, 1.2],
        By = Float32[1.2, 1.3, 1.4],
        Bz = Float32[1.4, 1.5, 1.6],
    )
    params = Dict{Symbol, Any}(:mass => 0.42f0)
    prdf = IO.ParticleDataFrame(df, params)
    mass_source = MassFromParams(:mass)

    input, catalog = build_input(
        prdf,
        mass_source;
        scalars = [:P],
        gradients = [:P],
        divergences = [:v],
        curls = [:B]
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
    @test catalog.ordered_names == (
        :P,
        Symbol("∇", "P", "ˣ"),
        Symbol("∇", "P", "ʸ"),
        Symbol("∇", "P", "ᶻ"),
        Symbol("∇⋅", "v"),
        Symbol("∇×", "B", "ˣ"),
        Symbol("∇×", "B", "ʸ"),
        Symbol("∇×", "B", "ᶻ"),
    )

    df_missing = select(df, Not(:h))
    prdf_missing = IO.ParticleDataFrame(df_missing, params)
    @test_throws ArgumentError build_input(
        prdf_missing,
        mass_source;
        scalars = Symbol[]
    )
end
