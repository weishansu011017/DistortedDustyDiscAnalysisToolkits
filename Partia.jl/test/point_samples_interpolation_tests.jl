using Test
using Random
using Partia

@static if !isdefined(@__MODULE__, :make_grid_interpolation_fixture)
    include("grid_interpolation_test_common.jl")
end

@testset "PointSamples interpolation -- CPU consistency" begin
    input, catalog, LBVH = make_grid_interpolation_fixture()
    grid_template = make_point_samples_template()
    result = PointSamples_interpolation(CPUComputeBackend(), grid_template, input, catalog, itpSymmetric)

    scalar_slot = ki_mod.scalar_index(catalog, :temp)
    div_slots = ki_mod.div_slots(catalog, :v)

    @test result.names == catalog.ordered_names
    @test length(result.grids) == length(catalog.ordered_names)
    @test result.grids[1].coor === result.grids[2].coor
    @test result.grids[1].coor == grid_template.coor

    for i in eachindex(grid_template.grid)
        point = (
            grid_template.coor[1][i],
            grid_template.coor[2][i],
            grid_template.coor[3][i],
        )
        ha = brute_nearest_h(input, point)
        expected_scalar = brute_quantity(input, point, ha, scalar_slot, itpSymmetric)
        expected_div = brute_divergence(input, point, ha, div_slots, itpSymmetric)

        @test isapprox(result.grids[1].grid[i], expected_scalar; atol = 1.0e-12, rtol = 1.0e-10)
        @test isapprox(result.grids[2].grid[i], expected_div; atol = 1.0e-12, rtol = 1.0e-10)
    end
end

@testset "LineSamples interpolation -- CPU scatter consistency" begin
    input, catalog, _ = make_line_interpolation_fixture()
    grid_template = make_line_samples_template()
    result = LineSamples_interpolation(CPUComputeBackend(), grid_template, input, catalog, itpScatter)

    scalar_slots = catalog.scalar_slots

    @test result.names == catalog.ordered_names
    @test length(result.grids) == length(catalog.ordered_names)
    @test result.grids[1].origin === result.grids[2].origin
    @test result.grids[1].direction === result.grids[2].direction
    @test result.grids[1].origin == grid_template.origin
    @test result.grids[1].direction == grid_template.direction

    for i in eachindex(grid_template.grid)
        origin = (
            grid_template.origin[1][i],
            grid_template.origin[2][i],
            grid_template.origin[3][i],
        )
        direction = (
            grid_template.direction[1][i],
            grid_template.direction[2][i],
            grid_template.direction[3][i],
        )

        expected = ntuple(
            j -> brute_line_integrated_quantity(input, origin, direction, 0.0, scalar_slots[j], itpScatter),
            length(scalar_slots),
        )

        @test isapprox(result.grids[1].grid[i], expected[1]; atol = 1.0e-12, rtol = 1.0e-10)
        @test isapprox(result.grids[2].grid[i], expected[2]; atol = 1.0e-12, rtol = 1.0e-10)
    end
end

@testset "LineSamples interpolation -- unsupported modes" begin
    line_input, line_catalog, _ = make_line_interpolation_fixture()
    line_template = make_line_samples_template()

    @test_throws ArgumentError LineSamples_interpolation(CPUComputeBackend(), line_template, line_input, line_catalog, itpGather)
    @test_throws ArgumentError LineSamples_interpolation(CPUComputeBackend(), line_template, line_input, line_catalog, itpSymmetric)

    point_input, point_catalog, _ = make_grid_interpolation_fixture()
    @test_throws MethodError LineSamples_interpolation(CPUComputeBackend(), line_template, point_input, point_catalog, itpScatter)
end

@testset "PointSamples interpolation -- analytic linear-field regression" begin
    input, catalog, h = make_uniform_cloud_3d(12; eta = 1.2, variable_h = true)
    grid_template = make_analytic_point_samples()

    for strategy in (itpGather, itpScatter, itpSymmetric)
        result = PointSamples_interpolation(CPUComputeBackend(), grid_template, input, catalog, strategy)

        for i in eachindex(grid_template.grid)
            point = (
                grid_template.coor[1][i],
                grid_template.coor[2][i],
                grid_template.coor[3][i],
            )
            s_ref = analytic_scalar(point...)
            g_ref = analytic_grad_scalar(point...)
            div_ref = analytic_divA(point...)
            curl_ref = analytic_curlA(point...)

            @test isapprox(result.grids[1].grid[i], s_ref; atol = 2e-2, rtol = 1e-2)
            @test isapprox(result.grids[2].grid[i], g_ref[1]; atol = 6e-2, rtol = 1e-2)
            @test isapprox(result.grids[3].grid[i], g_ref[2]; atol = 6e-2, rtol = 1e-2)
            @test isapprox(result.grids[4].grid[i], g_ref[3]; atol = 6e-2, rtol = 1e-2)
            @test isapprox(result.grids[5].grid[i], div_ref; atol = 1e-1, rtol = 2e-2)
            @test isapprox(result.grids[6].grid[i], curl_ref[1]; atol = 5e-2, rtol = 1e-2)
            @test isapprox(result.grids[7].grid[i], curl_ref[2]; atol = 5e-2, rtol = 1e-2)
            @test isapprox(result.grids[8].grid[i], curl_ref[3]; atol = 5e-2, rtol = 1e-2)
        end
    end
end
