using Test
using Random
using Partia

@static if !isdefined(@__MODULE__, :make_grid_interpolation_fixture)
    include("grid_interpolation_test_common.jl")
end

@testset "StructuredGrid transform -- flatten and restore" begin
    grid = make_structured_grid_template()
    grid.grid .= reshape(collect(1.0:length(grid.grid)), size(grid))

    flattened = Partia.Grids.flatten(Cartesian, grid)
    restored = Partia.Grids.restore_struct(Cartesian, flattened, grid.axes)

    @test restored.grid == grid.grid
    @test restored.axes == grid.axes
    @test restored.size == size(grid)
end

@testset "StructuredGrid transform -- coordinate dispatch" begin
    for (coord, grid) in (
        (Cartesian, make_structured_grid_template()),
        (Polar, make_polar_grid_template()),
        (Cylindrical, make_cylindrical_grid_template()),
        (Spherical, make_spherical_grid_template()),
    )
        grid.grid .= reshape(collect(1.0:length(grid.grid)), size(grid))
        flattened = Partia.Grids.flatten(coord, grid)
        expected = explicit_cartesian_coords(coord, grid)

        @test flattened.grid == vec(grid.grid)
        @test length(flattened.coor) == length(expected)
        @test all(isapprox.(flattened.coor[1], expected[1]; atol = 1.0e-12, rtol = 1.0e-12))
        @test all(isapprox.(flattened.coor[2], expected[2]; atol = 1.0e-12, rtol = 1.0e-12))
        if length(expected) == 3
            @test all(isapprox.(flattened.coor[3], expected[3]; atol = 1.0e-12, rtol = 1.0e-12))
        end

        restored = Partia.Grids.restore_struct(coord, flattened, grid.axes)
        @test restored.grid == grid.grid
        @test restored.axes == grid.axes
        @test restored.size == size(grid)
    end
end

@testset "StructuredGrid restore -- coordinate dispatch mismatch is rejected" begin
    cyl_grid = make_cylindrical_grid_template()
    flattened = Partia.Grids.flatten(Cylindrical, cyl_grid)

    @test_throws ArgumentError Partia.Grids.restore_struct(Cartesian, flattened, cyl_grid.axes)
end

@testset "StructuredGrid interpolation -- flattened consistency" begin
    input, catalog, _ = make_grid_interpolation_fixture()
    structured_template = make_structured_grid_template()
    flattened_template = Partia.Grids.flatten(Cartesian, structured_template)

    point_result = PointSamples_interpolation(
        CPUComputeBackend(),
        flattened_template,
        input,
        catalog,
        itpSymmetric,
    )
    structured_result = StructuredGrid_interpolation(
        CPUComputeBackend(),
        Cartesian,
        structured_template,
        input,
        catalog,
        itpSymmetric,
    )

    @test structured_result.names == point_result.names
    @test structured_result.names == catalog.ordered_names

    for i in eachindex(structured_result.grids)
        @test structured_result.grids[i].axes == structured_template.axes
        @test isapprox(vec(structured_result.grids[i].grid), point_result.grids[i].grid; atol = 1.0e-12, rtol = 1.0e-10)
    end
end

@testset "StructuredGrid interpolation -- flattened consistency by coordinate system" begin
    input, catalog, _ = make_grid_interpolation_fixture()

    for (coord, template) in (
        (Cartesian, make_structured_grid_template()),
        (Cylindrical, make_cylindrical_grid_template()),
        (Spherical, make_spherical_grid_template()),
    )
        flattened_template = Partia.Grids.flatten(coord, template)
        point_result = PointSamples_interpolation(
            CPUComputeBackend(),
            flattened_template,
            input,
            catalog,
            itpSymmetric,
        )
        structured_result = StructuredGrid_interpolation(
            CPUComputeBackend(),
            coord,
            template,
            input,
            catalog,
            itpSymmetric,
        )

        @test structured_result.names == point_result.names
        @test structured_result.names == catalog.ordered_names

        for i in eachindex(structured_result.grids)
            @test structured_result.grids[i].axes == template.axes
            @test isapprox(vec(structured_result.grids[i].grid), point_result.grids[i].grid; atol = 1.0e-12, rtol = 1.0e-10)
        end
    end
end

@testset "StructuredGrid interpolation -- analytic linear-field regression" begin
    input, catalog, _ = make_uniform_cloud_3d(12; eta = 1.2, variable_h = true)
    for (coord_name, coord, structured_template) in (
        ("Cartesian", Cartesian, make_analytic_structured_grid()),
        ("Cylindrical", Cylindrical, make_analytic_cylindrical_grid()),
        ("Spherical", Spherical, make_analytic_spherical_grid()),
    )
        @testset "$coord_name grid" begin
            sample_coords = explicit_cartesian_coords(coord, structured_template)

            for strategy in (itpGather, itpScatter, itpSymmetric)
                result = StructuredGrid_interpolation(CPUComputeBackend(), coord, structured_template, input, catalog, strategy)

                q_grid = vec(result.grids[1].grid)
                gradx_grid = vec(result.grids[2].grid)
                grady_grid = vec(result.grids[3].grid)
                gradz_grid = vec(result.grids[4].grid)
                div_grid = vec(result.grids[5].grid)
                curlx_grid = vec(result.grids[6].grid)
                curly_grid = vec(result.grids[7].grid)
                curlz_grid = vec(result.grids[8].grid)

                for i in eachindex(q_grid)
                    point = (
                        sample_coords[1][i],
                        sample_coords[2][i],
                        sample_coords[3][i],
                    )
                    s_ref = analytic_scalar(point...)
                    g_ref = analytic_grad_scalar(point...)
                    div_ref = analytic_divA(point...)
                    curl_ref = analytic_curlA(point...)

                    @test isapprox(q_grid[i], s_ref; atol = 2e-2, rtol = 1e-2)
                    @test isapprox(gradx_grid[i], g_ref[1]; atol = 6e-2, rtol = 1e-2)
                    @test isapprox(grady_grid[i], g_ref[2]; atol = 6e-2, rtol = 1e-2)
                    @test isapprox(gradz_grid[i], g_ref[3]; atol = 6e-2, rtol = 1e-2)
                    @test isapprox(div_grid[i], div_ref; atol = 1e-1, rtol = 2e-2)
                    @test isapprox(curlx_grid[i], curl_ref[1]; atol = 5e-2, rtol = 1e-2)
                    @test isapprox(curly_grid[i], curl_ref[2]; atol = 5e-2, rtol = 1e-2)
                    @test isapprox(curlz_grid[i], curl_ref[3]; atol = 5e-2, rtol = 1e-2)
                end
            end
        end
    end
end
