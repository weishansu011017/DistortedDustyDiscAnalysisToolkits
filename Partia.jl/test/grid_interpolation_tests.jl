# ────────────────────────────────────────────────────────────────────────────
#  Test: Grid Interpolation -- PointSamples and StructuredGrid
# ────────────────────────────────────────────────────────────────────────────
#
#  What this file tests
#  ────────────────────
#  Validates the formal grid/sample interpolation layer on the CPU:
#
#  1. `PointSamples_interpolation`
#     -- Output names follow the interpolation catalog ordering.
#     -- Interpolated scalar / divergence fields match the corresponding
#        single-point kernel evaluations at each sample location.
#     -- Returned grids preserve the sample coordinates.
#
#  2. `flatten` / `restore_struct`
#     -- Flattening a `StructuredGrid` and restoring it back to the same axes
#        reproduces the original grid values and axes exactly.
#
#  3. `LineSamples_interpolation`
#     -- Scalar line-integrated outputs match the corresponding direct
#        line-integrated kernel evaluations for `itpScatter`.
#     -- Returned samples preserve line origin/direction geometry.
#     -- Unsupported strategies / non-scalar catalogs are rejected explicitly.
#
#  4. `StructuredGrid_interpolation`
#     -- Matches `PointSamples_interpolation` applied to the flattened grid.
#     -- Restores the structured axes and value layout correctly.
#
# ────────────────────────────────────────────────────────────────────────────

using Test
using Partia
using Partia.KernelInterpolation: _quantity_interpolate_kernel,
    _divergence_quantity_interpolate_kernel,
    _line_integrated_quantities_interpolate_kernel

ki_mod = Partia.KernelInterpolation


# ========================== Fixture builders ================================ #

function make_grid_interpolation_fixture()
    x = Float64[0.15, 0.35, 0.55, 0.75]
    y = Float64[0.20, 0.60, 0.25, 0.70]
    z = Float64[0.25, 0.45, 0.80, 0.30]
    h = Float64[0.28, 0.24, 0.26, 0.22]
    rho = Float64[1.0, 0.95, 1.05, 1.1]
    m = Float64[0.4, 0.45, 0.42, 0.38]
    temp = Float64[10.0, 11.0, 13.0, 12.0]
    vx = Float64[0.1, -0.2, 0.3, -0.1]
    vy = Float64[0.0, 0.25, -0.15, 0.2]
    vz = Float64[-0.05, 0.1, 0.2, -0.1]

    input, catalog = build_input(
        CPUComputeBackend(),
        x, y, z, h, rho, m, (temp, vx, vy, vz);
        column_names = (:temp, :vx, :vy, :vz),
        scalars = (:temp,),
        divergences = (:v,),
        smoothed_kernel = M4_spline,
    )

    LBVH = LinearBVH!(input, Val(3))
    return input, catalog, LBVH
end

function make_point_samples_template()
    x = Float64[0.20, 0.40, 0.65]
    y = Float64[0.25, 0.55, 0.50]
    z = Float64[0.30, 0.50, 0.35]
    return PointSamples(zeros(Float64, length(x)), (x, y, z))
end

function make_structured_grid_template()
    return StructuredGrid(
        Cartesian,
        (0.20, 0.70, 3),
        (0.25, 0.55, 2),
        (0.30, 0.60, 2),
    )
end

function make_line_interpolation_fixture()
    x = Float64[0.15, 0.35, 0.55, 0.75]
    y = Float64[0.20, 0.60, 0.25, 0.70]
    z = Float64[0.25, 0.45, 0.80, 0.30]
    h = Float64[0.28, 0.24, 0.26, 0.22]
    rho = Float64[1.0, 0.95, 1.05, 1.1]
    m = Float64[0.4, 0.45, 0.42, 0.38]
    temp = Float64[10.0, 11.0, 13.0, 12.0]
    vx = Float64[0.1, -0.2, 0.3, -0.1]

    input, catalog = build_input(
        CPUComputeBackend(),
        x, y, z, h, rho, m, (temp, vx);
        column_names = (:temp, :vx),
        scalars = (:temp, :vx),
        smoothed_kernel = M4_spline,
    )

    LBVH = LinearBVH!(input, Val(3))
    return input, catalog, LBVH
end

function make_line_samples_template()
    invsqrt2 = inv(sqrt(2.0))
    origin = (
        Float64[0.20, 0.40, 0.55],
        Float64[0.25, 0.35, 0.45],
        Float64[0.00, 0.10, 0.20],
    )
    direction = (
        Float64[0.0, invsqrt2, 1.0],
        Float64[0.0, 0.0, 0.0],
        Float64[1.0, invsqrt2, 0.0],
    )
    return LineSamples(zeros(Float64, length(origin[1])), origin, direction)
end


# ============================== Test body =================================== #

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
        ha = LBVH_find_nearest_h(LBVH, point)

        expected_scalar = _quantity_interpolate_kernel(input, point, ha, LBVH, scalar_slot, true, itpSymmetric)
        expected_div = _divergence_quantity_interpolate_kernel(input, point, ha, LBVH, div_slots..., itpSymmetric)

        @test isapprox(result.grids[1].grid[i], expected_scalar; atol = 1.0e-12, rtol = 1.0e-10)
        @test isapprox(result.grids[2].grid[i], expected_div; atol = 1.0e-12, rtol = 1.0e-10)
    end
end

@testset "StructuredGrid transform -- flatten and restore" begin
    grid = make_structured_grid_template()
    grid.grid .= reshape(collect(1.0:length(grid.grid)), size(grid))

    flattened = Partia.Grids.flatten(grid)
    restored = Partia.Grids.restore_struct(flattened, grid.axes)

    @test restored.grid == grid.grid
    @test restored.axes == grid.axes
    @test restored.size == size(grid)
end

@testset "LineSamples interpolation -- CPU scatter consistency" begin
    input, catalog, LBVH = make_line_interpolation_fixture()
    grid_template = make_line_samples_template()
    result = LineSamples_interpolation(CPUComputeBackend(), grid_template, input, catalog, itpScatter)

    scalar_slots = catalog.scalar_slots
    scalar_snormalization = catalog.scalar_snormalization

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

        expected = _line_integrated_quantities_interpolate_kernel(
            input,
            origin,
            direction,
            LBVH,
            scalar_slots,
            scalar_snormalization,
            itpScatter,
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

@testset "StructuredGrid interpolation -- flattened consistency" begin
    input, catalog, _ = make_grid_interpolation_fixture()
    structured_template = make_structured_grid_template()
    flattened_template = Partia.Grids.flatten(structured_template)

    point_result = PointSamples_interpolation(
        CPUComputeBackend(),
        flattened_template,
        input,
        catalog,
        itpSymmetric,
    )
    structured_result = StructuredGrid_interpolation(
        CPUComputeBackend(),
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
