# ──────────────────────────────────────────────────────────────────────────── #
#  Test: Type Stability — Core Numerical Kernels and Hot Helpers
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ─────────────────────
#  Representative `@inferred` checks for concrete, performance-critical code
#  paths that should stay type-stable:
#
#  1. SPH kernel evaluation
#     • `Smoothed_kernel_function*`
#     • `Smoothed_gradient_kernel_function`
#     • `line_integrated_kernel_function*`
#
#  2. Neighbor-search query helpers
#     • `LBVH_probe_neighbors`
#     • `LBVH_find_nearest`
#     • `LBVH_find_nearest_h`
#     • `LBVH_query!`
#
#  3. Single-point interpolation kernels
#     • density / quantity
#     • gradient / divergence / curl
#     • multi-quantity interpolation
#
#  4. Line-integrated interpolation kernels
#     • density
#     • single- and multi-quantity interpolation
#
#
#  Deliberately excluded here:
#  • I/O, logging, and orchestration-heavy wrappers
#  • APIs that intentionally return flexible container shapes
#  • broad generic method sweeps; only representative concrete inputs are used
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using StaticArrays
using Partia
using Partia.KernelInterpolation:
    _density_accumulation,
    _quantity_interpolate_accumulation,
    _gradient_quantity_accumulation,
    _divergence_quantity_accumulation,
    _curl_quantity_accumulation,
    _line_integrated_quantity_interpolate_accumulation,
    _density_kernel,
    _number_density_kernel,
    _quantity_interpolate_kernel,
    _quantities_interpolate_kernel,
    _gradient_density_kernel,
    _gradient_quantity_interpolate_kernel,
    _divergence_quantity_interpolate_kernel,
    _curl_quantity_interpolate_kernel,
    _line_integrated_density_kernel,
    _line_integrated_quantities_interpolate_kernel


# ========================== Fixture builders ================================ #

function make_type_stability_input_3d()
    kern = M4_spline()
    x = Float64[0.10, 0.22, 0.34, 0.46, 0.58, 0.70]
    y = Float64[0.15, 0.28, 0.20, 0.52, 0.44, 0.68]
    z = Float64[0.12, 0.18, 0.36, 0.48, 0.62, 0.40]
    m = Float64[0.4, 0.45, 0.42, 0.38, 0.41, 0.43]
    h = Float64[0.16, 0.18, 0.17, 0.19, 0.16, 0.18]
    ρ = Float64[1.0, 1.1, 0.95, 1.05, 1.02, 0.98]
    q1 = Float64[10.0, 11.5, 9.0, 12.0, 13.5, 8.5]
    q2 = Float64[0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    q3 = Float64[0.0, 0.25, -0.15, 0.2, -0.05, 0.1]
    input = InterpolationInput((x, y, z), m, h, ρ, (q1, q2, q3); smoothed_kernel = typeof(kern))
    lbvh = LinearBVH!(input, Val(3))
    return input, lbvh
end

function make_type_stability_input_line_integrated()
    kern = M4_spline()
    x = Float64[0.12, 0.26, 0.38, 0.51, 0.64]
    y = Float64[0.08, 0.30, 0.22, 0.57, 0.41]
    z = Float64[0.10, 0.24, 0.48, 0.36, 0.62]
    m = Float64[0.5, 0.43, 0.47, 0.39, 0.44]
    h = Float64[0.15, 0.17, 0.16, 0.18, 0.15]
    ρ = Float64[1.0, 1.08, 0.97, 1.02, 1.05]
    q1 = Float64[2.0, 2.5, 3.0, 3.5, 4.0]
    input = InterpolationInput((x, y, z), m, h, ρ, (q1,); smoothed_kernel = typeof(kern))
    lbvh = LinearBVH!(input, Val(3))
    return input, lbvh
end


# ============================== Test body =================================== #

# ── 1. Kernel evaluation ────────────────────────────────────────────── #

@testset "Type stability — kernel evaluation" begin
    kern = M4_spline()
    @inferred Smoothed_kernel_function_dimensionless(typeof(kern), 0.8, Val(3))
    @inferred Smoothed_kernel_function(typeof(kern), 0.12, 0.20, Val(3))
    @inferred Smoothed_kernel_function(typeof(kern), (0.1, 0.2, 0.3), (0.2, 0.0, 0.4), 0.20)
    @inferred Smoothed_gradient_kernel_function(typeof(kern), 0.05, -0.03, 0.08, 0.20)
    @inferred Smoothed_gradient_kernel_function(typeof(kern), (0.1, 0.2, 0.3), (0.2, 0.0, 0.4), 0.20)
    @inferred line_integrated_kernel_function_dimensionless(typeof(kern), 0.6)
    @inferred line_integrated_kernel_function(typeof(kern), 0.07, 0.20)
    @inferred line_integrated_kernel_function(typeof(kern), (0.1, 0.2), (0.3, 0.4), 0.20)
end

# ── 2. Neighbor-search query helpers ────────────────────────────────── #

@testset "Type stability — accumulation helpers" begin
    kern = M4_spline()
    ra = (0.15, 0.25, 0.35)
    rb = (0.28, 0.12, 0.40)
    Δr = 0.18
    mb = 0.42
    ρb = 1.08
    Ab = 2.5
    Axb = 0.1
    Ayb = -0.2
    Azb = 0.3
    h = 0.17

    @inferred _density_accumulation(ra, rb, mb, h, kern)
    @inferred _quantity_interpolate_accumulation(ra, rb, mb, ρb, Ab, h, kern)
    @inferred _gradient_quantity_accumulation(ra, rb, mb, ρb, Ab, h, kern)
    @inferred _divergence_quantity_accumulation(ra, rb, mb, ρb, Axb, Ayb, Azb, h, kern)
    @inferred _curl_quantity_accumulation(ra, rb, mb, ρb, Axb, Ayb, Azb, h, kern)
    @inferred _line_integrated_quantity_interpolate_accumulation(Δr, mb, ρb, Ab, h, kern)
end

# ── 3. Neighbor-search query helpers ────────────────────────────────── #

@testset "Type stability — LBVH query helpers" begin
    input, lbvh = make_type_stability_input_3d()
    point = (0.33, 0.27, 0.31)
    radius = 0.22
    pool = zeros(Int, input.Npart)

    @inferred LBVH_probe_neighbors(lbvh, point, radius)
    @inferred LBVH_find_nearest(lbvh, point)
    @inferred LBVH_find_nearest_h(lbvh, point)
    @inferred LBVH_query!(pool, lbvh, point, radius)
end

# ── 4. Single-point interpolation kernels ───────────────────────────── #

@testset "Type stability — single-point interpolation kernels" begin
    input, lbvh = make_type_stability_input_3d()
    point = (0.33, 0.27, 0.31)
    ha = 0.19

    for strategy in (itpGather, itpScatter, itpSymmetric)
        if strategy === itpScatter
            @inferred _density_kernel(input, point, lbvh, strategy)
            @inferred _number_density_kernel(input, point, lbvh, strategy)
            @inferred _quantity_interpolate_kernel(input, point, lbvh, 1, true, strategy)
            @inferred _quantities_interpolate_kernel(input, point, lbvh, (1, 2), (true, false), strategy)
            @inferred _gradient_density_kernel(input, point, lbvh, strategy)
            @inferred _gradient_quantity_interpolate_kernel(input, point, lbvh, 2, strategy)
            @inferred _divergence_quantity_interpolate_kernel(input, point, lbvh, 1, 2, 3, strategy)
            @inferred _curl_quantity_interpolate_kernel(input, point, lbvh, 1, 2, 3, strategy)
        else
            @inferred _density_kernel(input, point, ha, lbvh, strategy)
            @inferred _number_density_kernel(input, point, ha, lbvh, strategy)
            @inferred _quantity_interpolate_kernel(input, point, ha, lbvh, 1, true, strategy)
            @inferred _quantities_interpolate_kernel(input, point, ha, lbvh, (1, 2), (true, false), strategy)
            @inferred _gradient_density_kernel(input, point, ha, lbvh, strategy)
            @inferred _gradient_quantity_interpolate_kernel(input, point, ha, lbvh, 2, strategy)
            @inferred _divergence_quantity_interpolate_kernel(input, point, ha, lbvh, 1, 2, 3, strategy)
            @inferred _curl_quantity_interpolate_kernel(input, point, ha, lbvh, 1, 2, 3, strategy)
        end
    end
end

# ── 5. Line-integrated interpolation kernels ────────────────────────── #

@testset "Type stability — line-integrated interpolation kernels" begin
    input, lbvh = make_type_stability_input_line_integrated()
    origin = (0.30, 0.35, 0.00)
    direction = (0.0, 0.0, 1.0)
    ha = 0.18

    for strategy in (itpGather, itpScatter, itpSymmetric)
        if strategy === itpScatter
            @inferred _line_integrated_density_kernel(input, origin, direction, lbvh, strategy)
            @inferred _line_integrated_quantities_interpolate_kernel(input, origin, direction, lbvh, (1,), (true,), strategy)
        else
            @inferred _line_integrated_density_kernel(input, origin, direction, ha, lbvh, strategy)
            @inferred _line_integrated_quantities_interpolate_kernel(input, origin, direction, ha, lbvh, (1,), (true,), strategy)
        end
    end
end
