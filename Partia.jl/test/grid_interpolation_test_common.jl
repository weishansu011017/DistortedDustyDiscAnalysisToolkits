using Test
using Random
using Partia

ki_mod = Partia.KernelInterpolation

@static if !isdefined(@__MODULE__, :support_radius)
    include("interpolation_test_common.jl")
end
@static if !isdefined(@__MODULE__, :analytic_scalar)
    include("interpolation_analytic_test_common.jl")
end


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

    input = InterpolationInput(
        x, y, z, m, h, rho, (temp, vx, vy, vz);
        smoothed_kernel = M4_spline,
    )
    catalog = InterpolationCatalog(
        (:temp, :vx, :vy, :vz), Val(3);
        scalars = (:temp,),
        divergences = (:v,),
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

function make_polar_grid_template()
    return StructuredGrid(
        Polar,
        (0.25, 0.55, 3),
        (0.30, 1.10, 2),
    )
end

function make_cylindrical_grid_template()
    return StructuredGrid(
        Cylindrical,
        (0.25, 0.55, 3),
        (0.30, 1.10, 2),
        (0.20, 0.50, 2),
    )
end

function make_spherical_grid_template()
    return StructuredGrid(
        Spherical,
        (0.50, 0.70, 2),
        (0.35, 0.85, 2),
        (0.80, 1.10, 2),
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

    input = InterpolationInput(
        x, y, z, m, h, rho, (temp, vx);
        smoothed_kernel = M4_spline,
    )
    catalog = InterpolationCatalog(
        (:temp, :vx), Val(3);
        scalars = (:temp, :vx),
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

function make_analytic_point_samples()
    x = Float64[0.35, 0.45, 0.55]
    y = Float64[0.40, 0.50, 0.45]
    z = Float64[0.45, 0.35, 0.50]
    return PointSamples(zeros(Float64, length(x)), (x, y, z))
end

function make_analytic_structured_grid()
    return StructuredGrid(
        Cartesian,
        (0.35, 0.55, 3),
        (0.35, 0.55, 3),
        (0.35, 0.55, 3),
    )
end

function make_analytic_cylindrical_grid()
    return StructuredGrid(
        Cylindrical,
        (0.35, 0.60, 3),
        (0.55, 0.95, 3),
        (0.25, 0.45, 3),
    )
end

function make_analytic_spherical_grid()
    return StructuredGrid(
        Spherical,
        (0.55, 0.75, 3),
        (0.45, 0.75, 3),
        (0.85, 1.05, 3),
    )
end

function explicit_cartesian_coords(::Type{Cartesian}, grid::StructuredGrid{3,TF}) where {TF<:AbstractFloat}
    return ntuple(d -> begin
        out = similar(vec(grid.grid))
        L = LinearIndices(grid.size)
        @inbounds for I in CartesianIndices(grid.size)
            out[L[I]] = grid.axes[d][I[d]]
        end
        out
    end, 3)
end

function explicit_cartesian_coords(::Type{Polar}, grid::StructuredGrid{2,TF}) where {TF<:AbstractFloat}
    x = similar(vec(grid.grid))
    y = similar(vec(grid.grid))
    L = LinearIndices(grid.size)
    @inbounds for I in CartesianIndices(grid.size)
        i = L[I]
        s = grid.axes[1][I[1]]
        ϕ = grid.axes[2][I[2]]
        x[i] = s * cos(ϕ)
        y[i] = s * sin(ϕ)
    end
    return (x, y)
end

function explicit_cartesian_coords(::Type{Cylindrical}, grid::StructuredGrid{3,TF}) where {TF<:AbstractFloat}
    x = similar(vec(grid.grid))
    y = similar(vec(grid.grid))
    z = similar(vec(grid.grid))
    L = LinearIndices(grid.size)
    @inbounds for I in CartesianIndices(grid.size)
        i = L[I]
        s = grid.axes[1][I[1]]
        ϕ = grid.axes[2][I[2]]
        zi = grid.axes[3][I[3]]
        x[i] = s * cos(ϕ)
        y[i] = s * sin(ϕ)
        z[i] = zi
    end
    return (x, y, z)
end

function explicit_cartesian_coords(::Type{Spherical}, grid::StructuredGrid{3,TF}) where {TF<:AbstractFloat}
    x = similar(vec(grid.grid))
    y = similar(vec(grid.grid))
    z = similar(vec(grid.grid))
    L = LinearIndices(grid.size)
    @inbounds for I in CartesianIndices(grid.size)
        i = L[I]
        r = grid.axes[1][I[1]]
        ϕ = grid.axes[2][I[2]]
        θ = grid.axes[3][I[3]]
        x[i] = r * sin(θ) * cos(ϕ)
        y[i] = r * sin(θ) * sin(ϕ)
        z[i] = r * cos(θ)
    end
    return (x, y, z)
end


# =========================== Brute-force references ========================== #

@inline function brute_nearest_h(input, point::NTuple{3,T}) where {T<:AbstractFloat}
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    best_idx = 1
    best_d2 = typemax(T)
    @inbounds for i in eachindex(x)
        dx = point[1] - x[i]
        dy = point[2] - y[i]
        dz = point[3] - z[i]
        d2 = dx*dx + dy*dy + dz*dz
        if d2 < best_d2
            best_d2 = d2
            best_idx = i
        end
    end
    return input.h[best_idx]
end
