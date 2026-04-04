"""
    Grid transformations between `StructuredGrid` and `PointSamples`.

This file provides the bridge between:

- `StructuredGrid`, which stores values on tensor-product axes
- `PointSamples`, which stores one explicit Cartesian sample coordinate per point

The key design point is that SPH interpolation kernels operate on explicit
Cartesian sample coordinates. A structured grid therefore has to be flattened
into `PointSamples` before interpolation.

For Cartesian grids, flattening is a direct reshaping plus axis expansion.
For non-Cartesian grids such as `Polar`, `Cylindrical`, and `Spherical`,
flattening additionally converts the grid coordinates into Cartesian sample
coordinates through the coordinate-dispatched `coordinate_grid(::Type{COORD}, ...)`
methods defined in `StructuredGrid.jl`.

The inverse operation, `restore_struct(::Type{COORD}, ...)`, does not attempt
to invert Cartesian coordinates back into axis coordinates. Instead, it takes
the provided axes, regenerates their Cartesian `coordinate_grid` under the same
coordinate-system dispatch, compares that against `PointSamples.coor`, and only
then reshapes the flattened value array back to the original structured-grid
shape and reattaches the original axes.
"""

"""
    flatten(::Type{COORD}, grid::StructuredGrid)

Flatten a `StructuredGrid` into a `PointSamples` object using the specified
coordinate-system dispatch.

This function always returns explicit Cartesian sample coordinates, because that
is the representation required by the interpolation kernels.

# Parameters
- `::Type{COORD}`:
  Coordinate-system tag controlling how the structured-grid axes are interpreted.
  Supported values are `Cartesian`, `Polar`, `Cylindrical`, and `Spherical`.
- `grid::StructuredGrid`:
  Structured grid whose values and axes define the sampling geometry.

# Returns
- `PointSamples`:
  A flattened sample container whose `grid` field is `vec(grid.grid)` and whose
  coordinates are the Cartesian locations of all structured-grid points.

# Notes
- `vec(grid.grid)` shares storage with the original structured-grid values.
- The coordinate transformation, when needed, happens inside
  `coordinate_grid(::Type{COORD}, grid)`.
"""
function flatten(::Type{Cartesian}, grid::StructuredGrid{D,TF}) where {D,TF<:AbstractFloat}
    coor = coordinate_grid(Cartesian, grid)
    return PointSamples(vec(grid.grid), coor)
end

function flatten(::Type{Polar}, grid::StructuredGrid{2,TF}) where {TF<:AbstractFloat}
    coor = coordinate_grid(Polar, grid)
    return PointSamples(vec(grid.grid), coor)
end

function flatten(::Type{Cylindrical}, grid::StructuredGrid{3,TF}) where {TF<:AbstractFloat}
    coor = coordinate_grid(Cylindrical, grid)
    return PointSamples(vec(grid.grid), coor)
end

function flatten(::Type{Spherical}, grid::StructuredGrid{3,TF}) where {TF<:AbstractFloat}
    coor = coordinate_grid(Spherical, grid)
    return PointSamples(vec(grid.grid), coor)
end

"""
    restore_struct(::Type{COORD}, grid::PointSamples{D,TF}, axes::NTuple{D,V};
                   atol::Real = 1.0e-8, rtol::Real = 1.0e-8) where
                   {COORD,D,TF<:AbstractFloat, V<:AbstractVector{TF}}

Restore a `StructuredGrid` from a flattened `PointSamples`.

This routine validates `grid.coor` by regenerating the Cartesian
`coordinate_grid(::Type{COORD}, ...)` from the provided `axes` and comparing
the two coordinate sets directly. No inverse coordinate transformation is used.
It then:

1. reshapes `grid.grid` back to the tensor-product size implied by `axes`
2. reattaches the provided axes to construct a `StructuredGrid`

# Parameters
- `::Type{COORD}`:
  Coordinate-system tag used to interpret the provided axes before comparing
  them against the flattened sample coordinates.
- `grid::PointSamples{D,TF}`:
  Flattened output values.
- `axes::NTuple{D,V}`:
  Original structured-grid axes that define the target tensor-product shape.

# Keyword Arguments
| Keyword | Default | Description |
|---|---|---|
| `atol` | `1.0e-8` | Absolute tolerance used in the coordinate-consistency check. |
| `rtol` | `1.0e-8` | Relative tolerance used in the coordinate-consistency check. |

# Returns
- `StructuredGrid`:
  Structured grid with values reshaped to the size implied by `axes`.
"""
function restore_struct(::Type{COORD}, grid::PointSamples{D,TF}, axes::NTuple{D,V}; atol::Real = 1.0e-8, rtol::Real = 1.0e-8) where {COORD<:AbstractCoordinateSystem,D,TF<:AbstractFloat, V <: AbstractVector{TF}}
    size = ntuple(i -> length(axes[i]), D)

    expected_template = StructuredGrid(reshape(similar(grid.grid), size), axes, size)
    expected_coor = coordinate_grid(COORD, expected_template)
    _coordinate_grid_isapprox(grid.coor, expected_coor; atol = atol, rtol = rtol) ||
        throw(ArgumentError("PointSamples coordinates do not match the provided axes under $(COORD) dispatch"))

    return StructuredGrid(reshape(grid.grid, size), axes, size)
end
