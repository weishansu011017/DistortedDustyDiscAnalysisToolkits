"""
    flatten(grid::StructuredGrid)

Flatten a `StructuredGrid` into a one-dimensional `PointSamples`.

- The data array `grid.grid` is reshaped with `vec`, producing a flat `Vector{TF}`.
- The coordinate grid (from `coordinate_grid(grid)`) is flattened to 
  `Vector{NTuple{D,TF}}`, where `D` is the dimension of Grid, and `TF` is the type of float.
- The returned `PointSamples` pairs the flattened values with their corresponding
  coordinates.

# Parameters
- `grid::StructuredGrid` : Structured grid object to be flattened.

# Returns
- `PointSamples{D, TF, Vector{TF}, Vector{NTuple{D, TF}}}` : A grid containing
  flat values and their coordinates.

# Notes
- The flattened `grid.grid` returned by `vec` shares the same underlying memory
  as the original `grid.grid`. No data copy occurs, so modifying one will also
  affect the other.
"""
function flatten(grid::StructuredGrid)
    coor = coordinate_grid(grid)
    return PointSamples(vec(grid.grid), coor)
end

"""
    restore_struct(grid::PointSamples{D,TF}, axes::NTuple{D,V}) where {D,TF<:AbstractFloat, V <: AbstractVector{TF}}

Restore a `StructuredGrid` from a flattened `PointSamples`. The coordinates stored
in `grid.coor` are validated against `axes` using `isapprox` with its default tolerances.

# Parameters
- `grid::PointSamples{D,TF}` : Flattened grid to be restored.
- `axes::NTuple{D,V}` : Target coordinate axes defining the output shape.

# Returns
- `StructuredGrid` : Grid reshaped to match `axes`, with values from `grid.grid`.

"""
function restore_struct(grid::PointSamples{D,TF}, axes::NTuple{D,V}) where {D,TF<:AbstractFloat, V <: AbstractVector{TF}}
    size = ntuple(i -> length(axes[i]), D)

    if !isapprox(grid, axes)
        throw(ArgumentError("DimensionalMismatch: Coordinate mismatch between PointSamples and axes!"))
    end

    return StructuredGrid(reshape(grid.grid, size), axes, size)
end
