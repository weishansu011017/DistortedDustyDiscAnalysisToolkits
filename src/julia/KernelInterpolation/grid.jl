"""
The general grid construction for SPH interpolation
    by Wei-Shan Su,
    September 21, 2025
"""

# Dispatch tag for constructing grid
abstract type AbstractCoordinateSystem end

struct Cartesian       <: AbstractCoordinateSystem end 
struct Polar           <: AbstractCoordinateSystem end        # (s, ϕ)
struct Cylindrical     <: AbstractCoordinateSystem end        # (s, ϕ, z)  
struct Spherical       <: AbstractCoordinateSystem end        # (r, ϕ, θ)

# Axis specification tuple `(xmin, xmax, xn)`.
"""
    AxisParam{TF} = Tuple{TF, TF, Int}

Axis specification tuple `(xmin, xmax, xn)`.

# Type Parameters
- `TF <: AbstractFloat` : Floating-point type for axis endpoints.

# Fields / Layout
- `xmin::TF` : Axis minimum.
- `xmax::TF` : Axis maximum.
- `xn::Int`  : Number of points (length).

# Examples
```julia
const AxisParam{TF} = Tuple{TF, TF, Int}
x = (0.0, 1.0, 256)  # AxisParam{Float64}
y = (0.0f0, 2.0f0, 128)  # AxisParam{Float32}
```
"""
const AxisParam{TF} = Tuple{TF, TF, Int}

# General Grid definition
abstract type AbstractInterpolationGrid end

"""
    Base.length(grid::GRID) where {GRID <: AbstractInterpolationGrid}

Return the number of elements stored in the grid values array.

This delegates to `length(grid.grid)`, i.e. the length of the internal 
storage vector for grid values.

# Parameters
- `grid::GRID` : Any concrete subtype of `AbstractInterpolationGrid`.

# Returns
- `Int` : The number of stored grid values.
"""
@inline Base.length(grid :: GRID) where {GRID <: AbstractInterpolationGrid} = length(grid.grid)

"""
    GeneralGrid{D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: AbstractVector{NTuple{D, TF}}}  <: AbstractInterpolationGrid

A generic grid container for interpolation, storing both grid values and 
their corresponding coordinates.
`coor` is intended to be immutable and can be shared across grids.

# Type Parameters
- `D` : Dimensionality of the grid.
- `TF <: AbstractFloat` : Floating-point element type.
- `VG <: AbstractVector{TF}` : Storage type for grid values.
- `VC <: AbstractVector{NTuple{D, TF}}` : Storage type for grid coordinates.

# Fields
- `grid :: VG` : Vector containing the grid values.
- `coor :: VC` : Vector containing the coordinates of each grid point.
"""
struct GeneralGrid{D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: AbstractVector{NTuple{D, TF}}} <: AbstractInterpolationGrid
    grid :: VG
    coor :: VC
end

function Adapt.adapt_structure(to, x :: GeneralGrid)
    GeneralGrid(
        Adapt.adapt(to, x.grid),
        Adapt.adapt(to, x.coor)
    )
end

"""
    similar(grid::GeneralGrid)

Construct a new `GeneralGrid` with fresh storage for values but sharing
the same coordinate container as the input grid.

# Parameters
- `grid::GeneralGrid` : Template grid to copy structure from.

# Returns
- `GeneralGrid` : A grid with independent value storage (`grid.grid`)
  and shared coordinates (`grid.coor`).
"""
function Base.similar(grid::GeneralGrid)
    return GeneralGrid(similar(grid.grid), grid.coor)
end

"""
    Base.isapprox(grid::GeneralGrid{D,TF}, axes::NTuple{D,<:AbstractVector}; atol::Real=1.0e-8, rtol::Real=1.0e-8) :: Bool where {D,TF<:AbstractFloat}

Check whether the coordinates stored in a `GeneralGrid` match the given `axes`
(up to numerical tolerance).

# Parameters
- `grid::GeneralGrid{D,TF}` : Grid whose coordinates will be checked.
- `axes::NTuple{D,<:AbstractVector}` : Target coordinate axes.

# Keyword Arguments
| Name            | Default  | Description                                    |
|-----------------|----------|------------------------------------------------|
| `atol::Real`    | `1.0e-8` | Absolute tolerance for floating-point comparison. |
| `rtol::Real`    | `1.0e-8` | Relative tolerance for floating-point comparison. |

# Returns
- `Bool` : `true` if all coordinates match within tolerance, otherwise `false`.
"""
function Base.isapprox(grid::GeneralGrid{D,TF}, axes::NTuple{D,<:AbstractVector}; atol :: Real = 1.0e-8, rtol :: Real = 1.0e-8) :: Bool where {D,TF <: AbstractFloat}
    size_expected = ntuple(i -> length(axes[i]), D)
    rscoor = reshape(grid.coor, size_expected)

    inds = CartesianIndices(size_expected)
    @inbounds for I in inds
        coor_tuple = rscoor[I]
        @inbounds @simd for d in 1:D
            val_expected = axes[d][I[d]]
            val_actual   = coor_tuple[d]
            if !isapprox(val_actual, val_expected; atol=atol, rtol=rtol)
                return false
            end
        end
    end
    return true
end

"""
    restore_struct(grid::GeneralGrid{D,TF}, axes::NTuple{D,V}) where {D,TF<:AbstractFloat, V <: AbstractVector{TF}}

Restore a `StructuredGrid` from a flattened `GeneralGrid`. The coordinates stored
in `grid.coor` are validated against `axes` using `isapprox` with its default tolerances.

# Parameters
- `grid::GeneralGrid{D,TF}` : Flattened grid to be restored.
- `axes::NTuple{D,V}` : Target coordinate axes defining the output shape.

# Returns
- `StructuredGrid` : Grid reshaped to match `axes`, with values from `grid.grid`.

"""
function restore_struct(grid::GeneralGrid{D,TF}, axes::NTuple{D,V}) where {D,TF<:AbstractFloat, V <: AbstractVector{TF}}
    size = ntuple(i -> length(axes[i]), D)

    if !isapprox(grid, axes)
        throw(ArgumentError("DimensionalMismatch: Coordinate mismatch between GeneralGrid and axes!"))
    end

    return StructuredGrid(reshape(grid.grid, size), axes, size)
end

"""
    batch_GeneralGrid(grid::GeneralGrid{D, TF, VG, VC}, batch_size::Int)

Split a `GeneralGrid` into contiguous batches, each containing at most `batch_size` points.
The function preserves ordering of both `grid.grid` and `grid.coor`, returning a statically-sized
NTuple where each element is a `GeneralGrid{D, TF, VG, VC}` containing a slice of the input data.

Batch `i` contains points from index  
`(i−1)·batch_size + 1` to `min(i·batch_size, N)`  
where `N = length(grid)`.

# Parameters
- `grid::GeneralGrid{D, TF, VG, VC}`  
  Input grid containing scalar field values and coordinate tuples.

- `batch_size::Int`  
  Maximum number of points in each batch.

# Returns
`NTuple{B, GeneralGrid{D, TF, VG, VC}}` where `B = cld(N, batch_size)`  
and `N = length(grid)`.  
Each returned `GeneralGrid` contains:
- `grid.grid[start:stop]`  
- `grid.coor[start:stop]`  
with correct contiguous slicing.
"""
function batch_GeneralGrid(grid::GeneralGrid{D,TF,VG,VC}, batch_size::Int) where {D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:AbstractVector{NTuple{D,TF}}}
    npoints = length(grid)
    num_batches = cld(npoints, batch_size)

    return ntuple(i -> begin
        start = (i-1)*batch_size + 1
        stop  = min(i*batch_size, npoints)
        GeneralGrid{D,TF,VG,VC}(grid.grid[start:stop], grid.coor[start:stop])
    end, num_batches)
end

"""
    merge_GeneralGrid(grids::AbstractVector{GeneralGrid{D,TF,VG,VC}})

Merge a vector of `GeneralGrid` batches back into a single `GeneralGrid`.
The function concatenates all `grid.grid` and `grid.coor` fields in order,
restoring the original point ordering before batching.

# Parameters
- `grids::AbstractVector{GeneralGrid{D,TF,VG,VC}}`
  A vector of batched `GeneralGrid` objects, typically produced by
  `batch_GeneralGrid`.

# Returns
`GeneralGrid{D,TF,VG,VC}` with all batched segments concatenated in order.
"""
function merge_GeneralGrid(grids :: V) where {D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:AbstractVector{NTuple{D,TF}}, GG <: GeneralGrid{D,TF,VG,VC}, V <: AbstractVector{GG}}
    # Concatenate scalar values and coordinates
    merged_grid = vcat(map(g -> g.grid, grids)...)
    merged_coor = vcat(map(g -> g.coor, grids)...)

    return GeneralGrid{D,TF,VG,VC}(merged_grid, merged_coor)
end
# structured grid (Cartesian/Cylindrical... etc)
"""
    StructuredGrid{D, TF<:AbstractFloat, V<:AbstractVector{TF}, A<:AbstractArray{TF,D}} <: AbstractInterpolationGrid

A structured grid container, storing values in an N-dimensional array
together with coordinate axes for each dimension.

# Type Parameters
- `D` : Dimensionality of the grid.
- `TF <: AbstractFloat` : Floating-point element type.
- `V <: AbstractVector{TF}` : Type of each axis coordinate vector.
- `A <: AbstractArray{TF,D}` : Storage type for grid values.

# Fields
- `grid :: A` : N-dimensional array of grid values.
- `axes :: NTuple{D,V}` : Tuple of coordinate vectors, one per dimension.
- `size :: NTuple{D,Int}` : Logical size of the grid (cached from axes).
"""
struct StructuredGrid{D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}} <: AbstractInterpolationGrid 
    grid :: A
    axes :: NTuple{D, V}
    size :: NTuple{D, Int}
end

function Adapt.adapt_structure(to, x :: SG) where {D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}, SG <: StructuredGrid{D, TF, V, A}}
    StructuredGrid(
        Adapt.adapt(to, x.grid),
        ntuple(i -> Adapt.adapt(to, x.axes[i]), D),
        Adapt.adapt(to, x.coor)
    )
end

## Extent Base functions
"""
    Base.size(grid::StructuredGrid)

Return the logical size (dimensions) of the structured grid, as stored in
the `size` field of the object.

# Parameters
- `grid::StructuredGrid` : Grid object.

# Returns
- `NTuple{D,Int}` : Dimensions of the grid.
"""
@inline Base.size(grid :: StructuredGrid) = grid.size

"""
    Base.size(grid::StructuredGrid, d::Integer)

Return the extent of the `d`-th dimension of a structured grid.

# Parameters
- `grid::StructuredGrid` : Grid object.
- `d::Integer` : Dimension index (1-based).

# Returns
- `Int` : Size of the `d`-th dimension.
"""
@inline Base.size(grid::StructuredGrid, d::Integer) = grid.size[d]

## Functions
"""
    coordinate(grid::StructuredGrid{D, TF}, element::NTuple{D, Int}) where {D, TF <: AbstractFloat}

Return the physical coordinate corresponding to a grid index.

# Parameters
- `grid::StructuredGrid{D}`  
  The structured grid container.
- `element::NTuple{D, Int}`  
  Index tuple (e.g., `(i, j, k)`) of the grid point.

# Returns
- `::NTuple{D, TF}`  
  A tuple of coordinates `(x, y, z, ...)` corresponding to the grid point.
"""
@inline function coordinate(grid::StructuredGrid{D, TF}, element::NTuple{D, Int}) where {D, TF <: AbstractFloat}
    return ntuple(i -> grid.axes[i][element[i]], D)
end

"""
    coordinate_grid(grid::StructuredGrid{D, TF}) where {D, TF<:AbstractFloat}

Generate a full coordinate grid from the `StructuredGrid` definition.

For each logical index `(i, j, ...)`, returns the physical coordinate
`(x, y, ...)` using `grid.axes`.

# Parameters
- `grid::StructuredGrid{D,TF}` : The structured grid container.

# Returns
- `Array{NTuple{D,TF}, D}` : Same shape as `grid.grid`, each element is the
  coordinate tuple at that grid point.
"""
function coordinate_grid(grid::StructuredGrid{D, TF}) where {D, TF<:AbstractFloat}
    coords = Array{NTuple{D, TF}, D}(undef, grid.size...)
    @inbounds for I in CartesianIndices(coords)
        coords[I] = coordinate(grid, Tuple(I))
    end
    return coords
end

"""
    flatten(grid::StructuredGrid)

Flatten a `StructuredGrid` into a one-dimensional `GeneralGrid`.

- The data array `grid.grid` is reshaped with `vec`, producing a flat `Vector{TF}`.
- The coordinate grid (from `coordinate_grid(grid)`) is flattened to 
  `Vector{NTuple{D,TF}}`, where `D` is the dimension of Grid, and `TF` is the type of float.
- The returned `GeneralGrid` pairs the flattened values with their corresponding
  coordinates.

# Parameters
- `grid::StructuredGrid` : Structured grid object to be flattened.

# Returns
- `GeneralGrid{D, TF, Vector{TF}, Vector{NTuple{D, TF}}}` : A grid containing
  flat values and their coordinates.

# Notes
- The flattened `grid.grid` returned by `vec` shares the same underlying memory
  as the original `grid.grid`. No data copy occurs, so modifying one will also
  affect the other.
"""
function flatten(grid::StructuredGrid)
    coor = coordinate_grid(grid)
    return GeneralGrid(vec(grid.grid), vec(coor))
end

"""
    reduce_mean(grid::StructuredGrid{D,TF,V,A}, dim::Int=1) where {D,TF<:AbstractFloat,V<:AbstractVector{TF},A<:AbstractArray{TF,D}}

Average `grid.grid` along dimension `dim` and drop that dimension.
Axes and size are reduced accordingly.

# Parameters
- `grid::StructuredGrid{D,TF,V,A}` : Input structured grid.
- `dim::Int=1` : Dimension to average over (1-based).

# Returns
- `StructuredGrid{D-1,TF,V,A2}` : Structured grid with one fewer dimension, where `A2 <: AbstractArray{TF, D-1}`.

"""
function reduce_mean(grid::StructuredGrid{D,TF,V,A}, dim::Int=1) where {D,TF<:AbstractFloat,V<:AbstractVector{TF},A<:AbstractArray{TF,D}}
    1 ≤ dim ≤ D || throw(ArgumentError("dim must be in 1:$D, got $dim"))
    D == 1      && throw(ArgumentError("cannot reduce a 1D grid to 0D StructuredGrid"))

    # reduce values and drop the reduced dimension
    vals = reduce_mean(grid.grid, dim)

    # build new axes and size by removing the `dim`-th entry
    rem = ntuple(i -> (i < dim ? i : i + 1), D - 1)
    new_axes = ntuple(i -> grid.axes[rem[i]], D - 1)
    new_size = ntuple(i -> grid.size[rem[i]], D - 1)

    return StructuredGrid(vals, new_axes, new_size)
end

## Constructing
### Arbitrary input
"""
        GeneralGrid(x::V, y::V, z::V) where {T<:AbstractFloat, V<:AbstractVector{T}}

Construct a 3D `GeneralGrid` from particle coordinates. The input vectors are
copied into an array of position tuples and paired with a zero-initialised value
buffer, yielding a dense, order-preserving grid container that can be filled by
subsequent interpolation routines.

# Parameters
- `x, y, z::AbstractVector{T}`: Particle positions along each Cartesian axis.

# Returns
- `GeneralGrid{3, T, Vector{T}, Vector{NTuple{3, T}}}`: Grid with zeroed values
    and `(x, y, z)` coordinate tuples.
"""
function GeneralGrid(x :: V, y :: V, z :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    N = length(x)
    (length(y) == N) || throw(ArgumentError("y length mismatch"))
    (length(z) == N) || throw(ArgumentError("z length mismatch"))

    coords = Vector{NTuple{3, T}}(undef, N)
    @inbounds @simd for i in eachindex(x, y, z)
        coords[i] = (x[i], y[i], z[i])
    end
    vals = zeros(T, N)
    return GeneralGrid{3, T, Vector{T}, Vector{NTuple{3, T}}}(vals, coords)
end

"""
    GeneralGrid(x::V, y::V) where {T<:AbstractFloat, V<:AbstractVector{T}}

Construct a 2D `GeneralGrid` from planar coordinates. The returned grid stores
zeroed values and the `(x, y)` coordinate tuples, ready for interpolation or
resampling passes.

# Parameters
- `x, y::AbstractVector{T}`: Particle positions along each axis.

# Returns
- `GeneralGrid{2, T, Vector{T}, Vector{NTuple{2, T}}}`: Grid with zeroed values
  and `(x, y)` coordinate tuples.
"""
function GeneralGrid(x :: V, y :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    N = length(x)
    (length(y) == N) || throw(ArgumentError("y length mismatch"))

    coords = Vector{NTuple{2, T}}(undef, N)
    @inbounds @simd for i in eachindex(x, y)
        coords[i] = (x[i], y[i])
    end
    vals = zeros(T, N)
    return GeneralGrid{2, T, Vector{T}, Vector{NTuple{2, T}}}(vals, coords)
end

### Cartesian
"""
    StructuredGrid(::Type{Cartesian}, params::Vararg{AxisParam{TF}}) where {TF<:AbstractFloat}

Construct a Cartesian `StructuredGrid` from axis specifications.

Each axis is described by an `AxisParam{TF} = (xmin::TF, xmax::TF, n::Int)`,
where `xmin` and `xmax` define the interval and `n` is the number of grid points.  
All axes must share the same floating-point type `TF`.

# Parameters
- `params::Vararg{AxisParam{TF}}`  
  A list of axis definitions. The number of axes determines the dimension `D`.

# Returns
- `StructuredGrid{D,TF,Vector{TF},Array{TF,D}}` :  
  A structured grid with fields:
  - `grid` : zero-initialized `Array{TF,D}` of shape given by `(n₁, n₂, …, nD)`.
  - `axes` : `NTuple{D,Vector{TF}}`, each axis created by `collect(LinRange(...))`.
  - `size` : `NTuple{D,Int}`, storing grid dimensions.

# Notes
- Axes are stored as `Vector{TF}` (via `collect`) for compatibility with GPU
  adaptation and to ensure concrete vector storage.
- Passing mixed precision (e.g. `Float32` and `Float64` together) is not allowed.

# Examples
```julia
# 2D Cartesian grid: x ∈ [0,1] with 11 points, y ∈ [0,2] with 21 points
grid = StructuredGrid(Cartesian, (0.0, 1.0, 11), (0.0, 2.0, 21))

size(grid)        # (11, 21)
grid.axes[1][1:3] # [0.0, 0.1, 0.2]
```
"""
function StructuredGrid(::Type{Cartesian}, params::Vararg{AxisParam{TF}}) where {TF<:AbstractFloat}
    D    = length(params)
    sz   = ntuple(i -> params[i][3], D)
    axes = ntuple(i -> collect(LinRange{TF}(TF(params[i][1]), TF(params[i][2]), params[i][3])), D)
    vals = zeros(TF, sz...)
    return StructuredGrid{D, TF, Vector{TF}, Array{TF, D}}(vals, axes, sz)
end

# Polar (2D)
"""
    StructuredGrid(::Type{Polar}, sparams::AxisParam{TF}, ϕparams::AxisParam{TF}) where {TF<:AbstractFloat}

Construct a 2D polar `StructuredGrid` with radial `s`, angular `ϕ` (half-open [ϕmin, ϕmax)).

# Parameters
- `sparams::AxisParam{TF}` = `(smin::TF, smax::TF, ns::Int)`
- `ϕparams::AxisParam{TF}` = `(ϕmin::TF, ϕmax::TF, nϕ::Int)`  with `0 ≤ ϕmin < ϕmax ≤ 2π`

# Returns
- `StructuredGrid{2,TF,Vector{TF},Array{TF,2}}`
"""
function StructuredGrid(::Type{Polar}, sparams::AxisParam{TF}, ϕparams::AxisParam{TF}) where {TF<:AbstractFloat}
    smin, smax, ns = sparams
    ϕmin, ϕmax, nϕ = ϕparams

    (ns ≥ 1 && nϕ ≥ 1) || throw(ArgumentError("ns and nϕ must be ≥ 1"))
    (smin ≥ zero(TF) && smax > smin) || throw(ArgumentError("radial range must satisfy 0 ≤ smin < smax"))
    (ϕmin ≥ zero(TF) && (ϕmax ≤ TF(2π) || isapprox(ϕmax, TF(2π))) && ϕmax > ϕmin) ||
        throw(ArgumentError("angular range must satisfy 0 ≤ ϕmin < ϕmax ≤ 2π"))

    saxes = collect(LinRange{TF}(smin, smax, ns))

    # half-open [ϕmin, ϕmax): nϕ points, step = (ϕmax - ϕmin)/nϕ
    Δϕ = (ϕmax - ϕmin) / nϕ
    ϕaxes = collect(range(ϕmin, step=Δϕ, length=nϕ))

    axes = (saxes, ϕaxes)
    sz   = (ns, nϕ)
    vals = zeros(TF, sz...)
    return StructuredGrid{2, TF, Vector{TF}, Array{TF, 2}}(vals, axes, sz)
end

# Polar Slice (3D)
"""
    StructuredGrid(::Type{Cylindrical}, sparams::AxisParam{TF}, ϕparams::AxisParam{TF}, zconst::TF) where {TF<:AbstractFloat}

3D cylindrical grid slice at fixed `z = zconst`. Axes = `(s, ϕ, z)`, with `z` a length-1 axis.
`ϕ` uses half-open interval `[ϕmin, ϕmax)`.

# Parameters
- `sparams::AxisParam{TF}` = `(smin::TF, smax::TF, ns::Int)`
- `ϕparams::AxisParam{TF}` = `(ϕmin::TF, ϕmax::TF, nϕ::Int)`  with `0 ≤ ϕmin < ϕmax ≤ 2π`
- `zconst::TF`

# Returns
- `StructuredGrid{3,TF,Vector{TF},Array{TF,3}}`
"""
function StructuredGrid(::Type{Cylindrical}, sparams::AxisParam{TF}, ϕparams::AxisParam{TF}, zconst::TF) where {TF<:AbstractFloat}
    smin, smax, ns = sparams
    ϕmin, ϕmax, nϕ = ϕparams
    nz = 1

    (ns ≥ 1 && nϕ ≥ 1 && nz ≥ 1) || throw(ArgumentError("ns, nϕ, nz must be ≥ 1"))
    (smin ≥ zero(TF) && smax > smin) || throw(ArgumentError("radial range must satisfy 0 ≤ smin < smax"))
    (ϕmin ≥ zero(TF) && (ϕmax ≤ TF(2π) || isapprox(ϕmax, TF(2π))) && ϕmax > ϕmin) ||
        throw(ArgumentError("angular range must satisfy 0 ≤ ϕmin < ϕmax ≤ 2π"))

    saxes = collect(LinRange{TF}(smin, smax, ns))
    zaxes = TF[zconst]

    # half-open [ϕmin, ϕmax): nϕ points, step = (ϕmax - ϕmin)/nϕ
    Δϕ = (ϕmax - ϕmin) / nϕ
    ϕaxes = collect(range(ϕmin, step=Δϕ, length=nϕ))

    axes = (saxes, ϕaxes, zaxes)
    sz   = (ns, nϕ, nz)
    vals = zeros(TF, sz...)
    return StructuredGrid{3, TF, Vector{TF}, Array{TF, 3}}(vals, axes, sz)
end

# Cylindrical (3D)
"""
    StructuredGrid(::Type{Cylindrical}, sparams::AxisParam{TF}, ϕparams::AxisParam{TF}, zparams::AxisParam{TF}) where {TF<:AbstractFloat}

Construct a 3D cylindrical `StructuredGrid` with radial `s`, angular `ϕ` (half-open [ϕmin, ϕmax)),
and axial `z`.

# Parameters
- `sparams::AxisParam{TF}` = `(smin::TF, smax::TF, ns::Int)`
- `ϕparams::AxisParam{TF}` = `(ϕmin::TF, ϕmax::TF, nϕ::Int)`  with `0 ≤ ϕmin < ϕmax ≤ 2π`
- `zparams::AxisParam{TF}` = `(zmin::TF, zmax::TF, nz::Int)`

# Returns
- `StructuredGrid{3,TF,Vector{TF},Array{TF,3}}`
"""
function StructuredGrid(::Type{Cylindrical}, sparams::AxisParam{TF}, ϕparams::AxisParam{TF}, zparams::AxisParam{TF}) where {TF<:AbstractFloat}
    smin, smax, ns = sparams
    ϕmin, ϕmax, nϕ = ϕparams
    zmin, zmax, nz = zparams

    (ns ≥ 1 && nϕ ≥ 1 && nz ≥ 1) || throw(ArgumentError("ns, nϕ, nz must be ≥ 1"))
    (smin ≥ zero(TF) && smax > smin) || throw(ArgumentError("radial range must satisfy 0 ≤ smin < smax"))
    (ϕmin ≥ zero(TF) && (ϕmax ≤ TF(2π) || isapprox(ϕmax, TF(2π))) && ϕmax > ϕmin) ||
        throw(ArgumentError("angular range must satisfy 0 ≤ ϕmin < ϕmax ≤ 2π"))
    (zmax > zmin) || throw(ArgumentError("axial range must satisfy zmin < zmax"))

    saxes = collect(LinRange{TF}(smin, smax, ns))
    zaxes = collect(LinRange{TF}(zmin, zmax, nz))

    # half-open [ϕmin, ϕmax): nϕ points, step = (ϕmax - ϕmin)/nϕ
    Δϕ = (ϕmax - ϕmin) / nϕ
    ϕaxes = collect(range(ϕmin, step=Δϕ, length=nϕ))

    axes = (saxes, ϕaxes, zaxes)
    sz   = (ns, nϕ, nz)
    vals = zeros(TF, sz...)
    return StructuredGrid{3, TF, Vector{TF}, Array{TF, 3}}(vals, axes, sz)
end



# Spherical Shell (3D)
"""
    StructuredGrid(::Type{Spherical}, rconst::TF,
                   ϕparams::AxisParam{TF},
                   θparams::AxisParam{TF}) where {TF<:AbstractFloat}

Construct a spherical shell structured grid with fixed radius `r = rconst`
and angular axes `(ϕ, θ)`.

# Parameters
- `rconst::TF`  
  Constant radius of the spherical shell. Must satisfy `rconst ≥ 0`.

- `ϕparams::AxisParam{TF}`  
  Tuple `(ϕmin, ϕmax, nϕ)` defining azimuthal extent and number of points.  
  Constructed as half-open interval `[ϕmin, ϕmax)`, with spacing  
  `Δϕ = (ϕmax - ϕmin) / nϕ`. Must satisfy `0 ≤ ϕmin < ϕmax ≤ 2π`, `nϕ ≥ 1`.

- `θparams::AxisParam{TF}`  
  Tuple `(θmin, θmax, nθ)` defining polar angle (colatitude) range and number of points.  
  The grid is sampled uniformly in `cos(θ)` and mapped back by `acos`, producing 
  nearly equal-area spacing on the spherical shell. Must satisfy `0 ≤ θmin < θmax ≤ π`, `nθ ≥ 1`.

# Returns
- `StructuredGrid{3,TF,Vector{TF},Array{TF,3}}`  
  A 3D grid with:
  - `axes = ( [rconst], ϕaxes, θaxes )`  
  - `size = (1, nϕ, nθ)`  
  - `grid = zeros(TF, 1, nϕ, nθ)`

# Notes
- This constructor is for **spherical shells** (`r` fixed).  
- Azimuth `ϕ` is sampled on `[ϕmin, ϕmax)` to avoid duplication at `ϕmax = 2π`.  
- Polar angle `θ` is cos-sampled to prevent clustering near poles.
"""
function StructuredGrid(::Type{Spherical}, rconst :: TF, ϕparams::AxisParam{TF}, θparams::AxisParam{TF}) where {TF<:AbstractFloat}
    nr = 1
    ϕmin, ϕmax, nϕ = ϕparams
    θmin, θmax, nθ = θparams

    (nr ≥ 1 && nϕ ≥ 1 && nθ ≥ 1) ||
        throw(ArgumentError("nr, nϕ, nθ must be ≥ 1"))
    (rconst ≥ zero(TF)) ||
        throw(ArgumentError("radial range must satisfy r ≥ 0 "))
    (ϕmin ≥ zero(TF) && (ϕmax ≤ TF(2π) || isapprox(ϕmax, TF(2π))) && ϕmax > ϕmin) ||
        throw(ArgumentError("angular range must satisfy 0 ≤ ϕmin < ϕmax ≤ 2π"))
    (θmin ≥ zero(TF) && θmax ≤ TF(π) && θmax > θmin) ||
        throw(ArgumentError("polar range must satisfy 0 ≤ θmin < θmax ≤ π"))
    
    raxes = TF[rconst]
    # half-open [ϕmin, ϕmax): nϕ points, step = (ϕmax - ϕmin)/nϕ
    Δϕ = (ϕmax - ϕmin) / nϕ
    ϕaxes = collect(range(ϕmin, step=Δϕ, length=nϕ))

    μaxes = LinRange(cos(θmin), cos(θmax), nθ)  
    θaxes = acos.(μaxes)                       
    
    axes = (raxes, ϕaxes, θaxes)
    sz   = (nr, nϕ, nθ)
    vals = zeros(TF, sz...)
    return StructuredGrid{3, TF, Vector{TF}, Array{TF, 3}}(vals, axes, sz)
end


# Spherical (3D)
"""
    StructuredGrid(::Type{Spherical},
                   rparams::AxisParam{TF},
                   ϕparams::AxisParam{TF},
                   θparams::AxisParam{TF}) where {TF<:AbstractFloat}

Construct a spherical structured grid with axes `(r, ϕ, θ)`.

# Parameters
- `rparams::AxisParam{TF}`  
  Tuple `(rmin, rmax, nr)` defining radial extent and number of points.  
  Must satisfy `0 ≤ rmin < rmax`, `nr ≥ 1`.

- `ϕparams::AxisParam{TF}`  
  Tuple `(ϕmin, ϕmax, nϕ)` defining azimuthal extent and number of points.  
  Constructed as half-open interval `[ϕmin, ϕmax)`, with spacing  
  `Δϕ = (ϕmax - ϕmin) / nϕ`. Must satisfy `0 ≤ ϕmin < ϕmax ≤ 2π`, `nϕ ≥ 1`.

- `θparams::AxisParam{TF}`  
  Tuple `(θmin, θmax, nθ)` defining polar angle (colatitude) range and number of points.  
  The grid is sampled uniformly in `cos(θ)` and mapped back by `acos`, producing 
  nearly equal-area spacing on the sphere. Must satisfy `0 ≤ θmin < θmax ≤ π`, `nθ ≥ 1`.

# Returns
- `StructuredGrid{3,TF,Vector{TF},Array{TF,3}}`  
  A 3D spherical grid with:
  - `axes = (raxes, ϕaxes, θaxes)`  
  - `size = (nr, nϕ, nθ)`  
  - `grid = zeros(TF, nr, nϕ, nθ)`

# Notes
- `θ` is the polar angle (colatitude) measured from the +z axis, not latitude.  
- Azimuth `ϕ` is sampled on `[ϕmin, ϕmax)` to avoid duplication at `ϕmax = 2π`.  
- `θ` uses cos-sampling to avoid pole clustering.
"""
function StructuredGrid(::Type{Spherical}, rparams::AxisParam{TF}, ϕparams::AxisParam{TF}, θparams::AxisParam{TF}) where {TF<:AbstractFloat}
    rmin, rmax, nr = rparams
    ϕmin, ϕmax, nϕ = ϕparams
    θmin, θmax, nθ = θparams

    (nr ≥ 1 && nϕ ≥ 1 && nθ ≥ 1) ||
        throw(ArgumentError("nr, nϕ, nθ must be ≥ 1"))
    (rmin ≥ zero(TF) && rmax > rmin) ||
        throw(ArgumentError("radial range must satisfy 0 ≤ rmin < rmax"))
    (ϕmin ≥ zero(TF) && (ϕmax ≤ TF(2π) || isapprox(ϕmax, TF(2π))) && ϕmax > ϕmin) ||
        throw(ArgumentError("angular range must satisfy 0 ≤ ϕmin < ϕmax ≤ 2π"))
    (θmin ≥ zero(TF) && θmax ≤ TF(π) && θmax > θmin) ||
        throw(ArgumentError("polar range must satisfy 0 ≤ θmin < θmax ≤ π"))
    
    raxes = collect(LinRange{TF}(rmin, rmax, nr))
    # half-open [ϕmin, ϕmax): nϕ points, step = (ϕmax - ϕmin)/nϕ
    Δϕ = (ϕmax - ϕmin) / nϕ
    ϕaxes = collect(range(ϕmin, step=Δϕ, length=nϕ))

    μaxes = LinRange(cos(θmin), cos(θmax), nθ)  
    θaxes = acos.(μaxes)                       
    
    axes = (raxes, ϕaxes, θaxes)
    sz   = (nr, nϕ, nθ)
    vals = zeros(TF, sz...)
    return StructuredGrid{3, TF, Vector{TF}, Array{TF, 3}}(vals, axes, sz)
end