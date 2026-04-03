"""
    PointSamples{D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D, VG}} <: AbstractSamples{TF}

A generic *unstructured* grid container for interpolation, storing grid values together with
their coordinates.

Coordinates are stored as a `D`-tuple of vectors, where each vector contains the coordinates
for one dimension. The tuple itself is immutable, so the coordinate container can be shared
across multiple grids without changing its structure.

# Type Parameters
- `D` : Dimensionality of the grid.
- `TF <: AbstractFloat` : Floating-point element type.
- `VG <: AbstractVector{TF}` : Storage type for grid values.
- `VC <: NTuple{D, VG}` : Storage type for coordinates (a `D`-tuple of vectors, each of type `VG`).

# Fields
- `grid :: VG` :
    Vector of grid values of length `N`.
- `coor :: VC` :
    Coordinates in SoA form. For each dimension `d ∈ 1:D`, `coor[d]` is a vector of length `N`,
    and `coor[d][i]` is the coordinate of the `i`-th grid point along dimension `d`.

"""
struct PointSamples{D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, VG}} <: AbstractSamples{D, TF}
    grid :: VG
    coor :: VC

    # Inner constructor
    function PointSamples(grid::VG, coor::VC) where {D,TF <: AbstractFloat,VG <: AbstractVector{TF},VC <: NTuple{D,VG}}
        N = length(grid)

        @inbounds for d in 1:D
            length(coor[d]) == N || throw(ArgumentError("coor[$d] length mismatch"))
        end

        return new{D,TF,VG,VC}(grid, coor)
    end
end

function Adapt.adapt_structure(to, x :: PointSamples{D}) where {D}
    PointSamples(
        Adapt.adapt(to, x.grid),
        ntuple(i -> Adapt.adapt(to, x.coor[i]), D)
    )
end

"""
    similar(grid::PointSamples)

Construct a new `PointSamples` with fresh storage for values but sharing
the same coordinate container as the input grid.

# Parameters
- `grid::PointSamples` : Template grid to copy structure from.

# Returns
- `PointSamples` : A grid with independent value storage (`grid.grid`)
  and shared coordinates (`grid.coor`).
"""
function Base.similar(grid::PointSamples)
    # Geometry is taken from grids[1] under the contract that `similar(::LineSamples)`
    # shares `coor` across all output grids.
    return PointSamples(similar(grid.grid), grid.coor)
end

"""
    similar(grid::PointSamples, itype :: Type{T}) where {T}

Construct a new `PointSamples` with fresh storage for values but sharing
the same coordinate container as the input grid and with values of type `T`.

Note that the coordinate type also changes accordingly.

# Parameters
- `grid::PointSamples` : Template grid to copy structure from.
- `itype::Type{T}` : Desired element type for the new grid's values.

# Returns
- `PointSamples` : A grid with independent value storage (`grid.grid`) of type `T`
  and shared coordinates (`grid.coor`) of type `NTuple{D, T}`.
"""
function Base.similar(grid::PointSamples{3,TF}, ::Type{T}) where {TF<:AbstractFloat,T<:AbstractFloat}
    new_grid = similar(grid.grid, T)
    new_coor = ntuple(i -> similar(grid.coor[i], T), 3)

    @inbounds @simd for i in eachindex(new_grid)
        @inbounds begin
            xi = grid.coor[1][i]
            yi = grid.coor[2][i]
            zi = grid.coor[3][i]

            new_coor[1][i] = xi
            new_coor[2][i] = yi
            new_coor[3][i] = zi
        end 
    end
    return PointSamples(new_grid, new_coor)
end

"""
    similar(grid::PointSamples, itype :: Type{T}) where {T}

Construct a new `PointSamples` with fresh storage for values but sharing
the same coordinate container as the input grid and with values of type `T`.

Note that the coordinate type also changes accordingly.

# Parameters
- `grid::PointSamples` : Template grid to copy structure from.
- `itype::Type{T}` : Desired element type for the new grid's values.

# Returns
- `PointSamples` : A grid with independent value storage (`grid.grid`) of type `T`
  and shared coordinates (`grid.coor`) of type `NTuple{D, T}`.
"""
function Base.similar(grid::PointSamples{2,TF}, ::Type{T}) where {TF<:AbstractFloat,T<:AbstractFloat}
    new_grid = similar(grid.grid, T)
    new_coor = ntuple(i -> similar(grid.coor[i], T), 2)

    @inbounds @simd for i in eachindex(new_grid)
        @inbounds begin
            xi = grid.coor[1][i]
            yi = grid.coor[2][i]

            new_coor[1][i] = xi
            new_coor[2][i] = yi
        end 
    end
    return PointSamples(new_grid, new_coor)
end

"""
    Base.isapprox(grid::PointSamples{D,TF}, axes::NTuple{D,<:AbstractVector}; atol::Real=1.0e-8, rtol::Real=1.0e-8) :: Bool where {D,TF<:AbstractFloat}

Check whether the coordinates stored in a `PointSamples` match the given `axes`
(up to numerical tolerance).

# Parameters
- `grid::PointSamples{D,TF}` : Grid whose coordinates will be checked.
- `axes::NTuple{D,<:AbstractVector}` : Target coordinate axes.

# Keyword Arguments
| Name            | Default  | Description                                    |
|-----------------|----------|------------------------------------------------|
| `atol::Real`    | `1.0e-8` | Absolute tolerance for floating-point comparison. |
| `rtol::Real`    | `1.0e-8` | Relative tolerance for floating-point comparison. |

# Returns
- `Bool` : `true` if all coordinates match within tolerance, otherwise `false`.
"""
function Base.isapprox(grid::PointSamples{D,TF}, axes::NTuple{D,<:AbstractVector}; atol :: Real = 1.0e-8, rtol :: Real = 1.0e-8) :: Bool where {D,TF <: AbstractFloat}
    size_expected = ntuple(d -> length(axes[d]), D)
    N = prod(size_expected)

    length(grid) == N || return false
    @inbounds for d in 1:D
        length(grid.coor[d]) == N || return false
    end

    inds = CartesianIndices(size_expected)
    L = LinearIndices(size_expected)

    @inbounds for I in inds
        i = L[I]

        @inbounds @simd for d in 1:D
            val_expected = axes[d][I[d]]
            val_actual   = grid.coor[d][i]
            if !isapprox(val_actual, val_expected; atol=atol, rtol=rtol)
                return false
            end
        end
    end

    return true
end

function Base.permute!(grid::PointSamples{D,TF}, p :: AbstractVector{TI}) where {D,TF<:AbstractFloat, TI<:Integer}
    Base.permute!(grid.grid, p)
    @inbounds for i in 1:D
        Base.permute!(grid.coor[i], p)
    end
    return nothing
end

function Base.invpermute!(grid::PointSamples{D,TF}, p :: AbstractVector{TI}) where {D,TF<:AbstractFloat, TI<:Integer}
    Base.invpermute!(grid.grid, p)
    @inbounds for i in 1:D
        Base.invpermute!(grid.coor[i], p)
    end
    return nothing
end

"""
    batch_PointSamples(grid::PointSamples{D, TF, VG, VC}, batch_size::Int)

Split a `PointSamples` into contiguous batches, each containing at most `batch_size` points.
The function preserves the ordering of both `grid.grid` and coordinates `grid.coor`, returning
a statically-sized `NTuple` of `PointSamples`s.

Batch `b` contains points from index
`start = (b - 1) * batch_size + 1` to `stop = min(b * batch_size, N)`,
where `N = length(grid)`.

# Parameters
- `grid::PointSamples{D, TF, VG, VC}` :
  Input grid containing values `grid.grid` and coordinates `grid.coor` in SoA layout
  (`grid.coor[d][i]` is the `d`-th coordinate of the `i`-th point).
- `batch_size::Int` :
  Maximum number of points in each batch.

# Returns
`NTuple{B, PointSamples{D, TF, VG, VC}}` where `B = cld(N, batch_size)` and `N = length(grid)`.

Each returned `PointSamples` contains:
- `grid.grid[start:stop]`
- `ntuple(d -> grid.coor[d][start:stop], D)`

with `start:stop` defined by the batch index `b`.
"""
function batch_PointSamples(grid::PointSamples{D,TF,VG,VC}, batch_size::Int) where {D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D,VG}}
    npoints = length(grid)
    num_batches = cld(npoints, batch_size)

    return ntuple(b -> begin
        start = (b - 1) * batch_size + 1
        stop  = min(b * batch_size, npoints)

        grid_b = grid.grid[start:stop]
        coor_b = ntuple(d -> grid.coor[d][start:stop], D)

        PointSamples(grid_b, coor_b)
    end, num_batches)
end

"""
    merge_PointSamples(grids::AbstractVector{<:PointSamples{D,TF,VG,VC}})

Merge a collection of `PointSamples` batches into a single `PointSamples`.

The function concatenates all scalar values `g.grid` in order, and concatenates
coordinates in SoA form per dimension: for each `d = 1:D`, the merged coordinate
vector is `vcat(g.coor[d] for g in grids...)`. This restores the original point
ordering when `grids` is produced by `batch_PointSamples` without reordering.

# Parameters
- `grids::AbstractVector{<:PointSamples{D,TF,VG,VC}}` :
  A vector of batched `PointSamples` objects.

# Returns
`PointSamples{D,TF,VG,VC}` containing the concatenated values and coordinates.
"""
function merge_PointSamples(grids::V) where {D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D,VG}, GG<:PointSamples{D,TF,VG,VC}, V<:AbstractVector{GG}}
    merged_grid = vcat((g.grid for g in grids)...)
    merged_coor = ntuple(d -> vcat((g.coor[d] for g in grids)...), D)

    return PointSamples(merged_grid, merged_coor)
end


# Constructors
"""
    PointSamples(x::V, y::V, z::V) where {T<:AbstractFloat, V<:AbstractVector{T}}

Construct a 3D `PointSamples` from particle coordinates. The input vectors are
copied into an array of position tuples and paired with a zero-initialised value
buffer, yielding a dense, order-preserving grid container that can be filled by
subsequent interpolation routines.

# Parameters
- `x, y, z::AbstractVector{T}`: Particle positions along each Cartesian axis.

# Returns
- `PointSamples{3, T, Vector{T}, NTuple{3, Vector{T}}}`: Grid with zeroed values
    and `(x, y, z)` coordinate tuples.
"""
function PointSamples(x :: V, y :: V, z :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    N = length(x)
    (length(y) == N) || throw(ArgumentError("y length mismatch"))
    (length(z) == N) || throw(ArgumentError("z length mismatch"))

    coords = (x, y, z)
    vals = zeros(T, N)
    return PointSamples(vals, coords)
end

"""
    PointSamples(x::V, y::V) where {T<:AbstractFloat, V<:AbstractVector{T}}

Construct a 2D `PointSamples` from planar coordinates. The returned grid stores
zeroed values and the `(x, y)` coordinate tuples, ready for interpolation or
resampling passes.

# Parameters
- `x, y::AbstractVector{T}`: Particle positions along each axis.

# Returns
- `PointSamples{2, T, Vector{T}, NTuple{2, Vector{T}}}`: Grid with zeroed values
  and `(x, y)` coordinate tuples.
"""
function PointSamples(x :: V, y :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    N = length(x)
    (length(y) == N) || throw(ArgumentError("y length mismatch"))

    coords = (x, y)
    vals = zeros(T, N)
    return PointSamples(vals, coords)
end
