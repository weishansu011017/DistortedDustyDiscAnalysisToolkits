struct GeneralLineGrid{D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, VG}} <: AbstractGrid{TF}
    grid :: VG
    origin :: VC
    direction :: VC
end
function Adapt.adapt_structure(to, x :: GeneralLineGrid{D}) where {D}
    GeneralLineGrid(
        Adapt.adapt(to, x.grid),
        ntuple(i -> Adapt.adapt(to, x.origin[i]), D),
        ntuple(i -> Adapt.adapt(to, x.direction[i]), D),
    )
end

"""
    similar(grid::GeneralLineGrid)

Construct a new `GeneralLineGrid` with fresh storage for values but sharing
the same coordinate container as the input grid.

# Parameters
- `grid::GeneralLineGrid` : Template grid to copy structure from.

# Returns
- `GeneralLineGrid` : A grid with independent value storage (`grid.grid`)
  and shared coordinates (`grid.origin`, `grid.direction`).
"""
function Base.similar(grid::GeneralLineGrid)
    return GeneralLineGrid(similar(grid.grid), grid.origin, grid.direction)
end

"""
    similar(grid::GeneralLineGrid, itype :: Type{T}) where {T}

Construct a new `GeneralLineGrid` with fresh storage for values but sharing
the same coordinate container as the input grid and with values of type `T`.

Note that the coordinate type also changes accordingly.

# Parameters
- `grid::GeneralLineGrid` : Template grid to copy structure from.
- `itype::Type{T}` : Desired element type for the new grid's values.

# Returns
- `GeneralLineGrid` : A grid with independent value storage (`grid.grid`) of type `T`
  and with coordinates copied into new storage and converted to type `T`
"""
function Base.similar(grid::GeneralLineGrid{3,TF}, ::Type{T}) where {TF<:AbstractFloat,T<:AbstractFloat}
    new_grid = similar(grid.grid, T)
    new_origin = ntuple(i -> similar(grid.origin[i], T), 3)
    new_direction = ntuple(i -> similar(grid.direction[i], T), 3)

    @inbounds @simd for i in eachindex(new_grid)
        @inbounds begin
            xoi = grid.origin[1][i]
            yoi = grid.origin[2][i]
            zoi = grid.origin[3][i]

            xdi = grid.direction[1][i]
            ydi = grid.direction[2][i]
            zdi = grid.direction[3][i]

            new_origin[1][i] = xoi
            new_origin[2][i] = yoi
            new_origin[3][i] = zoi

            new_direction[1][i] = xdi
            new_direction[2][i] = ydi
            new_direction[3][i] = zdi
        end 
    end
    return GeneralLineGrid(new_grid, new_origin, new_direction)
end

"""
    similar(grid::GeneralLineGrid, itype :: Type{T}) where {T}

Construct a new `GeneralLineGrid` with fresh storage for values but sharing
the same coordinate container as the input grid and with values of type `T`.

Note that the coordinate type also changes accordingly.

# Parameters
- `grid::GeneralLineGrid` : Template grid to copy structure from.
- `itype::Type{T}` : Desired element type for the new grid's values.

# Returns
- `GeneralLineGrid` : A grid with independent value storage (`grid.grid`) of type `T`
  and with coordinates copied into new storage and converted to type `T`.
"""
function Base.similar(grid::GeneralLineGrid{2,TF}, ::Type{T}) where {TF<:AbstractFloat,T<:AbstractFloat}
    new_grid = similar(grid.grid, T)
    new_origin = ntuple(i -> similar(grid.origin[i], T), 2)
    new_direction = ntuple(i -> similar(grid.direction[i], T), 2)

    @inbounds @simd for i in eachindex(new_grid)
        @inbounds begin
            xoi = grid.origin[1][i]
            yoi = grid.origin[2][i]

            xdi = grid.direction[1][i]
            ydi = grid.direction[2][i]

            new_origin[1][i] = xoi
            new_origin[2][i] = yoi

            new_direction[1][i] = xdi
            new_direction[2][i] = ydi
        end 
    end
    return GeneralLineGrid(new_grid, new_origin, new_direction)
end

"""
    Base.isapprox(
        grid::GeneralLineGrid{D,TF},
        origin_axes::NTuple{D,<:AbstractVector},
        direction_axes::NTuple{D,<:AbstractVector};
        atol::Real = 1.0e-8,
        rtol::Real = 1.0e-8
    ) :: Bool where {D,TF<:AbstractFloat}

Check whether a `GeneralLineGrid` matches a tensor-product reference defined
by `origin_axes` and `direction_axes`, up to numerical tolerance.

This method interprets both the origin and direction fields of the grid as
being generated from separable (per-dimension) axes. The expected values are
constructed implicitly via tensor-product expansion:

- For each multi-index `I ∈ CartesianIndices(size_expected)`, where
  `size_expected[d] = length(origin_axes[d])`,
  the linear index `i = LinearIndices(size_expected)[I]` corresponds to:

    - Expected origin:
      `origin_expected[d] = origin_axes[d][I[d]]`

    - Expected direction:
      `direction_expected[d] = direction_axes[d][I[d]]`

The function returns `true` if and only if both:

- `grid.origin[d][i] ≈ origin_expected[d]`
- `grid.direction[d][i] ≈ direction_expected[d]`

hold for all `i` and all dimensions `d`, within the specified tolerances.

# Parameters
- `grid::GeneralLineGrid{D,TF}` :
  The grid whose origin and direction fields are to be validated.
- `origin_axes::NTuple{D,<:AbstractVector}` :
  Per-dimension axes defining the expected origin positions via tensor-product expansion.
- `direction_axes::NTuple{D,<:AbstractVector}` :
  Per-dimension axes defining the expected direction field via tensor-product expansion.

# Keyword Arguments
| Name         | Default  | Description                                      |
|--------------|----------|--------------------------------------------------|
| `atol::Real` | `1e-8`   | Absolute tolerance for floating-point comparison |
| `rtol::Real` | `1e-8`   | Relative tolerance for floating-point comparison |

# Returns
- `Bool` :
  `true` if both origin and direction fields match the expected tensor-product
  expansion within tolerance; otherwise `false`.
"""
function Base.isapprox(grid::GeneralLineGrid{D,TF}, origin_axes::NTuple{D,<:AbstractVector}, direction_axes::NTuple{D,<:AbstractVector}; atol::Real = 1.0e-8, rtol::Real = 1.0e-8)::Bool where {D,TF<:AbstractFloat}

    size_expected = ntuple(d -> length(origin_axes[d]), D)
    ntuple(d -> length(direction_axes[d]), D) == size_expected || return false

    N = prod(size_expected)

    length(grid) == N || return false
    @inbounds for d in 1:D
        length(grid.origin[d]) == N || return false
        length(grid.direction[d]) == N || return false
    end

    inds = CartesianIndices(size_expected)
    L = LinearIndices(size_expected)

    @inbounds for I in inds
        i = L[I]

        for d in 1:D
            origin_expected = origin_axes[d][I[d]]
            origin_actual   = grid.origin[d][i]

            direction_expected = direction_axes[d][I[d]]
            direction_actual   = grid.direction[d][i]

            if !isapprox(origin_actual, origin_expected; atol=atol, rtol=rtol) ||
               !isapprox(direction_actual, direction_expected; atol=atol, rtol=rtol)
                return false
            end
        end
    end

    return true
end

function Base.permute!(grid::GeneralLineGrid{D,TF}, p :: AbstractVector{TI}) where {D,TF<:AbstractFloat, TI<:Integer}
    Base.permute!(grid.grid, p)
    @inbounds for i in 1:D
        Base.permute!(grid.origin[i], p)
        Base.permute!(grid.direction[i], p)
    end
    return nothing
end

function Base.invpermute!(grid::GeneralLineGrid{D,TF}, p :: AbstractVector{TI}) where {D,TF<:AbstractFloat, TI<:Integer}
    Base.invpermute!(grid.grid, p)
    @inbounds for i in 1:D
        Base.invpermute!(grid.origin[i], p)
        Base.invpermute!(grid.direction[i], p)
    end
    return nothing
end

"""
    batch_GeneralLineGrid(grid::GeneralLineGrid{D, TF, VG, VC}, batch_size::Int)

Split a `GeneralLineGrid` into contiguous batches, each containing at most `batch_size` points.
The function preserves the ordering of both `grid.grid` and coordinates `grid.origin` and `grid.direction`, returning
a statically-sized `NTuple` of `GeneralLineGrid`s.

Batch `b` contains points from index
`start = (b - 1) * batch_size + 1` to `stop = min(b * batch_size, N)`,
where `N = length(grid)`.

# Parameters
- `grid::GeneralLineGrid{D, TF, VG, VC}` :
  Input grid containing values `grid.grid` and coordinates `grid.origin` and `grid.direction` in SoA layout
  (`grid.origin[d][i]` is the `d`-th coordinate of the `i`-th point).
- `batch_size::Int` :
  Maximum number of points in each batch.

# Returns
`NTuple{B, GeneralLineGrid{D, TF, VG, VC}}` where `B = cld(N, batch_size)` and `N = length(grid)`.

Each returned `GeneralLineGrid` contains:
- `grid.grid[start:stop]`
- `ntuple(d -> grid.origin[d][start:stop], D)`
- `ntuple(d -> grid.direction[d][start:stop], D)`

with `start:stop` defined by the batch index `b`.
"""
function batch_GeneralLineGrid(grid::GeneralLineGrid{D,TF,VG,VC}, batch_size::Int) where {D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D,VG}}
    npoints = length(grid)
    num_batches = cld(npoints, batch_size)

    return ntuple(b -> begin
        start = (b - 1) * batch_size + 1
        stop  = min(b * batch_size, npoints)

        grid_b = grid.grid[start:stop]
        origin_b = ntuple(d -> grid.origin[d][start:stop], D)
        direction_b = ntuple(d -> grid.direction[d][start:stop], D)

        GeneralLineGrid(grid_b, origin_b, direction_b)
    end, num_batches)
end

"""
    merge_GeneralLineGrid(grids::AbstractVector{<:GeneralLineGrid{D,TF,VG,VC}}) where {D,TF<:AbstractFloat,VG<:AbstractVector{TF},VC<:NTuple{D,VG}}

Merge a collection of `GeneralLineGrid` objects into a single `GeneralLineGrid`.

The function concatenates the scalar values `g.grid` in order, and concatenates
the line origins and directions in structure-of-arrays (SoA) form per dimension.
For each dimension `d = 1:D`, the merged coordinate vectors are constructed as

- `vcat(g.origin[d] for g in grids...)`
- `vcat(g.direction[d] for g in grids...)`

respectively.

This preserves the original sample ordering when `grids` was produced by
`batch_GeneralLineGrid` without any intervening reordering.

# Parameters
- `grids::AbstractVector{<:GeneralLineGrid{D,TF,VG,VC}}` :
  A vector of `GeneralLineGrid` objects to merge.

# Returns
- `GeneralLineGrid{D,TF,VG,VC}` :
  A single `GeneralLineGrid` whose fields are formed by concatenating:
  - `g.grid`
  - `g.origin[d]` for each dimension `d`
  - `g.direction[d]` for each dimension `d`

  across all input grids, in order.
"""
function merge_GeneralLineGrid(grids::V) where {D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D,VG}, GG<:GeneralLineGrid{D,TF,VG,VC}, V<:AbstractVector{GG}}
    merged_grid = vcat((g.grid for g in grids)...)
    merged_origin = ntuple(d -> vcat((g.origin[d] for g in grids)...), D)
    merged_direction = ntuple(d -> vcat((g.direction[d] for g in grids)...), D)

    return GeneralLineGrid(merged_grid, merged_origin, merged_direction)
end

# Constructors
function GeneralLineGrid(grid::VG, origin::VC, direction::VC) where {D,TF<:AbstractFloat,VG<:AbstractVector{TF},VC<:NTuple{D,VG}}
    N = length(grid)

    @inbounds for d in 1:D
        length(origin[d]) == N || throw(ArgumentError("origin[$d] length mismatch"))
        length(direction[d]) == N || throw(ArgumentError("direction[$d] length mismatch"))
    end

    return GeneralLineGrid{D,TF,VG,VC}(grid, origin, direction)
end

"""
    GeneralLineGrid(xo::V, yo::V, zo::V, xd::V, yd::V, zd::V) where {T<:AbstractFloat, V<:AbstractVector{T}}

Construct a 3D `GeneralLineGrid` with reference point and direction vectors. The input vectors are
stored in structure-of-arrays (SoA) form as `(xo, yo, zo)` and `(xd, yd, zd)` respectively, and paired with a zero-initialised value
buffer, yielding a dense, order-preserving grid container that can be filled by
subsequent interpolation routines.

# Parameters
- `xo, yo, zo::AbstractVector{T}`: Reference points along each Cartesian axis.
- `xd, yd, zd::AbstractVector{T}`: Direction vectors along each Cartesian axis.

# Returns
- `GeneralLineGrid{3, T, Vector{T}, NTuple{3, Vector{T}}}`: Grid with zeroed values
    and `(x, y, z)` coordinate tuples.
"""
function GeneralLineGrid(xo :: V, yo :: V, zo :: V, xd :: V, yd :: V, zd :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    N = length(xo)
    (length(yo) == N) || throw(ArgumentError("yo length mismatch"))
    (length(zo) == N) || throw(ArgumentError("zo length mismatch"))
    (length(xd) == N) || throw(ArgumentError("xd length mismatch"))
    (length(yd) == N) || throw(ArgumentError("yd length mismatch"))
    (length(zd) == N) || throw(ArgumentError("zd length mismatch"))

    origin = (xo, yo, zo)
    direction = (xd, yd, zd)
    vals = zeros(T, N)
    return GeneralLineGrid{3, T, Vector{T}, NTuple{3, Vector{T}}}(vals, origin, direction)
end

"""
    GeneralLineGrid(x::V, y::V) where {T<:AbstractFloat, V<:AbstractVector{T}}

Construct a 2D `GeneralLineGrid` from planar coordinates. The input vectors are
stored in structure-of-arrays (SoA) form as `(xo, yo)` and `(xd, yd)` respectively, and paired with a zero-initialised value
buffer, yielding a dense, order-preserving grid container that can be filled by
subsequent interpolation routines.

# Parameters
- `x, y::AbstractVector{T}`: Particle positions along each axis.

# Returns
- `GeneralLineGrid{2, T, Vector{T}, NTuple{2, Vector{T}}}`: Grid with zeroed values
  and `(x, y)` coordinate tuples.
"""
function GeneralLineGrid(x :: V, y :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    N = length(x)
    (length(y) == N) || throw(ArgumentError("y length mismatch"))

    origin = (x, y)
    direction = (zeros(T, N), zeros(T, N))
    vals = zeros(T, N)
    return GeneralLineGrid{2, T, Vector{T}, NTuple{2, Vector{T}}}(vals, origin, direction)
end
