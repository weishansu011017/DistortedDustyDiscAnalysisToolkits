
"""
    LineSamples{D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D, VG}} <: AbstractSamples{D, TF}

A generic *unstructured* sample container for interpolation, storing grid values together with
line primitives in structure-of-arrays (SoA) form.

Each sample is associated with one line, represented by an origin point and a direction vector.
Both `origin` and `direction` are stored as `D`-tuples of vectors, where each vector contains
the corresponding component for all samples. The tuples themselves are immutable, so the
geometric container structure can be shared across multiple sample objects without changing
its layout.

# Type Parameters
- `D` : Dimensionality of the embedding space.
- `TF <: AbstractFloat` : Floating-point element type.
- `VG <: AbstractVector{TF}` : Storage type for sample values.
- `VC <: NTuple{D, VG}` : Storage type for line origins and directions (a `D`-tuple of vectors,
  each of type `VG`).

# Fields
- `grid :: VG` :
    Vector of sample values of length `N`.
- `origin :: VC` :
    Line origins in SoA form. For each dimension `d ∈ 1:D`, `origin[d]` is a vector of length `N`,
    and `origin[d][i]` is the `d`-th component of the origin of the `i`-th line sample.
- `direction :: VC` :
    Line directions in SoA form. For each dimension `d ∈ 1:D`, `direction[d]` is a vector of length `N`,
    and `direction[d][i]` is the `d`-th component of the direction of the `i`-th line sample.
"""
struct LineSamples{D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, VG}} <: AbstractSamples{D, TF}
    grid :: VG
    origin :: VC
    direction :: VC

    # Inner constructor
    function LineSamples(grid::VG, origin::VC, direction::VC) where {D,TF<:AbstractFloat,VG<:AbstractVector{TF},VC<:NTuple{D,VG}}
        N = length(grid)

        @inbounds for d in 1:D
            length(origin[d]) == N || throw(ArgumentError("origin[$d] length mismatch"))
            length(direction[d]) == N || throw(ArgumentError("direction[$d] length mismatch"))
        end

        return new{D,TF,VG,VC}(grid, origin, direction)
    end
end

function Adapt.adapt_structure(to, x :: LineSamples{D}) where {D}
    LineSamples(
        Adapt.adapt(to, x.grid),
        ntuple(i -> Adapt.adapt(to, x.origin[i]), D),
        ntuple(i -> Adapt.adapt(to, x.direction[i]), D),
    )
end

"""
    similar(grid::LineSamples)

Construct a new `LineSamples` with fresh storage for sample values while
sharing the same geometric containers as the input object.

The returned object allocates a new `grid` vector, but reuses
`grid.origin` and `grid.direction` without copying them.

# Parameters
- `grid::LineSamples` :
  Template sample container whose geometric layout is reused.

# Returns
- `LineSamples` :
  A new `LineSamples` object with independent value storage and shared
  origin/direction containers.
"""
function Base.similar(grid::LineSamples)
    # Geometry is taken from grids[1] under the contract that `similar(::LineSamples)`
    # shares `origin` and `direction` across all output grids.
    return LineSamples(similar(grid.grid), grid.origin, grid.direction)
end

"""
    similar(grid::LineSamples, ::Type{T}) where {T<:AbstractFloat}

Construct a new `LineSamples` with value storage of element type `T`, and with
the origin and direction fields copied into newly allocated storage of the same
element type.

Unlike `similar(grid::LineSamples)`, this method does not share the geometric
containers. Instead, both `origin` and `direction` are reallocated and their
contents are copied after conversion to `T`.

# Parameters
- `grid::LineSamples` :
  Template sample container whose value and geometric data are used to initialise
  the new object.
- `::Type{T}` :
  Target floating-point element type for the returned value, origin, and direction
  arrays.

# Returns
- `LineSamples` :
  A new `LineSamples` object whose `grid`, `origin`, and `direction` fields are
  stored in newly allocated arrays with element type `T`.
"""
function Base.similar(grid::LineSamples{3,TF}, ::Type{T}) where {TF<:AbstractFloat,T<:AbstractFloat}
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
    return LineSamples(new_grid, new_origin, new_direction)
end

"""
    similar(grid::LineSamples, ::Type{T}) where {T<:AbstractFloat}

Construct a new `LineSamples` with value storage of element type `T`, and with
the origin and direction fields copied into newly allocated storage of the same
element type.

Unlike `similar(grid::LineSamples)`, this method does not share the geometric
containers. Instead, both `origin` and `direction` are reallocated and their
contents are copied after conversion to `T`.

# Parameters
- `grid::LineSamples` :
  Template sample container whose value and geometric data are used to initialise
  the new object.
- `::Type{T}` :
  Target floating-point element type for the returned value, origin, and direction
  arrays.

# Returns
- `LineSamples` :
  A new `LineSamples` object whose `grid`, `origin`, and `direction` fields are
  stored in newly allocated arrays with element type `T`.
"""
function Base.similar(grid::LineSamples{2,TF}, ::Type{T}) where {TF<:AbstractFloat,T<:AbstractFloat}
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
    return LineSamples(new_grid, new_origin, new_direction)
end

"""
    Base.isapprox(
        grid::LineSamples{D,TF},
        origin_axes::NTuple{D,<:AbstractVector},
        direction_axes::NTuple{D,<:AbstractVector};
        atol::Real = 1.0e-8,
        rtol::Real = 1.0e-8
    ) :: Bool where {D,TF<:AbstractFloat}

Check whether a `LineSamples` matches a tensor-product reference defined
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
- `grid::LineSamples{D,TF}` :
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
function Base.isapprox(grid::LineSamples{D,TF}, origin_axes::NTuple{D,<:AbstractVector}, direction_axes::NTuple{D,<:AbstractVector}; atol::Real = 1.0e-8, rtol::Real = 1.0e-8)::Bool where {D,TF<:AbstractFloat}

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

function Base.permute!(grid::LineSamples{D,TF}, p :: AbstractVector{TI}) where {D,TF<:AbstractFloat, TI<:Integer}
    Base.permute!(grid.grid, p)
    @inbounds for i in 1:D
        Base.permute!(grid.origin[i], p)
        Base.permute!(grid.direction[i], p)
    end
    return nothing
end

function Base.invpermute!(grid::LineSamples{D,TF}, p :: AbstractVector{TI}) where {D,TF<:AbstractFloat, TI<:Integer}
    Base.invpermute!(grid.grid, p)
    @inbounds for i in 1:D
        Base.invpermute!(grid.origin[i], p)
        Base.invpermute!(grid.direction[i], p)
    end
    return nothing
end

"""
    batch_LineSamples(grid::LineSamples{D, TF, VG, VC}, batch_size::Int)

Split a `LineSamples` into contiguous batches, each containing at most `batch_size` points.
The function preserves the ordering of both `grid.grid` and coordinates `grid.origin` and `grid.direction`, returning
a statically-sized `NTuple` of `LineSamples`s.

Batch `b` contains points from index
`start = (b - 1) * batch_size + 1` to `stop = min(b * batch_size, N)`,
where `N = length(grid)`.

# Parameters
- `grid::LineSamples{D, TF, VG, VC}` :
  Input grid containing values `grid.grid` and coordinates `grid.origin` and `grid.direction` in SoA layout
  (`grid.origin[d][i]` is the `d`-th coordinate of the `i`-th point).
- `batch_size::Int` :
  Maximum number of points in each batch.

# Returns
`NTuple{B, LineSamples{D, TF, VG, VC}}` where `B = cld(N, batch_size)` and `N = length(grid)`.

Each returned `LineSamples` contains:
- `grid.grid[start:stop]`
- `ntuple(d -> grid.origin[d][start:stop], D)`
- `ntuple(d -> grid.direction[d][start:stop], D)`

with `start:stop` defined by the batch index `b`.
"""
function batch_LineSamples(grid::LineSamples{D,TF,VG,VC}, batch_size::Int) where {D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D,VG}}
    npoints = length(grid)
    num_batches = cld(npoints, batch_size)

    return ntuple(b -> begin
        start = (b - 1) * batch_size + 1
        stop  = min(b * batch_size, npoints)

        grid_b = grid.grid[start:stop]
        origin_b = ntuple(d -> grid.origin[d][start:stop], D)
        direction_b = ntuple(d -> grid.direction[d][start:stop], D)

        LineSamples(grid_b, origin_b, direction_b)
    end, num_batches)
end

"""
    merge_LineSamples(grids::AbstractVector{<:LineSamples{D,TF,VG,VC}}) where {D,TF<:AbstractFloat,VG<:AbstractVector{TF},VC<:NTuple{D,VG}}

Merge a collection of `LineSamples` objects into a single `LineSamples`.

The function concatenates the scalar values `g.grid` in order, and concatenates
the line origins and directions in structure-of-arrays (SoA) form per dimension.
For each dimension `d = 1:D`, the merged coordinate vectors are constructed as

- `vcat(g.origin[d] for g in grids...)`
- `vcat(g.direction[d] for g in grids...)`

respectively.

This preserves the original sample ordering when `grids` was produced by
`batch_LineSamples` without any intervening reordering.

# Parameters
- `grids::AbstractVector{<:LineSamples{D,TF,VG,VC}}` :
  A vector of `LineSamples` objects to merge.

# Returns
- `LineSamples{D,TF,VG,VC}` :
  A single `LineSamples` whose fields are formed by concatenating:
  - `g.grid`
  - `g.origin[d]` for each dimension `d`
  - `g.direction[d]` for each dimension `d`

  across all input grids, in order.
"""
function merge_LineSamples(grids::V) where {D, TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D,VG}, GG<:LineSamples{D,TF,VG,VC}, V<:AbstractVector{GG}}
    merged_grid = vcat((g.grid for g in grids)...)
    merged_origin = ntuple(d -> vcat((g.origin[d] for g in grids)...), D)
    merged_direction = ntuple(d -> vcat((g.direction[d] for g in grids)...), D)

    return LineSamples(merged_grid, merged_origin, merged_direction)
end

# Constructors
"""
    LineSamples(xo::V, yo::V, zo::V, xd::V, yd::V, zd::V) where {T<:AbstractFloat, V<:AbstractVector{T}}

Construct a 3D `LineSamples` object from per-sample line origins and direction
vectors.

The returned object stores sample values in a newly allocated zero-initialised
vector, and stores the geometric data in structure-of-arrays (SoA) form as

- `origin = (xo, yo, zo)`
- `direction = (xd, yd, zd)`

# Parameters
- `xo, yo, zo :: AbstractVector{T}` :
  The `x`-, `y`-, and `z`-components of the line origins.
- `xd, yd, zd :: AbstractVector{T}` :
  The `x`-, `y`-, and `z`-components of the line directions.

# Returns
- `LineSamples{3, T, Vector{T}, NTuple{3, Vector{T}}}` :
  A 3D `LineSamples` object with zero-initialised sample values and SoA-form
  origin and direction fields.
"""
function LineSamples(xo :: V, yo :: V, zo :: V, xd :: V, yd :: V, zd :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    N = length(xo)
    (length(yo) == N) || throw(ArgumentError("yo length mismatch"))
    (length(zo) == N) || throw(ArgumentError("zo length mismatch"))
    (length(xd) == N) || throw(ArgumentError("xd length mismatch"))
    (length(yd) == N) || throw(ArgumentError("yd length mismatch"))
    (length(zd) == N) || throw(ArgumentError("zd length mismatch"))

    origin = (xo, yo, zo)
    direction = (xd, yd, zd)
    vals = zeros(T, N)
    return LineSamples(vals, origin, direction)
end

"""
    LineSamples(xo::V, yo::V, xd::V, yd::V) where {T<:AbstractFloat, V<:AbstractVector{T}}

Construct a 2D `LineSamples` object from per-sample line origins and direction
vectors.

The returned object stores sample values in a newly allocated zero-initialised
vector, and stores the geometric data in structure-of-arrays (SoA) form as

- `origin = (xo, yo)`
- `direction = (xd, yd)`

# Parameters
- `xo, yo :: AbstractVector{T}` :
  The `x`- and `y`-components of the line origins.
- `xd, yd :: AbstractVector{T}` :
  The `x`- and `y`-components of the line directions.

# Returns
- `LineSamples{2, T, Vector{T}, NTuple{2, Vector{T}}}` :
  A 2D `LineSamples` object with zero-initialised sample values and SoA-form
  origin and direction fields.
"""
function LineSamples(xo :: V, yo :: V, xd :: V, yd :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    N = length(xo)
    (length(yo) == N) || throw(ArgumentError("yo length mismatch"))
    (length(xd) == N) || throw(ArgumentError("xd length mismatch"))
    (length(yd) == N) || throw(ArgumentError("yd length mismatch"))

    origin = (xo, yo)
    direction = (xd, yd)
    vals = zeros(T, N)
    return LineSamples(vals, origin, direction)
end
