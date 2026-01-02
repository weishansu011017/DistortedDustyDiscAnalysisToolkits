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

# structured grid (Cartesian/Cylindrical... etc)
"""
    StructuredGrid{D, TF<:AbstractFloat, V<:AbstractVector{TF}, A<:AbstractArray{TF,D}} <: AbstractGrid{TF}

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
struct StructuredGrid{D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}} <: AbstractGrid{TF}
    grid :: A
    axes :: NTuple{D, V}
    size :: NTuple{D, Int}
end

function Adapt.adapt_structure(to, x :: SG) where {D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}, SG <: StructuredGrid{D, TF, V, A}}
    StructuredGrid(
        Adapt.adapt(to, x.grid),
        ntuple(i -> Adapt.adapt(to, x.axes[i]), D),
        x.size
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

Generate coordinates for all grid points defined by a `StructuredGrid`.

Coordinates are returned in a structure-of-arrays (SoA) layout compatible with `GeneralGrid`.
For each dimension `d = 1:D`, the returned vector `coor[d]` has length `N = prod(grid.size)`,
and `coor[d][i]` is the `d`-th coordinate of the `i`-th grid point, where `i` follows Julia's
column-major linear indexing of `grid.size`.

# Parameters
- `grid::StructuredGrid{D,TF}` :
  The structured grid container.

# Returns
`NTuple{D, AbstractVector{TF}}`:
- For each `d = 1:D`, `coor[d]` is a vector of length `N = prod(grid.size)`.
- The linear index `i` is consistent with `vec(grid.grid)`.
"""
function coordinate_grid(grid::StructuredGrid{D,TF}) where {D,TF<:AbstractFloat}
    sz = grid.size
    gv = vec(grid.grid)
    coor = ntuple(_ -> similar(gv), D)

    L = LinearIndices(sz)

    @inbounds for I in CartesianIndices(sz)
        i = L[I]
        @inbounds @simd for d in 1:D
            coor[d][i] = grid.axes[d][I[d]]
        end
    end

    return coor
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

# Constructors
## Cartesian
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

## Polar (2D)
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

## Polar Slice (3D)
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

## Cylindrical (3D)
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



## Spherical Shell (3D)
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


## Spherical (3D)
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