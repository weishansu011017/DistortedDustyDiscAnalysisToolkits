"""
The grid construction for SPH interpolation
    by Wei-Shan Su,
    June 21, 2024
"""

"""
    struct gridbackend{D, T <: AbstractFloat, V <: AbstractVector{T}, A <: AbstractArray{T, D}}

A structure for storing grid-based scalar or vector field data and its corresponding coordinate axes.

This structure packages a multi-dimensional grid (`grid`) together with its axis definitions (`axes`) and resolution (`dimension`). It is designed to be compatible with both CPU and GPU backends (e.g., `Array`, `CuArray`), and can be adapted using `Adapt.jl` for GPU kernel usage.

# Fields
- `grid::A`  
  The underlying D-dimensional data array, typically an `Array{T,D}` or `CuArray{T,D}`.

- `axes::NTuple{D, V}`  
  A tuple of coordinate vectors, each representing the grid points along one dimension. Each element is a 1D array (e.g., `LinRange`, `Vector`, or `CuVector`), aligned with the grid shape.

- `dimension::NTuple{D, Int}`  
  The number of divisions (i.e., resolution) in each dimension of the grid, usually equal to `size(grid)`.

# Notes
- The type parameters are:
  - `D`: Dimensionality of the grid (e.g., 2 for 2D, 3 for 3D)
  - `T`: Element type (e.g., `Float32`, `Float64`)
  - `V`: Vector type used in axes (e.g., `Vector{T}`, `SVector{N, T}`, `CuVector{T}`)
  - `A`: Array type used in grid (e.g., `Array{T, D}`, `CuArray{T, D}`)

"""
struct gridbackend{D, T <: AbstractFloat, V <: AbstractVector{T}, A <: AbstractArray{T, D}}
    grid :: A
    axes :: NTuple{D, V}
    dimension::NTuple{D, Int}
end


"""
    function meshgrid(arrays::AbstractVector...)
Mesh multiple arrays, get the meshgrid arrays

# Parameters
- `arrays :: AbstractVector`: The array that would be meshed. Can applied any number of arrays.

# Returns
- `Tuple`: The vector that contains all of the meshgrids

# Examples
```julia
x = LinRange(0,10,250)
y = LinRange(0,2π,301)
X, Y = meshgrid(x,y)
```
"""
function meshgrid(arrays::AbstractVector...)
    nd = length(arrays)
    grids = ntuple(i->repeat(
            reshape(arrays[i], ntuple(d->d==i ? length(arrays[i]) : 1,nd)...),
            ntuple(d->d==i ? 1 : length(arrays[d]), nd)...
        ), nd)
    return grids 
end

"""
    function meshgrid(gbe::gridbackend)
Mesh multiple arrays from 'gridbackend', get the meshgrid arrays

# Parameters
- `gbe :: gridbackend`: The 'gridbackend' that contain all of the axes information

# Returns
- `Tuple`: The tuple that contains all of the meshgrids

# Examples
```julia
x = LinRange(0,10,250)
y = LinRange(0,2π,301)
X, Y = meshgrid(x,y)
```
"""
function meshgrid(gbe::gridbackend)
    iaxes = gbe.axes
    meshgrids = meshgrid(iaxes...)
    return meshgrids
end

"""
    generate_empty_grid(imin, imax, iN; FloatType=Float64, VectorType=Vector, ArrayType=Array)

Construct an empty structured grid and its coordinate axes.

This function generates a D-dimensional grid structure with associated coordinate axes.
It returns a `gridbackend{D, T, V, A}` object, where:
- `grid` is an array of zeros with the specified size,
- `axes` is a tuple of D 1D coordinate vectors spanning the range from `imin[d]` to `imax[d]`,
- `dimension` stores the size of each axis.

The method accepts either tuples or vectors for input.

# Parameters
- `imin::NTuple{D, <:Real}` or `AbstractVector{<:Real}`  
  The lower bound of each axis.
- `imax::NTuple{D, <:Real}` or `AbstractVector{<:Real}`  
  The upper bound of each axis.
- `iN::NTuple{D, <:Integer}` or `AbstractVector{<:Integer}`  
  The number of grid points along each axis.

# Keyword Arguments
| Name         | Default     | Description                                  |
|--------------|-------------|----------------------------------------------|
| `FloatType`  | `Float64`   | Floating-point type for all grid values.     |
| `VectorType` | `Vector`    | Type constructor for 1D coordinate vectors.  |
| `ArrayType`  | `Array`     | Type constructor for the full grid array.    |

# Returns
- `gridbackend{D, FloatType, VectorType{FloatType}, ArrayType{FloatType,D}}`  
  An initialized `gridbackend` object containing zero-initialized data and associated coordinate axes.

# Notes
- The NTuple variant will fail at compile-time if the dimensions are inconsistent.
- The Vector variant performs a runtime check for matching lengths.
- The grid values are initialized with `zero(FloatType)`.

# Example
```julia
imin = (0.0, 0.0)
imax = (1.0, 1.0)
iN   = (100, 200)
g    = generate_empty_grid(imin, imax, iN; FloatType=Float32, VectorType=SVector, ArrayType=CuArray)
```
"""
function generate_empty_grid(imin :: NTuple{D, <:Real}, imax :: NTuple{D, <:Real}, iN :: NTuple{D, <:Integer}; 
    FloatType::Type{<:AbstractFloat} = Float64, VectorType::Type{<:AbstractVector} = Vector, ArrayType::Type{<:AbstractArray} = Array) where {D}
    dimension = ntuple(i -> iN[i], D)
    VT :: Type = VectorType{FloatType}
    AT :: Type = ArrayType{FloatType, D}
    iaxes :: NTuple{D, VT} = ntuple(i -> VT(collect(LinRange(imin[i], imax[i], iN[i]))),D)
    grid = ArrayType{FloatType, D}(undef, dimension...)
    fill!(grid, zero(FloatType))
    return gridbackend{D, FloatType, VT, AT}(grid, iaxes, dimension)
end

function generate_empty_grid(imin :: AbstractVector{<:Real}, imax :: AbstractVector{<:Real}, iN :: AbstractVector{Int}; 
    FloatType::Type{<:AbstractFloat} = Float64, VectorType::Type{<:AbstractVector} = Vector, ArrayType::Type{<:AbstractArray} = Array)
    if length(imin) == length(imax) && length(imax) == length(iN)
        nothing
    else
        error("GridGeneratingError: Illegal input value.")
    end
    D = length(iN)
    iminT = ntuple(i -> imin[i],D)
    imaxT = ntuple(i -> imax[i],D)
    iNT = ntuple(i -> iN[i],D)
    return generate_empty_grid(iminT, imaxT, iNT, FloatType=FloatType, VectorType=VectorType, ArrayType=ArrayType)
end

"""
    generate_empty_grid(iaxes::NTuple{D, V}, ::Type{T}, ::Type{A}) -> gridbackend{D, T, V, A}

Generate an empty structured grid initialized with zeros based on provided axes and array types.

This function creates a `gridbackend` struct with specified axis definitions and underlying array type.
Each axis is represented by an `AbstractVector{T}` (e.g., `LinRange` or `SVector`), and the data array
(`grid`) is allocated using the given `ArrayType`, filled with zeros of type `T`.

# Parameters
- `iaxes::NTuple{D, V}`  
  A tuple of 1D coordinate arrays for each dimension. Each element must be a vector of type `V <: AbstractVector{T}`.
- `::Type{T}`  
  The floating-point type used for both the grid and axes (e.g., `Float64` or `Float32`).
- `::Type{A}`  
  The concrete array type for the data grid, e.g., `Array{T,D}` or `CuArray{T,D}`.

# Returns
- `::gridbackend{D, T, V, A}`  
  A structured grid container containing the data array and axes metadata.

# Example
```julia
iaxes = ntuple(i -> collect(range(0.0, 1.0; length=32)), 3)
grid = generate_empty_grid(iaxes, Float64, Array{Float64, 3})
```
"""
function generate_empty_grid(iaxes::NTuple{D, V}, ::Type{T}, ::Type{A}) where {D, T <: AbstractFloat, V <: AbstractVector{T}, A <: AbstractArray{T, D}}
    dimension = ntuple(i -> length(iaxes[i]), D)
    grid = A(undef, dimension...)
    fill!(grid, zero(T))
    return gridbackend{D, T, V, A}(grid, iaxes, dimension)
end

"""
    @inline coordinate(gbe::gridbackend{D, T, V, A}, element::NTuple{D, Int}) -> NTuple{D, T}

Return the physical coordinate corresponding to a grid index.

# Parameters
- `gbe::gridbackend{D, T, V, A}`  
  The structured grid container.
- `element::NTuple{D, Int}`  
  Index tuple (e.g., `(i, j, k)`) of the grid point.

# Returns
- `::NTuple{D, T}`  
  A tuple of coordinates `(x, y, z, ...)` corresponding to the grid point.
"""
@inline function coordinate(gbe::gridbackend{D, T, V, A}, element::NTuple{D, Int}) where {D, T <: AbstractFloat, V <: AbstractVector{T}, A <: AbstractArray{T, D}}
    return ntuple(i -> gbe.axes[i][element[i]], D)
end

"""
    generate_coordinate_grid(gbe::gridbackend{D, T, V, A}) -> Array{NTuple{D, T}, D}

Generate a full coordinate grid from a `gridbackend` definition.

For each grid index `(i, j, ...)`, returns a corresponding physical coordinate `(x, y, ...)` using `gbe.axes`.

# Parameters
- `gbe::gridbackend{D, T, V, A}`  
  The structured grid container, with axes and scalar field.

# Returns
- `::Array{NTuple{D, T}, D}`  
  An array of the same shape as `gbe.grid`, where each element is a tuple of physical coordinates at that grid point.

# Example
```julia
gbe = generate_empty_grid((0.0, 0.0), (1.0, 1.0), (3, 3))
coord_grid = generate_coordinate_grid(gbe)
coord_grid[2, 3]  # returns something like (0.5, 1.0)
```
"""
function generate_coordinate_grid(gbe::gridbackend{D, T, V, A}) where {D, T <: AbstractFloat, V <: AbstractVector{T}, A <: AbstractArray{T, D}}
    coordinates_array = Array{NTuple{D, T}, D}(undef, gbe.dimension...)

    for idx in CartesianIndices(coordinates_array)
        element_index = Tuple(idx)
        coordinates_array[idx] = coordinate(gbe, element_index)
    end
    return coordinates_array
end

"""
    grid_reduction(array::AbstractArray, averaged_axis_id::Int64 = 1) -> AbstractArray

Reduce a multidimensional array by averaging over one axis.

This function computes the mean along the specified axis and returns the result with that dimension removed, i.e., the output has one fewer dimension than the input.

# Parameters
- `array::AbstractArray`  
  The input array of arbitrary dimension to be reduced.
- `averaged_axis_id::Int64`  
  The axis over which to compute the mean (1-based indexing). Default is `1`.

# Returns
- `::AbstractArray`  
  A new array with dimension `ndims(array) - 1`, containing the mean values over the specified axis.

# Examples
```julia
A = rand(4, 3, 2)
B = grid_reduction(A, 2)  # average over the 2nd axis → size(B) = (4, 2)
```
"""
function grid_reduction(array::AbstractArray, averaged_axis_id::Int64 = 1)
    return dropdims(mean(array, dims = averaged_axis_id), dims = averaged_axis_id)
end

_array_type_reduced(::Type{A}, ::Val{D}) where {T, D, A<:AbstractArray{T, D}} = A.name.wrapper{T, D-1}
"""
    grid_reduction(gbe :: gridbackend, averaged_axis_id:: Int64 = 1)
Reducing the grid in the `gridbackend` by taking the average along a specific axes.

# Parameters
- `gbe :: gridbackend`: The 'gridbackend' that contain all of the axes information
- `averaged_axis_id :: Int64 = 1`: The axes that would be chosen for taking average along it.

# Returns
- `gridbackend`: The `gridbackend` that reduce from the original data.
"""
function grid_reduction(gbe::gridbackend{D, T, V, A}, averaged_axis_id::Int64 = 1) where {D, T <: AbstractFloat, V <: AbstractVector{T}, A <: AbstractArray{T, D}}
    new_data = grid_reduction(gbe.grid, averaged_axis_id)

    newaxes = collect(gbe.axes)
    deleteat!(newaxes, averaged_axis_id)
    newaxes_tuple::NTuple{D-1, V} = Tuple(newaxes)

    Ared = _array_type_reduced(A, Val(D))
    newgrid = generate_empty_grid(newaxes_tuple, T, Ared)

    if size(newgrid.grid) != size(new_data)
        error("ReductionError: Failed to reduce the gridbackend.")
    end

    newgrid.grid .= new_data
    return newgrid
end

"""
    disc_grid_generator(rmin::Real, rmax::Real, in::NTuple{2, Int};
        FloatType::Type{<:AbstractFloat}=Float64,
        VectorType::Type{<:AbstractVector}=Vector,
        ArrayType::Type{<:AbstractArray}=Array
    ) -> gridbackend{2, FloatType, VectorType{FloatType}, ArrayType{FloatType,2}}

Generate a 2D structured polar grid in (r, ϕ) space for disk-like geometry.

This utility builds a polar grid with uniform radial and angular resolution,
and returns a `gridbackend` object that stores both the coordinate axes and the data array.

The azimuthal angle ϕ ranges from 0 to just below 2π to avoid duplication at the periodic boundary.

# Parameters
- `rmin::Real`  
  Inner radius of the disk (must be ≥ 0).
- `rmax::Real`  
  Outer radius of the disk (must be > rmin).
- `in::NTuple{2, Int}`  
  Tuple `(Nr, Nϕ)` specifying the number of radial and angular divisions.

# Keyword Arguments
| Name         | Default     | Description                                  |
|--------------|-------------|----------------------------------------------|
| `FloatType`  | `Float64`   | Floating-point type for all grid values.     |
| `VectorType` | `Vector`    | Type constructor for 1D coordinate vectors.  |
| `ArrayType`  | `Array`     | Type constructor for the full grid array.    |

# Returns
- `gridbackend{2, FloatType, VectorType{FloatType}, ArrayType{FloatType,2}}`  
  A polar grid backend representing the (r, ϕ) domain with zero-initialized data.

# Notes
The azimuthal coordinate uses  
ϕ ∈ [0, 2π - Δϕ)  
to ensure periodicity without duplicating the endpoint.
"""
function disc_grid_generator(rmin :: Real, rmax :: Real, in :: NTuple{2, Int};
    FloatType::Type{<:AbstractFloat} = Float64, VectorType::Type{<:AbstractVector} = Vector, ArrayType::Type{<:AbstractArray} = Array)
    T = FloatType
    ϕmin = T(0.0)
    ϕmax = (2*T(π) - (2*T(π) / (in[2] + 1)))
    imin = (T(rmin), ϕmin)
    imax = (T(rmax), ϕmax)
    gbe = generate_empty_grid(imin, imax, in, FloatType=FloatType, VectorType=VectorType, ArrayType=ArrayType)
    return gbe
end

"""
    cylinder_grid_generator(rmin::Real, rmax::Real, zmin::Real, zmax::Real, in::NTuple{3, Int};
        FloatType::Type{<:AbstractFloat}=Float64,
        VectorType::Type{<:AbstractVector}=Vector,
        ArrayType::Type{<:AbstractArray}=Array
    ) -> gridbackend{3, FloatType, VectorType{FloatType}, ArrayType{FloatType,3}}

Generate a 3D structured polar grid in (r, ϕ, z) space for disk-like geometry.

This function creates a cylindrical grid covering the radial range `[rmin, rmax]`, the azimuthal angle `[0, 2π)`,
and the vertical extent `[zmin, zmax]`, and returns a `gridbackend` containing the initialized axes and data array.

To preserve angular periodicity, the azimuthal coordinate spans up to `2π - Δϕ`.

# Parameters
- `rmin::Real`  
  Inner radius of the disk.
- `rmax::Real`  
  Outer radius of the disk.
- `zmin::Real`  
  Lower bound in the vertical direction.
- `zmax::Real`  
  Upper bound in the vertical direction.
- `in::NTuple{3, Int}`  
  Number of grid points in `(Nr, Nϕ, Nz)` directions.

# Keyword Arguments
| Name         | Default     | Description                                  |
|--------------|-------------|----------------------------------------------|
| `FloatType`  | `Float64`   | Floating-point type for all grid values.     |
| `VectorType` | `Vector`    | Type constructor for 1D coordinate vectors.  |
| `ArrayType`  | `Array`     | Type constructor for the full grid array.    |

# Returns
- `gridbackend{3, FloatType, VectorType{FloatType}, ArrayType{FloatType,3}}`  
  A cylindrical (r, ϕ, z) grid with zero-initialized values.

# Notes
The angular coordinate ϕ is truncated at `2π - Δϕ` to avoid overlapping the periodic endpoint.
"""
function cylinder_grid_generator(rmin :: Real, rmax :: Real, zmin :: Real, zmax :: Real, in :: NTuple{3, Int};
    FloatType::Type{<:AbstractFloat} = Float64, VectorType::Type{<:AbstractVector} = Vector, ArrayType::Type{<:AbstractArray} = Array)
    T = FloatType
    ϕmin = T(0.0)
    ϕmax = (2*T(π) - (2*T(π) / (in[2] + 1)))
    imin = (T(rmin), ϕmin, T(zmin))
    imax = (T(rmax), ϕmax, T(zmax))
    gbe = generate_empty_grid(imin, imax, in, FloatType=FloatType, VectorType=VectorType, ArrayType=ArrayType)
    return gbe
end

"""
    func2gbe(rmin, rmax, in::NTuple{3, Int}; func, FloatType=Float64, VectorType=Vector, ArrayType=Array)

Generate a `gridbackend` in cylindrical coordinates and populate it with values from a user-defined function `f(s, ϕ)`.

This function constructs a 2D grid in (s, ϕ) space using the specified radial bounds, then evaluates the provided function `func` at each grid point's physical location to fill the data array.

# Parameters
- `rmin::Real`, `rmax::Real`  
  Radial bounds of the disk (minimum and maximum radius).
- `in::NTuple{2, Int}`  
  Number of grid points along `(s, ϕ)` dimensions.

# Keyword Arguments
| Name         | Default     | Description                                  |
|--------------|-------------|----------------------------------------------|
| `func`       | —           | Function `f(s, ϕ)` to evaluate on the grid.  |
| `FloatType`  | `Float64`   | Floating-point type for all grid values.     |
| `VectorType` | `Vector`    | Type constructor for 1D coordinate vectors.  |
| `ArrayType`  | `Array`     | Type constructor for the full grid array.    |

# Returns
- `::gridbackend{2, FloatType, VectorType{FloatType}, ArrayType{FloatType,2}}`  
  A structured grid containing `func(s, ϕ)` evaluated at each point.
"""
function func2gbe(rmin :: Real, rmax :: Real, in::NTuple{2, Int}; func,
    FloatType::Type{<:AbstractFloat} = Float64, VectorType::Type{<:AbstractVector} = Vector, ArrayType::Type{<:AbstractArray} = Array)
    gbe = disc_grid_generator(rmin, rmax, in, FloatType=FloatType, VectorType=VectorType, ArrayType=ArrayType)
    coordinate = generate_coordinate_grid(gbe)
    @inbounds for i in eachindex(coordinate)
        gbe.grid[i] = func(coordinate[i]...)
    end
    return gbe
end

"""
    func2gbe(rmin, rmax, zmin, zmax, in::NTuple{3, Int}; func, FloatType=Float64, VectorType=Vector, ArrayType=Array)

Generate a `gridbackend` in cylindrical coordinates and populate it with values from a user-defined function `f(s, ϕ, z)`.

This function constructs a 3D grid in (s, ϕ, z) space using the specified radial and vertical bounds, then evaluates the provided function `func` at each grid point's physical location to fill the data array.

# Parameters
- `rmin::Real`, `rmax::Real`  
  Radial bounds of the disk (minimum and maximum radius).
- `zmin::Real`, `zmax::Real`  
  Vertical bounds of the disk.
- `in::NTuple{3, Int}`  
  Number of grid points along `(s, ϕ, z)` dimensions.

# Keyword Arguments
| Name         | Default     | Description                                  |
|--------------|-------------|----------------------------------------------|
| `func`       | —           | Function `f(s, ϕ, z)` to evaluate on the grid.|
| `FloatType`  | `Float64`   | Floating-point type for all grid values.     |
| `VectorType` | `Vector`    | Type constructor for 1D coordinate vectors.  |
| `ArrayType`  | `Array`     | Type constructor for the full grid array.    |

# Returns
- `::gridbackend{3, FloatType, VectorType{FloatType}, ArrayType{FloatType,3}}`  
  A structured grid containing `func(s, ϕ, z)` evaluated at each point.
"""
function func2gbe(rmin :: Real, rmax :: Real, zmin :: Real, zmax :: Real, in::NTuple{3, Int}; func,
    FloatType::Type{<:AbstractFloat} = Float64, VectorType::Type{<:AbstractVector} = Vector, ArrayType::Type{<:AbstractArray} = Array)
    gbe = cylinder_grid_generator(rmin, rmax, zmin, zmax, in, FloatType=FloatType, VectorType=VectorType, ArrayType=ArrayType)
    coordinate = generate_coordinate_grid(gbe)
    @inbounds for i in eachindex(coordinate)
        gbe.grid[i] = func(coordinate[i]...)
    end
    return gbe
end
