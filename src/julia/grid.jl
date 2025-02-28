"""
The grid construction for SPH interpolation
    by Wei-Shan Su,
    June 21, 2024
"""

"""
    struct gridbackend
The struct for storeing data and axes. 

# Fields
- `grid :: Array`: A ndarray for storeing data.
- `axes :: Vector{LinRange{Float64,Int64}}`: The axes array for each dimension of grid.
- `dimension :: Vector{Int}`: The dimension of the grid (how much do the axes been separated.) e.g 3x3x4 matrix => [3,3,4]
"""
struct gridbackend
    grid::Array
    axes::Vector{LinRange{Float64,Int64}}
    dimension::Vector{Int64}
end

"""
    function meshgrid(arrays::AbstractVector...)
Mesh multiple arrays, get the meshgrid arrays

# Parameters
- `arrays :: AbstractVector`: The array that would be meshed. Can applied any number of arrays.

# Returns
- `Vector`: The vector that contains all of the meshgrids

# Examples
```julia
x = LinRange(0,10,250)
y = LinRange(0,2π,301)
meshgrids = meshgrid(x,y)
```
"""
function meshgrid(arrays::AbstractVector...)
    nd::Int = length(arrays)
    grids::Vector = [
        repeat(
            reshape(a, ntuple(d -> d == i ? length(a) : 1, nd)...),
            ntuple(d -> d == i ? 1 : length(arrays[d]), nd)...,
        ) for (i, a) in enumerate(arrays)
    ]
    return grids
end

"""
    function meshgrid(gbe::gridbackend)
Mesh multiple arrays from 'gridbackend', get the meshgrid arrays

# Parameters
- `gbe :: gridbackend`: The 'gridbackend' that contain all of the axes information

# Returns
- `Vector`: The vector that contains all of the meshgrids

# Examples
```julia
x = LinRange(0,10,250)
y = LinRange(0,2π,301)
meshgrids = meshgrid(x,y)
```
"""
function meshgrid(gbe::gridbackend)
    iaxes = gbe.axes
    meshgrids = meshgrid(iaxes...)
    return meshgrids
end

"""
    function generate_empty_grid(imin::Vector{Float64}, imax::Vector{Float64}, dimension::Vector{Int}, type::Type = Float64)
Generate an 'gridbackend'

# Parameters
- `imin::Vector{Float64}`: The minimum value of each axes. e.g.[x1min, x2min, x3min]
- `imax::Vector{Float64}`: The maximum value of each axes. e.g.[x1max, x2max, x3max]
- `dimension::Vector{Int}`: The dimension of the grid (how much does the axes been separated.) e.g [x1n, x2n, x3n]
- `type::Type = Float64`: Type of the data e.g. Float64

# Returns
- `gridbackend`: An 'gridbackend' that contain all of the axes information with empty data storage.

# Examples
```julia
# radial parameters
rmin :: Float64 = 0.0
rmax :: Float64 = 100.0
rn :: Int64 = 251

# azimuthal parameters
ϕmin :: Float64 = 0.0
ϕmax :: Float64 = 2π
ϕn :: Int64 = 301

# Constructing gbe
imin :: Vector{Float64} = [rmin, ϕmin]
imax :: Vector{Float64} = [rmax, ϕmax]
dimensions :: Vector{Int64} = [rn,ϕn]

grid :: gridbackend = generate_empty_grid(imin, imax, dimensions)
```
"""
function generate_empty_grid(
    imin::Vector{Float64},
    imax::Vector{Float64},
    dimension::Vector{Int},
    type::Type = Float64,
)
    if size(imin) == size(imax) == size(dimension)
        nothing
    else
        error("GridGeneratingError: Illegal input value.")
    end
    iaxes = Array{LinRange}(undef, length(dimension))
    for (i, num) in enumerate(dimension)
        iaxes[i] = LinRange(imin[i], imax[i], num)
    end
    grid::Array = zeros(type, dimension...)
    return gridbackend(grid, iaxes, dimension)
end

"""
    generate_empty_grid(iaxes::Vector{LinRange{Float64,Int64}}, type::Type = Float64)
Generate an 'gridbackend'

# Parameters
- `iaxes::Vector{LinRange{Float64,Int64}}`: The `LinRange` array with each axes
- `type::Type = Float64`: Type of the data e.g. Float64

# Returns
- `gridbackend`: An 'gridbackend' that contain all of the axes information with empty data storage.

# Examples
```julia
# Initialize iaxes
iaxes = Vector{LinRange{Float64,Int64}}(undef,2)
# Add axes
iaxes[1] = LinRange(0.0,100.0,251)
iaxes[2] = LinRange(0.0,2π,301)

grid :: gridbackend = generate_empty_grid(iaxes)
```
"""
function generate_empty_grid(iaxes::Vector{LinRange{Float64,Int64}}, type::Type = Float64)
    dimension::Vector{Int64} = length.(iaxes)
    grid::Array = zeros(type, dimension...)
    return gridbackend(grid, iaxes, dimension)
end

"""
    coordinate(gbe :: gridbackend, element :: Tuple)
Finding the corresponding coordinates for a specific element with given indices in the array of `gridbackend` data.

# Parameters
- `gbe :: gridbackend`: The 'gridbackend' that contain all of the axes information
- `element :: Tuple`: The indices of the element.

# Returns
- `Vector`: The coordinates vector

# Examples
```julia
grid :: gridbackend = generate_empty_grid(imin, imax, dimensions)
target_indices = [2,2,3]
coordinates = coordinate(grid, target_indices)
```
"""
function coordinate(gbe::gridbackend, element::Tuple)
    """
    Returning the corresponding coordinate of specific element.
    """
    if length(gbe.dimension) != length(element)
        error("GridLodingError: Mismatching of dimension between input element and grid.")
    end

    result::Vector = zeros(Float64, length(gbe.dimension))
    for i in eachindex(result)
        result[i] = gbe.axes[i][element[i]]
    end
    return result
end

"""
    generate_coordinate_grid(gbe :: gridbackend)
Finding the corresponding coordinates for a specific element for the entire array of `gridbackend` data.

# Parameters
- `gbe :: gridbackend`: The 'gridbackend' that contain all of the axes information

# Returns
- `Array{Vector{Float64}}`: The array that contains all of the vector of coordinate for each element in the data.

# Examples
```julia
grid :: gridbackend = generate_empty_grid(imin, imax, dimensions)
coordinates_array = generate_coordinate_grid(grid)
```
"""
function generate_coordinate_grid(gbe::gridbackend)
    """
    Returning an Array which contain the coordinate of the coorespoing element.
    """
    dims = gbe.dimension

    coordinates_array = Array{Vector{Float64},length(dims)}(undef, dims...)
    for idx in CartesianIndices(coordinates_array)
        element_index = Tuple(idx)
        coordinates_array[idx] = coordinate(gbe, element_index)
    end
    return coordinates_array
end

"""
    grid_reduction(gbe :: gridbackend, averaged_axis_id:: Int64 = 1)
Reducing the grid in the `gridbackend` by taking the average along a specific axes.

# Parameters
- `gbe :: gridbackend`: The 'gridbackend' that contain all of the axes information
- `averaged_axis_id :: Int64 = 1`: The axes that would be chosen for taking average along it.

# Returns
- `gridbackend`: The `gridbackend` that reduce from the original data.
"""
function grid_reduction(gbe::gridbackend, averaged_axis_id::Int64 = 1)
    new_data::Array =
        dropdims(mean(gbe.grid, dims = averaged_axis_id), dims = averaged_axis_id)
    newaxes = deepcopy(gbe.axes)
    deleteat!(newaxes, averaged_axis_id)
    type = typeof(newaxes[1][1])
    newgrid::gridbackend = generate_empty_grid(newaxes, type)
    if !(size(newgrid.grid) == size(new_data))
        error("ReductionError: Failed to reduce the gridbackend.")
    else
        newgrid.grid .= new_data
    end
    return newgrid
end

"""
    grid_reduction(gbe_dict :: Dict, averaged_axis_id:: Int64 = 1)
Reducing all of the grid in the dictionary by taking the average along a specific axis.

# Parameters
- `gbe_dict :: Dict`: The dictionary that contains all of the 'gridbackend'
- `averaged_axis_id :: Int64 = 1`: The axis that would be chosen for taking average along it.

# Returns
- `Dict`: The The dictionary that contains all of the 'gridbackend' from the original data.
"""
function grid_reduction(gbe_dict::Dict, averaged_axis_id::Int64 = 1)
    newdict = Dict()
    for key in keys(gbe_dict)
        gbe = gbe_dict[key]
        newdict[key] = grid_reduction(gbe, averaged_axis_id)
    end
    return newdict
end

"""
    grid_reduction(gbe_dict :: Dict, averaged_axis_id:: Int64 = 1)
Reducing array by taking average along a axis with given id

# Parameters
- `array :: Array`: The array with n-dimension.
- `averaged_axis_id :: Int64 = 1`: The axis that would be chosen for taking average along it.

# Returns
- `Array`: The array which has been reduce to (n-1)-dimension.
"""
function grid_reduction(array::Array, averaged_axis_id::Int64 = 1)
    return dropdims(mean(array, dims = averaged_axis_id), dims = averaged_axis_id)
end

"""
    disk_2d_grid_generator(imin::Vector ,imax::Vector, in::Vector)
Generate a face-on disk grid base on the polar/cylindrical coordinate (r,ϕ).

Note: If the azimuthal range is equal to 2π, the upper limit will be subtracted by 
    ϕmax_new = ϕmax - (2π/(ϕn+1))

# Parameters
- `imin :: Vector`: The minimum value of radial and azimuthal grid [rmin, ϕmin]
- `imax :: Vector`: The maximum value of radial and azimuthal grid [rmax, ϕmax]
- `in :: Vector`: The numbers of separattion of radial and azimuthal grid [rn, ϕn]

# Returns
- `gridbackend`: An 'gridbackend' that contain all of the axes information with empty data storage.

# Examples
```julia
rmin :: Float64 = 10.0
rmax :: Float64 = 100.0
rn :: Int64 = 251
ϕn :: Int64 = 301
grid :: gridbackend = disk_2d_grid_generator([rmin, 0.0], [rmax, 2π], [rn, ϕn])
```
"""
function disk_2d_grid_generator(imin::Vector, imax::Vector, in::Vector)
    if size(imin) == size(imax) == size(in)
        nothing
    else
        error("GridGeneratingError: Illegal input value.")
    end
    if (imax[2] > 2 * pi) || (imin[2] < 0)
        error("GridGeneratingError: Illegal theta value.")
    end
    if (imax[2] - imin[2]) == 2 * pi
        imax[2] -= (2 * pi / (in[2] + 1))
    end
    backend = generate_empty_grid(imin, imax, in)
    return backend
end

"""
    disk_3d_grid_generator(imin::Vector ,imax::Vector, in::Vector)
Generate a disk grid base on the cylindrical coordinate (r,ϕ,z).
Note: If the azimuthal range is equal to 2π, the upper limit will be subtracted by 
    ϕmax_new = ϕmax - (2π/(ϕn+1))

# Parameters
- `imin :: Vector`: The minimum value of radial and azimuthal grid [rmin, ϕmin, zmin]
- `imax :: Vector`: The maximum value of radial and azimuthal grid [rmax, ϕmax, zmax]
- `in :: Vector`: The numbers of separattion of radial and azimuthal grid [rn, ϕn, zn]

# Returns
- `gridbackend`: An 'gridbackend' that contain all of the axes information with empty data storage.

# Examples
```julia
rmin :: Float64 = 10.0
rmax :: Float64 = 100.0
zmin :: Float64 = -8
zmax :: Float64 = 8
rn :: Int64 = 251
ϕn :: Int64 = 16
zn :: Int64 = 101
grid :: gridbackend = disk_3d_grid_generator([rmin, 0.0, zmin], [rmax, 2π, zmax], [rn, ϕn, zn])
```
"""
function disk_3d_grid_generator(imin::Vector, imax::Vector, in::Vector)
    if size(imin) == size(imax) == size(in)
        nothing
    else
        error("GridGeneratingError: Illegal input value.")
    end
    if (imax[2] > 2 * pi) || (imin[2] < 0)
        error("GridGeneratingError: Illegal phi value.")
    end
    if (imax[2] - imin[2]) == 2 * pi
        imax[2] -= (2 * pi / (in[2] + 1))
    end
    backend = generate_empty_grid(imin, imax, in)
    return backend
end

"""
    func2gbe(imin::Vector, imax::Vector, in::Vector; func)
Transfer the function into a gridbackend.

# Parameters
- `imin :: Vector`: The minimum value of grid e.g[rmin, ϕmin, zmin]
- `imax :: Vector`: The maximum value of grid e.g[rmax, ϕmax, zmax]
- `in :: Vector`: The numbers of separattion of grid e.g[rn, ϕn, zn]

# Keyword arguments
- `func`: The function that contain the information.

# Returns
- `gridbackend`: An 'gridbackend' that contain all of the axes information with data storage.
"""
function func2gbe(imin::Vector, imax::Vector, in::Vector; func)
    if size(imin) == size(imax) == size(in)
        nothing
    else
        error("GridGeneratingError: Illegal input value.")
    end
    if (imax[2] > 2 * pi) || (imin[2] < 0)
        error("GridGeneratingError: Illegal phi value.")
    end
    if (imax[2] - imin[2]) == 2 * pi
        imax[2] -= (2 * pi / (in[2] + 1))
    end
    iLinRange = Vector{LinRange{Float64,Int64}}(undef,length(in))
    for i in eachindex(in)
        iLinRange[i] = LinRange(imin[i],imax[i],in[i])
    end
    meshgrids = meshgrid(iLinRange...)
    grid :: Array = Array{Float64}(undef,in...)
    grid = func.(meshgrids...)
    return gridbackend(grid,iLinRange,in)
end
