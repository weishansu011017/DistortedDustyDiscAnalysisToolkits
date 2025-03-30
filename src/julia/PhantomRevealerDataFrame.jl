"""
The PhantomRevealerDataFrame data Structure
    by Wei-Shan Su,
    June 23, 2024

Those methods with prefix `add` would store the result into the original data, and prefix `get` would return the value. 
Becarful, the methods with suffix `!` would change the inner state of its first argument!

# Structure:
    ## struct definition
        PhantomRevealerDataFrame
    ## Method
        ### get some quantity
        ### add some physical quantity into .
        ### rotating or translating the coordinate
"""

abstract type PhantomRevealerDataStructures end
"""
    struct PhantomRevealerDataFrame <: PhantomRevealerDataStructures
A data structure for storing the read dumpfile from SPH simulation.

# Fields
- `dfdata` :: The main data/particles information storage.
- `params` :: Global values stored in the dump file (time step, initial momentum, hfact, Courant factor, etc).
"""
struct PhantomRevealerDataFrame <: PhantomRevealerDataStructures
    dfdata::DataFrame
    params::Dict
end

# Extending index operation
# Single row and single column indexing
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, row_ind::Integer, col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[row_ind, col_ind]
end

# Multiple rows and single column indexing
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, row_inds::AbstractVector, col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[row_inds, col_ind]
end

# All rows and a single column indexing
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, ::Colon, col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[:, col_ind]
end

# Direct reference to a single column
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, ::typeof(!), col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[!, col_ind]
end

# Multiple rows and multiple columns indexing
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, row_inds::AbstractVector, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    return PhantomRevealerDataFrame(prdf.dfdata[row_inds, col_inds],prdf.params)
end

# Boolean vector indexing for rows and multiple columns
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, bool_mask::AbstractVector{Bool}, col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[bool_mask, col_ind]  # Vector
end

@inline function Base.getindex(prdf::PhantomRevealerDataFrame, bool_mask::AbstractVector{Bool}, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    return PhantomRevealerDataFrame(prdf.dfdata[bool_mask, col_inds], prdf.params)
end

@inline function Base.getindex(prdf::PhantomRevealerDataFrame, bool_mask::AbstractVector{Bool}, ::Colon)
    return PhantomRevealerDataFrame(prdf.dfdata[bool_mask, :], prdf.params)
end


# All rows and multiple columns indexing
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, ::Colon, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    return PhantomRevealerDataFrame(prdf.dfdata[:, col_inds],prdf.params)
end

# Single row and multiple columns assignment
@inline function Base.setindex!(prdf::PhantomRevealerDataFrame, value, row_ind::Integer, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    prdf.dfdata[row_ind, col_inds] = value
end

# Multiple rows and multiple columns assignment
@inline function Base.setindex!(prdf::PhantomRevealerDataFrame, value, row_inds::AbstractVector, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    prdf.dfdata[row_inds, col_inds] = value
end

# Extend `setindex!` to support `!` with column names for PhantomRevealerDataFrame
@inline function Base.setindex!(prdf::PhantomRevealerDataFrame, v::AbstractVector, ::typeof(!), col_ind::Union{Symbol, String})
    # Check if the length of `v` matches the number of rows in the DataFrame
    if nrow(prdf.dfdata) != length(v)
        throw(ArgumentError("New column must have the same length as the number of rows in the DataFrame."))
    end

    # Handle the assignment to an existing or new column
    if hasproperty(prdf.dfdata, Symbol(col_ind))
        prdf.dfdata[!, col_ind] = v  # Update existing column
    else
        prdf.dfdata[!, col_ind] = v  # Add new column
    end

    return prdf
end


#Method
"""
    print_params(data::PhantomRevealerDataFrame, pause::Bool=false)
Print out the `params` dictionary.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `pause :: Bool=false`: Pause the program after printing.
"""
function print_params(data::PhantomRevealerDataStructures, pause::Bool = false)
    allkeys = sort(collect(keys(data.params)))
    for key in allkeys
        println("$(key) => $(data.params[key])")
    end
    if pause
        readline()
    end
end

"""
    get_dim(data::PhantomRevealerDataFrame)
Get the dimension of simulation.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

#Returns
- 'Int64': The dimension of simulation of SPH data.
"""
function get_dim(data::PhantomRevealerDataFrame)
    return hasproperty(data.dfdata, "z") ? 3 : 2
end

"""
    get_time(data::PhantomRevealerDataFrame)
Get the time of simulation in code unit.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

#Returns
- 'Float64': The time of simulation.
"""
function get_time(data::PhantomRevealerDataFrame)
    if haskey(data.params,"time")
        return data.params["time"]
    elseif haskey(data.params,"Time")
        return data.params["Time"]
    else
        return 
    end
end

"""
    get_code_unit(data::PhantomRevealerDataFrame)
Get the value of converting from code unit to cgs.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
- 'Float64': Unit of distance
- 'Float64': Unit of mass
- 'Float64': Unit of time
- 'Float64': Unit of magnetic field
"""
function get_code_unit(data::PhantomRevealerDataStructures)
    udist = data.params["udist"]
    umass = data.params["umass"]
    utime = data.params["utime"]
    umagfd = data.params["umagfd"]
    return udist, umass, utime, umagfd
end

"""
    get_npart(data::PhantomRevealerDataFrame)
Get the number of particles in `data`.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
-`Int64`: The number of particles.
"""
function get_npart(data::PhantomRevealerDataFrame)
    return nrow(data.dfdata)
end

"""
    add_mean_h!(data::PhantomRevealerDataFrame)
Calculate the average smoothed radius of particles, storing into the `params` field.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
function add_mean_h!(data::PhantomRevealerDataFrame)
    data.params["h_mean"] = mean(data.dfdata[!, "h"])
end

"""
    get_truncated_radius(data::PhantomRevealerDataFrame, h::Float32=-1.0f0 , poffset::Float64=0.5, smoothed_kernal:: Function = M5_spline)
Get the proper truncated radius in unit length. 

The parameters `poffset` is used for doing a small positive offset on the result to make sure that all of the particles would be found while doing the `KDTree` searching.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `h :: Float32 = -1.0f0`: The smoothed radius. If `h < 0` then calling the average value of smoothed radius of particles.
- `poffset :: Float64=0.5`: A small modification to prevent the non-searched particles. Taking `0.0` would be the proper truncated radius.  
- `smoothed_kernal :: Function = M5_spline`: The smoothed kernel function.

# Returns
-`Float64`: The (modified) truncated radius with respect to the given kernel function
"""
function get_truncated_radius(
    data::PhantomRevealerDataFrame,
    h::Float32 = -1.0f0,
    poffset::Float64 = 0.5,
    smoothed_kernal::Function = M5_spline,
)
    if (h < 0)
        if !(haskey(data.params, "h_mean"))
            add_mean_h!(data)
        end
        h = data.params["h_mean"]
    end
    radius::Float64 =
        Float64((KernelFunctionValid()[nameof(smoothed_kernal)] + poffset) * h)
    return radius
end

"""
    get_unit_G(data::PhantomRevealerDataFrame)
Get the Gravitational constant G in code unit.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
-`Float64`: The Gravitational constant G in code unit.
"""
function get_unit_G(data::PhantomRevealerDataFrame)
    params = data.params
    udist = params["udist"]
    umass = params["umass"]
    utime = params["utime"]
    G_cgs = udist^3 * utime^(-2) * umass^(-1)
    G = G_cgs / 6.672041000000001e-8
    return G
end

"""
    get_general_coordinate(data::PhantomRevealerDataFrame,particle_index::Int)
Get the general coordinate `(x,y,z,vx,vy,vz)` of a specific particles.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `particle_index :: Int`: The index of particles

# Returns
- `Vector`: The general coordinate of the particle with given index.
"""
function get_general_coordinate(data::PhantomRevealerDataFrame, particle_index::Int)
    coordinate = Vector{Float64}(undef, 6)
    variable = String["x", "y", "z", "vx", "vy", "vz"]
    for (i, var) in enumerate(variable)
        coordinate[i] = data.dfdata[particle_index, var]
    end
    return coordinate
end

"""
    Generate_KDtree(data::PhantomRevealerDataFrame ,dim::Int)
Generate the kd tree of data in given dimension of space.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `dim::Int`: The dimension where kd tree is going to be constructed.

# Returns
- `KDTree`: The KDtree of SPH particles.
"""
function Generate_KDtree(data::PhantomRevealerDataFrame, dim::Int)
    if (dim == 2)
        position_array = hcat(data.dfdata[!, :x], data.dfdata[!, :y])'
    elseif (dim == 3)
        position_array = hcat(data.dfdata[!, :x], data.dfdata[!, :y], data.dfdata[!, :z])'
    end
    kdtree = KDTree(position_array)
    return kdtree
end

"""
    KDtreeRadiusFilter(data::PhantomRevealerDataFrame, kdtree::KDTree, target::Vector, radius::Float64, coordinate_flag::String = "cart")
Mask the particles which is located far from target.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `kdtree :: KDTree`: The kd tree of data.
- `target :: Vector`: Target position.
- `radius :: Float64`: The threshold of mask distance. Those particles which has a distance from the target that is larger then `radius` would be masked.
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Returns
- `PhantomRevealerDataFrame`: Masked data

# Example
```julia
prdf_list, prdf_sinks = read_phantom(dumpfile_00000)
data = prdf_list[1]
truncated_radius = get_truncated_radius(data)
kdtree3d :: KDTree = Generate_KDtree(data, 3)

target :: Vector = [10.0, 3.1415, 0.0] # In polar/cylindrical coordinate
coordinate_flag :: String = "polar"
filtered_data :: PhantomRevealerDataFrame = KDtreeRadiusFilter(data, kdtree3d, target, truncated_radius, coordinate_flag)
```
"""
function KDtreeRadiusFilter(
    data::PhantomRevealerDataFrame,
    kdtree::KDTree,
    target::Vector,
    radius::Float64,
    coordinate_flag::String = "cart",
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    reference_point is in "2D"
    "cart" = cartitian
    "polar" = polar
    """
    if coordinate_flag == "polar"
        target = _cylin2cart(target)
    end
    dim = length(first(kdtree.data))
    if (dim != length(target))
        error(
            "DimensionalError: The kdtree is constructed in $(dim)-d, but the given target is in $(length(target))-d.",
        )
    end
    kdtf_dfdata = data.dfdata[inrange(kdtree, target, radius), :]
    kdtf_data = PhantomRevealerDataFrame(kdtf_dfdata, data.params)
    return kdtf_data
end

"""
    KDtreeNearNeighborsFilter(data::PhantomRevealerDataFrame, kdtree::KDTree, target::Vector, N::Int64, coordinate_flag::String = "cart")

Filter out particles that are not among the `N` nearest neighbors of a given target position.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data stored in a `PhantomRevealerDataFrame`.
- `kdtree :: KDTree`: The KDTree constructed from `data`.
- `target :: Vector`: The reference position to search for nearest neighbors.
- `N :: Int64`: The number of nearest neighbors to retain.
- `coordinate_flag :: String = "cart"`: The coordinate system of `target`. Allowed values: `"cart"` (Cartesian), `"polar"` (polar/cylindrical).

# Returns
- `PhantomRevealerDataFrame`: A new data frame containing only the `N` nearest neighbors.

# Example
```julia
prdf_list, prdf_sinks = read_phantom(dumpfile_00000)
data = prdf_list[1]
truncated_radius = get_truncated_radius(data)
kdtree3d :: KDTree = Generate_KDtree(data, 3)

target :: Vector = [10.0, 3.1415, 0.0] # In polar/cylindrical coordinates
coordinate_flag :: String = "polar"
filtered_data :: PhantomRevealerDataFrame = KDtreeNearNeighborsFilter(data, kdtree3d, target, 100, coordinate_flag)
```
"""
function KDtreeNearNeighborsFilter(
    data::PhantomRevealerDataFrame,
    kdtree::KDTree,
    target::Vector,
    N::Int64,
    coordinate_flag::String = "cart",
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    reference_point is in "2D"
    "cart" = cartitian
    "polar" = polar
    """
    if coordinate_flag == "polar"
        target = _cylin2cart(target)
    end
    dim = length(first(kdtree.data))
    if (dim != length(target))
        error(
            "DimensionalError: The kdtree is constructed in $(dim)-d, but the given target is in $(length(target))-d.",
        )
    end
    idxs, _ = knn(kdtree, target, N, true)
    kdtf_dfdata = data.dfdata[idxs, :]
    kdtf_data = PhantomRevealerDataFrame(kdtf_dfdata, data.params)
    return kdtf_data
end

"""
    get_rnorm_ref(data::PhantomRevealerDataFrame, reference_position::Vector{Float64})
Get the array of distance between particles and the reference_position.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `reference_position::Vector{Float64}`: The reference point to estimate the distance.

# Returns
- `Vector`: The array of distance between particles and the reference_position.
"""
function get_rnorm_ref(data::PhantomRevealerDataFrame, reference_position::Vector{Float64})
    xt, yt, zt = reference_position
    x, y, z = data.dfdata[!, "x"], data.dfdata[!, "y"], data.dfdata[!, "z"]
    rnorm::Vector = sqrt.((x .- xt) .^ 2 + (y .- yt) .^ 2 + (z .- zt) .^ 2)
    return rnorm
end

"""
    get_r_ref(data::PhantomRevealerDataFrame,reference_position::Vector{Float64})
Get the array of distance and the relative offset between particles and the reference_position.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `reference_position::Vector{Float64}`: The reference point to estimate the distance.

# Returns
- `Vector`: The array of distance between particles and the reference_position.
- `Array`: The the relative offset between particles and the reference_position.
"""
function get_r_ref(data::PhantomRevealerDataFrame, reference_position::Vector{Float64})
    xt, yt, zt = reference_position
    x = xt .- data.dfdata[!, "x"]
    y = yt .- data.dfdata[!, "y"]
    z = zt .- data.dfdata[!, "z"]
    xyz::Array = hcat(x, y, z)
    rnorm::Vector = sqrt.(x .^ 2 + y .^ 2 + z .^ 2)
    return rnorm, xyz
end

"""
    get_snorm_ref(data::PhantomRevealerDataFrame, reference_position::Vector{Float64})
Get the array of distance between particles and the reference_position ON THE XY-PLANE PROJECTION.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `reference_position::Vector{Float64}`: The reference point to estimate the distance.

# Returns
- `Vector`: The array of distance between particles and the reference_position ON THE XY-PLANE PROJECTION.
"""
function get_snorm_ref(data::PhantomRevealerDataFrame, reference_position::Vector{Float64})
    if length(reference_position) == 2
        xt, yt = reference_position
    elseif length(reference_position) == 3
        xt, yt, zt = reference_position
    else
        error("DimensionalError: Wrong length for reference_position.")
    end
    x, y = data.dfdata[!, "x"], data.dfdata[!, "y"]
    snorm::Vector = sqrt.((x .- xt) .^ 2 + (y .- yt) .^ 2)
    return snorm
end

"""
    get_s_ref(data::PhantomRevealerDataFrame,reference_position::Vector{Float64})
Get the array of distance and the relative offset between particles and the reference_position ON THE XY-PLANE PROJECTION.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `reference_position::Vector{Float64}`: The reference point to estimate the distance.

# Returns
- `Vector`: The array of distance between particles and the reference_position ON THE XY-PLANE PROJECTION.
- `Array`: The the relative offset between particles and the reference_position ON THE XY-PLANE PROJECTION..
"""
function get_s_ref(data::PhantomRevealerDataFrame, reference_position::Vector{Float64})
    if length(reference_position) == 2
        xt, yt = reference_position
    elseif length(reference_position) == 3
        xt, yt, zt = reference_position
    else
        error("DimensionalError: Wrong length for reference_position.")
    end
    x = xt .- data.dfdata[!, "x"]
    y = yt .- data.dfdata[!, "y"]
    xy = hcat(x, y)
    snorm = sqrt.(x .^ 2 + y .^ 2)
    return snorm, xy
end

"""
    get_snorm(data::PhantomRevealerDataFrame)
Get the array of distance between particles and the origin ON THE XY-PLANE PROJECTION.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
- `Vector`: The array of distance between particles and the origin ON THE XY-PLANE PROJECTION.
"""
function get_snorm(data::PhantomRevealerDataFrame)
    return get_snorm_ref(data, [0.0, 0.0, 0.0])
end


"""
    COM2star!(data_list, sinks_data:: PhantomRevealerDataFrame,sink_particle_id::Int)
Transfer the coordinate to another coordinate with locating star at the origin.

# Parameters
- `data_list`: The array/single file which contains all of the data that would be transfered
- `sinks_data :: PhantomRevealerDataFrame`: The data which contains the sink star.
- `sink_particle_id :: Int`: The id of star that would be located at the origin.

# Example
```julia
# Transfer to the primary star-based coodinate(id=1)
prdf_list = read_phantom(dumpfile_00000)
sinks_data = prdf_list[end]         # The last data which is read from `read_phantom()` would always be the sinks data. 
COM2star!(prdf_list, sinks_data, 1)
```
"""
function COM2star!(data_list, sinks_data::PhantomRevealerDataFrame, sink_particle_id::Int)
    if (isa(data_list, Array))
        nothing
    elseif (isa(data_list, PhantomRevealerDataFrame))
        data_list = [data_list]
    else
        error("LoadError: Invaild Input in COM2star!")
    end
    general_coordinateQ1 = get_general_coordinate(sinks_data, sink_particle_id)
    variable = ["x", "y", "z", "vx", "vy", "vz"]
    for data in data_list
        for (i, var) in enumerate(variable)
            data.dfdata[:, var] .-= general_coordinateQ1[i]
        end
        data.params["COM_coordinate"] .-= general_coordinateQ1
        data.params["Origin_sink_id"] = sink_particle_id
        data.params["Origin_sink_mass"] = sinks_data.dfdata[sink_particle_id, "m"]
    end
end

"""
    star2COM!(data_list::Array)
Transfer the coordinate to COM coordinate.

# Parameters
- `data_list :: Array`: The array which contains all of the data that would be transfered

# Example
```julia
# Transfer to the primary star-based coodinate(id=1), and then transfer back.
prdf_list = read_phantom(dumpfile_00000)
sinks_data = prdf_list[end]         # The last data which is read from `read_phantom()` would always be the sinks data. 
println(prdf_list[1].params["Origin_sink_id"])  # print: -1
COM2star!(prdf_list, sinks_data, 1)
println(prdf_list[1].params["Origin_sink_id"])  # print: 1
star2COM!(prdf_list)
println(prdf_list[1].params["Origin_sink_id"])  # print: -1
```
"""
function star2COM!(data_list::Array)
    if (isa(data_list, Array))
        nothing
    elseif (isa(data_list, PhantomRevealerDataFrame))
        data_list = [data_list]
    else
        error("LoadError: Invaild Input in COM2star!")
    end
    variable = ["x", "y", "z", "vx", "vy", "vz"]
    for data in data_list
        COM_coordinate = data.params["COM_coordinate"]
        for (i, var) in enumerate(variable)
            data.dfdata[:, var] .-= COM_coordinate[i]
        end
        data.params["COM_coordinate"] .-= COM_coordinate
        data.params["Origin_sink_id"] = -1
        data.params["Origin_sink_mass"] = NaN
    end
end

"""
    add_cylindrical!(data::PhantomRevealerDataFrame)
Add the cylindrical/polar coordinate (s,ϕ) and corresponding velocity (vs, vϕ) into the data

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
function add_cylindrical!(data::PhantomRevealerDataFrame)
    data.dfdata[!, "s"] = sqrt.(data.dfdata[!, "x"] .^ 2 + data.dfdata[!, "y"] .^ 2)
    data.dfdata[!, "ϕ"] = atan.(data.dfdata[!, "y"], data.dfdata[!, "x"])
    sintheta = sin.(data.dfdata[!, "ϕ"])
    costheta = cos.(data.dfdata[!, "ϕ"])
    data.dfdata[!, "vs"] =
        (costheta .* data.dfdata[!, "vx"] + sintheta .* data.dfdata[!, "vy"])
    data.dfdata[!, "vϕ"] =
        (costheta .* data.dfdata[!, "vy"] - sintheta .* data.dfdata[!, "vx"])
end

"""
    add_norm!(data::PhantomRevealerDataFrame)
Add the length of position vector and velocity vector in 3D.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
function add_norm!(data::PhantomRevealerDataFrame)
    data.dfdata[!, "vrnorm"] =
        sqrt.(
            data.dfdata[!, "vx"] .^ 2 +
            data.dfdata[!, "vy"] .^ 2 +
            data.dfdata[!, "vz"] .^ 2
        )
    data.dfdata[!, "rnorm"] =
        sqrt.(
            data.dfdata[!, "x"] .^ 2 + data.dfdata[!, "y"] .^ 2 + data.dfdata[!, "z"] .^ 2
        )
end

"""
    add_norm!(data::DataFrame)
Add the length of position vector and velocity vector in 3D.

# Parameters
- `data :: DataFrame`: The SPH data that is stored in `DataFrame` 
"""
function add_norm!(dfdata::DataFrame)
    dfdata[!, "vrnorm"] =
        sqrt.(dfdata[!, "vx"] .^ 2 + dfdata[!, "vy"] .^ 2 + dfdata[!, "vz"] .^ 2)
    dfdata[!, "rnorm"] =
        sqrt.(dfdata[!, "x"] .^ 2 + dfdata[!, "y"] .^ 2 + dfdata[!, "z"] .^ 2)
end

"""
    add_rho!(data::PhantomRevealerDataFrame)
Add the local density of disk for each particles

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
function add_rho!(data::PhantomRevealerDataFrame)
    particle_mass = data.params["mass"]
    hfact = data.params["hfact"]
    d = get_dim(data)
    data.dfdata[!, "rho"] = particle_mass .* (hfact ./ data.dfdata[!, "h"]) .^ (d)
end

"""
    add_Kepelarian_azimuthal_velocity!(data::PhantomRevealerDataFrame)
Add the Kepelarian azimuthal velocity for each particles.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_Kepelarian_azimuthal_velocity!(data::PhantomRevealerDataFrame)
    if !(hasproperty(data.dfdata, "s"))
        add_cylindrical!(data)
    end
    G = get_unit_G(data)
    M = data.params["Origin_sink_mass"]
    data.dfdata[!, "vϕ_k"] = sqrt.((G * M) ./ data.dfdata[!, "s"])
    data.dfdata[!, "vϕ-vϕ_k"] = data.dfdata[!, "vϕ"] - data.dfdata[!, "vϕ_k"]
end

"""
    add_Kepelarian_angular_velocity!(data::PhantomRevealerDataFrame)
Add the Kepelarian angular velocity for each particles.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_Kepelarian_angular_velocity!(data::PhantomRevealerDataFrame)
    if !(hasproperty(data.dfdata, "s"))
        add_cylindrical!(data)
    end
    G = get_unit_G(data)
    M = data.params["Origin_sink_mass"]
    data.dfdata[!, "Ω_k"] = sqrt.((G * M) ./ (data.dfdata[!, "s"]).^3)
end

"""
    add_kinetic_energy!(data::PhantomRevealerDataFrame)
Add the Kinetic energy for each particles in current frame.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_kinetic_energy!(data::PhantomRevealerDataFrame)
    if !(hasproperty(data.dfdata, "vrnorm"))
        add_norm!(data)
    end
    dfdata = data.dfdata
    particle_mass = data.params["mass"]
    data.dfdata[!, "KE"] = (particle_mass / 2) .* dfdata[!, "vrnorm"]
end

"""
    add_specialized_angular_momentum!(data::PhantomRevealerDataFrame)
Add the specialized angular momentum vector for each particles in current frame.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_specialized_angular_momentum!(data::PhantomRevealerDataFrame)
    """add the angluar momentum w.r.t the current origin"""
    data.dfdata[!, "lx"] =
        (data.dfdata[!, "y"] .* data.dfdata[!, "vz"]) .-
        (data.dfdata[!, "z"] .* data.dfdata[!, "vy"])
    data.dfdata[!, "ly"] =
        (data.dfdata[!, "z"] .* data.dfdata[!, "vx"]) .-
        (data.dfdata[!, "x"] .* data.dfdata[!, "vz"])
    data.dfdata[!, "lz"] =
        (data.dfdata[!, "x"] .* data.dfdata[!, "vy"]) .-
        (data.dfdata[!, "y"] .* data.dfdata[!, "vx"])
    data.dfdata[!, "lnorm"] =
        sqrt.(
            data.dfdata[!, "lx"] .^ 2 +
            data.dfdata[!, "ly"] .^ 2 +
            data.dfdata[!, "lz"] .^ 2
        )
end

"""
    add_disk_normalized_angular_momentum!(data::PhantomRevealerDataFrame, rmin::Float64, rmax::Float64)
Add the normalized angular momentum vector of disk for each particles in current frame.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `rmin :: Float64`: The inner radius of disk.
- `rmax :: Float64`: The outer radius of disk.
"""
function add_disk_normalized_angular_momentum!(
    data::PhantomRevealerDataFrame,
    rmin::Float64,
    rmax::Float64,
)
    """calculate the disk angular momentum"""
    if !(hasproperty(data.dfdata, "lx")) ||
       !(hasproperty(data.dfdata, "ly")) ||
       !(hasproperty(data.dfdata, "lz"))
        add_specialized_angular_momentum!(data)
    end
    snorm = get_snorm(data)
    ldisk = zeros(Float64, 3)
    disk_particles = (snorm .> rmin) .& (snorm .< rmax)
    for (i, dir) in enumerate(["lx", "ly", "lz"])
        ldisk[i] = mean(data.dfdata[disk_particles, dir])
    end
    ldisk ./= norm(ldisk)
    data.params["ldisk"] = ldisk
end

"""
    add_tilt!(data::PhantomRevealerDataFrame, rmin::Float64, rmax::Float64)
Add the tilt of particles. 

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `rmin :: Float64`: The inner radius of disk.
- `rmax :: Float64`: The outer radius of disk.
"""
function add_tilt!(data::PhantomRevealerDataFrame, rmin::Float64, rmax::Float64)
    if !(hasproperty(data.dfdata, "lx")) ||
       !(hasproperty(data.dfdata, "ly")) ||
       !(hasproperty(data.dfdata, "lz"))
        add_disk_normalized_angular_momentum!(data, rmin, rmax)
    end
    if !(hasproperty(data.dfdata, "rnorm"))
        add_norm!(data)
    end
    rlproject =
        (
            data.dfdata[!, "x"] .* data.dfdata[!, "lx"] +
            data.dfdata[!, "y"] .* data.dfdata[!, "ly"] +
            data.dfdata[!, "z"] .* data.dfdata[!, "lz"]
        ) ./ data.dfdata[!, "lnorm"]
    nonzero_rnorm = data.dfdata[!, "rnorm"] .!= 0
    data.dfdata[!, "tilt"] =
        asin.(rlproject[nonzero_rnorm] ./ data.dfdata[nonzero_rnorm, "rnorm"])
end

"""
    rotate_to_disk_L!(data_list::Array, rmin::Float64, rmax::Float64, target_laxis::Union{Nothing, Vector{Float64}} = nothing)
Rotate the whole data to make z become angular_momentum_vector of disk.
Will take the angular momentum information from the first file. 

if no (data_list[1].params['ldisk']) => Make it

R = RxRy => rotate y axis and then x axis

        1      0      0
Rx = [  0   cos(ϕx) -sin(ϕx)]
        0   sin(ϕx)  cos(ϕx) 

    cos(ϕy)  0     sin(ϕy)
Ry = [  0      1      0     ]
    -sin(ϕy)  0     cos(ϕy)

l = (lx,ly,lz), lxz = N(lx,0,lz), N = 1/√lx^2 + lz^2
θy = Nlz, θx = N(lx^2 + lz^2) = 1/N

# Parameters
- `data_list :: Array`: The array which contains all of the data that would be transfered
- `rmin :: Float64`: The inner radius of disk.
- `rmax :: Float64`: The outer radius of disk.
- `target_laxis :: Union{Nothing, Vector{Float64}} = nothing`: The target 
"""
function rotate_to_disk_L!(
    data_list::Array,
    rmin::Float64,
    rmax::Float64,
    target_laxis::Union{Nothing,Vector{Float64}} = nothing
)
    for data in data_list
        if (data.params["Origin_sink_id"] == -1)
            COM2star!(data, data_list[end], 1)
        end
    end
    if isnothing(target_laxis)
        if !(haskey(data_list[1].params, "ldisk"))
            add_disk_normalized_angular_momentum!(data_list[1], rmin, rmax)
            laxis = data_list[1].params["ldisk"]
        end
    else
        laxis = target_laxis
    end
    if laxis[3] < 0
        laxis = -laxis
    end
    lx, ly, lz = laxis
    N = 1 / sqrt(lx^2 + lz^2)
    cosϕy = N * lz
    sinϕy = sin(acos(cosϕy))
    cosϕx = 1 / N
    sinϕx = sin(acos(cosϕx))
    sinsinxy = sinϕx * sinϕy
    cossinxy = cosϕx * sinϕy
    sincosxy = sinϕx * cosϕy
    coscosxy = cosϕx * cosϕy
    for data in data_list
        dfdata = data.dfdata
        copydfdata = deepcopy(dfdata)
        dfdata[!, "x"] =
            cosϕy * copydfdata[!, "x"] + 0 * copydfdata[!, "y"] + sinϕy * copydfdata[!, "z"]
        dfdata[!, "y"] =
            sinsinxy * copydfdata[!, "x"] + cosϕx * copydfdata[!, "y"] -
            sincosxy * copydfdata[!, "z"]
        dfdata[!, "z"] =
            -cossinxy * copydfdata[!, "x"] +
            sinϕx * copydfdata[!, "y"] +
            coscosxy * copydfdata[!, "z"]
        dfdata[!, "vx"] =
            cosϕy * copydfdata[!, "vx"] +
            0 * copydfdata[!, "vy"] +
            sinϕy * copydfdata[!, "vz"]
        dfdata[!, "vy"] =
            sinsinxy * copydfdata[!, "vx"] + cosϕx * copydfdata[!, "vy"] -
            sincosxy * copydfdata[!, "vz"]
        dfdata[!, "vz"] =
            -cossinxy * copydfdata[!, "vx"] +
            sinϕx * copydfdata[!, "vy"] +
            coscosxy * copydfdata[!, "vz"]
        add_disk_normalized_angular_momentum!(data, rmin, rmax)
    end
    return laxis
end

"""
    add_eccentricity!(data::PhantomRevealerDataFrame)
Add the eccentricity for each particle with respect to current origin.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_eccentricity!(data::PhantomRevealerDataFrame)
    if !(haskey(data.params, "Origin_sink_id")) || (data.params["Origin_sink_id"] == -1)
        error(
            "OriginLocatedError: Wrong origin located. Please use COM2star!() to transfer the coordinate.",
        )
    end
    G = get_unit_G(data)
    M1 = data.params["Origin_sink_mass"]
    μ = G * M1
    dfdata = data.dfdata
    if !(hasproperty(dfdata, "rnorm")) || !(hasproperty(dfdata, "vrnorm"))
        add_norm!(dfdata)
    end
    x, y, z = dfdata[!, "x"], dfdata[!, "y"], dfdata[!, "z"]
    vx, vy, vz = dfdata[!, "vx"], dfdata[!, "vy"], dfdata[!, "vz"]
    rnorm = dfdata[!, "rnorm"]
    vrnorm = dfdata[!, "vrnorm"]
    rdotv = (x .* vx) .+ (y .* vy) .+ (z .* vz)
    vrnorm2 = vrnorm .^ 2
    invrnorm = 1 ./ rnorm
    ex = ((vrnorm2 ./ μ) .- invrnorm) .* x - (rdotv ./ μ) .* vx
    ey = ((vrnorm2 ./ μ) .- invrnorm) .* y - (rdotv ./ μ) .* vy
    ez = ((vrnorm2 ./ μ) .- invrnorm) .* z - (rdotv ./ μ) .* vz
    dfdata[!, "e"] = sqrt.(ex .^ 2 + ey .^ 2 + ez .^ 2)
end

"""
    get_disk_mass(data::PhantomRevealerDataFrame, sink_data::PhantomRevealerDataFrame, disk_radius::Float64=120.0, sink_particle_id::Int64=1)
Get the mass of disk around the sink particle with given ID.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `sink_data :: PhantomRevealerDataFrame`: The data which contains the sink star.
- `disk_radius :: Float64 = 120.0`: The radius of disk.
- `sink_particle_id :: Int64 = 1`: The ID of sink particles in `sink_data`

# Return 
- `Float64`: The mass of disk around specific sink particle with given ID.
"""
function get_disk_mass(
    data::PhantomRevealerDataFrame,
    sink_data::PhantomRevealerDataFrame,
    disk_radius::Float64 = 120.0,
    sink_particle_id::Int64 = 1
)
    data_cp = deepcopy(data)
    particle_mass = data_cp.params["mass"]
    if data_cp.params["Origin_sink_id"] != sink_particle_id
        COM2star!(data_cp, sink_data, sink_particle_id)
    end
    kdtree = Generate_KDtree(data_cp, get_dim(data_cp))
    kdtf_data = KDtreeRadiusFilter(
        data_cp,
        kdtree,
        zeros(Float64, get_dim(data_cp)),
        disk_radius,
        "cart",
    )
    return particle_mass * Float64(nrow(kdtf_data.dfdata))
end

"""
    Analysis_params_recording(data::PhantomRevealerDataFrame)
Generate the dictionary for recording the basic properties of dumpfile.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`

# Returns
- `Dict{String, Any}`: The dictionary of parameters.
"""
function Analysis_params_recording(data::PhantomRevealerDataFrame)
    record_params = ["Origin_sink_id","Origin_sink_mass","grainsize","graindens","qfacdisc","udist","umass","utime","umagfd"]
    params = Dict{String,Any}()
    data_params = data.params
    params["time"] = get_time(data)
    for rparam in record_params
        if haskey(data_params,rparam)
            params[rparam] = data_params[rparam]
        end
    end
    return params
end
