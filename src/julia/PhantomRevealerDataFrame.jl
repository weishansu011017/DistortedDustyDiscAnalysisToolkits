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


# Single row, all columns -> PRDF (1-row)
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, row::Integer, ::Colon)
    return PhantomRevealerDataFrame(prdf.dfdata[row:row, :], prdf.params)
end

# Single row, multiple columns -> PRDF (1-row)
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, row::Integer,
                               cols::Union{Vector{Symbol},Vector{String},Vector{Int}})
    return PhantomRevealerDataFrame(prdf.dfdata[row:row, cols], prdf.params)
end

# Multiple rows, all columns -> PRDF
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, rows::AbstractVector, ::Colon)
    return PhantomRevealerDataFrame(prdf.dfdata[rows, :], prdf.params)
end

# All rows, all columns -> PRDF 
@inline function Base.getindex(prdf::PhantomRevealerDataFrame, ::Colon, ::Colon)
    return PhantomRevealerDataFrame(prdf.dfdata[:, :], prdf.params)
end

# Single row and single columns assignment
@inline function Base.setindex!(prdf::PhantomRevealerDataFrame, value, row_ind::Integer, col_ind::Union{String, Symbol})
    prdf.dfdata[row_ind, col_ind] = value
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

# Extend `names` function for PhantomRevealerDataFrame
@inline function Base.names(prdf :: PhantomRevealerDataFrame)
    return names(prdf.dfdata)
end

# Constant
const KDSearching_scratch = Ref{Vector{Vector{Int}}}()

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
    @inline get_dim(data::PhantomRevealerDataFrame)
Get the dimension of simulation.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

#Returns
- 'Int64': The dimension of simulation of SPH data.
"""
@inline function get_dim(data::PhantomRevealerDataFrame)
    return hasproperty(data.dfdata, "z") ? 3 : 2
end

"""
    @inline get_time(data::PhantomRevealerDataFrame)
Get the time of simulation in code unit.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

#Returns
- 'Float64': The time of simulation.
"""
@inline function get_time(data::PhantomRevealerDataFrame)
    if haskey(data.params,"time")
        return data.params["time"]
    elseif haskey(data.params,"Time")
        return data.params["Time"]
    else
        return 
    end
end

"""
    @inline get_code_unit(data::PhantomRevealerDataFrame)
Get the value of converting from code unit to cgs.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
- 'Float64': Unit of distance
- 'Float64': Unit of mass
- 'Float64': Unit of time
- 'Float64': Unit of magnetic field
"""
@inline function get_code_unit(data::PhantomRevealerDataStructures)
    udist = data.params["udist"]
    umass = data.params["umass"]
    utime = data.params["utime"]
    umagfd = data.params["umagfd"]
    return udist, umass, utime, umagfd
end

"""
    @inline get_npart(data::PhantomRevealerDataFrame)
Get the number of particles in `data`.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
-`Int64`: The number of particles.
"""
@inline function get_npart(data::PhantomRevealerDataFrame)
    return nrow(data.dfdata)
end

"""
    @inline get_init_npart(data::PhantomRevealerDataFrame)
Get the number of particles in `data`.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
-`Int64`: The number of particles.
"""
@inline function get_init_npart(data::PhantomRevealerDataFrame)
    itype = data.params["itype"]
    return data.params[string("npartoftype", itype == 1 ? "" : string("_", itype), )]
end

"""
    @inline add_mean_h!(data::PhantomRevealerDataFrame)
Calculate the average smoothed radius of particles, storing into the `params` field.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
@inline function add_mean_h!(data::PhantomRevealerDataFrame)
    data.params["h_mean"] = mean(data.dfdata[!, "h"])
end

"""
    @inline get_unit_G(data::PhantomRevealerDataFrame)
Get the Gravitational constant G in code unit.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
-`Float64`: The Gravitational constant G in code unit.
"""
@inline function get_unit_G(data::PhantomRevealerDataFrame) :: Float64
    params = data.params
    udist = params["udist"]
    umass = params["umass"]
    utime = params["utime"]
    G_cgs :: Float64 = udist^3 * utime^(-2) * umass^(-1)
    G ::Float64 = G_cgs / 6.672041000000001e-8
    return G
end

"""
    @inline get_general_coordinate(data::PhantomRevealerDataFrame,particle_index::Int)
Get the general coordinate `(x,y,z,vx,vy,vz)` of a specific particles.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `particle_index :: Int`: The index of particles

# Returns
- `Vector`: The general coordinate of the particle with given index.
"""
@inline function get_general_coordinate(data::PhantomRevealerDataFrame, particle_index::Int) :: Vector{Float64}
    coordinate :: Vector{Float64} = Vector{Float64}(undef, 6)
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
    get_Neighbor_indices(tree::KDTree, target::NTuple{D, T}, radius_multiplier::T, h_array::AbstractVector{T})

Finds neighboring particle indices around a target point using a KDTree.

This function first identifies the index `i₀` of the nearest neighbor to the `target` using `knn`, then uses the corresponding smoothing length `h_array[i₀]` scaled by `radius_multiplier` to define a search radius. It then populates a thread-local buffer with all indices within this radius using `inrange!`.

# Parameters
- `tree::KDTree`: The KDTree spatial index used for neighbor search.
- `target::NTuple{D, T}`: The coordinates of the target point in D dimensions.
- `radius_multiplier::T`: A multiplier applied to the smoothing length to define the neighbor search radius.
- `h_array::AbstractVector{T}`: An array of per-particle smoothing lengths or characteristic scales.

# Returns
- `idxs::Vector{Int}`: Indices of neighboring particles within the computed radius (thread-local scratch buffer; not heap-allocated).
- `ha::T`: The smoothing length of the nearest particle to the target, i.e., `h_array[i₀]`.

"""
function get_Neighbor_indices(tree :: KDTree, target :: NTuple{D, T}, radius_multiplier::T, h_array :: AbstractVector{T}) where {T<:AbstractFloat, D}
    tid = threadid()
    idxs = KDSearching_scratch[][tid]
    empty!(idxs)
    i0 = knn(tree, SVector{D, T}(target), 1)[1][1]
    ha = h_array[i0]
    radius = radius_multiplier * ha
    inrange!(idxs, tree, SVector{D, T}(target), radius)
    return idxs, ha
end

"""
    get_SymmetricNeighbor_indices(tree::KDTree, target::NTuple{D, T}, radius_multiplier::T, h_array::AbstractVector{T}) -> Tuple{Vector{Int}, T}

Return the symmetric SPH neighbor indices for a given target point based on the enlarged kernel support.

This function enforces the symmetric condition between the target point and all its neighbors by ensuring that every neighbor `j` satisfies:

    ‖x_a - x_b‖ < max(κ * h_a, κ * h_b)

where `κ = radius_multiplier`, `h_a` is the smoothing length of the target point, and `h_b` is that of the neighbor.  
The final radius is iteratively expanded until this criterion is satisfied for all selected neighbors.

# Parameters
- `tree::KDTree`  
  KDTree containing the particle positions, assumed to match the ordering of `h_array`.

- `target::NTuple{D, T}`  
  Target point in D-dimensional space (typically 2D or 3D).

- `radius_multiplier::T`  
  Support radius scaling factor `κ` (e.g., 2.0, 2.5, 3.0 for M4/M5/M6 kernels).

- `h_array::AbstractVector{T}`  
  Smoothing lengths for all particles, must align with `tree`.

# Returns
- `::Tuple{Vector{Int}, T}`  
  A tuple `(idxs, ha)` where `idxs` is the set of symmetric neighbor indices,  
  and `ha` is the smoothing length of the nearest particle (used as the "target").

"""
function get_SymmetricNeighbor_indices(tree :: KDTree, target :: NTuple{D, T}, radius_multiplier :: T, h_array :: AbstractVector{T}) where {T<:AbstractFloat, D}
    tid = threadid()
    idxs = KDSearching_scratch[][tid]
    empty!(idxs)
    
    i0 = knn(tree, SVector{D, T}(target), 1)[1][1]
    ha = h_array[i0]

    rad = radius_multiplier * ha
    inrange!(idxs, tree, SVector{D, T}(target), rad)

    while !isempty(idxs)
        hb = h_array[idxs]
        max_hb = maximum(hb)
        new_rad = radius_multiplier * max_hb
        new_rad ≤ rad && break
        rad = new_rad
        empty!(idxs)
        inrange!(idxs, tree, SVector{D, T}(target), rad)
    end
    return idxs, ha
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
    KDtreeSymmetricFilter(
        data::PhantomRevealerDataFrame,
        kdtree::KDTree,
        target::Vector,
        radius::Float64,
        coordinate_flag::String = "cart"
    )

Filter out particles based on symmetric smoothing-length criteria, including any neighbors for which the distance to `target` is less than `n*h_i` or less than `n*h_j`.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data stored in a `PhantomRevealerDataFrame`.
- `kdtree :: KDTree`: The KDTree constructed from `data`.
- `target :: Vector`: The reference position to search around.
- `radius :: Float64`: The initial cutoff distance `n * h_i`, where `h_i` is the smoothing length of the particle at `target` and `n` is the kernel support multiplier.
- `coordinate_flag :: String = "cart"`: The coordinate system of `target`. Allowed values:
"""
function KDtreeSymmetricFilter(
    data::PhantomRevealerDataFrame,
    tree::KDTree,
    target::AbstractVector{<:Real};
    kernel::Type{K} = M5_spline,
) where {K<:AbstractSPHKernel}

    tar = coord === :cylin ? _cylin2cart(target) : target

    dim_tree = size(tree.data, 1)
    length(tar) == dim_tree || error("KDTree is $dim_tree-D but target is $(length(tar))-D.")

    # Check kernel validity
    hT = eltype(data.dfdata[!, "h"])
    hasmethod(KernelFunctionValid, Tuple{Type{K}, Type{hT}}) ||
        error("No KernelFunctionValid(...) method for kernel $(K).")
    coeff = KernelFunctionValid(kernel, hT)

    # Use the shared symmetric neighbor finder
    idxs, _ = get_SymmetricNeighbor_indices(tree, tar, coeff, data.dfdata[!, "h"])

    return PhantomRevealerDataFrame(data.dfdata[idxs, :], data.params)
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

function _apply_coordinate_shift!(data :: PhantomRevealerDataFrame, new_origin :: NTuple{6, TF}) where {TF <: AbstractFloat}
    xo, yo, zo, vxo, vyo, vzo = new_origin
    x  = data.dfdata.x
    y  = data.dfdata.y
    z  = data.dfdata.z
    vx = data.dfdata.vx
    vy = data.dfdata.vy
    vz = data.dfdata.vz
    @inbounds @simd for i in eachindex(x)
        xi = x[i]; yi = y[i]; zi = z[i]
        vxi = vx[i]; vyi = vy[i]; vzi = vz[i]

        x[i] = xi - xo
        y[i] = yi - yo
        z[i] = zi - zo
        vx[i] = vxi - vxo
        vy[i] = vyi - vyo
        vz[i] = vzi - vzo
    end
end

"""
    COM2star!(data_list :: V,  sink_particle_id :: Int) where {D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
Transfer the coordinate to another coordinate with locating star at the origin.
Assuming the sink data is stored as the final `PhantomRevealerDataFrame` in the Vector

# Parameters
- `data_list :: V`: The array which contains all of the data that would be transfered
- `sink_particle_id :: Int`: The id of star that would be located at the origin.

# Example
```julia
# Transfer to the primary star-based coodinate(id=1)
prdf_list = read_phantom(dumpfile_00000)
COM2star!(prdf_list, 1)
```
"""
function COM2star!(data_list :: V, sink_particle_id :: Int) where {D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
    sinks_data = data_list[end]
    new_origin = (sinks_data.dfdata.x[sink_particle_id], 
                  sinks_data.dfdata.y[sink_particle_id],
                  sinks_data.dfdata.z[sink_particle_id],
                  sinks_data.dfdata.vx[sink_particle_id],
                  sinks_data.dfdata.vy[sink_particle_id],
                  sinks_data.dfdata.vz[sink_particle_id])

    @inbounds for data in data_list
        _apply_coordinate_shift!(data, new_origin)
    end
    sinks_data.params["COM_coordinate"] .-= new_origin
    sinks_data.params["Origin_sink_id"][] = sink_particle_id
    sinks_data.params["Origin_sink_mass"][] = sinks_data.dfdata.m[sink_particle_id]
    return nothing
end

"""
    star2COM!(data_list :: V) where {D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
Transfer the coordinate to COM coordinate.

# Parameters
- `data_list :: V`: The array which contains all of the data that would be transfered

# Example
```julia
# Transfer to the primary star-based coodinate(id=1), and then transfer back.
prdf_list = read_phantom(dumpfile_00000)
println(prdf_list[1].params["Origin_sink_id"][])  # print: -1
COM2star!(prdf_list, 1)
println(prdf_list[1].params["Origin_sink_id"][])  # print: 1
star2COM!(prdf_list)
println(prdf_list[1].params["Origin_sink_id"][])  # print: -1
```
"""

function star2COM!(data_list :: V) where {D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
    COM_position = data_list[1].params["COM_coordinate"]
    new_origin = ntuple(i -> COM_position[i], 6)

    @inbounds for data in data_list
        _apply_coordinate_shift!(data, new_origin)
    end
    data_list[1].params["COM_coordinate"] .-= new_origin
    data_list[1].params["Origin_sink_id"][] = -1
    data_list[1].params["Origin_sink_mass"][] = NaN64
    return nothing
end


@inline function _rotational_matrix(lx :: TF, ly :: TF, lz :: TF) :: NTuple{9, TF} where {TF <: AbstractFloat}
    # Normalize target axis
    lnorm = sqrt(lx^2 + ly^2 + lz^2)
    lx /= lnorm
    ly /= lnorm
    lz /= lnorm

    # Constructing rotating elements
    ρ = sqrt(lx^2 + lz^2)
    if ρ ≤ eps(TF)
        cosϕy = one(TF) ;  sinϕy = zero(TF)
        cosϕx = zero(TF);  sinϕx = sign(ly)
        cosψ  = one(TF) ;  sinψ = zero(TF)
    else
        invρ  = 1/ρ
        cosϕy = lz*invρ
        sinϕy = -lx*invρ   
        cosϕx = ρ
        sinϕx = ly
        ψ = atan(lx, -ly*lz)
        cosψ, sinψ = cos(ψ), sin(ψ)
    end

    # Rotational matrix R
    r11 = cosψ * cosϕy + sinψ * sinϕx * sinϕy
    r12 = sinψ * cosϕx
    r13 = cosψ * sinϕy - sinψ * sinϕx * cosϕy

    r21 = -sinψ * cosϕy + cosψ * sinϕx * sinϕy
    r22 =  cosψ * cosϕx
    r23 = -sinψ * sinϕy - cosψ * sinϕx * cosϕy

    r31 = -cosϕx * sinϕy
    r32 =  sinϕx
    r33 =  cosϕx * cosϕy
    return (r11,r12,r13,r21,r22,r23,r31,r32,r33)
end 

function _apply_zaxis_orientation!(data :: PhantomRevealerDataFrame, R :: NTuple{9, TF}) where {TF <: AbstractFloat}
    # Inplace rotation
    x  = data.dfdata.x
    y  = data.dfdata.y
    z  = data.dfdata.z
    vx = data.dfdata.vx
    vy = data.dfdata.vy
    vz = data.dfdata.vz
    r11,r12,r13,r21,r22,r23,r31,r32,r33 = R
    @inbounds @simd for i in eachindex(x)
        xi,yi,zi = x[i],y[i],z[i]
        vxi,vyi,vzi = vx[i],vy[i],vz[i]
        x[i]  = muladd(r13,zi, muladd(r12,yi, r11*xi))
        y[i]  = muladd(r23,zi, muladd(r22,yi, r21*xi))
        z[i]  = muladd(r33,zi, muladd(r32,yi, r31*xi))
        vx[i] = muladd(r13,vzi, muladd(r12,vyi, r11*vxi))
        vy[i] = muladd(r23,vzi, muladd(r22,vyi, r21*vxi))
        vz[i] = muladd(r33,vzi, muladd(r32,vyi, r31*vxi))
    end
end

"""
    set_zaxis_orientation!(data::PhantomRevealerDataFrame, target_zaxis::NTuple{3,TF}; inverse::Bool=false) where {TF<:AbstractFloat}

Rotate all positions (x,y,z) and velocities (vx,vy,vz) in-place so that the z-axis aligns with the given `target_zaxis`. The rotation is orthogonal (det=1), preserving lengths and inner products. By construction, the new x-axis lies in the original xy-plane:
    x' = normalize(ẑ × l̂),   y' = z' × x',   z' = l̂

where ẑ = (0,0,1) and l̂ = normalized target_zaxis.

        1      0      0
Rx = [  0   cos(ϕx) -sin(ϕx)]
        0   sin(ϕx)  cos(ϕx) 

    cos(ϕy)  0     sin(ϕy)
Ry = [  0      1      0     ]
    -sin(ϕy)  0     cos(ϕy)

# Parameters
- `data::PhantomRevealerDataFrame`: SPH data container with fields `x,y,z,vx,vy,vz`.
- `target_zaxis::NTuple{3,TF}`: Vector specifying the new z-axis (not required to be normalized).

# Keyword Arguments
- `inverse::Bool=false`: If `false`, apply the forward rotation (v' = R * v).
                         If `true`, apply the inverse rotation (v' = R' * v).

"""
function set_zaxis_orientation!(data :: PhantomRevealerDataFrame, target_zaxis :: NTuple{3, TF}; inverse :: Bool = false) where {TF<: AbstractFloat}
    Rdefault = _rotational_matrix(target_zaxis...)
    if inverse
        R = (Rdefault[1], Rdefault[4], Rdefault[7], Rdefault[2], Rdefault[5], Rdefault[8], Rdefault[3], Rdefault[6], Rdefault[9])
    else
        R = Rdefault
    end    
    _apply_zaxis_orientation!(data, R)
end

"""
    set_zaxis_orientation!(data_list :: V, target_zaxis :: NTuple{3, TF}; inverse :: Bool = false) where {TF<: AbstractFloat, D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}

Rotate all positions (x,y,z) and velocities (vx,vy,vz) in-place so that the z-axis aligns with the given `target_zaxis`. The rotation is orthogonal (det=1), preserving lengths and inner products. By construction, the new x-axis lies in the original xy-plane:
    x' = normalize(ẑ × l̂),   y' = z' × x',   z' = l̂

where ẑ = (0,0,1) and l̂ = normalized target_zaxis.

# Parameters
- `data_list :: V`: The array which contains all of the data that would be transfered
- `target_zaxis::NTuple{3,TF}`: Vector specifying the new z-axis (not required to be normalized).

# Keyword Arguments
- `inverse::Bool=false`: If `false`, apply the forward rotation (v' = R * v).
                         If `true`, apply the inverse rotation (v' = R' * v).

"""
function set_zaxis_orientation!(data_list :: V, target_zaxis :: NTuple{3, TF}; inverse :: Bool = false) where {TF<: AbstractFloat, D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
    Rdefault = _rotational_matrix(target_zaxis...)
    if inverse
        R = (Rdefault[1], Rdefault[4], Rdefault[7], Rdefault[2], Rdefault[5], Rdefault[8], Rdefault[3], Rdefault[6], Rdefault[9])
    else
        R = Rdefault
    end
    @inbounds for data in data_list
        _apply_zaxis_orientation!(data, R)
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
    add_Sigma!(data::PhantomRevealerDataFrame, smoothed_kernel::Type{K} = M4_spline; Identical_particles::Bool = true) where {K<:AbstractSPHKernel}

Compute and add the column density `Sigma` to the dataset.

This function evaluates the surface density at each particle by integrating the SPH kernel in 3D along the line-of-sight (z-axis), assuming an axisymmetric disk projected onto the xy-plane. The computed values are stored in `data[!,"Sigma"]`.

# Parameters
- `data::PhantomRevealerDataFrame`: The input dataset containing SPH particle properties.
- `smoothed_kernel::Type{K}`: The smoothing kernel type (default: `M4_spline`). Must be a subtype of `AbstractSPHKernel`.
- `Identical_particles::Bool=true`: If true, all particles are assumed to have the same mass (`data.params["mass"]`). If false, use `data[!,"m"]` as mass array.

# Notes
- The integration uses kernel weights from `LOSint_Smoothed_kernel_function`, assuming vertical integration through a 3D kernel.
"""
function add_Sigma!(data::PhantomRevealerDataFrame, smoothed_kernel :: Type{K} = M4_spline; Identical_particles::Bool=true) where {K<:AbstractSPHKernel}
    N = get_npart(data)
    Sigma = zeros(Float64, N)
    x = data[!,"x"]
    y = data[!,"y"]
    h = data[!,"h"]
    truncated_radius = KernelFunctionValid(smoothed_kernel, eltype(h)) .* h
    m = Identical_particles ? fill(data.params["mass"], N) : data[!,"m"]
    kdtree2d = Generate_KDtree(data, 2)

    points = [zeros(Float64, 2) for _ = 1:Threads.nthreads()]
    @inbounds @threads for i in 1:N
        tid = threadid()
        idxs = KDSearching_scratch[][tid]
        point = points[tid]
        empty!(idxs)
        xi = point[1] = x[i]
        yi = point[2] = y[i]
        inrange!(idxs, kdtree2d, point, truncated_radius[i])
        mW = 0.0
        @inbounds for j in idxs
            rab = sqrt((xi-x[j])^2 + (yi-y[j])^2)
            mW += m[i] * LOSint_Smoothed_kernel_function(smoothed_kernel, rab, h[i], Val(3))
        end 
        Sigma[i] = mW
    end
    data[!,"Sigma"] = Sigma
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
    M = data.params["Origin_sink_mass"][]
    data.dfdata[!, "vϕk"] = sqrt.((G * M) ./ data.dfdata[!, "s"])
    data.dfdata[!, "vrelϕ"] = data.dfdata[!, "vϕ"] - data.dfdata[!, "vϕk"]
end

"""
    add_Kepelarian_angular_velocity!(data::PhantomRevealerDataFrame)
Add the Kepelarian angular velocity for each particles.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_Kepelarian_angular_velocity!(data::PhantomRevealerDataFrame)
    xs = data[!,"x"]
    ys = data[!,"y"]
    zs = data[!,"z"]
    G = get_unit_G(data)
    M = data.params["Origin_sink_mass"][]
    μ = G * M
    Ωk = zeros(Float64, get_npart(data))
    @inbounds @simd for i in eachindex(Ωk)
        x = xs[i]
        y = ys[i]
        z = zs[i]
        r = sqrt(x*x + y*y + z*z)
        Ωk[i] = sqrt(μ / (r^3))
    end
    data.dfdata[!, "Ωk"] = Ωk
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
    M1 = data.params["Origin_sink_mass"][]
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
        COM2star!(data_cp, sink_particle_id)
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
