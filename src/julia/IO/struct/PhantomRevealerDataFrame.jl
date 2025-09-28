"""
The PhantomRevealerDataFrame data Structure
    by Wei-Shan Su,
    September 28, 2025

Those methods with prefix `add` would store the result into the original data, and prefix `get` would return the value. 
Becarful, the methods with suffix `!` would change the inner state of its first argument!
"""

"""
    struct PhantomRevealerDataFrame <: PhantomRevealerDataStructures
A data structure for storing the read dumpfile from SPH simulation.

# Fields
- `dfdata` :: The main data/particles information storage.
- `params` :: Global values stored in the dump file (time step, initial momentum, hfact, Courant factor, etc).
"""
struct PhantomRevealerDataFrame
    dfdata :: DataFrame
    params :: Dict
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

# Other methods
"""
    print_params(data::PhantomRevealerDataFrame, pause::Bool=false)
Print out the `params` dictionary.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `pause :: Bool=false`: Pause the program after printing.
"""
function print_params(data::PhantomRevealerDataFrame, pause::Bool = false)
    allkeys = sort(collect(keys(data.params)))
    @inbounds for key in allkeys
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
    if haskey(data.params,:time)
        return data.params[:time]
    elseif haskey(data.params,:Time)
        return data.params[:Time]
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
@inline function get_code_unit(data::PhantomRevealerDataFrame)
    udist = data.params[:udist]
    umass = data.params[:umass]
    utime = data.params[:utime]
    umagfd = data.params[:umagfd]
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
    itype = data.params[:itype]
    return data.params[Symbol(string("npartoftype", itype == 1 ? "" : string("_", itype), ))]
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
    udist = params[:udist]
    umass = params[:umass]
    utime = params[:utime]
    G_cgs :: Float64 = udist^3 * utime^(-2) * umass^(-1)
    G ::Float64 = G_cgs / 6.672041000000001e-8
    return G
end