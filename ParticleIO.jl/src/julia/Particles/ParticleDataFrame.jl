"""
The ParticleDataFrame data Structure
    by Wei-Shan Su,
    September 28, 2025

Those methods with prefix `add` would store the result into the original data, and prefix `get` would return the value. 
Becarful, the methods with suffix `!` would change the inner state of its first argument!
"""

"""
    struct ParticleDataFrame <: PartiaDataStructures
A data structure for storing the read dumpfile from SPH simulation.

# Fields
- `dfdata` :: The main data/particles information storage.
- `params` :: Global values stored in the dump file (time step, initial momentum, hfact, Courant factor, etc).
"""
struct ParticleDataFrame
    dfdata :: DataFrame
    params :: Dict
end

# Extending index operation
# Single row and single column indexing
@inline function Base.getindex(prdf::ParticleDataFrame, row_ind::Integer, col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[row_ind, col_ind]
end

# Multiple rows and single column indexing
@inline function Base.getindex(prdf::ParticleDataFrame, row_inds::AbstractVector, col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[row_inds, col_ind]
end

# All rows and a single column indexing
@inline function Base.getindex(prdf::ParticleDataFrame, ::Colon, col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[:, col_ind]
end

# Direct reference to a single column
@inline function Base.getindex(prdf::ParticleDataFrame, ::typeof(!), col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[!, col_ind]
end

# Multiple rows and multiple columns indexing
@inline function Base.getindex(prdf::ParticleDataFrame, row_inds::AbstractVector, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    return ParticleDataFrame(prdf.dfdata[row_inds, col_inds],prdf.params)
end

# Boolean vector indexing for rows and multiple columns
@inline function Base.getindex(prdf::ParticleDataFrame, bool_mask::AbstractVector{Bool}, col_ind::Union{Symbol, String, Int})
    return prdf.dfdata[bool_mask, col_ind]  # Vector
end

@inline function Base.getindex(prdf::ParticleDataFrame, bool_mask::AbstractVector{Bool}, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    return ParticleDataFrame(prdf.dfdata[bool_mask, col_inds], prdf.params)
end

@inline function Base.getindex(prdf::ParticleDataFrame, bool_mask::AbstractVector{Bool}, ::Colon)
    return ParticleDataFrame(prdf.dfdata[bool_mask, :], prdf.params)
end


# All rows and multiple columns indexing
@inline function Base.getindex(prdf::ParticleDataFrame, ::Colon, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    return ParticleDataFrame(prdf.dfdata[:, col_inds],prdf.params)
end


# Single row, all columns -> PRDF (1-row)
@inline function Base.getindex(prdf::ParticleDataFrame, row::Integer, ::Colon)
    return ParticleDataFrame(prdf.dfdata[row:row, :], prdf.params)
end

# Single row, multiple columns -> PRDF (1-row)
@inline function Base.getindex(prdf::ParticleDataFrame, row::Integer,
                               cols::Union{Vector{Symbol},Vector{String},Vector{Int}})
    return ParticleDataFrame(prdf.dfdata[row:row, cols], prdf.params)
end

# Multiple rows, all columns -> PRDF
@inline function Base.getindex(prdf::ParticleDataFrame, rows::AbstractVector, ::Colon)
    return ParticleDataFrame(prdf.dfdata[rows, :], prdf.params)
end

# All rows, all columns -> PRDF 
@inline function Base.getindex(prdf::ParticleDataFrame, ::Colon, ::Colon)
    return ParticleDataFrame(prdf.dfdata[:, :], prdf.params)
end

# Single row and single columns assignment
@inline function Base.setindex!(prdf::ParticleDataFrame, value, row_ind::Integer, col_ind::Union{String, Symbol})
    prdf.dfdata[row_ind, col_ind] = value
end

# Single row and multiple columns assignment
@inline function Base.setindex!(prdf::ParticleDataFrame, value, row_ind::Integer, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    prdf.dfdata[row_ind, col_inds] = value
end

# Multiple rows and multiple columns assignment
@inline function Base.setindex!(prdf::ParticleDataFrame, value, row_inds::AbstractVector, col_inds::Union{Vector{Symbol}, Vector{String}, Vector{Int}})
    prdf.dfdata[row_inds, col_inds] = value
end

# Extend `setindex!` to support `!` with column names for ParticleDataFrame
@inline function Base.setindex!(prdf::ParticleDataFrame, v::AbstractVector, ::typeof(!), col_ind::Union{Symbol, String})
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

# Extend `names` function for ParticleDataFrame
@inline function Base.names(prdf :: ParticleDataFrame)
    return names(prdf.dfdata)
end

# Other methods
"""
    print_params(data::ParticleDataFrame, pause::Bool=false)
Print out the `params` dictionary.

# Parameters
- `data :: ParticleDataFrame`: The SPH data that is stored in `ParticleDataFrame` 
- `pause :: Bool=false`: Pause the program after printing.
"""
function print_params(data::ParticleDataFrame, pause::Bool = false)
    allkeys = sort(collect(keys(data.params)))
    @inbounds for key in allkeys
        println("$(key) => $(data.params[key])")
    end
    if pause
        readline()
    end
end

"""
    @inline get_dim(data::ParticleDataFrame)
Get the dimension of simulation.

# Parameters
- `data :: ParticleDataFrame`: The SPH data that is stored in `ParticleDataFrame` 

#Returns
- 'Int64': The dimension of simulation of SPH data.
"""
@inline function get_dim(data::ParticleDataFrame)
    return hasproperty(data.dfdata, "z") ? 3 : 2
end

"""
    @inline get_time(data::ParticleDataFrame)
Get the time of simulation in code unit.

# Parameters
- `data :: ParticleDataFrame`: The SPH data that is stored in `ParticleDataFrame` 

#Returns
- 'Float64': The time of simulation.
"""
@inline function get_time(data::ParticleDataFrame)
    if haskey(data.params,:time)
        return data.params[:time]
    elseif haskey(data.params,:Time)
        return data.params[:Time]
    else
        return 
    end
end

"""
    get_code_unit(data::ParticleDataFrame, ::Type{TF}=Float64) where {TF<:AbstractFloat}

Extract Phantom code units from `data.params` and return them as a typed dictionary.

The function reads the required unit scalars `:udist`, `:umass`, `:utime`, and `:umagfd`
from `data.params`, converts each value to `TF`, and returns a `Dict{Symbol,TF}`.

# Parameters
- `data::ParticleDataFrame`: Input particle dataset whose `params` stores code-unit scalars.
- `::Type{TF}=Float64`: Target floating-point type used for the returned unit values.

# Returns
- `Dict{Symbol,TF}`: Dictionary containing the four required code units:
  `:umass`, `:udist`, `:utime`, `:umagfd`.
"""
@inline function get_code_unit(data::ParticleDataFrame, :: Type{TF} = Float64) where {TF <: AbstractFloat}
    udist = TF(data.params[:udist])
    umass = TF(data.params[:umass])
    utime = TF(data.params[:utime])
    umagfd = TF(data.params[:umagfd])

    code_units = Dict{Symbol, TF}()
    code_units[:umass] = umass
    code_units[:udist] = udist
    code_units[:utime] = utime
    code_units[:umagfd] = umagfd

    return code_units
end

"""
    @inline get_npart(data::ParticleDataFrame)
Get the number of particles in `data`.

# Parameters
- `data :: ParticleDataFrame`: The SPH data that is stored in `ParticleDataFrame` 

# Returns
-`Int64`: The number of particles.
"""
@inline function get_npart(data::ParticleDataFrame)
    return nrow(data.dfdata)
end

"""
    @inline get_init_npart(data::ParticleDataFrame)
Get the number of particles in `data`.

# Parameters
- `data :: ParticleDataFrame`: The SPH data that is stored in `ParticleDataFrame` 

# Returns
-`Int64`: The number of particles.
"""
@inline function get_init_npart(data::ParticleDataFrame)
    itype = data.params[:itype]
    return data.params[Symbol(string("npartoftype", itype == 1 ? "" : string("_", itype), ))]
end

"""
    @inline get_unit_G(data::ParticleDataFrame)
Get the Gravitational constant G in code unit.

# Parameters
- `data :: ParticleDataFrame`: The SPH data that is stored in `ParticleDataFrame` 

# Returns
-`Float64`: The Gravitational constant G in code unit.
"""
@inline function get_unit_G(data::ParticleDataFrame) :: Float64
    params = data.params
    udist = params[:udist]
    umass = params[:umass]
    utime = params[:utime]
    G_cgs :: Float64 = udist^3 * utime^(-2) * umass^(-1)
    G ::Float64 = G_cgs / 6.672041000000001e-8
    return G
end