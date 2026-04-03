const _COORDINATE = ("x", "y", "z")

"""
    write_GridDataset(gd::GridDataset{L,TF,G}, filename::String="PRGridDataset.h5") where {L,D,TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D,VG}, G<:PointSamples{D,TF,VG,VC}}

Serialize a `GridDataset` and write it to an HDF5 file.  
All grids in the dataset **must share the same coordinate vectors**, verified by
strict identity comparison (`===`). If any grid does not share coordinates with
the first grid, an `ArgumentError` is thrown.

The resulting HDF5 file has the following structure:

- `params/`  
  Stores all key–value pairs from `gd.params`
- `data/names`  
  Names associated with each grid
- `data/coord/<x|y|z>`  
  Shared coordinate vectors (taken from the first grid)
- `data/grids/<i>`  
  The numerical array of the `i`-th grid

# Parameters
- `gd::GridDataset{L,TF,G}`  
  A dataset containing multiple `PointSamples` objects that must share identical coordinates
- `filename::String="PRGridDataset.h5"`  
  Output HDF5 filename

"""
function write_GridDataset(gd::GridDataset{L,TF,G}, filename::String="PRGridDataset.h5") where {L,D,TF<:AbstractFloat, VG<:AbstractVector{TF}, VC<:NTuple{D,VG}, G <: PointSamples{D,TF,VG,VC}}
    # Check whether all the grid share the same coor
    g1 = gd.data.grids[1]
    @inbounds for i in 2:L
        gi = gd.data.grids[i]
        for d in 1:D
            (gi.coor[d] === g1.coor[d]) || throw(ArgumentError("PointSamples coor not shared: grid=$i dim=$d"))
        end
    end
    # Write HDF5
    h5open(filename, "w") do f
        # schema:
        ## params
        ## data/names
        ## data/grids/<i>/...(grid data)
        ## data/coord/<x> or <y> (or <z> if D = 3)/...(coor vector)

        # 1) params
        pg = create_group(f, "params")
        for (k,v) in gd.params
            write(pg, String(k), v)
        end

        # 2) data
        dg = create_group(f, "data")
        write(dg, "names", collect(String.(gd.data.names)))

        # shared coordinates (take from first grid)
        coordg = create_group(dg, "coord")
        g1 = gd.data.grids[1]
        @inbounds for d in 1:D
            write(coordg, _COORDINATE[d], g1.coor[d])
        end

        # values
        grids_g = create_group(dg, "grids")
        @inbounds for i in 1:L
            write(grids_g, string(i), gd.data.grids[i].grid)
        end
    end
    return nothing
end

"""
    write_GridDataset(gd::GridDataset{L,TF,G}, filename::String="PRGridDataset.h5") where
{L,D,TF<:AbstractFloat, V<:AbstractVector{TF}, A<:AbstractArray{TF,D}, G<:StructuredGrid{D,TF,V,A}}

Serialize a `GridDataset` composed of `StructuredGrid`s and write it to an HDF5 file.  
All grids in the dataset **must share the same axes vectors**, verified by strict
identity comparison (`===`). If any grid does not share axes with the first grid,
an `ArgumentError` is thrown.

The resulting HDF5 file has the following structure:

- `params/`  
  Stores all key–value pairs from `gd.params`
- `data/names`  
  Names associated with each grid
- `data/axes/<x|y|z>`  
  Shared axes vectors (taken from the first grid)
- `data/grids/<i>`  
  The numerical array of the `i`-th grid

# Parameters
- `gd::GridDataset{L,TF,G}`  
  A dataset containing multiple `StructuredGrid` objects that must share identical axes
- `filename::String="PRGridDataset.h5"`  
  Output HDF5 filename
"""
function write_GridDataset(gd::GridDataset{L,TF,G}, filename::String="PRGridDataset.h5") where {L,D,TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}, G <: StructuredGrid{D,TF,V, A}}
    # Check whether all the grid share the same coor
    g1 = gd.data.grids[1]
    @inbounds for i in 2:L
        gi = gd.data.grids[i]
        for d in 1:D
            (gi.axes[d] === g1.axes[d]) || throw(ArgumentError("StructuredGrid axes not shared: grid=$i dim=$d"))
        end
    end

    # Write HDF5
    h5open(filename, "w") do f
        # schema:
        ## params
        ## data/names
        ## data/grids/<i>/...(grid data)
        ## data/axes/<x> or <y> (or <z> if D = 3)/...(axes vector)

        # 1) params
        pg = create_group(f, "params")
        for (k,v) in gd.params
            write(pg, String(k), v)
        end

        # 2) data
        dg = create_group(f, "data")
        write(dg, "names", collect(String.(gd.data.names)))

        # shared axes
        axesg = create_group(dg, "axes")
        @inbounds for d in 1:D
            write(axesg, _COORDINATE[d], g1.axes[d])
        end

        # values
        grids_g = create_group(dg, "grids")
        @inbounds for i in 1:L
            write(grids_g, string(i), gd.data.grids[i].grid)
        end
    end
    return nothing
end

"""
    write_GridBundle(
        gb::GridBundle{L,G},
        filename::String="PRGridDataset.h5";
        code_units::Dict{Symbol,TF},
        operation_name::String="",
        params::Union{Nothing,Dict{Symbol,Union{String,Int,Bool,TF}}}=nothing
    ) where {L,TF<:AbstractFloat,G<:AbstractGrid{TF}}

Construct a `GridDataset` from a `GridBundle` and write it to an HDF5 file.
This function augments metadata, validates required code units, and records
system and schema information before delegating the actual I/O to
`write_GridDataset`.

The following metadata are automatically added or enforced:

- Schema identification (`schema_name`, `schema_id`)
- System kernel (`Sys.KERNEL`)
- Floating-point data type (`TF`)
- Grid concrete type
- Code units
- File identifier derived from `operation_name`

The `code_units` dictionary **must** contain all required unit keys; otherwise,
an `ArgumentError` is thrown.

# Parameters
- `gb::GridBundle{L,G}`  
  A bundle of grids to be serialized
- `filename::String="PRGridDataset.h5"`  
  Output HDF5 filename

# Keyword Arguments
| Name             | Type                                                         | Default | Description |
|------------------|--------------------------------------------------------------|---------|-------------|
| `code_units`     | `Dict{Symbol,TF}`                                            | —       | Code unit definitions; must include `:umass`, `:udist`, `:utime`, `:umagfd` |
| `operation_name` | `String`                                                     | `""`    | Operation name used to generate a unique file identifier |
| `params`         | `Union{Nothing,Dict{Symbol,Union{String,Int,Bool,TF}}}`      | `nothing` | Optional additional metadata stored in the dataset |

"""
function write_GridBundle(gb :: GridBundle{L, G}, filename :: String = "PRGridDataset.h5"; code_units :: Dict{Symbol, TF}, operation_name :: String = "", params :: Union{Nothing, Dict{Symbol, Union{String, Int, Bool, TF}}} = nothing) where {L, TF <: AbstractFloat, G <: AbstractGrid{TF}}
    p = isnothing(params) ? GridDataset_params_TYPE(TF)() : copy(params)
    
    # Check required code units
    required = (:umass, :udist, :utime, :umagfd)
    notfound  = Tuple(k for k in required if !haskey(code_units, k))
    isempty(notfound) || throw(ArgumentError("code_units missing required keys: $(notfound)"))
    
    # Record the name of schema
    p[:schema_name]      = "GridDataset"
    p[:schema_id]        = "GridDataset.v1"         

    # Record current system
    p[:system_kernel] = string(Sys.KERNEL)

    # Record data type
    p[:dtype] = string(TF)

    # Record type of grid
    p[:grid_type] = string(nameof(G))

    # Record code units
    for (k,v) in code_units
        p[k] = v
    end

    # Add file_identifier
    p[:file_identifier] = file_identifier(operation_name)

    # Construct a `GridDataset`
    gd = GridDataset{L, TF, G}(gb, p)

    write_GridDataset(gd, filename)
    return nothing
end
