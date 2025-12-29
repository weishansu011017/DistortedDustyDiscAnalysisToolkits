"""
    read_GridDataset(filename::String="PRGridDataset.h5")

Read an HDF5 file produced by `write_GridDataset` / `write_GridBundle` and reconstruct a
fully-typed `GridDataset{L,TF,G}`.

This function validates the on-disk schema (`/params`, `/data`), parses `TF` and the grid
base type from `params`, infers `D` and `L` from the stored coordinate/axes vectors and
the number of grids, then constructs concrete grid types:

- `StructuredGrid{D,TF,V,A}` when `params[:grid_type] == :StructuredGrid`
- `GeneralGrid{D,TF,VG,VC}` when `params[:grid_type] == :GeneralGrid`

Shared vectors (`axes` or `coord`) are read once and reused across all grids, matching the
writer-side assumption that coordinate/axes vectors are shared among grids.

# Parameters
- `filename::String="PRGridDataset.h5"`  
  Input HDF5 filename.

# Returns
- `GridDataset{L,TF,G}`  
  A reconstructed dataset where:
  - `L` is inferred from the number of entries under `data/grids`
  - `TF` is inferred from `params[:dtype]`
  - `G` is inferred from `params[:grid_type]` and concretized from file contents
  - `D` is inferred from the number of stored coordinate/axes vectors
"""
function read_GridDataset(filename::String="PRGridDataset.h5")
    h5open(filename, "r") do f
        haskey(f, "params") || throw(ArgumentError("Schema error: missing /params"))
        haskey(f, "data")   || throw(ArgumentError("Schema error: missing /data"))
        pg = f["params"]
        dg = f["data"]

        params = _read_params(pg)
        haskey(params, :dtype)     || throw(ArgumentError("params missing :dtype"))
        haskey(params, :grid_type) || throw(ArgumentError("params missing :grid_type"))

        TF = _parse_float_type(String(params[:dtype]))
        base = Symbol(params[:grid_type])

        names_vec = Symbol.(read(dg, "names"))

        if base === :StructuredGrid
            shared_vecs, D = _read_shared_vectors(dg, "axes")
            size = ntuple(i -> length(shared_vecs[i]), Val(D))
            arrays, L = _read_dense_grids(dg, TF)
            length(names_vec) == L || throw(ArgumentError("names length != number of grids"))

            V = typeof(shared_vecs[1])          
            A = typeof(arrays[1])               
            G = PhantomRevealer.StructuredGrid{D, TF, V, A} 

            grids = ntuple(i -> G(arrays[i], shared_vecs, size), Val(L))
            names = ntuple(i -> names_vec[i], Val(L))

            gb = GridBundle{L, G}(grids, names)
            return GridDataset{L, TF, G}(gb, params)

        elseif base === :GeneralGrid
            shared_vecs, D = _read_shared_vectors(dg, "coord")
            arrays, L = _read_dense_grids(dg, TF)
            length(names_vec) == L || throw(ArgumentError("names length != number of grids"))

            VG = typeof(shared_vecs[1])         
            VC = typeof(shared_vecs)           
            G  = PhantomRevealer.GeneralGrid{D, TF, VG, VC} 

            grids = ntuple(i -> G(arrays[i], shared_vecs), Val(L))
            names = ntuple(i -> names_vec[i], Val(L))

            gb = GridBundle{L, G}(grids, names)
            return GridDataset{L, TF, G}(gb, params)

        else
            throw(ArgumentError("Unsupported params[:grid_type] = $(params[:grid_type])"))
        end
    end
end

function _ordered_keys(g::HDF5.Group)
    ks = String.(keys(g))
    if all(k -> k in ks, _COORDINATE)
        return collect(_COORDINATE[1:length(ks)])
    else
        sort!(ks)
        return ks
    end
end

function _read_params(pg::HDF5.Group)
    p = Dict{Symbol, Any}()
    for k in keys(pg)
        p[Symbol(k)] = read(pg, k)
    end
    return p
end

function _parse_float_type(s::AbstractString)
    s = strip(s)
    s == "Float16"  && return Float16
    s == "Float32"  && return Float32
    s == "Float64"  && return Float64
    s == "BigFloat" && return BigFloat
    throw(ArgumentError("Unsupported dtype string in params[:dtype]: $s"))
end

function _grid_type_from_params(s::AbstractString)
    s = Symbol(s)

    isdefined(PhantomRevealer, s) || throw(ArgumentError("Unknown grid_type in params[:grid_type]: $(repr(s))"))

    G = getfield(PhantomRevealer, s)

    (G <: PhantomRevealer.AbstractGrid) || throw(ArgumentError("params[:grid_type] is not a subtype of AbstractGrid: $(G)"))
    return G
end

function _read_shared_vectors(dg::HDF5.Group, groupname::String)
    haskey(dg, groupname) || throw(ArgumentError("Schema error: missing data/$groupname"))
    vg = dg[groupname]
    ord = _ordered_keys(vg)
    vecs = map(ord) do k
        read(vg, k)
    end
    return Tuple(vecs), length(vecs) # (shared_vecs, D)
end

function _read_dense_grids(dg::HDF5.Group, ::Type{TF}) where {TF <: AbstractFloat}
    haskey(dg, "grids") || throw(ArgumentError("Schema error: missing data/grids"))
    gg = dg["grids"]
    L = length(keys(gg))
    arrays = Vector{Array{TF}}(undef, L)
    @inbounds for i in 1:L
        arrays[i] = read(gg, string(i))
    end
    return arrays, L
end