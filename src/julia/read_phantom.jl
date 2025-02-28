function _read_fortran_block(fp, bytesize)
    """ Helper function to read Fortran data, which is also buffered before and after by 4 bytes."""
    start_tag = read(fp, 4)
    data = read(fp, bytesize)
    end_tag = read(fp, 4)

    if (start_tag != end_tag)
        error("Fortran tags mismatch.")
    end
    return data
end

function _read_capture_pattern(fp)
    """ Phantom dump validation plus default real and int sizes."""
    # 4 byte Fortran tag
    start_tag = read(fp, 4)

    def_int_dtype = Int32
    def_real_dtype = Float32


    # integer 1 == 060769
    i1 = read(fp, def_int_dtype)

    # assert i1. Try 8-byte int if fails.
    if (i1 != 60769)
        seek(fp, position(fp) - 8)  # rewind based on current file position

        def_int_dtype = Int64

        i1 = read(fp, Int64)
        # retry assert
        if i1 != 60769
            error("CapturePatternError: i1 mismatch. Is this a Phantom data file?")
        end
    end

    # real 1 == integer 2 == 060878
    r1 = read(fp, def_real_dtype)
    i2 = read(fp, def_int_dtype)

    if ((i2 != 60878) || !(Float32(i2) == r1))
        seek(fp, position(fp) - 8)  # rewind based on current file position

        def_real_dtype = Float64
        r1 = read(fp, def_real_dtype)
        i2 = read(fp, def_int_dtype)

        # retry assert
        if ((i2 != 60878) || !(Float32(i2) == r1))
            error("CapturePatternError: i1 mismatch. Is this a Phantom data file?")
        end
    end

    # iversion -- we don't actually check this
    iversion = read(fp, def_int_dtype)

    # integer 3 == 690706
    i3 = read(fp, def_int_dtype)
    if (i3 != 690706)
        error("CapturePatternError: i3 mismatch. Is this a Phantom data file?")
    end

    # 4 byte Fortran tag
    end_tag = read(fp, 4)

    # assert tags equal
    if (start_tag != end_tag)
        error("CapturePatternError: Fortran tags mismatch. Is this a Phantom data file?")
    end

    return def_int_dtype, def_real_dtype
end

function _read_file_identifier(fp)
    """ Read the 100 character file identifier. Contains code version and date information. """
    data = _read_fortran_block(fp, 100)
    return strip(String(data))
end

function _rename_duplicates(keys)
    seen = Dict()

    for (i, key) in enumerate(keys)
        if !haskey(seen, key)
            seen[key] = 1
        else
            seen[key] += 1
            keys[i] = keys[i] * "_$(seen[key])"
        end
    end
    return keys
end

function _read_global_header_block(fp, dtype)
    nvars_data = _read_fortran_block(fp, 4)
    nvars = reinterpret(Int32, nvars_data)[1]
    keys = String[]
    data = dtype[]
    if nvars > 0
        # each tag is 16 characters in length
        keys_data = _read_fortran_block(fp, 16 * nvars)
        keys = [strip(String(keys_data[i:i+15])) for i = 1:16:length(keys_data)]

        data_block = _read_fortran_block(fp, sizeof(dtype) * nvars)
        data = reinterpret(dtype, data_block)
    end
    return keys, data
end

function _read_global_header(fp, def_int_dtype, def_real_dtype)
    """ Read global variables. """
    dtypes = [def_int_dtype, Int8, Int16, Int32, Int64, def_real_dtype, Float32, Float64]
    keys = String[]
    data = []
    for dtype in dtypes
        new_keys, new_data = _read_global_header_block(fp, dtype)
        append!(keys, new_keys)
        append!(data, new_data)
    end
    keys = _rename_duplicates(keys)
    global_vars = Dict{String,Any}()
    for (i, key) in enumerate(keys)
        global_vars[key] = data[i]
    end

    return global_vars
end

function _read_array_block(fp, df, n, nums, def_int_dtype, def_real_dtype)

    dtypes = [def_int_dtype, Int8, Int16, Int32, Int64, def_real_dtype, Float32, Float64]
    for i in eachindex(nums)
        dtype = dtypes[i]
        for j = 1:nums[i]
            tag = strip(String(_read_fortran_block(fp, 16)))
            raw_data = _read_fortran_block(fp, sizeof(dtype) * n)
            data = reinterpret(dtype, raw_data)
            df[!, tag] = data
        end
    end
    return df
end

function _read_array_blocks(fp, def_int_dtype, def_real_dtype)
    """ Read particle data. Block 2 is always for sink particles?"""

    raw_data = _read_fortran_block(fp, 4)
    nblocks = reinterpret(Int32, raw_data)[1]

    n = []
    nums = []
    for i = 1:nblocks
        start_tag = read(fp, 4)

        raw_data = read(fp, 8)
        value = reinterpret(Int64, raw_data)[1]
        push!(n, value)

        raw_data = read(fp, 32)
        values = reinterpret(Int32, raw_data)
        selected_values = values[1:8]
        push!(nums, selected_values)

        end_tag = read(fp, 4)
        if (start_tag != end_tag)
            error("Fortran tags mismatch in array blocks.")
        end
    end
    df = DataFrame()
    df_sinks = DataFrame()
    for i = 1:nblocks
        # This assumes the second block is only for sink particles.
        # I believe this is a valid assumption as I think this is what splash assumes.
        # For now we will just append sinks to the end of the data frame.
        if i == 2
            df_sinks = _read_array_block(
                fp,
                df_sinks,
                n[i],
                nums[i],
                def_int_dtype,
                def_real_dtype,
            )
        else
            df = _read_array_block(fp, df, n[i], nums[i], def_int_dtype, def_real_dtype)
        end
    end
    return df, df_sinks
end

"""
    function read_phantom(filename::String, separate_types::String = "sinks", ignore_inactive::Bool = true) :: Union{PhantomRevealerDataFrame,Vector}

Note: This implementation is inspired by the function of the same name in the 'saracen' python package.
Check out the documentation of `saracen <https://sarracen.readthedocs.io/en/latest/index.html>`_

Read data from a Phantom dump file.

This reads the native binary format of Phantom dump files, which in turn were derived from the binary file format
used by sphNG.

Global values stored in the dump file (time step, initial momentum, hfact, Courant factor, etc) are stored within the
data frame in the field `params`.

# Parameters
- `filename::String`: Name of the file to be loaded.
- `separate_types::String = "sinks"`: Whether to separate different particle types into several dataframes. ``None`` returns all particle types in one
        data frame. '`sinks`' separates sink particles into a second dataframe, and '`all`' returns all particle types in
        different dataframes.
- `ignore_inactive::Bool = true`: If true, particles with negative smoothing length will not be read on import. These are
        typically particles that have been accreted onto a sink particle or are otherwise inactive.

# Returns
- `PhantomRevealerDataFrame` or `Vector`: PhantomRevealerDataFrame or Array of PhantomRevealerDataFrames

# Examples
## Example 1: By default, SPH particles are grouped into one data frame and sink particles into a second data frame.
```julia
prdf, prdf_sinks = read_phantom("dumpfile_00000")
```
## Example 2: A dump file containing multiple particle types, say gas + dust + sinks, can separated into their own data frames
    by specifying `separate_types="all"`.
```julia
prdf_gas, prdf_dust, prdf_sinks = read_phantom("multiple_types_00000", separate_types="all")
```

# Notes
See the `Phantom documentation <https://phantomsph.readthedocs.io/en/latest/dumpfile.html>`_ for a full description
    of the Phantom binary file format.
You should also check out the function of the same name in `sarracen documentation <https://sarracen.readthedocs.io/en/latest/api/sarracen.read_phantom.html>`_ 
    for further investigation.
"""
function read_phantom(
    filename::String,
    separate_types::String = "sinks",
    ignore_inactive::Bool = true,
) :: Union{PhantomRevealerDataFrame,Vector}
    open(filename, "r") do fp
        def_int_dtype, def_real_dtype = _read_capture_pattern(fp)
        file_identifier = _read_file_identifier(fp)

        header_vars = _read_global_header(fp, def_int_dtype, def_real_dtype)
        header_vars["file_identifier"] = file_identifier
        header_vars["COM_coordinate"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #General Coordinate of origin
        header_vars["Origin_sink_id"] = -1
        header_vars["Origin_sink_mass"] = NaN

        df, df_sinks = _read_array_blocks(fp, def_int_dtype, def_real_dtype)

        if ignore_inactive
            df = df[df[!, :"h"].>0, :]
        end

        if (separate_types == "all") &&
           (hasproperty(df, :"itype")) &&
           (length(unique(df.itype)) > 1)
            df_list = []
            for group in groupby(df, :"itype")
                itype::Int = Int(group[!, "itype"][1])
                mass_key = itype == 1 ? "massoftype" : "massoftype_$(itype)"
                group_clean = select(
                    group,
                    [col for col in names(group) if !any(ismissing, group[!, col])],
                )
                params_group = merge(header_vars, Dict("mass" => header_vars[mass_key]))
                push!(df_list, PhantomRevealerDataFrame(group_clean, params_group))
            end
            if !(isempty(df_sinks))
                push!(df_list, PhantomRevealerDataFrame(df_sinks, header_vars))
            end
            return df_list
        end

        if ((separate_types == "sinks") || (separate_types == "all")) &&
           !(isempty(df_sinks))
            params = merge(header_vars, Dict("mass" => header_vars["massoftype"]))
            df = PhantomRevealerDataFrame(df, params)
            df_sinks = PhantomRevealerDataFrame(df_sinks, header_vars)
            return [df, df_sinks]
        end
        combined_df = vcat(df, df_sinks, cols = :union)
        params = merge(header_vars, Dict("mass" => header_vars["massoftype"]))
        df = PhantomRevealerDataFrame(combined_df, params)
        return df
    end
end
