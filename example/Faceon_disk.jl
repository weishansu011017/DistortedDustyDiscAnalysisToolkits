using PhantomRevealer

"""
Analysis the data in a face-on annulus grid.
    by Wei-Shan Su,
    July 17, 2024
"""

function Disk_Faceon_interpolation(filepath :: String)
    @info "-------------------------------------------------------"
    # ------------------------------PARAMETER SETTING------------------------------
    Analysis_tag :: String = "Faceon_disk"
    # parameters of radial axis
    smin :: Float64 = 10.0
    smax :: Float64 = 175.0
    sn :: Int64 = 331
    
    # parameters of azimuthal axis
    ϕmin :: Float64 = 0.0
    ϕmax :: Float64 = 2π
    ϕn :: Int64 = 351
  
    # Other parameters
    column_names :: Vector{String} = ["e"]									# The quantities that would be interpolate except for surface density `Sigma`.
    mid_column_names :: Vector{String} = ["rho","vs","vϕ","vz","vϕ-vϕ_k"]   # The quantities that would be interpolate in the midplane.
    Origin_sinks_id :: Int64 = 1											# The id of sink at the middle of disk for analysis.
    smoothed_kernel :: Function = M6_spline
    h_mode :: String = "closest"
    DiskMass_OuterRadius :: Float64 = 175.0                                 # The outer radius of disk while estimating the mass of disk

    # Output setting
    File_prefix :: String = "Faceon"
    # -----------------------------------------------------------------------------
    # Packaging parameters
    sparams :: Tuple{Float64,Float64,Int} = (smin, smax, sn)
    ϕparams :: Tuple{Float64,Float64,Int} = (ϕmin, ϕmax, ϕn)
    columns_order :: Vector = ["Sigma", column_names..., (mid_column_names.*"m")...] # construct a ordered column names (Those quantities with taking mid-plane average will have a suffix "m")
    
    # Load file
    prdf_list :: Vector{PhantomRevealerDataFrame} = read_phantom(filepath, "all")
    COM2star!(prdf_list, prdf_list[end], Origin_sinks_id)
    datag :: PhantomRevealerDataFrame = prdf_list[1]
    datad :: PhantomRevealerDataFrame = prdf_list[2]
    sinks_data :: PhantomRevealerDataFrame = prdf_list[3]
    
    # Add extra quantity for interpolation 
    add_rho!(datag)
    add_rho!(datad)
    add_Sigma!(datag)
    add_Sigma!(datad)
    add_cylindrical!(datag)
    add_cylindrical!(datad)
    add_eccentricity!(datag)
    add_eccentricity!(datad)
    add_Kepelarian_azimuthal_velocity!(datag)
    add_Kepelarian_azimuthal_velocity!(datad)

    
    # Make the `params` field
    time :: Float64 = get_time(datag)
    params :: Dict{String, Any} = Analysis_params_recording(datag)
    params["GasDiskMass"] = get_disk_mass(datag, sinks_data, DiskMass_OuterRadius, Origin_sinks_id)
    params["DustDiskMass"] = get_disk_mass(datad, sinks_data, DiskMass_OuterRadius, Origin_sinks_id)
    
    # Calculate the midplane of gaseous disk
    if isempty(mid_column_names)
        midz_func = nothing
        midz_gbe = nothing
    else
        midz_func = Disk_2D_midplane_function_generator(datag,(10.0,smax, 166))
        # Transfer the midplane interpolation function into gridbackend.
        imin = [smin,ϕmin]
        imax = [smax,ϕmax]
        in = [sn,ϕn]
        midz_gbe = func2gbe(func=midz_func, imin, imax,in)
    end

    # Interpolation
    grids_gas :: Dict{String, gridbackend} = Disk_2D_FaceOn_Grid_analysis(datag, sparams, ϕparams, column_names=column_names, mid_column_names=mid_column_names, midz_func=midz_func, smoothed_kernel=smoothed_kernel, h_mode=h_mode)
    grids_dust :: Dict{String, gridbackend} = Disk_2D_FaceOn_Grid_analysis(datad, sparams, ϕparams, column_names=column_names, mid_column_names=mid_column_names, midz_func=midz_func, smoothed_kernel=smoothed_kernel, h_mode=h_mode)
    
    # Combine these dictionaries of grids with suffix
    final_dict = create_grids_dict(["g","d"], [grids_gas, grids_dust])

    # Packaging the result
    Result_buffer :: Analysis_result_buffer = Analysis_result_buffer(time, final_dict, columns_order, params, midz_gbe)
    Result :: Analysis_result = buffer2output(Result_buffer)
    
    # Write out the result to HDF5
    Write_HDF5(Analysis_tag,filepath, Result, File_prefix)
    @info "-------------------------------------------------------"
end

function main()
    # Commendline variable setting
    if length(ARGS) < 1
        println("Usage: julia Faceon_disk.jl <filename>")
        exit(1)
    end

    files = ARGS             

    First_logging()

    for file in files
        @info "File: $file"
        @time_and_print begin
            Disk_Faceon_interpolation(file)
        end 
    end

    @info "\nEnd analysis!"
end

main()