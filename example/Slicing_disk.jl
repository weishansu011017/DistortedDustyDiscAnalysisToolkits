using PhantomRevealer
using LaTeXStrings
using CairoMakie

"""
Slice the disk for checking the edge-on vertical structure.
    by Wei-Shan Su,
    Febuary 28, 2025
"""

function Slicing_disk(file::String)
    @info "-------------------------------------------------------"
    # ------------------------------PARAMETER SETTING------------------------------
    Analysis_tag :: String = "Slicing_disk"
    # General setting
    smoothed_kernel :: Function = M6_spline                                             # Allowed function: M4_spline, M5_spline, M6_spline, C2_Wendland, C4_Wendland, C6_Wendland
    h_mode :: String = "closest"                                                        # Allowed mode: "mean", "closest"
    DiskMass_OuterRadius :: Float64 = 140.0                                             # The outer radius of disk while estimating the mass of disk

    # Output setting
    File_prefix :: String = "Slice"
    HDF5 :: Bool = false                                                                 # Extract the final result as HDF5 format
    figure :: Bool = true                                                              # Extract the final result as figure

    # Disk generating setting (Base on cylindrical coordinate (s,ϕ,z))
    Origin_sinks_id = 1                                                                 # The id of sink which is located at the middle of disk.
    ## s
    smin :: Float64 = 10.0                                                              # The minimum radius of disk.
    smax :: Float64  = 140.0                                                            # The maximum radius of disk.
    sn :: Int64 = 141                                                                   # The number of separation alone the radial direction on the disk.

    ## ϕ
    ϕmin :: Float64  = 0.0                                                              # The lower bound of angle for analysis.
    ϕmax :: Float64  = 2π                                                               # The upper bound of angle for analysis.
    ϕn :: Int64 = 24                                                                    # The number of separation alone the azimuthal direction on the disk.
    
    ## z
    zmin :: Float64  = -19.0                                                            # The lower bound of height from the z = 0 plane.
    zmax :: Float64  = 19.0                                                             # The upper bound of height from the z = 0 plane.
    zn :: Int64 = 120                                                                   # The number of separation alone the vertical direction on the disk.
    
    column_names :: Vector{String} = ["vs", "vϕ", "vz"]                                 # The columns that would be interpolated
    gradient_column_names :: Vector{String} = []                                        # The columns that would be interpolated its gradient value
    divergence_column_names :: Vector{String} = ["v"]                                   # The vector quantities that would be interpolated its divergence
    curl_column_names :: Vector{String} = ["v"]                                         # The vector quantities that would be interpolated its curl
    
    Figure_format :: String = "pdf"
    figsize :: Tuple = (8,8)
    dpi = 450
    slabel = latexstring(L"$r$ [au]")
    zlabel = latexstring(L"$z$ [au]")
    colormap_rho :: String = "RdYlGn"
    colormap_v :: String = "RdBu"
    colormap_vorticity :: String = "RdBu"
    clim_rho :: Vector = [7.1e-18,1e-14]
    clim_vs :: Vector = [-1.5e+5,1.5e+5]
    clim_vz :: Vector = [-1.5e+5,1.5e+5]
    clim_vorticity :: Vector = [-25.5,25.5]
    Slice_ϕ :: Union{Nothing,Float64} = 0.0                                             # The azimuthal angle of vertical structure (in degree). Taking azimuthally averaging if `nothing`.
    sreverse :: Bool = true
    # -----------------------------------------------------------------------------
    # Setup info
    initial_logging(get_analysis_info(file))

    # Packaging the parameters
    sparams :: Tuple{Float64,Float64,Int} = (smin, smax, sn)
    ϕparams :: Tuple{Float64,Float64,Int} = (ϕmin, ϕmax, ϕn)
    zparams :: Tuple{Float64,Float64,Int} = (zmin, zmax, zn)

    # Modified the column name for constructing the output file
    modified_grad_column_names :: Vector{String} = []
    for gvcolumn_name in gradient_column_names
        push!(modified_grad_column_names, "∇$(gvcolumn_name)s")
        push!(modified_grad_column_names, "∇$(gvcolumn_name)ϕ")
        push!(modified_grad_column_names, "∇$(gvcolumn_name)z")
    end
    modified_diver_column_names :: Vector{String} = []
    for dvcolumn_name in divergence_column_names
        push!(modified_diver_column_names, "∇⋅$(dvcolumn_name)")
    end
    modified_curl_column_names :: Vector{String} = []
    for cvcolumn_name in curl_column_names
        push!(modified_curl_column_names, "∇×$(cvcolumn_name)s")
        push!(modified_curl_column_names, "∇×$(cvcolumn_name)ϕ")
        push!(modified_curl_column_names, "∇×$(cvcolumn_name)z")
    end
    columns_order :: Vector = ["rho", "∇rhos", "∇rhoϕ","∇rhoz", column_names...,modified_grad_column_names...,modified_diver_column_names...,modified_curl_column_names...] # construct a ordered column names 

    # Read file
    prdf_list :: Vector{PhantomRevealerDataFrame} = read_phantom(file,"all")
    sinks_data :: PhantomRevealerDataFrame = prdf_list[end]
    COM2star!(prdf_list, sinks_data ,Origin_sinks_id)
    datag :: PhantomRevealerDataFrame = prdf_list[1]
    datad :: PhantomRevealerDataFrame = prdf_list[2]
    
    # Get time stamp of the dumpfile
    time = get_time(datag)

    # Get params
    params = Analysis_params_recording(datag)
    diskg_mass = get_disk_mass(datag, sinks_data, DiskMass_OuterRadius , Origin_sinks_id)
    diskd_mass = get_disk_mass(datad, sinks_data, DiskMass_OuterRadius , Origin_sinks_id)
    params["MassGaseousDisk"] = diskg_mass
    params["MassDustyDisk"] = diskd_mass

    # Add the cylindrical parameters
    add_cylindrical!(datag)
    add_cylindrical!(datad)

    # Main_analysis
    grids_dictg :: Dict{String, gridbackend} = Disk_3D_Grid_analysis(datag, sparams, ϕparams, zparams, column_names=column_names, gradient_column_names=gradient_column_names, divergence_column_names=divergence_column_names, curl_column_names=curl_column_names, smoothed_kernel=smoothed_kernel, h_mode=h_mode)
    grids_dictd :: Dict{String, gridbackend} = Disk_3D_Grid_analysis(datad, sparams, ϕparams, zparams, column_names=column_names, gradient_column_names=gradient_column_names, divergence_column_names=divergence_column_names, curl_column_names=curl_column_names, smoothed_kernel=smoothed_kernel, h_mode=h_mode)

    # Packaging the grids dictionary
    final_dict = create_grids_dict(["g","d"], [grids_dictg, grids_dictd])

    # Packaging the result
    Result_buffer :: Analysis_result_buffer = Analysis_result_buffer(time, final_dict, columns_order,params)
    Result :: Analysis_result = buffer2output(Result_buffer)

    # Write the file to HDF5
    if HDF5
        Write_HDF5(Analysis_tag,file, Result, File_prefix)
    end

    # Construct the figure
    if figure
        # Get the number for new filename
        filename = splitext(file)[1]
        number_data = extract_number(filename)

        # Modified Colormap to have white color at the bottom
        colormap_rho_modified = colormap_with_base(colormap_rho)

        # Packaging parameters
        clims = [clim_rho, clim_vs,clim_vz, clim_vorticity]

        transfer_cgs!(Result)

        # Finding column_index
        target_columns :: Vector{String} = ["rho_g","rho_d","vs_g","vs_d","vz_g","vz_d","∇×vϕ_g","∇×vϕ_d"]
        target_column_index :: Vector{Int64} = zeros(Int64, length(target_columns))
        for key in keys(Result.column_names)
            value = strip(Result.column_names[key])  
            actual_name = strip(split(value)[end], ['[', ']'])  
            for (i, target) in enumerate(target_columns)
                if actual_name == target 
                    target_column_index[i] = key
                end
            end
        end

        # Setup plotting target
        timestamp = Result.time
        s = Result.axes[1]
        ϕ = Result.axes[2]
        z = Result.axes[3]
        rho_label = L"$\rho$ [g\ cm$^{-3}$]"
        vs_label = L"$v_r$ [cm\ s$^{-1}$]"
        vz_label = L"$v_z$ [cm\ s$^{-1}$]"
        Vorticity_label = L"Vorticity [$10^{-10}$ s$^{-1}$]"

        # Determining the azimuthal angle
        if !(isnothing(Slice_ϕ))
            Slice_ϕ_index = value2closestvalueindex(ϕ,Slice_ϕ*(π/180))
            Slice_ϕ_real = ϕ[Slice_ϕ_index]*(180/π)
            anatonate_label = latexstring(L"$t = ",Int64(round(timestamp)), L"$ yr, $\phi$ = ", Int64(round(Slice_ϕ_real)),L"^{\circ}")
        else
            anatonate_label = latexstring(L"$t = ",Int64(round(timestamp)), L"$ yr")
        end
        
        reduced_array = Vector{Array{Float64}}(undef, length(target_columns))
        for (i,index) in enumerate(target_column_index)
            if isnothing(Slice_ϕ)
                reduced_array[i] = grid_reduction(Result.data_dict[index],2)
            else
                reduced_array[i] = Result.data_dict[index][:,Slice_ϕ_index,:]
            end
        end

        rhog, rhod, vsg, vsd, vzg, vzd, curlvϕg, curlvϕd = reduced_array
        threadshold = 1.15*clims[1][1]
        vsg[rhog .< threadshold] .= NaN
        vsd[rhod .< threadshold] .= NaN
        vzg[rhog .< threadshold] .= NaN
        vzd[rhod .< threadshold] .= NaN
        curlvϕg[rhog .< threadshold] .= NaN
        curlvϕd[rhod .< threadshold] .= NaN


        # Preparing plotting backend
        activate_backend("Cairo")
        Fax = FigureAxes(4,2,figsize=figsize,sharex=true, sharey=true)
        heatmap!(Fax.axes[1,1],s ,z ,rhog ,colormap=colormap_rho_modified, colorrange=clim_rho, colorscale=log10)
        set_annotation!(Fax,(1,1),latexstring(anatonate_label," (Gas)"),fontsize=14)
        heatmap!(Fax.axes[1,2],s ,z ,rhod ,colormap=colormap_rho_modified, colorrange=clim_rho, colorscale=log10)
        set_annotation!(Fax,(1,2),latexstring(anatonate_label," (Dust)"),fontsize=14)
        set_colorbar!(Fax,(1,1),clabel=rho_label,link_row_colormap=true)
        Fax.axes[1,1].xreversed = sreverse
        Fax.axes[1,2].xreversed = sreverse

        heatmap!(Fax.axes[2,1],s ,z ,vsg ,colormap= colormap_v, colorrange=clim_vs, colorscale=Symlog10Scale(clim_vs...))
        set_annotation!(Fax,(2,1),latexstring(anatonate_label," (Gas)"),fontsize=14)
        heatmap!(Fax.axes[2,2],s ,z ,vsd ,colormap= colormap_v, colorrange=clim_vs, colorscale=Symlog10Scale(clim_vs...))
        set_annotation!(Fax,(2,2),latexstring(anatonate_label," (Dust)"),fontsize=14)
        set_colorbar!(Fax,(2,1),clabel=vs_label,link_row_colormap=true)
        Fax.axes[2,1].xreversed = sreverse
        Fax.axes[2,2].xreversed = sreverse

        heatmap!(Fax.axes[3,1],s ,z ,vzg ,colormap= colormap_v, colorrange=clim_vz, colorscale=Symlog10Scale(clim_vz...))
        set_annotation!(Fax,(3,1),latexstring(anatonate_label," (Gas)"),fontsize=14)
        heatmap!(Fax.axes[3,2],s ,z ,vzd ,colormap= colormap_v, colorrange=clim_vz, colorscale=Symlog10Scale(clim_vz...))
        set_annotation!(Fax,(3,2),latexstring(anatonate_label," (Dust)"),fontsize=14)
        set_colorbar!(Fax,(3,1),clabel=vz_label,link_row_colormap=true)
        Fax.axes[3,1].xreversed = sreverse
        Fax.axes[3,2].xreversed = sreverse

        heatmap!(Fax.axes[4,1],s ,z ,-curlvϕg*1e10 ,colormap= colormap_vorticity, colorrange=clim_vorticity, colorscale=identity)
        set_annotation!(Fax,(4,1),latexstring(anatonate_label," (Gas)"),fontsize=14)
        heatmap!(Fax.axes[4,2],s ,z ,-curlvϕd*1e10 ,colormap= colormap_vorticity, colorrange=clim_vorticity, colorscale=identity)
        set_annotation!(Fax,(4,2),latexstring(anatonate_label," (Dust)"),fontsize=14)
        set_colorbar!(Fax,(4,1),clabel=Vorticity_label,link_row_colormap=true)
        Fax.axes[4,1].xreversed = sreverse
        Fax.axes[4,2].xreversed = sreverse


        set_xlabel!(Fax,slabel)
        set_ylabel!(Fax,zlabel)

        if isnothing(Slice_ϕ)
            output_filename = "$(File_prefix)_$(number_data)_aziave.$(Figure_format)"
        else
            output_filename = "$(File_prefix)_$(number_data)_$(Slice_ϕ)deg.$(Figure_format)"
        end
        save_Fig!(Fax, output_filename, dpi)
        close_Fig!(Fax)
    end
    
    @info "-------------------------------------------------------"
end

function main()
    # Commendline variable setting
    if length(ARGS) < 1
        println("Usage: julia Slicing_disk.jl <filename>")
        exit(1)
    end

    files = ARGS             

    First_logging()

    for file in files
        @info "File: $file"
        @time_and_print begin
            Slicing_disk(file)
        end 
    end

    @info "\nEnd analysis!"
end

main()