"""
The grid SPH interpolation
    by Wei-Shan Su,
    June 27, 2024

Generate a `gridbackend`, calculate the value for each column.

# Progression
Step 1. Generate a 3d KDTree to speed up the searching of particles

Step 2. Generate a grid for analysis. The grid is base on a struct gridbackend which is defined in the "grid.jl" file

Step 3. Calculate the result for each point by using the SPH intepolation. The calculation for each point is defined in "physical_quantity.jl"
"""

"""
    Disk_3D_Grid_analysis(
        data::PhantomRevealerDataFrame,
        s_params::Tuple{Float64,Float64,Int},
        ϕ_params::Tuple{Float64,Float64,Int},
        z_params::Tuple{Float64,Float64,Int};
        column_names::Union{Nothing,Vector{String}}=nothing,
        gradient_column_names::Union{Nothing,Vector{String}}=nothing,
        divergence_column_names::Union{Nothing,Vector{String}}=nothing,
        curl_column_names::Union{Nothing,Vector{String}}=nothing,
        smoothed_kernel::Function = M5_spline,
        h_mode::String = "closest",
        Identical_particles::Bool = true
    )
Calculate the SPH interpolation on a grid that is described as a cylindrical coordinate (s,ϕ,z) for a disk.

The density `rho` and its gradient vector `∇rho` will be calculated automatically. The `∇rho` values are returned in cylindrical coordinates (∇rho_s, ∇rho_ϕ, ∇rho_z).

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data stored in `PhantomRevealerDataFrame`. 
- `s_params :: Tuple{Float64,Float64,Int}`: The radial parameters [smin, smax, sn].
- `ϕ_params :: Tuple{Float64,Float64,Int}`: The azimuthal parameters [ϕmin, ϕmax, ϕn].
- `z_params :: Tuple{Float64,Float64,Int}`: The vertical parameters [zmin, zmax, zn].

# Keyword Arguments
- `column_names :: Union{Nothing, Vector{String}}=nothing`: The quantities to interpolate. If `nothing`, no additional quantities will be interpolated.
- `gradient_column_names :: Union{Nothing, Vector{String}}=nothing`: The gradient value of quantities to interpolate. If `nothing`, no additional quantities will be interpolated.
- `divergence_column_names :: Union{Nothing, Vector{String}}=nothing`: The divergence value of quantities to interpolate. For vector quantities (e.g., data columns named "vx", "vy", "vz"), you only need to provide the common prefix of the vector name, such as "v". If `nothing`, no additional quantities will be interpolated.
- `curl_column_names :: Union{Nothing, Vector{String}}=nothing`: The curl value of quantities to interpolate. For vector quantities (e.g., data columns named "vx", "vy", "vz"), you only need to provide the common prefix of the vector name, such as "v". If `nothing`, no additional quantities will be interpolated.
- `smoothed_kernel :: Function = M5_spline`: The kernel function for SPH interpolation.
- `h_mode :: String = "closest"`: The mode for determining the smoothing radius. Allowed values are `"closest"` and `"mean"`.
- `Identical_particles :: Bool = true`: Whether the particles are identical (default: `true`).

# Returns
- `Dict{String, gridbackend}`: A dictionary containing the interpolated results in the form of `gridbackend`.

# Examples
```julia
data :: PhantomRevealerDataFrame = read_phantom("dumpfile_00000", "all")[1]
add_cylindrical!(data)
s_params :: Tuple{Float64,Float64,Int} = (10.0,100.0,91)
ϕ_params :: Tuple{Float64,Float64,Int} = (0.0,2π,12)
z_params :: Tuple{Float64,Float64,Int} = (0.0,30.0,151)
smoothed_kernel :: Function = M6_spline
column_names :: Vector = ["vs", "vϕ", "vz"]

result :: Dict{String, gridbackend} = Disk_3D_Grid_analysis(
    data, s_params, ϕ_params, z_params;
    column_names=column_names,
    smoothed_kernel=smoothed_kernel
)
println(keys(result))  # Output: ["rho", "∇rhos", "∇rhoϕ", "∇rhoz", "vs", "vϕ", "vz"]
```
"""
function Disk_3D_Grid_analysis(
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int},
    ϕ_params::Tuple{Float64,Float64,Int},
    z_params::Tuple{Float64,Float64,Int};
    column_names::Union{Nothing,Vector{String}}=nothing,
    gradient_column_names::Union{Nothing,Vector{String}}=nothing,
    divergence_column_names::Union{Nothing,Vector{String}}=nothing,
    curl_column_names::Union{Nothing,Vector{String}}=nothing,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    Identical_particles::Bool=true
)
    function wrap_dens(data::PhantomRevealerDataFrame, point::Array)::Float64
        return density(data, point, smoothed_kernel, h_mode, "polar",Identical_particles=Identical_particles)
    end
    function wrap_graddens(data::PhantomRevealerDataFrame, point::Array)::Vector
        return gradient_density(data, point, smoothed_kernel, h_mode, "polar",Identical_particles=Identical_particles)
    end
    function wrap_quant(data::PhantomRevealerDataFrame, point::Array)::Dict{String,Float64}
        return quantity_intepolate(
            data,
            point,
            column_names,
            smoothed_kernel,
            h_mode,
            "polar",
            Identical_particles=Identical_particles
        )
    end
    function wrap_gradquant(data::PhantomRevealerDataFrame, point::Array, column_name::String,density_value::Union{Nothing,Float64}=nothing, quantity_value::Union{Nothing,Float64} = nothing)
        return gradient_quantity_intepolate(
            data,
            point,
            column_name,
            smoothed_kernel,
            h_mode,
            "polar",
            Identical_particles=Identical_particles,
            density_value=density_value,
            quantity_value=quantity_value
        )
    end
    function wrap_diverquant(data::PhantomRevealerDataFrame, point::Array, column_name::String,density_value::Union{Nothing,Float64}=nothing, quantity_value::Union{Nothing,Vector{Float64}} = nothing)
        return divergence_quantity_intepolate(
            data,
            point,
            column_name,
            smoothed_kernel,
            h_mode,
            "polar",
            Identical_particles=Identical_particles,
            density_value=density_value,
            quantity_value=quantity_value,
            quantity_coordinate_flag="polar"
        )
    end
    function wrap_curlquant(data::PhantomRevealerDataFrame, point::Array, column_name::String,density_value::Union{Nothing,Float64}=nothing, quantity_value::Union{Nothing,Vector{Float64}} = nothing)
        return curl_quantity_intepolate(
            data,
            point,
            column_name,
            smoothed_kernel,
            h_mode,
            "polar",
            Identical_particles=Identical_particles,
            density_value=density_value,
            quantity_value=quantity_value,
            quantity_coordinate_flag="polar"
        )
    end
    @info "Start 3D disk grid analysis."
    # Add necessary quantities
    add_necessary_quantity!(data)
    
    # The column subfix in cylindrical coordinate system
    column_suffixes = ["s", "ϕ", "z"]

    # Checking data before interpolation
    ###############################
    # Checking the necessity of intepolation
    # Regular intepolation
    columnNotEmpty = true
    if isnothing(column_names) || isempty(column_names)
        columnNotEmpty = false
    end

    # Gradient intepolation
    gradcolumnNotEmpty = true
    if isnothing(gradient_column_names) || isempty(gradient_column_names)
        gradcolumnNotEmpty = false
    end

    # Divergence intepolation
    divercolumnNotEmpty = true
    if isnothing(divergence_column_names) || isempty(divergence_column_names)
        divercolumnNotEmpty = false
    end

    # Curl intepolation
    curlcolumnNotEmpty = true
    if isnothing(curl_column_names) || isempty(curl_column_names)
        curlcolumnNotEmpty = false
    end

    ###############################
    if (data.params["Origin_sink_id"] == -1)
        error("IntepolateError: Wrong origin located!")
    end

    # Check missing columns. Also checking if the regular intepolation also intepolate the same column in the first deriviative intepolation to reduce the error of estimation.
    if columnNotEmpty
        for column_name in column_names
            if !(hasproperty(data.dfdata, column_name))
                error("IntepolateError: Missing column name $column_name !")
            end
        end
    end

    if gradcolumnNotEmpty
        grad_value_exist :: Vector{Bool} = Vector{Bool}(undef,length(gradient_column_names))
        for column_name in gradient_column_names
            if !(hasproperty(data.dfdata, column_name))
                error("IntepolateError: Missing column name $column_name !")
            end
        end
        for (i,gradcolumn) in enumerate(gradient_column_names)
            if gradcolumn in column_names
                grad_value_exist[i] = true
            else
                grad_value_exist[i] = false
            end
        end
    end

    if divercolumnNotEmpty
        diver_value_exist :: Vector{Bool} = Vector{Vector{Bool}}(undef,length(divergence_column_names))
        for (i,rawdivercolumn) in enumerate(divergence_column_names)
            for suffix in column_suffixes
                divercolumn = rawdivercolumn * suffix
                if !(hasproperty(data.dfdata, divercolumn))
                    error("IntepolateError: Missing column name $divercolumn !")
                end
                if divercolumn in column_names
                    diver_value_exist[i] = true
                else
                    diver_value_exist[i] = false
                    break
                end
            end
        end
    end

    if curlcolumnNotEmpty
        curl_value_exist :: Vector{Bool} = Vector{Bool}(undef,length(curl_column_names))
        for (i,rawcurlcolumn) in enumerate(curl_column_names)
            for suffix in column_suffixes
                curlcolumn = rawcurlcolumn * suffix
                if !(hasproperty(data.dfdata, curlcolumn))
                    error("IntepolateError: Missing column name $curlcolumn !")
                end
                if curlcolumn in column_names
                    curl_value_exist[i] = true
                else
                    curl_value_exist[i] = false
                    break
                end
            end
        end
    end

    # Generate kd tree in 3D space
    kdtree3d = Generate_KDtree(data, 3)

    # Generate Edge-on grid 
    imin::Vector = [s_params[1], ϕ_params[1], z_params[1]]
    imax::Vector = [s_params[2], ϕ_params[2], z_params[2]]
    iaxen::Vector = [s_params[3], ϕ_params[3], z_params[3]]
    empty_gridbackend::gridbackend = disk_3d_grid_generator(imin, imax, iaxen)

    # Generate the coordinate array for the grid interpolation
    gridv::Array{Vector{Float64}} = generate_coordinate_grid(empty_gridbackend)

    # Preparation of result dictionary
    Result_dict = Dict{String,gridbackend}()
    Result_dict["rho"] = deepcopy(empty_gridbackend)
    Result_dict["∇rhos"] = deepcopy(empty_gridbackend)
    Result_dict["∇rhoϕ"] = deepcopy(empty_gridbackend)
    Result_dict["∇rhoz"] = deepcopy(empty_gridbackend)
    if columnNotEmpty
        for column_name in column_names
            (column_name == "rho") && continue
            Result_dict[column_name] = deepcopy(empty_gridbackend)
        end
    end
    if gradcolumnNotEmpty
        for column_name in gradient_column_names
            (column_name == "rho") && continue
            Result_dict["∇$(column_name)s"] = deepcopy(empty_gridbackend)
            Result_dict["∇$(column_name)ϕ"] = deepcopy(empty_gridbackend)
            Result_dict["∇$(column_name)z"] = deepcopy(empty_gridbackend)
        end
    end
    if divercolumnNotEmpty
        for column_name in divergence_column_names
            (column_name == "rho") && continue
            Result_dict["∇⋅$(column_name)"] = deepcopy(empty_gridbackend)
        end
    end
    if curlcolumnNotEmpty
        for column_name in curl_column_names
            (column_name == "rho") && continue
            Result_dict["∇×$(column_name)s"] = deepcopy(empty_gridbackend)
            Result_dict["∇×$(column_name)ϕ"] = deepcopy(empty_gridbackend)
            Result_dict["∇×$(column_name)z"] = deepcopy(empty_gridbackend)
        end
    end

    # Prepare a roughly truncate radius for KD-tree filtering.
    roughly_truncated_radius::Float64 =
        get_truncated_radius(data, -1.0f0, 0.5, smoothed_kernel)
    # Iteration
    @threads for i in eachindex(gridv)
        target = gridv[i]
        kdtf_data = KDtree_filter(data, kdtree3d, target, roughly_truncated_radius, "polar") # New data that has been filtered.
        Result_dict["rho"].grid[i] = wrap_dens(kdtf_data, target)
        ∇dens = wrap_graddens(kdtf_data, target)
        Result_dict["∇rhos"].grid[i] = ∇dens[1]
        Result_dict["∇rhoϕ"].grid[i] = ∇dens[2]
        Result_dict["∇rhoz"].grid[i] = ∇dens[3]
        if columnNotEmpty
            quantity_interpolation_dict::Dict{String,Float64} = wrap_quant(kdtf_data, target)
            if all(key -> haskey(Result_dict, key), keys(quantity_interpolation_dict))
                for key in keys(quantity_interpolation_dict)
                    Result_dict[key].grid[i] = quantity_interpolation_dict[key]
                end
            else
                error("IntepolateError: Missing column name!")
            end
        end 
        if gradcolumnNotEmpty
            input_density = Result_dict["rho"].grid[i]
            grad_quantity_interpolation_dict::Dict{String,Float64} = Dict{String,Float64}()
            for n in eachindex(gradient_column_names)
                column_name = gradient_column_names[n]
                input_value = nothing
                if grad_value_exist[n]
                    input_value = Result_dict[column_name].grid[i]
                end
                buffer_array = wrap_gradquant(kdtf_data, target,column_name,input_density,input_value)
                grad_quantity_interpolation_dict["∇$(column_name)s"],grad_quantity_interpolation_dict["∇$(column_name)ϕ"],grad_quantity_interpolation_dict["∇$(column_name)z"] = buffer_array
            end
            if all(key -> haskey(Result_dict, key), keys(grad_quantity_interpolation_dict))
                for key in keys(grad_quantity_interpolation_dict)
                    Result_dict[key].grid[i] = grad_quantity_interpolation_dict[key]
                end
            else
                error("IntepolateError: Missing column name!")
            end
        end
        if divercolumnNotEmpty
            input_density = Result_dict["rho"].grid[i]
            diver_quantity_interpolation_dict::Dict{String,Float64} = Dict{String,Float64}()
            for n in eachindex(divergence_column_names)
                column_name = divergence_column_names[n]
                input_value = nothing
                if diver_value_exist[n]
                    input_value = zeros(Float64,3)
                    input_value[1] = Result_dict["$(column_name)s"].grid[i]
                    input_value[2] = Result_dict["$(column_name)ϕ"].grid[i]
                    input_value[3] = Result_dict["$(column_name)z"].grid[i]
                end
                diver_quantity_interpolation_dict["∇⋅$(column_name)"] = wrap_diverquant(kdtf_data, target,column_name,input_density,input_value)
            end
            if all(key -> haskey(Result_dict, key), keys(diver_quantity_interpolation_dict))
                for key in keys(diver_quantity_interpolation_dict)
                    Result_dict[key].grid[i] = diver_quantity_interpolation_dict[key]
                end
            else
                error("IntepolateError: Missing column name!")
            end
        end  
        if curlcolumnNotEmpty
            input_density = Result_dict["rho"].grid[i]
            curl_quantity_interpolation_dict::Dict{String,Float64} = Dict{String,Float64}()
            for n in eachindex(curl_column_names)
                column_name = curl_column_names[n]
                input_value = nothing
                if curl_value_exist[n]
                    input_value = zeros(Float64,3)
                    input_value[1] = Result_dict["$(column_name)s"].grid[i]
                    input_value[2] = Result_dict["$(column_name)ϕ"].grid[i]
                    input_value[3] = Result_dict["$(column_name)z"].grid[i]
                end
                buffer_array = wrap_curlquant(kdtf_data, target,column_name,input_density,input_value)
                curl_quantity_interpolation_dict["∇×$(column_name)s"],curl_quantity_interpolation_dict["∇×$(column_name)ϕ"],curl_quantity_interpolation_dict["∇×$(column_name)z"] = buffer_array
            end
            if all(key -> haskey(Result_dict, key), keys(curl_quantity_interpolation_dict))
                for key in keys(curl_quantity_interpolation_dict)
                    Result_dict[key].grid[i] = curl_quantity_interpolation_dict[key]
                end
            else
                error("IntepolateError: Missing column name!")
            end
        end  
    end

    @info "End 3D disk grid analysis."
    return Result_dict
end

"""
    Disk_scale_height_analysis(edgeon_data_3D :: Dict{String, gridbackend})
Calculate the scale height of disk from existing edgeon data.

# Parameters
- `edgeon_data_3D :: Dict{String, gridbackend}`: The result from a edge-on analysis.

# Returns
- `Interpolations.Extrapolation`: The interpolation object of scale height as the function of `s`
"""
function Disk_scale_height_analysis(edgeon_data_3D :: Dict{String, gridbackend})
    rho_grid = grid_reduction(edgeon_data_3D["rho"], 2)
    s_separate = rho_grid.dimension[1]
    z_separate = rho_grid.dimension[2]
    s_array = rho_grid.axes[1]
    z_array = collect(rho_grid.axes[2])
    H_array = zeros(Float64, s_separate)
    @threads for i = 1:s_separate
        rho_array = rho_grid.grid[i, :]
        rho_sum = cumsum(rho_array) ./ maximum(rho_array)
        rho_cdf = rho_sum ./ maximum(rho_sum)
        for j = 1:z_separate
            cdfrho = rho_cdf[j]
            if cdfrho > 0.99999999
                H_array[i] = z_array[j]
                break
            end
        end
    end
    result = CubicSplineInterpolation(s_array, H_array, extrapolation_bc = Line())
    return result
end

"""
    Disk_scale_height_analysis(data::PhantomRevealerDataFrame, s_params::Tuple{Float64,Float64,Int}, ϕ_params :: Tuple{Float64,Float64,Int} = (0.0,2π,8) ,z_params::Tuple{Float64,Float64,Int} = (0.0, 28.0, 70),smoothed_kernel:: Function = M5_spline,h_mode::String="closest")
Calculate the scale height of disk.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `s_params :: Tuple{Float64,Float64,Int}`: The radial parameters with [smin, smax, sn]
- `ϕ_params :: Tuple{Float64,Float64,Int}`: The azimuthal parameters with [ϕmin, ϕmax, ϕn]
- `z_params :: Tuple{Float64,Float64,Int}`: The height parameters with [zmin, zmax, zn]
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")

# Returns
- `Interpolations.Extrapolation`: The interpolation object of scale height as the function of `s`
"""
function Disk_scale_height_analysis(
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int},
    ϕ_params::Tuple{Float64,Float64,Int} = (0.0, 2π, 8),
    z_params::Tuple{Float64,Float64,Int} = (0.0, 28.0, 70),
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest";
    Identical_particles::Bool=true
)
    edgeon_data_3D = Disk_3D_Grid_analysis(
        data,
        s_params,
        ϕ_params,
        z_params,
        smoothed_kernel=smoothed_kernel,
        h_mode=h_mode,
        Identical_particles=Identical_particles
    )
    result = Disk_scale_height_analysis(edgeon_data_3D)
    return result
end

"""
    Disk_2D_midplane_function_generator(edgeon_data_3D::Dict{String, gridbackend}, circular_axis::Int64=[2])
Generate an function f(s,ϕ) which represent the midplane of the disk.

# Parameters
- `edgeon_data_3D :: Dict{String, gridbackend}`: The result from a edge-on analysis.
- `azimuthal_circular :: Bool = true`: Boolean of a circular azimuthal grid.

# Returns
- `Interpolations.Extrapolation`: The interpolation object of midplane as the function of `s` and `ϕ`.
"""
function Disk_2D_midplane_function_generator(edgeon_data_3D::Dict{String, gridbackend}, azimuthal_circular::Bool = true)
    rho_gbe :: gridbackend = deepcopy(edgeon_data_3D["rho"])
    z_grid :: Array = zeros(Float64,rho_gbe.dimension[1],rho_gbe.dimension[2])
    z_array = rho_gbe.axes[3]
    grid3d = rho_gbe.grid
    @threads for idx in CartesianIndices(z_grid)
        rhos ::Vector{Float64} = grid3d[idx[1],idx[2],:]
        if sum(rhos) == 0.0
            z_grid[idx] = NaN64
        else
            z_grid[idx] = z_array[findmax(rhos)[2]]
        end
    end
    if azimuthal_circular
        z_grid = hcat(z_grid,z_grid[:,1])
        rho_gbe.axes[2] = LinRange(0.0,2π,length(rho_gbe.axes[2])+1)
    end
    zfunc :: Interpolations.Extrapolation = LinearInterpolation((rho_gbe.axes[1],rho_gbe.axes[2]),z_grid)
    return zfunc
end

"""
    Disk_2D_midplane_function_generator(   
        data::PhantomRevealerDataFrame,
        s_params::Tuple{Float64,Float64,Int} = (10.0, 120.0, 111),
        ϕ_params::Tuple{Float64,Float64,Int} = (0.0, 2π, 24),
        z_params::Tuple{Float64,Float64,Int} = (-28.0, 28.0, 100),
        smoothed_kernel::Function = M5_spline,
        h_mode::String = "closest"
    )
Generate an function f(s,ϕ) which represent the midplane of the disk.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `s_params :: Tuple{Float64,Float64,Int}`: The radial parameters with [smin, smax, sn]
- `ϕ_params :: Tuple{Float64,Float64,Int}`: The azimuthal parameters with [ϕmin, ϕmax, ϕn]
- `z_params :: Tuple{Float64,Float64,Int}`: The height parameters with [zmin, zmax, zn]

# Keyword Arguments
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `Identical_particles :: Bool = true`: Whether the particles are identical (default: `true`).

# Returns
- `Interpolations.Extrapolation`: The interpolation object of midplane as the function of `s` and `ϕ`.
"""
function Disk_2D_midplane_function_generator(   
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int} = (10.0, 120.0, 111),
    ϕ_params::Tuple{Float64,Float64,Int} = (0.0, 2π, 24),
    z_params::Tuple{Float64,Float64,Int} = (-28.0, 28.0, 100);
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    Identical_particles::Bool=true
)
    edgeon_data_3D = Disk_3D_Grid_analysis(
        data,
        s_params,
        ϕ_params,
        z_params,
        smoothed_kernel=smoothed_kernel,
        h_mode=h_mode,
        Identical_particles=Identical_particles
    )
    if ϕ_params[2] == 2π
        circ_flag = true
    else
        circ_flag = false
    end
    zfunc :: Interpolations.Extrapolation = Disk_2D_midplane_function_generator(edgeon_data_3D,circ_flag)
    return zfunc
end

"""
    Disk_2D_FaceOn_Grid_analysis(
        data::PhantomRevealerDataFrame,
        s_params::Tuple{Float64,Float64,Int},
        ϕ_params::Tuple{Float64,Float64,Int};
        column_names::Union{Nothing,Vector{String}}=nothing,
        mid_column_names::Union{Nothing,Vector{String}}=nothing,
        midz_func::Union{Nothing,Interpolations.Extrapolation}=nothing,
        smoothed_kernel::Function = M5_spline,
        h_mode::String = "closest",
        Identical_particles::Bool = true
    )
Calculate the SPH interpolation on a face-on grid described in polar coordinates (s,ϕ) for a disk.

Note: The gradient values of arbitrary quantities are not yet supported.

The surface density `Sigma` and its gradient vector `∇Sigma` will be calculated automatically. The `∇Sigma` values are returned in polar/cylindrical coordinates (∇Sigma_s, ∇Sigma_ϕ).

For quantities calculated using mid-plane averages, the results will have a suffix `m`. For example, `rho` becomes `rhom`, `vs` becomes `vsm`, etc.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data stored in `PhantomRevealerDataFrame`. 
- `s_params :: Tuple{Float64,Float64,Int}`: The radial parameters [smin, smax, sn].
- `ϕ_params :: Tuple{Float64,Float64,Int}`: The azimuthal parameters [ϕmin, ϕmax, ϕn].

# Keyword Arguments
- `column_names :: Union{Nothing,Vector{String}}=nothing`: The quantities to interpolate. If `nothing`, only density-related values will be interpolated.
- `mid_column_names :: Union{Nothing,Vector{String}}=nothing`: Quantities to evaluate by taking mid-plane averages.
- `midz_func :: Union{Nothing,Interpolations.Extrapolation}=nothing`: The interpolation function for the z-component of midplane `H_mid(s)`.
- `smoothed_kernel :: Function = M5_spline`: The kernel function for SPH interpolation.
- `h_mode :: String="closest"`: The mode for determining the smoothing radius. Allowed values are `"closest"` and `"mean"`.
- `Identical_particles :: Bool = true`: Whether the particles are identical (default: `true`).

# Returns
- `Dict{String, gridbackend}`: A dictionary containing the interpolated results in the form of `gridbackend`.

# Examples
```julia
data :: PhantomRevealerDataFrame = read_phantom("./temp/st15co_00107", "all")[1]
add_cylindrical!(data)
s_params :: Tuple{Float64,Float64,Int} = (10.0,100.0,91)
ϕ_params :: Tuple{Float64,Float64,Int} = (0.0,2π,12)
smoothed_kernel :: Function = M6_spline
midz_func = Disk_2D_midplane_function_generator(data, s_params)

column_names :: Vector = ["e"]
mid_column_names :: Vector = ["rho", "vs", "vϕ"]

result :: Dict{String, gridbackend} = Disk_2D_FaceOn_Grid_analysis(
    data, s_params, ϕ_params;
    column_names=column_names,
    mid_column_names=mid_column_names,
    midz_func=midz_func,
    smoothed_kernel=smoothed_kernel
)
println(keys(result))  # Output: ["Sigma", "∇Sigmas", "∇Sigmaϕ", "e", "rhom", "vsm", "vϕm"]
```
"""
function Disk_2D_FaceOn_Grid_analysis(
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int},
    ϕ_params::Tuple{Float64,Float64,Int};
    column_names::Union{Nothing,Vector{String}}=nothing,
    mid_column_names::Union{Nothing,Vector{String}}=nothing,
    midz_func::Union{Nothing,Interpolations.Extrapolation}=nothing,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    Identical_particles::Bool=true
)
    function wrap_surf_dens(data::PhantomRevealerDataFrame, point::Array)::Float64
        return surface_density(data, point, smoothed_kernel, h_mode, "polar",Identical_particles=Identical_particles)
    end
    function wrap_grad_surf_dens(
        data::PhantomRevealerDataFrame,
        point::Array,
    )::Vector{Float64}
        return gradient_surface_density(data, point, smoothed_kernel, h_mode, "polar",Identical_particles=Identical_particles)
    end
    function wrap_quant(data::PhantomRevealerDataFrame, point::Array)::Dict{String,Float64}
        return quantity_intepolate(
            data,
            point,
            mid_column_names,
            smoothed_kernel,
            h_mode,
            "polar",
            Identical_particles=Identical_particles
        )
    end
    function wrap_quant2D(
        data::PhantomRevealerDataFrame,
        point::Array,
        Sigmai::Float64,
    )::Dict{String,Float64}
        return quantity_intepolate_2D(
            data,
            point,
            Sigmai,
            column_names,
            smoothed_kernel,
            h_mode,
            "polar",
            Identical_particles=Identical_particles
        )
    end
    @info "Start 2D disk grid analysis."
    # Add necessary quantities
    add_necessary_quantity!(data)

    # Checking data before interpolation
    ###############################
    # Checking the necessity of intepolation
    # Regular intepolation
    columnNotEmpty = true
    if isnothing(column_names)
        columnNotEmpty = false
    elseif isempty(column_names)
        columnNotEmpty = false
    end
    # Midplane intepolation
    midcolumnNotEmpty = true
    if isnothing(mid_column_names)
        midcolumnNotEmpty = false
    elseif isempty(mid_column_names)
        columnNotEmpty = false
    end
    if (data.params["Origin_sink_id"] == -1)
        error("IntepolateError: Wrong origin located!")
    end
    ###############################

    if columnNotEmpty
        for column_name in column_names
            if !(hasproperty(data.dfdata, column_name))
                error("IntepolateError: Missing column name $column_name !")
            end
        end
    end
    if midcolumnNotEmpty
        for column_name in mid_column_names
            if !(hasproperty(data.dfdata, column_name))
                error("IntepolateError: Missing column name $column_name !")
            end
        end
    end
    # Generate kd tree
    kdtree3d = Generate_KDtree(data, 3)
    kdtree2d = Generate_KDtree(data, 2)

    # Generate Face-on grid 
    imin::Vector = [s_params[1], ϕ_params[1]]
    imax::Vector = [s_params[2], ϕ_params[2]]
    iaxen::Vector = [s_params[3], ϕ_params[3]]
    empty_gridbackend::gridbackend = disk_2d_grid_generator(imin, imax, iaxen)

    # Generate the coordinate array for the grid interpolation
    gridv::Array{Vector{Float64}} = generate_coordinate_grid(empty_gridbackend)

    # Preparation of result dictionary
    Result_dict = Dict{String,gridbackend}()
    Result_dict["Sigma"] = deepcopy(empty_gridbackend)
    Result_dict["∇Sigmas"] = deepcopy(empty_gridbackend)
    Result_dict["∇Sigmaϕ"] = deepcopy(empty_gridbackend)
    if columnNotEmpty
        for column_name in column_names
            (column_name == "Sigma") && continue
            Result_dict[column_name] = deepcopy(empty_gridbackend)
        end
    end
    if midcolumnNotEmpty
        imid_column_names = mid_column_names .* "m"
        for column_name in imid_column_names
            Result_dict[column_name] = deepcopy(empty_gridbackend)
        end
    end

    # Prepare a roughly truncate radius for KD-tree filtering.
    roughly_truncated_radius::Float64 =
        get_truncated_radius(data, -1.0f0, 0.5, smoothed_kernel)

    # Iteration
    @threads for i in eachindex(gridv)
        target = gridv[i]
        # 2D intepolation
        kdtf_data2d =
            KDtree_filter(data, kdtree2d, target, roughly_truncated_radius, "polar") # New data that has been filtered.
        Sigmai = wrap_surf_dens(kdtf_data2d, target)
        Result_dict["Sigma"].grid[i] = Sigmai
        ∇dens = wrap_grad_surf_dens(kdtf_data2d, target)
        Result_dict["∇Sigmas"].grid[i] = ∇dens[1]
        Result_dict["∇Sigmaϕ"].grid[i] = ∇dens[2]
        if columnNotEmpty
            quantity_interpolation_dict::Dict{String,Float64} = wrap_quant2D(kdtf_data2d, target, Sigmai)
            if all(key -> haskey(Result_dict, key), keys(quantity_interpolation_dict))
                for key in keys(quantity_interpolation_dict)
                    Result_dict[key].grid[i] = quantity_interpolation_dict[key]
                end
            else
                error("IntepolateError: Missing column name!")
            end
        end
        # 3D midplane intepolation
        if midcolumnNotEmpty
            midz = midz_func(target...)
            if midz == NaN64
                for j in eachindex(mid_column_names)
                    Result_dict[imid_column_names[j]].grid[i] = NaN64
                end
            else
                target3D = [target..., midz]
                kdtf_data3d = KDtree_filter(data, kdtree3d, target3D, roughly_truncated_radius, "polar") # New data that has been filtered.
                mid_interpolation_dict::Dict{String,Float64} = wrap_quant(kdtf_data3d, target3D)
                for j in eachindex(mid_column_names)
                    Result_dict[imid_column_names[j]].grid[i] = mid_interpolation_dict[mid_column_names[j]]
                end
            end
        end
        
    end
    @info "End 2D disk grid analysis."
    return Result_dict
end

"""
    gridbackend_Grid_analysis(
        data::PhantomRevealerDataFrame,
        grid::gridbackend;
        column_names::Union{Nothing,Vector{String}}=nothing,
        smoothed_kernel::Function = M5_spline,
        h_mode::String = "closest",
        coordinate_flag::String = "cart",
        Identical_particles::Bool = true
    )
Calculate SPH interpolation on a given grid described as `gridbackend`.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data stored in `PhantomRevealerDataFrame`. 
- `grid :: gridbackend`: The grid for interpolation.

# Keyword Arguments
- `column_names :: Union{Nothing,Vector{String}}=nothing`: The quantities to interpolate. If `nothing`, only density is calculated.
- `smoothed_kernel :: Function = M5_spline`: The kernel function for SPH interpolation.
- `h_mode :: String = "closest"`: The mode for determining the smoothing radius. Allowed values are `"closest"` and `"mean"`.
- `coordinate_flag :: String = "cart"`: The coordinate system used for target points. Allowed values are `"cart"` and `"polar"`.
- `Identical_particles :: Bool = true`: Whether the particles are identical (default: `true`).

# Returns
- `Dict{String, gridbackend}`: A dictionary containing the interpolated results in the form of `gridbackend`.
"""
function gridbackend_Grid_analysis(
    data::PhantomRevealerDataFrame,
    grid::gridbackend;
    column_names::Union{Nothing,Vector{String}}=nothing,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag:: String = "cart",
    Identical_particles::Bool=true
)   
    function generate_functions(grid_dimension::Int)
        if grid_dimension == 2
            dens_name = "Sigma"
            kdtree = Generate_KDtree(data, 2)
            
            wrap_dens = (data, point) -> begin
                surface_density(data, point, smoothed_kernel, h_mode, coordinate_flag,Identical_particles=Identical_particles)
            end
            
            wrap_quant = (data, point, deni) -> begin
                quantity_intepolate_2D(
                    data,
                    point,
                    deni,
                    column_names,
                    smoothed_kernel,
                    h_mode,
                    coordinate_flag,
                    Identical_particles=Identical_particles
                )
            end

            return wrap_dens, wrap_quant, dens_name, kdtree

        elseif grid_dimension == 3
            dens_name = "rho"
            kdtree = Generate_KDtree(data, 3)
            
            wrap_dens = (data, point) -> begin
                density(data, point, smoothed_kernel, h_mode, coordinate_flag,Identical_particles=Identical_particles)
            end

            wrap_quant = (data, point, deni) -> begin
                quantity_intepolate(
                    data,
                    point,
                    column_names,
                    smoothed_kernel,
                    h_mode,
                    coordinate_flag,
                    Identical_particles=Identical_particles
                )
            end

            return wrap_dens, wrap_quant, dens_name, kdtree

        else
            error("DimensionMismatch: Unable to generate the KDTree.")
        end
    end

    wrap_dens, wrap_quant, dens_name, kdtree = generate_functions(length(grid.dimension))
    @info "Start grid analysis. "
    # Generate the coordinate array for the grid interpolation
    gridv::Array{Vector{Float64}} = generate_coordinate_grid(grid)

    # Preparation of result dictionary
    Result_dict = Dict{String,gridbackend}()
    Result_dict[dens_name] = deepcopy(grid)
    for column_name in column_names
        (column_name == dens_name) && continue
        Result_dict[column_name] = deepcopy(grid)
    end

    # Prepare a roughly truncate radius for KD-tree filtering.
    roughly_truncated_radius::Float64 =
        get_truncated_radius(data, -1.0f0, 0.5, smoothed_kernel)

    # Iteration
    @threads for i in eachindex(gridv)
        target = gridv[i]
        kdtf_data = KDtree_filter(data, kdtree, target, roughly_truncated_radius, coordinate_flag) # New data that has been filtered.
        deni = wrap_dens(kdtf_data,target)
        Result_dict[dens_name].grid[i] = deni
        quantity_interpolation_dict::Dict{String,Float64} = wrap_quant(kdtf_data, target, deni)
        if all(key -> haskey(Result_dict, key), keys(quantity_interpolation_dict))
            for key in keys(quantity_interpolation_dict)
                Result_dict[key].grid[i] = quantity_interpolation_dict[key]
            end
        else
            error("IntepolateError: Missing column name!")
        end
    end
    @info "End grid analysis."
    return Result_dict
end
