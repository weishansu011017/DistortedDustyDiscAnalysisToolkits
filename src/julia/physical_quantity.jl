"""
The single point SPH interpolation
    by Wei-Shan Su,
    June 27, 2024


# Structure:
    ## Preparation
        add_necessary_quantity()
    ## 3D interpolation
        ### density
        ### gradient density
        ### 3D quantity interpolation
    ## 2D interpolation
        ### surface density
        ### gradient surface density
        ### 2D quantity interpolation
"""
function add_necessary_quantity!(data::PhantomRevealerDataFrame)
    if !(hasproperty(data.dfdata, "rho"))
        add_rho!(data)
    end
end

function _easy_estimate_h_intepolate(dfdata::DataFrame, rnorm::Vector)
    """
    Give a specific smoothed radius to calculate
    Assume data has been transfered to the reference_point-based coordinate
    """
    if (isempty(rnorm))
        return NaN32
    end
    min_index = argmin(rnorm)
    return dfdata[min_index, "h"]
end

"""
    estimate_h_intepolate(data::PhantomRevealerDataFrame, rnorm::Vector,mode::String ="closest")
Provide a smoothed radius for interpolation.

mode: 
"closest": Choose h of the closest particles.
"mean": use the mean value of h

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `rnorm :: Vector`: The distance between reference point and every particles. 
- `mode :: String ="closest"`: The mode of interpolation. (Allowed value: "closest", "mean")

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].

# Returns
- `Float32`: The smoothed radius h.
"""
function estimate_h_intepolate(
    data::PhantomRevealerDataFrame,
    rnorm::Vector,
    mode::String = "closest",
)
    dfdata = data.dfdata
    if !(haskey(data.params, "h_mean"))
        add_mean_h!(data)
    end
    if (mode == "mean")
        return data.params["h_mean"]
    elseif (mode == "closest")
        return _easy_estimate_h_intepolate(dfdata, rnorm)
    else
        error("IntepolateError: Invaild mode of h calculattion.")
    end
end

"""
    density(data::PhantomRevealerDataFrame, reference_point::Vector, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the density ρ at the given reference point by SPH interpolation

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value:  "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].

# Returns
- `Float64`: The value of density at the reference point.

# Example
```julia
prdf_list = read_phantom(dumpfile_00000,"all")
COM2star!(prdf_list, prdf_list[end],1)
data :: PhantomRevealerDataFrame = prdf_list[1]
reference_point :: Vector = [25.0, π/2, 0.0]  # Choosing the cylindrical coordinate
smoothed_kernel :: Function = M6_spline
dens = density(data, reference_point, smoothed_kernel, "closest", "polar")
```
"""
function density(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    reference_point is in "3D"
    "cart" = cartitian
    "polar" = cylindrical
    """
    # Return 0.0 if no particle in the data
    if nrow(data.dfdata) == 0
        return 0.0
    end

    # Preparing and verifing the data
    if coordinate_flag == "polar"
        reference_point = _cylin2cart(reference_point)
    end

    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    rnorm = get_rnorm_ref(data, reference_point)
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(rnorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, rnorm, h_mode) # _easy_estimate_h_intepolate(dfdata, rnorm,)

    # Kernel interpolation
    if (h_intepolate == 0.0)
        density = 0.0
    else
        truncate_radius = truncate_multiplier * h_intepolate
        filtered_indices = findall(r -> r <= truncate_radius, rnorm)
        filtered_rnorm = rnorm[filtered_indices]
        filtered_m = mass_array[filtered_indices]
        density = sum(
            filtered_m .*
            Smoothed_kernel_function.(smoothed_kernel, h_intepolate, filtered_rnorm, 3),
        )
    end
    return density
end

"""
    number_density(data::PhantomRevealerDataFrame, reference_point::Vector, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart")
Calculate the number density n at the given reference point by SPH interpolation.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value:  "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Returns
- `Float64`: The value of number density at the reference point.

# Example
```julia
prdf_list = read_phantom(dumpfile_00000,"all")
COM2star!(prdf_list, prdf_list[end],1)
data :: PhantomRevealerDataFrame = prdf_list[1]
reference_point :: Vector = [25.0, π/2, 0.0]  # Choosing the cylindrical coordinate
smoothed_kernel :: Function = M6_spline
numdens = number_density(data, reference_point, smoothed_kernel, "closest", "polar")
```
"""
function number_density(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart"
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    reference_point is in "3D"
    "cart" = cartitian
    "polar" = cylindrical
    """
    # Return 0.0 if no particle in the data
    if nrow(data.dfdata) == 0
        return 0.0
    end

    # Preparing and verifing the data
    if coordinate_flag == "polar"
        reference_point = _cylin2cart(reference_point)
    end

    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    rnorm = get_rnorm_ref(data, reference_point)
    h_intepolate = estimate_h_intepolate(data, rnorm, h_mode) # _easy_estimate_h_intepolate(dfdata, rnorm,)

    # Kernel interpolation
    if (h_intepolate == 0.0)
        numdensity = 0.0
    else
        truncate_radius = truncate_multiplier * h_intepolate
        filtered_indices = findall(r -> r <= truncate_radius, rnorm)
        filtered_rnorm = rnorm[filtered_indices]
        numdensity = sum(
            Smoothed_kernel_function.(smoothed_kernel, h_intepolate, filtered_rnorm, 3),
        )
    end
    return numdensity
end

"""
    gradient_density(data::PhantomRevealerDataFrame, reference_point::Vector, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the grident of density ∇ρ at the given reference point by SPH interpolation

The gradient density would be estimated by

∇ρ(r) = (1/ρ(r))∑_b m_b*ρ_b*∇W(r-r_b) -∑_b m_b*∇W(r-r_b)

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].

# Returns
- `Vector`: The gradient vector of density at the reference point. The coordinate would be as same as the input(depends on "coordinate_flag").

# Example
```julia
prdf_list = read_phantom(dumpfile_00000,"all")
COM2star!(prdf_list, prdf_list[end],1)
data :: PhantomRevealerDataFrame = prdf_list[1]
reference_point :: Vector = [25.0, π/2, 0.0]  # Choosing the cylindrical coordinate
smoothed_kernel :: Function = M6_spline
∇dens = gradient_density(data, reference_point, smoothed_kernel, "closest", "polar")
```
"""
function gradient_density(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true,
    density_value::Union{Nothing,Float64}=nothing
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    reference_point is in "2D"
    "cart" = cartitian
    "polar" = polar
    """
    grad_density::Vector = Vector{Float64}(undef, 3)
    # Return NaN if no particle in the data
    if nrow(data.dfdata) == 0
        fill!(grad_density, NaN)
        return grad_density
    end

    # Preparing and verifing the data
    if coordinate_flag == "polar"
        ϕ =  reference_point[2]
        reference_point = _cylin2cart(reference_point)
    end
    if !(hasproperty(data.dfdata, "rho"))
        add_rho!(data)
    end
    if isnothing(density_value)
        density_value = density(data,reference_point,smoothed_kernel,h_mode,"cart",Identical_particles=Identical_particles)
    end
    
    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    rnorm, xyzref = get_r_ref(data, reference_point)
    density_array = data.dfdata[!,"rho"]
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(rnorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, rnorm, h_mode) #_easy_estimate_h_intepolate(dfdata, rnorm, 1.0)

    # Kernel interpolation
    if (h_intepolate == 0.0)
        fill!(grad_density, NaN)
        return grad_density
    else
        truncate_radius = truncate_multiplier * h_intepolate
        mask_rnorm = rnorm .< truncate_radius
        xyz_filtered = xyzref[mask_rnorm, :]
        filtered_m = mass_array[mask_rnorm]
        filtered_ρb = density_array[mask_rnorm]

        # bufferf represent the first term of formula, buffers represent the second term of formula
        bufferf_array = zeros(Float64, size(xyz_filtered))
        buffers_array = zeros(Float64, size(xyz_filtered))
        for i = 1:size(xyz_filtered)[1]
            bufferf_array[i, :] =
                (filtered_ρb[i] * filtered_m[i]) .* Smoothed_gradient_kernel_function(
                    smoothed_kernel,
                    h_intepolate,
                    xyz_filtered[i, :],
                ) ./density_value
            buffers_array[i, :] =
                filtered_m[i] .* Smoothed_gradient_kernel_function(
                    smoothed_kernel,
                    h_intepolate,
                    xyz_filtered[i, :],
                )
            
        end
        for j in eachindex(grad_density)
            grad_density[j] = sum(bufferf_array[:, j] .- buffers_array[:,j])
        end

        if coordinate_flag == "polar"
            cylindrical_grad_density = _vector_cart2cylin(ϕ,grad_density...)
            return cylindrical_grad_density
        else
            return grad_density
        end
    end
end

"""
    quantity_intepolate(data::PhantomRevealerDataFrame, reference_point::Vector, column_names::Vector{String}, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the value of every requested quantities at the given reference point by SPH interpolation.
Those points whose neighborhood has no particles around it would be label as `NaN`



# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `column_names :: Vector{String}`: The quantities that would be interpolated.
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].

# Returns
- `Dict{String, Float64}`: A dictionary that contains the value of quantities at the reference point.

# Example
```julia
prdf_list = read_phantom(dumpfile_00000,"all")
COM2star!(prdf_list, prdf_list[end],1)
data :: PhantomRevealerDataFrame = prdf_list[1]
column_names :: Vector{String} = ["vr", "vϕ", "vz", "e", "St"]
reference_point :: Vector = [25.0, π/2, 0.0]  # Choosing the cylindrical coordinate
smoothed_kernel :: Function = M6_spline
quantities_dict :: Dict{Float64} = quantity_intepolate(data, reference_point, column_names, smoothed_kernel, "closest", "polar")
```
"""
function quantity_intepolate(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    column_names::Vector{String},
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    "cart" = cartitian
    "polar" = cylindrical
    """
    # Constructing the column names for interpolation
    working_column_names = copy(column_names)
    divided_column = "rho"
    rho_flag = false
    if divided_column in working_column_names
        rho_flag = true
        deleteat!(working_column_names, findall(x->x==divided_column, working_column_names))
    else
        rho_flag = false
    end
    quantity_result = Dict{String,Float64}()

    # Return NaN if no particle in the data
    if nrow(data.dfdata) == 0
        for column_name in working_column_names
            quantity_result[column_name] = NaN
            if rho_flag
                quantity_result[divided_column] = 0.0
            end
        end
        return quantity_result
    end

    # Preparing and verifing the data
    if !(hasproperty(data.dfdata, "rho"))
        add_rho!(data)
    end
    for column_name in working_column_names
        if !(hasproperty(data.dfdata, column_name))
            error("IntepolateError: No matching column '$(column_name)'.")
        end
    end
    if coordinate_flag == "polar"
        reference_point = _cylin2cart(reference_point)
    end
    
    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    rnorm = get_rnorm_ref(data, reference_point)
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(rnorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, rnorm, h_mode)

    # Kernel interpolation
    dfdata = data.dfdata
    if (h_intepolate == 0.0)
        for column_name in working_column_names
            quantity_result[column_name] = NaN
            if rho_flag
                quantity_result[divided_column] = 0.0
            end
        end
    else
        truncate_radius = truncate_multiplier * h_intepolate
        indices = findall(x -> x <= truncate_radius, rnorm)
        filtered_dfdata = dfdata[indices, :]
        filtered_rnorm = rnorm[indices]
        filtered_m = mass_array[indices]
        if rho_flag
            quantity_result[divided_column] = sum(filtered_m .* (Smoothed_kernel_function.(smoothed_kernel, h_intepolate, filtered_rnorm,3)))
        end
        for column_name in working_column_names
            filtered_dfdata[!, column_name] ./= filtered_dfdata[!, divided_column]
            quantity_result[column_name] = sum(filtered_m .* (filtered_dfdata[!, column_name]) .* (Smoothed_kernel_function.(smoothed_kernel, h_intepolate, filtered_rnorm,3)))
        end
    end
    return quantity_result
end

"""
    gradient_quantity_intepolate(data::PhantomRevealerDataFrame, reference_point::Vector, column_name::String, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the gradient value of every requested quantities at the given reference point by SPH interpolation.
Those points whose neighborhood has no particles around it would be label as `NaN`

The gradient value of any arbitrary scalar quantity A would be estimated by

∇A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))∇W(r-r_b)

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `column_name :: String`: The quantity that would be interpolated.
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].
- `density_value :: Union{Nothing,Float64} = nothing`: The density of fluid at reference_point. If `nothing` will calculate it automatically.
- `quantity_value :: Union{Nothing,Float64} = nothing`: The value of quantitiy at reference_point. If `nothing` will calculate it automatically.

# Returns
- `Vector`: The gradient vector of density at the reference point. The coordinate would be as same as the input(depends on "coordinate_flag").

# Example
```julia
prdf_list = read_phantom(dumpfile_00000,"all")
COM2star!(prdf_list, prdf_list[end],1)
data :: PhantomRevealerDataFrame = prdf_list[1]
column_name :: String = ["vx"]
reference_point :: Vector = [25.0, π/2, 0.0]  # Choosing the cylindrical coordinate
smoothed_kernel :: Function = M6_spline
∇vx = gradient_quantity_intepolate(data, reference_point, column_name, smoothed_kernel, "closest", "polar")
```
"""
function gradient_quantity_intepolate(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    column_name::String,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true,
    density_value::Union{Nothing,Float64} = nothing,
    quantity_value::Union{Nothing,Float64} = nothing
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    "cart" = cartitian
    "polar" = cylindrical
    """
    grad_quant::Vector = Vector{Float64}(undef, 3)
    # Return NaN if no particle in the data
    if nrow(data.dfdata) == 0
        fill!(grad_quant, NaN)
        return grad_quant
    end

    # Preparing and verifing the data
    if !(hasproperty(data.dfdata, column_name))
        error("IntepolateError: No matching column '$(column_name)'.")
    end
    if coordinate_flag == "polar"
        ϕ =  reference_point[2]
        reference_point = _cylin2cart(reference_point)
    end

    # Estimating necessary quantity if not provided
    if isnothing(density_value)
        density_value = density(data,reference_point,smoothed_kernel,h_mode,"cart",Identical_particles=Identical_particles)
    end
    if isnothing(quantity_value)
        quantity_value = quantity_intepolate(data,reference_point,[column_name],smoothed_kernel,h_mode,"cart",Identical_particles=Identical_particles)[column_name]
    end

    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    rnorm, xyzref = get_r_ref(data, reference_point)
    quantity_array = data.dfdata[!,column_name]
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(rnorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, rnorm, h_mode) #_easy_estimate_h_intepolate(dfdata, rnorm, 1.0)
    
    # Kernel interpolation
    if (h_intepolate == 0.0)
        fill!(grad_quant, NaN)
        return grad_quant
    else
        truncate_radius = truncate_multiplier * h_intepolate
        mask_rnorm = rnorm .< truncate_radius
        xyz_filtered = xyzref[mask_rnorm, :]
        filtered_m = mass_array[mask_rnorm]
        filtered_Ab = quantity_array[mask_rnorm]

        # bufferf represent the first term of formula, buffers represent the second term of formula
        buffer_array = zeros(Float64, size(xyz_filtered))
        for i = 1:size(xyz_filtered)[1]
            buffer_array[i, :] =
                ((filtered_Ab[i] - quantity_value) * filtered_m[i]) .* Smoothed_gradient_kernel_function(
                    smoothed_kernel,
                    h_intepolate,
                    xyz_filtered[i, :],
                ) ./density_value 
        end
        for j in eachindex(grad_quant)
            grad_quant[j] = sum(buffer_array[:, j])
        end

        if coordinate_flag == "polar"
            cylindrical_grad_quant = _vector_cart2cylin(ϕ,grad_quant...)
            return cylindrical_grad_quant
        else
            return grad_quant
        end
    end
end

"""
    divergence_quantity_intepolate(data::PhantomRevealerDataFrame, reference_point::Vector, column_name::String, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the divergence value of every requested quantities at the given reference point by SPH interpolation.
Those points whose neighborhood has no particles around it would be label as `NaN`

The divergence of any arbitrary vector quantity A would be estimated by

∇⋅A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))⋅∇W(r-r_b)

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `column_name :: String`: The quantity that would be interpolated. For vector quantities (e.g., data columns named "vx", "vy", "vz"), you only need to provide the common prefix of the vector name, such as "v".
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].
- `density_value :: Union{Nothing,Float64} = nothing`: The density of fluid at reference_point. If `nothing` will calculate it automatically.
- `quantity_value :: Union{Nothing,Vector{Float64}} = nothing`: The vector of quantitiy at reference_point. If `nothing` will calculate it automatically.
- `quantity_coordinate_flag :: String = "cart"`: The coordinate system of `quantity_value`.

# Returns
- `Float64`: The divergence of quantity at the reference point.

"""
function divergence_quantity_intepolate(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    column_name::String,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true,
    density_value::Union{Nothing,Float64} = nothing,
    quantity_value::Union{Nothing,Vector{Float64}} = nothing,
    quantity_coordinate_flag::String = "cart"
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    "cart" = cartitian
    "polar" = cylindrical
    """
    # Return NaN if no particle in the data
    if nrow(data.dfdata) == 0
        return NaN
    end

    # Constructing the column names for interpolation
    cart_column_names :: Vector{String} = ["$(column_name)x","$(column_name)y","$(column_name)z"]
    cylin_column_names :: Vector{String} = ["$(column_name)s","$(column_name)ϕ","$(column_name)z"]
    vector_column_names :: Union{Nothing,Vector{String}} = nothing
    cylin_components_flag = false
    for vcolumn_name in cart_column_names
        if !(hasproperty(data.dfdata,vcolumn_name))
            if coordinate_flag == "polar"
                for cvcolumn_name in cylin_column_names
                    if !(hasproperty(data.dfdata,cvcolumn_name))
                        error("IntepolateError: No matching vector component in columns!")
                    end
                end
                cylin_components_flag = true
                break
            else
                error("IntepolateError: No matching vector component in columns!")
            end
        end
    end
    if cylin_components_flag
        vector_column_names = cylin_column_names
    else
        vector_column_names = cart_column_names
    end

    # Preparing and verifing the data
    if coordinate_flag == "polar"
        ϕ =  reference_point[2]
        reference_point = _cylin2cart(reference_point)
    end
    
    # Estimating necessary quantity if not provided
    if isnothing(density_value)
        density_value = density(data,reference_point,smoothed_kernel,h_mode,"cart",Identical_particles=Identical_particles)
    end
    if isnothing(quantity_value)
        temp_quantity_value_vector = zeros(Float64,3)
        for (i,vcolumn_name) in enumerate(vector_column_names)
            temp_quantity_value_vector[i] = quantity_intepolate(data,reference_point,[vcolumn_name],smoothed_kernel,h_mode,"cart",Identical_particles=Identical_particles)[vcolumn_name]
        end
        if cylin_components_flag
            quantity_value_vector = _vector_cylin2cart(temp_quantity_value_vector)
        else
            quantity_value_vector = temp_quantity_value_vector
        end
    else
        if quantity_coordinate_flag == "polar"
            quantity_value_vector = _vector_cylin2cart(ϕ,quantity_value...)
        else
            quantity_value_vector = quantity_value
        end
    end

    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    rnorm, xyzref = get_r_ref(data, reference_point)
    quantity_array = zeros(Float64,length(rnorm),3)
    for (i,vcolumn_name) in enumerate(vector_column_names)
        quantity_array[:,i] = data.dfdata[!,vcolumn_name]
    end
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(rnorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, rnorm, h_mode) #_easy_estimate_h_intepolate(dfdata, rnorm, 1.0)

    # Kernel interpolation
    if (h_intepolate == 0.0)
        return NaN
    else
        truncate_radius = truncate_multiplier * h_intepolate
        mask_rnorm = rnorm .< truncate_radius
        xyz_filtered = xyzref[mask_rnorm, :]
        filtered_m = mass_array[mask_rnorm]
        filtered_Ab = quantity_array[mask_rnorm,:]
        filtered_AbsAa = filtered_Ab .- quantity_value_vector'
        buffer_array = zeros(Float64, length(filtered_m))
        for i = 1:size(xyz_filtered)[1]
            buffer_array[i] = dot(filtered_AbsAa[i,:] , Smoothed_gradient_kernel_function(
                smoothed_kernel,
                h_intepolate,
                xyz_filtered[i, :],
            ))
        end
        buffer_array .*= (filtered_m./density_value)
        diver_quant = sum(buffer_array)
        return diver_quant
    end
end

"""
    curl_quantity_intepolate(data::PhantomRevealerDataFrame, reference_point::Vector, column_name::String, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the curl value of every requested quantities at the given reference point by SPH interpolation.
Those points whose neighborhood has no particles around it would be label as `NaN`

The curl of any arbitrary vector quantity A would be estimated by

∇×A(r) = -(1/ρ(r))∑_b m_b*(A_b-A(r))×∇W(r-r_b)

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `column_name :: String`: The quantity that would be interpolated. For vector quantities (e.g., data columns named "vx", "vy", "vz"), you only need to provide the common prefix of the vector name, such as "v".
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].
- `density_value :: Union{Nothing,Float64} = nothing`: The density of fluid at reference_point. If `nothing` will calculate it automatically.
- `quantity_value :: Union{Nothing,Vector{Float64}} = nothing`: The vector of quantitiy at reference_point. If `nothing` will calculate it automatically.
- `quantity_coordinate_flag :: String = "cart"`: The coordinate system of `quantity_value`.

# Returns
- `Vector{Float64}`: The curl of quantity at the reference point.

"""
function curl_quantity_intepolate(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    column_name::String,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true,
    density_value::Union{Nothing,Float64} = nothing,
    quantity_value::Union{Nothing,Vector{Float64}} = nothing,
    quantity_coordinate_flag::String = "cart"
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    "cart" = cartitian
    "polar" = cylindrical
    """
    curl_quant::Vector = Vector{Float64}(undef, 3)
    # Return NaN if no particle in the data
    if nrow(data.dfdata) == 0
        fill!(curl_quant, NaN)
        return curl_quant
    end

    # Constructing the column names for interpolation
    cart_column_names :: Vector{String} = ["$(column_name)x","$(column_name)y","$(column_name)z"]
    cylin_column_names :: Vector{String} = ["$(column_name)s","$(column_name)ϕ","$(column_name)z"]
    vector_column_names :: Union{Nothing,Vector{String}} = nothing
    cylin_components_flag = false
    for vcolumn_name in cart_column_names
        if !(hasproperty(data.dfdata,vcolumn_name))
            if coordinate_flag == "polar"
                for cvcolumn_name in cylin_column_names
                    if !(hasproperty(data.dfdata,cvcolumn_name))
                        error("IntepolateError: No matching vector component in columns!")
                    end
                end
                cylin_components_flag = true
                break
            else
                error("IntepolateError: No matching vector component in columns!")
            end
        end
    end
    if cylin_components_flag
        vector_column_names = cylin_column_names
    else
        vector_column_names = cart_column_names
    end

    # Preparing and verifing the data
    if coordinate_flag == "polar"
        ϕ =  reference_point[2]
        reference_point = _cylin2cart(reference_point)
    end
    
    # Estimating necessary quantity if not provided
    if isnothing(density_value)
        density_value = density(data,reference_point,smoothed_kernel,h_mode,"cart",Identical_particles=Identical_particles)
    end
    if isnothing(quantity_value)
        temp_quantity_value_vector = zeros(Float64,3)
        for (i,vcolumn_name) in enumerate(vector_column_names)
            temp_quantity_value_vector[i] = quantity_intepolate(data,reference_point,[vcolumn_name],smoothed_kernel,h_mode,"cart",Identical_particles=Identical_particles)[vcolumn_name]
        end
        if cylin_components_flag
            quantity_value_vector = _vector_cylin2cart(temp_quantity_value_vector)
        else
            quantity_value_vector = temp_quantity_value_vector
        end
    else
        if quantity_coordinate_flag == "polar"
            quantity_value_vector = _vector_cylin2cart(ϕ,quantity_value...)
        else
            quantity_value_vector = quantity_value
        end
    end

    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    rnorm, xyzref = get_r_ref(data, reference_point)
    quantity_array = zeros(Float64,length(rnorm),3)
    for (i,vcolumn_name) in enumerate(vector_column_names)
        quantity_array[:,i] = data.dfdata[!,vcolumn_name]
    end
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(rnorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, rnorm, h_mode) #_easy_estimate_h_intepolate(dfdata, rnorm, 1.0)
    
    # Kernel intepolation
    if (h_intepolate == 0.0)
        fill!(curl_quant, NaN)
        return curl_quant
    else
        truncate_radius = truncate_multiplier * h_intepolate
        mask_rnorm = rnorm .< truncate_radius
        xyz_filtered = xyzref[mask_rnorm, :]
        filtered_m = mass_array[mask_rnorm]
        filtered_Ab = quantity_array[mask_rnorm,:]
        filtered_AbsAa = filtered_Ab .- quantity_value_vector'

        buffer_array = zeros(Float64, size(xyz_filtered))
        for i = 1:size(xyz_filtered)[1]
            buffer_array[i, :] =
                -cross(filtered_AbsAa[i,:], Smoothed_gradient_kernel_function(
                    smoothed_kernel,
                    h_intepolate,
                    xyz_filtered[i, :],
                )) 
            buffer_array[i,:] .*= (filtered_m[i]/density_value) 
        end
        for j in eachindex(curl_quant)
            curl_quant[j] = sum(buffer_array[:, j])
        end

        if coordinate_flag == "polar"
            cylindrical_curl_quant = _vector_cart2cylin(ϕ,curl_quant...)
            return cylindrical_curl_quant
        else
            return curl_quant
        end
    end
end

"""
    surface_density(data::PhantomRevealerDataFrame, reference_point::Vector, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the surface density Σ at the given reference point by SPH interpolation

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].

# Returns
- `Float64`: The value of surface density at the reference point.

# Example
```julia
prdf_list = read_phantom(dumpfile_00000,"all")
COM2star!(prdf_list, prdf_list[end],1)
data :: PhantomRevealerDataFrame = prdf_list[1]
reference_point :: Vector = [25.0, π/2]  # Choosing the cylindrical coordinate
smoothed_kernel :: Function = M6_spline
surf_dens :: Vector = surface_density(data, reference_point, smoothed_kernel, "closest", "polar")
```
"""
function surface_density(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    reference_point is in "2D"
    "cart" = cartitian
    "polar" = polar
    """
    # Return 0.0 if no particle in the data
    if nrow(data.dfdata) == 0
        return 0.0
    end
    # Preparing and verifing the data
    if coordinate_flag == "polar"
        reference_point = _cylin2cart(reference_point)
    end

    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    snorm = get_snorm_ref(data, reference_point)
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(snorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, snorm, h_mode) #_easy_estimate_h_intepolate(dfdata, rnorm, 1.0)

    # Kernel interpolation
    if (h_intepolate == 0.0)
        surface_density = 0.0
    else
        truncate_radius = truncate_multiplier * h_intepolate
        filtered_indices = findall(r -> r <= truncate_radius, snorm)
        filtered_snorm = snorm[filtered_indices]
        filtered_m = mass_array[filtered_indices]
        surface_density = sum(
            filtered_m .*
            Smoothed_kernel_function.(smoothed_kernel, h_intepolate, filtered_snorm, 2),
        )
    end
    return surface_density
end

"""
    gradient_surface_density(data::PhantomRevealerDataFrame, reference_point::Vector, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the gredient of surface density ∇Σ at the given reference point by SPH interpolation

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].

# Returns
- `Vector`: The gradient vector of surface density at the reference point. The coordinate would be as same as the input(depends on "coordinate_flag").

# Example
```julia
prdf_list = read_phantom(dumpfile_00000,"all")
COM2star!(prdf_list, prdf_list[end],1)
data :: PhantomRevealerDataFrame = prdf_list[1]
reference_point :: Vector = [25.0, π/2]  # Choosing the cylindrical coordinate
smoothed_kernel :: Function = M6_spline
grad_surf_dens :: Vector = grad_surface_density(data, reference_point, smoothed_kernel, "closest", "polar")
```
"""
function gradient_surface_density(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    reference_point is in "2D"
    "cart" = cartitian
    "polar" = polar
    """
    # Return NaN if no particle in the data
    grad_surface_density::Vector = Vector{Float64}(undef, 2)
    if nrow(data.dfdata) == 0
        fill!(grad_surface_density, NaN)
        return grad_surface_density
    end

    # Preparing and verifing the data
    if coordinate_flag == "polar"
        ϕ =  reference_point[2]
        reference_point = _cylin2cart(reference_point)
    end

    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    snorm, xyref = get_s_ref(data, reference_point)
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(snorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, snorm, h_mode) #_easy_estimate_h_intepolate(dfdata, rnorm, 1.0)

    # Kernel interpolation
    if (h_intepolate == 0.0)
        fill!(grad_surface_density, NaN)
        return grad_surface_density
    else
        truncate_radius = truncate_multiplier * h_intepolate
        mask_snorm = snorm .< truncate_radius
        xy_filtered = xyref[mask_snorm, :]
        filtered_m = mass_array[mask_snorm]
        buffer_array = zeros(Float64, size(xy_filtered))
        for i = 1:size(xy_filtered)[1]
            buffer_array[i, :] =
                filtered_m[i] .* Smoothed_gradient_kernel_function(
                    smoothed_kernel,
                    h_intepolate,
                    xy_filtered[i, :],
                )
        end
        for j in eachindex(grad_surface_density)
            grad_surface_density[j] = sum(buffer_array[:, j])
        end
        if coordinate_flag == "polar"
            cylindrical_grad_surface_density = _vector_cart2cylin(ϕ,grad_surface_density...)
            return cylindrical_grad_surface_density
        else
            return grad_surface_density
        end
    end
end

"""
    quantity_intepolate_2D(data::PhantomRevealerDataFrame, reference_point::Vector, column_names::Vector{String}, smoothed_kernel:: Function = M5_spline,h_mode::String="closest", coordinate_flag::String = "cart";Identical_particles::Bool=true)
Calculate the value of every requested quantities at the given reference point by SPH interpolation ON THE 2D PLANE.
Those points whose neighborhood has no particles around it would be label as `NaN`

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
- `reference_point :: Vector`: The reference point for interpolation.
- `Sigmai :: Float64`: The surface density in current location.
- `column_names :: Vector{String}`: The quantities that would be interpolated.
- `smoothed_kernel :: Function = M5_spline`: The Kernel function for interpolation.
- `h_mode :: String="closest"`: The mode for finding a proper smoothed radius. (Allowed value: "closest", "mean")
- `coordinate_flag :: String = "cart"`: The coordinate system that is used for given the target. Allowed value: ("cart", "polar") 

# Keyword argument
- `Identical_particles :: Bool = true`: Whether all the particles is identical in `data`. If false, particle mass would try to access the mass column i.e. data.data_dict["m"].

# Returns
- `Dict{String, Float64}`: A dictionary that contains the value of quantities at the reference point ON THE 2D PLANE.

# Example
```julia
prdf_list = read_phantom(dumpfile_00000,"all")
COM2star!(prdf_list, prdf_list[end],1)
data :: PhantomRevealerDataFrame = prdf_list[1]
column_names :: Vector{String} = ["vr", "vϕ", "e", "St"]
reference_point :: Vector = [25.0, π/2]  # Choosing the cylindrical coordinate
smoothed_kernel :: Function = M6_spline
quantities_dict :: Dict{Float64} = quantity_intepolate2D(data, reference_point, column_names, smoothed_kernel, "closest", "polar")
```
"""
function quantity_intepolate_2D(
    data::PhantomRevealerDataFrame,
    reference_point::Vector,
    Sigmai::Float64,
    column_names::Vector{String},
    smoothed_kernel::Function = M5_spline,
    h_mode::String = "closest",
    coordinate_flag::String = "cart";
    Identical_particles::Bool=true
)
    """
    Here recommended to use a single type of particle.
    coordinate_flag is the coordinate system that the reference_point is given
    reference_point is in "2D"
    "cart" = cartitian
    "polar" = polar
    """
    # Constructing the column names for interpolation
    working_column_names = copy(column_names)
    quantity_result = Dict{String,Float64}()

    # Return NaN if no particle in the data
    if nrow(data.dfdata) == 0 && Sigmai == 0.0
        for column_name in working_column_names
            quantity_result[column_name] = NaN
        end
        return quantity_result
    end
    
    # Preparing and verifing the data
    for column_name in column_names
        if !(hasproperty(data.dfdata, column_name))
            error("IntepolateError: No matching column '$(column_name)'.")
        end
    end
    if coordinate_flag == "polar"
        reference_point = _cylin2cart(reference_point)
    end

    # Evaluating a proper smoothed radius
    truncate_multiplier = KernelFunctionValid()[nameof(smoothed_kernel)]
    snorm = get_snorm_ref(data, reference_point)
    if Identical_particles
        particle_mass = data.params["mass"]
        mass_array = fill(particle_mass,length(snorm))
    else
        mass_array = data.dfdata[!,"m"]
    end
    h_intepolate = estimate_h_intepolate(data, snorm, h_mode)

    # Kernel interpolation
    dfdata = data.dfdata
    if (h_intepolate == 0.0)
        for column_name in working_column_names
            quantity_result[column_name] = NaN
        end
    else
        truncate_radius = truncate_multiplier * h_intepolate
        indices = findall(x -> x <= truncate_radius, snorm)
        filtered_dfdata = dfdata[indices, :]
        filtered_m = mass_array[indices]
        filtered_snorm = snorm[indices]
        for column_name in working_column_names
            filtered_dfdata[!, column_name] ./= Sigmai
            quantity_result[column_name] = sum(filtered_m .* (filtered_dfdata[!, column_name]) .* (Smoothed_kernel_function.(smoothed_kernel,h_intepolate,filtered_snorm,2)))
        end
    end
    return quantity_result
end
