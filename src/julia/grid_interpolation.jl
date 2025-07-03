"""
The grid SPH interpolation
    by Wei-Shan Su,
    June 27, 2024

Generate a `gridbackend`, calculate the value for each column.

# Progression
Step 1. Generate a 3d KDTree to speed up the searching of particles

Step 2. Generate a grid for analysis. The grid is base on a struct gridbackend which is defined in the "grid.jl" file

Step 3. Calculate the result for each point by using the SPH interpolation. The calculation for each point is defined in "physical_quantity.jl"
"""


"""
Cylinder_Grid_analysis(
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int},
    ϕn :: Int,
    z_params::Tuple{Float64,Float64,Int};
    column_names::Vector{Symbol}=Symbol[],
    gradient_column_names::Vector{Symbol}=Symbol[],
    divergence_column_names::Vector{Symbol}=Symbol[],
    curl_column_names::Vector{Symbol}=Symbol[],
    smoothed_kernel::Type{K}=M5_spline,
    Identical_particles::Bool=true
) where {K<:AbstractSPHKernel}

Perform 3D SPH interpolation and differential analysis over a cylindrical grid.

This function generates a cylindrical grid in (s, ϕ, z) coordinates and evaluates interpolated physical quantities (e.g., density, gradients, divergences, curls) from SPH particle data. Results are stored in gridbackend objects matching the geometry of the grid.

# Parameters
- `data`: The input SPH particle data (`PhantomRevealerDataFrame`).
- `s_params`: Tuple specifying (s_min, s_max, s_resolution).
- `ϕn`: Number of bins in the azimuthal direction (ϕ).
- `z_params`: Tuple specifying (z_min, z_max, z_resolution).

# Keyword Arguments
| Argument                | Default       | Description                                                  |
|-------------------------|---------------|--------------------------------------------------------------|
| `column_names`          | `Symbol[]`    | List of scalar field names to interpolate.                   |
| `gradient_column_names` | `Symbol[]`    | List of scalar fields to compute ∇field in cylindrical form. |
| `divergence_column_names` | `Symbol[]`  | List of vector fields to compute ∇⋅field. DO NOT PASS THE DIRECTION SUFFIX                   |
| `curl_column_names`     | `Symbol[]`    | List of vector fields to compute ∇×field in cylindrical form. DO NOT PASS THE DIRECTION SUFFIX     |
| `smoothed_kernel`       | `M5_spline`   | SPH kernel type used in interpolation.                       |
| `Identical_particles`   | `true`        | Whether all particles have identical mass.                   |

# Returns
A tuple of interpolated results (in the order of column names, gradients, divergences, curls), each entry being a `gridbackend`. If `:rho` is included, the corresponding density and ∇ρ fields are prepended to the output.

The result ordering matches the `result_column_names` string vector constructed internally.

"""
function Cylinder_Grid_analysis(
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int},
    ϕn :: Int,
    z_params::Tuple{Float64,Float64,Int};
    column_names::Vector{Symbol}=Symbol[],
    gradient_column_names::Vector{Symbol}=Symbol[],
    divergence_column_names::Vector{Symbol}=Symbol[],
    curl_column_names::Vector{Symbol}=Symbol[],
    smoothed_kernel::Type{K}=M5_spline,
    Identical_particles::Bool=true
) where {K<:AbstractSPHKernel}
    @info "Start 3D cylinder grid analysis."
    
    # The column subfix in cartisian coordinate system
    column_suffixes = ["x", "y", "z"]
    cylin_column_suffixes = ["s", "ϕ", "z"]

    # Checking data before interpolation
    ###############################
    if (data.params["Origin_sink_id"] == -1)
        error("IntepolateError: Wrong origin located!")
    end

    # Filter rho in column names
    pure_column_names = [column for column in column_names if column != :rho]
    pure_gradient_column_names = [column for column in gradient_column_names if column != :rho]
    pure_divergence_column_names = [column for column in divergence_column_names if column != :rho]
    pure_curl_column_names = [column for column in curl_column_names if column != :rho]
    
    # Check missing columns. 
    required_column_names :: Vector{Symbol} = [pure_column_names...,
                                               pure_gradient_column_names..., 
                                               [Symbol(cn, sf) for sf in column_suffixes for cn in pure_divergence_column_names]...,  
                                               [Symbol(cn, sf) for sf in column_suffixes for cn in pure_curl_column_names]...,]
    for column in required_column_names
        if !(hasproperty(data.dfdata, column))
            error("IntepolateError: Missing column name $column !")
        end
    end

    # Generate kd tree in 3D space
    kdtree3d = Generate_KDtree(data, 3)
    
    # Generate Edge-on grid 
    iN = (s_params[3], ϕn, z_params[3])
    empty_gridbackend::gridbackend = cylinder_grid_generator(s_params[1], s_params[2], z_params[1], z_params[2], iN)

    # Generate the coordinate array for the grid interpolation
    gridv = generate_coordinate_grid(empty_gridbackend)

    # Preparation of result dictionary
    Ncolumnitp = length(pure_column_names)
    Ngradcolumnitp = length(pure_gradient_column_names)
    Ndivcolumnitp = length(pure_divergence_column_names)
    Ncurlcolumnitp = length(pure_curl_column_names)

    # Check whether density is in the column name
    hasrho = false
    rhoitp = nothing
    if :rho in column_names
        hasrho = true
        rhoitp = deepcopy(empty_gridbackend)
    end

    has∇rho = false
    ∇rhoitp = nothing
    if :rho in gradient_column_names
        has∇rho = true
        ∇rhoitp = ntuple(i -> deepcopy(empty_gridbackend), 3)
    end

    # Constructing result Tuple
    columnitp_Tuple = ntuple(i -> deepcopy(empty_gridbackend), Ncolumnitp)
    gradcolumnitp_Tuple = ntuple(j -> ntuple(i -> deepcopy(empty_gridbackend),3),  Ngradcolumnitp)
    divcolumnitp_Tuple = ntuple(i -> deepcopy(empty_gridbackend), Ndivcolumnitp)
    curlcolumnitp_Tuple = ntuple(j -> ntuple(i -> deepcopy(empty_gridbackend), 3), Ncurlcolumnitp)

    
    

    # Make the overall column names
    result_column_names :: Vector{String} = [string.(pure_column_names)..., 
                                            [string("∇", cn, sf) for sf in cylin_column_suffixes for cn in pure_gradient_column_names]..., 
                                            [string("∇⋅", cn) for cn in pure_divergence_column_names]...,  
                                            [string("∇×", cn, sf) for sf in cylin_column_suffixes for cn in pure_curl_column_names]...,]
    if has∇rho
        pushfirst!(result_column_names, "∇rhoz")
        pushfirst!(result_column_names, "∇rhoy")
        pushfirst!(result_column_names, "∇rhox")
    end
    if hasrho
        pushfirst!(result_column_names, "rho")
    end
    # Prepare the full InterpolationInput
    input_normal_interpolation = InterpolationInput(data, pure_column_names, smoothed_kernel, Identical_particles=Identical_particles)
    input_first_deriviative_interpolation = InterpolationInput(data, required_column_names, smoothed_kernel, Identical_particles=Identical_particles)

    # Prepare h_array
    h_array = input_normal_interpolation.h

    # Get Valid Kernel range
    multiplier = KernelFunctionValid(K, eltype(h_array))

    # Generate workspace
    workspace = zeros(Float64, Ncolumnitp)

    # Iteration
    @threads for i in eachindex(gridv)
        target_cylin = gridv[i]
        ϕ = target_cylin[2]
        target = _cylin2cart(target_cylin)
        neighbor_indices, ha = get_Neighbor_indices(kdtree3d, target, multiplier, h_array)

        # Interpolate density
        if hasrho
            rhoitp.grid[i] = density(input_normal_interpolation, target, ha, neighbor_indices, itpGather)
        end
        if has∇rho
            ∇rhoitp[1].grid[i], ∇rhoitp[2].grid[i], ∇rhoitp[3].grid[i] = gradient_density(input_normal_interpolation, target, ha, neighbor_indices, itpGather)
        end

        # Get subarray for interpolation
        buf = ntuple(n -> @view(columnitp_Tuple[n].grid[i]), Ncolumnitp)

        # quantities intepolate
        quantities_interpolate!(buf, workspace, input_normal_interpolation, target, ha, neighbor_indices, itpGather)

        # Gradient intepolate
        for j in eachindex(pure_gradient_column_names)
            grad_result_cart = gradient_quantity_intepolate(input_first_deriviative_interpolation, target, ha, neighbor_indices, j, itpGather)
            gradcolumnitp_Tuple[j][1].grid[i], gradcolumnitp_Tuple[j][2].grid[i], gradcolumnitp_Tuple[j][3].grid[i] = _vector_cart2cylin(ϕ, grad_result_cart...)
        end

        # Divergence
        for j in eachindex(pure_divergence_column_names)
            ingrident = (3*(j-1) + 1, 3*(j-1) + 2, 3*(j-1) + 3)
            divcolumnitp_Tuple[j].grid[i] = divergence_quantity_intepolate(input_first_deriviative_interpolation, target, ha, neighbor_indices, ingrident..., itpGather)
        end

        # Curl intepolate
        for j in eachindex(pure_curl_column_names)
            ingrident = (3*(j-1) + 1, 3*(j-1) + 2, 3*(j-1) + 3)
            curl_result_cart = curl_quantity_intepolate(input_first_deriviative_interpolation, target, ha, neighbor_indices, ingrident..., itpGather)
            curlcolumnitp_Tuple[j][1].grid[i], curlcolumnitp_Tuple[j][2].grid[i], curlcolumnitp_Tuple[j][3].grid[i] = _vector_cart2cylin(ϕ, curl_result_cart...)
        end
    end

    # Construct result
    result_value = typeof(empty_gridbackend)[]
    if hasrho
        push!(result_value, rhoitp)
    end
    if has∇rho
        push!(result_value, ∇rhoitp...)
    end
    push!(result_value, columnitp_Tuple...)
    for subtuple in gradcolumnitp_Tuple
        push!(result_value, subtuple...)
    end
    push!(result_value, divcolumnitp_Tuple...)
    for subtuple in curlcolumnitp_Tuple
        push!(result_value, subtuple...)
    end

    @assert length(result_column_names) == length(result_value)

    result = Dict{String, typeof(empty_gridbackend)}()
    for (key, val) in zip(result_column_names, result_value)
        result[key] = val
    end
    @info "End 3D cylinder grid analysis."
    return result
end

"""
Disc_Grid_analysis(
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int},
    ϕn::Int,
    ϕnmid::Int = 16,
    z_params_mid::Tuple{Float64,Float64,Int} = (-28.0, 28.0, 100);
    midz_func :: Union{Nothing, Interpolations.Extrapolation, Function} = nothing,
    column_names::Vector{Symbol}=Symbol[],
    midplane_column_names::Vector{Symbol}=Symbol[],
    smoothed_kernel::Type{K}=M5_spline,
    Identical_particles::Bool=true
) where {K<:AbstractSPHKernel}

Perform SPH interpolation over a 2D cylindrical disc grid with optional midplane sampling.

This function interpolates physical quantities onto a 2D (s, ϕ) cylindrical grid using SPH data. If midplane quantities are requested, the function computes the midplane height at each grid point and interpolates corresponding 3D values at that height. Midplane height can be automatically computed via vertical SPH interpolation or supplied by user-defined `midz_func`.

# Parameters
- `data`: The SPH particle dataset (`PhantomRevealerDataFrame`).
- `s_params`: Tuple `(s_min, s_max, s_resolution)` defining radial bins.
- `ϕn`: Number of bins in azimuthal angle ϕ.
- `ϕnmid`: Number of azimuthal bins used for midplane estimation (default: 16).
- `z_params_mid`: Tuple `(z_min, z_max, z_resolution)` for midplane vertical density integration.

# Keyword Arguments
| Name                  | Default         | Description                                                                 |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| `midz_func`           | `nothing`       | Optional function to supply midplane height. If `nothing`, computed internally. |
| `column_names`        | `Symbol[]`      | Scalar fields to interpolate along the 2D LOS (line-of-sight) projection.  |
| `midplane_column_names` | `Symbol[]`    | Scalar fields to interpolate at the computed midplane height (3D).         |
| `smoothed_kernel`     | `M5_spline`     | SPH kernel used in all interpolations.                                     |
| `Identical_particles` | `true`          | Whether all particles share the same mass.                                 |

# Returns
A `Dict{String, gridbackend}` mapping quantity names to their interpolated grid results.
"""
function Disc_Grid_analysis(
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int},
    ϕn::Int,
    ϕnmid::Int = 16,
    z_params_mid::Tuple{Float64,Float64,Int} = (-28.0, 28.0, 100);
    midz_func :: Union{Nothing, Interpolations.Extrapolation, Function} = nothing,
    column_names::Vector{Symbol}=Symbol[],
    midplane_column_names::Vector{Symbol}=Symbol[],
    smoothed_kernel::Type{K}=M5_spline,
    Identical_particles::Bool=true
) where {K<:AbstractSPHKernel}
    @info "Start 2D disc grid analysis."

    # Checking data before interpolation
    ###############################
    if (data.params["Origin_sink_id"] == -1)
        error("IntepolateError: Wrong origin located!")
    end

    # Filter Sigma/rho in column names
    pure_column_names = [column for column in column_names if (column ∉ (:rho, :Sigma))]
    pure_midplane_column_names = [column for column in midplane_column_names if (column ∉ (:rho, :Sigma))]
    
    # Check missing columns. 
    required_column_names :: Vector{Symbol} = [pure_column_names...,
                                               pure_midplane_column_names...]
    for column in required_column_names
        if !(hasproperty(data.dfdata, column))
            error("IntepolateError: Missing column name $column !")
        end
    end

    # Generate kd tree in 3D space
    kdtree2d = Generate_KDtree(data, 2)
    kdtree3d = Generate_KDtree(data, 3)

    # Generate Face-on grid 
    iN = (s_params[3], ϕn)
    empty_gridbackend::gridbackend = disc_grid_generator(s_params[1], s_params[2], iN)

    # Generate the coordinate array for the grid interpolation
    gridv = generate_coordinate_grid(empty_gridbackend)

    # Preparation of result dictionary
    Ncolumnitp = length(pure_column_names)
    Nmidcolumnitp = length(pure_midplane_column_names)


    # Check whether density is in the column name
    hasSigma = false
    if :Sigma in column_names
        hasSigma = true
        Sigmaitp = deepcopy(empty_gridbackend)
    end

    hasrho = false
    if :rho in midplane_column_names
        hasrho = true
        rhoitp = deepcopy(empty_gridbackend)
    end

    # Constructing result Tuple
    columnitp_Tuple = ntuple(i -> deepcopy(empty_gridbackend), Ncolumnitp)
    midcolumnitp_Tuple = ntuple(i -> deepcopy(empty_gridbackend), Nmidcolumnitp)

    # Make the overall column names
    result_column_names :: Vector{String} = [string.(pure_column_names)..., 
                                            [string(cn,"m") for cn in pure_midplane_column_names]...]
    if hasrho
        pushfirst!(result_column_names, "rhom")
    end
    if hasSigma
        pushfirst!(result_column_names, "Sigma")
    end
    # Prepare the full InterpolationInput
    input_interpolation = InterpolationInput(data, pure_column_names, smoothed_kernel, Identical_particles=Identical_particles)
    input_midplane_interpolation = InterpolationInput(data, pure_midplane_column_names, smoothed_kernel, Identical_particles=Identical_particles)

    # Prepare h_array
    h_array = input_interpolation.h

    # Get Valid Kernel range
    multiplier = KernelFunctionValid(K, eltype(h_array))

    # Generate workspace
    workspace2d = zeros(get_type(input_interpolation), Ncolumnitp)
    workspace3d = zeros(get_type(input_midplane_interpolation), Nmidcolumnitp)

    # Prepare midplane
    midz = nothing
    if !isempty(midplane_column_names)
        if isnothing(midz_func)
            iN3d = (s_params[3], ϕnmid, z_params_mid[3])
            rhoitp3d::gridbackend = cylinder_grid_generator(s_params[1], s_params[2], z_params_mid[1], z_params_mid[2], iN3d)
            gridv3d = generate_coordinate_grid(rhoitp3d)
            @threads for i in eachindex(gridv3d)
                target_cylin = gridv3d[i]
                target = _cylin2cart(target_cylin)
                nbur, ha = get_Neighbor_indices(kdtree3d, target, multiplier, h_array)
                rhoitp3d.grid[i] = density(input_midplane_interpolation, target, ha, nbur, itpGather)
            end
            midz_func = Disc_2D_midplane_function_generator(rhoitp3d)
            midz = func2gbe(s_params[1], s_params[2], iN, func=midz_func)
        else
            midz = func2gbe(s_params[1], s_params[2], iN, func=midz_func)
        end
    end

    # Iteration
    @threads for i in eachindex(gridv)
        target_cylin = gridv[i]
        target = _cylin2cart(target_cylin)
        neighbor_indices, ha = get_Neighbor_indices(kdtree2d, target, multiplier, h_array)

        # Interpolate 2D
        if hasSigma
            Sigmaitp.grid[i] = LOS_density(input_interpolation, target, ha, neighbor_indices, itpGather)
        end

        # Get subarray for interpolation
        buf = ntuple(n -> @view(columnitp_Tuple[n].grid[i]), Ncolumnitp)
        LOS_quantities_interpolate!(buf, workspace2d, input_interpolation, target, ha, neighbor_indices, itpGather)

        # Interpolate midplane
        if !isempty(midplane_column_names)
            if !isfinite(midz.grid[i])
                for n in 1:Nmidcolumnitp
                    midcolumnitp_Tuple[n].grid[i] = get_type(input_midplane_interpolation)(NaN)
                end
                continue
            end
            mid_target = (target..., midz.grid[i])
            mid_neighbor_indices, midha = get_Neighbor_indices(kdtree3d, mid_target, multiplier, h_array)
            if hasrho
                rhoitp.grid[i] = density(input_midplane_interpolation, mid_target, midha, mid_neighbor_indices, itpGather)
            end
            # Get subarray for interpolation
            midbuf = ntuple(n -> @view(midcolumnitp_Tuple[n].grid[i]), Nmidcolumnitp)

            # quantities intepolate
            quantities_interpolate!(midbuf, workspace3d, input_midplane_interpolation, mid_target, midha, mid_neighbor_indices, itpGather)
        end
    end

    # Construct result
    result_value = typeof(empty_gridbackend)[]
    if hasSigma
        push!(result_value, Sigmaitp)
    end
    if hasrho
        push!(result_value, rhoitp)
        
    end
    push!(result_value, columnitp_Tuple...)
    push!(result_value, midcolumnitp_Tuple...)

    @assert length(result_column_names) == length(result_value)

    result = Dict{String, typeof(empty_gridbackend)}()
    for (key, val) in zip(result_column_names, result_value)
        result[key] = val
    end
    @info "End 2D disc grid analysis."
    return result
end

function Disc_2D_midplane_function_generator(rho_gbe::gridbackend)
    s_array = rho_gbe.axes[1]
    z_array = rho_gbe.axes[3]
    z_grid :: Array = zeros(Float64,rho_gbe.dimension[1],rho_gbe.dimension[2])
    
    grid3d = rho_gbe.grid
    @threads for idx in CartesianIndices(z_grid)
        rhos ::Vector{Float64} = grid3d[idx[1],idx[2],:]
        if sum(rhos) == 0.0
            z_grid[idx] = NaN64
        else
            z_grid[idx] = z_array[findmax(rhos)[2]]
        end
    end
    z_grid = hcat(z_grid,z_grid[:,1])
    ϕ_array = LinRange(0.0,2π,length(rho_gbe.axes[2])+1)
    zfunc :: Interpolations.Extrapolation = LinearInterpolation((s_array,ϕ_array),z_grid)
    return zfunc
end

function Disc_2D_midplane_function_generator(   
    data::PhantomRevealerDataFrame,
    s_params::Tuple{Float64,Float64,Int} = (10.0, 120.0, 111),
    ϕn::Int = 16,
    z_params::Tuple{Float64,Float64,Int} = (-28.0, 28.0, 100);
    smoothed_kernel::Type{K} = M5_spline,
    Identical_particles::Bool=true
    ) where {K<:AbstractSPHKernel}
    # Generate a simple cylinder
    iN = (s_params[3], ϕn, z_params[3])
    rhoitp = cylinder_grid_generator(s_params[1], s_params[2], z_params[1], z_params[2], iN)
    gridv = generate_coordinate_grid(rhoitp)
    input = InterpolationInput(data, Symbol[], smoothed_kernel, Identical_particles=Identical_particles)
    kdtree = Generate_KDtree(data, 3)
    multiplier = KernelFunctionValid(K, eltype(input.h))
    @threads for i in eachindex(gridv)
        target_cylin = gridv[i]
        target = _cylin2cart(target_cylin)

        nbur, ha = get_Neighbor_indices(kdtree, target, multiplier, input.h)

        rhoitp.grid[i] = density(input, target, ha, nbur)
    end
    return Disc_2D_midplane_function_generator(rhoitp)
end
