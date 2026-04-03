"""
    build_input(data::ParticleDataFrame,
                mass_from_column::MassFromColumn;
                scalars::Tuple{Vararg{Symbol}}=(),
                gradients::Tuple{Vararg{Symbol}}=(),
                divergences::Tuple{Vararg{Symbol}}=(),
                curls::Tuple{Vararg{Symbol}}=(),
                smoothed_kernel::Type{K}=M5_spline) where {K<:AbstractSPHKernel}

Construct a CPU-side `InterpolationInput` and its corresponding
`InterpolationCatalog` from a `ParticleDataFrame`, using a mass column selected
by `mass_from_column`.

This method determines the required extra quantity columns from the requested
scalar, gradient, divergence, and curl names, verifies that all required
columns are present in `data`, materializes those columns, and forwards the
result to the lower-level CPU `build_input` helper.

The base interpolation fields are read from `data[:,:x]`, `data[:,:y]`,
`data[:,:z]`, `data[:,:h]`, and `data[:,:rho]`. Particle mass is taken from
the column specified by `mass_from_column.name`.

# Parameters
- `data::ParticleDataFrame`: Particle dataset containing the base interpolation
  fields and all requested extra quantity columns.
- `mass_from_column::MassFromColumn`: Mass-column selector indicating which
  column in `data` should be used as the particle mass input.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `scalars` | `Tuple{Vararg{Symbol}}` | `()` | Names of extra scalar quantities to interpolate directly. |
| `gradients` | `Tuple{Vararg{Symbol}}` | `()` | Names of extra scalar quantities whose spatial gradients should be computed. |
| `divergences` | `Tuple{Vararg{Symbol}}` | `()` | Base names of extra vector quantities whose divergences should be computed. |
| `curls` | `Tuple{Vararg{Symbol}}` | `()` | Base names of extra vector quantities whose curls should be computed. |
| `smoothed_kernel` | `Type{K}` where `K<:AbstractSPHKernel` | `M5_spline` | SPH kernel type used when constructing the `InterpolationInput`. |

# Returns
- `Tuple{InterpolationInput,InterpolationCatalog{3,N,G,Div,C,L}}`: A pair
  consisting of the materialized CPU interpolation input and the corresponding
  3D interpolation catalog for the requested extra quantities.

"""
function build_input(
    data::ParticleDataFrame,
    mass_from_column::MassFromColumn;
    scalars::Tuple{Vararg{Symbol}} = (),
    gradients::Tuple{Vararg{Symbol}} = (),
    divergences::Tuple{Vararg{Symbol}} = (),
    curls::Tuple{Vararg{Symbol}} = (),
    smoothed_kernel::Type{K} = M5_spline,
) where {K<:AbstractSPHKernel}
    column_names = _quantity_column_names(scalars, gradients, divergences, curls)
    missing_columns = _missing_particle_columns(data, column_names, mass_from_column.name)
    if !isempty(missing_columns)
        missing_list = join(string.(missing_columns), ", ")
        throw(ArgumentError("Missing columns in ParticleDataFrame: " * missing_list))
    end

    return Partia.KernelInterpolation.build_input(
        Partia.KernelInterpolation.CPUComputeBackend(),
        data[!, :x],
        data[!, :y],
        data[!, :z],
        data[!, :h],
        data[!, :rho],
        data[!, mass_from_column.name];
        column_names = column_names,
        quantity_columns = _materialized_columns(data, column_names),
        scalars = scalars,
        gradients = gradients,
        divergences = divergences,
        curls = curls,
        smoothed_kernel = smoothed_kernel,
    )
end

"""
    build_input(data::ParticleDataFrame,
                mass_from_params::MassFromParams;
                scalars::Tuple{Vararg{Symbol}}=(),
                gradients::Tuple{Vararg{Symbol}}=(),
                divergences::Tuple{Vararg{Symbol}}=(),
                curls::Tuple{Vararg{Symbol}}=(),
                smoothed_kernel::Type{K}=M5_spline) where {K<:AbstractSPHKernel}

Construct a CPU-side `InterpolationInput` and its corresponding
`InterpolationCatalog` from a `ParticleDataFrame`, using a particle mass value
stored in `data.params`.

This method determines the required extra quantity columns from the requested
scalar, gradient, divergence, and curl names, verifies that all required data
columns are present in `data`, checks that the requested mass parameter exists
in `data.params`, constructs a constant mass vector of length `get_npart(data)`,
and forwards the result to the lower-level CPU `build_input` helper.

The base interpolation fields are read from `data[:,:x]`, `data[:,:y]`,
`data[:,:z]`, `data[:,:h]`, and `data[:,:rho]`. Particle mass is taken from
`data.params[mass_from_params.name]` and broadcast to all particles.

# Parameters
- `data::ParticleDataFrame`: Particle dataset containing the base interpolation
  fields, the requested extra quantity columns, and the parameter table
  `data.params`.
- `mass_from_params::MassFromParams`: Mass selector indicating which parameter
  in `data.params` should be used as the particle mass value.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `scalars` | `Tuple{Vararg{Symbol}}` | `()` | Names of extra scalar quantities to interpolate directly. |
| `gradients` | `Tuple{Vararg{Symbol}}` | `()` | Names of extra scalar quantities whose spatial gradients should be computed. |
| `divergences` | `Tuple{Vararg{Symbol}}` | `()` | Base names of extra vector quantities whose divergences should be computed. |
| `curls` | `Tuple{Vararg{Symbol}}` | `()` | Base names of extra vector quantities whose curls should be computed. |
| `smoothed_kernel` | `Type{K}` where `K<:AbstractSPHKernel` | `M5_spline` | SPH kernel type used when constructing the `InterpolationInput`. |

# Returns
- `Tuple{InterpolationInput,InterpolationCatalog{3,N,G,Div,C,L}}`: A pair
  consisting of the materialized CPU interpolation input and the corresponding
  3D interpolation catalog for the requested extra quantities.

"""
function build_input(
    data::ParticleDataFrame,
    mass_from_params::MassFromParams;
    scalars::Tuple{Vararg{Symbol}} = (),
    gradients::Tuple{Vararg{Symbol}} = (),
    divergences::Tuple{Vararg{Symbol}} = (),
    curls::Tuple{Vararg{Symbol}} = (),
    smoothed_kernel::Type{K} = M5_spline,
) where {K<:AbstractSPHKernel}
    column_names = _quantity_column_names(scalars, gradients, divergences, curls)
    missing_columns = _missing_particle_columns(data, column_names)
    if !isempty(missing_columns)
        missing_list = join(string.(missing_columns), ", ")
        throw(ArgumentError("Missing columns in ParticleDataFrame: " * missing_list))
    end
    haskey(data.params, mass_from_params.name) || throw(
        ArgumentError("Missing parameter in ParticleDataFrame.params: $(mass_from_params.name)"),
    )

    N = get_npart(data)
    particle_mass = data.params[mass_from_params.name]

    return Partia.KernelInterpolation.build_input(
        Partia.KernelInterpolation.CPUComputeBackend(),
        data[!, :x],
        data[!, :y],
        data[!, :z],
        data[!, :h],
        data[!, :rho],
        fill(particle_mass, N);
        column_names = column_names,
        quantity_columns = _materialized_columns(data, column_names),
        scalars = scalars,
        gradients = gradients,
        divergences = divergences,
        curls = curls,
        smoothed_kernel = smoothed_kernel,
    )
end

# Toolbox
@inline function _missing_particle_columns(
    data::ParticleDataFrame,
    column_names::Tuple{Vararg{Symbol}},
    mass_column::Union{Nothing,Symbol} = nothing,
)
    available_cols = Set(Symbol.(names(data)))
    missing_columns = Symbol[]

    for base in (:x, :y, :z, :h, :rho)
        if !(base in available_cols) && !(base in missing_columns)
            push!(missing_columns, base)
        end
    end
    if !isnothing(mass_column) && !(mass_column in available_cols) && !(mass_column in missing_columns)
        push!(missing_columns, mass_column)
    end
    for name in column_names
        if !(name in available_cols) && !(name in missing_columns)
            push!(missing_columns, name)
        end
    end

    return missing_columns
end

@inline function _quantity_column_names(
    scalars::Tuple{Vararg{Symbol}},
    gradients::Tuple{Vararg{Symbol}},
    divergences::Tuple{Vararg{Symbol}},
    curls::Tuple{Vararg{Symbol}},
)
    column_names = Symbol[]

    for name in scalars
        name in column_names || push!(column_names, name)
    end
    for name in gradients
        name in column_names || push!(column_names, name)
    end
    for name in divergences
        for comp in Partia.KernelInterpolation._vector_components(name, Val(3))
            comp in column_names || push!(column_names, comp)
        end
    end
    for name in curls
        for comp in Partia.KernelInterpolation._vector_components(name, Val(3))
            comp in column_names || push!(column_names, comp)
        end
    end

    return tuple(column_names...)
end

@inline function _materialized_columns(data::ParticleDataFrame, column_names::Tuple{Vararg{Symbol}})
    return ntuple(i -> data[!, column_names[i]], length(column_names))
end