"""
The New single point SPH interpolation
    by Wei-Shan Su,
    June 28, 2025

"""

# Input and output structure
abstract type AbstractInterpolationInput end

"""
    InterpolationInput{NCOLUMN, T, V, K}

Immutable SPH input container for read-only interpolation queries.

This struct is designed for fast and safe interpolation on CPU and GPU. It contains all the necessary SPH quantities. The `quant` field holds `NCOLUMN` scalar fields (e.g., pressure, temperature) in a tuple.

**This struct is fully isbits and read-only** — all internal data should be pre-filled and never mutated. Intended for bulk interpolation kernels.

# Type Parameters
- `NCOLUMN`: Number of scalar fields in `quant`.
- `T`: Floating-point type (e.g., `Float32` or `Float64`).
- `V`: An `AbstractVector{T}`, the vector type used throughout.
- `K`: Type of SPH kernel used, must be a concrete `AbstractSPHKernel`.

# Fields
- `Npart::Int64` — Number of active (valid) particles within the batch.
- `ha::T` — Target smoothing length for the reference point.
- `smoothed_kernel::Type{K}` — SPH kernel function instance.
- `x, y, z::V` — Particle positions in Cartesian coordinates.
- `m::V` — Particle masses.
- `h::V` — Particle smoothing lengths.
- `ρ::V` — Particle densities.
- `quant::NTuple{NCOLUMN, V}` — Tuple of per-field scalar data arrays (e.g., pressure, temperature).
"""
struct InterpolationInput{NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel} <: AbstractInterpolationInput
    Npart       :: Int64
    smoothed_kernel :: Type{K}
    x               :: V
    y               :: V
    z               :: V
    m               :: V
    h               :: V
    ρ               :: V
    quant           :: NTuple{NCOLUMN, V}
end

# Some useful function 
## Get "Valid" range of data (the other would be 0)
Base.length(inp::Input) where {Input<:AbstractInterpolationInput} = inp.Npart   # N

# Check the "Valid" length of data for each fields
function Base.checkbounds(inp::Input) where {Input<:AbstractInterpolationInput}
    N = inp.Npart
    @assert N isa Integer && N ≥ 0 "Invalid Npart: $N"
    for name in fieldnames(typeof(inp))
        name === :quant && continue  
        val = getfield(inp, name)
        if val isa AbstractVector
            @assert N ≤ length(val) "$name is shorter than Npart ($N)"
        end
    end

    # Check all fields inside NamedTuple
    for (k, v) in inp.quant
        @assert N ≤ length(v) "quant[$k] is shorter than Npart ($N)"
    end
    return true
end

"""
    get_type(input::InterpolationInput) -> Type{T}

Return the floating-point type `T` used in the `InterpolationInput`.

This is the type used for all scalar fields and positions in the interpolation system.
"""
@inline function get_type(input::InterpolationInput{NCOLUMN, T, V, K}) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return T
end

# Generate IntepolationInput
"""
    InterpolationInput(data::PhantomRevealerDataFrame, column_names::Vector{Symbol}, smoothed_kernel::K; Identical_particles = true)

Construct a SPH interpolation container from `PhantomRevealerDataFrame`.

This function extracts particle information from `data`, promotes types to a unified floating-point type, and stores them in the structure. The resulting structure is designed for efficient GPU access and supports SPH kernel-based interpolation.

# Parameters
- `data::PhantomRevealerDataFrame`  
  The particle dataset to extract information from.
- `column_names::Vector{Symbol}`  
  List of scalar physical quantities (e.g., `[:P, :T]`) to interpolate, stored in `quant`.
- `smoothed_kernel::K`  
  The smoothing kernel type to use (e.g., `M5_spline`), must subtype `AbstractSPHKernel`.

# Keyword Arguments
| Name                | Default     | Description                                                        |
|---------------------|-------------|--------------------------------------------------------------------|
| `Identical_particles` | `true`    | If `true`, all particles are assumed to have identical mass. Uses `data.params["mass"]` for all entries. If `false`, reads mass column from `data.dfdata`.

# Returns
- `::InterpolationInput{NCOLUMN, T, Vector{T}, K}`  
  A statically sized struct containing interpolable particle fields: position, smoothing length, mass, density, and scalar fields in `quant`.

# Notes
- All values are promoted to a consistent floating-point type `T`.
"""
function InterpolationInput(data::PhantomRevealerDataFrame, column_names::Vector{Symbol}, smoothed_kernel::Type{K}; Identical_particles=true) where {K<:AbstractSPHKernel}
    N = get_npart(data)
    # Promote all to unified type
    Tprom = promote_type(
        eltype(data[!, :x]), eltype(data[!, :y]), eltype(data[!, :z]),
        eltype(data[!, :h]), eltype(data[!, :rho]),
    )


    x = data[!, :x]
    y = data[!, :y]
    z = data[!, :z]
    h = data[!, :h]
    ρ = data[!, :rho]
    m = nothing
    if Identical_particles
      particle_mass = data.params["mass"]
      m = fill(particle_mass, N)
    else
      m = data[!, :m]
    end

    # Build quant 
    NCOLUMN = length(column_names)
    quant = ntuple(i -> Vector{Tprom}(data[!, column_names[i]]), NCOLUMN)
    
    return InterpolationInput{NCOLUMN, Tprom, Vector{Tprom}, K}(
        N, smoothed_kernel, x, y, z, m, h, ρ, quant
    )
end

# Determine interpolation type
@enum InterpolationStrategy begin
    itpGather
    itpScatter
    itpSymmetric
end


# Kernel interpolation
## Density
@inline function _density_kernel(input::AbstractInterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T}
    # Return 0.0 if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        return 0.0
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    Ktyp = input.smoothed_kernel

    # Initialize counter
    rho :: T = zero(T)

    @inbounds for i in neighbor_indices
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end
        rho += ms[i] * W
    end
    return rho
end

"""
    density(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) -> T

Compute SPH density at a given reference point using the input particle data.

This function computes the SPH density via a summation over all filtered particles in `input`, using the kernel function specified by `smoothed_kernel`. Internally, it dispatches to a low-level `_density_kernel` implementation optimized for performance and static typing.

# Parameters
- `input::InterpolationInput{...}`  
  Preprocessed read-only SPH particle data container. Must be constructed using `InterpolationInput(...)`.
- `reference_point::NTuple{3, T}`  
  Cartesian coordinate (x, y, z) of the interpolation location, in the same unit as particle positions.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `ρ_interp::T` — Interpolated density at the reference point, computed by SPH summation.

# Notes
- The kernel is symmetrized using the target smoothing length `ha` and particle-specific `h[i]`.

"""
function density(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T,neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return _density_kernel(input, reference_point, ha, neighbor_indices, itp_strategy)
end

## Number density
@inline function _number_density_kernel(input::AbstractInterpolationInput, reference_point::NTuple{3, T}, ha :: T,neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T}
    # Return 0.0 if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        return 0.0
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    hs = input.h
    Ktyp = input.smoothed_kernel

    # Initialize counter
    n :: T = zero(T)

    @inbounds for i in neighbor_indices
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end
        n += W
    end
    return n
end

"""
    number_density(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) -> T

Compute SPH number density at a given reference point using particle data.

This function evaluates the particle number density — the kernel-weighted sum of 1 per particle — at a specified spatial location. It calls `_number_density_kernel` internally, which performs a summation over all active particles in the input.

# Parameters
- `input::InterpolationInput{...}`  
  Preprocessed, read-only SPH data container holding fixed-length particle arrays.
- `reference_point::NTuple{3, T}`  
  The Cartesian coordinate (x, y, z) where the number density is evaluated.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `n_interp::T` — Interpolated SPH number density at the reference point.
"""
function number_density(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return _number_density_kernel(input, reference_point, ha, neighbor_indices, itp_strategy)
end

## Single quantity intepolation
@inline function _quantity_intepolate_kernel(input::AbstractInterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T}
    # Return 0.0 if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        return 0.0
    end
    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    As = input.quant[column_idx]
    Ktyp = input.smoothed_kernel

    # Initialize counter
    A :: T = zero(T)
    mWlρ :: T = zero(T)

    @inbounds for i in neighbor_indices
        mb = ms[i]
        ρb = ρs[i]
        Ab = As[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        mbWlρb = mb * W/ρb
        mWlρ += mbWlρb
        A += Ab * mbWlρb
    end
    # Shepard normalization
    A /= mWlρ
    return A
end

"""
    quantity_intepolate(input::InterpolationInput{...}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, column_idx::Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) -> T

Interpolate a specific scalar quantity at a reference point using SPH kernel smoothing.

This function computes the SPH-interpolated value of the `column_idx`-th scalar field in the `quant` tuple. It uses symmetric kernel averaging (`W = 0.5(Wa + Wb)`) and Shepard normalization to ensure consistency and stability.

# Parameters
- `input::InterpolationInput{...}`  
  Pre-filled SPH data container holding all necessary physical fields.
- `reference_point::NTuple{3, T}`  
  The spatial location (x, y, z) in Cartesian coordinates where interpolation is evaluated.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `column_idx::Int64`  
  The index of the scalar quantity to interpolate, referring to the `quant` tuple (starting from 1).
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `A_interp::T` — Interpolated scalar value at the specified location.

# Notes
- Interpolation uses symmetric SPH kernels with per-particle and target smoothing lengths.
- `column_idx` must be within the bounds `1:NCOLUMN`, where `NCOLUMN` is the number of fields stored.
"""
function quantity_intepolate(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return _quantity_intepolate_kernel(input, reference_point, ha, neighbor_indices, column_idx, itp_strategy)
end

## Muti-columns intepolation
@inline function _quantities_interpolate_kernel!(output :: O, input::AbstractInterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T, O<:AbstractVector{T}}
    # Return 0.0 if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        fill!(output, T(NaN))
        return
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    vals = input.quant    
    Ktyp = input.smoothed_kernel
    
    # Whether we need to check the length of output and vals? (AbstractVector{T}) <-> (NamedTuple{FN, NTuple{NCOLUMN, V}})
    @assert length(output) == length(vals)

    # Initialize counter
    mWlρ :: T = zero(T)
    fill!(output, 0.0)

    @inbounds for i in neighbor_indices
        mb = ms[i]
        ρb = ρs[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        # Counting
        mbWlρb = mb * W/ρb
        mWlρ += mbWlρb           # Prepare for Shapard Normalization
        
        @inbounds for j in eachindex(output)
            output[j] += mbWlρb * vals[j][i]
        end             
    end
    # Shapard Normalization
    @inbounds for j in eachindex(output)
        output[j] /= mWlρ
    end
end


"""
    quantities_interpolate(input::InterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) -> Vector{T}

Interpolate all scalar fields in `input.quant` at a 3D reference point using SPH with Shepard normalization.

This function returns a vector of interpolated values corresponding to each physical quantity in `input.quant`. The interpolation is symmetric, using both the reference and particle smoothing lengths.

# Parameters
- `input::InterpolationInput{...}`  
  A read-only container with SPH particle data and scalar fields
- `reference_point::NTuple{3, T}`  
  3D Cartesian coordinate (x, y, z) where interpolation is evaluated.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `::Vector{T}`  
  A vector of length `NCOLUMN`, each entry corresponding to the interpolated value of a scalar field (e.g., pressure, temperature).

# Notes
- Uses symmetric kernel averaging between `ha` and `hᵢ`.
- Shepard normalization is applied to ensure consistency.
- Output order matches the order of scalar fields in `input.quant`.
"""
function quantities_interpolate(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    workspace = zeros(T, NCOLUMN)
    if NCOLUMN == 0
      return workspace
    end
    _quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbor_indices, itp_strategy)
    return workspace
end

function quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    if NCOLUMN == 0
      return nothing
    end  
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
      _quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbor_indices, itp_strategy)
end

function quantities_interpolate!(buffer :: NTuple{NCOLUMN, SA}, workspace :: Vector{T}, input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, SA<:AbstractArray{T, 0}}
    if NCOLUMN == 0
      return nothing
    end
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
    _quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbor_indices, itp_strategy)
    @inbounds for i in eachindex(buffer)
        buffer[i][] = workspace[i]
    end
end

## LOS density interpolation (Column / Surface density)
@inline function _LOS_density_kernel(input::AbstractInterpolationInput, reference_point::NTuple{2, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T}
    # Return 0.0 if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        return zero(T)
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    ms = input.m
    hs = input.h
    Ktyp = input.smoothed_kernel

    # Initialize counter
    Sigma :: T = zero(T)

    @inbounds for i in neighbor_indices
        mb = ms[i]
        rb :: NTuple{2, T} = (xs[i], ys[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
          W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        # Counting
        Sigma += mb * W     
    end
    return Sigma
end

"""
    LOS_density(input::InterpolationInput{...}, reference_point::NTuple{2, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) -> T

Compute the **line-of-sight (LOS) column density** at a 2D position by integrating SPH particle contributions along the z-axis.

This function projects the 3D SPH density field onto the x–y plane, computing the surface density at a given (x, y) coordinate. It internally calls a kernel `_LOS_density_kernel` that performs SPH summation with symmetric kernel averaging.

# Parameters
- `input::InterpolationInput{...}`  
  Immutable, read-only SPH input container containing positions, mass, density, smoothing length, and kernel information.
- `reference_point::NTuple{2, T}`  
  A 2D coordinate (x, y) specifying the location on the projection plane at which the LOS column density is evaluated.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `Σ::T`  
  Total column density at the given location, i.e., integrated \rho(z)  dz estimated via SPH.

# Notes
- Assumes the LOS is aligned with the z-axis.
- Requires full particle positions and densities to be preloaded.
- Output is a scalar in units of mass per area.
"""
function LOS_density(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{2, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return _LOS_density_kernel(input, reference_point, ha, neighbor_indices, itp_strategy)
end


## LOS quantities interpolation
@inline function _LOS_quantities_interpolate_kernel!(output :: O, input::AbstractInterpolationInput, reference_point::NTuple{2, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T, O<:AbstractVector{T}}
    # Return 0.0 if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        fill!(output, T(NaN))
        return
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    ms = input.m
    hs = input.h
    ρs = input.ρ
    vals = input.quant    
    Ktyp = input.smoothed_kernel
    
    # Whether we need to check the length of output and vals? (AbstractVector{T}) <-> (NamedTuple{FN, NTuple{NCOLUMN, V}})
    @assert length(output) == length(vals)

    # Initialize counter
    mWlρ :: T = zero(T)
    fill!(output, 0.0)

    @inbounds for i in neighbor_indices
        mb = ms[i]
        ρb = ρs[i]
        rb :: NTuple{2, T} = (xs[i], ys[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
          W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        # Counting
        mbWlρb = mb * W/ρb
        mWlρ += mbWlρb           # Prepare for Shapard Normalization
        
        @inbounds for j in eachindex(output)
            output[j] += mbWlρb * vals[j][i]
        end             
    end
    # Shapard Normalization
    @inbounds for j in eachindex(output)
        output[j] /= mWlρ
    end
end

"""
    LOS_quantities_interpolate(input::InterpolationInput, reference_point::NTuple{2, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) -> Vector{T}

Compute the line-of-sight (LOS) projection of all scalar fields in `input.quant` at a 2D sky-plane location.

This function performs SPH-based interpolation of each scalar field along the line-of-sight direction (usually the z-axis), evaluated at a fixed (x, y) point on the image or projection plane. The result is a vector of LOS-integrated quantities (e.g., column density, projected temperature).

# Parameters
- `input::InterpolationInput{...}`  
  A read-only, isbits SPH input container with all scalar quantities
- `reference_point::NTuple{2, T}`  
  The 2D Cartesian coordinate (x, y) in the projection plane.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `::Vector{T}`  
  A vector of length `NCOLUMN`, each entry corresponding to the LOS-projected value of a scalar field in `input.quant`.

# Notes
- The kernel smoothing is performed in 3D but the integration is projected onto the 2D plane.
- This function is intended for generating projected maps (e.g., surface density, emission measure).
- Output order matches the ordering of scalar fields in `input.quant`.
"""
function LOS_quantities_interpolate(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{2, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    workspace = zeros(T, NCOLUMN)
    if NCOLUMN == 0
      return workspace
    end
    _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbor_indices, itp_strategy)
    return workspace
end

function LOS_quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{2, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where { NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    if NCOLUMN == 0
      return nothing
    end
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
    _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbor_indices, itp_strategy)
end

function LOS_quantities_interpolate!(buffer :: NTuple{NCOLUMN, SA}, workspace :: Vector{T}, input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{2, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, SA<:AbstractArray{T, 0}}
    if NCOLUMN == 0
      return nothing
    end
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
    _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbor_indices, itp_strategy)
    @inbounds for i in eachindex(buffer)
        buffer[i][] = workspace[i]
    end
end

"""
∇ρ(r) = (1/ρ(r))∑_b m_b*(ρ_b-ρ(r))∇W(r-r_b)
      = (1/ρ(r))((∑_b m_b*ρ_b*∇W(r-r_b))  - ρ(r)(∑_b m_b*∇W(r-r_b))
      = (1/ρ(r))((∑_b m_b*ρ_b*∇W(r-r_b)) - ∑_b m_b*∇W(r-r_b)
"""
# Single column gradient density intepolation
@inline function _gradient_density_kernel(input::AbstractInterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T}
  # Return (NaN) if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        return (T(NaN), T(NaN), T(NaN))
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    Ktyp = input.smoothed_kernel

    # Initialize counter
    ∇ρxf :: T = zero(T)
    ∇ρyf :: T = zero(T)
    ∇ρzf :: T = zero(T)
    ∇ρxb :: T = zero(T)
    ∇ρyb :: T = zero(T)
    ∇ρzb :: T = zero(T)

    ρ :: T = zero(T)

    @inbounds for i in neighbor_indices
      mb = ms[i]
      ρb = ρs[i]
      rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
      W :: T = zero(T)
        if itp_strategy == itpGather
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end
      ∂xW :: T = zero(T)
      ∂yW :: T = zero(T)
      ∂zW :: T = zero(T)
      if itp_strategy == itpGather
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
      elseif itp_strategy == itpScatter
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
      elseif itp_strategy == itpSymmetric
        ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
        ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
        ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
        ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
      end

      # Counting
      ## Normal
      mbW = mb * W
      ρ += mbW                                # ρ(r)

      # Gradient
      mb∂xW = mb * ∂xW
      mb∂yW = mb * ∂yW
      mb∂zW = mb * ∂zW

      ∇ρxf += mb∂xW * ρb
      ∇ρyf += mb∂yW * ρb
      ∇ρzf += mb∂zW * ρb
      ∇ρxb += mb∂xW
      ∇ρyb += mb∂yW
      ∇ρzb += mb∂zW
    end
    if iszero(ρ)
        return Tuple{T, T, T}(T(NaN), T(NaN), T(NaN))
    end

    # Construct gradient
    ∇ρxf /= ρ
    ∇ρyf /= ρ
    ∇ρzf /= ρ

    # Final result
    ∇ρx = (∇ρxf - ∇ρxb)
    ∇ρy = (∇ρyf - ∇ρyb)
    ∇ρz = (∇ρzf - ∇ρzb)
    return (∇ρx, ∇ρy, ∇ρz)
end

"""
    gradient_density(input::InterpolationInput{NCOLUMN, T, V, K},
                     reference_point::NTuple{3, T},
                     ha::T,
                     neighbor_indices::AbstractVector{<:Integer},
                     itp_strategy :: InterpolationStrategy = itpSymmetric)

Compute the SPH gradient of the density field at a given point using symmetric kernel formulation.

This function wraps `_gradient_density_kernel(...)` and evaluates the gradient of density
at a specified 3D location using the provided neighbor indices and SPH kernel.

# Parameters
- `input::InterpolationInput{NCOLUMN, T, V, K}`  
  The interpolation data container, including particle properties and kernel type.
- `reference_point::NTuple{3, T}`  
  The physical position where the gradient is evaluated.
- `ha::T`  
  The smoothing length of the reference point.
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighbor particles found via KDTree or similar neighbor search.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `::NTuple{3, T}`  
  The gradient of the density field, ∇ρ, at the reference point.  
  If `neighbor_indices` is empty or ρ is zero, returns `(NaN, NaN, NaN)`.
"""
function gradient_density(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return _gradient_density_kernel(input, reference_point, ha, neighbor_indices, itp_strategy)
end

"""
∇A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))∇W(r-r_b)
      = (1/ρ(r))((∑_b m_b*A_b*∇W(r-r_b))  - A(r)(∑_b m_b*∇W(r-r_b))
      = ∇Af - ∇Ab
"""
# Single column gradient value intepolation
@inline function _gradient_quantity_intepolate_kernel(input::AbstractInterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T}
    # Return (NaN) if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        return (T(NaN), T(NaN), T(NaN))
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    As = input.quant[column_idx]
    Ktyp = input.smoothed_kernel

    # Initialize counter

    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)
    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)

    mWlρ :: T = zero(T)
    A :: T = zero(T)
    ρ :: T = zero(T)

    @inbounds for i in neighbor_indices
        mb = ms[i]
        ρb = ρs[i]
        Ab = As[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end
        ∂xW :: T = zero(T)
        ∂yW :: T = zero(T)
        ∂zW :: T = zero(T)
        if itp_strategy == itpGather
          ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
          ∂xW = ∇W[1]
          ∂yW = ∇W[2]
          ∂zW = ∇W[3]
        elseif itp_strategy == itpScatter
          ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
          ∂xW = ∇W[1]
          ∂yW = ∇W[2]
          ∂zW = ∇W[3]
        elseif itp_strategy == itpSymmetric
          ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
          ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
          ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
          ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
          ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
        end

        # Counting
        ## Normal
        mbW = mb * W
        mWlρ += mbW/ρb                          # Shepard normalization for A(r)
        ρ += mbW                                # ρ(r)
        A += (mbW * Ab)/ρb                      # A(r)

        # Gradient
        mb∂xW = mb * ∂xW
        mb∂yW = mb * ∂yW
        mb∂zW = mb * ∂zW

        ∇Axf += mb∂xW * Ab
        ∇Ayf += mb∂yW * Ab
        ∇Azf += mb∂zW * Ab
        ∇Axb += mb∂xW
        ∇Ayb += mb∂yW
        ∇Azb += mb∂zW
    end
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    A /= mWlρ

    # Construct gradient
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A

    # Final result
    ∇Ax = (∇Axf - ∇Axb)/ρ
    ∇Ay = (∇Ayf - ∇Ayb)/ρ
    ∇Az = (∇Azf - ∇Azb)/ρ
    return (∇Ax, ∇Ay, ∇Az)
end

"""
    gradient_quantity_intepolate(input::InterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, column_idx::Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) -> NTuple{3, T}

Estimate the gradient ∇A of a single scalar quantity at a 3D reference point via SPH interpolation.

The gradient value of any arbitrary scalar quantity A would be estimated by

∇A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))∇W(r-r_b)

This function returns the spatial gradient of a given scalar field in `input.quant[column_idx]`, computed using a symmetrized SPH formulation with Shepard normalization. The kernel gradient is averaged over both the target smoothing length `ha` and each particle's `h`.

# Parameters
- `input::InterpolationInput{...}`  
  Immutable, read-only container for SPH particle data.
- `reference_point::NTuple{3, T}`  
  The 3D Cartesian coordinates at which to evaluate the gradient.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `column_idx::Int64`  
  Index of the target scalar field in the `input.quant` tuple.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `::NTuple{3, T}`  
  The interpolated gradient vector (∂x A, ∂y A, ∂z A). If no valid particles are present, returns `(NaN, NaN, NaN)`.

# Notes
- Uses symmetrized kernel and its gradient:  
   W = 0.5 (W_{ah} + W_{bh}), ∇W = 0.5 (∇W_{ah} + ∇W_{bh}) 
"""
function gradient_quantity_intepolate(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return _gradient_quantity_intepolate_kernel(input, reference_point, ha, neighbor_indices, column_idx, itp_strategy)
end


# Single column divergence value intepolation
"""
    ∇⋅A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))⋅∇W(r-r_b)
           = (1/ρ(r)) * ((∑_b m_b*A_b⋅∇W(r-r_b)))- A(r)⋅(∑_b m_b*∇W(r-r_b)))
           = ∇⋅A(r)
"""
@inline function _divergence_quantity_intepolate_kernel(input::AbstractInterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T}
    # Return 0.0 if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        return T(NaN)
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    Axs = input.quant[Ax_column_idx]
    Ays = input.quant[Ay_column_idx]
    Azs = input.quant[Az_column_idx]
    Ktyp = input.smoothed_kernel

    # Initialize counter
    ∇Af :: T = zero(T)
    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)

    mWlρ :: T = zero(T)
    Ax :: T = zero(T)
    Ay :: T = zero(T)
    Az :: T = zero(T)
    ρ :: T = zero(T)
    @inbounds for i in neighbor_indices
        mb = ms[i]
        ρb = ρs[i]
        Axb = Axs[i]
        Ayb = Ays[i]
        Azb = Azs[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        ∂xW :: T = zero(T)
        ∂yW :: T = zero(T)
        ∂zW :: T = zero(T)
        if itp_strategy == itpGather
          ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
          ∂xW = ∇W[1]
          ∂yW = ∇W[2]
          ∂zW = ∇W[3]
        elseif itp_strategy == itpScatter
          ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
          ∂xW = ∇W[1]
          ∂yW = ∇W[2]
          ∂zW = ∇W[3]
        elseif itp_strategy == itpSymmetric
          ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
          ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
          ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
          ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
          ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
        end

        # Counting
        ## Normal
        mbW = mb * W
        mWlρ += mbW/ρb                          # Shepard normalization for A(r)
        ρ += mbW                                # ρ(r)
        Ax += (mbW * Axb)/ρb                    # Ax(r)
        Ay += (mbW * Ayb)/ρb                    # Ay(r)
        Az += (mbW * Azb)/ρb                    # Az(r)

        # Gradient
        mb∂xW = mb * ∂xW
        mb∂yW = mb * ∂yW
        mb∂zW = mb * ∂zW

        ∇Af += mb∂xW * Axb + mb∂yW * Ayb + mb∂zW * Azb
        ∇Axb += mb∂xW
        ∇Ayb += mb∂yW
        ∇Azb += mb∂zW
    end
    if iszero(ρ)
        return Tuple{T, T, T}(T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    Ax /= mWlρ
    Ay /= mWlρ
    Az /= mWlρ

    # Construct gradient
    ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb

    # Final result
    ∇A = (∇Af - ∇Ab)/ρ

    return ∇A
end

"""
    divergence_quantity_intepolate(input::InterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, Ax_column_idx::Int64, Ay_column_idx::Int64, Az_column_idx::Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) -> T

Estimate the divergence ∇·A of a vector quantity at a 3D reference point via SPH interpolation.

The divergence of any arbitrary vector quantity A would be estimated by

∇⋅A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))⋅∇W(r-r_b)

This function computes the divergence at a given position from three scalar fields stored in `input.quant`, identified by their corresponding indices. It symmetrizes the kernel gradient using both the target smoothing length and each particle’s individual smoothing length.

# Parameters
- `input::InterpolationInput{...}`  
  Read-only, GPU-friendly container for SPH particle data.
- `reference_point::NTuple{3, T}`  
  The 3D Cartesian coordinates at which the divergence is computed.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `Ax_column_idx::Int64`, `Ay_column_idx::Int64`, `Az_column_idx::Int64`  
  Indices of the scalar fields representing the x, y, and z components of the vector field \vec{A}.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `::T`  
  The interpolated divergence ∇·A at the reference point. Returns `NaN` if no valid particles are present or if the local density is zero.

# Notes
- Uses symmetrized kernel value and gradient:  

  W = 0.5 (W_{ah} + W_{bh}), ∇W = 0.5 (∇W_{ah} + ∇W_{bh})

- Includes normalization via the estimated ρ(r) and subtraction of \vec{A}(r) to ensure conservative and stable estimates.
"""
function divergence_quantity_intepolate(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return _divergence_quantity_intepolate_kernel(input, reference_point, ha, neighbor_indices, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy)
end

"""
∇×A(r) = -(1/ρ(r))∑_b m_b*(A_b-A(r))×∇W(r-r_b)
       = -(1/ρ(r)) * ((∑_b m_b*A_b×∇W(r-r_b)) - A(r)×(∑_b m_b*∇W(r-r_b)))
       = -(1/ρ(r))*(∇×Af - ∇×Ab)
"""
@inline function _curl_quantity_intepolate_kernel(input::AbstractInterpolationInput, reference_point::NTuple{3, T}, ha :: T,neighbor_indices :: AbstractVector{<:Integer}, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {T}
    # Return 0.0 if no particle in the data
    Npart :: Int64 = length(neighbor_indices)
    if Npart == 0
        return (T(NaN), T(NaN), T(NaN))
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    Axs = input.quant[Ax_column_idx]
    Ays = input.quant[Ay_column_idx]
    Azs = input.quant[Az_column_idx]
    Ktyp = input.smoothed_kernel

    # Initialize counter
    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)

    m∂xW :: T = zero(T)
    m∂yW :: T = zero(T)
    m∂zW :: T = zero(T)

    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)

    mWlρ :: T = zero(T)
    Ax :: T = zero(T)
    Ay :: T = zero(T)
    Az :: T = zero(T)
    ρ :: T = zero(T)
    @inbounds for i in neighbor_indices
        mb = ms[i]
        ρb = ρs[i]
        Axb = Axs[i]
        Ayb = Ays[i]
        Azb = Azs[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])

        W :: T = zero(T)
        if itp_strategy == itpGather
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
          W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
          W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end
        
        ∂xW :: T = zero(T)
        ∂yW :: T = zero(T)
        ∂zW :: T = zero(T)
        if itp_strategy == itpGather
          ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
          ∂xW = ∇W[1]
          ∂yW = ∇W[2]
          ∂zW = ∇W[3]
        elseif itp_strategy == itpScatter
          ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
          ∂xW = ∇W[1]
          ∂yW = ∇W[2]
          ∂zW = ∇W[3]
        elseif itp_strategy == itpSymmetric
          ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
          ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
          ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
          ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
          ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
        end

        # Counting
        ## Normal
        mbW = mb * W
        mWlρ += mbW/ρb                          # Shepard normalization for A(r)
        ρ += mbW                                # ρ(r)
        Ax += (mbW * Axb)/ρb                    # Ax(r)
        Ay += (mbW * Ayb)/ρb                    # Ay(r)
        Az += (mbW * Azb)/ρb                    # Az(r)

        # Gradient
        mb∂xW = mb * ∂xW
        mb∂yW = mb * ∂yW
        mb∂zW = mb * ∂zW

        ∇Axf += Ayb * mb∂zW -  Azb * mb∂yW
        ∇Ayf += Azb * mb∂xW -  Axb * mb∂zW
        ∇Azf += Axb * mb∂yW -  Ayb * mb∂xW
        m∂xW += mb∂xW
        m∂yW += mb∂yW
        m∂zW += mb∂zW
    end
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    Ax /= mWlρ
    Ay /= mWlρ
    Az /= mWlρ

    # Construct gradient
    ∇Axb = Ay * m∂zW - Az * m∂yW
    ∇Ayb = Az * m∂xW - Ax * m∂zW
    ∇Azb = Ax * m∂yW - Ay * m∂xW

    # Final result
    ∇Ax = -(∇Axf - ∇Axb)/ρ
    ∇Ay = -(∇Ayf - ∇Ayb)/ρ
    ∇Az = -(∇Azf - ∇Azb)/ρ

    return (∇Ax, ∇Ay, ∇Az)
end

"""
    curl_quantity_intepolate(input::InterpolationInput, reference_point::NTuple{3, T}, ha :: T,neighbor_indices :: AbstractVector{<:Integer}, Ax_column_idx::Int64, Ay_column_idx::Int64, Az_column_idx::Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) -> NTuple{3, T}

Estimate the curl ∇×A of a vector field at a 3D reference point via symmetrized SPH interpolation.

This function computes the curl of a 3-component vector field stored in `input.quant`, specified by the three column indices. The formulation follows Price (2012), Eq. (79), using a symmetrized kernel and gradient with Shepard normalization. The expression is:

∇×A(r) = - (1 / ρ(r)) * [ ∑₍b₎ m_b * A_b × ∇W(r - r_b) - A(r) × ∑₍b₎ m_b * ∇W(r - r_b) ]

# Parameters
- `input::InterpolationInput{...}`  
  Immutable, read-only container for SPH particle data.
- `reference_point::NTuple{3, T}`  
  The Cartesian position at which to compute the curl.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbor_indices::AbstractVector{<:Integer}`  
  Indices of neighboring particles satisfying the kernel support condition; typically obtained via KDTree or cell-linked list.
- `Ax_column_idx`, `Ay_column_idx`, `Az_column_idx`  
  Column indices of the three components A_x, A_y, A_z of the vector field.
- `itp_strategy::InterpolationStrategy=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `::NTuple{3, T}`  
  The curl vector (∂yAz - ∂zAy, ∂zAx - ∂xAz, ∂xAy - ∂yAx) evaluated at the given point. Returns `(NaN, NaN, NaN)` if no valid particles exist.

# Notes
- Uses symmetrized SPH kernel average:  
  W = 0.5(W_a + W_b), ∇W = 0.5(∇W_a + ∇W_b)
- The result is scaled by `1/ρ(r)` to maintain consistency with SPH conventions.
- The negative sign follows the antisymmetric form in Price (2012).
"""
function curl_quantity_intepolate(input::InterpolationInput{NCOLUMN, T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbor_indices :: AbstractVector{<:Integer}, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}
    return _curl_quantity_intepolate_kernel(input, reference_point,ha , neighbor_indices, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy)
end
