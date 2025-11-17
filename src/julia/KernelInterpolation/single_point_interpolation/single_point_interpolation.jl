"""
The New single point SPH interpolation
    by Wei-Shan Su,
    October 31, 2025
"""

# Kernel interpolation
## Density
"""
    density(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) -> T

Compute SPH density at a given reference point using the input particle data.

This function computes the SPH density via a summation over all filtered particles in `input`, using the kernel function specified by `smoothed_kernel`. Internally, it dispatches to a low-level `_density_kernel` implementation optimized for performance and static typing.

# Parameters
- `input::InterpolationInput{...}`  
  Preprocessed read-only SPH particle data container. Must be constructed using `build_input(...)`.
- `reference_point::NTuple{3, T}`  
  Cartesian coordinate (x, y, z) of the interpolation location, in the same unit as particle positions.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `ρ_interp::T` — Interpolated density at the reference point, computed by SPH summation.

# Notes
- The kernel is symmetrized using the target smoothing length `ha` and particle-specific `h[i]`.

"""
function density(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _density_kernel(input, reference_point, ha, neighbors, itp_strategy)
end

## Number density
"""
    number_density(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) -> T

Compute SPH number density at a given reference point using particle data.

This function evaluates the particle number density — the kernel-weighted sum of 1 per particle — at a specified spatial location. It calls `_number_density_kernel` internally, which performs a summation over all active particles in the input.

# Parameters
- `input::InterpolationInput{...}`  
  Preprocessed, read-only SPH data container holding fixed-length particle arrays.
- `reference_point::NTuple{3, T}`  
  The Cartesian coordinate (x, y, z) where the number density is evaluated.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `n_interp::T` — Interpolated SPH number density at the reference point.
"""
function number_density(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _number_density_kernel(input, reference_point, ha, neighbors, itp_strategy)
end

## Single quantity intepolation
"""
    quantity_interpolate(input::InterpolationInput{...}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, column_idx::Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) -> T

Interpolate a specific scalar quantity at a reference point using SPH kernel smoothing.

This function computes the SPH-interpolated value of the `column_idx`-th scalar field in the `quant` tuple. It uses symmetric kernel averaging (`W = 0.5(Wa + Wb)`) and Shepard normalization to ensure consistency and stability.

# Parameters
- `input::InterpolationInput{...}`  
  Pre-filled SPH data container holding all necessary physical fields.
- `reference_point::NTuple{3, T}`  
  The spatial location (x, y, z) in Cartesian coordinates where interpolation is evaluated.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `column_idx::Int64`  
  The index of the scalar quantity to interpolate, referring to the `quant` tuple (starting from 1).
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
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
function quantity_interpolate(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, column_idx :: Int64, ShepardNormalization :: Bool = true,itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _quantity_interpolate_kernel(input, reference_point, ha, neighbors, column_idx, ShepardNormalization, itp_strategy)
end

## Muti-columns intepolation
"""
    quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN},
                           reference_point::NTuple{3, T},
                           ha::T,
                           neighbors::NeighborSelection,
                           ShepardNormalization::NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN),
                           itp_strategy::Type{ITPSTRATEGY} = itpSymmetric)

Interpolate multiple scalar fields at a 3D reference point using SPH kernel summation with optional Shepard normalization.  
Each scalar field registered in `input.quant` is interpolated independently, and the results are returned in a fixed ordering consistent with the input catalog.

# Parameters
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  Immutable container holding SPH particle positions, masses, smoothing lengths, and `NCOLUMN` scalar fields to be interpolated.
- `reference_point::NTuple{3, T}`  
  Cartesian coordinates `(x, y, z)` where the interpolation is evaluated.
- `ha::T`  
  Smoothing length associated with the interpolation point.
- `neighbors::NeighborSelection`  
  Structure containing the selected particle indices and their count.
- `ShepardNormalization::NTuple{NCOLUMN, Bool}`  
  Boolean mask specifying which scalar fields should apply Shepard normalization.
- `itp_strategy::Type{ITPSTRATEGY}`  
  Kernel evaluation rule:
  - `itpGather`: use only the target-point smoothing length `hₐ`.
  - `itpScatter`: use only particle smoothing lengths `hᵢ`.
  - `itpSymmetric`: average both contributions, `0.5*(W(hₐ)+W(hᵢ))`.

# Returns
- `NTuple{NCOLUMN, T}`  
  A statically sized tuple containing the interpolated values for all scalar fields listed in `input.quant`, in the same field order.
"""
function quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _quantities_interpolate_kernel(input, reference_point, ha, neighbors, ShepardNormalization , itp_strategy)
end

function quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M),itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy, M}
    return _quantities_interpolate_kernel(input, reference_point, ha, neighbors, columns, ShepardNormalization, itp_strategy)
end

function quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
    if NCOLUMN == 0
        return nothing
    end  
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
    _quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, ShepardNormalization, itp_strategy)
end

function quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy, M}
    @assert length(workspace) == M "Length of `workspace` should match `columns`."
    if M == 0
        return nothing
    end
    _quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, columns, ShepardNormalization, itp_strategy)
    return nothing
end

function quantities_interpolate!(buffer :: NTuple{NCOLUMN, SA}, workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, SA<:AbstractArray{T, 0}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    if NCOLUMN == 0
      return nothing
    end
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
    _quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, ShepardNormalization, itp_strategy)
    @inbounds for i in eachindex(buffer)
        buffer[i][] = workspace[i]
    end
end

function quantities_interpolate!(buffer :: NTuple{M, SA}, workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {M, NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, SA<:AbstractArray{T, 0}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    @assert length(workspace) == M "Length of `workspace` should match `columns`."
    if M == 0
      return nothing
    end
    _quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, columns, ShepardNormalization, itp_strategy)
    @inbounds for i in eachindex(buffer)
      buffer[i][] = workspace[i]
    end
    return nothing
end

## LOS density interpolation (Column / Surface density)
"""
    LOS_density(input::InterpolationInput{...}, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) -> T

Compute the **line-of-sight (LOS) column density** at a 2D position by integrating SPH particle contributions along the z-axis.

This function projects the 3D SPH density field onto the x–y plane, computing the surface density at a given (x, y) coordinate. It internally calls a kernel `_LOS_density_kernel` that performs SPH summation with symmetric kernel averaging.

# Parameters
- `input::InterpolationInput{...}`  
  Immutable, read-only SPH input container containing positions, mass, density, smoothing length, and kernel information.
- `reference_point::NTuple{2, T}`  
  A 2D coordinate (x, y) specifying the location on the projection plane at which the LOS column density is evaluated.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
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
function LOS_density(input::InterpolationInput{T, V, K}, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _LOS_density_kernel(input, reference_point, ha, neighbors, itp_strategy)
end


## LOS quantities interpolation
"""
    LOS_quantities_interpolate(input::InterpolationInput, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) -> Vector{T}

Compute the line-of-sight (LOS) projection of all scalar fields in `input.quant` at a 2D sky-plane location.

This function performs SPH-based interpolation of each scalar field along the line-of-sight direction (usually the z-axis), evaluated at a fixed (x, y) point on the image or projection plane. The result is a vector of LOS-integrated quantities (e.g., column density, projected temperature).

# Parameters
- `input::InterpolationInput{...}`  
  A read-only, isbits SPH input container with all scalar quantities
- `reference_point::NTuple{2, T}`  
  The 2D Cartesian coordinate (x, y) in the projection plane.
- `ha::T`  
  Target smoothing length used at the interpolation point (same type as interpolated fields).
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
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
function LOS_quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
    workspace = zeros(T, NCOLUMN)
    if NCOLUMN == 0
      return workspace
    end
  _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, ShepardNormalization, itp_strategy)
    return workspace
end

function LOS_quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy, M}
    workspace = zeros(T, M)
    if M == 0
        return workspace
    end
  _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, columns, ShepardNormalization, itp_strategy)
    return workspace
end

function LOS_quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
    if NCOLUMN == 0
      return nothing
    end
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
  _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, ShepardNormalization, itp_strategy)
end

function LOS_quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy, M}
    @assert length(workspace) == M "Length of `workspace` should match `columns`."
    if M == 0
        return nothing
    end
  _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, columns, ShepardNormalization, itp_strategy)
  return nothing
end

function LOS_quantities_interpolate!(buffer :: NTuple{NCOLUMN, SA}, workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection,ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, SA<:AbstractArray{T, 0}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    if NCOLUMN == 0
      return nothing
    end
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
  _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, ShepardNormalization, itp_strategy)
    @inbounds for i in eachindex(buffer)
        buffer[i][] = workspace[i]
    end
end

  function LOS_quantities_interpolate!(buffer :: NTuple{M, SA}, workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {M, NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, SA<:AbstractArray{T, 0}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    @assert length(workspace) == M "Length of `workspace` should match `columns`."
    if M == 0
      return nothing
    end
    _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, neighbors, columns, ShepardNormalization, itp_strategy)
    @inbounds for i in eachindex(buffer)
      buffer[i][] = workspace[i]
    end
    return nothing
  end

# Single column gradient density intepolation
"""
    gradient_density(input::InterpolationInput{T, V, K},
                     reference_point::NTuple{3, T},
                     ha::T,
                     neighbors::NeighborSelection,
                     itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric)

Compute the SPH gradient of the density field at a given point using symmetric kernel formulation.

This function wraps `_gradient_density_kernel(...)` and evaluates the gradient of density
at a specified 3D location using the provided neighbor indices and SPH kernel.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  The interpolation data container, including particle properties and kernel type.
- `reference_point::NTuple{3, T}`  
  The physical position where the gradient is evaluated.
- `ha::T`  
  The smoothing length of the reference point.
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
  Kernel interpolation strategy controlling how the smoothing length is applied to W(r,h).  
  - `itpGather`: Use only `h_a`, the smoothing length centered at the target point. (Hernquist & Katz (1989), Price (2012))
  - `itpScatter`: Use only `h_b`, the smoothing length from each source particle. (Price (2007, SPLASH), Monaghan (1992))
  - `itpSymmetric`: Use averaged kernel value `0.5*(W(h_a) + W(h_b))`. (Monaghan (1992))

# Returns
- `::NTuple{3, T}`  
  The gradient of the density field, ∇ρ, at the reference point.  
  If `neighbors` is empty or ρ is zero, returns `(NaN, NaN, NaN)`.
"""
function gradient_density(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _gradient_density_kernel(input, reference_point, ha, neighbors, itp_strategy)
end

# Single column gradient value intepolation
"""
    gradient_quantity_interpolate(input::InterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, column_idx::Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) -> NTuple{3, T}

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
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `column_idx::Int64`  
  Index of the target scalar field in the `input.quant` tuple.
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
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
function gradient_quantity_interpolate(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, column_idx :: Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _gradient_quantity_interpolate_kernel(input, reference_point, ha, neighbors, column_idx, itp_strategy)
end

# Single column divergence value intepolation
"""
    divergence_quantity_interpolate(input::InterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbors::NeighborSelection, Ax_column_idx::Int64, Ay_column_idx::Int64, Az_column_idx::Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) -> T

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
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `Ax_column_idx::Int64`, `Ay_column_idx::Int64`, `Az_column_idx::Int64`  
  Indices of the scalar fields representing the x, y, and z components of the vector field \vec{A}.
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
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
function divergence_quantity_interpolate(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _divergence_quantity_interpolate_kernel(input, reference_point, ha, neighbors, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy)
end

# Single column curl value intepolation
"""
    curl_quantity_interpolate(input::InterpolationInput, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, Ax_column_idx::Int64, Ay_column_idx::Int64, Az_column_idx::Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) -> NTuple{3, T}

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
- `neighbors::NeighborSelection`  
  Neighbor selection holding the particle indices and count.
- `Ax_column_idx`, `Ay_column_idx`, `Az_column_idx`  
  Column indices of the three components A_x, A_y, A_z of the vector field.
- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`: 
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
function curl_quantity_interpolate(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _curl_quantity_interpolate_kernel(input, reference_point, ha, neighbors, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy)
end
