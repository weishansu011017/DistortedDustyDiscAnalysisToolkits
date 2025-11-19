"""
The New single point SPH interpolation
    by Wei-Shan Su,
    October 31, 2025
"""

# Kernel interpolation
## Density
"""
    density(input::InterpolationInput{T, V, K},
            reference_point::NTuple{3, T},
            ha::T,
            LBVH::LinearBVH,
            itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> T

Compute SPH density at a given reference point using a Linear Bounding Volume
Hierarchy (LBVH) for neighbour search.

This variant of `density` performs neighbour discovery via the supplied `LBVH`,
querying particles whose AABBs intersect the smoothing region centred at
`reference_point` with smoothing length `ha`. The resulting particle subset is
passed into a low-level `_density_kernel` implementation optimised for static
types and high-performance SPH interpolation.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  Preprocessed, read-only SPH particle data container constructed using
  `build_input(...)`.
- `reference_point::NTuple{3, T}`  
  Spatial position (x, y, z) where the density is evaluated.
- `ha::T`  
  Target smoothing length at the interpolation point.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy built over particle positions. Used to
  identify candidate neighbours whose bounding boxes intersect the kernel
  support radius.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy determining how `h_a` and `h_b` contribute to
  the smoothing kernel:  
  - `itpGather`: use only the target-point smoothing length `h_a`.  
  - `itpScatter`: use only particle smoothing lengths `h_b`.  
  - `itpSymmetric`: use the averaged kernel `0.5*(W(h_a)+W(h_b))`.

# Returns
- `ρ_interp::T`  
  Interpolated SPH density at `reference_point`, computed via kernel summation
  over all LBVH-selected neighbours.
"""
function density(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _density_kernel(input, reference_point, ha, LBVH, itp_strategy)
end

## Number density
"""
    number_density(input::InterpolationInput{T, V, K},
                   reference_point::NTuple{3, T},
                   ha::T,
                   LBVH::LinearBVH,
                   itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> T

Compute the SPH number density at a given reference point using a Linear
Bounding Volume Hierarchy (LBVH) for neighbour search.

This variant of `number_density` identifies neighbouring particles through the
supplied `LBVH`, selecting leaf AABBs intersecting the spherical support region
centred at `reference_point` with smoothing length `ha`. The selected particles
are passed to a low-level `_number_density_kernel` optimised for static typing
and high-performance SPH interpolation.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  Preprocessed, read-only SPH particle data container constructed via
  `build_input(...)`.
- `reference_point::NTuple{3, T}`  
  Cartesian coordinates (x, y, z) of the evaluation point.
- `ha::T`  
  Target smoothing length at the interpolation location.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy constructed from particle positions. Used to
  determine candidate neighbours inside the kernel support.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Strategy controlling the use of `h_a` and `h_b` in the kernel evaluation:  
  - `itpGather`: use only the target-point smoothing length `h_a`.  
  - `itpScatter`: use only particle smoothing lengths `h_b`.  
  - `itpSymmetric`: average kernel value `0.5*(W(h_a)+W(h_b))`.

# Returns
- `n_interp::T`  
  Interpolated SPH number density at `reference_point`, computed by summation
  over all neighbours returned by the LBVH search.
"""
function number_density(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _number_density_kernel(input, reference_point, ha, LBVH, itp_strategy)
end

## Single quantity intepolation
"""
    quantity_interpolate(input::InterpolationInput{T, V, K},
                         reference_point::NTuple{3, T},
                         ha::T,
                         LBVH::LinearBVH,
                         column_idx::Int64,
                         ShepardNormalization::Bool = true,
                         itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> T

Interpolate an arbitrary SPH quantity at a given reference point using a Linear
Bounding Volume Hierarchy (LBVH) for neighbour search.

This routine selects neighbouring particles by intersecting the LBVH leaf AABBs
with the spherical kernel support of radius `ha` centred at `reference_point`.
The filtered neighbours are then passed to the low-level
`_quantity_interpolate_kernel`, which performs SPH interpolation of the
requested particle field indexed by `column_idx`. Optional Shepard
normalization can be applied to enforce partition-of-unity behaviour.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  Preprocessed SPH particle data container created via `build_input(...)`.
- `reference_point::NTuple{3, T}`  
  Spatial coordinates (x, y, z) where the field is interpolated.
- `ha::T`  
  Target smoothing length at the interpolation point.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy built over particle positions. Determines
  candidate neighbours via AABB–kernel intersection.
- `column_idx::Int64`  
  Index of the particle field/column to interpolate.
- `ShepardNormalization::Bool = true`  
  When true, apply Shepard normalization to the kernel weights.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy specifying how `h_a` and `h_b` contribute to
  the kernel evaluation:  
  - `itpGather`: use only `h_a`.  
  - `itpScatter`: use only `h_b`.  
  - `itpSymmetric`: use the averaged kernel value `0.5*(W(h_a)+W(h_b))`.

# Returns
- `q_interp::T`  
  Interpolated SPH field value at `reference_point`, obtained from summation
  over neighbours selected via LBVH traversal.
"""
function quantity_interpolate(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int64, ShepardNormalization :: Bool = true,itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _quantity_interpolate_kernel(input, reference_point, ha, LBVH, column_idx, ShepardNormalization, itp_strategy)
end

## Muti-columns intepolation
"""
    quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN},
                           reference_point::NTuple{3, T},
                           ha::T,
                           LBVH::LinearBVH,
                           ShepardNormalization::NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN),
                           itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> NTuple{NCOLUMN, T}

Interpolate all registered scalar fields in an `InterpolationInput` at a 3D
location using an LBVH-accelerated neighbour search.

The LBVH traversal identifies all particles whose kernel support intersects the
sphere of radius `ha` centred at `reference_point`. For each neighbour, the SPH
kernel contribution is accumulated into a compile-time sized accumulator
(`NTuple{NCOLUMN, T}`). This avoids dynamic allocation and ensures GPU
compatibility. Shepard normalization can be toggled independently per field.

# Parameters
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  SPH particle dataset containing `NCOLUMN` scalar fields.
- `reference_point::NTuple{3, T}`  
  Spatial location `(x, y, z)` at which interpolation is performed.
- `ha::T`  
  Smoothing length associated with the interpolation point.
- `LBVH::LinearBVH`  
  Bounding volume hierarchy used for neighbour discovery.
- `ShepardNormalization::NTuple{NCOLUMN, Bool}`  
  Per-field Shepard-normalization flags.
- `itp_strategy::Type{ITPSTRATEGY}`  
  Kernel policy determining whether the contribution uses  
  `hₐ`, `hᵢ`, or their symmetric average.

# Returns
- `::MVector{NCOLUMN, T}`  
  Fixed-size MVector containing the interpolated values of the requested fields,
  in the same order as specified in `columns`.
"""
function quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _quantities_interpolate_kernel(input, reference_point, ha, LBVH, ShepardNormalization , itp_strategy)
end

"""
    quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN},
                           reference_point::NTuple{3, T},
                           ha::T,
                           LBVH::LinearBVH,
                           columns::NTuple{M, Int},
                           ShepardNormalization::NTuple{M, Bool} = ntuple(_ -> true, M),
                           itp_strategy::Type{ITPSTRATEGY} = itpSymmetric)
        -> MVector{M, T}

Interpolate a selected subset of SPH particle scalar fields at a 3D location
using an LBVH-accelerated neighbour search.

Only the fields specified by `columns` are evaluated. The LBVH traversal
identifies particles whose kernel support intersects the sphere of radius `ha`
centred at `reference_point`. For each neighbour, contributions are accumulated
into a compile-time sized `MVector{M,T}`, ensuring full type stability and GPU
compatibility. Shepard normalization can be independently enabled for each
requested field.

# Parameters
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  SPH particle dataset containing `NCOLUMN` scalar fields.
- `reference_point::NTuple{3, T}`  
  Cartesian coordinates `(x, y, z)` where interpolation is evaluated.
- `ha::T`  
  Smoothing length associated with the interpolation point.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy used for neighbour discovery.
- `columns::NTuple{M, Int}`  
  Indices of the particle fields to be interpolated.  
  `M` is the number of requested fields.
- `ShepardNormalization::NTuple{M, Bool}`  
  Flags enabling per-field Shepard normalization.
- `itp_strategy::Type{ITPSTRATEGY}`  
  Kernel interpolation policy determining the use of `hₐ`, `hᵢ`, or their
  symmetric average.

# Returns
- `::MVector{M, T}`  
  Fixed-size MVector containing the interpolated values of the requested fields,
  in the same order as specified in `columns`.
"""
function quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M),itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy, M}
    return _quantities_interpolate_kernel(input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end

## Multiple-Quantity Interpolation (In-Place)
"""
    quantities_interpolate!(workspace::Vector{T},
                            input::InterpolationInput{T, V, K, NCOLUMN},
                            reference_point::NTuple{3, T},
                            ha::T,
                            LBVH::LinearBVH,
                            ShepardNormalization::NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN),
                            itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> Nothing

In-place SPH interpolation of all particle fields stored in an
`InterpolationInput`, using LBVH-based neighbour search.

This routine computes interpolated values for **all `NCOLUMN` scalar fields**
and writes them directly into the provided `workspace` array. Neighbours are
selected by intersecting LBVH leaf AABBs with the kernel support region of
radius `ha` around `reference_point`. No memory allocation occurs except for
caller-provided buffers. The core computation is delegated to the in-place
`_quantities_interpolate_kernel!`.

# Parameters
- `workspace::Vector{T}`  
  Preallocated output buffer of length `NCOLUMN`. Filled with interpolated
  values.
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  Preprocessed SPH particle data with `NCOLUMN` scalar fields.
- `reference_point::NTuple{3, T}`  
  Evaluation position (x, y, z).
- `ha::T`  
  Target smoothing length at the interpolation location.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy for neighbour selection.
- `ShepardNormalization::NTuple{NCOLUMN, Bool}`  
  Per-field Shepard normalization flags. Defaults to all `true`.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy specifying how `h_a` and `h_b` are combined:  
  - `itpGather`: use only `h_a`.  
  - `itpScatter`: use only `h_b`.  
  - `itpSymmetric`: average kernel value `0.5*(W(h_a)+W(h_b))`.

# Returns
`nothing`. The results are written directly into `workspace`.
"""
function quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
    if NCOLUMN == 0
        return nothing
    end  
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
    _quantities_interpolate_kernel!(workspace, input, reference_point, ha, LBVH, ShepardNormalization, itp_strategy)
end

"""
    quantities_interpolate!(workspace::Vector{T},
                            input::InterpolationInput{T, V, K, NCOLUMN},
                            reference_point::NTuple{3, T},
                            ha::T,
                            LBVH::LinearBVH,
                            columns::NTuple{M, Int},
                            ShepardNormalization::NTuple{M, Bool} = ntuple(_ -> true, M),
                            itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> Nothing

In-place interpolation of a **selected subset** of particle fields using an
LBVH-based neighbour search.

This variant of `quantities_interpolate!` computes interpolated values for only
the fields specified by `columns`. It writes the results into the provided
`workspace`, whose length must match the number of selected fields `M`. The
LBVH is used to select neighbouring particles whose bounding boxes intersect
the smoothing sphere of radius `ha` centred at `reference_point`. The
in-place `_quantities_interpolate_kernel!` performs the SPH summation and (if
enabled) Shepard normalization.

# Parameters
- `workspace::Vector{T}`  
  Preallocated output buffer of length `M`. Filled with interpolated values.
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  Preprocessed SPH particle dataset containing `NCOLUMN` scalar fields.
- `reference_point::NTuple{3, T}`  
  Spatial coordinates (x, y, z) for interpolation.
- `ha::T`  
  Target smoothing length at the interpolation point.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy used to identify candidate neighbours.
- `columns::NTuple{M, Int}`  
  Indices of the particle fields to interpolate.
- `ShepardNormalization::NTuple{M, Bool} = ntuple(_ -> true, M)`  
  Per-field Shepard normalization flags corresponding to `columns`.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy controlling how `h_a` and `h_b` are combined:  
  - `itpGather`: use only `h_a`.  
  - `itpScatter`: use only `h_b`.  
  - `itpSymmetric`: average kernel `0.5*(W(h_a) + W(h_b))`.

# Returns
`nothing`. Results are written directly into `workspace`.
"""
function quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy, M}
    @assert length(workspace) == M "Length of `workspace` should match `columns`."
    if M == 0
        return nothing
    end
    _quantities_interpolate_kernel!(workspace, input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
    return nothing
end

## LOS density interpolation (Column / Surface density)
"""
    LOS_density(input::InterpolationInput{T, V, K},
                reference_point::NTuple{2, T},
                ha::T,
                LBVH::LinearBVH,
                itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> T

Compute the SPH line-of-sight (LOS) density at a 2-D reference point using an
LBVH for neighbour identification.

This function evaluates the projected (column) density along the line of sight
by summing contributions from particles whose AABBs intersect the circular
kernel support of radius `ha` in the 2-D projected space. The LBVH efficiently
selects candidate neighbours, and `_LOS_density_kernel` performs the
projection-aware SPH density summation.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  Preprocessed 3-D SPH particle data container.
- `reference_point::NTuple{2, T}`  
  2-D projected spatial position (e.g., image-plane coordinates).
- `ha::T`  
  Smoothing length associated with the projected kernel.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy built over particle positions, projected for
  LOS search.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy defining how `h_a` and `h_b` contribute in the
  projected kernel:
  - `itpGather`: use only `h_a`.  
  - `itpScatter`: use only `h_b`.  
  - `itpSymmetric`: average kernel value.

# Returns
- `Σ_interp::T`  
  Projected SPH column density at the given 2-D reference point.
"""
function LOS_density(input::InterpolationInput{T, V, K}, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _LOS_density_kernel(input, reference_point, ha, LBVH, itp_strategy)
end


## LOS quantities interpolation
"""
    LOS_quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN},
                               reference_point::NTuple{2, T},
                               ha::T,
                               LBVH::LinearBVH,
                               ShepardNormalization::NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN),
                               itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> Vector{T}

Interpolate all SPH particle fields in *projected (2-D) space* using an LBVH
for neighbour selection, producing line-of-sight (LOS)–weighted quantities.

This routine computes LOS-projected interpolated values for all `NCOLUMN` scalar
fields. Candidate neighbours are obtained by intersecting LBVH leaf AABBs with
the circular kernel support of radius `ha` centred at the 2-D projection
`reference_point`. The resulting particle subset is passed into
`_LOS_quantities_interpolate_kernel!`, which performs projection-aware SPH
summation and optional Shepard normalization.

# Parameters
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  Preprocessed SPH particle dataset containing `NCOLUMN` scalar fields.
- `reference_point::NTuple{2, T}`  
  2-D projected coordinates where LOS quantities are evaluated.
- `ha::T`  
  Smoothing length used for the projected kernel support.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy providing neighbour candidates in
  projected space.
- `ShepardNormalization::NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN)`  
  Per-field Shepard normalization flags for LOS interpolation.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy (`itpGather`, `itpScatter`, `itpSymmetric`).

# Returns
- `::MVector{NCOLUMN, T}`  
  Fixed-size MVector containing the interpolated values of the requested fields,
  in the same order as specified in `columns`.
"""
function LOS_quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _LOS_quantities_interpolate_kernel(input, reference_point, ha, LBVH, ShepardNormalization , itp_strategy)
end

"""
    LOS_quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN},
                               reference_point::NTuple{2, T},
                               ha::T,
                               LBVH::LinearBVH,
                               columns::NTuple{M, Int},
                               ShepardNormalization::NTuple{M, Bool} = ntuple(_ -> true, M),
                               itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> Vector{T}

Interpolate a selected subset of SPH particle fields in projected (2-D) space
using an LBVH for neighbour detection, producing line-of-sight (LOS) weighted
quantities.

This variant computes LOS-projected interpolated values only for the fields
listed in `columns`. An output workspace of size `M` is allocated. Neighbours
are identified through LBVH intersection tests between particle AABBs and the
circular kernel support of radius `ha` centered at the 2-D `reference_point`.
Computation is handled by `_LOS_quantities_interpolate_kernel!`, which performs
the LOS-aware SPH summation with optional Shepard normalization.

# Parameters
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  Preprocessed SPH dataset containing `NCOLUMN` scalar fields.
- `reference_point::NTuple{2, T}`  
  2-D projected evaluation position.
- `ha::T`  
  Kernel smoothing length used in the projected space.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy used to select neighbour candidates.
- `columns::NTuple{M, Int}`  
  Indices of the fields to interpolate.
- `ShepardNormalization::NTuple{M, Bool} = ntuple(_ -> true, M)`  
  Per-field Shepard normalization flags corresponding to `columns`.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy (`itpGather`, `itpScatter`, `itpSymmetric`).

# Returns
- `::MVector{M, T}`  
  Fixed-size MVector containing the interpolated values of the requested fields,
  in the same order as specified in `columns`.
"""
function LOS_quantities_interpolate(input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy, M}
    return _quantities_interpolate_kernel(input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end

"""
    LOS_quantities_interpolate!(workspace::Vector{T},
                                input::InterpolationInput{T, V, K, NCOLUMN},
                                reference_point::NTuple{2, T},
                                ha::T,
                                LBVH::LinearBVH,
                                ShepardNormalization::NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN),
                                itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> Nothing

In-place interpolation of all SPH particle fields in projected (2-D) space
using LBVH-based neighbour search, producing line-of-sight (LOS) weighted
quantities.

This function computes LOS-projected interpolated values for all `NCOLUMN`
fields and writes the results directly into the user-provided `workspace`
buffer. Neighbours are selected by intersecting LBVH leaf AABBs with the
circular kernel support of radius `ha` centred at the 2-D projected position
`reference_point`. The core computation is performed by
`_LOS_quantities_interpolate_kernel!`, which evaluates LOS-weighted SPH
summations with optional Shepard normalization.

# Parameters
- `workspace::Vector{T}`  
  Preallocated buffer of length `NCOLUMN`. Filled with LOS-projected
  interpolated values.
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  Preprocessed SPH dataset containing `NCOLUMN` scalar particle fields.
- `reference_point::NTuple{2, T}`  
  Evaluation point in 2-D projected space.
- `ha::T`  
  Smoothing length associated with the projected kernel.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy used to identify neighbour candidates.
- `ShepardNormalization::NTuple{NCOLUMN, Bool}`  
  Per-field Shepard normalization flags. Must match the number of fields.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation mode (`itpGather`, `itpScatter`, `itpSymmetric`).

# Returns
`nothing`. Results are written directly into `workspace`.
"""
function LOS_quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
    if NCOLUMN == 0
      return nothing
    end
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
  _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, LBVH, ShepardNormalization, itp_strategy)
end

"""
    LOS_quantities_interpolate!(workspace::Vector{T},
                                input::InterpolationInput{T, V, K, NCOLUMN},
                                reference_point::NTuple{2, T},
                                ha::T,
                                LBVH::LinearBVH,
                                columns::NTuple{M, Int},
                                ShepardNormalization::NTuple{M, Bool} = ntuple(_ -> true, M),
                                itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> Nothing

In-place interpolation of a selected subset of SPH particle fields in projected
(2-D) space using LBVH-based neighbour search, producing line-of-sight (LOS)
weighted quantities.

This method computes LOS-projected SPH interpolated values only for the fields
listed in `columns`. The caller-provided `workspace` must have length `M`,
matching the number of selected fields. Neighbouring particles are determined
via LBVH intersection between leaf AABBs and the projected kernel support
(circle of radius `ha`). The computation is performed by
`_LOS_quantities_interpolate_kernel!`, which applies LOS-weighted SPH summation
and optional Shepard normalization.

# Parameters
- `workspace::Vector{T}`  
  Output buffer of length `M` to be filled in-place.
- `input::InterpolationInput{T, V, K, NCOLUMN}`  
  Preprocessed SPH particle data.
- `reference_point::NTuple{2, T}`  
  2-D projected evaluation location.
- `ha::T`  
  Smoothing length for the projected kernel.
- `LBVH::LinearBVH`  
  LBVH used for neighbour identification in projected space.
- `columns::NTuple{M, Int}`  
  Indices of scalar fields to interpolate.
- `ShepardNormalization::NTuple{M, Bool} = ntuple(_ -> true, M)`  
  Shepard normalization flags corresponding to the selected fields.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy (`itpGather`, `itpScatter`, `itpSymmetric`).

# Returns
`nothing`. All results are written directly into `workspace`.
"""
function LOS_quantities_interpolate!(workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy, M}
    @assert length(workspace) == M "Length of `workspace` should match `columns`."
    if M == 0
        return nothing
    end
  _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
  return nothing
end

function LOS_quantities_interpolate!(buffer :: NTuple{NCOLUMN, SA}, workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH,ShepardNormalization :: NTuple{NCOLUMN, Bool} = ntuple(_ -> true, NCOLUMN), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, SA<:AbstractArray{T, 0}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    if NCOLUMN == 0
      return nothing
    end
    @assert length(workspace) == NCOLUMN "Length of `workspace` should be identical as NCOLUMN."
  _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, LBVH, ShepardNormalization, itp_strategy)
    @inbounds for i in eachindex(buffer)
        buffer[i][] = workspace[i]
    end
end

  function LOS_quantities_interpolate!(buffer :: NTuple{M, SA}, workspace :: Vector{T}, input::InterpolationInput{T, V, K, NCOLUMN}, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool} = ntuple(_ -> true, M), itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {M, NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, SA<:AbstractArray{T, 0}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    @assert length(workspace) == M "Length of `workspace` should match `columns`."
    if M == 0
      return nothing
    end
    _LOS_quantities_interpolate_kernel!(workspace, input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
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
                     LBVH::LinearBVH,
                     itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> NTuple{3, T}

Compute the SPH density gradient ∇ρ at a 3-D reference point using an LBVH for
efficient neighbour selection.

This routine evaluates the spatial gradient of the SPH density field by
performing kernel-gradient summation over particles whose AABBs intersect the
spherical support region of radius `ha` centred at `reference_point`. Neighbour
selection is handled by the LBVH, while `_gradient_density_kernel` performs the
actual SPH gradient computation using the specified interpolation strategy.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  Preprocessed SPH particle data structure.
- `reference_point::NTuple{3, T}`  
  Evaluation point (x, y, z) in 3-D space.
- `ha::T`  
  Smoothing length at the interpolation point.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy used for neighbour queries.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy controlling the use of `h_a` and `h_b`:  
  - `itpGather` — use only `h_a`  
  - `itpScatter` — use only `h_b`  
  - `itpSymmetric` — use the averaged kernel gradient

# Returns
- `∇ρ::NTuple{3, T}`  
  The SPH density gradient evaluated at `reference_point`.
"""
function gradient_density(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _gradient_density_kernel(input, reference_point, ha, LBVH, itp_strategy)
end

# Single column gradient value intepolation
"""
    gradient_quantity_interpolate(input::InterpolationInput{T, V, K},
                                  reference_point::NTuple{3, T},
                                  ha::T,
                                  LBVH::LinearBVH,
                                  column_idx::Int64,
                                  itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> NTuple{3, T}

Compute the spatial gradient of an arbitrary SPH quantity at a 3-D reference
point using LBVH-based neighbour search.

This function evaluates the gradient ∇Q of the particle field indexed by
`column_idx`. Neighbours are selected by intersecting LBVH leaf AABBs with the
kernel support sphere of radius `ha`. The low-level
`_gradient_quantity_interpolate_kernel` performs the SPH gradient summation
using the specified kernel interpolation strategy.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  Preprocessed SPH particle data.
- `reference_point::NTuple{3, T}`  
  3-D position (x, y, z) where the quantity gradient is evaluated.
- `ha::T`  
  Smoothing length at the evaluation point.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy used to identify neighbouring particles.
- `column_idx::Int64`  
  Index of the scalar field whose gradient is to be interpolated.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy (`itpGather`, `itpScatter`, `itpSymmetric`).

# Returns
- `∇Q::NTuple{3, T}`  
  The interpolated gradient of the selected quantity at `reference_point`.
"""
function gradient_quantity_interpolate(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _gradient_quantity_interpolate_kernel(input, reference_point, ha, LBVH, column_idx, itp_strategy)
end

# Single column divergence value intepolation
"""
    divergence_quantity_interpolate(input::InterpolationInput{T, V, K},
                                    reference_point::NTuple{3, T},
                                    ha::T,
                                    LBVH::LinearBVH,
                                    Ax_column_idx::Int64,
                                    Ay_column_idx::Int64,
                                    Az_column_idx::Int64,
                                    itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> T

Compute the divergence of a vector-valued SPH quantity at a 3-D reference point
using LBVH-based neighbour search.

This routine evaluates the divergence ∇·A of a vector field **A = (Aₓ, Aᵧ, A_z)**,
where each component is stored in the SPH dataset as a separate scalar column,
given by `Ax_column_idx`, `Ay_column_idx`, and `Az_column_idx`. Neighbours are
selected through LBVH AABB–sphere intersection using the smoothing length `ha`.
The SPH divergence summation is performed by
`_divergence_quantity_interpolate_kernel`.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  Preprocessed SPH particle dataset.
- `reference_point::NTuple{3, T}`  
  3-D position (x, y, z) where the divergence is evaluated.
- `ha::T`  
  Smoothing length at the evaluation point.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy used for neighbour discovery.
- `Ax_column_idx::Int64`  
  Column index of the x-component of the vector field.
- `Ay_column_idx::Int64`  
  Column index of the y-component of the vector field.
- `Az_column_idx::Int64`  
  Column index of the z-component of the vector field.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy (`itpGather`, `itpScatter`, `itpSymmetric`).

# Returns
- `∇·A::T`  
  Divergence of the vector field evaluated at `reference_point`.
"""
function divergence_quantity_interpolate(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _divergence_quantity_interpolate_kernel(input, reference_point, ha, LBVH, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy)
end

# Single column curl value intepolation
"""
    curl_quantity_interpolate(input::InterpolationInput{T, V, K},
                              reference_point::NTuple{3, T},
                              ha::T,
                              LBVH::LinearBVH,
                              Ax_column_idx::Int64,
                              Ay_column_idx::Int64,
                              Az_column_idx::Int64,
                              itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) -> NTuple{3, T}

Compute the curl of a vector-valued SPH quantity at a 3-D reference point using
LBVH-based neighbour search.

This function evaluates the curl **∇×A** of a vector field  
A = (Aₓ, Aᵧ, A_z), where each component is stored in a separate scalar column
indexed by `Ax_column_idx`, `Ay_column_idx`, and `Az_column_idx`. Neighbours are
collected by checking intersection of LBVH leaf AABBs with the spherical kernel
support of radius `ha`. The SPH curl summation is implemented in
`_curl_quantity_interpolate_kernel`.

# Parameters
- `input::InterpolationInput{T, V, K}`  
  Preprocessed SPH particle data.
- `reference_point::NTuple{3, T}`  
  3-D Cartesian point (x, y, z) where the curl is evaluated.
- `ha::T`  
  Smoothing length at the interpolation point.
- `LBVH::LinearBVH`  
  Linear bounding volume hierarchy used for neighbour detection.
- `Ax_column_idx::Int64`  
  Column index of the x-component of the vector field.
- `Ay_column_idx::Int64`  
  Column index of the y-component of the vector field.
- `Az_column_idx::Int64`  
  Column index of the z-component of the vector field.
- `itp_strategy::Type{ITPSTRATEGY} = itpSymmetric`  
  Kernel interpolation strategy (`itpGather`, `itpScatter`, `itpSymmetric`).

# Returns
- `∇×A::NTuple{3, T}`  
  The curl of the vector field evaluated at `reference_point`.
"""
function curl_quantity_interpolate(input::InterpolationInput{T, V, K}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, ITPSTRATEGY <: AbstractInterpolationStrategy}
  return _curl_quantity_interpolate_kernel(input, reference_point, ha, LBVH, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy)
end
