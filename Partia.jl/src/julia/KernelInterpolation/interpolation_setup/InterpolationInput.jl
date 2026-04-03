"""
InterpolationInput.jl
    by Wei-Shan Su
    October 31, 2025

Definition of the immutable input structure for SPH interpolation.

This file defines the abstract interface and concrete implementation of
`InterpolationInput`, a statically-typed, GPU-compatible data container used by
the interpolation kernels in `KernelInterpolation`.

Coordinates are stored as `coord::NTuple{D,V}` rather than separate `x/y/z`
fields. The storage is dimension-aware, while the current interpolation kernels
still mainly target the 3D path via `InterpolationInput{3,...}`.
"""

"""
    InterpolationInput{D, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, NCOLUMN}

Immutable SPH input container for read-only interpolation queries.

This struct is designed for fast and safe interpolation on CPU and GPU. It
contains all particle quantities required by the SPH interpolation kernels.
Coordinates are stored as `coord::NTuple{D,V}`, so the spatial dimension is
carried in the type rather than inferred from separate `x/y/z` fields.

# Type Parameters
- `D`: Spatial dimension.
- `T`: Floating-point type (e.g. `Float32` or `Float64`).
- `V`: An `AbstractVector{T}`, the vector type used throughout.
- `K`: Type of SPH kernel used. Must be a concrete `AbstractSPHKernel`.
- `NCOLUMN`: Number of scalar fields stored in `quant`.

# Fields
- `Npart::Int64`: Number of active (valid) particles within the batch.
- `smoothed_kernel::K`: SPH kernel function instance.
- `coord::NTuple{D,V}`: Particle coordinates, e.g. `(x, y, z)` for `D == 3`.
- `m::V`: Particle masses.
- `h::V`: Particle smoothing lengths.
- `ρ::V`: Particle densities.
- `quant::NTuple{NCOLUMN,V}`: Tuple of per-field scalar data arrays.
"""
struct InterpolationInput{D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN}
    Npart           :: Int64
    smoothed_kernel :: K
    coord           :: NTuple{D, V}
    m               :: V
    h               :: V
    ρ               :: V
    quant           :: NTuple{NCOLUMN, V}
end

function Adapt.adapt_structure(to, x :: InterpolationInput{D, T, V, K, NCOLUMN}) where {D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN}
    InterpolationInput(
        x.Npart,
        Adapt.adapt(to, x.smoothed_kernel),
        ntuple(i -> Adapt.adapt(to, x.coord[i]), Val(D)),
        Adapt.adapt(to, x.m),
        Adapt.adapt(to, x.h),
        Adapt.adapt(to, x.ρ),
        ntuple(i -> Adapt.adapt(to, x.quant[i]), Val(NCOLUMN)),
    )
end

"""
    InterpolationInput(coord::NTuple{D,V}, m::V, h::V, ρ::V, quant::NTuple{NCOLUMN,V}; smoothed_kernel::Type{K} = M5_spline) where {D, NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}

Construct an interpolation input from materialized particle columns.

# Parameters
- `coord::NTuple{D,V}`: Coordinate tuple, such as `(x, y, z)`.
- `m::V`: Particle masses.
- `h::V`: Particle smoothing lengths.
- `ρ::V`: Particle densities.
- `quant::NTuple{NCOLUMN,V}`: Tuple of scalar field columns.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `smoothed_kernel` | `Type{K}` | `M5_spline` | Kernel type used to construct the stored kernel instance. |

# Returns
- `InterpolationInput{D,T,V,K,NCOLUMN}`: Interpolation input with validated column lengths.
"""
function InterpolationInput(coord :: NTuple{D, V}, m :: V, h :: V, ρ :: V, quant :: NTuple{NCOLUMN, V}; smoothed_kernel :: Type{K} = M5_spline) where {D, NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}
    Npart = length(m)
    @inbounds for d in 1:D
        length(coord[d]) == Npart || throw(
            DimensionMismatch("coord[$d] length $(length(coord[d])) != Nparticles $Npart"),
        )
    end
    length(h) == Npart || throw(DimensionMismatch("h length $(length(h)) != Nparticles $Npart"))
    length(ρ) == Npart || throw(DimensionMismatch("ρ length $(length(ρ)) != Nparticles $Npart"))
    @inbounds for j in 1:NCOLUMN
        length(quant[j]) == Npart || throw(
            DimensionMismatch("quant[$j] length $(length(quant[j])) != Nparticles $Npart"),
        )
    end

    return InterpolationInput{D, T, V, K, NCOLUMN}(
        Npart,
        smoothed_kernel(),
        coord,
        m,
        h,
        ρ,
        quant,
    )
end


# Basic constructors
"""
    InterpolationInput(x::V, y::V, m::V, h::V, ρ::V, quant::NTuple{NCOLUMN,V}; smoothed_kernel::Type{K} = M5_spline) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}

Construct a 2D interpolation input from separate coordinate vectors.

# Parameters
- `x::V`: Particle x-coordinates.
- `y::V`: Particle y-coordinates.
- `m::V`: Particle masses.
- `h::V`: Particle smoothing lengths.
- `ρ::V`: Particle densities.
- `quant::NTuple{NCOLUMN,V}`: Tuple of scalar field columns.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `smoothed_kernel` | `Type{K}` | `M5_spline` | Kernel type used to construct the stored kernel instance. |

# Returns
- `InterpolationInput{2,T,V,K,NCOLUMN}`: 2D interpolation input.
"""
@inline function InterpolationInput(x :: V, y :: V, m :: V, h :: V, ρ :: V, quant :: NTuple{NCOLUMN, V}; smoothed_kernel::Type{K} = M5_spline) where {NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}
    return InterpolationInput((x, y), m, h, ρ, quant; smoothed_kernel = smoothed_kernel)
end

"""
    InterpolationInput(x::V, y::V, z::V, m::V, h::V, ρ::V, quant::NTuple{NCOLUMN,V}; smoothed_kernel::Type{K} = M5_spline) where {NCOLUMN, T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel}

Construct a 3D interpolation input from separate coordinate vectors.

# Parameters
- `x::V`: Particle x-coordinates.
- `y::V`: Particle y-coordinates.
- `z::V`: Particle z-coordinates.
- `m::V`: Particle masses.
- `h::V`: Particle smoothing lengths.
- `ρ::V`: Particle densities.
- `quant::NTuple{NCOLUMN,V}`: Tuple of scalar field columns.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `smoothed_kernel` | `Type{K}` | `M5_spline` | Kernel type used to construct the stored kernel instance. |

# Returns
- `InterpolationInput{3,T,V,K,NCOLUMN}`: 3D interpolation input.
"""
function InterpolationInput(x :: V, y :: V, z :: V, m :: V, h :: V, ρ :: V, quant :: NTuple{NCOLUMN, V}; smoothed_kernel :: Type{K} = M5_spline) where {NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}
    return InterpolationInput((x, y, z), m, h, ρ, quant; smoothed_kernel = smoothed_kernel)
end


# Some useful function
## Get "Valid" range of data (the other would be 0)
@inline Base.length(input :: InterpolationInput) = input.Npart

## Get element type of the input
@inline Base.eltype(:: InterpolationInput{D, T}) where {D, T <: AbstractFloat} = T

## Get dimension of the input
@inline spatial_dimension(:: InterpolationInput{D}) where {D} = D

## Coordinate accessors
@inline get_coord(input :: InterpolationInput{D}) where {D} = input.coord
@inline get_xcoord(input :: InterpolationInput{D}) where {D} = input.coord[1]
@inline get_ycoord(input :: InterpolationInput{D}) where {D} = input.coord[2]
@inline get_zcoord(input :: InterpolationInput{3}) = input.coord[3]


# Check the "Valid" length of data for each fields
function Base.checkbounds(input::InterpolationInput)
    N = input.Npart
    @assert N isa Integer && N >= 0 "Invalid Npart: $N"

    @inbounds for d in 1:spatial_dimension(input)
        @assert N <= length(input.coord[d]) "coord[$d] is shorter than Npart ($N)"
    end

    @assert N <= length(input.m) "m is shorter than Npart ($N)"
    @assert N <= length(input.h) "h is shorter than Npart ($N)"
    @assert N <= length(input.ρ) "ρ is shorter than Npart ($N)"

    @inbounds for (k, v) in enumerate(input.quant)
        @assert N <= length(v) "quant[$k] is shorter than Npart ($N)"
    end
    return true
end

# Input helper for LBVH
## 3D path
function LinearBVH!(input::InterpolationInput{3}, ::Val{3}; CodeType::Type{TI} = UInt64) where {TI<:Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    enc = MortonEncoding(x, y, z, input.h, CodeType = CodeType)
    order = enc.order

    Base.permute!(x, order)
    Base.permute!(y, order)
    Base.permute!(z, order)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    Base.permute!(input.ρ, order)
    for column in input.quant
        Base.permute!(column, order)
    end

    brt = BinaryRadixTree(enc)
    return LinearBVH(enc, brt)
end

## 2D path
function LinearBVH!(input::InterpolationInput{2}, ::Val{2}; CodeType::Type{TI} = UInt64) where {TI<:Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)

    enc = MortonEncoding(x, y, input.h, CodeType = CodeType)
    order = enc.order

    Base.permute!(x, order)
    Base.permute!(y, order)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    Base.permute!(input.ρ, order)
    for column in input.quant
        Base.permute!(column, order)
    end

    brt = BinaryRadixTree(enc)
    return LinearBVH(enc, brt)
end
