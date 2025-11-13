"""
InterpolationInput.jl  
    by Wei-Shan Su  
    October 31, 2025  

Definition of the immutable input structure for SPH single-point interpolation.  

This file defines the abstract interface and concrete implementation of `InterpolationInput`,
a statically-typed, GPU-compatible data container used for all SPH interpolation routines
(e.g. density, gradient, divergence, LOS projection). The structure enforces type safety and 
read-only semantics to ensure reproducibility and performance in both CPU and GPU execution contexts.

# Overview
Each interpolation kernel in the `single_point_interpolation` system operates on an
`InterpolationInput` instance, which encapsulates all particle quantities required for SPH summation.

An `InterpolationInput` stores:
- particle positions `(x, y, z)`,
- smoothing lengths `h`,
- masses `m`,
- densities `ρ`,
- a tuple of arbitrary scalar fields `quant` (e.g., temperature, pressure),
- and the kernel type `smoothed_kernel`.

All fields are stored as immutable vectors (`AbstractVector{T}`), and the structure is designed
to be `isbits` when used with `CuDeviceVector` or `SVector` types, enabling full GPU compatibility.

# Provided Definitions
- `AbstractInterpolationInput` — abstract supertype for interpolation input objects.  
- `InterpolationInput{T,V,K,NCOLUMN}` — concrete immutable struct storing particle data.  
- `Base.length(::InterpolationInput)` — returns the number of valid particles (`Npart`).  
- `Base.checkbounds(::InterpolationInput)` — validates internal vector lengths.  
- `get_type(::InterpolationInput)` — retrieves the floating-point element type `T`.  

# Design Notes
- This structure is read-only; data should be pre-filled and never mutated after construction.  
- `quant` is stored as an `NTuple{NCOLUMN, V}` to guarantee homogeneity and GPU safety.  
- Intended for use with all SPH interpolation kernels in `single_point_interpolation/single_point_interpolation.jl`.

# Dependencies
- Requires kernel definitions from `KernelInterpolation/kernel_function.jl`.
- Can be constructed using `constructor.jl`, which converts a `PhantomRevealerDataFrame` to a valid input object.

"""


# Input and output structure
abstract type AbstractInterpolationInput end

"""
    InterpolationInput{T, V, K, NCOLUMN}

Immutable SPH input container for read-only interpolation queries.

This struct is designed for fast and safe interpolation on CPU and GPU. It contains all the necessary SPH quantities. The `quant` field holds `NCOLUMN` scalar fields (e.g., pressure, temperature) in a tuple.

**This struct is fully isbits and read-only** — all internal data should be pre-filled and never mutated. Intended for bulk interpolation kernels.

# Type Parameters
- `T`: Floating-point type (e.g., `Float32` or `Float64`).
- `V`: An `AbstractVector{T}`, the vector type used throughout.
- `K`: Type of SPH kernel used, must be a concrete `AbstractSPHKernel`.
- `NCOLUMN`: Number of scalar fields in `quant`.

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
struct InterpolationInput{T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, NCOLUMN} <: AbstractInterpolationInput
    Npart           :: Int64
    smoothed_kernel :: K
    x               :: V
    y               :: V
    z               :: V
    m               :: V
    h               :: V
    ρ               :: V
    quant           :: NTuple{NCOLUMN, V}
end

function Adapt.adapt_structure(to, x :: ITPINPUT) where {T<:AbstractFloat, V<:AbstractVector{T}, K<:AbstractSPHKernel, NCOLUMN, ITPINPUT <: InterpolationInput{T, V, K, NCOLUMN}}
    InterpolationInput(
        x.Npart,
        Adapt.adapt(to, x.smoothed_kernel),
        Adapt.adapt(to, x.x),
        Adapt.adapt(to, x.y),
        Adapt.adapt(to, x.z),
        Adapt.adapt(to, x.m),
        Adapt.adapt(to, x.h),
        Adapt.adapt(to, x.ρ),
        ntuple(i->Adapt.adapt(to, x.quant[i]), NCOLUMN)
    )
end

# Some useful function 
## Get "Valid" range of data (the other would be 0)
Base.length(inp::T) where {T<:AbstractInterpolationInput} = inp.Npart        # N

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
    for (k, v) in enumerate(inp.quant)
        @assert N ≤ length(v) "quant[$k] is shorter than Npart ($N)"
    end
    return true
end

"""
    get_type(::InterpolationInput) -> Type{T}

Return the floating-point type `T` used in the `InterpolationInput`.

This is the type used for all scalar fields and positions in the interpolation system.
"""
@inline function get_type(::InterpolationInput{T}) where {T<:AbstractFloat}
    return T
end

@inline function _apply_permutation!(data::AbstractVector, perm::AbstractVector)
    len_perm = length(perm)
    if len_perm == 0
        return data
    end
    @boundscheck len_perm <= length(data) || throw(ArgumentError("permutation longer than data"))
    tmp = similar(data, len_perm)
    @inbounds for i in 1:len_perm
        tmp[i] = data[Int(perm[i])]
    end
    copyto!(data, tmp)
    return data
end

function LinearBVH!(inp::InterpolationInput, ::Val{3}; CodeType :: Type{TI} = UInt64) where {TI<:Unsigned}
    enc = MortonEncoding(inp.x, inp.y, inp.z, CodeType=CodeType)
    order = enc.order

    _apply_permutation!(inp.x, order)
    _apply_permutation!(inp.y, order)
    _apply_permutation!(inp.z, order)
    _apply_permutation!(inp.m, order)
    _apply_permutation!(inp.h, order)
    _apply_permutation!(inp.ρ, order)
    for column in inp.quant
        _apply_permutation!(column, order)
    end

    brt = BinaryRadixTree(enc)
    LBVH = LinearBVH(enc, brt)
    return LBVH
end

function LinearBVH!(inp::InterpolationInput, ::Val{2}; CodeType :: Type{TI} = UInt64) where {TI<:Unsigned}
    enc = MortonEncoding(inp.x, inp.y, CodeType=CodeType)
    order = enc.order

    _apply_permutation!(inp.x, order)
    _apply_permutation!(inp.y, order)
    _apply_permutation!(inp.m, order)
    _apply_permutation!(inp.h, order)
    _apply_permutation!(inp.ρ, order)
    for column in inp.quant
        _apply_permutation!(column, order)
    end

    brt = BinaryRadixTree(enc)
    LBVH = LinearBVH(enc, brt)
    return LBVH
end

