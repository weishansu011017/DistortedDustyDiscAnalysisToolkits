"""
Splash-style tabulated full line-integrated SPH kernel functions.
    by Wei-Shan Su,
    October 31, 2025

This file provides tabulated full line-integrated versions of the standard 3D
SPH smoothing kernels. It is intended for efficient evaluation of column
density and related line-integrated quantities by replacing the 3D kernel
`W(r, h)` with its full line-integrated counterpart.
"""

# Line-integrated kernel lookup
@inline function _lin_lut(q, Q::SVector{N,T}, I::SVector{N,T}) where {N,T}
    dq   = Q[2] - Q[1]            
    idxf = q / dq + 1             
    i    = clamp(Int(floor(idxf)), 1, N-1)
    t    = idxf - i             
    return I[i]*(1-t) + I[i+1]*t
end

for (K, Qsym, Inormsym) in (
    (M4_spline,   :_M4_spline_Q,   _M4_spline_Inorm),
    (M5_spline,   :_M5_spline_Q,   _M5_spline_Inorm),
    (M6_spline,   :_M6_spline_Q,   _M6_spline_Inorm),
    (C2_Wendland, :_C2_Wendland_Q, _C2_Wendland_Inorm),
    (C4_Wendland, :_C4_Wendland_Q, _C4_Wendland_Inorm),
    (C6_Wendland, :_C6_Wendland_Q, _C6_Wendland_Inorm),
)
    @eval @inline lookup_line_integrated_kernel(::Type{$K}, q_perp::T) where {T <: AbstractFloat} =
    _lin_lut(T(q_perp), $Qsym, $Inormsym,)
end

"""
    line_integrated_kernel_function_dimensionless(
        ::Type{K},
        q_perp::T,
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

Evaluate the **dimensionless line-integrated SPH kernel**
for a given dimensionless transverse separation `q_perp`.

This function returns the tabulated value of the line-integrated
kernel shape function,

    I(q⊥) = ∫ w(√(q⊥² + q∥²)) dq∥ ,

where `q⊥` (represented in code as `q_perp`) is the dimensionless transverse
separation from the integration line. The integration is truncated at the
kernel support radius.

If `q_perp` exceeds the kernel support (`q_perp ≥ q_max`), the function returns zero.

# Parameters
- `::Type{K}`
  SPH kernel type, where `K <: AbstractSPHKernel`.
- `q_perp::T`
  Dimensionless transverse separation from the integration line.

# Returns
- `T`
  Dimensionless line-integrated kernel value.

# Notes
- This function is **dimensionless** and does not include any physical
  prefactors involving `h`.
- The support cutoff is determined by `KernelFunctionValid(K, T)`.
- Intended for use in column-density or other line-integrated quantity calculations.
- Performs no heap allocation and is suitable for hot loops.
"""
@inline function line_integrated_kernel_function_dimensionless(::Type{K}, q_perp::T) where {K <: AbstractSPHKernel, T <: AbstractFloat}
    q_perp ≥ KernelFunctionValid(K, T) && return zero(T)
    Iq = lookup_line_integrated_kernel(K, q_perp)
    return Iq
end

"""
    line_integrated_kernel_function(
        ::Type{<:AbstractSPHKernel},
        r::T,
        h::T
    ) where {T<:AbstractFloat}

Evaluate the **line-integrated SPH kernel** at transverse distance `r`
for smoothing length `h`.

This function evaluates the tabulated full line-integrated kernel associated
with a 3D SPH smoothing kernel,

    I(r, h) = ∫ W(√(r² + s²), h) ds ,

where `r` is the transverse distance from the integration line. Internally,
the computation is performed through the dimensionless transverse coordinate
`q_perp = r / h`, followed by the physical `1 / h` prefactor.

This is the splash-style full line integration lookup and is intended for
column-density or other line-integrated quantity evaluations. It is not a
replacement for the original 3D kernel in volumetric interactions.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  SPH kernel type.
- `r::T`  
  Transverse distance from the integration line.
- `h::T`  
  Smoothing length.

# Returns
- `T`  
  Physical line-integrated kernel value, including the `1 / h` scaling.

"""
@inline function line_integrated_kernel_function(::Type{K}, r::T, h::T) where {K <: AbstractSPHKernel, T <: AbstractFloat}
    invh = inv(h)
    q_perp = r * invh
    I_dimless = line_integrated_kernel_function_dimensionless(K, q_perp)
    return invh * I_dimless
end

@inline function line_integrated_kernel_function(::Type{K}, r::T, h::S) where {K <: AbstractSPHKernel, T <: AbstractFloat, S <: AbstractFloat}
    rp, hp = promote(r, h)
    return line_integrated_kernel_function(K, rp, hp)
end

"""
    line_integrated_kernel_function(
        ::Type{<:AbstractSPHKernel},
        ra::NTuple{2,T},
        rb::NTuple{2,T},
        h::T
    ) where {T<:AbstractFloat}

Evaluate the **line-integrated SPH kernel** from two transverse-plane
coordinates and a smoothing length.

This method computes the transverse separation

    r = √((xₐ − x_b)² + (yₐ − y_b)²)

in the plane perpendicular to the integration line, then evaluates the
line-integrated kernel value

    I(r, h) = ∫ W(√(r² + s²), h) ds .

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  SPH kernel type.
- `ra::NTuple{2,T}`  
  2D transverse-plane coordinates of the evaluation point.
- `rb::NTuple{2,T}`  
  2D transverse-plane coordinates of the source point.
- `h::T`  
  Smoothing length.

# Returns
- `T`  
  Physical line-integrated kernel value, including the `1 / h` scaling.
"""
@inline function line_integrated_kernel_function(::Type{K}, ra::NTuple{2,T}, rb::NTuple{2,T}, h::T) where {K <: AbstractSPHKernel, T <: AbstractFloat}
    rax, ray = ra
    rbx, rby = rb
    Δx = rax - rbx
    Δy = ray - rby
    r2 = Δx * Δx + Δy * Δy
    r = sqrt(r2)
    return line_integrated_kernel_function(K, r, h) 
end
