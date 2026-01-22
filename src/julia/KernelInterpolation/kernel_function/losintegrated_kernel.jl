"""
Line‐of‐Sight (LOS) integrated SPH kernel functions.  
    by Wei-Shan Su,  
    October 31, 2025  

This module provides the **projected (LOS-integrated)** versions of the standard SPH
smoothing kernels.  It allows efficient evaluation of surface densities or
column–integrated quantities by replacing the 3-D kernel  W(r,h)  with its
line-of-sight integral.
"""

# Line-of-Sight Integrated Kernel function
@inline function _lin_lut(q, Q::SVector{N,T}, I::SVector{N,T}) where {N,T}
    dq   = Q[2] - Q[1]            
    idxf = q / dq + 1             
    i    = clamp(Int(floor(idxf)), 1, N-1)
    t    = idxf - i             
    return I[i]*(1-t) + I[i+1]*t
end

for (K, Qsym, Isym, Inormsym) in (
    (M4_spline,   :_M4_spline_Q,   :_M4_spline_I, _M4_spline_Inorm),
    (M5_spline,   :_M5_spline_Q,   :_M5_spline_I, _M5_spline_Inorm),
    (M6_spline,   :_M6_spline_Q,   :_M6_spline_I, _M6_spline_Inorm),
    (C2_Wendland, :_C2_Wendland_Q, :_C2_Wendland_I, _C2_Wendland_Inorm),
    (C4_Wendland, :_C4_Wendland_Q, :_C4_Wendland_I, _C4_Wendland_Inorm),
    (C6_Wendland, :_C6_Wendland_Q, :_C6_Wendland_I, _C6_Wendland_Inorm),
)
@eval @inline lookup_LOS(::Type{$K}, q::T) where {T<:AbstractFloat} =
    _lin_lut(T(q), $Qsym, $Inormsym,)
end

"""
    LOSint_Smoothed_kernel_function_dimensionless(
        ::Type{K},
        qxy::T,
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

Evaluate the **dimensionless line-of-sight (LOS) integrated SPH kernel**
for a given transverse separation `qxy`.

This function returns the precomputed or tabulated value of the LOS-integrated
kernel shape function,

    I(q_xy) = ∫ w(√(q_xy² + q_z²)) dq_z ,

where `q_xy = r_xy / h` is the dimensionless projected distance perpendicular
to the line of sight. The integration is truncated at the kernel support radius.

If `qxy` exceeds the kernel support (`qxy ≥ q_max`), the function returns zero.

# Parameters
- `::Type{K}`  
  SPH kernel type, where `K <: AbstractSPHKernel`.
- `qxy::T`  
  Dimensionless projected (transverse) separation, `qxy = r_xy / h`.

# Returns
- `T`  
  Dimensionless LOS-integrated kernel value.

# Notes
- This function is **dimensionless** and does not include any physical
  prefactors involving `h`.
- The support cutoff is determined by `KernelFunctionValid(K, T)`.
- Intended for use in column-density or projected-quantity calculations.
- Performs no heap allocation and is suitable for hot loops.
"""
@inline function LOSint_Smoothed_kernel_function_dimensionless(::Type{K}, qxy::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    qxy ≥ KernelFunctionValid(K, T) && return zero(T)
    Iq = lookup_LOS(K, qxy)
    return Iq
end

"""
    LOSint_Smoothed_kernel_function(
        ::Type{<:AbstractSPHKernel},
        r::T,
        h::T
    ) where {T<:AbstractFloat}

Evaluate the **line-of-sight (LOS) integrated SPH kernel** at a projected
(semi-2D) distance `r`.

This function evaluates the kernel obtained by **integrating a 3D SPH kernel
along the line-of-sight direction (z)**, i.e.

    I(r_xy, h) = ∫ W(√(r_xy² + z²), see h) dz ,

where `r = r_xy` is the distance in the plane perpendicular to the LOS.
Internally, the computation is performed using the dimensionless projected
coordinate `qxy = r / h`, followed by the appropriate physical scaling.

⚠️ **Important:**  
This function is only valid for quantities that are explicitly defined as
**integrals along the z-direction** (e.g. column density, projected mass,
LOS-integrated scalar fields). It must **not** be used for full 3D kernel
evaluations or force/gradient calculations.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  SPH kernel type.
- `r::T`  
  Projected (transverse) distance in the plane orthogonal to the line of sight
  (i.e. √(dx² + dy²)).
- `h::T`  
  Smoothing length.

# Returns
- `T`  
  LOS-integrated kernel value with physical units (includes the `1/h` scaling).

"""
@inline function LOSint_Smoothed_kernel_function(::Type{K}, r::T, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    invh = inv(h)
    qxy = r * invh
    I_dimless = LOSint_Smoothed_kernel_function_dimensionless(K, qxy)
    return invh * I_dimless
end

@inline function LOSint_Smoothed_kernel_function(::Type{K}, r::T, h::S) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat}
    rp, hp = promote(r, h)
    return LOSint_Smoothed_kernel_function(K, rp, hp)
end

"""
    LOSint_Smoothed_kernel_function(
        ::Type{<:AbstractSPHKernel},
        ra::NTuple{2,T},
        rb::NTuple{2,T},
        h::T
    ) where {T<:AbstractFloat}

Evaluate the **line-of-sight (LOS) integrated SPH kernel** using projected
2D coordinates of two points.

This method computes the projected separation

    r = √((xₐ − x_b)² + (yₐ − y_b)²)

in the plane perpendicular to the line of sight, and evaluates the
LOS-integrated kernel value

    I(r, h) = ∫ W(√(r² + z²), h) dz .

⚠️ **Important:**  
This function is only valid for kernels that have been **explicitly integrated
along the z-direction**. The input coordinates `ra` and `rb` must therefore
represent **2D projected positions (x, y)**. Any z-coordinate information is
assumed to have already been integrated out and must not be supplied here.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  SPH kernel type.
- `ra::NTuple{2,T}`  
  Projected 2D position `(x, y)` of the evaluation point.
- `rb::NTuple{2,T}`  
  Projected 2D position `(x, y)` of the source particle.
- `h::T`  
  Smoothing length.

# Returns
- `T`  
  LOS-integrated kernel value with physical units (includes the `1/h` scaling).
"""
@inline function LOSint_Smoothed_kernel_function(::Type{K}, ra::NTuple{2,T}, rb::NTuple{2,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    rax, ray = ra
    rbx, rby = rb
    Δx = rax - rbx
    Δy = ray - rby
    r2 = Δx * Δx + Δy * Δy
    r = sqrt(r2)
    return LOSint_Smoothed_kernel_function(K, r, h) 
end

