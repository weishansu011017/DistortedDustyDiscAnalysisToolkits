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
        ::Type{<:AbstractSPHKernel},
        r::T,
        h::T,
        ::Val{D},
    ) -> T

Return the **dimension-less line-of-sight integral**

    I(q_xy) = ∫_{-R}^{R} Ẇ( √(q_xy² + q_z²) ) dq_z        (1)

already multiplied by the normalization constant **Cᴰ** that belongs to
the *original* spatial dimension **D**.

* `D` is the kernel’s original dimension (before the LOS integration).  
  – `Val(3)` ⇒ a 3-D kernel projected onto 2-D  
  – `Val(2)` ⇒ a 2-D kernel projected onto 1-D  
* `r` – in-plane distance ‖rₐ − r_b‖  
* `h` – smoothing length (same precision as `r`)

The function returns zero if `q_xy = r/h` is outside the kernel support.

### Example
```julia
# Dimension-less LOS integral of M4 spline with Float32 precision
I = LOSint_Smoothed_kernel_function_dimensionless(
        M4_spline, 0.8f0, 0.1f0, Val(3))
```
"""
@inline function LOSint_Smoothed_kernel_function_dimensionless(::K, r::T, h::T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    qxy = r / h
    qxy ≥ KernelFunctionValid(K, T) && return zero(T)
    Iq = lookup_LOS(K, qxy)
    return Iq
end

"""
    LOSint_Smoothed_kernel_function(
      ::Type{<:AbstractSPHKernel},
      r::T,
      h::T,
      ::Val{D}
    ) -> T

Compute the **full** (dimensional) line-of-sight integrated SPH kernel:

    W_LOS(r,h) = h^(−D) · ∫ Ẇ(√(q_xy² + q_z²)) dq_z

where
- `q_xy = r / h`
- `R    = KernelFunctionValid(K, T)`
- `Ẇ(q) = K()(q)` is the dimensionless kernel shape
- `D`   is the compile-time dimension tag (`Val{D}`)

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  The kernel functor type (e.g. `M4_spline`, `C2_Wendland`).
- `r::T`  
  In-plane distance ‖rₐ − r_b‖.
- `h::T`  
  Smoothing length.
- `::Val{D}`  
  Dimension before integration (1, 2, or 3).

# Returns
- `T`  
  The LOS-integrated kernel value, including the factor `h^(−D)`, in precision `T`.

# Example
```julia
val = LOSint_Smoothed_kernel_function(
    C2_Wendland, 1.2f0, 0.5f0, Val(3)
)
```
"""
@inline function LOSint_Smoothed_kernel_function(::K, r::T, h::T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}

    inv_hDminus1 = inv(h^(D-1))         
    I_dimless = LOSint_Smoothed_kernel_function_dimensionless(K, r, h, Val(D))
    return inv_hDminus1 * I_dimless
end

@inline function LOSint_Smoothed_kernel_function(::K, r::T, h::S, d::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, D}
    rp, hp = promote(r, h)
    return LOSint_Smoothed_kernel_function(K, rp, hp, d)
end

@inline function LOSint_Smoothed_kernel_function(::K, ra::AbstractVector{T}, rb::AbstractVector{T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    r2 = zero(T)
    @inbounds for i in eachindex(ra, rb)
        Δ = ra[i] - rb[i]
        r2 += Δ * Δ
    end
    r = sqrt(r2)
    dim = length(ra) + 1
    inv_hDminus1 = inv(h^(dim-1))         
    I_dimless = LOSint_Smoothed_kernel_function_dimensionless(K, r, h, Val(dim))
    return inv_hDminus1 * I_dimless
end

@inline function LOSint_Smoothed_kernel_function(::K, ra::AbstractVector{T}, rb::AbstractVector{S}, h::R) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, R<:AbstractFloat}
    P = promote_type(T, S, R)
    rap = P.(ra)    
    rbp = P.(rb)
    hp  = P(h)
    return LOSint_Smoothed_kernel_function(K, rap, rbp, hp)
end

@inline function LOSint_Smoothed_kernel_function(::K, ra::NTuple{D,T}, rb::NTuple{D,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    r2 = zero(T)
    @inbounds for i in eachindex(ra, rb)
        Δ = ra[i] - rb[i]
        r2 += Δ * Δ
    end
    r = sqrt(r2)
    inv_hDminus1 = inv(h^(D))         
    I_dimless = LOSint_Smoothed_kernel_function_dimensionless(K, r, h, Val(D + 1))
    return inv_hDminus1 * I_dimless
end

@inline function LOSint_Smoothed_kernel_function(::K, ra::NTuple{D,T}, rb::NTuple{D,S}, h::R) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, R<:AbstractFloat, D}
    P = promote_type(T, S, R)
    rap = ntuple(i -> P(ra[i]), D)
    rbp = ntuple(i -> P(rb[i]), D)
    hp  = P(h)
    return LOSint_Smoothed_kernel_function(K, rap, rbp, hp)
end
