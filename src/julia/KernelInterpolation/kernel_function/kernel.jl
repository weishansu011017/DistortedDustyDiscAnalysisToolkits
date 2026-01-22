"""
Kernel functions and smoothed forms for Smoothed Particle Hydrodynamics (SPH).
    by Wei-Shan Su,
    October 31, 2025

This module provides a unified and extensible framework for SPH kernel functions,
including both the dimensionless analytic forms and their smoothed (dimensional) 
representations, as well as gradient evaluations.

# Overview

For a given SPH kernel type `K <: AbstractSPHKernel`, the smoothing kernel is defined as:

    W(r, h) = h^{-D} · C_norm(D) · f(q),
    q = |r| / h,

where  
- `h` is the smoothing length,  
- `D` is the spatial dimension (1, 2, or 3),  
- `f(q)` is the **dimensionless kernel function**, and  
- `C_norm(D)` is the normalization constant ensuring ∫W = 1.

The kernel family and normalization constants follow *Price (2018, JCoPh, 378, 257)*.


# Extending the Kernel Set

To implement a new kernel type, define the following components:

1. **Type definition**
   ```julia
   struct MyKernel <: AbstractSPHKernel end
   struct _dMyKernel <: AbstractSPHKernel end
   parenttype(::Type{_dMyKernel}) = 
   ```
2. **Kernel shape and derivative**
   ```julia
   (::MyKernel)(q::T) where {T<:AbstractFloat} = ...
   (::_dMyKernel)(q::T) where {T<:AbstractFloat} = ...
   ```

3. **Support radius and normalization**
   ```julia
   KernelFunctionValid(::Type{MyKernel}, ::Type{T}) where {T<:AbstractFloat} = ...
   KernelFunctionnorm(::Type{MyKernel}, ::Val{D}, ::Type{T}) where {T<:AbstractFloat} = ...
   ```

4. **Typical neighbor count**
   ```julia
   KernelFunctionNneigh(::Type{MyKernel}) = ...
   ```
Once defined, all smoothed and gradient functions such as
`Smoothed_kernel_function` and `Smoothed_gradient_kernel_function`
will automatically work with the new kernel type.
"""

"""
    abstract type AbstractSPHKernel

An abstract type for collecting all kernel function for Smoothed particles hydrodynamics.
"""
abstract type AbstractSPHKernel end

@inline KernelFunctionValid(::Type{K}) where {K<:AbstractSPHKernel} = KernelFunctionValid(parenttype(K))
@inline KernelFunctionNneigh(::Type{K}) where {K<:AbstractSPHKernel} = KernelFunctionNneigh(parenttype(K))

# Calculating influence by Smoothed Function
# Dimensionless Kernel
"""
    Smoothed_kernel_function_dimensionless(
        ::Type{K},
        q::T,
        ::Val{D}
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}

Compute the **dimensionless** SPH smoothing kernel value `w(q)` for a given kernel type,
where the dimensionless separation is

```math
q = r/h .
```
This function returns the normalised dimensionless kernel value in D dimensions,
i.e. the kernel shape K()(q) multiplied by the dimension-dependent normalisation factor.

# Parameters
	- ::Type{K}
Kernel type (e.g. M4_spline, C2_Wendland) with K <: AbstractSPHKernel.
	- q::T
Dimensionless distance q = r/h, with T <: AbstractFloat.
	- ::Val{D}
Dimension tag. Use Val(1), Val(2), or Val(3).

Returns
	- :: T
The dimensionless kernel value w(q) (normalised for D dimensions).
"""
@inline function Smoothed_kernel_function_dimensionless(::Type{K}, q :: T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    return KernelFunctionnorm(K, Val(D), T) * K()(q)
end


# Dimensional Kernel
"""
    Smoothed_kernel_function(
      ::Type{<:AbstractSPHKernel},
      r::T,
      h::T,
      ::Val{D}
    ) where {T<:AbstractFloat, D} -> T

Compute the SPH kernel  

  W(r,h) = h^{-D} w(q)
  q = r/h,

in `D` dimensions, where w is the dimensionless kernel.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  The kernel functor type (e.g. `M4_spline`, `C2_Wendland`).
- `r::T`  
  Distance between two particles, precision `T<:AbstractFloat`.
- `h::T`  
  Smoothing length (same precision as `r`).
- `::Val{D}`  
  Dimension tag: use `Val(1)`, `Val(2)`, or `Val(3)`.

# Returns
- `T`  
  The kernel value W(r,h) in the same precision as the inputs.

# Examples
```julia
# 2D, Float32
W2f32 = Smoothed_kernel_function(M4_spline, 0.8f0, 0.1f0, Val(2))

# 3D, Float64
W3f64 = Smoothed_kernel_function(C2_Wendland, 1.2, 0.5, Val(3))
```
"""
@inline function Smoothed_kernel_function(::Type{K}, r::T, h::T, ::Val{1}) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    invh = inv(h)
    q = r * invh
    return invh * Smoothed_kernel_function_dimensionless(K, q, Val(1))
end
@inline function Smoothed_kernel_function(::Type{K}, r::T, h::T, ::Val{2}) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    invh = inv(h)
    inv_hD = invh * invh
    q = r * invh
    return inv_hD * Smoothed_kernel_function_dimensionless(K, q, Val(2))
end
@inline function Smoothed_kernel_function(::Type{K}, r::T, h::T, ::Val{3}) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    invh = inv(h)
    inv_hD = invh * invh * invh
    q = r * invh
    return inv_hD * Smoothed_kernel_function_dimensionless(K, q, Val(3))
end

@inline function Smoothed_kernel_function(::Type{K}, r::T, h::S, d::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, D}
    rp, hp = promote(r, h)
    return Smoothed_kernel_function(K, rp, hp, d)
end

"""
    Smoothed_kernel_function(
        ::Type{K},
        ra::NTuple{D,T},
        rb::NTuple{D,T},
        h::T
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}

Compute the **physical** SPH smoothing kernel value `W(|ra-rb|, h)` in `D` dimensions
from two coordinate tuples.

This method is a thin wrapper that:
1. computes the Euclidean distance `r = |ra - rb|` from `ra` and `rb`, and
2. dispatches to the scalar-distance kernel implementation
   `Smoothed_kernel_function(K, r, h, Val(D))`.

# Parameters
- `::Type{K}`  
  Kernel type with `K <: AbstractSPHKernel` (e.g. `M4_spline`, `C2_Wendland`).
- `ra::NTuple{D,T}`  
  Coordinate of point/particle `a` in `D` dimensions.
- `rb::NTuple{D,T}`  
  Coordinate of point/particle `b` in `D` dimensions.
- `h::T`  
  Smoothing length.

# Returns
- `T`  
  The kernel value `W(|ra-rb|, h)` in `D` dimensions.
"""
@inline function Smoothed_kernel_function(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    r2 = zero(T)
    @inbounds for i in eachindex(ra, rb)
        Δ = ra[i] - rb[i]
        r2 += Δ * Δ
    end
    r = sqrt(r2)
    return Smoothed_kernel_function(K, r, h, Val(D))
end

# ∇W(ra-rb,h)
"""
    Smoothed_gradient_kernel_function_dimensionless(
        ::Type{K},
        Δx::T,
        h::T
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

    Smoothed_gradient_kernel_function_dimensionless(
        ::Type{K},
        Δx::T, Δy::T,
        h::T
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

    Smoothed_gradient_kernel_function_dimensionless(
        ::Type{K},
        Δx::T, Δy::T, Δz::T,
        h::T
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

Compute the **dimensionless gradient** of an SPH kernel using scalar displacement
components in 1D, 2D, or 3D.

Let `rab = ra - rb` with components `(Δx, Δy, Δz)` and define the dimensionless
distance `q = |rab| / h`.  
This function evaluates the gradient of the dimensionless kernel shape function,
including the dimension-dependent normalisation factor.

The returned value corresponds to

    ∇w(q) = (dw/dq) * (rab / |rab|)

scaled by `KernelFunctionnorm(K, Val(D), T)`, but **does not** include the outer
physical prefactor `1 / h^D`. That scaling should be applied by higher-level
routines when constructing the full kernel gradient.

These scalar-component APIs are intended for extremely hot loops and avoid any
container allocation.

# Parameters
- `::Type{K}`  
  SPH kernel type, where `K <: AbstractSPHKernel`.
- `Δx`, `Δy`, `Δz`  
  Displacement components between two particles.
- `h`  
  Smoothing length.

# Returns
- 1D: `T`
- 2D: `NTuple{2,T}`
- 3D: `NTuple{3,T}`

Dimensionless gradient components. If the separation is zero, all components
are returned as zero.
"""
@inline function Smoothed_gradient_kernel_function_dimensionless(::Type{K}, Δx :: T, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    if iszero(Δx)
      return zero(T)
    end
    r     = Δx
    q     = r / h
    coeff = KernelFunctionDiff(K, q) * KernelFunctionnorm(K, Val(1), T) / r

    return Δx * coeff
end

@inline function Smoothed_gradient_kernel_function_dimensionless(::Type{K}, Δx :: T, Δy :: T, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    r2 = Δx * Δx + Δy * Δy
    if iszero(r2)
      zeroT = zero(T)
      return (zeroT, zeroT)
    end
    r     = sqrt(r2)
    invr  = inv(r)
    q     = r / h
    coeff = KernelFunctionDiff(K, q) * KernelFunctionnorm(K, Val(2), T) * invr

    return (Δx * coeff, Δy * coeff)
end

@inline function Smoothed_gradient_kernel_function_dimensionless(::Type{K}, Δx :: T, Δy :: T, Δz :: T, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    r2 = Δx * Δx + Δy * Δy + Δz * Δz
    if iszero(r2)
      zeroT = zero(T)
      return (zeroT, zeroT, zeroT)
    end
    r     = sqrt(r2)
    invr  = inv(r)
    q     = r / h
    coeff = KernelFunctionDiff(K, q) * KernelFunctionnorm(K, Val(3), T) * invr

    return (Δx * coeff, Δy * coeff, Δz * coeff)
end

"""
    Smoothed_gradient_kernel_function(
        ::Type{<:AbstractSPHKernel},
        Δx::T, [Δy::T, Δz::T], h::T
    ) where {T<:AbstractFloat}

Compute the **physical (dimensionful) gradient of the SPH smoothing kernel**
\\( \\nabla W(\\mathbf{r}, h) \\) in 1D, 2D, or 3D, using **scalar displacement components**
as input.

This is the **performance-critical, allocation-free API** intended for hot loops
(e.g. neighbor interactions, grid interpolation, GPU kernels).

The function internally evaluates the **dimensionless kernel gradient**
\\( \\nabla w(q) \\) with \\( q = r/h \\), and applies the correct dimensional
scaling:

\\[
\\nabla W = h^{-(D+1)} \\, \\nabla w
\\]

where `D` is the spatial dimension.

---

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  SPH kernel type (e.g. `M4_spline`, `C2_Wendland`).

- `Δx::T`, `Δy::T`, `Δz::T`  
  Cartesian displacement components between two particles:
  - 1D: `Δx`
  - 2D: `(Δx, Δy)`
  - 3D: `(Δx, Δy, Δz)`

- `h::T`  
  Smoothing length.

All arguments must share the same floating-point type `T`.

# Returns
- **1D:** `T`  
- **2D:** `(T, T)`  
- **3D:** `(T, T, T)`  

The physical kernel gradient components in each dimension.

# Examples
```julia
# 1D
dWdx = Smoothed_gradient_kernel_function(M4_spline, Δx, h)

# 2D
dWdx, dWdy = Smoothed_gradient_kernel_function(M4_spline, Δx, Δy, h)

# 3D
dWdx, dWdy, dWdz = Smoothed_gradient_kernel_function(M4_spline, Δx, Δy, Δz, h)
```
"""
@inline function Smoothed_gradient_kernel_function(::Type{K}, Δx :: T, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    invh = inv(h)
    inv_hDp1 = invh * invh
    ws = Smoothed_gradient_kernel_function_dimensionless(K, Δx, h)
    return inv_hDp1 * ws
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, Δx :: T, Δy :: T, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    invh = inv(h)
    inv_hDp1 = invh * invh * invh
    gx, gy = Smoothed_gradient_kernel_function_dimensionless(K, Δx, Δy, h)
    return (inv_hDp1 * gx, inv_hDp1 * gy)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, Δx :: T, Δy :: T, Δz :: T, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    invh = inv(h)
    inv_hDp1 = invh * invh * invh * invh
    gx, gy, gz = Smoothed_gradient_kernel_function_dimensionless(K, Δx, Δy, Δz, h)
    return (inv_hDp1 * gx, inv_hDp1 * gy, inv_hDp1 * gz)
end

"""
    Smoothed_gradient_kernel_function(
        ::Type{K},
        rab::NTuple{2,T},
        h::T
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

    Smoothed_gradient_kernel_function(
        ::Type{K},
        rab::NTuple{3,T},
        h::T
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

Compute the **physical SPH kernel gradient** ∇W using a displacement tuple
`rab = ra - rb` in 2D or 3D.

This is a **thin convenience wrapper** that destructures the displacement tuple
into scalar components and forwards the computation to the corresponding
scalar-based kernel implementation. No additional arithmetic, memory allocation,
or approximation is introduced at this level.

Internally, the gradient is evaluated as

    ∇W = (1 / h^{D+1}) ∇w(q),   with q = |rab| / h

where `∇w(q)` is the dimensionless kernel gradient and `D` is the spatial
dimension.

# Parameters
- `::Type{K}`  
  SPH kernel type, where `K <: AbstractSPHKernel`.
- `rab::NTuple{D,T}`  
  Displacement vector between two particles (`ra - rb`), with `D = 2` or `3`.
- `h::T`  
  Smoothing length.

# Returns
- `NTuple{D,T}`  
  Physical kernel gradient components in `D` dimensions.
"""
@inline function Smoothed_gradient_kernel_function(::Type{K}, rab::NTuple{2,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    Δx, Δy = rab
    return Smoothed_gradient_kernel_function(K, Δx, Δy, h)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, rab::NTuple{3,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    Δx, Δy, Δz = rab
    return Smoothed_gradient_kernel_function(K, Δx, Δy, Δz, h)
end

"""
    Smoothed_gradient_kernel_function(
        ::Type{K},
        ra::NTuple{2,T},
        rb::NTuple{2,T},
        h::T
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

    Smoothed_gradient_kernel_function(
        ::Type{K},
        ra::NTuple{3,T},
        rb::NTuple{3,T},
        h::T
    ) where {K<:AbstractSPHKernel, T<:AbstractFloat}

Compute the **physical SPH kernel gradient** ∇W between two particle positions
`ra` and `rb` in 2D or 3D.

This function computes the displacement `rab = ra - rb` component-wise and
forwards the calculation to the scalar-based kernel gradient implementation.
It introduces no additional approximation or memory allocation beyond the
coordinate subtraction.

Internally, the gradient is evaluated as

    ∇W(ra - rb, h) = (1 / h^{D+1}) ∇w(q),   with q = |ra - rb| / h

where `∇w(q)` is the dimensionless kernel gradient and `D` is the spatial
dimension.

# Parameters
- `::Type{K}`  
  SPH kernel type, where `K <: AbstractSPHKernel`.
- `ra::NTuple{D,T}`  
  Position of particle *a*.
- `rb::NTuple{D,T}`  
  Position of particle *b*.
- `h::T`  
  Smoothing length.

# Returns
- `NTuple{D,T}`  
  Physical kernel gradient components in `D` dimensions.

"""
@inline function Smoothed_gradient_kernel_function(::Type{K}, ra::NTuple{2,T}, rb::NTuple{2,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    rax, ray = ra
    rbx, rby = rb
    Δx = rax - rbx
    Δy = ray - rby
    return Smoothed_gradient_kernel_function(K, Δx, Δy, h)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, ra::NTuple{3,T}, rb::NTuple{3,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    rax, ray, raz = ra
    rbx, rby, rbz = rb
    Δx = rax - rbx
    Δy = ray - rby
    Δz = raz - rbz
    return Smoothed_gradient_kernel_function(K, Δx, Δy, Δz, h)
end

