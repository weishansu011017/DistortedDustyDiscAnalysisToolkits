"""
The Kernel function for SPH interpolation calculation.
    by Wei-Shan Su,
    June 21, 2024

The Kernel function is given as following (Price2018)

    W(r-r_a,h) = (Cnorm/h^d)f(q)

where q = |r-r_a|/h, d is the dimention of Kernel function, h is the smoothed radius(dimensional-depended),
and f(q) is the non-normalized Kernel function which would be refered as "Kernel function(Non-normalized)".

IMPORTANT: Personally, I will NOT recommend using M4 spline function since M4 may not have enough accuracy while estimating result.

# Structure:
    ## Kernel function(Non-normalized)
        M4 B-spline
        differentiated M4 B-spline

        M5 B-spline
        differentiated M5 B-spline

        M6 B-spline
        differentiated M6 B-spline

        Wendland C2
        differentiated Wendland C2

        Wendland C4
        differentiated Wendland C4

        Wendland C6
        differentiated Wendland C6

    ## Calculating influence by Smoothed Function
        Smoothed_kernel_function (2 Method): Calculating W(r-r_a, h).
        Smoothed_gradient_kernel_function (2 Method): Calculating ∇W(r-r_a, h).
        Smoothed_dh_kernel_function (2 Method): Calculating
        
"""

"""
    abstract type AbstractSPHKernel

An abstract type for collecting all kernel function for Smoothed particles hydrodynamics.
"""
abstract type AbstractSPHKernel end

# Type calling of function
struct M4_spline <: AbstractSPHKernel end
struct _dM4_spline <: AbstractSPHKernel end
struct M5_spline <: AbstractSPHKernel end
struct _dM5_spline <: AbstractSPHKernel end
struct M6_spline <: AbstractSPHKernel end
struct _dM6_spline <: AbstractSPHKernel end
struct C2_Wendland <: AbstractSPHKernel end
struct _dC2_Wendland <: AbstractSPHKernel end
struct C4_Wendland <: AbstractSPHKernel end
struct _dC4_Wendland <: AbstractSPHKernel end
struct C6_Wendland <: AbstractSPHKernel end
struct _dC6_Wendland <: AbstractSPHKernel end

# Defince parent type of deriviative kernel
parenttype(::Type{_dM4_spline}) = M4_spline
parenttype(::Type{_dM5_spline}) = M5_spline
parenttype(::Type{_dM6_spline}) = M6_spline
parenttype(::Type{_dC2_Wendland}) = C2_Wendland
parenttype(::Type{_dC4_Wendland}) = C4_Wendland
parenttype(::Type{_dC6_Wendland}) = C6_Wendland

# Kernel Functions
# M4 B-spline
@inline function (::M4_spline)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if zero(T) <= q < T(1)
        return (T(0.25) * (T(2) - q)^3 - (T(1) - q)^3)
    elseif T(1) <= q < T(2)
        return (T(0.25) * (T(2) - q)^3)
    else 
        return zero(T)
    end
end

@inline function (::_dM4_spline)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if zero(T) <= q < T(1)
        return (T(-0.75) * (T(2) - q)^2 + T(3) * (T(1) - q)^2)
    elseif T(1) <= q < T(2)
        return (T(-0.75) * (T(2) - q)^2)
    else
        return zero(T)
    end
end

# M5 B-spline
@inline function (::M5_spline)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if zero(T) <= q < T(0.5)
        return ((T(2.5) - q)^4 - T(5) * (T(1.5) - q)^4 + T(10) * (T(0.5) - q)^4)
    elseif T(0.5) <= q < T(1.5)
        return ((T(2.5) - q)^4 - T(5) * (T(1.5) - q)^4)
    elseif T(1.5) <= q < T(2.5)
        return ((T(2.5) - q)^4)
    else
        return zero(T)
    end
end

@inline function (::_dM5_spline)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if zero(T) <= q < T(0.5)
        return (T(-4) * (T(2.5) - q)^3 + T(20) * (T(1.5) - q)^3 - T(40) * (T(0.5) - q)^3)
    elseif T(0.5) <= q < T(1.5)
        return (-T(4) * (T(2.5) - q)^3 + T(20) * (T(1.5) - q)^3)
    elseif T(1.5) <= q < T(2.5)
        return (T(-4) * (T(2.5) - q)^3)
    else
        return zero(T)
    end
end

# M6 B-spline
@inline function (::M6_spline)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if zero(T) <= q < T(1)
        return ((T(3) - q)^5 - T(6) * (T(2) - q)^5 + T(15) * (T(1) - q)^5)
    elseif T(1) <= q < T(2)
        return ((T(3) - q)^5 - T(6) * (T(2) - q)^5)
    elseif T(2) <= q < T(3)
        return ((T(3) - q)^5)
    else
        return zero(T)
    end
end

@inline function (::_dM6_spline)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if zero(T) <= q < T(1)
        return (T(-5) * (T(3) - q)^4 + T(30) * (T(2) - q)^4 - T(75) * (T(1) - q)^4)
    elseif T(1) <= q < T(2)
        return (T(-5) * (T(3) - q)^4 + T(30) * (T(2) - q)^4)
    elseif T(2) <= q < T(3)
        return (T(-5) * (T(3) - q)^4)
    else
        return zero(T)
    end
end

# Wendland C2
@inline function (::C2_Wendland)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if q < T(2)
        return (((T(1) - T(0.5) * q)^4) * (T(2) * q + T(1)))
    else
        return zero(T)
    end
end

@inline function (::_dC2_Wendland)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if q < T(2)
        return (((T(1) - T(0.5) * q)^4) * T(2) - ((T(1) - T(0.5) * q)^3) * (T(4) * q + T(2)))
    else
        return zero(T)
    end
end

# Wendland C4
@inline function (::C4_Wendland)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if q < T(2)
        return (((T(1) - T(0.5) * q)^6) * ((T(35) / T(12)) * (q^2) + T(3) * q + T(1)))
    else
        return zero(T)
    end
end

@inline function (::_dC4_Wendland)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if q < T(2)
        return (
            ((T(1) - T(0.5) * q)^6) * ((T(35) / T(6)) * q + T(3)) -
            ((T(1) - T(0.5) * q)^5) * ((T(35) / T(4)) * (q^2) + T(9) * q + T(3))
        )
    else
        return zero(T)
    end
end

# Wendland C6
@inline function (::C6_Wendland)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if q < T(2)
        return (((T(1) - T(0.5) * q)^8) * (T(4) * (q^3) + T(6.25) * (q^2) + T(4) * q + T(1)))
    else
        return zero(T)
    end
end

@inline function (::_dC6_Wendland)(q :: T) where {T<:AbstractFloat}
    if q < zero(T)
        return T(NaN)
    end
    if q < T(2)
        return (
            ((T(1) - T(0.5) * q)^8) * (T(12) * (q^2) + T(12.5) * q + T(4)) -
            ((T(1) - T(0.5) * q)^7) * (T(16) * (q^3) + T(25) * (q^2) + T(16) * q + T(4))
        )
    else
        return zero(T)
    end
end

# Function constant
"""
    KernelFunctionValid(::Type{<:AbstractSPHKernel}, ::Type{T}) where {T<:AbstractFloat} -> T

Return the support radius (in units of the smoothing length `h`) for the given
SPH kernel type, cast to floating‐point precision `T`.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  The kernel functor type (e.g. `M4_spline`, `C2_Wendland`, etc.).
- `::Type{T}`  
  Desired output precision (`Float32` or `Float64`).

# Returns
- `T`  
  The support radius of the kernel (in units of `h`), converted to type `T`.

# Examples
```julia
julia> KernelFunctionValid(M5_spline, Float64)
2.5

julia> KernelFunctionValid(C2_Wendland, Float32)
2.0f0
````
"""
@inline KernelFunctionValid(::Type{M4_spline}, ::Type{T}) where {T<:AbstractFloat} = T(2.0)
@inline KernelFunctionValid(::Type{M5_spline}, ::Type{T}) where {T<:AbstractFloat} = T(2.5)
@inline KernelFunctionValid(::Type{M6_spline}, ::Type{T}) where {T<:AbstractFloat} = T(3.0)
@inline KernelFunctionValid(::Type{C2_Wendland}, ::Type{T}) where {T<:AbstractFloat} = T(2.0)
@inline KernelFunctionValid(::Type{C4_Wendland}, ::Type{T}) where {T<:AbstractFloat} = T(2.0)
@inline KernelFunctionValid(::Type{C6_Wendland}, ::Type{T}) where {T<:AbstractFloat} = T(2.0)
@inline KernelFunctionValid(::Type{K}) where {K<:AbstractSPHKernel} = KernelFunctionValid(parenttype(K))

"""
    KernelFunctionnorm(
      ::Type{<:AbstractSPHKernel},
      ::Val{D},
      ::Type{T}
    ) -> T

Return the normalization constant for the given SPH kernel type in `D` dimensions,
expressed in floating-point precision `T<:AbstractFloat`.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  The SPH kernel functor type (e.g. `M4_spline`, `C2_Wendland`).
- `::Val{D}`  
  A compile-time dimension tag (`Val(1)`, `Val(2)`, or `Val(3)`).
- `::Type{T}`  
  The desired output precision (`Float32` or `Float64`).

# Returns
- `T`  
  The normalization constant of the kernel in `D` dimensions, cast to type `T`.

# Examples
```julia
c32 = KernelFunctionnorm(C2_Wendland, Val(2), Float32)
# → 7f0 / (4f0 * πf0)

c64 = KernelFunctionnorm(M4_spline, Val(3), Float64)
# → 1.0 / π
```
"""
@inline KernelFunctionnorm(::Type{M4_spline}, ::Val{1}, ::Type{T}) where {T<:AbstractFloat} = T(4) / T(3)
@inline KernelFunctionnorm(::Type{M4_spline}, ::Val{2}, ::Type{T}) where {T<:AbstractFloat} = T(10) / (T(7) * T(π))
@inline KernelFunctionnorm(::Type{M4_spline}, ::Val{3}, ::Type{T}) where {T<:AbstractFloat} = T(1) / T(π)
@inline KernelFunctionnorm(::Type{M5_spline}, ::Val{1}, ::Type{T}) where {T<:AbstractFloat} = T(1) / T(24)
@inline KernelFunctionnorm(::Type{M5_spline}, ::Val{2}, ::Type{T}) where {T<:AbstractFloat} = T(96) / (T(1199) * T(π))
@inline KernelFunctionnorm(::Type{M5_spline}, ::Val{3}, ::Type{T}) where {T<:AbstractFloat} = T(0.05) / T(π)
@inline KernelFunctionnorm(::Type{M6_spline}, ::Val{1}, ::Type{T}) where {T<:AbstractFloat} = T(1) / T(120)
@inline KernelFunctionnorm(::Type{M6_spline}, ::Val{2}, ::Type{T}) where {T<:AbstractFloat} = T(7) / (T(478) * T(π))
@inline KernelFunctionnorm(::Type{M6_spline}, ::Val{3}, ::Type{T}) where {T<:AbstractFloat} = T(1) / (T(120) * T(π))
@inline KernelFunctionnorm(::Type{C2_Wendland}, ::Val{1}, ::Type{T}) where {T<:AbstractFloat} = T(5) / T(8)
@inline KernelFunctionnorm(::Type{C2_Wendland}, ::Val{2}, ::Type{T}) where {T<:AbstractFloat} = T(7) / (T(4) * T(π))
@inline KernelFunctionnorm(::Type{C2_Wendland}, ::Val{3}, ::Type{T}) where {T<:AbstractFloat} = T(21) / (T(16) * T(π))
@inline KernelFunctionnorm(::Type{C4_Wendland}, ::Val{1}, ::Type{T}) where {T<:AbstractFloat} = T(3) / T(4)
@inline KernelFunctionnorm(::Type{C4_Wendland}, ::Val{2}, ::Type{T}) where {T<:AbstractFloat} = T(9) / (T(4) * T(π))
@inline KernelFunctionnorm(::Type{C4_Wendland}, ::Val{3}, ::Type{T}) where {T<:AbstractFloat} = T(495) / (T(256) * T(π))
@inline KernelFunctionnorm(::Type{C6_Wendland}, ::Val{1}, ::Type{T}) where {T<:AbstractFloat} = T(64) / T(55)
@inline KernelFunctionnorm(::Type{C6_Wendland}, ::Val{2}, ::Type{T}) where {T<:AbstractFloat} = T(78) / (T(28) * T(π))
@inline KernelFunctionnorm(::Type{C6_Wendland}, ::Val{3}, ::Type{T}) where {T<:AbstractFloat} = T(1365) / (T(512) * T(π))

"""
    KernelFunctionDiff(::Type{<:AbstractSPHKernel}, q::T) where {T<:AbstractFloat}

Return the value of the derivative of the kernel function at dimensionless radius `q`.

# Examples
```julia
dw32 = KernelFunctionDiff(M4_spline, 0.7f0)  # Float32
dw64 = KernelFunctionDiff(M4_spline, 0.7)    # Float64
```
"""
@inline KernelFunctionDiff(::Type{M4_spline}, q :: T) where {T<:AbstractFloat} = _dM4_spline()(q)
@inline KernelFunctionDiff(::Type{M5_spline}, q :: T) where {T<:AbstractFloat} = _dM5_spline()(q)
@inline KernelFunctionDiff(::Type{M6_spline}, q :: T) where {T<:AbstractFloat} = _dM6_spline()(q)
@inline KernelFunctionDiff(::Type{C2_Wendland}, q :: T) where {T<:AbstractFloat} = _dC2_Wendland()(q)
@inline KernelFunctionDiff(::Type{C4_Wendland}, q :: T) where {T<:AbstractFloat} = _dC4_Wendland()(q)
@inline KernelFunctionDiff(::Type{C6_Wendland}, q :: T) where {T<:AbstractFloat} = _dC6_Wendland()(q)

"""
    KernelFunctionNneigh(::Type{<:AbstractSPHKernel}) -> Int

Return the typical number of neighbors associated with the kernel function.

# Examples
```julia
nneigh = KernelFunctionNneigh(M6_spline)
# → 112
```
"""
@inline KernelFunctionNneigh(::Type{M4_spline}) = 57
@inline KernelFunctionNneigh(::Type{M5_spline}) = 113
@inline KernelFunctionNneigh(::Type{M6_spline}) = 112
@inline KernelFunctionNneigh(::Type{C2_Wendland}) = 92
@inline KernelFunctionNneigh(::Type{C4_Wendland}) = 137
@inline KernelFunctionNneigh(::Type{C6_Wendland}) = 356
@inline KernelFunctionNneigh(::Type{K}) where {K<:AbstractSPHKernel} = KernelFunctionNneigh(parenttype(K))


# Calculating influence by Smoothed Function
# Dimensionless Kernel
"""
    Smoothed_kernel_function_dimensionless(
        ::Type{<:AbstractSPHKernel},
        r::T,
        h::T,
        ::Val{D}
    ) where {T<:AbstractFloat, D}

Compute the **dimensionless** kernel value `w(q)` with `q = r/h``
for a given SPH kernel type in `D` dimensions.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  The kernel functor type (e.g. `M4_spline`, `C2_Wendland`).
- `r::T`  
  Euclidean distance between two particles, precision `T<:AbstractFloat`.
- `h::T`  
  Smoothing length (same precision as `r`).
- `::Val{D}`  
  Dimension tag: use `Val(1)`, `Val(2)`, or `Val(3)` to select 1D/2D/3D.

# Returns
- `T`  
  The dimensionless kernel value `w(q)` in the same precision as inputs.

# Examples
```julia
# 2D, Float32
w2f32 = Smoothed_kernel_function_dimensionless(M4_spline, 0.8f0, 0.1f0, Val(2))

# 3D, Float64
w3f64 = Smoothed_kernel_function_dimensionless(C2_Wendland, 1.2, 0.5, Val(3))
```
"""
@inline function Smoothed_kernel_function_dimensionless(::Type{K}, r :: T, h :: T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    q :: T = r / h
    return KernelFunctionnorm(K, Val(D), T) * K()(q)
end

@inline function Smoothed_kernel_function_dimensionless(::Type{K}, r :: T, h :: S, d ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, D}
    rp, hp = promote(r, h)
    return Smoothed_kernel_function_dimensionless(K, rp, hp, d)
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
@inline function Smoothed_kernel_function(::Type{K}, r::T, h::T, d::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    inv_hD = inv(h^D)
    return inv_hD * Smoothed_kernel_function_dimensionless(K, r, h, d)
end

@inline function Smoothed_kernel_function(::Type{K}, r::T, h::S, d::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, D}
    rp, hp = promote(r, h)
    return Smoothed_kernel_function(K, rp, hp, d)
end

"""
    Smoothed_kernel_function(
      ::Type{<:AbstractSPHKernel},
      ra::AbstractVector{T},
      rb::AbstractVector{T},
      h::T
    ) where {T<:AbstractFloat, D} -> T

Compute W(r_a-r_b,h) by first measuring (r_a-r_b) in `D`-dimensions.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  The kernel functor type (e.g. `M4_spline`, `C2_Wendland`).

- `ra, rb::AbstractVector{T}`  
  Position vectors of equal length `D`, with element type `T<:AbstractFloat`.

- `h::T`  
  Smoothing length (same precision `T`).

# Returns
- `T`  
  The kernel value w in the same precision as the inputs.

# Examples
```julia
ra = [0.0f0, 0.1f0, 0.2f0]
rb = [0.1f0, 0.0f0, 0.3f0]
h  = 0.05f0

Wval = Smoothed_kernel_function(M4_spline, ra, rb, h)
````
"""
@inline function Smoothed_kernel_function(::Type{K}, ra::AbstractVector{T}, rb::AbstractVector{T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    r2 = zero(T)
    @inbounds for i in eachindex(ra, rb)
        Δ = ra[i] - rb[i]
        r2 += Δ * Δ
    end
    r = sqrt(r2)
    dim = length(ra)
    return Smoothed_kernel_function(K, r, h, Val(dim))
end

@inline function Smoothed_kernel_function(::Type{K}, ra::AbstractVector{T}, rb::AbstractVector{S}, h::R) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, R<:AbstractFloat}
    P = promote_type(T, S, R)
    rap = P.(ra)    
    rbp = P.(rb)
    hp  = P(h)
    return Smoothed_kernel_function(K, rap, rbp, hp)
end

@inline function Smoothed_kernel_function(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    r2 = zero(T)
    @inbounds for i in eachindex(ra, rb)
        Δ = ra[i] - rb[i]
        r2 += Δ * Δ
    end
    r = sqrt(r2)
    return Smoothed_kernel_function(K, r, h, Val(D))
end

@inline function Smoothed_kernel_function(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,S}, h::R) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, R<:AbstractFloat, D}
    P = promote_type(T, S, R)
    rap = ntuple(i -> P(ra[i]), D)
    rbp = ntuple(i -> P(rb[i]), D)
    hp  = P(h)
    return Smoothed_kernel_function(K, rap, rbp, hp)
end





# ∇W(ra-rb,h)
@inline function _smoothed_grad_dimless!(out::AbstractVector{T}, ::Type{K}, rab::AbstractVector{T}, h::T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    r2 = zero(T)
    @inbounds for i in eachindex(rab)
        Δ = rab[i]
        r2 += Δ*Δ
    end
    r = sqrt(r2)
    q     = r / h
    coeff = KernelFunctionDiff(K, q) * KernelFunctionnorm(K, Val(D), T)

    @inbounds for i in eachindex(rab)
        out[i] = (rab[i] / r) * coeff
    end
    return out 
end

@inline function _smoothed_grad_dimless!(out::AbstractVector{T}, ::Type{K}, ra::AbstractVector{T}, rb::AbstractVector{T}, h::T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    r2 = zero(T)
    @inbounds for i in 1:D
        Δ = ra[i] - rb[i]
        r2 += Δ*Δ
    end
    r = sqrt(r2)
    q     = r / h
    coeff = KernelFunctionDiff(K, q) * KernelFunctionnorm(K, Val(D), T)

    @inbounds for i in eachindex(rab)
        out[i] = ((ra[i] - rb[i]) / r) * coeff
    end
    return out 
end

@inline function _smoothed_grad_dimless(::Type{K}, rab::NTuple{D,T}, h::T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    r2 = zero(T)
    @inbounds for i in 1:D
        r2 += rab[i]^2
    end
    r = sqrt(r2)
    q = r / h
    coeff = KernelFunctionDiff(K, q) * KernelFunctionnorm(K, Val(D), T)

    return ntuple(i -> (rab[i] / r) * coeff, D)
end

@inline function _smoothed_grad_dimless(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,T}, h::T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    r2 = zero(T)
    @inbounds for i in 1:D
        Δ = ra[i] - rb[i]
        r2 += Δ*Δ
    end
    r = sqrt(r2)
    q     = r / h
    coeff = KernelFunctionDiff(K, q) * KernelFunctionnorm(K, Val(D), T)

    return ntuple(i -> ((ra[i] - rb[i]) / r) * coeff, D)
end

"""
    Smoothed_gradient_kernel_function_dimensionless(
      ::Type{<:AbstractSPHKernel},
      rab::AbstractVector{T},
      h::T
    ) where {T<:AbstractFloat} -> Vector{T}

Compute the **dimensionless** gradient of the SPH kernel at a displacement `rab` 
and smoothing length `h`.

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  The kernel functor type (e.g. `M4_spline`, `C2_Wendland`).
- `rab::AbstractVector{T}`  
  Displacement vector r_a - r_b, with element type `T<:AbstractFloat`.
- `h::T`  
  Smoothing length (same precision as `rab`).

# Returns
- `Vector{T}`  
  A vector of the same length as `rab` containing w
  in the same floating-point precision.

# Examples
```julia
rab = [0.1f0, -0.2f0, 0.3f0]
h   = 0.05f0
grad_dimless = Smoothed_gradient_kernel_function_dimensionless(
    M4_spline, rab, h
)
```
"""
@inline function Smoothed_gradient_kernel_function_dimensionless(::Type{K}, rab::AbstractVector{T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    dim = length(rab)
    out = similar(rab)
    return _smoothed_grad_dimless!(out, K, rab, h, Val(dim))
end

@inline function Smoothed_gradient_kernel_function_dimensionless(::Type{K}, ra::AbstractVector{T}, rb::AbstractVector{T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    dim = length(ra)
    out = similar(ra)
    return _smoothed_grad_dimless!(out, K, ra, rb, h, Val(dim))
end

@inline function Smoothed_gradient_kernel_function_dimensionless(::Type{K}, rab::NTuple{D,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    return _smoothed_grad_dimless(K, rab, h, Val(D))
end

@inline function Smoothed_gradient_kernel_function_dimensionless(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    rab = ntuple(i -> ra[i] - rb[i], D)
    return _smoothed_grad_dimless(K, rab, h, Val(D))
end

"""
    Smoothed_gradient_kernel_function(
      ::Type{<:AbstractSPHKernel},
      rab::AbstractVector{T},
      h::T
    ) where {T<:AbstractFloat} -> Vector{T}

Compute the **full** SPH gradient for displacement `rab = ra - rb` and smoothing
length `h`.  The result is

    ∇W(r_ab, h) = h^{-(D+1)} · ĥ_{ab} · (dŴ/dq)(q),

where

- q = ‖rab‖ / h  
- ĥ_{ab} = rab / ‖rab‖  
- D = length(rab)

# Parameters
- `::Type{<:AbstractSPHKernel}`  
  The kernel functor type (e.g. `M4_spline`, `C2_Wendland`).
- `rab::AbstractVector{T}`  
  Displacement vector `ra - rb`, of element type `T<:AbstractFloat`.
- `h::T`  
  Smoothing length (same precision as `rab`).

# Returns
- `Vector{T}`  
  A vector of the same length as `rab`, containing the gradient ∇W(r_ab,h)
  in the same floating-point precision `T`.

# Examples
```julia
rab = [0.1, -0.2]      # Vector{Float64}
h   = 0.1
grad_full = Smoothed_gradient_kernel_function(
    C2_Wendland, rab, h)
```
"""
@inline function Smoothed_gradient_kernel_function(::Type{K}, rab::AbstractVector{T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    D = length(rab)
    inv_hDp1 = inv(h^(D+1))
    return inv_hDp1 * Smoothed_gradient_kernel_function_dimensionless(K, rab, h)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, rab::AbstractVector{T}, h::S) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat}
    P = promote_type(T, S)
    rabp = P.(rab)    
    hp  = P(h)
    return Smoothed_gradient_kernel_function(K, rabp, hp)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, ra::AbstractVector{T}, rb::AbstractVector{T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
    D = length(ra)
    inv_hDp1 = inv(h^(D+1))
    return inv_hDp1 * Smoothed_gradient_kernel_function_dimensionless(K, ra, rb, h)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, ra::AbstractVector{T}, rb::AbstractVector{S}, h::R) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, R<:AbstractFloat}
    P = promote_type(T, S, R)
    rap = P.(ra)    
    rbp = P.(rb)  
    hp  = P(h)
    return Smoothed_gradient_kernel_function(K, rap, rbp, hp)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, rab::NTuple{D,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    inv_hDp1 = inv(h^(D+1))
    ws = Smoothed_gradient_kernel_function_dimensionless(K, rab, h)
    return ntuple(i -> inv_hDp1 * ws[i], D)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, rab::NTuple{D,T}, h::S) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, D}
    P = promote_type(T, S)
    rabp = ntuple(i -> convert(P, rab[i]), D)
    hp   = convert(P, h)
    return Smoothed_gradient_kernel_function(K, rabp, hp)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
    inv_hDp1 = inv(h^(D+1))
    ws = Smoothed_gradient_kernel_function_dimensionless(K, ra, rb, h)
    return ntuple(i -> inv_hDp1 * ws[i], D)
end

@inline function Smoothed_gradient_kernel_function(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,S}, h::R) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, R<:AbstractFloat, D}
    P = promote_type(T, S, R)
    rap = ntuple(i -> convert(P, ra[i]), D)
    rbp = ntuple(i -> convert(P, rb[i]), D)
    hp   = convert(P, h)
    return Smoothed_gradient_kernel_function(K, rap, rbp, hp)
end


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

already multiplied by the normalisation constant **Cᴰ** that belongs to
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
@inline function LOSint_Smoothed_kernel_function_dimensionless(::Type{K}, r::T, h::T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
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
@inline function LOSint_Smoothed_kernel_function(::Type{K}, r::T, h::T, ::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}

    inv_hDminus1 = inv(h^(D-1))         
    I_dimless = LOSint_Smoothed_kernel_function_dimensionless(K, r, h, Val(D))
    return inv_hDminus1 * I_dimless
end

@inline function LOSint_Smoothed_kernel_function(::Type{K}, r::T, h::S, d::Val{D}) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, D}
    rp, hp = promote(r, h)
    return LOSint_Smoothed_kernel_function(K, rp, hp, d)
end

@inline function LOSint_Smoothed_kernel_function(::Type{K}, ra::AbstractVector{T}, rb::AbstractVector{T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat}
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

@inline function LOSint_Smoothed_kernel_function(::Type{K}, ra::AbstractVector{T}, rb::AbstractVector{S}, h::R) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, R<:AbstractFloat}
    P = promote_type(T, S, R)
    rap = P.(ra)    
    rbp = P.(rb)
    hp  = P(h)
    return LOSint_Smoothed_kernel_function(K, rap, rbp, hp)
end

@inline function LOSint_Smoothed_kernel_function(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,T}, h::T) where {K<:AbstractSPHKernel, T<:AbstractFloat, D}
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

@inline function LOSint_Smoothed_kernel_function(::Type{K}, ra::NTuple{D,T}, rb::NTuple{D,S}, h::R) where {K<:AbstractSPHKernel, T<:AbstractFloat, S<:AbstractFloat, R<:AbstractFloat, D}
    P = promote_type(T, S, R)
    rap = ntuple(i -> P(ra[i]), D)
    rbp = ntuple(i -> P(rb[i]), D)
    hp  = P(h)
    return LOSint_Smoothed_kernel_function(K, rap, rbp, hp)
end
