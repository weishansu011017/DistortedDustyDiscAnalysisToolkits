"""
The Kernel function for SPH interpolation calculation.
    by Wei-Shan Su,
    June 21, 2024

The Kernel function is given as following (Price2018)

    W(r-r_a,h) = (Cnorm/h^d)f(q)

where q = |r-r_a|/h, d is the dimention of Kernel function, h is the smoothed radius(dimensional-depended),
and f(q) is the non-normalized Kernel function which would be refered as "Kernel function(Non-normalized)".

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

@inline KernelFunctionValid(::Type{K}) where {K<:AbstractSPHKernel} = KernelFunctionValid(parenttype(K))
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


