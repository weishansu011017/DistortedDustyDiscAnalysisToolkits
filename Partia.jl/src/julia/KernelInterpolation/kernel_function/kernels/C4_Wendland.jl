# Type calling of function
struct C4_Wendland <: AbstractSPHKernel end
struct _dC4_Wendland <: AbstractSPHKernel end

# Defince parent type of deriviative kernel
parenttype(::Type{_dC4_Wendland}) = C4_Wendland

# Kernel Functions
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
```
"""
@inline KernelFunctionValid(::Type{C4_Wendland}, ::Type{T}) where {T<:AbstractFloat} = T(2.0)

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
@inline KernelFunctionnorm(::Type{C4_Wendland}, ::Val{1}, ::Type{T}) where {T<:AbstractFloat} = T(3) / T(4)
@inline KernelFunctionnorm(::Type{C4_Wendland}, ::Val{2}, ::Type{T}) where {T<:AbstractFloat} = T(9) / (T(4) * T(π))
@inline KernelFunctionnorm(::Type{C4_Wendland}, ::Val{3}, ::Type{T}) where {T<:AbstractFloat} = T(495) / (T(256) * T(π))

"""
    KernelFunctionDiff(::Type{<:AbstractSPHKernel}, q::T) where {T<:AbstractFloat}

Return the value of the derivative of the kernel function at dimensionless radius `q`.

# Examples
```julia
dw32 = KernelFunctionDiff(M4_spline, 0.7f0)  # Float32
dw64 = KernelFunctionDiff(M4_spline, 0.7)    # Float64
```
"""
@inline KernelFunctionDiff(::Type{C4_Wendland}, q :: T) where {T<:AbstractFloat} = _dC4_Wendland()(q)

"""
    KernelFunctionNneigh(::Type{<:AbstractSPHKernel}) -> Int

Return the typical number of neighbors associated with the kernel function.

# Examples
```julia
nneigh = KernelFunctionNneigh(M6_spline)
# → 112
```
"""
@inline KernelFunctionNneigh(::Type{C4_Wendland}) = 137