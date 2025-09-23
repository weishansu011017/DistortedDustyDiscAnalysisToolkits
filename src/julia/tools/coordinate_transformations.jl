"""
Coordinate transformations modules.
    by Wei-Shan Su,
    September 24, 2025
"""

# Cartesian ⟹ Cylindrical/Polar
# Coordinate transform 
@inline function _cart2cylin(x :: T, y :: T) where {T<:AbstractFloat}
    s = sqrt(x*x + y*y)
    ϕ = mod(atan(y, x), 2π)
    return (T(s), T(ϕ))
end

@inline function _cart2cylin(x :: T, y :: T, z :: T) where {T<:AbstractFloat}
    s = sqrt(x*x + y*y)
    ϕ = mod(atan(y, x), 2π)
    return (T(s), T(ϕ), T(z))
end

@inline function _cart2cylin(point::NTuple{2, T}) where {T<:AbstractFloat}
    x, y = point
    return _cart2cylin(x, y)
end

@inline function _cart2cylin(point::NTuple{3, T}) where {T<:AbstractFloat}
    x, y, z = point
    return _cart2cylin(x, y, z)
end

@inline function _cart2cylin(point::AbstractVector{<:AbstractFloat})
    x, y = @inbounds point[1], point[2]
    if length(point) > 2
        return _cart2cylin(x, y, point[3])
    else
        return _cart2cylin(x, y)
    end
end

# Vector transform 
@inline function _vector_cart2cylin(ϕ::T, Ax::T, Ay::T) where {T<:AbstractFloat}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ*Ax + sinϕ*Ay),
            T(-sinϕ*Ax + cosϕ*Ay))
end

@inline function _vector_cart2cylin(ϕ::T, Ax::T, Ay::T, Az::T) where {T<:AbstractFloat}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ*Ax + sinϕ*Ay),
            T(-sinϕ*Ax + cosϕ*Ay),
            T(Az))
end

@inline function _vector_cart2cylin(ϕ::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    Ax, Ay = @inbounds A[1], A[2]
    if length(A) > 2
        return _vector_cart2cylin(ϕ, Ax, Ay, A[3])
    else
        return _vector_cart2cylin(ϕ, Ax, Ay)
    end
end

@inline function _vector_cart2cylin(ϕ::T, A::NTuple{2, T}) where {T<:AbstractFloat}
    Ax, Ay = @inbounds A[1], A[2]
    return _vector_cart2cylin(ϕ, Ax, Ay)
end

@inline function _vector_cart2cylin(ϕ::T, A::NTuple{3, T}) where {T<:AbstractFloat}
    Ax, Ay, Az = @inbounds A[1], A[2], A[3]
    return _vector_cart2cylin(ϕ, Ax, Ay, Az)
end

@inline function _vector_cart2cylin(x::T, y::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    ϕ = mod(atan(y, x), 2π)
    return _vector_cart2cylin(ϕ, A)
end


# Cylindrical/Polar ⟹ Cartesian
# Coordinate transform 
@inline function _cylin2cart(s :: T, ϕ :: T) where {T<:AbstractFloat}
    x = s * cos(ϕ)
    y = s * sin(ϕ)
    return (T(x), T(y))
end

@inline function _cylin2cart(s :: T, ϕ :: T, z :: T) where {T<:AbstractFloat}
    x = s * cos(ϕ)
    y = s * sin(ϕ)
    return (T(x), T(y), T(z))
end


@inline function _cylin2cart(point::NTuple{2, T}) where {T<:AbstractFloat}
    s, ϕ = point
    return _cylin2cart(s, ϕ)
end

@inline function _cylin2cart(point::NTuple{3, T}) where {T<:AbstractFloat}
    s, ϕ, z = point
    return _cylin2cart(s, ϕ, z)
end

@inline function _cylin2cart(point::AbstractVector{<:AbstractFloat})
    s, ϕ = @inbounds point[1], point[2]
    if length(point) > 2
        return _cylin2cart(s, ϕ, point[3])
    else
        return _cylin2cart(s, ϕ)
    end
end

# Vector transform 
@inline function _vector_cylin2cart(ϕ::T, As::T, Aϕ::T) where {T<:AbstractFloat}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ*As - sinϕ*Aϕ),
            T(sinϕ*As + cosϕ*Aϕ))
end

@inline function _vector_cylin2cart(ϕ::T, As::T, Aϕ::T, Az::T) where {T<:AbstractFloat}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ*As - sinϕ*Aϕ),
            T(sinϕ*As + cosϕ*Aϕ),
            T(Az))
end

@inline function _vector_cylin2cart(ϕ::T, A::Tuple{T, T}) where {T<:AbstractFloat}
    As, Aϕ = @inbounds A[1], A[2]
    return _vector_cylin2cart(ϕ, As, Aϕ)
end

@inline function _vector_cylin2cart(ϕ::T, A::Tuple{T, T, T}) where {T<:AbstractFloat}
    As, Aϕ, Az = @inbounds A[1], A[2], A[3]
    return _vector_cylin2cart(ϕ, As, Aϕ, Az)
end

@inline function _vector_cylin2cart(ϕ::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    As, Aϕ = @inbounds A[1], A[2]
    if length(A) > 2
        return _vector_cylin2cart(ϕ, As, Aϕ, A[3])
    else
        return _vector_cylin2cart(ϕ, As, Aϕ)
    end
end

@inline function _vector_cylin2cart(x::T, y::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    ϕ = mod(atan(y, x), 2π)
    return _vector_cylin2cart(ϕ, A)
end

# # Cartesian ⟹ Spherical
# # Coordinate transform 
# @inline function _cart2sph(x :: T, y :: T) where {T<:AbstractFloat}
#     r = sqrt(x*x + y*y)
#     ϕ = mod(atan(y, x), 2π)
#     return (T(r), T(ϕ))
# end

# @inline function _cart2sph(x::T, y::T, z::T) where {T<:AbstractFloat}
#     r = sqrt(x*x + y*y + z*z)
#     ϕ = mod(atan(y, x), 2π)
#     if iszero(r)
#         θ = zero(T)              
#     else
#         θ = acos(clamp(z/r, -one(T), one(T)))
#     end
#     return (T(r), T(ϕ), T(θ))            
# end


# @inline function _cart2sph(point::NTuple{2, T}) where {T<:AbstractFloat}
#     x, y = point
#     return _cart2sph(x, y)
# end

# @inline function _cart2sph(point::NTuple{3, T}) where {T<:AbstractFloat}
#     x, y, z = point
#     return _cart2sph(x, y, z)
# end

# @inline function _cart2sph(point::AbstractVector{<:AbstractFloat})
#     x, y = @inbounds point[1], point[2]
#     if length(point) > 2
#         return _cart2sph(x, y, point[3])
#     else
#         return _cart2sph(x, y)
#     end
# end

# # Vector transform 
# @inline function _vector_cart2sph(ϕ::T, Ax::T, Ay::T) where {T<:AbstractFloat}
#     cosϕ, sinϕ = cos(ϕ), sin(ϕ)
#     return (T(cosϕ*Ax + sinϕ*Ay),
#             T(-sinϕ*Ax + cosϕ*Ay))
# end

# @inline function _vector_cart2sph(ϕ::T, Ax::T, Ay::T, Az::T) where {T<:AbstractFloat}
#     cosϕ, sinϕ = cos(ϕ), sin(ϕ)
#     return (T(cosϕ*Ax + sinϕ*Ay),
#             T(-sinϕ*Ax + cosϕ*Ay),
#             T(Az))
# end

# @inline function _vector_cart2sph(ϕ::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
#     Ax, Ay = @inbounds A[1], A[2]
#     if length(A) > 2
#         return _vector_cart2sph(ϕ, Ax, Ay, A[3])
#     else
#         return _vector_cart2sph(ϕ, Ax, Ay)
#     end
# end

# @inline function _vector_cart2sph(ϕ::T, A::NTuple{2, T}) where {T<:AbstractFloat}
#     Ax, Ay = @inbounds A[1], A[2]
#     return _vector_cart2sph(ϕ, Ax, Ay)
# end

# @inline function _vector_cart2sph(ϕ::T, A::NTuple{3, T}) where {T<:AbstractFloat}
#     Ax, Ay, Az = @inbounds A[1], A[2], A[3]
#     return _vector_cart2sph(ϕ, Ax, Ay, Az)
# end

# @inline function _vector_cart2sph(x::T, y::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
#     ϕ = mod(atan(y, x), 2π)
#     return _vector_cart2sph(ϕ, A)
# end