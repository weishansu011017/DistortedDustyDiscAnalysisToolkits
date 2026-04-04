"""
Coordinate transformations modules.
    by Wei-Shan Su,
    September 24, 2025
"""

# Cartesian -> Cylindrical/Polar
# Coordinate transform
@inline function _cart2cylin(x::T, y::T) where {T<:AbstractFloat}
    s = sqrt(x * x + y * y)
    ϕ = mod(atan(y, x), 2π)
    return (T(s), T(ϕ))
end

@inline function _cart2cylin(x::T, y::T, z::T) where {T<:AbstractFloat}
    s = sqrt(x * x + y * y)
    ϕ = mod(atan(y, x), 2π)
    return (T(s), T(ϕ), T(z))
end

@inline function _cart2cylin(point::NTuple{2,T}) where {T<:AbstractFloat}
    x, y = point
    return _cart2cylin(x, y)
end

@inline function _cart2cylin(point::NTuple{3,T}) where {T<:AbstractFloat}
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
    return (T(cosϕ * Ax + sinϕ * Ay), T(-sinϕ * Ax + cosϕ * Ay))
end

@inline function _vector_cart2cylin(ϕ::T, Ax::T, Ay::T, Az::T) where {T<:AbstractFloat}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ * Ax + sinϕ * Ay), T(-sinϕ * Ax + cosϕ * Ay), T(Az))
end

@inline function _vector_cart2cylin(ϕ::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    Ax, Ay = @inbounds A[1], A[2]
    if length(A) > 2
        return _vector_cart2cylin(ϕ, Ax, Ay, A[3])
    else
        return _vector_cart2cylin(ϕ, Ax, Ay)
    end
end

@inline function _vector_cart2cylin(ϕ::T, A::NTuple{2,T}) where {T<:AbstractFloat}
    Ax, Ay = A
    return _vector_cart2cylin(ϕ, Ax, Ay)
end

@inline function _vector_cart2cylin(ϕ::T, A::NTuple{3,T}) where {T<:AbstractFloat}
    Ax, Ay, Az = A
    return _vector_cart2cylin(ϕ, Ax, Ay, Az)
end

@inline function _vector_cart2cylin(x::T, y::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    ϕ = mod(atan(y, x), 2π)
    return _vector_cart2cylin(ϕ, A)
end


# Cylindrical/Polar -> Cartesian
# Coordinate transform
@inline function _cylin2cart(s::T, ϕ::T) where {T<:AbstractFloat}
    x = s * cos(ϕ)
    y = s * sin(ϕ)
    return (T(x), T(y))
end

@inline function _cylin2cart(s::T, ϕ::T, z::T) where {T<:AbstractFloat}
    x = s * cos(ϕ)
    y = s * sin(ϕ)
    return (T(x), T(y), T(z))
end

@inline function _cylin2cart(point::NTuple{2,T}) where {T<:AbstractFloat}
    s, ϕ = point
    return _cylin2cart(s, ϕ)
end

@inline function _cylin2cart(point::NTuple{3,T}) where {T<:AbstractFloat}
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
    return (T(cosϕ * As - sinϕ * Aϕ), T(sinϕ * As + cosϕ * Aϕ))
end

@inline function _vector_cylin2cart(ϕ::T, As::T, Aϕ::T, Az::T) where {T<:AbstractFloat}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ * As - sinϕ * Aϕ), T(sinϕ * As + cosϕ * Aϕ), T(Az))
end

@inline function _vector_cylin2cart(ϕ::T, A::NTuple{2,T}) where {T<:AbstractFloat}
    As, Aϕ = A
    return _vector_cylin2cart(ϕ, As, Aϕ)
end

@inline function _vector_cylin2cart(ϕ::T, A::NTuple{3,T}) where {T<:AbstractFloat}
    As, Aϕ, Az = A
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


# Cartesian -> Spherical
# Coordinate transform
@inline function _cart2sph(x::T, y::T, z::T) where {T<:AbstractFloat}
    r = sqrt(x * x + y * y + z * z)
    ϕ = mod(atan(y, x), 2π)
    θ = iszero(r) ? zero(T) : acos(clamp(z / r, -one(T), one(T)))
    return (T(r), T(ϕ), T(θ))
end

@inline function _cart2sph(point::NTuple{3,T}) where {T<:AbstractFloat}
    x, y, z = point
    return _cart2sph(x, y, z)
end

@inline function _cart2sph(point::AbstractVector{<:AbstractFloat})
    x, y, z = @inbounds point[1], point[2], point[3]
    return _cart2sph(x, y, z)
end

# Vector transform
@inline function _vector_cart2sph(ϕ::T, θ::T, Ax::T, Ay::T, Az::T) where {T<:AbstractFloat}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    cosθ, sinθ = cos(θ), sin(θ)
    Ar = sinθ * cosϕ * Ax + sinθ * sinϕ * Ay + cosθ * Az
    Aϕ = -sinϕ * Ax + cosϕ * Ay
    Aθ = cosθ * cosϕ * Ax + cosθ * sinϕ * Ay - sinθ * Az
    return (T(Ar), T(Aϕ), T(Aθ))
end

@inline function _vector_cart2sph(ϕ::T, θ::T, A::NTuple{3,T}) where {T<:AbstractFloat}
    Ax, Ay, Az = A
    return _vector_cart2sph(ϕ, θ, Ax, Ay, Az)
end

@inline function _vector_cart2sph(ϕ::T, θ::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    Ax, Ay, Az = @inbounds A[1], A[2], A[3]
    return _vector_cart2sph(ϕ, θ, Ax, Ay, Az)
end

@inline function _vector_cart2sph(x::T, y::T, z::T, A::NTuple{3,T}) where {T<:AbstractFloat}
    _, ϕ, θ = _cart2sph(x, y, z)
    return _vector_cart2sph(ϕ, θ, A)
end

@inline function _vector_cart2sph(x::T, y::T, z::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    _, ϕ, θ = _cart2sph(x, y, z)
    return _vector_cart2sph(ϕ, θ, A)
end


# Spherical -> Cartesian
# Coordinate transform
@inline function _sph2cart(r::T, ϕ::T, θ::T) where {T<:AbstractFloat}
    sinθ = sin(θ)
    x = r * sinθ * cos(ϕ)
    y = r * sinθ * sin(ϕ)
    z = r * cos(θ)
    return (T(x), T(y), T(z))
end

@inline function _sph2cart(point::NTuple{3,T}) where {T<:AbstractFloat}
    r, ϕ, θ = point
    return _sph2cart(r, ϕ, θ)
end

@inline function _sph2cart(point::AbstractVector{<:AbstractFloat})
    r, ϕ, θ = @inbounds point[1], point[2], point[3]
    return _sph2cart(r, ϕ, θ)
end

# Vector transform
@inline function _vector_sph2cart(ϕ::T, θ::T, Ar::T, Aϕ::T, Aθ::T) where {T<:AbstractFloat}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    cosθ, sinθ = cos(θ), sin(θ)
    Ax = sinθ * cosϕ * Ar - sinϕ * Aϕ + cosθ * cosϕ * Aθ
    Ay = sinθ * sinϕ * Ar + cosϕ * Aϕ + cosθ * sinϕ * Aθ
    Az = cosθ * Ar - sinθ * Aθ
    return (T(Ax), T(Ay), T(Az))
end

@inline function _vector_sph2cart(ϕ::T, θ::T, A::NTuple{3,T}) where {T<:AbstractFloat}
    Ar, Aϕ, Aθ = A
    return _vector_sph2cart(ϕ, θ, Ar, Aϕ, Aθ)
end

@inline function _vector_sph2cart(ϕ::T, θ::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    Ar, Aϕ, Aθ = @inbounds A[1], A[2], A[3]
    return _vector_sph2cart(ϕ, θ, Ar, Aϕ, Aθ)
end

@inline function _vector_sph2cart(r::T, ϕ::T, θ::T, A::NTuple{3,T}) where {T<:AbstractFloat}
    return _vector_sph2cart(ϕ, θ, A)
end

@inline function _vector_sph2cart(r::T, ϕ::T, θ::T, A::AbstractVector{<:T}) where {T<:AbstractFloat}
    return _vector_sph2cart(ϕ, θ, A)
end
