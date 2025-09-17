"""
Useful mathematical toolkit.
    by Wei-Shan Su,
    June 27, 2024
"""

# Cartesian ⟹ Cylindrical/Polar
# Coordinate transform 
@inline function _cart2cylin(x :: T, y :: T) where {T<:Real}
    s = sqrt(x*x + y*y)
    ϕ = mod(atan(y, x), 2π)
    return (T(s), T(ϕ))
end

@inline function _cart2cylin(x :: T, y :: T, z :: T) where {T<:Real}
    s = sqrt(x*x + y*y)
    ϕ = mod(atan(y, x), 2π)
    return (T(s), T(ϕ), T(z))
end

@inline function _cart2cylin(point::AbstractVector{<:Real})
    x, y = @inbounds point[1], point[2]
    if length(point) > 2
        return _cart2cylin(x, y, point[3])
    else
        return _cart2cylin(x, y)
    end
end

@inline function _cart2cylin(point::Tuple{T,T}) where {T<:Real}
    x, y = point
    return _cart2cylin(x, y)
end

@inline function _cart2cylin(point::Tuple{T,T,T}) where {T<:Real}
    x, y, z = point
    return _cart2cylin(x, y, z)
end

# Vector transform 
@inline function _vector_cart2cylin(ϕ::T, A::AbstractVector{<:T}) where {T<:Real}
    Ax, Ay = @inbounds A[1], A[2]
    if length(A) > 2
        return _vector_cart2cylin(ϕ, Ax, Ay, A[3])
    else
        return _vector_cart2cylin(ϕ, Ax, Ay)
    end
end

@inline function _vector_cart2cylin(ϕ::T, A::Tuple{T, T}) where {T<:Real}
    Ax, Ay = @inbounds A[1], A[2]
    return _vector_cart2cylin(ϕ, Ax, Ay)
end

@inline function _vector_cart2cylin(ϕ::T, A::Tuple{T, T, T}) where {T<:Real}
    Ax, Ay, Az = @inbounds A[1], A[2], A[3]
    return _vector_cart2cylin(ϕ, Ax, Ay, Az)
end

@inline function _vector_cart2cylin(x::T, y::T, A::AbstractVector{<:T}) where {T<:Real}
    ϕ = mod(atan(y, x), 2π)
    return _vector_cart2cylin(ϕ, A)
end

@inline function _vector_cart2cylin(ϕ::T, Ax::T, Ay::T) where {T<:Real}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ*Ax + sinϕ*Ay),
            T(-sinϕ*Ax + cosϕ*Ay))
end

@inline function _vector_cart2cylin(ϕ::T, Ax::T, Ay::T, Az::T) where {T<:Real}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ*Ax + sinϕ*Ay),
            T(-sinϕ*Ax + cosϕ*Ay),
            T(Az))
end


# Cylindrical/Polar ⟹ Cartesian
# Coordinate transform 
@inline function _cylin2cart(s :: T, ϕ :: T) where {T<:Real}
    x = s * cos(ϕ)
    y = s * sin(ϕ)
    return (T(x), T(y))
end

@inline function _cylin2cart(s :: T, ϕ :: T, z :: T) where {T<:Real}
    x = s * cos(ϕ)
    y = s * sin(ϕ)
    return (T(x), T(y), T(z))
end


@inline function _cylin2cart(point::AbstractVector{<:Real})
    s, ϕ = @inbounds point[1], point[2]
    if length(point) > 2
        return _cylin2cart(s, ϕ, point[3])
    else
        return _cylin2cart(s, ϕ)
    end
end

@inline function _cylin2cart(point::Tuple{T,T}) where {T<:Real}
    s, ϕ = point
    return _cylin2cart(s, ϕ)
end

@inline function _cylin2cart(point::Tuple{T,T,T}) where {T<:Real}
    s, ϕ, z = point
    return _cylin2cart(s, ϕ, z)
end

# Vector transform 
@inline function _vector_cylin2cart(ϕ::T, A::AbstractVector{<:T}) where {T<:Real}
    As, Aϕ = @inbounds A[1], A[2]
    if length(A) > 2
        return _vector_cylin2cart(ϕ, As, Aϕ, A[3])
    else
        return _vector_cylin2cart(ϕ, As, Aϕ)
    end
end

@inline function _vector_cylin2cart(ϕ::T, A::Tuple{T, T}) where {T<:Real}
    As, Aϕ = @inbounds A[1], A[2]
    return _vector_cylin2cart(ϕ, As, Aϕ)
end

@inline function _vector_cylin2cart(ϕ::T, A::Tuple{T, T, T}) where {T<:Real}
    As, Aϕ, Az = @inbounds A[1], A[2], A[3]
    return _vector_cylin2cart(ϕ, As, Aϕ, Az)
end

@inline function _vector_cylin2cart(x::T, y::T, A::AbstractVector{<:T}) where {T<:Real}
    ϕ = mod(atan(y, x), 2π)
    return _vector_cylin2cart(ϕ, A)
end

@inline function _vector_cylin2cart(ϕ::T, As::T, Aϕ::T) where {T<:Real}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ*As - sinϕ*Aϕ),
            T(sinϕ*As + cosϕ*Aϕ))
end

@inline function _vector_cylin2cart(ϕ::T, As::T, Aϕ::T, Az::T) where {T<:Real}
    cosϕ, sinϕ = cos(ϕ), sin(ϕ)
    return (T(cosϕ*As - sinϕ*Aϕ),
            T(sinϕ*As + cosϕ*Aϕ),
            T(Az))
end

"""
    _Integral_1d(x::AbstractVector ,y::AbstractVector, inteval::Vector)
Intergral a discrete function f(x) in a interval in 1D. 

# Parameters
- `x :: AbstractVector`: The x value.
- `y :: AbstractVector`: The f(x) value.
- `inteval :: Vector`: The intergral range.

# Returns
- `Float64`: The integral result.
"""
function _Integral_1d(x::AbstractVector, y::AbstractVector, inteval::Vector)
    spline = CubicSplineInterpolation(x, y, extrapolation_bc = Line())
    f_interp(x) = spline(x)
    integral, error = quadgk(f_interp, inteval[1], inteval[2])
    return integral
end

"""
    value2closestvalueindex(array::AbstractVector, Union{Float64,Int64})
Find the index of value which is the closest value to a given target in a array.

# Parameters
- `array::AbstractVector`: The array.
- `target::Union{Float64,Int64}`: The target for finding.

# Returns
- `Int64`: The index of closest value.
"""
function value2closestvalueindex(array::AbstractVector, target::Union{Float64,Int64})
    target_index = argmin(abs.(target .- array))
    return target_index
end

"""
    find_array_max_index(y::AbstractVector)
Find the index of maximum value in a array.

# Parameters
- `y :: AbstractVector`: The array.

# Returns
- `Int64`: The index of maximum value.
"""
function find_array_max_index(y::AbstractVector)
    arange = 1:length(y)
    spline(x) = CubicSplineInterpolation(arange, y, extrapolation_bc = Line())(x)
    dspline(x) = ForwardDiff.derivative(spline, x)
    ddspline(x) = ForwardDiff.derivative(dspline, x)

    closest_to_zero = Inf
    target_index = 0
    for i in arange
        dspline_value = dspline(i)
        if abs(dspline_value) < closest_to_zero
            ddspline_value = ddspline(i)
            signature = sign(ddspline_value)
            if signature <= 0
                closest_to_zero = abs(dspline_value)
                target_index = i
            end
        end
    end

    if target_index == 0
        error("SearchMaxError: The maximum value is not found.")
    end

    return target_index
end

"""
    astrounit2KeperianAngularVelocity(r :: Float64,M :: Float64)
Calculate the Keperian angular velocity in cgs by giving the parameters in au and M⊙

# Parameters
- `r :: Float64`: The radius to the center of the system in au.
- `M :: Float64`: The mass of the center star in M⊙

# Returns
- `Float64`: The Keperian angular velocity in cgs.
"""
function astrounit2KeperianAngularVelocity(r::Float64,M::Float64)
    r_cgs = 14959787069100.0 * r
    M_cgs =1.9891e33* M
    G = 6.67e-8
    Ω = sqrt((G*M_cgs)/r_cgs^3)
    return Ω
end

"""
    replace_inf_with_nan!(array::Array)

Replaces all infinite (`Inf`, `-Inf`) values in the given array with `NaN`.

# Parameters
- `array::Array`: The input array containing numerical values.

# Returns
- `Array`: A modified version of the input array where all infinite values are replaced with `NaN`.
"""
function replace_inf_with_nan!(array::Array)
    mask = isinf.(array)
    array[mask] .= NaN
    return array
end

"""
    Binning2OneDimBoxes(array::AbstractArray, boxes::AbstractVector)

Assigns elements of `array` to bins defined by `boxes`, returning an array of bin indices.

This function takes an input `array` and assigns each element to the closest bin index 
defined in `boxes`, which must be sorted in ascending order. The output is an array of 
integers indicating the bin index for each element in the original `array`.

# Parameters
- `array::AbstractArray`  
    - The input array containing numerical values to be binned.
- `boxes::AbstractVector`  
    - A sorted vector defining bin edges. The values in `boxes` should be in ascending order.

# Returns
- `result_bin_array::AbstractArray{Int64}`  
    - An array of the same shape as `array`, where each element is replaced by its bin index.
"""
function Binning2OneDimBoxes(array::AbstractArray, boxes::AbstractVector)
    if !issorted(boxes)
        error("NonSortedError: Only a sorted boxes set is allowed")
    end
    original_size = size(array)
    flatten_array = vec(array)
    bin_index = zeros(Int64,length(flatten_array))
    for (i,element) in enumerate(flatten_array)
        closest_index = value2closestvalueindex(boxes,element)
        neiboredge = boxes[closest_index]
        if ((neiboredge <= element) && (closest_index != length(boxes)))
            bin_index[i] = closest_index
        elseif ((neiboredge > element) && (closest_index != 1))
            bin_index[i] = closest_index - 1
        else
            bin_index[i] = -1
        end
    end
    result_bin_array = reshape(bin_index,original_size...)
    return result_bin_array
end

"""
    AssignBinIndices(array::AbstractArray, boxes::AbstractVector; mode::Symbol = :bin)

Assign each element in `array` to a bin defined by `boxes`, using either binning or nearest-neighbor matching.

# Parameters
- `array :: AbstractArray`: The array of values to assign to bins.
- `boxes :: AbstractVector`: A sorted vector defining bin **edges** (for `mode=:bin`) or **center points** (for `:nearest`).
- `mode :: Symbol`: 
    - `:bin`: Use standard binning. `boxes` are treated as bin edges.
    - `:nearest`: Match each value to the closest value in `boxes`.

# Returns
- `Array`: An array of the same shape as `array`, filled with bin indices (1-based). `-1` for out-of-range (in `:bin` mode).
"""
function AssignBinIndices(array::AbstractArray, boxes::AbstractVector; mode::Symbol = :bin)
    if !issorted(boxes)
        error("Boxes must be sorted in ascending order.")
    end

    original_size = size(array)
    flatten_array = vec(array)
    bin_index = similar(flatten_array, Int)

    if mode == :bin
        for (i, val) in enumerate(flatten_array)
            idx = searchsortedlast(boxes, val)
            if 1 <= idx < length(boxes)
                bin_index[i] = idx
            else
                bin_index[i] = -1  # value out of bin range
            end
        end

    elseif mode == :nearest
        for (i, val) in enumerate(flatten_array)
            _, idx = findmin(abs.(boxes .- val))
            bin_index[i] = idx
        end

    else
        error("Unknown mode: $mode. Must be :bin or :nearest.")
    end

    return reshape(bin_index, original_size...)
end

"""
    @inline function angular_distance(θ1, θ2) 

Calculate the angular distances between (θ1, θ2) 

# Parameters
- `θ1 :: Real`: The first angle.
- `θ2 :: Real`: The second angle.

# Returns
- `Float64`: The angular distances between (θ1, θ2).
"""
@inline function angular_distance(θ1 :: Real, θ2 :: Real) :: Float64
    return abs(mod(θ1 - θ2 + π, 2π) - π)
end

"""
    @inline function k2pitch(k::Real)::Float64

Convert the logarithmic spiral parameter `k` to pitch angle in degrees.

The pitch angle is defined as the angle between the spiral arm and a circle,
and relates to `k` through: `pitch = arctangent(k)` in radians, converted to degrees.

# Parameters
- `k::Real`: The logarithmic spiral parameter in the function `r(θ) = a * exp(kθ)`.

# Returns
- `Float64`: Pitch angle in degrees.
"""
@inline function k2pitch(k :: Real) :: Float64
    return rad2deg(tan(k))
end

"""
    @inline function pitch2k(pitchdeg::Real)::Float64

Convert the pitch angle in degrees to the logarithmic spiral parameter `k`.

The logarithmic spiral parameter `k` determines the tightness of the spiral arm.
It is derived from the pitch angle via: `k = tan(pitch)` where pitch is in radians.

# Parameters
- `pitchdeg::Real`: Pitch angle in degrees.

# Returns
- `Float64`: The logarithmic spiral parameter `k`.
"""
@inline function pitch2k(pitchdeg :: Real) :: Float64
    return atan(deg2rad(pitchdeg))
end

"""
    marco _def_nanfunc(funcname)

Define a new `nan*` variant of the statistical function that ignores `NaN` values.

This macro generates two method overloads:
- `nanfunc(A::AbstractArray{T})`: returns the scalar result after removing all `NaN`s.
- `nanfunc(A::AbstractArray{T}, dims::Integer)`: performs reduction along the specified dimension, skipping `NaN`s and returning `T(NaN)` if the slice is entirely `NaN`.

# Parameters
- `funcname::Symbol`: The base function name (e.g., `:mean`, `:maximum`) to wrap.

# Generated
- A new function named `nanfuncname` will be defined in the current scope.
"""
macro _def_nanfunc(funcname)
    fname = Symbol(:nan, funcname)          
    quote
        @inline function $(fname)(A::AbstractArray{T}) where {T<:Number}
            vals = filter(!isnan, A)
            return isempty(vals) ? T(NaN) : $(funcname)(vals)
        end

        @inline function $(fname)(A::AbstractArray{T}, dims::Integer) where {T<:Number}
            if ndims(A) == 1
                return [nanmean(A)]
            else
                result = map(x -> isempty(x) ? T(NaN) : mean(x), eachslice(A, dims))
                return dropdims(result, dims)
            end
        end
    end |> esc        
end


@_def_nanfunc mean
@_def_nanfunc median
@_def_nanfunc std
@_def_nanfunc maximum
@_def_nanfunc minimum