"""
Useful mathematical toolkit.
    by Wei-Shan Su,
    June 27, 2024
"""

"""
    _cart2cylin(point::Vector)
Transfer a cartesian coordinate into cylindrical coordinate.

# Parameters
- `point :: Vector`: The point with a set of cartesian coordinate.

# Returns
- `Vector`: The point with a set of cylindrical coordinate.
"""
function _cart2cylin(point::Vector)
    s = sqrt(point[1]^2 + point[2]^2)
    if point[2] >= 0.0
        theta = acos(point[1]/s)
    else
        theta = abs(acos(point[1]/s) - 2π)
    end
    if length(point) == 2
        return [s, theta]
    else
        return [s, theta, point[3]]
    end
end

"""
    _cylin2cart(point::Vector)
Transfer a cylindrical coordinate into cartesian coordinate.

# Parameters
- `point :: Vector`: The point with a set of cylindrical coordinate..

# Returns
- `Vector`: The point with a set of cartesian coordinate
"""
function _cylin2cart(point::Vector)
    x = point[1] * cos(point[2])
    y = point[1] * sin(point[2])
    if length(point) == 2
        return [x, y]
    else
        return [x, y, point[3]]
    end
end

"""
    _vector_cart2cylin(ϕ::Float64, Ax::Float64, Ay::Float64, Az::Union{Nothing,Float64}=nothing)
Transform a vector from Cartesian coordinates to cylindrical coordinates.

# Parameters
- `ϕ :: Float64`: The azimuth angle of the position of the vector (in radians).
- `Ax :: Float64`: The x-component of the vector.
- `Ay :: Float64`: The y-component of the vector.
- `Az :: Union{Nothing, Float64} = nothing`: The z-component of the vector (optional).

# Returns
- `Vector{Float64}`: The vector in cylindrical coordinates, with components `[Ar, Aϕ, Az]` where:
  - `Ar`: Radial component.
  - `Aϕ`: Azimuthal component.
  - `Az`: Axial (z) component (included if `Az` is provided).
"""
function _vector_cart2cylin(
    ϕ::Float64,
    Ax::Float64,
    Ay::Float64,
    Az::Union{Nothing, Float64} = nothing
)::Vector{Float64}
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)

    cylin_vector = Vector{Float64}(undef, isnothing(Az) ? 2 : 3)

    cylin_vector[1] = cosϕ * Ax + sinϕ * Ay  # Radial component
    cylin_vector[2] = -sinϕ * Ax + cosϕ * Ay # Azimuthal component
    if !isnothing(Az)
        cylin_vector[3] = Az # Axial component remains unchanged
    end

    return cylin_vector
end

"""
    _vector_cylin2cart(ϕ::Float64, As::Float64, Aϕ::Float64, Az::Union{Nothing, Float64} = nothing)
Transform a vector from cylindrical coordinates to Cartesian coordinates.

# Parameters
- `ϕ :: Float64`: The azimuth angle of the position of the vector (in radians).
- `As :: Float64`: The radial component of the vector.
- `Aϕ :: Float64`: The azimuthal component of the vector.
- `Az :: Union{Nothing, Float64} = nothing`: The vertical (z) component of the vector (optional).

# Returns
- `Vector{Float64}`: The vector in Cartesian coordinates, with components `[Ax, Ay, Az]` where:
  - `Ax`: x-component.
  - `Ay`: y-component.
  - `Az`: z-component (included if `Az` is provided).
"""
function _vector_cylin2cart(
    ϕ::Float64,
    As::Float64,
    Aϕ::Float64,
    Az::Union{Nothing, Float64} = nothing
)::Vector{Float64}
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)

    # Determine the length of the resulting vector
    cart_vector = Vector{Float64}(undef, isnothing(Az) ? 2 : 3)

    # Transform components
    cart_vector[1] = cosϕ * As - sinϕ * Aϕ  # x-component
    cart_vector[2] = sinϕ * As + cosϕ * Aϕ  # y-component
    if !isnothing(Az)
        cart_vector[3] = Az # z-component remains unchanged
    end

    return cart_vector
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
