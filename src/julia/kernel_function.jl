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
    
    ## Constants
        _TRUNCATED_RADIUS: The truncated radius of kernel function in the units of "h".(Smoothing radius)(For Building up the KD-Tree Searching)
        _CNORM: The normalized constant Cnorm of Kernel function(dimensional-depended), the indices represent the dimension of Cnorm. 
        _DKERNEL: The differentiated kernel reference dictionary.
        KernelFunctionValid(), KernelFunctionnorm(), KernelFunctionDiff(): A shortcut for calling these constants.

    ## Calculating influence by Smoothed Function
        Smoothed_kernel_function (2 Method): Calculating W(r-r_a, h).
        Smoothed_greident_kernel_function (2 Method): Calculating ∇W(r-r_a, h).
        Smoothed_dh_kernel_function (2 Method): Calculating
        
"""

# Kernel Functions
# M4 B-spline
function M4_spline(q)
    if 0 <= q < 1
        return (0.25 * (2 - q)^3 - (1 - q)^3)
    elseif 1 <= q < 2
        return (0.25 * (2 - q)^3)
    elseif q >= 2
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

function _dM4_spline(q)
    if 0 <= q < 1
        return (-0.75 * (2 - q)^2 + 3 * (1 - q)^2)
    elseif 1 <= q < 2
        return (-0.75 * (2 - q)^2)
    elseif q >= 2
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

# M5 B-spline
function M5_spline(q)
    if 0 <= q < 0.5
        return ((2.5 - q)^4 - 5 * (1.5 - q)^4 + 10 * (0.5 - q)^4)
    elseif 0.5 <= q < 1.5
        return ((2.5 - q)^4 - 5 * (1.5 - q)^4)
    elseif 1.5 <= q < 2.5
        return ((2.5 - q)^4)
    elseif q >= 2.5
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

function _dM5_spline(q)
    if 0 <= q < 0.5
        return (-4 * (2.5 - q)^3 + 20 * (1.5 - q)^3 - 40 * (0.5 - q)^3)
    elseif 0.5 <= q < 1.5
        return (-4 * (2.5 - q)^3 + 20 * (1.5 - q)^3)
    elseif 1.5 <= q < 2.5
        return (-4 * (2.5 - q)^3)
    elseif q >= 2.5
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

# M6 B-spline
function M6_spline(q)
    if 0 <= q < 1
        return ((3 - q)^5 - 6 * (2 - q)^5 + 15 * (1 - q)^5)
    elseif 1 <= q < 2
        return ((3 - q)^5 - 6 * (2 - q)^5)
    elseif 2 <= q < 3
        return ((3 - q)^5)
    elseif q >= 3
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

function _dM6_spline(q)
    if 0 <= q < 1
        return (-5 * (3 - q)^4 + 30 * (2 - q)^4 - 75 * (1 - q)^4)
    elseif 1 <= q < 2
        return (-5 * (3 - q)^4 + 30 * (2 - q)^4)
    elseif 2 <= q < 3
        return (-5 * (3 - q)^4)
    elseif q >= 3
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

# Wendland C2
function C2_Wendland(q)
    if q < 2
        return (((1 - 0.5 * q)^4) * (2 * q + 1))
    elseif q >= 2
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

function _dC2_Wendland(q)
    if q < 2
        return (((1 - 0.5 * q)^4) * 2 - ((1 - 0.5 * q)^3) * (4 * q + 2))
    elseif q >= 2
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

# Wendland C4
function C4_Wendland(q)
    if q < 2
        return (((1 - 0.5 * q)^6) * ((35 / 12) * (q^2) + 3 * q + 1))
    elseif q >= 2
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

function _dC4_Wendland(q)
    if q < 2
        return (
            ((1 - 0.5 * q)^6) * ((35 / 6) * q + 3) -
            ((1 - 0.5 * q)^5) * ((35 / 4) * (q^2) + 9 * q + 3)
        )
    elseif q >= 2
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

# Wendland C6
function C6_Wendland(q)
    if q < 2
        return (((1 - 0.5 * q)^8) * (4 * (q^3) + 6.25 * (q^2) + 4 * q + 1))
    elseif q >= 2
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

function _dC6_Wendland(q)
    if q < 2
        return (
            ((1 - 0.5 * q)^8) * (12 * (q^2) + 12.5 * q + 4) -
            ((1 - 0.5 * q)^7) * (16 * (q^3) + 25 * (q^2) + 16 * q + 4)
        )
    elseif q >= 2
        return 0.0
    else
        error("Kernel Error: Invalid input!")
    end
end

# Function constant

"""
    _TRUNCATED_RADIUS
The truncated radius of kernel function in the units of "h".(Smoothing radius)(For Building up the KD-Tree Searching)
"""
const _TRUNCATED_RADIUS = Dict(
    :M4_spline => 2.0,
    :M5_spline => 2.5,
    :M6_spline => 3.0,
    :C2_Wendland => 2.0,
    :C4_Wendland => 2.0,
    :C6_Wendland => 2.0,
)

"""
    _CNORM
The normalized constant Cnorm of Kernel function(dimensional-depended), the indices represent the dimension of Cnorm. 
"""
const _CNORM = Dict(
    :M4_spline => [4 / 3, 10 / (7 * pi), 1 / pi],
    :M5_spline => [1 / 24, 96 / (1199 * pi), 0.05 / pi],
    :M6_spline => [120^(-1), 7 / (478 * pi), (120 * pi)^(-1)],
    :C2_Wendland => [5 / 8, 7 / (4 * pi), 21 / (16 * pi)],
    :C4_Wendland => [3 / 4, 9 / (4 * pi), 495 / (256 * pi)],
    :C6_Wendland => [64 / 55, 78 / (28 * pi), 1365 / (512 * pi)],
)

"""
    _DKERNEL
The differentiated kernel reference dictionary.
"""
const _DKERNEL = Dict(
    :M4_spline => _dM4_spline,
    :M5_spline => _dM5_spline,
    :M6_spline => _dM6_spline,
    :C2_Wendland => _dC2_Wendland,
    :C4_Wendland => _dC4_Wendland,
    :C6_Wendland => _dC6_Wendland,
)


"""
    _Nneigh
The mean neighbour number of kernel provided by Price et al.(2018).
"""
const _Nneigh = Dict(
    :M4_spline => 57,
    :M5_spline => 113,
    :M6_spline => 112,
    :C2_Wendland => 92,
    :C4_Wendland => 137,
    :C6_Wendland => 356,
)

"""
    function KernelFunctionValid()
Call the dictionary of truncated radius of kernel function.

# Returns
- _TRUNCATED_RADIUS

# Examples
```julia
radius = KernelFunctionValid()[:M4_spline]
println(radius) 
>> 2.0
```
"""
function KernelFunctionValid()
    return _TRUNCATED_RADIUS
end

"""
    function KernelFunctionnorm()
Call the dictionary of normalized constant of kernel function.

# Returns
- _CNORM

# Examples
```julia
Cnorm = KernelFunctionnorm()[:M4_spline]
println(Cnorm) 
>> [4/3, 10/7π, π]
```
"""
function KernelFunctionnorm()
    return _CNORM
end

"""
    function KernelFunctionDiff()
Call the dictionary of differentiated reference of kernel function.

# Returns
- _DKERNEL

# Examples
```julia
dfun = KernelFunctionDiff()[:M4_spline]
println(nameof(dfun)) 
>> :_dM4_spline
```
"""
function KernelFunctionDiff()
    return _DKERNEL
end

"""
    function KernelFunctioNneigh()
Call the mean neighbour number of kernel function.

# Returns
- _Nneigh

# Examples
```julia
Nneigh = KernelFunctionNneigh()[:M4_spline]
println(Nneigh) 
>> 58
```
"""
function KernelFunctionNneigh()
    return _Nneigh
end

# Calculating influence by Smoothed Function
# W(ra-rb,h)
"""
    function Smoothed_kernel_function_dimensionless(f::Function, h::Union{Float32,Float64}, ra::Vector, rb::Vector)

Calculating dimensionless part of W(ra-rb,h).

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `ra::Vector`: Position a in Vector.
- `rb::Vector`: Position b in Vector.

# Returns
- `Float64`: W(ra-rb,h)
"""
function Smoothed_kernel_function_dimensionless(
    f::Function,
    h::Union{Float32,Float64},
    ra::Vector,
    rb::Vector,
)
    r::Float64 = norm(ra - rb)
    q::Float64 = r / h
    dim::Int32 = length(ra)
    influence::Float64 = f(q) * KernelFunctionnorm()[nameof(f)][dim]
    return influence
end

"""
    function Smoothed_kernel_function_dimensionless(f::Function, h::Union{Float32,Float64}, r::Float64, dim::Int)

Calculating dimensionless part of W(ra-rb,h).

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `r::Float64`: The geometric distance bewteen ra and rb.
- `dim:Int`: The dimension for calculating.

# Returns
- `Float64`: W(ra-rb,h)
"""
function Smoothed_kernel_function_dimensionless(
    f::Function,
    h::Union{Float32,Float64},
    r::Float64,
    dim::Int,
)
    q::Float64 = r / h
    influence::Float64 = f(q) * KernelFunctionnorm()[nameof(f)][dim]
    return influence
end

"""
    function Smoothed_kernel_function(f::Function, h::Union{Float32,Float64}, ra::Vector, rb::Vector)

Calculating W(ra-rb,h).

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `ra::Vector`: Position a in Vector.
- `rb::Vector`: Position b in Vector.

# Returns
- `Float64`: W(ra-rb,h)
"""
function Smoothed_kernel_function(
    f::Function,
    h::Union{Float32,Float64},
    ra::Vector,
    rb::Vector,
)  
    dim::Int32 = length(ra)
    hr::Float64 = h^(-dim)
    influence::Float64 = hr * Smoothed_kernel_function_dimensionless(f,h,ra,rb)
    return influence
end

"""
    function Smoothed_kernel_function(f::Function, h::Union{Float32,Float64}, r::Float64, dim::Int)

Calculating W(ra-rb,h).

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `r::Float64`: The geometric distance bewteen ra and rb.
- `dim:Int`: The dimension for calculating.

# Returns
- `Float64`: W(ra-rb,h)
"""
function Smoothed_kernel_function(
    f::Function,
    h::Union{Float32,Float64},
    r::Float64,
    dim::Int,
)
    q::Float64 = r / h
    hr::Float64 = h^(-dim)
    influence::Float64 = hr * Smoothed_kernel_function_dimensionless(f,h,r,dim)
    return influence
end

# ∇W(ra-rb,h)
"""
    Smoothed_greident_kernel_function_dimensionless(f::Function, h::Union{Float32,Float64}, ra::Vector, rb::Vector)

Calculating dimensionless part of ∇W(ra-rb,h).

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `ra::Vector`: Position a in Vector.
- `rb::Vector`: Position b in Vector.

# Returns
- `Vector`: ∇W(ra-rb,h)
"""
function Smoothed_greident_kernel_function_dimensionless(
    f::Function,
    h::Union{Float32,Float64},
    ra::Vector,
    rb::Vector,
)
    rab::Vector = ra - rb
    hatrab::Vector = rab ./ norm(rab)
    q::Float64 = norm(rab) / h
    dim::Int32 = length(rab)
    F_ab::Float64 =
        KernelFunctionDiff()[nameof(f)](q) * KernelFunctionnorm()[nameof(f)][dim]
    influence::Vector = hatrab .* F_ab
    return influence
end

"""
    Smoothed_greident_kernel_function_dimensionless(f::Function, h::Union{Float32,Float64}, rab::Vector)

Calculating dimensionless part of ∇W(ra-rb,h).

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `rab::Vector`: Vector of ra-rb

# Returns
- `Vector`: ∇W(ra-rb,h)
"""
function Smoothed_greident_kernel_function_dimensionless(
    f::Function,
    h::Union{Float32,Float64},
    rab::Vector,
)
    hatrab::Vector = rab ./ norm(rab)
    q::Float64 = norm(rab) / h
    dim::Int32 = length(rab)
    F_ab::Float64 =
        KernelFunctionDiff()[nameof(f)](q) * KernelFunctionnorm()[nameof(f)][dim]
    influence::Vector = hatrab .* F_ab
    return influence
end
"""
    Smoothed_greident_kernel_function(f::Function, h::Union{Float32,Float64}, ra::Vector, rb::Vector)

Calculating ∇W(ra-rb,h).

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `ra::Vector`: Position a in Vector.
- `rb::Vector`: Position b in Vector.

# Returns
- `Vector`: ∇W(ra-rb,h)
"""
function Smoothed_greident_kernel_function(
    f::Function,
    h::Union{Float32,Float64},
    ra::Vector,
    rb::Vector,
)
    dim::Int32 = length(rab)
    hr::Float64 = h^(-(dim + 1))
    influence::Vector = hr .* Smoothed_greident_kernel_function_dimensionless(f,h,ra,rb)
    return influence
end

"""
    Smoothed_greident_kernel_function(f::Function, h::Union{Float32,Float64}, rab)

Calculating ∇W(ra-rb,h).

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `rab::Vector`: Vector of ra-rb

# Returns
- `Vector`: ∇W(ra-rb,h)
"""
function Smoothed_greident_kernel_function(
    f::Function,
    h::Union{Float32,Float64},
    rab::Vector,
)
    dim::Int32 = length(rab)
    hr::Float64 = h^(-(dim + 1))
    influence::Vector = hr .* Smoothed_greident_kernel_function_dimensionless(f,h,rab)
    return influence
end

# ∂W(ra-rb,h)/∂h
"""
    Smoothed_dh_kernel_function(f::Function, h::Union{Float32,Float64}, ra::Vector, rb::Vector)

Calculating ∂W(ra-rb,h)/∂h.

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `ra::Vector`: Position a in Vector.
- `rb::Vector`: Position b in Vector.

# Returns
- `Float64`: ∂W(ra-rb,h)/∂h
"""
function Smoothed_dh_kernel_function(
    f::Function,
    h::Union{Float32,Float64},
    ra::Vector,
    rb::Vector,
)
    r::Float64 = norm(ra - rb)
    dim::Int32 = length(ra)
    q::Float64 = r / h
    hr::Float64 = h^(-(dim + 1))
    influence::Float64 =
        -hr *
        (3 * f(q) + q * KernelFunctionDiff()[nameof(f)](q)) *
        KernelFunctionnorm()[nameof(f)][dim]
    return influence
end

"""
    Smoothed_dh_kernel_function(f::Function, h::Union{Float32,Float64}, r::Float64, dim::Int)

Calculating ∂W(ra-rb,h)/∂h.

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `r::Float64`: The geometric distance bewteen ra and rb.
- `dim:Int`: The dimension for calculating.

# Returns
- `Float64`: ∂W(ra-rb,h)/∂h
"""
function Smoothed_dh_kernel_function(
    f::Function,
    h::Union{Float32,Float64},
    r::Float64,
    dim::Int,
)
    q::Float64 = r / h
    hr::Float64 = h^(-(dim + 1))
    influence::Float64 =
        -hr *
        (3 * f(q) + q * KernelFunctionDiff()[nameof(f)](q)) *
        KernelFunctionnorm()[nameof(f)][dim]
    return influence
end

# Line-of-Sight Integrated Kernel function
"""
    function LOSint_Smoothed_kernel_function_dimensionless(f::Function, h::Union{Float32,Float64}, ra::Vector, rb::Vector)

Calculating dimensionless part of Line-of-Sight Integrated ∫W(ra-rb,h)dqz.

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `r::Float64`: The geometric distance bewteen ra and rb.
- `dim:Int`: The dimension for calculating. (before integration)

# Returns
- `Float64`: ∫W(ra-rb,h)dqz
"""
function LOSint_Smoothed_kernel_function_dimensionless(
    f::Function,
    h::Union{Float32,Float64},
    r::Float64,
    dim::Int,
)
    qxy::Float64 = r / h
    R :: Float64 = KernelFunctionValid()[nameof(f)]
    if qxy > R
        return 0
    else
        qz_min = -(sqrt(R^2 - qxy^2))
        qz_max = sqrt(R^2 - qxy^2)
        integrand(qz) = f(sqrt(qxy^2 + qz^2))
        result, _ = quadgk(integrand, qz_min, qz_max)
        influence::Float64 = result * KernelFunctionnorm()[nameof(f)][dim]
        return influence
    end
end

"""
    function LOSint_Smoothed_kernel_function(f::Function, h::Union{Float32,Float64}, ra::Vector, rb::Vector)

Calculating dimensionless part of Line-of-Sight Integrated ∫W(ra-rb,h)dqz.

# Parameters
- `f::Function`: Kernel function. 
- `h::Union{Float32,Float64}`: Smoothed length.
- `r::Float64`: The geometric distance bewteen ra and rb.
- `dim:Int`: The dimension for calculating. (before integration)

# Returns
- `Float64`: (1/h^(dim-1))∫W(ra-rb,h)dqz
"""
function LOSint_Smoothed_kernel_function(
    f::Function,
    h::Union{Float32,Float64},
    r::Float64,
    dim::Int,
)   
    hr :: Float64 = h^(-dim+1)
    influence :: Float64 = hr * LOSint_Smoothed_kernel_function_dimensionless(f,h,r,dim)
    return influence
end
