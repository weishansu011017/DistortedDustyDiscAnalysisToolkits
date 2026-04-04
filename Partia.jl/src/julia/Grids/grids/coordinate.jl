# Dispatch tag for constructing grid
abstract type AbstractCoordinateSystem end

struct Cartesian       <: AbstractCoordinateSystem end 
struct Polar           <: AbstractCoordinateSystem end        # (s, ϕ)
struct Cylindrical     <: AbstractCoordinateSystem end        # (s, ϕ, z)  
struct Spherical       <: AbstractCoordinateSystem end        # (r, ϕ, θ)


@inline function _coordinate_grid_isapprox(
    actual::NTuple{D,VA},
    expected::NTuple{D,VE};
    atol::Real = 1.0e-8,
    rtol::Real = 1.0e-8,
) where {D, T <: AbstractFloat,VA <: AbstractVector{T}, VE <: AbstractVector{T}}
    @inbounds for d in 1:D
        length(actual[d]) == length(expected[d]) || return false
        for i in eachindex(actual[d], expected[d])
            isapprox(actual[d][i], expected[d][i]; atol = atol, rtol = rtol) || return false
        end
    end
    return true
end
