# Dispatch tag for constructing grid
abstract type AbstractCoordinateSystem end

struct Cartesian       <: AbstractCoordinateSystem end 
struct Polar           <: AbstractCoordinateSystem end        # (s, ϕ)
struct Cylindrical     <: AbstractCoordinateSystem end        # (s, ϕ, z)  
struct Spherical       <: AbstractCoordinateSystem end        # (r, ϕ, θ)