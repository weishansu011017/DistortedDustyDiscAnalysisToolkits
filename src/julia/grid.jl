"""
The general grid construction for SPH interpolation
    by Wei-Shan Su,
    September 21, 2025
"""

# General Grid definition
abstract type AbstractInterpolationGrid end

struct GeneralGrid{D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: AbstractVector{NTuple{D, TF}}} <: AbstractInterpolationGrid
    grid :: VG
    coor :: VC
end

Base.length(::GeneralGrid)

# structured grid (Cartesian/Cylindrical... etc)
abstract type AbstractStructuredGrid{D} <: AbstractInterpolationGrid end
