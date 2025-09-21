"""
    The calculation of properties for different equation of state in Phantom
            by Wei-Shan Su,
            September 19, 2025

"""

# EOS Types
abstract type AbstractEOS end
struct Adiabatic <: AbstractEOS end
struct Isothermal <: AbstractEOS end
struct LocallyIsothermal <: AbstractEOS end

# Unit Type
abstract type AbstractUnit end
struct SIUnit <: AbstractUnit end
struct CGSUnit <: AbstractUnit end
## Astonomical Type
abstract type AstronomicalUnit <: AbstractUnit end
struct StarUnit <: AstronomicalUnit end             # mass: M⊙, dist: AU
struct GalacticUnit <: AstronomicalUnit end         # velocity: km/s, distance: kpc

# Sound speed from Equation of State
## Sound speed evaluation (Same type)
@inline function SoundSpeed(::Type{Adiabatic}, u::T, γ::T) where {T<:AbstractFloat}
    if u < zero(T)
        return T(NaN)
    elseif γ < one(T)
        return T(NaN)
    else
        return sqrt(γ * (γ - one(T)) * u)
    end
end
@inline SoundSpeed(::Type{Isothermal}, cs::T) where {T<:AbstractFloat} = cs
@inline function SoundSpeed(::Type{LocallyIsothermal}, r::T, cs0::T, q::T) where {T<:AbstractFloat}
    if r ≤ zero(T)
        return T(NaN)
    elseif cs0 < zero(T)
        return T(NaN)
    else
        return cs0 * r^(-q)
    end
end


## Sound speed evaluation (Promote)
@inline function SoundSpeed(::Type{Adiabatic}, u::AbstractFloat, γ::AbstractFloat)
    up, γp = promote(u, γ)
    T = typeof(γp)
    if up < zero(T)
        return T(NaN)
    elseif γp < one(T)
        return T(NaN)
    else
        return sqrt(γp * (γp - one(T)) * up)
    end
end
@inline function SoundSpeed(::Type{LocallyIsothermal}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat)
    rp, cs0p, qp = promote(r, cs0, q)
    T = typeof(rp)
    if rp ≤ zero(T)
        return T(NaN)
    elseif cs0p < zero(T)
        return T(NaN)
    else
        return cs0p * rp^(-qp)
    end
end

# Pressure from Equation of State
## Pressure evaluation (Same type)
@inline function Pressure(::Type{Adiabatic}, ρ::T, u::T, γ::T) where {T<:AbstractFloat}
    if ρ < zero(T)
        return T(NaN)
    elseif u < zero(T)
        return T(NaN)
    elseif γ < one(T)
        return T(NaN)
    else
        return (γ - one(T)) * ρ * u
    end
end
@inline function Pressure(::Type{Isothermal}, ρ::T, cs::T) where {T<:AbstractFloat} 
    if ρ < zero(T)
        return T(NaN)
    else
        return ρ * cs^2
    end
end
   
@inline function Pressure(::Type{LocallyIsothermal}, ρ::T, r::T, cs0::T, q::T) where {T<:AbstractFloat} 
    if ρ < zero(T)
        return T(NaN)
    elseif r ≤ zero(T)
        return T(NaN)
    elseif cs0 < zero(T)
        return T(NaN)
    else
        return ρ * (cs0 * r^(-q))^2    
    end
end                       

## Pressure evaluation (Promote)
@inline function Pressure(::Type{Adiabatic}, ρ::AbstractFloat, u::AbstractFloat, γ::AbstractFloat)
    ρp, up, γp = promote(ρ, u, γ)
    T = typeof(ρp)
    if ρp < zero(T)
        return T(NaN)
    elseif up < zero(T)
        return T(NaN)
    elseif γp < one(T)
        return T(NaN)
    else
        return (γp - one(T)) * ρp * up
    end
end

@inline function Pressure(::Type{Isothermal}, ρ::AbstractFloat, cs::AbstractFloat)
    ρp, csp = promote(ρ, cs)
    T = typeof(ρp)
    if ρp < zero(T)
        return T(NaN)
    else
        return ρp * csp^2
    end
end

@inline function Pressure(::Type{LocallyIsothermal}, ρ::AbstractFloat, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat)
    ρp, rp, cs0p, qp = promote(ρ, r, cs0, q)  
    T = typeof(ρp)
    if ρp < zero(T)
        return T(NaN)
    elseif rp ≤ zero(T)
        return T(NaN)
    elseif cs0p < zero(T)
        return T(NaN)
    else
        return ρp * (cs0p * rp^(-qp))^2    
    end
end

# Temperature from Equation of state
## Temperature evaluation (Same type)
### SI
@inline function Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::T, γ::T, μ::T) where {T<:AbstractFloat}
    # u: m² s⁻²
    mplkB = T(0.00012114751277768644)               # K kg J⁻¹
    if μ < zero(T)
        return T(NaN)
    elseif u < zero(T)
        return T(NaN)
    elseif γ < one(T)
        return T(NaN)
    else
        return mplkB * (μ * (γ - one(T)) * u)
    end
end

@inline function Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::T, μ::T) where {T<:AbstractFloat}
    # u: m² s⁻²
    mplkB = T(0.00012114751277768644)               # K kg J⁻¹
    if μ < zero(T)
        return T(NaN)
    elseif cs < zero(T)
        return T(NaN)
    else
        return mplkB * (μ * cs^2)
    end
end

@inline function Temperature(::Type{LocallyIsothermal}, ::Type{SIUnit}, r::T, cs0::T, q::T, μ::T) where {T<:AbstractFloat}
    # u: m² s⁻²
    mplkB = T(0.00012114751277768644)               # K kg J⁻¹
    if μ < zero(T)
        return T(NaN)
    elseif r ≤ zero(T)
        return T(NaN)
    elseif cs0 < zero(T)
        return T(NaN)
    else
        cs = cs0 * r^(-q)
        return mplkB * (μ * cs^2)
    end
end

### CGS
@inline function Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::T, γ::T, μ::T) where {T<:AbstractFloat}
    # u: cm² s⁻²
    mplkB = T(1.2114751277768644e-8)               # K g erg⁻¹
    if μ < zero(T)
        return T(NaN)
    elseif u < zero(T)
        return T(NaN)
    elseif γ < one(T)
        return T(NaN)
    else
        return mplkB * (μ * (γ - one(T)) * u)
    end
end

@inline function Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::T, μ::T) where {T<:AbstractFloat}
    # u: cm² s⁻²
    mplkB = T(1.2114751277768644e-8)               # K g erg⁻¹
    if μ < zero(T)
        return T(NaN)
    elseif cs < zero(T)
        return T(NaN)
    else
        return mplkB * (μ * cs^2)
    end
end

@inline function Temperature(::Type{LocallyIsothermal}, ::Type{CGSUnit}, r::T, cs0::T, q::T, μ::T) where {T<:AbstractFloat}
    # u: cm² s⁻²
    mplkB = T(1.2114751277768644e-8)               # K g erg⁻¹
    if μ < zero(T)
        return T(NaN)
    elseif r ≤ zero(T)
        return T(NaN)
    elseif cs0 < zero(T)
        return T(NaN)
    else
        cs = cs0 * r^(-q)
        return mplkB * (μ * cs^2)
    end
end

### Galatic
@inline function Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::T, γ::T, μ::T) where {T<:AbstractFloat}
    # u: km² s⁻²
    mplkB = T(121.14751277768644)               # K s² km⁻²
    if μ < zero(T)
        return T(NaN)
    elseif u < zero(T)
        return T(NaN)
    elseif γ < one(T)
        return T(NaN)
    else
        return mplkB * (μ * (γ - one(T)) * u)
    end
end

@inline function Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::T, μ::T) where {T<:AbstractFloat}
    # u: km² s⁻²
    mplkB = T(121.14751277768644)               # K s² km⁻²
    if μ < zero(T)
        return T(NaN)
    elseif cs < zero(T)
        return T(NaN)
    else
        return mplkB * (μ * cs^2)
    end
end

@inline function Temperature(::Type{LocallyIsothermal}, ::Type{GalacticUnit}, r::T, cs0::T, q::T, μ::T) where {T<:AbstractFloat}
    # u: km² s⁻²
    mplkB = T(121.14751277768644)               # K s² km⁻²
    if μ < zero(T)
        return T(NaN)
    elseif r ≤ zero(T)
        return T(NaN)
    elseif cs0 < zero(T)
        return T(NaN)
    else
        cs = cs0 * r^(-q)
        return mplkB * (μ * cs^2)
    end
end

## Temperature evaluation (Promote)
### SI
@inline function Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)
    # u: m² s⁻²
    up, γp, μp = promote(u, γ, μ)
    T = typeof(up)
    mplkB = T(0.00012114751277768644)               # K kg J⁻¹
    if μp < zero(T)
        return T(NaN)
    elseif up < zero(T)
        return T(NaN)
    elseif γp < one(T)
        return T(NaN)
    else
        return mplkB * (μp * (γp - one(T)) * up)
    end
end

@inline function Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::AbstractFloat, μ::AbstractFloat)
    # u: m² s⁻²
    csp, μp = promote(cs, μ)
    T = typeof(csp)
    mplkB = T(0.00012114751277768644)               # K kg J⁻¹
    if μp < zero(T)
        return T(NaN)
    elseif csp < zero(T)
        return T(NaN)
    else
        return mplkB * (μp * csp^2)
    end
end

@inline function Temperature(::Type{LocallyIsothermal}, ::Type{SIUnit}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat, μ::AbstractFloat)
    # u: m² s⁻²
    rp, cs0p, qp, μp = promote(r, cs0, q, μ)
    T = typeof(cs0p)
    mplkB = T(0.00012114751277768644)               # K kg J⁻¹
    if μp < zero(T)
        return T(NaN)
    elseif rp ≤ zero(T)
        return T(NaN)
    elseif cs0p < zero(T)
        return T(NaN)
    else
        cs = cs0p * rp^(-qp)
        return mplkB * (μp * cs^2)
    end
end

### CGS
@inline function Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)
    # u: cm² s⁻²
    up, γp, μp = promote(u, γ, μ)
    T = typeof(up)
    mplkB = T(1.2114751277768644e-8)               # K g erg⁻¹
    if μp < zero(T)
        return T(NaN)
    elseif up < zero(T)
        return T(NaN)
    elseif γp < one(T)
        return T(NaN)
    else
        return mplkB * (μp * (γp - one(T)) * up)
    end
end

@inline function Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::AbstractFloat, μ::AbstractFloat)
    # u: cm² s⁻²
    csp, μp = promote(cs, μ)
    T = typeof(csp)
    mplkB = T(1.2114751277768644e-8)               # K g erg⁻¹
    if μp < zero(T)
        return T(NaN)
    elseif csp < zero(T)
        return T(NaN)
    else
        return mplkB * (μp * csp^2)
    end
end

@inline function Temperature(::Type{LocallyIsothermal}, ::Type{CGSUnit}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat, μ::AbstractFloat)
    # u: cm² s⁻²
    rp, cs0p, qp, μp = promote(r, cs0, q, μ)
    T = typeof(cs0p)
    mplkB = T(1.2114751277768644e-8)               # K g erg⁻¹
    if μp < zero(T)
        return T(NaN)
    elseif rp ≤ zero(T)
        return T(NaN)
    elseif cs0p < zero(T)
        return T(NaN)
    else
        cs = cs0p * rp^(-qp)
        return mplkB * (μp * cs^2)
    end
end

### Galatic
@inline function Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)
    # u: km² s⁻²
    up, γp, μp = promote(u, γ, μ)
    T = typeof(up)
    mplkB = T(121.14751277768644)               # K s² km⁻²
    if μp < zero(T)
        return T(NaN)
    elseif up < zero(T)
        return T(NaN)
    elseif γp < one(T)
        return T(NaN)
    else
        return mplkB * (μp * (γp - one(T)) * up)
    end
end

@inline function Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::AbstractFloat, μ::AbstractFloat)
    # u: km² s⁻²
    csp, μp = promote(cs, μ)
    T = typeof(csp)
    mplkB = T(121.14751277768644)               # K s² km⁻²
    if μp < zero(T)
        return T(NaN)
    elseif csp < zero(T)
        return T(NaN)
    else
        return mplkB * (μp * csp^2)
    end
end

@inline function Temperature(::Type{LocallyIsothermal}, ::Type{GalacticUnit}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat, μ::AbstractFloat)
    # u: km² s⁻²
    rp, cs0p, qp, μp = promote(r, cs0, q, μ)
    T = typeof(cs0p)
    mplkB = T(121.14751277768644)               # K s² km⁻²
    if μp < zero(T)
        return T(NaN)
    elseif rp ≤ zero(T)
        return T(NaN)
    elseif cs0p < zero(T)
        return T(NaN)
    else
        cs = cs0p * rp^(-qp)
        return mplkB * (μp * cs^2)
    end
end