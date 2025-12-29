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
"""
    SoundSpeed(::Type{Adiabatic}, u::T, γ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic sound speed from the specific internal energy `u` and 
adiabatic index `γ`, using

    c_s = √[ γ (γ - 1) u ].
    
Returns `NaN` if `u < 0` or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic sound speed.
- `u::T` : Specific internal energy.
- `γ::T` : Adiabatic index.

# Returns
- `T` : The computed sound speed, or `NaN` if input is unphysical.
"""
@inline function SoundSpeed(::Type{Adiabatic}, u::T, γ::T) :: T where {T<:AbstractFloat} 
    if u < zero(T)
        return T(NaN)
    elseif γ < one(T)
        return T(NaN)
    else
        return sqrt(γ * (γ - one(T)) * u)
    end
end

"""
    SoundSpeed(::Type{Isothermal}, cs::T)  where {T<:AbstractFloat}

Return the isothermal sound speed. In an isothermal equation of state, the 
sound speed is constant and equal to the input value.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal sound speed.
- `cs::T` : Prescribed constant sound speed.

# Returns
- `T` : The same value `cs`, representing the isothermal sound speed.
"""
@inline SoundSpeed(::Type{Isothermal}, cs::T) where {T<:AbstractFloat} = cs

"""
    SoundSpeed(::Type{LocallyIsothermal}, r::T, cs0::T, q::T) :: T where {T<:AbstractFloat}

Compute the locally isothermal sound speed profile, defined as

    c_s(r) = c_{s0} * r^(-q),

where `c_{s0}` is a reference sound speed and `q` is the radial power-law index.  
Returns `NaN` if `r ≤ 0` or `cs0 < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal sound speed.
- `r::T` : Radial position.
- `cs0::T` : Reference sound speed at `r = 1`.
- `q::T` : Power-law exponent controlling radial dependence.

# Returns
- `T` : The locally isothermal sound speed at radius `r`, or `NaN` if input is unphysical.
"""
@inline function SoundSpeed(::Type{LocallyIsothermal}, r::T, cs0::T, q::T) :: T where {T<:AbstractFloat}
    if r ≤ zero(T)
        return T(NaN)
    elseif cs0 < zero(T)
        return T(NaN)
    else
        return cs0 * r^(-q)
    end
end


## Sound speed evaluation (Promote)
"""
    SoundSpeed(::Type{Adiabatic}, u::AbstractFloat, γ::AbstractFloat)

Compute the adiabatic sound speed with automatic type promotion, using

    c_s = √[ γ (γ - 1) u ],

where `u` is the specific internal energy and `γ` is the adiabatic index.  
Returns `NaN` if `u < 0` or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic sound speed.
- `u::AbstractFloat` : Specific internal energy.
- `γ::AbstractFloat` : Adiabatic index.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  computed sound speed, or `NaN` if input is unphysical.
"""
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

"""
    SoundSpeed(::Type{LocallyIsothermal}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat)

Compute the locally isothermal sound speed with automatic type promotion, defined as

    c_s(r) = c_{s0} * r^(-q),

where `c_{s0}` is a reference sound speed and `q` is the radial power-law index.  
Returns `NaN` if `r ≤ 0` or `cs0 < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal sound speed.
- `r::AbstractFloat` : Radial position.
- `cs0::AbstractFloat` : Reference sound speed at `r = 1`.
- `q::AbstractFloat` : Power-law exponent controlling radial dependence.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  locally isothermal sound speed at radius `r`, or `NaN` if input is unphysical.
"""
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
"""
    Pressure(::Type{Adiabatic}, ρ::T, u::T, γ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic gas pressure using the equation of state

    P = (γ - 1) ρ u,

where `ρ` is the mass density, `u` is the specific internal energy, and `γ` is the adiabatic index.  
Returns `NaN` if `ρ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic pressure calculation.
- `ρ::T` : Mass density.
- `u::T` : Specific internal energy.
- `γ::T` : Adiabatic index.

# Returns
- `T` : The computed pressure, or `NaN` if input is unphysical.
"""
@inline function Pressure(::Type{Adiabatic}, ρ::T, u::T, γ::T) :: T where {T<:AbstractFloat}
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

"""
    Pressure(::Type{Isothermal}, ρ::T, cs::T) :: T where {T<:AbstractFloat}

Compute the isothermal gas pressure using the equation of state

    P = ρ c_s²,

where `ρ` is the mass density and `c_s` is the isothermal sound speed.  
Returns `NaN` if `ρ < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal pressure calculation.
- `ρ::T` : Mass density.
- `cs::T` : Constant isothermal sound speed.

# Returns
- `T` : The computed pressure, or `NaN` if input is unphysical.
"""
@inline function Pressure(::Type{Isothermal}, ρ::T, cs::T) :: T  where {T<:AbstractFloat}
    if ρ < zero(T)
        return T(NaN)
    else
        return ρ * cs^2
    end
end

"""
    Pressure(::Type{LocallyIsothermal}, ρ::T, r::T, cs0::T, q::T) :: T where {T<:AbstractFloat}

Compute the locally isothermal gas pressure profile, defined as

    P(r) = ρ [ c_{s0} * r^(-q) ]²,

where `ρ` is the mass density, `c_{s0}` is a reference sound speed, and `q` is the radial power-law index.  
Returns `NaN` if `ρ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal pressure calculation.
- `ρ::T` : Mass density.
- `r::T` : Radial position.
- `cs0::T` : Reference sound speed at `r = 1`.
- `q::T` : Power-law exponent controlling radial dependence.

# Returns
- `T` : The locally isothermal pressure at radius `r`, or `NaN` if input is unphysical.
"""
@inline function Pressure(::Type{LocallyIsothermal}, ρ::T, r::T, cs0::T, q::T) :: T  where {T<:AbstractFloat}
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
"""
    Pressure(::Type{Adiabatic}, ρ::AbstractFloat, u::AbstractFloat, γ::AbstractFloat)

Compute the adiabatic gas pressure with automatic type promotion, using the equation of state

    P = (γ - 1) ρ u,

where `ρ` is the mass density, `u` is the specific internal energy, and `γ` is the adiabatic index.  
Returns `NaN` if `ρ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic pressure calculation.
- `ρ::AbstractFloat` : Mass density.
- `u::AbstractFloat` : Specific internal energy.
- `γ::AbstractFloat` : Adiabatic index.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  computed pressure, or `NaN` if input is unphysical.
"""
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

"""
    Pressure(::Type{Isothermal}, ρ::AbstractFloat, cs::AbstractFloat)

Compute the isothermal gas pressure with automatic type promotion, using the equation of state

    P = ρ c_s²,

where `ρ` is the mass density and `c_s` is the isothermal sound speed.  
Returns `NaN` if `ρ < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal pressure calculation.
- `ρ::AbstractFloat` : Mass density.
- `cs::AbstractFloat` : Constant isothermal sound speed.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  computed pressure, or `NaN` if input is unphysical.
"""
@inline function Pressure(::Type{Isothermal}, ρ::AbstractFloat, cs::AbstractFloat)
    ρp, csp = promote(ρ, cs)
    T = typeof(ρp)
    if ρp < zero(T)
        return T(NaN)
    else
        return ρp * csp^2
    end
end

"""
    Pressure(::Type{LocallyIsothermal}, ρ::AbstractFloat, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat)

Compute the locally isothermal gas pressure with automatic type promotion, defined as

    P(r) = ρ [ c_{s0} * r^(-q) ]²,

where `ρ` is the mass density, `c_{s0}` is a reference sound speed, and `q` is the radial power-law index.  
Returns `NaN` if `ρ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal pressure calculation.
- `ρ::AbstractFloat` : Mass density.
- `r::AbstractFloat` : Radial position.
- `cs0::AbstractFloat` : Reference sound speed at `r = 1`.
- `q::AbstractFloat` : Power-law exponent controlling radial dependence.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  locally isothermal pressure at radius `r`, or `NaN` if input is unphysical.
"""
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
"""
    Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic gas temperature in SI units from the specific internal energy, 
adiabatic index, and mean molecular weight. The formula is

    T = (m_p / k_B) * μ (γ - 1) u,

where `m_p` is the proton mass and `k_B` is the Boltzmann constant.  
Here, the constant factor `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]` in SI units.  
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `u::T` : Specific internal energy (m² s⁻²).
- `γ::T` : Adiabatic index.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}
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

"""
    Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}

Compute the isothermal gas temperature in SI units from the isothermal sound speed and 
mean molecular weight. The formula is

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed.  
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.  
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `cs::T` : Isothermal sound speed.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}
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

"""
    Temperature(::Type{LocallyIsothermal}, ::Type{SIUnit}, r::T, cs0::T, q::T, μ::T) :: T where {T<:AbstractFloat}

Compute the locally isothermal gas temperature in SI units from the radial position, 
reference sound speed, radial power-law index, and mean molecular weight. The formula is

    T(r) = (m_p / k_B) * μ [ c_{s0} * r^(-q) ]²,

where `c_{s0}` is the reference sound speed at `r = 1` and `q` is the radial power-law index.  
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.  
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `c_{s0} < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `r::T` : Radial position.
- `cs0::T` : Reference sound speed at `r = 1`.
- `q::T` : Power-law exponent controlling radial dependence.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{LocallyIsothermal}, ::Type{SIUnit}, r::T, cs0::T, q::T, μ::T) :: T where {T<:AbstractFloat}
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
"""
    Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic gas temperature in CGS units from the specific internal energy, 
adiabatic index, and mean molecular weight. The formula is

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in cm² s⁻².  
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.  
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `u::T` : Specific internal energy (cm² s⁻²).
- `γ::T` : Adiabatic index.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}
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

"""
    Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}

Compute the isothermal gas temperature in CGS units from the isothermal sound speed 
and mean molecular weight. The formula is

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in cm s⁻¹.  
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.  
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `cs::T` : Isothermal sound speed (cm s⁻¹).
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}
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

"""
    Temperature(::Type{LocallyIsothermal}, ::Type{CGSUnit}, r::T, cs0::T, q::T, μ::T) :: T where {T<:AbstractFloat}

Compute the locally isothermal gas temperature in CGS units from the radial position, 
reference sound speed, radial power-law index, and mean molecular weight. The formula is

    T(r) = (m_p / k_B) * μ [ c_{s0} * r^(-q) ]²,

where `c_{s0}` is the reference sound speed at `r = 1` and `q` is the radial power-law index.  
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.  
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `c_{s0} < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `r::T` : Radial position.
- `cs0::T` : Reference sound speed at `r = 1` (cm s⁻¹).
- `q::T` : Power-law exponent controlling radial dependence.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{LocallyIsothermal}, ::Type{CGSUnit}, r::T, cs0::T, q::T, μ::T) :: T where {T<:AbstractFloat}
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
"""
    Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic gas temperature in Galactic units from the specific internal energy, 
adiabatic index, and mean molecular weight. The formula is

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in km² s⁻².  
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.  
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `u::T` : Specific internal energy (km² s⁻²).
- `γ::T` : Adiabatic index.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}
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

"""
    Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}

Compute the isothermal gas temperature in Galactic units from the isothermal sound speed 
and mean molecular weight. The formula is

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in km s⁻¹.  
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.  
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `cs::T` : Isothermal sound speed (km s⁻¹).
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}
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

"""
    Temperature(::Type{LocallyIsothermal}, ::Type{GalacticUnit}, r::T, cs0::T, q::T, μ::T) :: T where {T<:AbstractFloat}

Compute the locally isothermal gas temperature in Galactic units from the radial position, 
reference sound speed, radial power-law index, and mean molecular weight. The formula is

    T(r) = (m_p / k_B) * μ [ c_{s0} * r^(-q) ]²,

where `c_{s0}` is the reference sound speed at `r = 1` and `q` is the radial power-law index.  
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.  
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `c_{s0} < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `r::T` : Radial position.
- `cs0::T` : Reference sound speed at `r = 1` (km s⁻¹).
- `q::T` : Power-law exponent controlling radial dependence.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{LocallyIsothermal}, ::Type{GalacticUnit}, r::T, cs0::T, q::T, μ::T) :: T where {T<:AbstractFloat}
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
"""
    Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)

Compute the adiabatic gas temperature in SI units with automatic type promotion, using

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in m² s⁻².  
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.  
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `u::AbstractFloat` : Specific internal energy (m² s⁻²).
- `γ::AbstractFloat` : Adiabatic index.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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

"""
    Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::AbstractFloat, μ::AbstractFloat)

Compute the isothermal gas temperature in SI units with automatic type promotion, using

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in m s⁻¹.  
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.  
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `cs::AbstractFloat` : Isothermal sound speed (m s⁻¹).
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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

"""
    Temperature(::Type{LocallyIsothermal}, ::Type{SIUnit}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat, μ::AbstractFloat)

Compute the locally isothermal gas temperature in SI units with automatic type promotion, using

    T(r) = (m_p / k_B) * μ [ c_{s0} * r^(-q) ]²,

where `c_{s0}` is the reference sound speed at `r = 1`, `q` is the radial power-law index, 
and `r` is the radial position.  
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.  
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `c_{s0} < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `r::AbstractFloat` : Radial position.
- `cs0::AbstractFloat` : Reference sound speed at `r = 1` (m s⁻¹).
- `q::AbstractFloat` : Power-law exponent controlling radial dependence.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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
"""
    Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)

Compute the adiabatic gas temperature in CGS units with automatic type promotion, using

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in cm² s⁻².  
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.  
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `u::AbstractFloat` : Specific internal energy (cm² s⁻²).
- `γ::AbstractFloat` : Adiabatic index.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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

"""
    Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::AbstractFloat, μ::AbstractFloat)

Compute the isothermal gas temperature in CGS units with automatic type promotion, using

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in cm s⁻¹.  
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.  
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `cs::AbstractFloat` : Isothermal sound speed (cm s⁻¹).
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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

"""
    Temperature(::Type{LocallyIsothermal}, ::Type{CGSUnit}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat, μ::AbstractFloat)

Compute the locally isothermal gas temperature in CGS units with automatic type promotion, using

    T(r) = (m_p / k_B) * μ [ c_{s0} * r^(-q) ]²,

where `c_{s0}` is the reference sound speed at `r = 1`, `q` is the radial power-law index, 
and `r` is the radial position.  
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.  
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `c_{s0} < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `r::AbstractFloat` : Radial position.
- `cs0::AbstractFloat` : Reference sound speed at `r = 1` (cm s⁻¹).
- `q::AbstractFloat` : Power-law exponent controlling radial dependence.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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
"""
    Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)

Compute the adiabatic gas temperature in Galactic units with automatic type promotion, using

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in km² s⁻².  
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.  
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `u::AbstractFloat` : Specific internal energy (km² s⁻²).
- `γ::AbstractFloat` : Adiabatic index.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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

"""
    Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::AbstractFloat, μ::AbstractFloat)

Compute the isothermal gas temperature in Galactic units with automatic type promotion, using

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in km s⁻¹.  
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.  
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `cs::AbstractFloat` : Isothermal sound speed (km s⁻¹).
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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

"""
    Temperature(::Type{LocallyIsothermal}, ::Type{GalacticUnit}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat, μ::AbstractFloat)

Compute the locally isothermal gas temperature in Galactic units with automatic type promotion, using

    T(r) = (m_p / k_B) * μ [ c_{s0} * r^(-q) ]²,

where `c_{s0}` is the reference sound speed at `r = 1`, `q` is the radial power-law index, 
and `r` is the radial position.  
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.  
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `c_{s0} < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `r::AbstractFloat` : Radial position.
- `cs0::AbstractFloat` : Reference sound speed at `r = 1` (km s⁻¹).
- `q::AbstractFloat` : Power-law exponent controlling radial dependence.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the 
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
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