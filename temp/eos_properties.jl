"""
    The calculation of properties for different equation of state in Phantom
            by Wei-Shan Su,
            Augest 12, 2024

"""

"""
    eos6_cs(s::Float64,cs0::Float64,q::Float64)
The sound speed of locally isothermal EOS(ieos=6)

        cs(r) = cs0 * (s^(-q))

# Parameters
- `s :: Float64`: The radial distance to the center star in cylindrical coordinate.
- `cs0 :: Float64`: The sound speed at s=1
- `q :: Float64`: The q-index in the formula.

# Returns
- `Float64`: The sound speed on the given radius.
"""
function eos6_cs(s::Float64,cs0::Float64,q::Float64)
    return cs0 * (s^(-q))
end

"""
    eos6_P(ρ::Float64,s::Float64,cs0::Float64,q::Float64)
The pressure of locally isothermal EOS(ieos=6)

        P = cs^2 * ρ

# Parameters
- `ρ :: Float64`: The local density of the fluid.
- `s :: Float64`: The radial distance to the center star in cylindrical coordinate.
- `cs0 :: Float64`: The sound speed at s=1
- `q :: Float64`: The q-index in the formula.

# Returns
- `Float64`: The pressure on the given radius.
"""
function eos6_P(ρ::Float64,s::Float64,cs0::Float64,q::Float64)
    return ρ * eos6_cs(s,cs0,q)^2
end