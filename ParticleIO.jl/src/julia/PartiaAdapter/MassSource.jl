abstract type AbstractMassSource end

struct MassFromColumn <: AbstractMassSource
    name::Symbol
end

struct MassFromParams <: AbstractMassSource
    name::Symbol
end
