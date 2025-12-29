struct GridBundle{L, G <: AbstractGrid}
    grids :: NTuple{L, G}
    names :: NTuple{L, Symbol}
end

"""
    gridtype(gb::GridBundle{L,G}) where {L, G<:AbstractGrid}

Return the concrete grid element type `G` stored in a `GridBundle{L,G}`.

This is a lightweight helper for extracting the bundle's grid type parameter,
useful for dispatch and schema/metadata recording.

# Parameters
- `gb::GridBundle{L,G}`: A `GridBundle` whose element grid type is `G <: AbstractGrid`.

# Returns
- `Type{G}`: The concrete grid element type carried by the bundle.
"""
@inline gridtype(:: GridBundle{L, G}) where {L, G <: AbstractGrid} = G

"""
    datatype(gb::GridBundle{L,G}) where {L, G<:AbstractGrid}

Return the floating-point element type parameter `TF` carried by the grid type `G`
stored in `GridBundle{L,G}`.

This is equivalent to calling `datatype(G)` on the bundle's grid element type.

# Parameters
- `gb::GridBundle{L,G}`: A `GridBundle` whose element grid type is `G <: AbstractGrid`.

# Returns
- `Type{TF}`: The floating-point element type parameter of `G`.
"""
@inline datatype(:: GridBundle{L, G}) where {L, G <: AbstractGrid} = datatype(G)