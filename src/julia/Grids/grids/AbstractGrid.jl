# General Grid definition
abstract type AbstractGrid{TF} end

"""
    Base.length(grid::GRID) where {GRID <: AbstractGrid}

Return the number of elements stored in the grid values array.

This delegates to `length(grid.grid)`, i.e. the length of the internal 
storage vector for grid values.

# Parameters
- `grid::GRID` : Any concrete subtype of `AbstractGrid`.

# Returns
- `Int` : The number of stored grid values.
"""
@inline Base.length(grid :: GRID) where {GRID <: AbstractGrid} = length(grid.grid)

"""
    datatype(::Type{GRID}) where {TF<:AbstractFloat, GRID<:AbstractGrid{TF}}

Return the floating-point element type parameter `TF` of an `AbstractGrid{TF}` type.

This method extracts `TF` purely from the parametric type, without inspecting any
stored arrays or values.

# Parameters
- `::Type{GRID}`: A concrete grid type `GRID <: AbstractGrid{TF}`.

# Returns
- `Type{TF}`: The floating-point element type parameter of the grid type.
"""
@inline datatype(::Type{GRID}) where {TF <: AbstractFloat, GRID <: AbstractGrid{TF}} = TF