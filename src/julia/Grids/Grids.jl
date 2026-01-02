module Grids
using .Threads
using Statistics
using Adapt

# Flag of coordinate system
include(joinpath(@__DIR__, "grids", "coordinate.jl"))

# AbstractGrid
include(joinpath(@__DIR__, "grids", "AbstractGrid.jl"))

# GeneralGrid
include(joinpath(@__DIR__, "grids", "GeneralGrid.jl"))

# StructuredGrid
include(joinpath(@__DIR__, "grids", "StructuredGrid.jl"))

# Transfromation between StructuredGrid and GeneralGrid
include(joinpath(@__DIR__, "grids", "transform.jl"))

# GridBundle
include(joinpath(@__DIR__,  "griddataset", "GridBundle.jl"))

# GridDataset
include(joinpath(@__DIR__,  "griddataset", "GridDataset.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end