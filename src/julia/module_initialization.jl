# Pkg Module
const MODULE_LIST = [
    :CairoMakie,
    :Clustering,
    :CSV,
    :Colors,
    :ColorSchemes,
    :DataFrames,
    :DataStructures,
    :Dates,
    :FilesystemDatastructures,
    :ForwardDiff,
    :GeometryBasics,
    :HDF5,
    :ImageFiltering,
    :Interpolations,
    :LaTeXStrings,
    :LinearAlgebra,
    :Logging,
    :LoggingExtras,
    :LsqFit,
    :Makie,
    :MakieCore,
    :NearestNeighbors,
    :Pkg,
    :Printf,
    :QuadGK,
    :Requires,
    :Statistics,
    :StatsBase,
]
for mod in MODULE_LIST
    @eval using $(Symbol(mod))
end
@eval using Base.Sys
@eval using Base.Threads
@eval using Base.Iterators
