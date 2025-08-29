# Pkg Module
const MODULE_LIST = [
    :BenchmarkTools,
    :CairoMakie,
    :Clustering,
    :CSV,
    :Colors,
    :ColorSchemes,
    :DataFrames,
    :DataStructures,
    :Dates,
    :FFTW,
    :FilesystemDatastructures,
    :ForwardDiff,
    :GeometryBasics,
    :HDF5,
    :ImageFiltering,
    :ImageMorphology,
    :Interpolations,
    :LaTeXStrings,
    :LinearAlgebra,
    :Logging,
    :LoggingExtras,
    :LsqFit,
    :Makie,
    :NearestNeighbors,
    :Pkg,
    :Printf,
    :QuadGK,
    :Requires,
    :StaticArrays,
    :Statistics,
    :StatsBase,
]
for mod in MODULE_LIST
    @eval using $(Symbol(mod))
end
@eval using Base.Sys
@eval using Base.Threads
@eval using Base.Iterators
