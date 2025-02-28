"""
Spiral detection in a Face-on data by using the ridge detection technique from Lindeberg(1996)(doi=10.1023/A:1008097225773)
    by Wei-Shan Su,
    Augest 30, 2024
"""

mutable struct pattern_tracer
    detected_points :: Vector{Tuple{Float64, Float64}}
    coord :: String
    flag :: Int64
end

function _tuples2matrix(v::Vector{Tuple{Float64, Float64}})
    matrix = vcat(transpose([t[1] for t in v]), transpose([t[2] for t in v]))
    return matrix
end

