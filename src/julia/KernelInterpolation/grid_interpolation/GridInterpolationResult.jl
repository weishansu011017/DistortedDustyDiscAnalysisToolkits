struct GridInterpolationResult{L}
    grids :: NTuple{L, <:AbstractInterpolationGrid}
    coors :: NTuple{L, Symbol}
end