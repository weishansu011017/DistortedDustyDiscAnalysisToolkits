struct GridInterpolationResult{L}
    grids :: NTuple{L, <:AbstractInterpolationGrid}
    order :: NTuple{L, Symbol}
end