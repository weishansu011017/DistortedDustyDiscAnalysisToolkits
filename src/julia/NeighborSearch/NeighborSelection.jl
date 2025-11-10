struct NeighborSelection{TI <: Integer, VI <: AbstractVector{TI}}
    pool :: VI
    count :: TI
    nearest :: TI
end

Base.length(result::NeighborSelection) = result.count

@inline function valid_indices(result::NeighborSelection)
    return @view result.pool[1:result.count]
end

@inline function nearest_index(result::NeighborSelection)
    return result.nearest
end
