using Test
using Statistics
using PhantomRevealer

const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const NS = PhantomRevealer.NeighborSearch
const MassFromParams = PhantomRevealer.MassFromParams

function _neighbor_selection(pool, stack, lbvh, strategy, point, multiplier, h_values)
    if strategy == KI.itpSymmetric || strategy == KI.itpScatter
        multiplier *= 2.5
    elseif strategy == KI.itpGather
        multiplier *= 1.5
    else
        throw(ArgumentError("Unknown strategy: $(strategy)"))
    end
    ha = mean(h_values)
    gather_radius = ha * multiplier
    selection = NS.LBVH_query!(pool, lbvh, point, gather_radius)
    nearest = NS.nearest_index(selection)
    ha = h_values[nearest]
    radius = multiplier * ha
    selection = NS.LBVH_query!(pool, lbvh, point, radius)
    nearest = selection.count == 0 ? nearest : NS.nearest_index(selection)
    return selection, h_values[nearest]
end
