"""
Linear Bounding Volume Hierarchy (LBVH) construction and traversal utilities  
    by Wei-Shan Su,  
    November 8, 2025
"""
struct AABB{D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    min :: NTuple{D, VF}
    max :: NTuple{D, VF}
end

struct LinearBVH{D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}, A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    enc :: MortonEncoding{D, TF, TI, VF, VI}
    brt :: BinaryRadixTree{TI, VI, A, B}
    leaf_aabb :: AABB{D, TF, VF}
    node_aabb :: AABB{D, TF, VF}
    root :: Int
end

@inline function _push!(stack::AbstractVector{Int}, top::Int, value::Int)
    new_top = top + 1
    @inbounds stack[new_top] = value
    return new_top
end

@inline function _pop!(stack::AbstractVector{Int}, top::Int)
    @inbounds value = stack[top]
    return value, top - 1
end

@inline function _find_root_index(brt::BinaryRadixTree)
    parents = brt.node_parent
    zero_parent = zero(eltype(parents))
    @inbounds for (idx, parent) in pairs(parents)
        if parent == zero_parent
            return idx
        end
    end
    return 0
end

@inline function _build_leaf_aabb!(LBVH::LinearBVH{D}) where {D}
    coords = LBVH.enc.coord
    leaf = LBVH.leaf_aabb
    @inbounds for d in 1:D
        copyto!(leaf.min[d], coords[d])
        copyto!(leaf.max[d], coords[d])
    end
    return nothing
end

@inline function _build_internal_aabb!(LBVH::LinearBVH{D}) where {D}
    brt = LBVH.brt
    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max

    L = brt.left_child
    R = brt.right_child
    LL = brt.is_leaf_left
    RR = brt.is_leaf_right

    nint = length(L)
    root = LBVH.root
    if nint == 0 || root == 0
        return nothing
    end

    visit = brt.visit_counter
    zero_visit = zero(eltype(visit))
    fill!(visit, zero_visit)
    one_visit = one(eltype(visit))

    stack = Vector{Int}(undef, max(1, 2 * nint + 8))
    top = _push!(stack, 0, root)

    while top > 0
        i, top = _pop!(stack, top)
        v = visit[i] + one_visit

        if v == one_visit
            visit[i] = v
            top = _push!(stack, top, i)
            if !RR[i]
                top = _push!(stack, top, R[i])
            end
            if !LL[i]
                top = _push!(stack, top, L[i])
            end
            continue
        end

        @inbounds for d in 1:D
            l = L[i]; r = R[i]
            lmin = LL[i] ? leaf_min[d][l] : node_min[d][l]
            rmin = RR[i] ? leaf_min[d][r] : node_min[d][r]
            lmax = LL[i] ? leaf_max[d][l] : node_max[d][l]
            rmax = RR[i] ? leaf_max[d][r] : node_max[d][r]
            node_min[d][i] = ifelse(lmin < rmin, lmin, rmin)
            node_max[d][i] = ifelse(lmax > rmax, lmax, rmax)
        end
        visit[i] = zero_visit
    end

    return nothing
end

@inline function _build_LBVH!(LBVH::LinearBVH)
    _build_leaf_aabb!(LBVH)
    _build_internal_aabb!(LBVH)
    return nothing
end

"""
        LinearBVH(enc::MortonEncoding, brt::BinaryRadixTree)

Assemble a linear bounding volume hierarchy from a Morton-encoded particle set
and its matching binary radix tree. This allocates per-leaf and per-node
axis-aligned bounding boxes, discovers the tree root, and precomputes the
hierarchical extent data required for subsequent neighbor queries.

# Parameters
- `enc::MortonEncoding`: Morton-sorted particle coordinates and permutation.
- `brt::BinaryRadixTree`: Connectivity generated from the same `enc` instance.

# Returns
- `LinearBVH`: Immutable hierarchy storing the encoding, tree topology, and
    bounding volumes.
"""
function LinearBVH(enc::MortonEncoding{D, TF, TI, VF, VI}, brt::BinaryRadixTree{TI, VI, A, B}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}, A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    nleaf = length(enc.coord[1])
    ninternal = max(nleaf - 1, 0)

    vproto = enc.coord[1]

    leaf_aabb = AABB(ntuple(_ -> similar(vproto, nleaf), D),
                     ntuple(_ -> similar(vproto, nleaf), D))
    node_aabb = AABB(ntuple(_ -> similar(vproto, ninternal), D),
                     ntuple(_ -> similar(vproto, ninternal), D))

    root = _find_root_index(brt)
    LBVH = LinearBVH{D, TF, TI, VF, VI, A, B}(enc, brt, leaf_aabb, node_aabb, root)
    _build_LBVH!(LBVH)
    return LBVH
end

"""
    LBVH_query!(pool, stack, lbvh, point, radius)

Traverse the LBVH around `point` and collect particle indices whose leaf AABBs
lie within the spherical query of radius `radius`. The provided buffers are
reused to avoid allocations; indices are written in Morton order into `pool`
and reported through a `NeighborSelection` handle.

# Parameters
- `pool::AbstractVector{Int}`: Scratch space receiving neighbor leaf indices.
- `stack::AbstractVector{Int}`: Scratch stack used by the depth-first traversal.
- `lbvh::LinearBVH{D,T}`: Hierarchy constructed from Morton-encoded particles.
- `point::NTuple{D,T}`: Query position in the same coordinate system as the BVH.
- `radius::Real`: Search radius; promoted to the LBVH element type if needed.

# Returns
- `NeighborSelection`: View of `pool` limited to the number of found neighbors.

# Notes
- When `lbvh.root == 0`, the query falls back to scanning all leaves.
"""
@inline function LBVH_query!(pool::VI, stack::VI, LBVH::LinearBVH{D, T},
                             point::NTuple{D, T},
                             radius::T) where {D, T <: AbstractFloat, VI <: AbstractVector{Int}}
    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max

    L = LBVH.brt.left_child
    R = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    root = LBVH.root

    r2 = radius * radius
    count = 0
    top = 0
    closest_idx = zero(eltype(pool))
    closest_dist2 = typemax(T)

    @inline function dist2_to_node(i::Int)
        s = zero(T)
        @inbounds for d in 1:D
            a = node_min[d][i]; b = node_max[d][i]; p = point[d]
            t = p < a ? (a - p) : (p > b ? (p - b) : zero(T))
            s += t * t
        end
        return s
    end

    @inline function leaf_dist2(i::Int)
        s = zero(T)
        @inbounds for d in 1:D
            a = leaf_min[d][i]; b = leaf_max[d][i]; p = point[d]
            t = p < a ? (a - p) : (p > b ? (p - b) : zero(T))
            s += t * t
        end
        return s
    end

    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            dist2 = leaf_dist2(leaf_idx)
            if dist2 <= r2
                count += 1
                @boundscheck count <= length(pool) || throw(BoundsError(pool, count))
                pool[count] = leaf_idx
                if dist2 < closest_dist2
                    closest_dist2 = dist2
                    closest_idx = leaf_idx
                end
            end
        end
        return NeighborSelection(pool, count, closest_idx)
    end

    top = _push!(stack, top, root)
    while top > 0
        node, top = _pop!(stack, top)
        if dist2_to_node(node) <= r2
            if LL[node]
                leaf_idx = L[node]
                dist2 = leaf_dist2(leaf_idx)
                if dist2 <= r2
                    count += 1
                    @inbounds pool[count] = leaf_idx
                    if dist2 < closest_dist2
                        closest_dist2 = dist2
                        closest_idx = leaf_idx
                    end
                end
            else
                top = _push!(stack, top, L[node])
            end

            if RR[node]
                leaf_idx = R[node]
                dist2 = leaf_dist2(leaf_idx)
                if dist2 <= r2
                    count += 1
                    @inbounds pool[count] = leaf_idx
                    if dist2 < closest_dist2
                        closest_dist2 = dist2
                        closest_idx = leaf_idx
                    end
                end
            else
                top = _push!(stack, top, R[node])
            end
        end
    end
    return NeighborSelection(pool, count, closest_idx)
end

@inline function LBVH_query!(pool::VI, stack::VI, LBVH::LinearBVH{D, T},
                             point::NTuple{D, T},
                             radius::S) where {D, T <: AbstractFloat, S <: AbstractFloat, VI <: AbstractVector{Int}}
    return LBVH_query!(pool, stack, LBVH, point, T(radius))
end

@inline function _nearest_leaf_index(stack::VI, LBVH::LinearBVH{D, T},
                                     point::NTuple{D, T}) where {D, T <: AbstractFloat, VI <: AbstractVector{Int}}
    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max

    L = LBVH.brt.left_child
    R = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    root = LBVH.root

    @inline function dist2_to_node(i::Int)
        s = zero(T)
        @inbounds for d in 1:D
            a = node_min[d][i]; b = node_max[d][i]; p = point[d]
            t = p < a ? (a - p) : (p > b ? (p - b) : zero(T))
            s += t * t
        end
        return s
    end

    @inline function leaf_dist2(i::Int)
        s = zero(T)
        @inbounds for d in 1:D
            a = leaf_min[d][i]; b = leaf_max[d][i]; p = point[d]
            t = p < a ? (a - p) : (p > b ? (p - b) : zero(T))
            s += t * t
        end
        return s
    end

    best_idx = zero(eltype(stack))
    best_dist2 = typemax(T)

    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            dist2 = leaf_dist2(leaf_idx)
            if dist2 < best_dist2
                best_dist2 = dist2
                best_idx = leaf_idx
            end
        end
        return best_idx
    end

    top = 0
    top = _push!(stack, top, root)
    while top > 0
        node, top = _pop!(stack, top)
        if dist2_to_node(node) <= best_dist2
            if LL[node]
                leaf_idx = L[node]
                dist2 = leaf_dist2(leaf_idx)
                if dist2 < best_dist2
                    best_dist2 = dist2
                    best_idx = leaf_idx
                end
            else
                top = _push!(stack, top, L[node])
            end

            if RR[node]
                leaf_idx = R[node]
                dist2 = leaf_dist2(leaf_idx)
                if dist2 < best_dist2
                    best_dist2 = dist2
                    best_idx = leaf_idx
                end
            else
                top = _push!(stack, top, R[node])
            end
        end
    end

    return best_idx
end

"""
    LBVH_query!(pool, stack, lbvh, point, radius_multiplier, h_array; atol = eps(T) * radius_multiplier)

Iteratively expand the spherical query until every neighbor selected by the LBVH
fulfils the symmetric SPH criterion `‖x_a - x_b‖ < max(κ h_a, κ h_b)`.

# Parameters
- `pool`, `stack`: Scratch buffers identical to the standard `LBVH_query!` entry point.
- `lbvh::LinearBVH{D,T}`: Hierarchy built from Morton-sorted particles.
- `point::NTuple{D,T}`: Query position.
- `radius_multiplier::T`: Kernel support factor `κ`.
- `h_array::AbstractVector{T}`: Per-particle smoothing lengths laid out in the
    same Morton order used to build `lbvh`.
- `atol::T`: Optional convergence tolerance for the expanding radius (defaults to
   a machine-epsilon scaled version of `radius_multiplier`).

# Returns
- `(NeighborSelection, T)`: The Morton-ordered neighbor selection and the
   smoothing length `h_a` of the nearest particle.
"""
@inline function LBVH_query!(pool::VI, stack::VI, LBVH::LinearBVH{D, T},
                             point::NTuple{D, T},
                             radius_multiplier::T, h_array::V;
                             atol::T = eps(T) * radius_multiplier) where {D, T <: AbstractFloat, V <: AbstractVector{T}, VI <: AbstractVector{Int}}
    nleaf = length(LBVH.enc.order)
    length(h_array) == nleaf || throw(ArgumentError("length(h_array) = $(length(h_array)) does not match LBVH leaf count $nleaf."))

    nearest_leaf = _nearest_leaf_index(stack, LBVH, point)
    zero_idx = zero(eltype(pool))
    nearest_leaf == 0 && return NeighborSelection(pool, zero_idx, zero_idx), zero(T)

    ha = h_array[nearest_leaf]
    tol = max(atol, zero(T))

    global_max_h = maximum(h_array)
    search_radius = radius_multiplier * max(ha, global_max_h)
    candidate = LBVH_query!(pool, stack, LBVH, point, search_radius)
    count = candidate.count
    count == 0 && return NeighborSelection(pool, zero_idx, zero_idx), ha

    coords = LBVH.enc.coord
    indices = candidate.pool
    kept = zero_idx
    closest_idx = zero_idx
    closest_dist2 = typemax(T)

    @inbounds for idx in 1:count
        leaf_idx = indices[idx]
        hb = h_array[leaf_idx]
        limit = radius_multiplier * max(ha, hb) + tol
        limit2 = limit * limit

        dist2 = zero(T)
        for d in 1:D
            δ = coords[d][leaf_idx] - point[d]
            dist2 += δ * δ
        end

        if dist2 <= limit2
            kept += one(eltype(pool))
            indices[kept] = leaf_idx
            if dist2 < closest_dist2
                closest_dist2 = dist2
                closest_idx = leaf_idx
            end
        end
    end

    kept == zero_idx && return NeighborSelection(pool, zero_idx, zero_idx), ha
    return NeighborSelection(indices, kept, closest_idx), ha
end

@inline function LBVH_query!(pool::VI, stack::VI, LBVH::LinearBVH{D, T},
                             point::NTuple{D, T},
                             radius_multiplier::S,
                             h_array::V;
                             atol::Union{Nothing, Real} = nothing) where {D, T <: AbstractFloat, S <: AbstractFloat, V <: AbstractVector{T}, VI <: AbstractVector{Int}}
    rm = T(radius_multiplier)
    tol = atol === nothing ? eps(T) * rm : T(atol)
    return LBVH_query!(pool, stack, LBVH, point, rm, h_array; atol = tol)
end

"""
        LBVH_query(lbvh::LinearBVH, point, radius)

Allocate scratch buffers and execute an LBVH neighbor search around `point`.
Convenience wrapper over [`LBVH_query!`](@ref) that returns the
`NeighborSelection` result while managing temporary storage internally.

# Parameters
- `lbvh::LinearBVH{D,T}`: Hierarchy previously built for the particle set.
- `point::NTuple{D,T}`: Query location.
- `radius::Real`: Search radius; internally promoted to `T`.

# Returns
- `NeighborSelection`: Neighbor indices aligned with the Morton ordering of
    `lbvh.enc`.
"""
@inline function LBVH_query(LBVH::LinearBVH{D, T}, point::NTuple{D, T}, radius::S) where {D, T <: AbstractFloat, S <: AbstractFloat}
    promoted_radius = T(radius)
    nleaf = length(LBVH.enc.coord[1])
    pool = zeros(Int, nleaf)
    stack_capacity = max(1, 2 * length(LBVH.brt.left_child) + 8)
    stack = Vector{Int}(undef, stack_capacity)
    result = LBVH_query!(pool, stack, LBVH, point, promoted_radius)
    return result
end

"""
    LBVH_query(lbvh::LinearBVH, point, radius_multiplier, h_array; atol = eps(T) * radius_multiplier)

Allocate buffers and execute a symmetric SPH neighbor query driven by the LBVH.
This mirrors [`LBVH_query!`](@ref) with smoothing-length aware radius expansion
but manages scratch storage automatically.

# Returns
- `(NeighborSelection, T)`: Morton-ordered neighbor selection and the nearest
   particle smoothing length.
"""
@inline function LBVH_query(LBVH::LinearBVH{D, T}, point::NTuple{D, T},
                            radius_multiplier::S,
                            h_array::V;
                            atol::Union{Nothing, Real} = nothing) where {D, T <: AbstractFloat, S <: AbstractFloat, V <: AbstractVector{T}}
    promoted_multiplier = T(radius_multiplier)
    tol = atol === nothing ? eps(T) * promoted_multiplier : T(atol)
    nleaf = length(LBVH.enc.coord[1])
    pool = zeros(Int, nleaf)
    stack_capacity = max(1, 2 * length(LBVH.brt.left_child) + 8)
    stack = Vector{Int}(undef, stack_capacity)
    result, ha = LBVH_query!(pool, stack, LBVH, point, promoted_multiplier, h_array; atol = tol)
    return result, ha
end