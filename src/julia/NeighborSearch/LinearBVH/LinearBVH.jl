"""
Linear Bounding Volume Hierarchy (LBVH) construction and traversal utilities  
    by Wei-Shan Su,  
    November 14, 2025
"""

################# Define structures #################
struct AABB{D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    min :: NTuple{D, VF}
    max :: NTuple{D, VF}
end

function Adapt.adapt_structure(to, x :: AB) where {D, AB <: AABB{D}}
    AABB(
        ntuple(i -> Adapt.adapt(to, x.min[i]), D),
        ntuple(i -> Adapt.adapt(to, x.max[i]), D)
    )
end

struct LinearBVH{D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}, A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    enc :: MortonEncoding{D, TF, TI, VF, VI}
    brt :: BinaryRadixTree{TI, VI, A, B}
    leaf_aabb :: AABB{D, TF, VF}
    node_aabb :: AABB{D, TF, VF}
    node_hmax :: VF
    root :: Int
end

function Adapt.adapt_structure(to, x :: LBVH) where {D, LBVH <: LinearBVH{D}}
    LinearBVH(
        Adapt.adapt(to, x.enc),
        Adapt.adapt(to, x.brt),
        Adapt.adapt(to, x.leaf_aabb),
        Adapt.adapt(to, x.node_aabb),
        Adapt.adapt(to, x.node_hmax),
        x.root
    )
end

################# Constructing LBVH #################
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
    node_hmax = similar(enc.h, ninternal)

    root = _find_root_index(brt)
    LBVH = LinearBVH{D, TF, TI, VF, VI, A, B}(enc, brt, leaf_aabb, node_aabb, node_hmax, root)
    _build_LBVH!(LBVH)
    return LBVH
end

# Toolbox
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

@inline function _dist2_to_node_aabb(node_min :: NTuple{D, VF} , node_max :: NTuple{D, VF}, point :: NTuple{D, TF}, idx :: Int) where {D,TF <: AbstractFloat, VF <: AbstractVector{TF}}
    T = typeof(point[1])
    zero_T = zero(T)
    s = zero_T
    @inbounds for d in eachindex(point)
        a = node_min[d][idx]
        b = node_max[d][idx]
        p = point[d]
        t = p < a ? (a - p) : (p > b ? (p - b) : zero_T)
        s += t * t
    end
    return s
end

@inline function _dist2_to_leaf_aabb(leaf_min :: NTuple{D, VF}, leaf_max :: NTuple{D, VF}, point :: NTuple{D, TF}, idx :: Int) where {D,TF <: AbstractFloat, VF <: AbstractVector{TF}}
    T = typeof(point[1])
    zero_T = zero(T)
    s = zero_T
    @inbounds for d in eachindex(point)
        a = leaf_min[d][idx]
        b = leaf_max[d][idx]
        p = point[d]
        t = p < a ? (a - p) : (p > b ? (p - b) : zero_T)
        s += t * t
    end
    return s
end

@inline function _process_leaf!(pool :: VI, count :: Int, leaf_idx :: Int, r2 :: T, point :: NTuple{D, T}, leaf_min :: NTuple{D, VF}, leaf_max :: NTuple{D, VF}, closest_idx :: Int, closest_dist2 :: T) where {D, T <: AbstractFloat, VI <: AbstractVector{Int}, VF <: AbstractVector{T}}
    dist2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, point, leaf_idx)
    if dist2 <= r2
        count += 1
        @inbounds pool[count] = leaf_idx
        if dist2 < closest_dist2
            closest_dist2 = dist2
            closest_idx = leaf_idx
        end
    end
    return count, closest_idx, closest_dist2
end

@inline function _next_internal_node(node::Int,
                                     left_child::A,
                                     right_child::A,
                                     is_leaf_left::B,
                                     is_leaf_right::B,
                                     node_parent) where {A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    while true
        @inbounds parent_raw = node_parent[node]
        parent = Int(parent_raw)
        if parent == 0
            return 0
        end
        if (!is_leaf_left[parent]) && left_child[parent] == node
            if is_leaf_right[parent]
                node = parent
                continue
            else
                return right_child[parent]
            end
        end
        node = parent
    end
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
    node_hmax = LBVH.node_hmax
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max
    leaf_h = LBVH.enc.h

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

    stack = zero(MVector{128, Int})
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

        @inbounds begin
            l = L[i]; r = R[i]
            hl = LL[i] ? leaf_h[l] : node_hmax[l]
            hr = RR[i] ? leaf_h[r] : node_hmax[r]
            node_hmax[i] = ifelse(hl > hr, hl, hr)
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

################# Traversal and Query #################
"""
    LBVH_probe_neighbors(LBVH, point, radius)

Probe all leaf AABBs in a Linear Bounding Volume Hierarchy (LinearBVH) to
find which leaves lie within a spherical query region of radius `radius`
centred at `point`. Returns the number of intersecting leaves, the index of
the closest leaf, and its squared distance.

# Parameters
- `LBVH::LinearBVH{D,T}`: Linear BVH structure containing node and leaf AABBs,
  child relationships, and root index.
- `point::NTuple{D,T}`: Query point in D-dimensional space.
- `radius::T`: Search radius.

# Returns
A 3-tuple `(count, closest_idx, closest_dist2)`:
- `count::Int`: Number of leaves whose bounding boxes intersect the sphere.
- `closest_idx::Int`: Index of the closest intersecting leaf (0 if none).
- `closest_dist2::T`: Minimum squared distance to an intersecting leaf
  (`typemax(T)` if none).
"""
@inline function LBVH_probe_neighbors(
    LBVH::LinearBVH{D, T},
    point::NTuple{D, T},
    radius::T
) where {D, T <: AbstractFloat}

    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    r2 = radius * radius
    count = 0
    closest_idx = 0
    closest_dist2 = typemax(T)

    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, point, leaf_idx)
            if d2 <= r2
                count += 1
                if d2 < closest_dist2
                    closest_dist2 = d2
                    closest_idx = leaf_idx
                end
            end
        end
        return count, closest_idx, closest_dist2
    end

    node = root
    while node != 0
        dist2_node = _dist2_to_node_aabb(node_min, node_max, point, node)
        if dist2_node <= r2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, point, leaf_idx)
                if d2 <= r2
                    count += 1
                    if d2 < closest_dist2
                        closest_dist2 = d2
                        closest_idx = leaf_idx
                    end
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, point, leaf_idx)
                if d2 <= r2
                    count += 1
                    if d2 < closest_dist2
                        closest_dist2 = d2
                        closest_idx = leaf_idx
                    end
                end
            end

            if !LL[node]
                node = L[node]
                continue
            end
            if !RR[node]
                node = R[node]
                continue
            end

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end

    return count, closest_idx, closest_dist2
end

"""
    LBVH_find_nearest(LBVH, point)

Find the nearest leaf AABB to a query point in a Linear Bounding Volume
Hierarchy (LinearBVH). Returns the index of the closest leaf and the
corresponding squared distance. Uses a stackless BVH traversal with
aggressive pruning based on current best distance.

# Parameters
- `LBVH::LinearBVH{D,T}`: Linear BVH structure containing node and leaf AABBs,
  child relations, parent pointers, and the root index.
- `point::NTuple{D,T}`: Query point in D-dimensional space.

# Keyword Arguments
(None)

# Returns
A 2-tuple `(best_idx, best_dist2)`:
- `best_idx::Int`: Index of the closest leaf (`0` if the BVH contains no nodes).
- `best_dist2::T`: Squared distance from `point` to the closest leaf AABB
  (`typemax(T)` if no leaves exist).
"""
@inline function LBVH_find_nearest(LBVH::LinearBVH{D,T}, point::NTuple{D,T}) where {D,T<:AbstractFloat}
    # AABB references
    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max

    # BVH tree topology
    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    parent = LBVH.brt.node_parent
    root = LBVH.root

    # If degenerate (no internal nodes)
    if root == 0
        nleaf = length(leaf_min[1])
        best_idx = 0
        best_dist2 = typemax(T)
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, point, leaf_idx)
            if d2 < best_dist2
                best_dist2 = d2
                best_idx = leaf_idx
            end
        end
        return best_idx, best_dist2
    end

    # Initial best distance set to +∞
    best_idx = 0
    best_dist2 = typemax(T)

    # Traverse with "manual stackless BVH" (your existing mechanism)
    node = root
    while node != 0
        # Bounding box distance test
        d2 = _dist2_to_node_aabb(node_min, node_max, point, node)

        if d2 > best_dist2
            # Too far → prune subtree
            node = _next_internal_node(node, L, R, LL, RR, parent)
            continue
        end

        @inbounds if LL[node]
            leaf_idx = L[node]
            d2leaf = _dist2_to_leaf_aabb(leaf_min, leaf_max, point, leaf_idx)
            if d2leaf < best_dist2
                best_dist2 = d2leaf
                best_idx = leaf_idx
            end
        elseif L[node] != 0
            # recurse into left node
            node = L[node]
            continue
        end

        @inbounds if RR[node]
            leaf_idx = R[node]
            d2leaf = _dist2_to_leaf_aabb(leaf_min, leaf_max, point, leaf_idx)
            if d2leaf < best_dist2
                best_dist2 = d2leaf
                best_idx = leaf_idx
            end
        elseif R[node] != 0
            node = R[node]
            continue
        end

        # Finished both children, climb back
        node = _next_internal_node(node, L, R, LL, RR, parent)
    end

    return best_idx, best_dist2
end


"""
    LBVH_query!(pool, lbvh, point, radius)

Depth–first traversal of a Linear BVH **without an explicit stack**.  
Traversal proceeds using per–node parent links (`node_parent`) to determine the
next internal node once a subtree has been fully processed. This eliminates
scratch–space requirements for a stack and is suitable for GPU kernels where
per–thread memory must remain minimal.

The algorithm performs:
- AABB rejection at each internal node.
- Direct leaf processing when a leaf child is encountered.
- Morton–order neighbor emission by visiting left before right children.
- Iterative DFS without recursion and without a software stack.

# Parameters
- `pool::AbstractVector{Int}`  
  Output buffer that receives leaf indices intersecting the spherical query.
  Written in Morton order; the valid prefix is returned through a
  `NeighborSelection` handle.

- `lbvh::LinearBVH{D,T}`  
  Linear Bounding Volume Hierarchy constructed from Morton–sorted primitives.

- `point::NTuple{D,T}`  
  Query position.

- `radius::Real`  
  Query radius; promotes to the BVH floating–point type.

# Returns
- `NeighborSelection`  
  A lightweight view into `pool` containing the number of neighbors and the
  closest leaf index with respect to squared distance.
"""

@inline function LBVH_query!(pool::VI, LBVH::LinearBVH{D, T},
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
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    r2 = radius * radius
    count = 0
    closest_idx = zero(eltype(pool))
    closest_dist2 = typemax(T)

    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            dist2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, point, leaf_idx)
            if dist2 <= r2
                count += 1
                @inbounds pool[count] = leaf_idx
                if dist2 < closest_dist2
                    closest_dist2 = dist2
                    closest_idx = leaf_idx
                end
            end
        end
        return NeighborSelection(pool, count, closest_idx)
    end

    node = root
    while node != 0
        dist2 = _dist2_to_node_aabb(node_min, node_max, point, node)
        if dist2 <= r2
            if LL[node]
                @inbounds leaf_idx = L[node]
                count, closest_idx, closest_dist2 = _process_leaf!(pool, count, leaf_idx, r2, point, leaf_min, leaf_max, closest_idx, closest_dist2)
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                count, closest_idx, closest_dist2 = _process_leaf!(pool, count, leaf_idx, r2, point, leaf_min, leaf_max, closest_idx, closest_dist2)
            end
            if !LL[node]
                @inbounds node = L[node]
                continue
            end
            if !RR[node]
                @inbounds node = R[node]
                continue
            end
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end

    return NeighborSelection(pool, count, closest_idx)
end

@inline function LBVH_query!(pool::VI, LBVH::LinearBVH{D, T},
                                       point::NTuple{D, T},
                                       radius::S) where {D, T <: AbstractFloat, S <: AbstractFloat, VI <: AbstractVector{Int}}
    return LBVH_query!(pool, LBVH, point, T(radius))
end
###################################################