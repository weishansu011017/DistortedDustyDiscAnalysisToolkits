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
@inline function LBVH_probe_neighbors(LBVH::LinearBVH{D, T}, point::NTuple{D, T}, radius::T) where {D, T <: AbstractFloat}
    # Initialize
    r2 = radius * radius
    count = 0
    closest_idx = 0
    closest_dist2 = typemax(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    @LBVH_gather_point_traversal LBVH point r2 leaf_idx p2leaf_d2 begin
        count += 1
        if p2leaf_d2 < closest_dist2
            closest_dist2 = p2leaf_d2
            closest_idx = leaf_idx
        end
    end

    return count, closest_idx, closest_dist2
end

"""
    LBVH_find_nearest(LBVH, point)

Find the nearest **leaf AABB** to a query point in a `LinearBVH`.

The search is performed with a **stackless depth-first traversal** driven by the
binary radix tree (BRT) tables `left` and `escape` (no explicit stack and no
parent-walking during traversal). The current best squared distance is used as
a tightening bound:

- For an internal node, if the point-to-AABB squared distance exceeds the current
  `best_dist2`, the whole subtree is pruned by jumping to `escape[node]`.
- For a leaf node, the leaf is considered only when its point-to-AABB squared
  distance is within the current bound, and `best_idx/best_dist2` are updated if
  it improves the best.

# Parameters
- `LBVH::LinearBVH{D,T}`:
  Linear BVH containing:
  - `LBVH.leaf_aabb` (leaf AABBs)
  - `LBVH.node_aabb` (internal AABBs)
  - `LBVH.brt.left` and `LBVH.brt.escape` (stackless traversal tables)
  - `LBVH.brt.root` and `LBVH.brt.nleaf`

- `point::NTuple{D,T}`:
  Query point in D-dimensional space. Values are assumed finite.

# Returns
A 2-tuple `(best_idx, best_dist2)`:
- `best_idx::Int`:
  Leaf index (1-based) of the closest leaf AABB. Returns `0` only if the BVH
  contains no leaves (should not happen if `nleaf ≥ 1`).
- `best_dist2::T`:
  Squared distance from `point` to the closest leaf AABB.
"""
@inline function LBVH_find_nearest(LBVH::LinearBVH{D,T}, point::NTuple{D,T}) where {D,T<:AbstractFloat}
    # Initial best distance set to +∞
    best_idx = 0
    best_dist2 = typemax(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    @LBVH_gather_point_traversal LBVH point r2 leaf_idx p2leaf_d2 begin
        if p2leaf_d2 < best_dist2
            best_dist2 = p2leaf_d2
            best_idx = leaf_idx
        end
    end
    return best_idx, best_dist2
end

"""
    LBVH_find_nearest_h(LBVH::LinearBVH{D,T}, point::NTuple{D,T}) where {D,T<:AbstractFloat}

Return the smoothing length `h` of the nearest particle (leaf) to `point` in a
`LinearBVH`.

This routine traverses the BVH using point-to-AABB squared distances for pruning.
With the current LBVH construction where each leaf AABB is degenerate
(`leaf_min == leaf_max == particle_position`), the leaf distance reduces to the
squared Euclidean distance between `point` and the particle position. The
returned value is `LBVH.leaf_h[best_idx]`, where `best_idx` is the closest leaf.

# Parameters
- `LBVH::LinearBVH{D,T}`  
  Bounding volume hierarchy built from Morton-sorted particle coordinates and
  associated per-particle smoothing lengths `LBVH.leaf_h`.
- `point::NTuple{D,T}`  
  Query point in the same coordinate space as the particles.

# Keyword Arguments
- None.

# Returns
- `h::T`  
  Smoothing length of the nearest particle (leaf) to `point`.

"""
@inline function LBVH_find_nearest_h(LBVH::LinearBVH{D,T}, point::NTuple{D,T}) where {D,T<:AbstractFloat}
    # Initial best distance set to +∞
    best_idx = 0
    best_dist2 = typemax(T)

    # Smoothed radius
    smoothed_radius = LBVH.leaf_h
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    @LBVH_gather_point_traversal LBVH point best_dist2 leaf_idx p2leaf_d2 begin
        if p2leaf_d2 < best_dist2
            best_dist2 = p2leaf_d2
            best_idx = leaf_idx
        end
    end

    best_idx == 0 && return T(NaN)   # 或 return (0, typemax(T))
    best_h = smoothed_radius[best_idx]
    return best_h
end

"""
    LBVH_query!(pool, LBVH, point, radius)

Collect leaf indices whose leaf AABBs intersect a spherical query region centered
at `point` with radius `radius`, using a **stackless depth-first traversal** of a
`LinearBVH`.

Traversal is driven by the BRT tables `left` and `escape` (no explicit stack and
no parent-walking). Nodes are visited in DFS preorder; accepted leaves are
written into `pool` in that visit order. For each visited leaf, the point-to-leaf
AABB squared distance is evaluated and compared against `radius^2`.

# Parameters
- `pool::AbstractVector{Int}`:
  Output buffer that receives accepted leaf indices (1-based). The function writes
  into `pool[1:count]`; the remaining entries are untouched.

- `LBVH::LinearBVH{D,T}`:
  Linear BVH containing:
  - `LBVH.leaf_aabb` and `LBVH.node_aabb`
  - `LBVH.brt.root`, `LBVH.brt.nleaf`, `LBVH.brt.left`, `LBVH.brt.escape`

- `point::NTuple{D,T}`:
  Query point in D-dimensional space.

- `radius`:
  Query radius. In the `radius::T` method, `r2 = radius*radius` is used directly.
  In the `radius::S` method, `radius` is promoted to `T`.

# Returns
- `NeighborSelection`:
  Handle/view describing the valid prefix of `pool` and the closest leaf among the
  accepted set (by squared distance). The neighbor count is `count`, and the
  closest leaf index is `closest_idx` (0 if no leaf is accepted).

# Notes
- The traversal order is determined by `left`/`escape` (DFS preorder in unified node
  ID space). This is not the same mechanism as the older parent-pointer traversal.
- Correctness assumes the BRT `left`/`escape` tables and leaf/internal ID mapping
  (`is_leaf_id`, `leaf_index`, `internal_index`) are consistent with the LBVH build.
"""
@inline function LBVH_query!(pool::VI, LBVH::LinearBVH{D, T},
                                       point::NTuple{D, T},
                                       radius::T) where {D, T <: AbstractFloat, VI <: AbstractVector{Int}}
    # Initializing
    r2 = radius * radius
    count = 0
    closest_idx = zero(eltype(pool))
    closest_dist2 = typemax(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    @LBVH_gather_point_traversal LBVH point r2 leaf_idx p2leaf_d2 begin
        count += 1
        @inbounds pool[count] = leaf_idx
        if p2leaf_d2 < closest_dist2
          closest_dist2 = p2leaf_d2
          closest_idx = leaf_idx
        end
    end
    return NeighborSelection(pool, count, closest_idx)
end

@inline function LBVH_query!(pool::VI, LBVH::LinearBVH{D, T},
                                       point::NTuple{D, T},
                                       radius::S) where {D, T <: AbstractFloat, S <: AbstractFloat, VI <: AbstractVector{Int}}
    return LBVH_query!(pool, LBVH, point, T(radius))
end