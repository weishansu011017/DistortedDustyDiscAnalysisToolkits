"""
    @_lbvh_gather_traversal(LBVH, reference_point, radius2, leaf_hit)

Emit a stackless LBVH traversal (depth-first, preorder) using the BRT `left` + `escape`
tables, intended for **gather-style** queries.

The macro expands into a loop that visits unified node IDs starting from `brt.root`:

- **Leaf node** (`is_leaf_id(node, nleaf)`):
  - Convert unified node ID → leaf index via `leaf_index(node, nleaf)` (1..nleaf).
  - Compute squared distance `d2 = _dist2_to_aabb(leaf_min, leaf_max, reference_point, leaf_idx)`.
  - If `d2 <= radius2`, execute `leaf_hit` with `leaf_idx` available in a local `let` block.
  - Advance to `node = escape[node]`.

- **Internal node**:
  - Convert unified node ID → internal index via `internal_index(node)` (1..ninternal).
  - Compute `d2_node = _dist2_to_aabb(node_min, node_max, reference_point, node_idx)`.
  - If `d2_node > radius2`, prune subtree by `node = escape[node]`.
  - Otherwise descend by `node = left[node]`.

Degenerate tree (`brt.root == 0`) falls back to brute-force over all leaves `1:nleaf`.

# Arguments
- `LBVH`:
  A `LinearBVH` whose `brt` provides `root::Int32`, `left::Vector{Int32}`, `escape::Vector{Int32}`, `nleaf::Int`,
  and whose AABBs are stored in `LBVH.node_aabb` and `LBVH.leaf_aabb`.
- `reference_point`:
  `NTuple{D, T}` query position.
- `radius2`:
  Squared query radius (same float type as distance computations).
- `leaf_hit`:
  An expression executed on each accepted leaf; inside it, local variables `leaf_idx` (1..nleaf) and `p2leaf_d2` are available.
"""
macro _lbvh_gather_traversal(LBVH, reference_point, radius2, leafsym, d2sym, leaf_hit)
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_min_       = gensym(:leaf_min)
    leaf_max_       = gensym(:leaf_max)
    brt_            = gensym(:brt)
    left_           = gensym(:left)
    escape_         = gensym(:escape)
    root_           = gensym(:root)
    nleaf_          = gensym(:nleaf)
    node_           = gensym(:node)
    node_idx_       = gensym(:node_idx)
    leaf_idx_       = gensym(:leaf_idx)
    d2_             = gensym(:d2)
    d2_node_        = gensym(:d2_node)

    # hygiene: avoid capturing user locals
    LBVH_       = esc(LBVH)
    rp_         = esc(reference_point)
    r2_         = esc(radius2)
    leafsym_    = esc(leafsym)
    d2sym_      = esc(d2sym) 

    # the user-provided code must run in caller scope
    hit_  = esc(leaf_hit)

    quote
        # LBVH data
        ## AABB
        $node_min_    = $LBVH_.node_aabb.min        # length = ninternal
        $node_max_    = $LBVH_.node_aabb.max        # length = ninternal
        $leaf_min_    = $LBVH_.leaf_aabb.min        # length = nleaf
        $leaf_max_    = $LBVH_.leaf_aabb.max        # length = nleaf

        ## BRT
        $brt_         = $LBVH_.brt
        $left_        = $brt_.left                  # length = ntotal (children valid for internal nodes)
        $escape_      = $brt_.escape                # length = ntotal

        ## Other information
        $root_        = $brt_.root
        $nleaf_       = $brt_.nleaf     

        # No internal node: brute force leaves
        if iszero($root_)
            @inbounds for $leaf_idx_ in 1:$nleaf_
                $d2_ = _dist2_to_aabb($leaf_min_, $leaf_max_, $rp_, $leaf_idx_)
                if $d2_ <= $r2_
                    $leafsym_ = $leaf_idx_
                    $d2sym_   = $d2_
                    $hit_
                end
            end
        else
            $node_ = $root_
            while !iszero($node_)
                # Leaf: process then jump by escape
                if is_leaf_id($node_, $nleaf_)
                    $leaf_idx_ = leaf_index($node_, $nleaf_)
                    $d2_ = _dist2_to_aabb($leaf_min_, $leaf_max_, $rp_, $leaf_idx_)
                    if $d2_ <= $r2_
                        $leafsym_ = $leaf_idx_
                        $d2sym_   = $d2_
                        $hit_
                    end
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: AABB reject => prune subtree
                $node_idx_ = internal_index($node_)
                $d2_node_ = _dist2_to_aabb($node_min_, $node_max_, $rp_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[internal_index($node_)]
            end
        end
        nothing
    end
end

"""
    @_lbvh_scatter_traversal(LBVH, reference_point, Kvalid, leaf_hit)

Internal macro: stackless DFS traversal over a LinearBVH/BinaryRadixTree using
`left` + `escape` links (no explicit stack, no recursion).

This is the **scatter** variant: the pruning radius is **node-dependent**.
For each leaf, the acceptance radius is `r = Kvalid * leaf_h[leaf_idx]`.
For each internal node, subtree pruning uses `r = Kvalid * node_hmax[node_idx]`.

# Arguments
- `LBVH`: `LinearBVH` holding `leaf_aabb`, `node_aabb`, `leaf_h`, `node_hmax`, and `brt`.
- `reference_point`: query point used in AABB distance tests.
- `Kvalid`: scalar multiplier converting smoothing length to search radius.
- `leaf_hit`: 
   An expression executed on each accepted leaf; inside it, local variables `leaf_idx` (1..nleaf), `p2leaf_d2` and `hb` are available.
"""
macro _lbvh_scatter_traversal(LBVH, reference_point, Kvalid, leafsym, d2sym, hbsym, leaf_hit)
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_min_       = gensym(:leaf_min)
    leaf_max_       = gensym(:leaf_max)
    brt_            = gensym(:brt)
    left_           = gensym(:left)
    escape_         = gensym(:escape)
    root_           = gensym(:root)
    nleaf_          = gensym(:nleaf)
    node_           = gensym(:node)
    node_idx_       = gensym(:node_idx)
    leaf_idx_       = gensym(:leaf_idx)
    d2_             = gensym(:d2)
    d2_node_        = gensym(:d2_node)
    r_              = gensym(:r)
    r2_             = gensym(:r2)
    leaf_h_         = gensym(:leaf_h)
    node_hmax_      = gensym(:node_hmax)
    hb_             = gensym(:hb)

    # hygiene: avoid capturing user locals
    LBVH_       = esc(LBVH)
    Kvalid_     = esc(Kvalid)
    rp_         = esc(reference_point)
    leafsym_    = esc(leafsym)
    d2sym_      = esc(d2sym) 
    hbsym_      = esc(hbsym)

    # the user-provided code must run in caller scope
    hit_  = esc(leaf_hit)

    quote
        # LBVH data
        ## AABB
        $node_min_    = $LBVH_.node_aabb.min        # length = ninternal
        $node_max_    = $LBVH_.node_aabb.max        # length = ninternal
        $leaf_min_    = $LBVH_.leaf_aabb.min        # length = nleaf
        $leaf_max_    = $LBVH_.leaf_aabb.max        # length = nleaf

        ## BRT
        $brt_         = $LBVH_.brt
        $left_        = $brt_.left                  # length = ntotal (children valid for internal nodes)
        $escape_      = $brt_.escape                # length = ntotal

        ## Other information
        $root_        = $brt_.root
        $nleaf_       = $brt_.nleaf     
        $leaf_h_      = $LBVH_.leaf_h
        $node_hmax_   = $LBVH_.node_hmax

        # No internal node: brute force leaves
        if iszero($root_)
            @inbounds for $leaf_idx_ in 1:$nleaf_
                $hb_    = $leaf_h_[$leaf_idx_]
                $r_     = $Kvalid_ * $hb_
                $r2_    = $r_ * $r_
                $d2_ = _dist2_to_aabb($leaf_min_, $leaf_max_, $rp_, $leaf_idx_)
                if $d2_ <= $r2_
                    $leafsym_ = $leaf_idx_
                    $d2sym_   = $d2_
                    $hbsym_   = $hb_
                    $hit_
                end
            end
        else
            $node_ = $root_
            while !iszero($node_)
                # Leaf: process then jump by escape
                if is_leaf_id($node_, $nleaf_)
                    $leaf_idx_ = leaf_index($node_, $nleaf_)
                    $hb_    = $leaf_h_[$leaf_idx_]
                    $r_     = $Kvalid_ * $hb_
                    $r2_    = $r_ * $r_
                    $d2_ = _dist2_to_aabb($leaf_min_, $leaf_max_, $rp_, $leaf_idx_)
                    if $d2_ <= $r2_
                        $leafsym_ = $leaf_idx_
                        $d2sym_   = $d2_
                        $hbsym_   = $hb_
                        $hit_
                    end
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: AABB reject => prune subtree
                $node_idx_ = internal_index($node_)
                $hb_    = $node_hmax_[$node_idx_]
                $r_     = $Kvalid_ * $hb_
                $r2_    = $r_ * $r_
                $d2_node_ = _dist2_to_aabb($node_min_, $node_max_, $rp_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[internal_index($node_)]
            end
        end
        nothing
    end
end

"""
    @_lbvh_symmetric_traversal(LBVH, reference_point, Kvalid, radius2, leaf_hit)

Internal macro: stackless DFS traversal over a LinearBVH/BinaryRadixTree using
`left` + `escape` links (no explicit stack, no recursion).

This is the **symmetric** variant: each candidate uses a per-node acceptance radius

    r2 = max(radius2, (Kvalid * h)^2)

where `h = leaf_h[leaf_idx]` for leaves and `h = node_hmax[node_idx]` for internal nodes.
This implements a symmetric gate between a fixed query radius (`radius2`) and an
SPH smoothing-radius scale (`Kvalid*h`).

# Arguments
- `LBVH`: `LinearBVH` holding `leaf_aabb`, `node_aabb`, `leaf_h`, `node_hmax`, and `brt`.
- `reference_point`: query point used in AABB distance tests.
- `Kvalid`: scalar multiplier converting smoothing length to search radius.
- `radius2`: base squared radius for the query.
- `leaf_hit`: 
   An expression executed on each accepted leaf; inside it, local variables `leaf_idx` (1..nleaf), `p2leaf_d2` and `hb` are available.
"""
macro _lbvh_symmetric_traversal(LBVH, reference_point, Kvalid, radius2, leafsym, d2sym, hbsym, leaf_hit)
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_min_       = gensym(:leaf_min)
    leaf_max_       = gensym(:leaf_max)
    brt_            = gensym(:brt)
    left_           = gensym(:left)
    escape_         = gensym(:escape)
    root_           = gensym(:root)
    nleaf_          = gensym(:nleaf)
    node_           = gensym(:node)
    node_idx_       = gensym(:node_idx)
    leaf_idx_       = gensym(:leaf_idx)
    d2_             = gensym(:d2)
    d2_node_        = gensym(:d2_node)
    r_              = gensym(:r)
    r2_             = gensym(:r2)
    leaf_h_         = gensym(:leaf_h)
    node_hmax_      = gensym(:node_hmax)
    hb_             = gensym(:hb)
    

    # hygiene: avoid capturing user locals
    LBVH_       = esc(LBVH)
    Kvalid_     = esc(Kvalid)
    radius2_    = esc(radius2)
    rp_         = esc(reference_point)
    leafsym_    = esc(leafsym)
    d2sym_      = esc(d2sym) 
    hbsym_      = esc(hbsym)

    # the user-provided code must run in caller scope
    hit_  = esc(leaf_hit)

    quote
        # LBVH data
        ## AABB
        $node_min_    = $LBVH_.node_aabb.min        # length = ninternal
        $node_max_    = $LBVH_.node_aabb.max        # length = ninternal
        $leaf_min_    = $LBVH_.leaf_aabb.min        # length = nleaf
        $leaf_max_    = $LBVH_.leaf_aabb.max        # length = nleaf

        ## BRT
        $brt_         = $LBVH_.brt
        $left_        = $brt_.left                  # length = ntotal
        $escape_      = $brt_.escape                # length = ntotal

        ## Other information
        $root_        = $brt_.root
        $nleaf_       = $brt_.nleaf     
        $leaf_h_      = $LBVH_.leaf_h
        $node_hmax_   = $LBVH_.node_hmax

        # No internal node: brute force leaves
        if iszero($root_)
            @inbounds for $leaf_idx_ in 1:$nleaf_
                $hb_    = $leaf_h_[$leaf_idx_]
                $r_     = $Kvalid_ * $hb_
                $r2_    = max($radius2_, $r_ * $r_)
                $d2_ = _dist2_to_aabb($leaf_min_, $leaf_max_, $rp_, $leaf_idx_)
                if $d2_ <= $r2_
                    $leafsym_ = $leaf_idx_
                    $d2sym_   = $d2_
                    $hbsym_   = $hb_
                    $hit_
                end
            end
        else
            $node_ = $root_
            while !iszero($node_)
                # Leaf: process then jump by escape
                if is_leaf_id($node_, $nleaf_)
                    $leaf_idx_ = leaf_index($node_, $nleaf_)
                    $hb_    = $leaf_h_[$leaf_idx_]
                    $r_     = $Kvalid_ * $hb_
                    $r2_    = max($radius2_, $r_ * $r_)
                    $d2_ = _dist2_to_aabb($leaf_min_, $leaf_max_, $rp_, $leaf_idx_)
                    if $d2_ <= $r2_
                        $leafsym_ = $leaf_idx_
                        $d2sym_   = $d2_
                        $hbsym_   = $hb_
                        $hit_
                    end
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: AABB reject => prune subtree
                $node_idx_ = internal_index($node_)
                $hb_    = $node_hmax_[$node_idx_]
                $r_     = $Kvalid_ * $hb_
                $r2_    = max($radius2_, $r_ * $r_)
                $d2_node_ = _dist2_to_aabb($node_min_, $node_max_, $rp_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[Int($node_)]
            end
        end
        nothing
    end
end