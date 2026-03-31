"""
    @LBVH_gather_point_traversal(LBVH, reference_point, radius2, leafsym, d2sym, leaf_hit)

Stackless DFS traversal over a LinearBVH/BinaryRadixTree using
`left` + `escape` links (no explicit stack, no recursion).

This is the **gather** variant: the pruning radius is given by the `radius2` input.

# Arguments
- `LBVH`:
  A `LinearBVH` whose `brt` provides `root::Int32`, `left::Vector{Int32}`, `escape::Vector{Int32}`, `nleaf::Int`,
  and whose internal-node AABBs are stored in `LBVH.node_aabb` while leaf
  primitives are stored in `LBVH.leaf_coor`.
- `reference_point`:
  `NTuple{D, T}` query position.
- `radius2`:
  Squared query radius (same float type as distance computations).
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared point-to-leaf distance.
- `leaf_hit`:
  An expression executed on each accepted leaf after assigning `leafsym` and
  `d2sym` in caller scope.
"""
macro LBVH_gather_point_traversal(LBVH, reference_point, radius2, leafsym, d2sym, leaf_hit)
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_coor_      = gensym(:leaf_coor)
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
        $leaf_coor_   = $LBVH_.leaf_coor            # length = nleaf

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
                $d2_ = _squared_distance_point_coords($rp_, $leaf_coor_, $leaf_idx_)
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
                    $d2_ = _squared_distance_point_coords($rp_, $leaf_coor_, $leaf_idx_)
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
                $d2_node_ = _squared_distance_point_aabb($rp_, $node_min_, $node_max_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[$node_idx_]
            end
        end
        nothing
    end
end

"""
    @LBVH_scatter_point_traversal(LBVH, reference_point, Kvalid, leafsym, d2sym, hbsym, leaf_hit)

Stackless DFS traversal over a LinearBVH/BinaryRadixTree using
`left` + `escape` links (no explicit stack, no recursion).

This is the **scatter** variant: the pruning radius is **node-dependent**.
For each leaf, the acceptance radius is `r = Kvalid * leaf_h[leaf_idx]`.
For each internal node, subtree pruning uses `r = Kvalid * node_hmax[node_idx]`.

# Arguments
- `LBVH`: `LinearBVH` holding `leaf_coor`, `node_aabb`, `leaf_h`, `node_hmax`, and `brt`.
- `reference_point`: query point used in AABB distance tests.
- `Kvalid`: scalar multiplier converting smoothing length to search radius.
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared point-to-leaf distance.
- `hbsym`:
  Caller-scope symbol that receives the smoothing length associated with the
  accepted leaf.
- `leaf_hit`: 
   An expression executed on each accepted leaf after assigning `leafsym`,
   `d2sym`, and `hbsym` in caller scope.
"""
macro LBVH_scatter_point_traversal(LBVH, reference_point, Kvalid, leafsym, d2sym, hbsym, leaf_hit)
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_coor_      = gensym(:leaf_coor)
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
        $leaf_coor_   = $LBVH_.leaf_coor            # length = nleaf

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
                $d2_ = _squared_distance_point_coords($rp_, $leaf_coor_, $leaf_idx_)
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
                    $d2_ = _squared_distance_point_coords($rp_, $leaf_coor_, $leaf_idx_)
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
                $d2_node_ = _squared_distance_point_aabb($rp_, $node_min_, $node_max_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[$node_idx_]
            end
        end
        nothing
    end
end

"""
    @LBVH_symmetric_point_traversal(LBVH, reference_point, Kvalid, radius2, leafsym, d2sym, hbsym, leaf_hit)

Stackless DFS traversal over a LinearBVH/BinaryRadixTree using
`left` + `escape` links (no explicit stack, no recursion).

This is the **symmetric** variant: each candidate uses a per-node acceptance radius

    r2 = max(radius2, (Kvalid * h)^2)

where `h = leaf_h[leaf_idx]` for leaves and `h = node_hmax[node_idx]` for internal nodes.
This implements a symmetric gate between a fixed query radius (`radius2`) and an
SPH smoothing-radius scale (`Kvalid*h`).

# Arguments
- `LBVH`: `LinearBVH` holding `leaf_coor`, `node_aabb`, `leaf_h`, `node_hmax`, and `brt`.
- `reference_point`: query point used in AABB distance tests.
- `Kvalid`: scalar multiplier converting smoothing length to search radius.
- `radius2`: base squared radius for the query.
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared point-to-leaf distance.
- `hbsym`:
  Caller-scope symbol that receives the smoothing length associated with the
  accepted leaf.
- `leaf_hit`: 
   An expression executed on each accepted leaf after assigning `leafsym`,
   `d2sym`, and `hbsym` in caller scope.
"""
macro LBVH_symmetric_point_traversal(LBVH, reference_point, Kvalid, radius2, leafsym, d2sym, hbsym, leaf_hit)
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_coor_      = gensym(:leaf_coor)
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
        $leaf_coor_   = $LBVH_.leaf_coor            # length = nleaf

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
                $d2_    = _squared_distance_point_coords($rp_, $leaf_coor_, $leaf_idx_)
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
                    $d2_ = _squared_distance_point_coords($rp_, $leaf_coor_, $leaf_idx_)
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
                $d2_node_ = _squared_distance_point_aabb($rp_, $node_min_, $node_max_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[$node_idx_]
            end
        end
        nothing
    end
end

"""
    @LBVH_gather_line_traversal(LBVH, line_origin, line_direction, radius2, leafsym, d2sym, leaf_hit)

Stackless DFS traversal over a `LinearBVH`/`BinaryRadixTree` using
`left` + `escape` links, with no explicit stack and no recursion.

This is the **gather** line-traversal variant: the pruning radius is given
directly by the input `radius2`. Internal nodes are pruned using a
conservative lower bound on the squared distance between the query line and
the node AABB, while leaf primitives are tested using the exact squared
distance between the query line and the leaf particle coordinate.

# Parameters
- `LBVH`:
  A `LinearBVH` whose `brt` provides `root::Int32`, `left::Vector{Int32}`,
  `escape::Vector{Int32}`, and `nleaf::Int`, whose internal-node AABBs are
  stored in `LBVH.node_aabb`, and whose leaf primitives are stored in
  `LBVH.leaf_coor`.
- `line_origin`:
  `NTuple{D,T}` giving the origin of the query line.
- `line_direction`:
  `NTuple{D,T}` giving the direction of the query line. This direction is
  assumed to be a unit vector.
- `radius2`:
  Squared query radius used for both internal-node pruning and leaf
  acceptance.
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared distance from the query line
  to the accepted leaf primitive.
- `leaf_hit`:
  An expression executed on each accepted leaf after assigning `leafsym` and
  `d2sym` in caller scope.
"""
macro LBVH_gather_line_traversal(LBVH, line_origin, line_direction, radius2, leafsym, d2sym, leaf_hit)
    # Leaf primitives are stored as particle coordinates, so the leaf-level
    # query uses the exact point-line squared distance. Internal nodes remain
    # AABBs and use a conservative lower bound for pruning.
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_coor_      = gensym(:leaf_coor)
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
    origin_     = esc(line_origin)
    direction_  = esc(line_direction)
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
        $leaf_coor_   = $LBVH_.leaf_coor            # length = nleaf

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
                $d2_ = _squared_distance_line_coords($origin_, $direction_, $leaf_coor_, $leaf_idx_)
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
                    $d2_ = _squared_distance_line_coords($origin_, $direction_, $leaf_coor_, $leaf_idx_)
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
                $d2_node_ = _squared_distance_line_aabb_lower_bound($origin_, $direction_, $node_min_, $node_max_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[$node_idx_]
            end
        end
        nothing
    end
end

"""
    @LBVH_scatter_line_traversal(LBVH, line_origin, line_direction, Kvalid, leafsym, d2sym, hbsym, leaf_hit)

Stackless DFS traversal over a `LinearBVH`/`BinaryRadixTree` using
`left` + `escape` links, with no explicit stack and no recursion.

This is the **scatter** line-traversal variant: the acceptance radius is
node-dependent. For each leaf, the acceptance radius is

    r = Kvalid * leaf_h[leaf_idx]

and for each internal node, subtree pruning uses

    r = Kvalid * node_hmax[node_idx]

Leaf primitives are tested using the exact squared distance between the query
line and the leaf particle coordinate, while internal nodes are pruned using
a conservative lower bound on the squared distance between the query line and
the node AABB.

# Parameters
- `LBVH`:
  A `LinearBVH` holding `leaf_coor`, `node_aabb`, `leaf_h`, `node_hmax`, and
  `brt`.
- `line_origin`:
  `NTuple{D,T}` giving the origin of the query line.
- `line_direction`:
  `NTuple{D,T}` giving the direction of the query line. This direction is
  assumed to be a unit vector.
- `Kvalid`:
  Scalar multiplier converting smoothing length to search radius.
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared distance from the query line
  to the accepted leaf primitive.
- `hbsym`:
  Caller-scope symbol that receives the smoothing length associated with the
  accepted leaf.
- `leaf_hit`:
  An expression executed on each accepted leaf after assigning `leafsym`,
  `d2sym`, and `hbsym` in caller scope.
"""
macro LBVH_scatter_line_traversal(LBVH, line_origin, line_direction, Kvalid, leafsym, d2sym, hbsym, leaf_hit)
    # Leaf primitives are stored as particle coordinates, so the leaf-level
    # query uses the exact point-line squared distance. Internal nodes remain
    # AABBs and use a conservative lower bound for pruning.
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_coor_      = gensym(:leaf_coor)
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
    origin_     = esc(line_origin)
    direction_  = esc(line_direction)
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
        $leaf_coor_   = $LBVH_.leaf_coor            # length = nleaf

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
                $d2_ = _squared_distance_line_coords($origin_, $direction_, $leaf_coor_, $leaf_idx_)
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
                    $d2_ = _squared_distance_line_coords($origin_, $direction_, $leaf_coor_, $leaf_idx_)
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
                $d2_node_ = _squared_distance_line_aabb_lower_bound($origin_, $direction_, $node_min_, $node_max_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[$node_idx_]
            end
        end
        nothing
    end
end

"""
    @LBVH_symmetric_line_traversal(LBVH, line_origin, line_direction, Kvalid, radius2, leafsym, d2sym, hbsym, leaf_hit)

Stackless DFS traversal over a `LinearBVH`/`BinaryRadixTree` using
`left` + `escape` links, with no explicit stack and no recursion.

This is the **symmetric** line-traversal variant: each candidate uses a
per-node acceptance radius

    r2 = max(radius2, (Kvalid * h)^2)

where `h = leaf_h[leaf_idx]` for leaves and `h = node_hmax[node_idx]` for
internal nodes. This implements a symmetric gate between a fixed query radius
(`radius2`) and an SPH smoothing-radius scale (`Kvalid*h`).

Leaf primitives are tested using the exact squared distance between the query
line and the leaf particle coordinate, while internal nodes are pruned using
a conservative lower bound on the squared distance between the query line and
the node AABB.

# Parameters
- `LBVH`:
  A `LinearBVH` holding `leaf_coor`, `node_aabb`, `leaf_h`, `node_hmax`, and
  `brt`.
- `line_origin`:
  `NTuple{D,T}` giving the origin of the query line.
- `line_direction`:
  `NTuple{D,T}` giving the direction of the query line. This direction is
  assumed to be a unit vector.
- `Kvalid`:
  Scalar multiplier converting smoothing length to search radius.
- `radius2`:
  Base squared radius for the query.
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared distance from the query line
  to the accepted leaf primitive.
- `hbsym`:
  Caller-scope symbol that receives the smoothing length associated with the
  accepted leaf.
- `leaf_hit`:
  An expression executed on each accepted leaf after assigning `leafsym`,
  `d2sym`, and `hbsym` in caller scope.
"""
macro LBVH_symmetric_line_traversal(LBVH, line_origin, line_direction, Kvalid, radius2, leafsym, d2sym, hbsym, leaf_hit)
    # Leaf primitives are stored as particle coordinates, so the leaf-level
    # query uses the exact point-line squared distance. Internal nodes remain
    # AABBs and use a conservative lower bound for pruning.
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    leaf_coor_      = gensym(:leaf_coor)
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
    origin_     = esc(line_origin)
    direction_  = esc(line_direction)
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
        $leaf_coor_   = $LBVH_.leaf_coor            # length = nleaf

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
                $r2_    = max($radius2_, $r_ * $r_)
                $d2_ = _squared_distance_line_coords($origin_, $direction_, $leaf_coor_, $leaf_idx_)
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
                    $d2_ = _squared_distance_line_coords($origin_, $direction_, $leaf_coor_, $leaf_idx_)
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
                $d2_node_ = _squared_distance_line_aabb_lower_bound($origin_, $direction_, $node_min_, $node_max_, $node_idx_)
                if $d2_node_ > $r2_
                    @inbounds $node_ = $escape_[Int($node_)]
                    continue
                end

                # Internal: descend to left child (DFS preorder)
                @inbounds $node_ = $left_[$node_idx_]
            end
        end
        nothing
    end
end
