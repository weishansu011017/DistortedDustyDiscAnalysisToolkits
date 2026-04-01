# ──────────────────────────────────────────────────────────────────────────── #
#  Test: NeighborSearch — Morton Encoding, BRT, and Linear BVH
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ─────────────────────
#  End-to-end validation of the spatial neighbour-search pipeline:
#
#  1. BinaryRadixTree — Structural invariants
#     • Root identity, node counts (2N − 1 total).
#     • Parent–child consistency (parent[root] = 0, valid IDs).
#     • Karras split rules: left/right child IDs match computed split
#       positions in the Morton-sorted code array.
#     • Escape links within bounds.
#     • Identical-code degenerate case.
#
#  2. LinearBVH — Construction, hmax, and queries
#     • AABB containment: each internal node's box encloses both children.
#     • Node hmax: per-node maximum smoothing length matches a recursive
#       reference computation.
#     • Neighbour queries: `LBVH_query!` returns exactly the same set as
#       an O(N²) brute-force scan.
#
#  3. Traversal pruning
#     • Scatter pruning: stackless traversal with per-particle radii
#       matches brute force.
#     • hmax pruning: node-level hmax never visits more nodes than a
#       global-hmax baseline.
#
#  All invariants are tested for 2D and 3D, across tree sizes N = 1…20,
#  using deterministic pseudo-random coordinates seeded by prime-offset
#  hashing.
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using Random
using PhantomRevealer
using PhantomRevealer.NeighborSearch

# ========================== Module aliases ================================== #

ns_mod = PhantomRevealer.NeighborSearch

# ========================== Helper functions ================================ #

# ── Coordinate generators ────────────────────────────────────────────── #

primes = (37, 61, 97, 131, 197, 263)

"""Build `dim` coordinate vectors of length `n` using prime-hashing."""
function build_coords(dim::Int, n::Int, offset::Int)
    coords = Vector{Vector{Float64}}(undef, dim)
    modulus = 1021
    for d in 1:dim
        prime = primes[d]
        shift = primes[d + dim]
        coords[d] = [Float64(mod(prime * i + shift * offset, modulus)) / modulus for i in 1:n]
    end
    return coords
end

function build_encoding(::Val{2}, n::Int, offset::Int)
    coords = build_coords(2, n, offset)
    h = fill(0.1, n)
    return MortonEncoding(coords[1], coords[2], h)
end

function build_encoding(::Val{3}, n::Int, offset::Int)
    coords = build_coords(3, n, offset)
    h = fill(0.1, n)
    return MortonEncoding(coords[1], coords[2], coords[3], h)
end

identical_encoding(::Val{2}, n::Int) =
    MortonEncoding(fill(0.5, n), fill(0.5, n), fill(0.5, n))

identical_encoding(::Val{3}, n::Int) =
    MortonEncoding(fill(0.5, n), fill(0.5, n), fill(0.5, n), fill(0.5, n))

# ── Brute-force neighbour search ─────────────────────────────────────── #

"""O(N²) brute-force: find all particle indices within `radius` of `point`."""
function brute_force_neighbors(enc, point::NTuple{D,T}, radius) where {D,T}
    coords = enc.coord
    r2 = radius * radius
    tol = eps(eltype(coords[1])) * 16
    n = length(coords[1])
    hits = Int[]
    for i in 1:n
        dist2 = zero(eltype(coords[1]))
        for d in 1:D
            δ = coords[d][i] - point[d]
            dist2 += δ * δ
        end
        if dist2 <= r2 + tol
            push!(hits, i)
        end
    end
    sort!(hits)
    return hits
end

# ── Recursive hmax reference ─────────────────────────────────────────── #

"""Compute expected `node_hmax` by recursive subtree traversal."""
function subtree_hmax_reference(enc, brt)
    nleaf = brt.nleaf
    nint = nleaf - 1
    out = zeros(eltype(enc.h), nint)
    root = brt.root
    root == 0 && return out

    function visit(node::Int32)
        if ns_mod.is_leaf_id(node, nleaf)
            return enc.h[ns_mod.leaf_index(node, nleaf)]
        end
        idx = ns_mod.internal_index(node)
        hl = visit(brt.left[idx])
        hr = visit(brt.right[idx])
        out[idx] = max(hl, hr)
        return out[idx]
    end
    visit(root)
    return out
end

# ── Scatter traversal reference ──────────────────────────────────────── #

"""Stackless traversal with per-particle radii (for scatter-pruning test)."""
function scatter_neighbors_reference(lbvh, point::NTuple{D,T}, Kvalid::T, hvec) where {D,T}
    node_min = lbvh.node_aabb.min
    node_max = lbvh.node_aabb.max
    leaf_coor = lbvh.leaf_coor
    brt = lbvh.brt
    left = brt.left
    escape = brt.escape
    nleaf = brt.nleaf
    hits = Int[]
    node = brt.root

    if node == 0
        @inbounds for leaf in 1:nleaf
            r2 = (Kvalid * hvec[leaf])^2
            d2 = ns_mod._squared_distance_point_coords(point, leaf_coor, leaf)
            d2 <= r2 && push!(hits, leaf)
        end
        sort!(hits)
        return hits
    end
    while node != 0
        if ns_mod.is_leaf_id(node, nleaf)
            leaf = ns_mod.leaf_index(node, nleaf)
            r2 = (Kvalid * hvec[leaf])^2
            d2 = ns_mod._squared_distance_point_coords(point, leaf_coor, leaf)
            d2 <= r2 && push!(hits, leaf)
            node = escape[Int(node)]
            continue
        end
        idx = ns_mod.internal_index(node)
        r2node = (Kvalid * lbvh.node_hmax[idx])^2
        d2node = ns_mod._squared_distance_point_aabb(point, node_min, node_max, idx)
        node = (d2node <= r2node) ? left[idx] : escape[idx]
    end
    sort!(hits)
    return hits
end

"""O(N) brute-force line query against leaf particle coordinates."""
function line_neighbors_reference(lbvh, origin::NTuple{D,T}, direction::NTuple{D,T}, radius2_of) where {D,T}
    nleaf = lbvh.brt.nleaf
    hits = Int[]
    @inbounds for leaf in 1:nleaf
        point = ntuple(d -> lbvh.leaf_coor[d][leaf], D)
        d2 = ns_mod._squared_distance_point_line(point, origin, direction)
        d2 <= radius2_of(leaf) && push!(hits, leaf)
    end
    sort!(hits)
    return hits
end

# ============================== Test body =================================== #

# ── 1. Binary Radix Tree — Structural invariants ─────────────────────── #

@testset "BinaryRadixTree — structural invariants" begin
    offsets = 0:4
    for offset in offsets, D in (Val(2), Val(3)), n in 1:20
        enc = build_encoding(D, n, offset)
        brt = BinaryRadixTree(enc)

        ntotal = 2n - 1

        # Node counts
        @test brt.nleaf == n
        @test length(brt.left)   == ntotal
        @test length(brt.right)  == ntotal
        @test length(brt.escape) == ntotal
        @test length(brt.parent) == ntotal

        # Root identity
        @test brt.root === (n >= 2 ? Int32(1) : Int32(0))
        if brt.root != 0
            @test brt.parent[Int(brt.root)] == 0
        end

        # All parent IDs valid
        @test all(p -> 0 <= p < Int32(ntotal + 1), brt.parent)

        # Karras split rules for internal nodes
        codes = enc.codes
        leaf_offset = n - 1
        for i in 1:(n - 1)
            lo, hi = ns_mod._find_range(codes, i)
            split = ns_mod._split_position(codes, lo, hi)

            l = brt.left[i]
            r = brt.right[i]

            @test ns_mod.is_internal_id(l, n) || ns_mod.is_leaf_id(l, n)
            @test ns_mod.is_internal_id(r, n) || ns_mod.is_leaf_id(r, n)

            if ns_mod.is_leaf_id(l, n)
                @test ns_mod.leaf_index(l, n) == lo
                @test l == Int32(leaf_offset + lo)
            else
                @test l == Int32(split)
            end

            if ns_mod.is_leaf_id(r, n)
                @test ns_mod.leaf_index(r, n) == hi
                @test r == Int32(leaf_offset + hi)
            else
                @test r == Int32(split + 1)
            end
        end

        # Escape links within bounds
        @test all(e -> 0 <= e <= Int32(ntotal), brt.escape)
    end
end

# ── 1b. Identical-code degenerate case ───────────────────────────────── #

@testset "BinaryRadixTree — identical Morton codes" begin
    for D in (Val(2), Val(3)), n in (1, 2, 8, 16)
        enc = identical_encoding(D, n)
        brt = BinaryRadixTree(enc)

        @test brt.root === (n >= 2 ? Int32(1) : Int32(0))
        if n >= 2
            @test brt.parent[Int(brt.root)] == 0
        end
        @test count(==(Int32(0)), brt.parent) == 1
    end
end

# ── 2a. LinearBVH — AABB containment ────────────────────────────────── #

@testset "LinearBVH — AABB containment" begin
    for D in (Val(2), Val(3)), n in (2, 8, 32)
        dim = typeof(D).parameters[1]
        coords = ntuple(_ -> collect(range(0.0, stop=1.0, length=n)), dim)
        h = fill(0.1, n)
        enc = dim == 2 ?
            MortonEncoding(coords[1], coords[2], h) :
            MortonEncoding(coords[1], coords[2], coords[3], h)
        brt = BinaryRadixTree(enc)
        lbvh = LinearBVH(enc, brt)

        @test lbvh.brt.root == (n >= 2 ? Int32(1) : Int32(0))
        @test length(lbvh.leaf_coor[1]) == n
        @test length(lbvh.node_aabb.min[1]) == n - 1

        # Leaf coordinates match sorted coordinates
        for d in 1:dim
            @test lbvh.leaf_coor[d] == enc.coord[d]
        end

        # Internal node AABBs enclose both children
        L = brt.left;  R = brt.right
        nmin = lbvh.node_aabb.min;  nmax = lbvh.node_aabb.max
        lcoor = lbvh.leaf_coor

        for i in 1:(n - 1)
            for d in 1:dim
                cmin_l = ns_mod.is_leaf_id(L[i], n) ? lcoor[d][ns_mod.leaf_index(L[i], n)] : nmin[d][ns_mod.internal_index(L[i])]
                cmin_r = ns_mod.is_leaf_id(R[i], n) ? lcoor[d][ns_mod.leaf_index(R[i], n)] : nmin[d][ns_mod.internal_index(R[i])]
                cmax_l = ns_mod.is_leaf_id(L[i], n) ? lcoor[d][ns_mod.leaf_index(L[i], n)] : nmax[d][ns_mod.internal_index(L[i])]
                cmax_r = ns_mod.is_leaf_id(R[i], n) ? lcoor[d][ns_mod.leaf_index(R[i], n)] : nmax[d][ns_mod.internal_index(R[i])]
                @test nmin[d][i] == min(cmin_l, cmin_r)
                @test nmax[d][i] == max(cmax_l, cmax_r)
            end
        end
    end
end

# ── 2b. LinearBVH — node hmax ───────────────────────────────────────── #

@testset "LinearBVH — node hmax" begin
    rng = MersenneTwister(0xBEEF)
    for dim in (2, 3)
        n = 64
        coords = ntuple(_ -> rand(rng, n), dim)
        h = rand(rng, n)
        enc = dim == 2 ?
            MortonEncoding(coords[1], coords[2], h) :
            MortonEncoding(coords[1], coords[2], coords[3], h)
        brt = BinaryRadixTree(enc)
        lbvh = LinearBVH(enc, brt)
        expected = subtree_hmax_reference(enc, brt)
        @test lbvh.node_hmax == expected
    end
end

# ── 2c. LinearBVH — neighbour queries ───────────────────────────────── #

@testset "LinearBVH — neighbour queries" begin
    x = [0.0, 1.0, 0.0, 1.0]
    y = [0.0, 0.0, 1.0, 1.0]
    z = [0.0, 0.0, 0.0, 0.0]
    h = fill(0.1, length(x))

    enc2 = MortonEncoding(x, y, h)
    enc3 = MortonEncoding(x, y, z, h)

    cases_2d = [((0.1, 0.1), 0.25), ((0.9, 0.9), 0.25), ((0.5, 0.5), 0.75)]
    cases_3d = [((0.1, 0.1, 0.0), 0.25), ((0.9, 0.9, 0.0), 0.25), ((0.5, 0.5, 0.0), 0.75)]

    for (enc, cases) in ((enc2, cases_2d), (enc3, cases_3d))
        brt = BinaryRadixTree(enc)
        lbvh = LinearBVH(enc, brt)
        pool = zeros(Int, length(enc.codes))

        for (point, radius) in cases
            expected = brute_force_neighbors(enc, point, radius)
            result = LBVH_query!(pool, lbvh, point, radius)
            got = sort(result.pool[1:result.count])
            @test got == expected
        end
    end
end

# ── 3a. Scatter pruning ─────────────────────────────────────────────── #

@testset "LinearBVH — scatter pruning matches brute force" begin
    rng = MersenneTwister(0xCAFE)
    n = 128
    coords = ntuple(_ -> rand(rng, n), 3)
    h = rand(rng, n) .* 0.2 .+ 0.05
    enc = MortonEncoding(coords[1], coords[2], coords[3], h)
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)
    Kvalid = 1.0
    point = (0.5, 0.5, 0.5)

    h_sorted = enc.h
    coords_sorted = enc.coord

    # Brute-force reference
    brute = Int[]
    @inbounds for i in 1:n
        r2 = (Kvalid * h_sorted[i])^2
        d2 = sum((coords_sorted[d][i] - point[d])^2 for d in 1:3)
        d2 <= r2 && push!(brute, i)
    end
    sort!(brute)

    accel = scatter_neighbors_reference(lbvh, point, Kvalid, h_sorted)
    @test accel == brute
end

# ── 3b. hmax pruning tightens traversal ──────────────────────────────── #

@testset "LinearBVH — hmax pruning tightens traversal" begin
    rng = MersenneTwister(0x1234)
    n = 256
    coords = ntuple(_ -> rand(rng, n), 3)
    h = rand(rng, n) .* 0.3 .+ 0.01
    enc = MortonEncoding(coords[1], coords[2], coords[3], h)
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)
    Kvalid = 1.0
    point = (0.1, 0.2, 0.3)

    global_hmax = maximum(h)
    node_min = lbvh.node_aabb.min
    node_max = lbvh.node_aabb.max
    left = brt.left
    escape = brt.escape
    nleaf = brt.nleaf

    # Count visits with global hmax
    visits_global = 0
    node = brt.root
    while node != 0
        if ns_mod.is_leaf_id(node, nleaf)
            node = escape[Int(node)]; continue
        end
        visits_global += 1
        idx = ns_mod.internal_index(node)
        r2 = (Kvalid * global_hmax)^2
        d2 = ns_mod._squared_distance_point_aabb(point, node_min, node_max, idx)
        node = (d2 <= r2) ? left[idx] : escape[idx]
    end

    # Count visits with per-node hmax
    visits_hmax = 0
    node = brt.root
    while node != 0
        if ns_mod.is_leaf_id(node, nleaf)
            node = escape[Int(node)]; continue
        end
        visits_hmax += 1
        idx = ns_mod.internal_index(node)
        r2 = (Kvalid * lbvh.node_hmax[idx])^2
        d2 = ns_mod._squared_distance_point_aabb(point, node_min, node_max, idx)
        node = (d2 <= r2) ? left[idx] : escape[idx]
    end

    @test visits_hmax <= visits_global
end

@testset "LinearBVH - line traversal matches brute force" begin
    x = [0.0, 0.1, 0.2, 0.3, 0.4]
    y = [0.0, 0.15, 0.3, 0.6, 1.0]
    h = [0.05, 0.2, 0.35, 0.55, 0.2]

    enc = MortonEncoding(x, y, h)
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)

    origin = (0.0, 0.0)
    direction = (1.0, 0.0)
    Kvalid = 1.0
    radius2 = 0.04

    gather_expected = line_neighbors_reference(lbvh, origin, direction, _ -> radius2)
    scatter_expected = line_neighbors_reference(lbvh, origin, direction, leaf -> (Kvalid * lbvh.leaf_h[leaf])^2)
    symmetric_expected = line_neighbors_reference(lbvh, origin, direction, leaf -> max(radius2, (Kvalid * lbvh.leaf_h[leaf])^2))

    gather_hits = Int[]
    leaf_idx = 0
    d2 = 0.0
    ns_mod.@LBVH_gather_line_traversal lbvh origin direction radius2 leaf_idx d2 begin
        push!(gather_hits, leaf_idx)
    end
    sort!(gather_hits)

    scatter_hits = Int[]
    hb = 0.0
    ns_mod.@LBVH_scatter_line_traversal lbvh origin direction Kvalid leaf_idx d2 hb begin
        push!(scatter_hits, leaf_idx)
    end
    sort!(scatter_hits)

    symmetric_hits = Int[]
    ns_mod.@LBVH_symmetric_line_traversal lbvh origin direction Kvalid radius2 leaf_idx d2 hb begin
        push!(symmetric_hits, leaf_idx)
    end
    sort!(symmetric_hits)

    @test gather_hits == gather_expected
    @test scatter_hits == scatter_expected
    @test symmetric_hits == symmetric_expected
end
