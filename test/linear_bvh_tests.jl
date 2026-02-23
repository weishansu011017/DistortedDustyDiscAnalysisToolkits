using Test
using Random
using PhantomRevealer

const NS = PhantomRevealer.NeighborSearch

function expected_neighbor_indices(enc, point::NTuple{D, T}, radius) where {D, T}
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

function subtree_hmax(enc, brt)
    nleaf = brt.nleaf
    nint = nleaf - 1
    out = zeros(eltype(enc.h), nint)
    root = brt.root
    root == 0 && return out

    function visit(node::Int32)
        if NS.is_leaf_id(node, nleaf)
            return enc.h[NS.leaf_index(node, nleaf)]
        end

        idx = NS.internal_index(node)
        l = brt.left[idx]; r = brt.right[idx]
        hl = visit(l)
        hr = visit(r)
        out[idx] = max(hl, hr)
        return out[idx]
    end

    visit(root)
    return out
end

function scatter_neighbors(lbvh, point::NTuple{D,T}, Kvalid::T, hvec) where {D,T}
    node_min = lbvh.node_aabb.min
    node_max = lbvh.node_aabb.max
    leaf_min = lbvh.leaf_aabb.min
    leaf_max = lbvh.leaf_aabb.max
    brt = lbvh.brt
    left = brt.left
    escape = brt.escape
    nleaf = brt.nleaf

    hits = Int[]
    node = brt.root

    if node == 0
        @inbounds for leaf in 1:nleaf
            r2 = (Kvalid * hvec[leaf])^2
            d2 = NS._squared_distance_point_aabb(leaf_min, leaf_max, point, leaf)
            if d2 <= r2
                push!(hits, leaf)
            end
        end
        sort!(hits)
        return hits
    end

    while node != 0
        if NS.is_leaf_id(node, nleaf)
            leaf = NS.leaf_index(node, nleaf)
            r2 = (Kvalid * hvec[leaf])^2
            d2 = NS._squared_distance_point_aabb(leaf_min, leaf_max, point, leaf)
            if d2 <= r2
                push!(hits, leaf)
            end
            node = escape[Int(node)]
            continue
        end

        idx = NS.internal_index(node)
        r2node = (Kvalid * lbvh.node_hmax[idx])^2
        d2node = NS._squared_distance_point_aabb(node_min, node_max, point, idx)
        node = (d2node <= r2node) ? left[idx] : escape[idx]
    end

    sort!(hits)
    return hits
end

@testset "LinearBVH construction invariants" begin
    for D in (Val(2), Val(3))
        for n in (2, 8, 32)
            dim = typeof(D).parameters[1]
            coords = ntuple(_ -> collect(range(0.0, stop=1.0, length=n)), dim)
            h = fill(0.1, n)
            enc = dim == 2 ? MortonEncoding(coords[1], coords[2], h) : MortonEncoding(coords[1], coords[2], coords[3], h)
            brt = BinaryRadixTree(enc)
            lbvh = LinearBVH(enc, brt)

            @test lbvh.brt.root == (n >= 2 ? Int32(1) : Int32(0))
            @test length(lbvh.leaf_aabb.min[1]) == n
            @test length(lbvh.node_aabb.min[1]) == n - 1

            for d in 1:dim
                @test lbvh.leaf_aabb.min[d] == enc.coord[d]
                @test lbvh.leaf_aabb.max[d] == enc.coord[d]
            end

            L = brt.left
            R = brt.right
            node_min = lbvh.node_aabb.min
            node_max = lbvh.node_aabb.max
            leaf_min = lbvh.leaf_aabb.min
            leaf_max = lbvh.leaf_aabb.max

            for i in 1:(n - 1) # internal ids
                for d in 1:dim
                    lmin = NS.is_leaf_id(L[i], n) ? leaf_min[d][NS.leaf_index(L[i], n)] : node_min[d][NS.internal_index(L[i])]
                    rmin = NS.is_leaf_id(R[i], n) ? leaf_min[d][NS.leaf_index(R[i], n)] : node_min[d][NS.internal_index(R[i])]
                    lmax = NS.is_leaf_id(L[i], n) ? leaf_max[d][NS.leaf_index(L[i], n)] : node_max[d][NS.internal_index(L[i])]
                    rmax = NS.is_leaf_id(R[i], n) ? leaf_max[d][NS.leaf_index(R[i], n)] : node_max[d][NS.internal_index(R[i])]
                    @test node_min[d][i] == min(lmin, rmin)
                    @test node_max[d][i] == max(lmax, rmax)
                end
            end
        end
    end
end

@testset "LinearBVH node hmax" begin
    rng = MersenneTwister(0xBEEF)
    for dim in (2, 3)
        n = 64
        coords = ntuple(_ -> rand(rng, n), dim)
        h = rand(rng, n)
        enc = dim == 2 ? MortonEncoding(coords[1], coords[2], h) : MortonEncoding(coords[1], coords[2], coords[3], h)
        brt = BinaryRadixTree(enc)
        lbvh = LinearBVH(enc, brt)
        expected = subtree_hmax(enc, brt)
        @test lbvh.node_hmax == expected
    end
end

@testset "LinearBVH neighbor queries" begin
    x = [0.0, 1.0, 0.0, 1.0]
    y = [0.0, 0.0, 1.0, 1.0]
    z = [0.0, 0.0, 0.0, 0.0]

    h = fill(0.1, length(x))
    enc2 = MortonEncoding(x, y, h)
    enc3 = MortonEncoding(x, y, z, h)

    for (enc, cases) in ((enc2, [((0.1, 0.1), 0.25), ((0.9, 0.9), 0.25), ((0.5, 0.5), 0.75)]),
                         (enc3, [((0.1, 0.1, 0.0), 0.25), ((0.9, 0.9, 0.0), 0.25), ((0.5, 0.5, 0.0), 0.75)]))
        brt = BinaryRadixTree(enc)
        lbvh = LinearBVH(enc, brt)

        pool = zeros(Int, length(enc.codes))
        for (point, radius) in cases
            expected = expected_neighbor_indices(enc, point, radius)

            result1 = LBVH_query!(pool, lbvh, point, radius)
            got = sort(result1.pool[1:result1.count])
            @test got == expected
        end
    end
end

@testset "Scatter pruning matches brute force" begin
    rng = MersenneTwister(0xCAFE)
    dim = 3
    n = 128
    coords = ntuple(_ -> rand(rng, n), dim)
    h = rand(rng, n) .* 0.2 .+ 0.05
    enc = MortonEncoding(coords[1], coords[2], coords[3], h)
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)
    Kvalid = 1.0
    point = (0.5, 0.5, 0.5)

    coords_sorted = enc.coord
    h_sorted = enc.h

    brute = Int[]
    @inbounds for i in 1:n
        r2 = (Kvalid * h_sorted[i])^2
        d2 = zero(eltype(h_sorted))
        for d in 1:dim
            δ = coords_sorted[d][i] - point[d]
            d2 += δ * δ
        end
        if d2 <= r2
            push!(brute, i)
        end
    end
    sort!(brute)

    accel = scatter_neighbors(lbvh, point, Kvalid, h_sorted)
    @test accel == brute
end

@testset "Node hmax pruning tightens traversal" begin
    rng = MersenneTwister(0x1234)
    dim = 3
    n = 256
    coords = ntuple(_ -> rand(rng, n), dim)
    h = rand(rng, n) .* 0.3 .+ 0.01
    enc = MortonEncoding(coords[1], coords[2], coords[3], h)
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)
    Kvalid = 1.0
    point = (0.1, 0.2, 0.3)

    # baseline: global max h
    global_hmax = maximum(h)
    node_visits_baseline = 0
    node_visits_hmax = 0

    brt = lbvh.brt
    node_min = lbvh.node_aabb.min; node_max = lbvh.node_aabb.max
    left = brt.left; escape = brt.escape
    nleaf = brt.nleaf

    node = brt.root
    while node != 0
        if NS.is_leaf_id(node, nleaf)
            node = escape[Int(node)]
            continue
        end
        node_visits_baseline += 1
        idx = NS.internal_index(node)
        r2node = (Kvalid * global_hmax)^2
        d2node = NS._squared_distance_point_aabb(node_min, node_max, point, idx)
        node = (d2node <= r2node) ? left[idx] : escape[idx]
    end

    node = brt.root
    while node != 0
        if NS.is_leaf_id(node, nleaf)
            node = escape[Int(node)]
            continue
        end
        node_visits_hmax += 1
        idx = NS.internal_index(node)
        r2node = (Kvalid * lbvh.node_hmax[idx])^2
        d2node = NS._squared_distance_point_aabb(node_min, node_max, point, idx)
        node = (d2node <= r2node) ? left[idx] : escape[idx]
    end

    @test node_visits_hmax <= node_visits_baseline
end


