using Test
using Random
using PhantomRevealer

const NS = PhantomRevealer.NeighborSearch

function expected_neighbor_indices(lbvh, point::NTuple{D, T}, radius) where {D, T}
    coords = lbvh.enc.coord
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

function expected_symmetric_neighbors(lbvh, point::NTuple{D, T}, radius_multiplier, h_array; atol=nothing) where {D, T}
    coords = lbvh.enc.coord
    n = length(coords[1])
    dist2 = zeros(T, n)
    nearest_idx = 0
    nearest_dist2 = typemax(T)

    @inbounds for i in 1:n
        d2 = zero(T)
        for d in 1:D
            δ = coords[d][i] - point[d]
            d2 += δ * δ
        end
        dist2[i] = d2
        if d2 < nearest_dist2
            nearest_dist2 = d2
            nearest_idx = i
        end
    end

    ha = h_array[nearest_idx]
    tol = atol === nothing ? eps(T) * radius_multiplier : T(atol)
    hits = Int[]
    @inbounds for i in 1:n
        limit = radius_multiplier * max(ha, h_array[i]) + tol
        if dist2[i] <= limit * limit
            push!(hits, i)
        end
    end
    sort!(hits)
    return hits, ha, dist2
end

@testset "LinearBVH construction invariants" begin
    for D in (Val(2), Val(3))
        for n in (2, 8, 32)
            dim = typeof(D).parameters[1]
            coords = ntuple(_ -> collect(range(0.0, stop=1.0, length=n)), dim)
            enc = dim == 2 ? MortonEncoding(coords[1], coords[2]) : MortonEncoding(coords[1], coords[2], coords[3])
            brt = BinaryRadixTree(enc)
            lbvh = LinearBVH(enc, brt)

            @test lbvh.root == NS._find_root_index(brt)
            @test length(lbvh.leaf_aabb.min[1]) == n
            @test length(lbvh.node_aabb.min[1]) == n - 1
            @test all(==(zero(eltype(brt.visit_counter))), brt.visit_counter)

            for d in 1:dim
                @test lbvh.leaf_aabb.min[d] == enc.coord[d]
                @test lbvh.leaf_aabb.max[d] == enc.coord[d]
            end

            L = brt.left_child
            R = brt.right_child
            LL = brt.is_leaf_left
            RR = brt.is_leaf_right
            node_min = lbvh.node_aabb.min
            node_max = lbvh.node_aabb.max
            leaf_min = lbvh.leaf_aabb.min
            leaf_max = lbvh.leaf_aabb.max

            for i in eachindex(L)
                for d in 1:dim
                    lmin = LL[i] ? leaf_min[d][L[i]] : node_min[d][L[i]]
                    rmin = RR[i] ? leaf_min[d][R[i]] : node_min[d][R[i]]
                    lmax = LL[i] ? leaf_max[d][L[i]] : node_max[d][L[i]]
                    rmax = RR[i] ? leaf_max[d][R[i]] : node_max[d][R[i]]
                    @test node_min[d][i] == min(lmin, rmin)
                    @test node_max[d][i] == max(lmax, rmax)
                end
            end
        end
    end
end

@testset "LinearBVH neighbor queries" begin
    x = [0.0, 1.0, 0.0, 1.0]
    y = [0.0, 0.0, 1.0, 1.0]
    z = [0.0, 0.0, 0.0, 0.0]

    enc2 = MortonEncoding(x, y)
    enc3 = MortonEncoding(x, y, z)

    for (enc, cases) in ((enc2, [((0.1, 0.1), 0.25), ((0.9, 0.9), 0.25), ((0.5, 0.5), 0.75)]),
                         (enc3, [((0.1, 0.1, 0.0), 0.25), ((0.9, 0.9, 0.0), 0.25), ((0.5, 0.5, 0.0), 0.75)]))
        brt = BinaryRadixTree(enc)
        lbvh = LinearBVH(enc, brt)

        pool = zeros(Int, length(enc.codes))
        stack = Vector{Int}(undef, max(1, 2 * length(brt.left_child) + 8))

        for (point, radius) in cases
            expected = expected_neighbor_indices(lbvh, point, radius)

            result1 = LBVH_query!(pool, stack, lbvh, point, radius)
            got = sort(result1.pool[1:result1.count])
            @test got == expected

            result2 = LBVH_query(lbvh, point, radius)
            @test result2.count == result1.count
            @test sort(result2.pool[1:result2.count]) == expected
        end
    end
end

@testset "LinearBVH identical points" begin
    n = 16
    coords = fill(0.5, n)
    enc = MortonEncoding(coords, coords)
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)

    pool = zeros(Int, n)
    stack = Vector{Int}(undef, max(1, 2 * length(brt.left_child) + 8))
    qpt = (0.5, 0.5)
    result = LBVH_query!(pool, stack, lbvh, qpt, 0.0)
    @test result.count == n
    @test sort(result.pool[1:result.count]) == collect(1:n)
end

@testset "LinearBVH symmetric SPH queries" begin
    Random.seed!(0xDEADBEEF)
    for dim in (2, 3)
        n = 128
        coords = ntuple(_ -> rand(Float64, n), dim)
        enc = dim == 2 ? MortonEncoding(coords[1], coords[2]) : MortonEncoding(coords[1], coords[2], coords[3])
        brt = BinaryRadixTree(enc)
        lbvh = LinearBVH(enc, brt)

        h_array = 0.05 .+ 0.45 .* rand(Float64, n)
        pool = zeros(Int, n)
        stack = Vector{Int}(undef, max(1, 2 * length(brt.left_child) + 8))

        for trial in 1:10
            point = ntuple(_ -> rand(Float64), dim)
            multiplier = rand(Float64) * 1.5 + 1.0

            expected_hits, expected_ha, dist2 = expected_symmetric_neighbors(lbvh, point, multiplier, h_array)

            selection1, ha1 = LBVH_query!(pool, stack, lbvh, point, multiplier, h_array)
            @test ha1 == expected_ha
            @test sort(selection1.pool[1:selection1.count]) == expected_hits

            Ttol = eltype(h_array)
            tol = eps(Ttol) * multiplier
            for idx in selection1.pool[1:selection1.count]
                limit = multiplier * max(ha1, h_array[idx]) + tol
                @test dist2[idx] <= limit * limit
            end

            selection2, ha2 = LBVH_query(lbvh, point, multiplier, h_array)
            @test ha2 == expected_ha
            @test sort(selection2.pool[1:selection2.count]) == expected_hits
            for idx in selection2.pool[1:selection2.count]
                limit = multiplier * max(ha2, h_array[idx]) + tol
                @test dist2[idx] <= limit * limit
            end
        end
    end
end
