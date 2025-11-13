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

@testset "LinearBVH stackless traversal" begin
    rng = MersenneTwister(42)
    n = 2048
    coords = ntuple(_ -> rand(rng, n), 3)
    enc = MortonEncoding(coords[1], coords[2], coords[3])
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)

    pool_stack = zeros(Int, n)
    pool_stackless = zeros(Int, n)
    stack = Vector{Int}(undef, max(1, 2 * length(brt.left_child) + 8))

    queries = 200
    points = [(rand(rng), rand(rng), rand(rng)) for _ in 1:queries]
    radii = [0.05 + 0.2 * rand(rng) for _ in 1:queries]

    for i in 1:queries
        sel_stack = LBVH_query!(pool_stack, stack, lbvh, points[i], radii[i])
        sel_stackless = LBVH_query!(pool_stackless, lbvh, points[i], radii[i])
        @test sel_stackless.count == sel_stack.count
        @test sel_stackless.nearest == sel_stack.nearest
        got_stack = sort(pool_stack[1:sel_stack.count])
        got_stackless = sort(pool_stackless[1:sel_stackless.count])
        @test got_stackless == got_stack
    end

    function run_stack!(iterations)
        for _ in 1:iterations
            for i in 1:queries
                LBVH_query!(pool_stack, stack, lbvh, points[i], radii[i])
            end
        end
        return nothing
    end

    function run_stackless!(iterations)
        for _ in 1:iterations
            for i in 1:queries
                LBVH_query!(pool_stackless, lbvh, points[i], radii[i])
            end
        end
        return nothing
    end

    run_stack!(1)
    run_stackless!(1)

    t_stack = @elapsed run_stack!(3)
    t_stackless = @elapsed run_stackless!(3)
    @info "LBVH traversal timings" stack_based = t_stack stackless = t_stackless
end

