import Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))

Pkg.activate(temp=true)
Pkg.develop(path=ROOT)
Pkg.instantiate()
Pkg.add(["NearestNeighbors"])  # ensure KDTree dependency

using PhantomRevealer
using PhantomRevealer.NeighborSearch
using NearestNeighbors
using Random
using Statistics

function load_dataset()
    dump_path = joinpath(ROOT, "test", "testinput", "testdumpfile_00000")
    prdfs = read_phantom(dump_path)
    isempty(prdfs) && error("read_phantom returned no particle containers")
    df = prdfs[1].dfdata
    names_df = Symbol.(names(df))
    coords_syms = Symbol[:x, :y]
    if :z in names_df
        push!(coords_syms, :z)
    end
    coords = [Float64.(df[!, sym]) for sym in coords_syms]
    base_h = Float64.(df[!, :h])
    return coords, base_h
end

function build_structures(coords)
    D = length(coords)
    n = length(coords[1])
    coords_matrix = Matrix{Float64}(undef, D, n)
    for (d, arr) in enumerate(coords)
        coords_matrix[d, :] = arr
    end
    points_tuple = [ntuple(d -> coords[d][i], D) for i in 1:n]
    enc = D == 3 ? MortonEncoding(coords[1], coords[2], coords[3]) : MortonEncoding(coords[1], coords[2])
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)
    pool = zeros(Int, length(enc.codes))
    stack_capacity = max(1, 2 * length(brt.left_child) + 8)
    stack = Vector{Int}(undef, stack_capacity)
    order = enc.order
    tree = KDTree(coords_matrix)
    return lbvh, pool, stack, order, tree, points_tuple, coords_matrix
end

coords, base_h = load_dataset()
lbvh, pool, stack, order, tree, points_tuple, coords_matrix = build_structures(coords)
D = length(coords)

function dist2_point(coords, point, idx)
    s = 0.0
    @inbounds for d in 1:length(coords)
        δ = coords[d][idx] - point[d]
        s += δ * δ
    end
    return s
end

n = length(coords[1])

function run_batch(batch_name, sample_indices; multipliers=(1.5, 2.0, 3.0), offset_scale=0.25, rng_seed=0)
    rng = MersenneTwister(rng_seed == 0 ? rand(UInt) : rng_seed)
    println("\n=== Batch $(batch_name) ===")
    for multiplier in multipliers
        radii = multiplier .* base_h
        bad_nearest = Int[]
        kd_mismatch = Int[]
        empty_hits = Int[]
        examples = NamedTuple[]

        for idx in sample_indices
            # sample point slightly offset from particle to obtain non-zero separations
            dir = randn!(rng, zeros(Float64, D))
            norm_dir = sqrt(sum(dir .^ 2))
            if norm_dir == 0.0
                dir[1] = 1.0
                norm_dir = 1.0
            end
            dir ./= norm_dir
            offset = offset_scale * base_h[idx] .* dir
            point = ntuple(d -> points_tuple[idx][d] + offset[d], D)
            result = LBVH_query!(pool, stack, lbvh, point, radii[idx])
            if result.count == 0
                push!(empty_hits, idx)
                continue
            end
            pool_indices = order[valid_indices(result)]
            best_pool_index = pool_indices[1]
            best_dist2 = dist2_point(coords, point, best_pool_index)
            @inbounds for leaf_idx in pool_indices
                d2 = dist2_point(coords, point, leaf_idx)
                if d2 < best_dist2
                    best_dist2 = d2
                    best_pool_index = leaf_idx
                end
            end

            nearest_lbvh = order[nearest_index(result)]
            if nearest_lbvh != best_pool_index
                push!(bad_nearest, idx)
            end

            kd_idx = knn(tree, collect(point), 1)[1][1]
            kd_dist = sqrt(sum((point .- coords_matrix[:, kd_idx]).^2))
            if nearest_lbvh != kd_idx
                push!(kd_mismatch, idx)
            end

            if length(examples) < 5
                push!(examples, (sample=idx,
                                 multiplier=multiplier,
                                 radius=radii[idx],
                                 count=result.count,
                                 nearest_lbvh=nearest_lbvh,
                                 lbvh_dist=sqrt(best_dist2),
                                 kd_idx=kd_idx,
                                 kd_dist=kd_dist))
            end
        end

        println("Multiplier $(multiplier)")
        println("  total samples: $(length(sample_indices))")
        println("  empty queries: $(length(empty_hits))")
        println("  nearest mismatch vs LBVH pool argmin: $(length(bad_nearest))")
        println("  nearest mismatch vs KDTree knn: $(length(kd_mismatch))")
        if !isempty(examples)
            println("  Example distances:")
            for ex in examples
                println("    sample $(ex.sample): radius $(ex.radius), count $(ex.count), lbvh dist $(ex.lbvh_dist), kd dist $(ex.kd_dist)")
            end
        end
        if !isempty(bad_nearest)
            println("    nearest mismatches (first 10): ", bad_nearest[1:min(end, 10)])
        end
        if !isempty(kd_mismatch)
            println("    KD mismatches (first 10): ", kd_mismatch[1:min(end, 10)])
        end
    end
end

sample_size = min(1_000, n)
Random.seed!(1234)
sample_indices_global = sample_size == n ? collect(1:n) : sort(randperm(n)[1:sample_size])
run_batch("global random", sample_indices_global)

if D == 3
    # choose indices closest to midplane (|z| smallest)
    Random.seed!(5678)
    zvals = coords[3]
    order_midplane = sortperm(abs.(zvals))[1:sample_size]
    run_batch("midplane slice", sort(order_midplane))

    # choose points far from origin along z to stress different neighborhood shapes
    Random.seed!(91011)
    high_z = sortperm(abs.(zvals), rev=true)[1:sample_size]
    run_batch("high-|z| slice", sort(high_z))
else
    # alternative selection for 2D data: radial ordering
    Random.seed!(5678)
    xvals = coords[1]; yvals = coords[2]
    radii = sqrt.(xvals .^ 2 .+ yvals .^ 2)
    center = sortperm(radii)[1:sample_size]
    run_batch("near origin", sort(center))

    Random.seed!(91011)
    outer = sortperm(radii, rev=true)[1:sample_size]
    run_batch("outer ring", sort(outer))
end
