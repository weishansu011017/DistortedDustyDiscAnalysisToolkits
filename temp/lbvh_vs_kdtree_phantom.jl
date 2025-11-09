import Pkg

const ROOT = normpath(joinpath(@__DIR__, ".."))

Pkg.activate(temp=true)
Pkg.develop(path=ROOT)
Pkg.instantiate()
Pkg.add(["BenchmarkTools", "NearestNeighbors"])

using PhantomRevealer
using PhantomRevealer.NeighborSearch
using NearestNeighbors
using BenchmarkTools
using Statistics
using Test
using Random

dump_path = joinpath(ROOT, "test", "testinput", "testdumpfile_00000")

println("Loading dumpfile: $(dump_path)")
prdfs = read_phantom(dump_path)
if isempty(prdfs)
    error("read_phantom returned no particle containers")
end

data = prdfs[1]
df = data.dfdata

names_df = Symbol.(names(df))
coords_syms = Symbol[:x, :y]
if :z in names_df
    push!(coords_syms, :z)
end
D = length(coords_syms)

coords = [Float64.(df[!, sym]) for sym in coords_syms]
base_h = Float64.(df[!, :h])

n = length(base_h)
@assert all(length(arr) == n for arr in coords)

coords_matrix = Matrix{Float64}(undef, D, n)
for (d, arr) in enumerate(coords)
    coords_matrix[d, :] = arr
end
points_tuple = [ntuple(d -> coords[d][i], D) for i in 1:n]

println("Building LBVH...")
enc = D == 3 ? MortonEncoding(coords[1], coords[2], coords[3]) : MortonEncoding(coords[1], coords[2])
brt = BinaryRadixTree(enc)
lbvh = LinearBVH(enc, brt)

pool = zeros(Int, length(enc.codes))
stack = Vector{Int}(undef, max(1, 2 * length(brt.left_child) + 8))
order = enc.order

println("Building KDTree...")
tree = KDTree(coords_matrix)

radius_multipliers = isempty(ARGS) ? [1.5, 2.0, 3.0] : parse.(Float64, ARGS)
println("Radius multipliers: $(radius_multipliers)")

Random.seed!(1234)
sample_count = min(1_000, n)
if sample_count == n
    sample_indices = collect(1:n)
    println("Validating all particles per multiplier.")
else
    sample_indices = sort(randperm(n)[1:sample_count])
    println("Validating $(sample_count) sampled particles per multiplier (seeded).")
end

function run_lbvh!(pool, stack, lbvh, points, radii, counts)
    for i in eachindex(radii)
        counts[i] = LBVH_query!(pool, stack, lbvh, points[i], radii[i])
    end
    return nothing
end

function run_kdtree!(tree, coords, radii, counts)
    @views for i in eachindex(radii)
        counts[i] = length(inrange(tree, coords[:, i], radii[i]))
    end
    return nothing
end

function validate_radius!(pool, stack, lbvh, order, tree, coords, points, radii, idxs)
    if isempty(idxs)
        return
    end
    @views for i in idxs
        count = LBVH_query!(pool, stack, lbvh, points[i], radii[i])
        lbvh_idx = sort!(Int.(order[pool[1:count]]))
        kd_idx = sort!(inrange(tree, coords[:, i], radii[i]))
        @test lbvh_idx == kd_idx
    end
    return nothing
end

for multiplier in radius_multipliers
    println("\nRadius multiplier: $(multiplier) x h")
    radii = multiplier .* base_h
    validate_radius!(pool, stack, lbvh, order, tree, coords_matrix, points_tuple, radii, sample_indices)

    lbvh_counts = Vector{Int}(undef, n)
    lbvh_time = @belapsed run_lbvh!($pool, $stack, $lbvh, $points_tuple, $radii, $lbvh_counts)
    avg_neighbors = mean(lbvh_counts)

    kd_counts = similar(lbvh_counts)
    kd_time = @belapsed run_kdtree!($tree, $coords_matrix, $radii, $kd_counts)
    @test lbvh_counts == kd_counts

    lbvh_per_query = lbvh_time / n * 1e6
    kd_per_query = kd_time / n * 1e6

    println("Average neighbors: $(avg_neighbors)")
    println("Neighbors range: $(minimum(lbvh_counts)) - $(maximum(lbvh_counts))")
    println("LBVH total query time over $n particles: $(lbvh_time) seconds ($(lbvh_per_query) μs/query)")
    println("KDTree total query time over $n particles: $(kd_time) seconds ($(kd_per_query) μs/query)")
    println("LBVH/KDTree time ratio: $(lbvh_time / kd_time)")
end
