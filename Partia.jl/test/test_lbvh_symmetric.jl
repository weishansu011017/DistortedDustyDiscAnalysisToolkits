import Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))

Pkg.activate(temp=true)
Pkg.develop(path=ROOT)
Pkg.instantiate()
Pkg.add("NearestNeighbors")

using Partia
using Partia.NeighborSearch
using NearestNeighbors
using Random

function build_structures(; n::Int=256, D::Int=3, seed::Int=2025)
    rng = MersenneTwister(seed)
    T = Float64
    coords = [rand(rng, T, n) for _ in 1:D]
    h_orig = 0.05 .+ 0.10 .* rand(rng, T, n)

    enc = D == 3 ? MortonEncoding(coords[1], coords[2], coords[3]) : MortonEncoding(coords[1], coords[2])
    brt = BinaryRadixTree(enc)
    lbvh = LinearBVH(enc, brt)

    order = enc.order
    h_lbvh = similar(h_orig)
    @inbounds for i in eachindex(order)
        h_lbvh[i] = h_orig[order[i]]
    end

    coords_matrix = Matrix{T}(undef, D, n)
    @inbounds for (d, arr) in enumerate(coords)
        coords_matrix[d, :] = arr
    end

    tree = KDTree(coords_matrix)
    pool = zeros(Int, n)
    stack = Vector{Int}(undef, max(1, 2 * length(brt.left_child) + 8))

    return (; coords, coords_matrix, h_orig, h_lbvh, lbvh, tree, pool, stack, order, T, D, n)
end

function kd_symmetric(tree::KDTree, point::AbstractVector{T}, κ::T, h::AbstractVector{T}; tol::T=eps(T)) where {T<:AbstractFloat}
    idx = knn(tree, point, 1)[1][1]
    ha = h[idx]
    radius = κ * ha
    idxs = inrange(tree, point, radius)
    isempty(idxs) && return Int[], ha

    while true
        max_h = maximum(@view h[idxs])
        new_radius = κ * max_h
        new_radius <= radius + tol && break
        radius = new_radius
        idxs = inrange(tree, point, radius)
        isempty(idxs) && break
    end

    sort!(idxs)
    return idxs, ha
end

function lbvh_symmetric!(pool::Vector{Int}, stack::Vector{Int}, lbvh::LinearBVH{D,T},
                          point::NTuple{D,T}, κ::T, h_lbvh::AbstractVector{T}; tol::T=eps(T)) where {D,T<:AbstractFloat}
    result, ha = LBVH_query!(pool, stack, lbvh, point, κ, h_lbvh; atol = tol * κ)
    count = result.count
    if count == 0
        return Int[], ha
    end
    leaf_indices = result.pool
    order = lbvh.enc.order
    neighbors = Vector{Int}(undef, count)
    @inbounds for i in 1:count
        neighbors[i] = order[leaf_indices[i]]
    end
    sort!(neighbors)
    return neighbors, ha
end

function run_trial(; n::Int=256, D::Int=3, κ::Float64=2.5, samples::Int=128, seed::Int=2025)
    data = build_structures(n=n, D=D, seed=seed)
    rng = MersenneTwister(seed + 1)
    idxs = samples >= data.n ? collect(1:data.n) : sort(randperm(rng, data.n)[1:samples])
    tol = eps(data.T)

    for idx in idxs
        point_vec = data.coords_matrix[:, idx]
        offset = 0.2 * data.h_orig[idx] .* randn(rng, data.D)
        query = point_vec .+ offset
        kd_neighbors, ha_kd = kd_symmetric(data.tree, query, data.T(κ), data.h_orig; tol=tol)
        result_neighbors, ha_lbvh = lbvh_symmetric!(data.pool, data.stack, data.lbvh,
                                                    Tuple(query), data.T(κ), data.h_lbvh;
                                                    tol=tol)
        kd_neighbors == result_neighbors || return false
        isapprox(ha_kd, ha_lbvh; atol=10 * tol) || return false
    end
    return true
end

println(run_trial())
println(run_trial(D=2))
