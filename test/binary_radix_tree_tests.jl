using Test
using PhantomRevealer.NeighborSearch

const NS = PhantomRevealer.NeighborSearch

const PRIMES = (37, 61, 97, 131, 197, 263)

function build_coords(dim::Int, n::Int, offset::Int)
    coords = Vector{Vector{Float64}}(undef, dim)
    modulus = 1021
    for d in 1:dim
        prime = PRIMES[d]
        shift = PRIMES[d + dim]
        coords[d] = [Float64(mod(prime * i + shift * offset, modulus)) / modulus for i in 1:n]
    end
    return coords
end

function build_encoding(::Val{2}, n::Int, offset::Int)
    coords = build_coords(2, n, offset)
    return MortonEncoding(coords[1], coords[2])
end

function build_encoding(::Val{3}, n::Int, offset::Int)
    coords = build_coords(3, n, offset)
    return MortonEncoding(coords[1], coords[2], coords[3])
end

identical_encoding(::Val{2}, n::Int) = MortonEncoding(fill(0.5, n), fill(0.5, n))

identical_encoding(::Val{3}, n::Int) = MortonEncoding(fill(0.5, n), fill(0.5, n), fill(0.5, n))

@testset "BinaryRadixTree invariants" begin
    offsets = 0:4
    for offset in offsets
        for D in (Val(2), Val(3))
            for n in 2:20
                enc = build_encoding(D, n, offset)
                brt = BinaryRadixTree(enc)

                @test length(brt.left_child) == n - 1
                @test length(brt.right_child) == n - 1
                @test length(brt.is_leaf_left) == n - 1
                @test length(brt.is_leaf_right) == n - 1
                @test length(brt.node_parent) == n - 1
                @test length(brt.visit_counter) == n - 1
                @test length(brt.leaf_parent) == n

                # Visit counters should stay zero after construction
                @test all(==(zero(eltype(brt.visit_counter))), brt.visit_counter)

                codes = enc.codes
                root_idx = findfirst(==(zero(eltype(brt.node_parent))), brt.node_parent)
                @test root_idx !== nothing
                for (idx, parent) in enumerate(brt.node_parent)
                    if idx == root_idx
                        @test parent == zero(eltype(brt.node_parent))
                    else
                        @test parent > zero(eltype(brt.node_parent))
                    end
                end

                for (idx, parent) in enumerate(brt.leaf_parent)
                    @test parent > zero(eltype(brt.leaf_parent))
                end

                for i in eachindex(brt.left_child)
                    lo, hi = NS._find_range(codes, i)
                    split = NS._split_position(codes, lo, hi)

                    l = brt.left_child[i]
                    r = brt.right_child[i]

                    if brt.is_leaf_left[i]
                        @test 1 <= l <= n
                        @test l == lo
                        @test brt.leaf_parent[l] == i
                    else
                        @test 1 <= l <= n - 1
                        @test l == split
                        @test brt.node_parent[l] == i
                    end

                    if brt.is_leaf_right[i]
                        @test 1 <= r <= n
                        @test r == hi
                        @test brt.leaf_parent[r] == i
                    else
                        @test 1 <= r <= n - 1
                        @test r == split + 1
                        @test brt.node_parent[r] == i
                    end

                    @test lo <= hi
                    @test lo <= l <= hi
                    @test lo <= r <= hi
                end
            end
        end
    end
end

@testset "BinaryRadixTree identical codes" begin
    for D in (Val(2), Val(3))
        for n in (2, 8, 16)
            enc = identical_encoding(D, n)
            brt = BinaryRadixTree(enc)

            @test count(==(zero(eltype(brt.node_parent))), brt.node_parent) == 1
            @test all(>(zero(eltype(brt.leaf_parent))), brt.leaf_parent)
        end
    end
end
