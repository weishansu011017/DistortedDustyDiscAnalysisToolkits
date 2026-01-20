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
    h = fill(0.1, n)
    return MortonEncoding(coords[1], coords[2], h)
end

function build_encoding(::Val{3}, n::Int, offset::Int)
    coords = build_coords(3, n, offset)
    h = fill(0.1, n)
    return MortonEncoding(coords[1], coords[2], coords[3], h)
end

identical_encoding(::Val{2}, n::Int) = MortonEncoding(fill(0.5, n), fill(0.5, n), fill(0.5, n))

identical_encoding(::Val{3}, n::Int) = MortonEncoding(fill(0.5, n), fill(0.5, n), fill(0.5, n), fill(0.5, n))

@testset "BinaryRadixTree invariants" begin
    offsets = 0:4
    for offset in offsets
        for D in (Val(2), Val(3))
            for n in 1:20
                enc = build_encoding(D, n, offset)
                brt = BinaryRadixTree(enc)

                ntotal = 2n - 1
                @test brt.root === (n >= 2 ? Int32(1) : Int32(0))
                @test brt.nleaf == n
                @test length(brt.left) == ntotal
                @test length(brt.right) == ntotal
                @test length(brt.escape) == ntotal
                @test length(brt.parent) == ntotal

                # parent[root] == 0 and all parents are valid IDs
                if brt.root != 0
                    @test brt.parent[Int(brt.root)] == 0
                end
                @test all(p -> p >= 0 && p < Int32(ntotal + 1), brt.parent)

                # Internal node children follow Karras split rules in unified ID space
                codes = enc.codes
                leaf_offset = n - 1
                for i in 1:(n - 1)  # internal ids
                    lo, hi = NS._find_range(codes, i)
                    split = NS._split_position(codes, lo, hi)

                    l = brt.left[i]
                    r = brt.right[i]

                    @test NS.is_internal_id(l, n) || NS.is_leaf_id(l, n)
                    @test NS.is_internal_id(r, n) || NS.is_leaf_id(r, n)

                    if NS.is_leaf_id(l, n)
                        @test NS.leaf_index(l, n) == lo
                        @test l == Int32(leaf_offset + lo)
                    else
                        @test l == Int32(split)
                    end

                    if NS.is_leaf_id(r, n)
                        @test NS.leaf_index(r, n) == hi
                        @test r == Int32(leaf_offset + hi)
                    else
                        @test r == Int32(split + 1)
                    end
                end

                # Escape links stay within [0, ntotal]
                @test all(e -> 0 <= e <= Int32(ntotal), brt.escape)
            end
        end
    end
end

@testset "BinaryRadixTree identical codes" begin
    for D in (Val(2), Val(3))
        for n in (1, 2, 8, 16)
            enc = identical_encoding(D, n)
            brt = BinaryRadixTree(enc)

            @test brt.root === (n >= 2 ? Int32(1) : Int32(0))
            if n >= 2
                @test brt.parent[Int(brt.root)] == 0
            end
            @test count(==(Int32(0)), brt.parent) == 1
        end
    end
end
