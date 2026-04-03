using Partia

x = [0.0, 1.0, 0.0, 1.0]
y = [0.0, 0.0, 1.0, 1.0]
z = [0.0, 0.0, 0.0, 0.0]

enc = MortonEncoding(x, y, z)
brt = BinaryRadixTree(enc)
lbvh = LinearBVH(enc, brt)

pool = zeros(Int, length(x))
stack = Vector{Int}(undef, max(1, 2 * length(brt.left_child) + 8))

point1 = (0.1, 0.1, 0.0)
radius = 0.3
count1 = LBVH_query!(pool, stack, lbvh, point1, radius)
println("Query point $(point1) radius=$(radius) -> $(count1) hits: $(pool[1:count1])")

point2 = (0.9, 0.9, 0.0)
count2 = LBVH_query!(pool, stack, lbvh, point2, radius)
println("Query point $(point2) radius=$(radius) -> $(count2) hits: $(pool[1:count2])")
