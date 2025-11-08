struct LinearBVH{D, TI <: Unsigned, TF <: AbstractFloat, VI <: AbstractVector{TI}, A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    enc :: MortonEncoding{D, TF, TI, VI}
    brt :: BinaryRadixTree{A, B}
end