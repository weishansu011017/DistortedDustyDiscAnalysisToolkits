@inline GridDataset_params_TYPE(::Type{T}) where {T <: AbstractFloat} = Dict{Symbol,Union{String, Int, Bool, T}}

struct GridDataset{L, TF <: AbstractFloat, G <: AbstractGrid{TF}}
    data   :: GridBundle{L,G}
    params :: Dict{Symbol,Union{String, Int, Bool, TF}}
end