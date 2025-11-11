struct InterpolationCatalog{N,G,D,C,L}
    scalar_names :: NTuple{N,Symbol}
    scalar_slots :: NTuple{N,Int}
    scalar_snormalization :: NTuple{N,Bool}   # Shepard normalization flags

    grad_names   :: NTuple{G,Symbol}
    grad_slots   :: NTuple{G,Int}

    div_names    :: NTuple{D,Symbol}
    div_slots    :: NTuple{D,NTuple{3,Int}}

    curl_names   :: NTuple{C,Symbol}
    curl_slots   :: NTuple{C,NTuple{3,Int}}

    ordered_names :: NTuple{L,Symbol}
end

const _DERIV_SUFFIXES = ("ˣ", "ʸ", "ᶻ")

@inline function _ordered_quantity_names(::Val{L}, scalars::NTuple{N,Symbol}, grads::NTuple{G,Symbol}, divs::NTuple{D,Symbol}, curls::NTuple{C,Symbol}) where {L,N,G,D,C}
    names = Symbol[]
    append!(names, scalars)

    for base in grads
        for suffix in _DERIV_SUFFIXES
            push!(names, Symbol("∇", string(base), suffix))
        end
    end

    for base in divs
        push!(names, Symbol("∇⋅", string(base)))
    end

    for base in curls
        for suffix in _DERIV_SUFFIXES
            push!(names, Symbol("∇×", string(base), suffix))
        end
    end

    return tuple(names...)::NTuple{L,Symbol}
end

@inline function _shepard_normalization_flag(scalar_names::NTuple{N,Symbol}) where {N}
    no_norm = (:ρ, :rho)
    return ntuple(i -> scalar_names[i] ∈ no_norm ? false : true, N)
end

function InterpolationCatalog(
    scalar_names::NTuple{N,Symbol}, scalar_slots::NTuple{N,Int},
    grad_names::NTuple{G,Symbol}, grad_slots::NTuple{G,Int},
    div_names::NTuple{D,Symbol}, div_slots::NTuple{D,NTuple{3,Int}},
    curl_names::NTuple{C,Symbol}, curl_slots::NTuple{C,NTuple{3,Int}},
) where {N,G,D,C}
    L = N + 3G + D + 3C
    ordered = _ordered_quantity_names(Val(L), scalar_names, grad_names, div_names, curl_names)
    scalar_norm = _shepard_normalization_flag(scalar_names)
    return InterpolationCatalog{N,G,D,C,L}(
        scalar_names, scalar_slots, scalar_norm, 
        grad_names, grad_slots,
        div_names, div_slots,
        curl_names, curl_slots,
        ordered
    )
end

scalar_index(cat::InterpolationCatalog, name::Symbol) =
    cat.scalar_slots[findfirst(==(name), cat.scalar_names)]

grad_slot(cat::InterpolationCatalog, name::Symbol) =
    cat.grad_slots[findfirst(==(name), cat.grad_names)]

div_slots(cat::InterpolationCatalog, name::Symbol) =
    cat.div_slots[findfirst(==(name), cat.div_names)]

curl_slots(cat::InterpolationCatalog, name::Symbol) =
    cat.curl_slots[findfirst(==(name), cat.curl_names)]

ordered_quantity_names(cat::InterpolationCatalog) = cat.ordered_names