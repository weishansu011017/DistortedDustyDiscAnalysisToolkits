struct InterpolationCatalog{N,G,D,C,L}
    scalar_names :: NTuple{N,Symbol}
    scalar_slots :: NTuple{N,Int}
    scalar_snormalization :: NTuple{N,Bool}         # Shepard normalization flags

    grad_names   :: NTuple{G,Symbol}
    grad_slots   :: NTuple{G,Int}

    div_names    :: NTuple{D,Symbol}
    div_slots    :: NTuple{D,NTuple{3,Int}}

    curl_names   :: NTuple{C,Symbol}
    curl_slots   :: NTuple{C,NTuple{3,Int}}

    ordered_names :: NTuple{L,Symbol}
end

struct InterpolationCatalogConcise{N,G,D,C}
    scalar_slots :: NTuple{N,Int}
    scalar_snormalization :: NTuple{N,Bool}         # Shepard normalization flags
    grad_slots   :: NTuple{G,Int}
    div_slots    :: NTuple{D,NTuple{3,Int}}
    curl_slots   :: NTuple{C,NTuple{3,Int}}
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

"""
InterpolationCatalog(scalar_names, scalar_slots,
                     grad_names,   grad_slots,
                     div_names,    div_slots,
                     curl_names,   curl_slots)

Structured registry describing which physical quantities an interpolation
routine should compute, together with their column locations in the underlying
particle dataset.

`InterpolationCatalog` groups scalar fields, gradient fields, divergence
targets, and curl targets into a single statically-typed catalog. It also
constructs an ordered list of all requested output quantity names, expanding
vector derivatives into component-wise entries, and generates Shepard
normalization flags for all scalar quantities.

# Parameters
- `scalar_names::NTuple{N,Symbol}`  
  Names of scalar quantities to interpolate (e.g. `(:ρ, :u, :T)`).
- `scalar_slots::NTuple{N,Int}`  
  Column indices of each scalar in the particle field array.
- `grad_names::NTuple{G,Symbol}`  
  Base names of quantities whose spatial gradients ∇A should be returned.
- `grad_slots::NTuple{G,Int}`  
  Column indices of each gradient target.
- `div_names::NTuple{D,Symbol}`  
  Names of vector fields for which divergence ∇⋅A is required.
- `div_slots::NTuple{D,NTuple{3,Int}}`  
  `(Ax, Ay, Az)` column indices for each divergence field.
- `curl_names::NTuple{C,Symbol}`  
  Names of vector fields for which curl ∇×A is required.
- `curl_slots::NTuple{C,NTuple{3,Int}}`  
  `(Ax, Ay, Az)` column indices for each curl field.

# Field Expansion Rules
- Each scalar contributes 1 output.
- Each gradient contributes 3 outputs, named  
  `(:∇Aˣ, :∇Aʸ, :∇Aᶻ)`.
- Each divergence contributes 1 output, named  
  `(:∇⋅A)`.
- Each curl contributes 3 outputs, named  
  `(:∇×Aˣ, :∇×Aʸ, :∇×Aᶻ)`.

# Shepard Normalization
Scalar quantities named `:ρ` or `:rho` automatically disable Shepard
normalization; all others default to enabled.

# Returns
- `InterpolationCatalog{N,G,D,C,L}`  
  A fully constructed catalog containing:
  - scalar names, slots, and normalization flags
  - gradient names and slots
  - divergence names and triple-component slots
  - curl names and triple-component slots
  - `ordered_names::NTuple{L,Symbol}` — all output quantity names in a
    deterministic, expanded order.

"""
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

"""
    to_concise_catalog(cat::InterpolationCatalog{N,G,D,C,L})
        -> InterpolationCatalogConcise{N,G,D,C}

Construct a concise interpolation catalog by stripping name information from a
full `InterpolationCatalog`. The resulting `InterpolationCatalogConcise`
retains only slot indices and Shepard–normalization flags, making it suitable
for high-performance interpolation paths where symbolic field names are not
required.

# Parameters
- `cat::InterpolationCatalog{N,G,D,C,L}`  
  Full catalog containing scalar, gradient, divergence, and curl quantity
  metadata, including symbolic names, slot locations, and normalization flags.

# Returns
- `InterpolationCatalogConcise{N,G,D,C}`  
  A compact catalog holding only:
  - `scalar_slots  :: NTuple{N,Int}`
  - `scalar_snormalization :: NTuple{N,Bool}`
  - `grad_slots    :: NTuple{G,Int}`
  - `div_slots     :: NTuple{D,NTuple{3,Int}}`
  - `curl_slots    :: NTuple{C,NTuple{3,Int}}`

This reduction eliminates name handling overhead and is intended for
performance-critical inner interpolation kernels.
"""
@inline function to_concise_catalog(catalog::InterpolationCatalog{N,G,D,C,L}) where {N,G,D,C,L}
    return InterpolationCatalogConcise{N,G,D,C}(
        catalog.scalar_slots,
        catalog.scalar_snormalization,
        catalog.grad_slots,
        catalog.div_slots,
        catalog.curl_slots,
    )
end