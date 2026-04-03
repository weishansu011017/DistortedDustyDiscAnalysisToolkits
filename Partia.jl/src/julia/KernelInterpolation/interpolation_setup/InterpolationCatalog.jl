

# For field suffixes
const _XYZ_SUFFIXES = (:x, :y, :z)
const _DERIV_SUFFIXES = ("ˣ", "ʸ", "ᶻ")



struct InterpolationCatalog{D, N, G, Div, C, L}
    scalar_names :: NTuple{N,Symbol}
    scalar_slots :: NTuple{N,Int}
    scalar_snormalization :: NTuple{N,Bool}         # Shepard normalization flags

    grad_names   :: NTuple{G,Symbol}
    grad_slots   :: NTuple{G,Int}

    div_names    :: NTuple{Div,Symbol}
    div_slots    :: NTuple{Div,NTuple{D,Int}}

    curl_names   :: NTuple{C,Symbol}
    curl_slots   :: NTuple{C,NTuple{D,Int}}

    ordered_names :: NTuple{L,Symbol}
end

struct InterpolationCatalogConcise{D, N, G, Div,C}
    scalar_slots :: NTuple{N,Int}
    scalar_snormalization :: NTuple{N,Bool}         # Shepard normalization flags
    grad_slots   :: NTuple{G,Int}
    div_slots    :: NTuple{Div,NTuple{D,Int}}
    curl_slots   :: NTuple{C,NTuple{D,Int}}
end

"""
    InterpolationCatalog(::Val{D};
                         scalar_names::Tuple{Vararg{Symbol}}=(),
                         scalar_slots::Tuple{Vararg{Int}}=(),
                         grad_names::Tuple{Vararg{Symbol}}=(),
                         grad_slots::Tuple{Vararg{Int}}=(),
                         div_names::Tuple{Vararg{Symbol}}=(),
                         div_slots::Tuple{Vararg{NTuple{D,Int}}}=(),
                         curl_names::Tuple{Vararg{Symbol}}=(),
                         curl_slots::Tuple{Vararg{NTuple{D,Int}}}=()) where {D}

Construct an `InterpolationCatalog` from explicitly provided quantity names and
their corresponding column locations in the particle dataset.

This constructor groups scalar fields, gradient fields, divergence targets, and
curl targets into a single statically-typed catalog. It also computes the
ordered expanded output quantity names and the Shepard-normalization flags for
all scalar quantities.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `scalar_names` | `Tuple{Vararg{Symbol}}` | `()` | Names of scalar quantities to interpolate directly. |
| `scalar_slots` | `Tuple{Vararg{Int}}` | `()` | Column indices of the scalar quantities in the particle field array. |
| `grad_names` | `Tuple{Vararg{Symbol}}` | `()` | Base names of scalar quantities whose spatial gradients should be returned. |
| `grad_slots` | `Tuple{Vararg{Int}}` | `()` | Column indices of the gradient targets. |
| `div_names` | `Tuple{Vararg{Symbol}}` | `()` | Names of vector fields for which divergence should be computed. |
| `div_slots` | `Tuple{Vararg{NTuple{D,Int}}}` | `()` | Component-wise column indices of the divergence targets. |
| `curl_names` | `Tuple{Vararg{Symbol}}` | `()` | Names of vector fields for which curl should be computed. |
| `curl_slots` | `Tuple{Vararg{NTuple{D,Int}}}` | `()` | Component-wise column indices of the curl targets. |

# Returns
- `InterpolationCatalog{D, N, G, Div, C, L}`: A fully constructed interpolation
  catalog containing quantity names, column-slot mappings, Shepard
  normalization flags, and the ordered expanded output quantity names.

"""
function InterpolationCatalog(::Val{D};
    scalar_names::Tuple{Vararg{Symbol}} = (),
    scalar_slots::Tuple{Vararg{Int}} = (),
    grad_names::Tuple{Vararg{Symbol}} = (),
    grad_slots::Tuple{Vararg{Int}} = (),
    div_names::Tuple{Vararg{Symbol}} = (),
    div_slots::Tuple{Vararg{NTuple{D,Int}}} = (),
    curl_names::Tuple{Vararg{Symbol}} = (),
    curl_slots::Tuple{Vararg{NTuple{D,Int}}} = (),
) where {D}
    N = length(scalar_names)
    G = length(grad_names)
    Div = length(div_names)
    C = length(curl_names)

    length(scalar_slots) == N || throw(DimensionMismatch("scalar_slots length $(length(scalar_slots)) != scalar_names length $N"))
    length(grad_slots)   == G || throw(DimensionMismatch("grad_slots length $(length(grad_slots)) != grad_names length $G"))
    length(div_slots)    == Div || throw(DimensionMismatch("div_slots length $(length(div_slots)) != div_names length $Div"))
    length(curl_slots)   == C || throw(DimensionMismatch("curl_slots length $(length(curl_slots)) != curl_names length $C"))

    L = N + D*G + Div + D*C  # Total number of output quantities after expansion
    ordered = _ordered_quantity_names(Val(L), Val(D), scalar_names, grad_names, div_names, curl_names)
    scalar_norm = _shepard_normalization_flag(scalar_names)

    return InterpolationCatalog{D, N, G, Div, C, L}(
        scalar_names, scalar_slots, scalar_norm,
        grad_names, grad_slots,
        div_names, div_slots,
        curl_names, curl_slots,
        ordered
    )
end

"""
    InterpolationCatalog(column_names::Tuple{Vararg{Symbol}}, ::Val{D};
                         scalars::Tuple{Vararg{Symbol}}=(),
                         gradients::Tuple{Vararg{Symbol}}=(),
                         divergences::Tuple{Vararg{Symbol}}=(),
                         curls::Tuple{Vararg{Symbol}}=()) where {D}

Construct an `InterpolationCatalog` from a tuple of dataset column names and
symbolic quantity requests.

This method resolves requested scalar, gradient, divergence, and curl targets
against `column_names`, converts them into the corresponding column indices,
and forwards the resulting slot tuples to the lower-level
`InterpolationCatalog` constructor. Vector-valued requests in `divergences`
and `curls` are expanded into component column names through
`_vector_components(..., Val(D))` before slot lookup.

# Parameters
- `column_names::Tuple{Vararg{Symbol}}`: Ordered tuple of available column
  names in the underlying particle dataset.
- `::Val{D}`: Compile-time spatial dimension used to expand vector quantity
  names into component-wise column names.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `scalars` | `Tuple{Vararg{Symbol}}` | `()` | Names of scalar quantities to interpolate directly. |
| `gradients` | `Tuple{Vararg{Symbol}}` | `()` | Names of scalar quantities whose spatial gradients should be computed. |
| `divergences` | `Tuple{Vararg{Symbol}}` | `()` | Base names of vector fields whose divergences should be computed. |
| `curls` | `Tuple{Vararg{Symbol}}` | `()` | Base names of vector fields whose curls should be computed. |

# Returns
- `InterpolationCatalog{D, N, G, Div, C, L}`: A fully constructed interpolation
  catalog with all requested quantity names resolved to their corresponding
  column slots.

"""
function InterpolationCatalog(column_names :: Tuple{Vararg{Symbol}}, :: Val{D};
    scalars :: Tuple{Vararg{Symbol}} = (),
    gradients::Tuple{Vararg{Symbol}} = (),
    divergences::Tuple{Vararg{Symbol}} = (),
    curls::Tuple{Vararg{Symbol}} = (),
) where {D}

    # Make sure column names are unique to avoid ambiguous slot lookups
    allunique(column_names) || throw(ArgumentError("column_names must be unique"))

    N = length(scalars)
    G = length(gradients)
    Div = length(divergences)
    C = length(curls)

    # Collect the name to column index mapping for all requested quantities
    column_index = Dict{Symbol,Int}(name => idx for (idx, name) in enumerate(column_names))

    # Get the index of each quantity in the column_names tuple.
    ## scalars
    scalar_slots = ntuple(i -> column_index[scalars[i]], N)

    ## gradients
    grad_slots = ntuple(i -> column_index[gradients[i]], G)

    ## divergences
    div_slots = ntuple(i ->begin
        comps = _vector_components(divergences[i], Val(D))
        ntuple(i -> column_index[comps[i]], D)
    end, Div)

    ## curls
    curl_slots = ntuple(i ->begin
        comps = _vector_components(curls[i], Val(D))
        ntuple(i -> column_index[comps[i]], D)
    end, C)

    return InterpolationCatalog(Val(D);
        scalar_names = scalars,
        scalar_slots = scalar_slots,
        grad_names = gradients,
        grad_slots = grad_slots,
        div_names = divergences,
        div_slots = div_slots,
        curl_names = curls,
        curl_slots = curl_slots,
    )
end

# Toolbox
@inline function _vector_components(name :: Symbol, ::Val{3}) :: NTuple{3, Symbol}
    return ntuple(i -> Symbol(name, _XYZ_SUFFIXES[i]), 3)
end

@inline function _vector_components(name :: Symbol, ::Val{2}) :: NTuple{2, Symbol}
    return ntuple(i -> Symbol(name, _XYZ_SUFFIXES[i]), 2)
end

@inline function _ordered_quantity_names(:: Val{L}, :: Val{D}, scalars :: NTuple{N,Symbol}, grads :: NTuple{G,Symbol}, divs :: NTuple{Div,Symbol}, curls :: NTuple{C,Symbol}) where {L, N, G, Div, C, D}
    names = Symbol[]
    append!(names, scalars)

    for base in grads
        for i in 1:D
             push!(names, Symbol("∇", string(base), _DERIV_SUFFIXES[i]))
        end
    end

    for base in divs
        push!(names, Symbol("∇⋅", string(base)))
    end

    for base in curls
        for i in 1:D
            push!(names, Symbol("∇×", string(base), _DERIV_SUFFIXES[i]))
        end
    end

    return tuple(names...) :: NTuple{L,Symbol}
end

@inline function _shepard_normalization_flag(scalar_names::NTuple{N,Symbol}) where {N}
    no_norm = (:ρ, :rho)
    return ntuple(i -> scalar_names[i] ∈ no_norm ? false : true, N)
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
    to_concise_catalog(cat::InterpolationCatalog{D, N, G, Div, C, L})
        -> InterpolationCatalogConcise{D, N, G, Div, C}

Construct a concise interpolation catalog by stripping name information from a
full `InterpolationCatalog`. The resulting `InterpolationCatalogConcise`
retains only slot indices and Shepard–normalization flags, making it suitable
for high-performance interpolation paths where symbolic field names are not
required.

# Parameters
- `cat::InterpolationCatalog{D, N, G, Div, C, L}`  
  Full catalog containing scalar, gradient, divergence, and curl quantity
  metadata, including symbolic names, slot locations, and normalization flags.

# Returns
- `InterpolationCatalogConcise{D, N, G, Div, C}`  
  A compact catalog holding only:
  - `scalar_slots  :: NTuple{N,Int}`
  - `scalar_snormalization :: NTuple{N,Bool}`
  - `grad_slots    :: NTuple{G,Int}`
  - `div_slots     :: NTuple{Div,NTuple{D,Int}}`
  - `curl_slots    :: NTuple{C,NTuple{D,Int}}`

This reduction eliminates name handling overhead and is intended for
performance-critical inner interpolation kernels.
"""
@inline function to_concise_catalog(catalog::InterpolationCatalog{D, N, G, Div, C, L}) where {D, N, G, Div, C, L}
    return InterpolationCatalogConcise{D, N, G, Div, C}(
        catalog.scalar_slots,
        catalog.scalar_snormalization,
        catalog.grad_slots,
        catalog.div_slots,
        catalog.curl_slots,
    )
end
