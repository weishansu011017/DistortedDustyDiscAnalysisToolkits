"""
Classical Hough Transform for detecting straight lines.
    by Wei-Shan Su,
    May 11, 2025
"""


# Logarithmic spiral as a function of s
@inline function _logarithmic_spiral_ϕ(s :: Real; a, k) :: Float64
    return mod((1/k)*(log(s/a)), 2π)
end

"""
    Hough_transform_logarithmic_spiral(
        pointset_binary::AbstractMatrix{Bool},
        axes::Tuple{AbstractVector{Float64},AbstractVector{Float64}},
        weight_array::Union{Nothing,AbstractMatrix{Float64}}=nothing;
        a_range::Tuple{Float64,Float64} = (30.0,300.0),
        k_range::Tuple{Float64,Float64} = (-0.5,-0.06),
        num_a_bins::Int               = 800,
        num_k_bins::Int               = 200
    ) -> (accumulator, a_array, k_array)

Compute the Hough‐space accumulator for logarithmic spirals defined by  
    ln s = ln a + kϕ  
from a binary ridge map, optionally with per-pixel weights.

# Parameters
- `pointset_binary::AbstractMatrix{Bool}`  
  A `(ns × nϕ)` Boolean mask marking ridge-detected pixels.
- `axes::Tuple{s_vals,ϕ_vals}`  
  Two vectors of length `ns` and `nϕ` giving the radial (`s_vals`) and angular (`ϕ_vals`) axes.
- `weight_array::Union{Nothing,AbstractMatrix{Float64}}`  
  Optional weights for each pixel (same shape as `pointset_binary`).  
  If `nothing`, uniform weight 1.0 is applied.

# Keyword Arguments
| kw           | default           | meaning                                                       |
|--------------|-------------------|---------------------------------------------------------------|
| `a_range`    | `(30.0,300.0)`    | Search range for spiral scale length *a* (same units as *s*). |
| `k_range`    | `(-0.5,-0.06)`    | Search range for pitch parameter *k*.                         |
| `num_a_bins` | `800`             | Number of bins along the ln *a* axis.                         |
| `num_k_bins` | `200`             | Number of bins along the *k* axis.                            |

# Returns
- `accumulator::Matrix{Float64}`  
  A `(num_a_bins × num_k_bins)` matrix of vote sums.
- `a_array::Vector{Float64}`  
  The *a* values (exp of ln *a* axis) at each radial bin.
- `k_array::Vector{Float64}`  
  The *k* values at each angular bin.
"""
function Hough_transform_logarithmic_spiral(pointset_binary :: AbstractMatrix{Bool}, axes::Tuple{AbstractArray{Float64},AbstractArray{Float64}}, weight_array::Union{Nothing, AbstractMatrix{Float64}}=nothing;
    a_range::Tuple{Float64,Float64} = (30.0, 300.0), k_range::Tuple{Float64,Float64} = (-0.5, -0.06), num_a_bins::Int = 800, num_k_bins::Int = 200, npattern :: Int64 = 2)
    # Assume binary image has the size (ns, nϕ)
    if size(pointset_binary) != (length(axes[1]),length(axes[2]))
        error("DimentionMismatch: The binary array must have the same size as (ns, nϕ), which requires ($(length(axes[1])),$(length(axes[2]))), but get $(size(pointset_binary))")
    end

    # Setup voting weighting
    if isnothing(weight_array)
        weight_array = ones(Float64, size(pointset_binary)...)
    else
        if size(pointset_binary) != size(weight_array)
            error("DimentionMismatch: The array of weight_array and pointset_binary should be in the same length, but got $(size(weight_array)) and $(size(pointset_binary))")
        end
    end

    @info "Start Hough Transform."
    @info "Find $(count(pointset_binary)) points for Hough transform."
    # Setup s phi
    S, Φ = meshgrid(axes...)
    s_array = S[pointset_binary]
    ϕ_array = Φ[pointset_binary]
    masked_weight_array = weight_array[pointset_binary]

    lns_array = log.(s_array)

    # Setup accumulator & axis
    accumulator = zeros(Float64, num_a_bins, num_k_bins)
    k_array = LinRange(k_range..., num_k_bins)
    lna_range = (log(a_range[1]),log(a_range[2]))
    lna_array = LinRange(lna_range..., num_a_bins)

    if (count(pointset_binary) <= 4 * npattern)
        a_array = exp.(lna_array)
        @info "End Hough Transform. No spiral detected due to not enough points has given!"
        return accumulator, collect(a_array), collect(k_array)
    end

    # Prepare multiple local bins
    nt         = nthreads()                        
    local_acc  = [zeros(Float64, num_a_bins, num_k_bins) for _ in 1:nt ]

    # Voting
    @threads for i in eachindex(lns_array)
        tid     = threadid()
        lns     = @inbounds lns_array[i]
        ϕ       = @inbounds ϕ_array[i]
        weight  = @inbounds masked_weight_array[i]
        @inbounds for k_idx in eachindex(k_array)
            k = k_array[k_idx]
            lna = lns - k * ϕ
            if lna >= lna_range[1] && lna <= lna_range[2]
                lna_idx = Int(floor((lna - lna_range[1]) / (lna_range[2] - lna_range[1]) * (num_a_bins - 1))) + 1
                local_acc[tid][lna_idx, k_idx] += weight
            end
        end
    end
    # Reduction
    for acc_t in local_acc
        @inbounds accumulator .+= acc_t
    end
    a_array = exp.(lna_array)

    @info "End Hough Transform."
    return accumulator, collect(a_array), collect(k_array)
end


