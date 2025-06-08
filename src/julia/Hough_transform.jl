"""
Classical Hough Transform for detecting straight lines..
    by Wei-Shan Su,
    May 11, 2025
"""

"""
    struct PatternResult
The struct for storeing specific pattern. 

# Fields
- `pointsset_binary :: AbstractArray{Bool}`: A Bitarray for locating the pattern.
- `axes :: Tuple`: The axes of `pointsset_binary`.
- `params :: Dict{String, Real}`: The 

"""
struct PatternResult
    pointsset_binary :: AbstractArray{Bool}
    axes             :: Tuple
    params           :: Dict{String, Real}
end


# Logarithmic spiral as a function of s
@inline function _logarithmic_spiral_ϕ(s :: Real; a, k) :: Float64
    return mod((1/k)*(log(s/a)), 2π)
end

"""
    function Hough_transform_fitting_single_arm_spiral(s_array::Vector{Float64}, ϕ_array::Vector{Float64}; 
        a_range::Tuple{Float64,Float64} = (1.0, 300.0), 
        k_range::Tuple{Float64,Float64} = (-0.5, -0.06), 
        num_a_bins::Int = 2000, 
        num_k_bins::Int = 200)

Fits a logarithmic spiral of the form:

    ϕ(s) = (1/k) * log(s/a)

using the Hough transform.

# Parameters
- `s_array`: Array of radial distances.
- `ϕ_array`: Array of azimuthal angles.

# Keyword arguments
- `a_range`: Fitting range of `a`.
- `k_range`: Fitting range of `k`.
- `num_a_bins`: Resolution in `a`.
- `num_k_bins`: Resolution in `k`.

# Returns
- `best_a`: The fitted `a` parameter.
- `best_k`: The fitted `k` parameter.
- `accumulator`: 2D accumulator matrix used in voting.
"""
function Hough_transform_fitting_logarithmic_spiral(s_array::Vector{Float64}, ϕ_array::Vector{Float64}, weight_array::Union{Nothing, Vector{Float64}}=nothing; a_range::Tuple{Float64,Float64} = (1.0, 300.0), k_range::Tuple{Float64,Float64} = (-0.5, -0.06), num_a_bins::Int = 2000, num_k_bins::Int = 200)
    # Check length 
    if length(s_array) != length(ϕ_array)
        error("DimentionMismatch: The array of s_array and ϕ_array should be in the same length, but got $(length(s_array)) and $(length(ϕ_array))")
    end
    
    # Setup voting weighting
    if isnothing(weight_array)
        weight_array = ones(Float64, length(s_array))
    else
        if length(weight_array) != length(ϕ_array)
            error("DimentionMismatch: The array of weight_array and ϕ_array should be in the same length, but got $(length(weight_array)) and $(length(ϕ_array))")
        end
    end
    
    # Take ln(s)
    lns_array = log.(s_array)

    # Setup accumulator & axis
    accumulator = zeros(Float64, num_a_bins, num_k_bins)
    k_array = LinRange(k_range..., num_k_bins)
    lna_range = (log(a_range[1]),log(a_range[2]))
    lna_array = LinRange(lna_range..., num_a_bins)
    
    # Voting
    for i in eachindex(lns_array)
        lns = lns_array[i]
        ϕ = ϕ_array[i]
        weight = weight_array[i]

        for k_idx in 1:num_k_bins
            k = k_array[k_idx]

            lna = lns - k * ϕ

            if lna >= lna_range[1] && lna <= lna_range[2]
                lna_idx = Int(floor((lna - lna_range[1]) / (lna_range[2] - lna_range[1]) * (num_a_bins - 1))) + 1
                accumulator[lna_idx, k_idx] += weight
            end
        end
    end
    return  accumulator, lna_array, k_array
end

function Hough_transform_fitting_logarithmic_spiral(pointset_binary :: AbstractMatrix{Bool}, axes::Tuple{AbstractArray{Float64},AbstractArray{Float64}}, weight_array::Union{Nothing, AbstractMatrix{Float64}}=nothing;p = 1.0, a_range::Tuple{Float64,Float64} = (30.0, 300.0), k_range::Tuple{Float64,Float64} = (-0.5, -0.06), num_a_bins::Int = 800, num_k_bins::Int = 200)
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
        weight_array .^= p
    end
    
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

    # Voting
    for i in eachindex(lns_array)
        lns = lns_array[i]
        ϕ = ϕ_array[i]
        weight = masked_weight_array[i]
        for k_idx in 1:num_k_bins
            k = k_array[k_idx]
    
            lna = lns - k * ϕ
    
            if lna >= lna_range[1] && lna <= lna_range[2]
                lna_idx = Int(floor((lna - lna_range[1]) / (lna_range[2] - lna_range[1]) * (num_a_bins - 1))) + 1
                accumulator[lna_idx, k_idx] += weight
            end
        end
    end
    return accumulator, k_array, lna_array
end

function Hough_transform_fitting_twolines(
    pointset_binary :: AbstractMatrix{Bool},
    axes::Tuple{AbstractArray{Float64},AbstractArray{Float64}},
    weight_array::Union{Nothing, AbstractMatrix{Float64}}=nothing; 
    a_range::Tuple{Float64,Float64} = (30.0, 300.0), k_range::Tuple{Float64,Float64} = (-0.5, -0.06), num_a_bins::Int = 800, num_k_bins::Int = 200, ϕend_spiral1=0.0,
    width = 5.0
)
    function _find_best_line(accumulator, k_array, lna_array)
        max_idx = argmax(accumulator)
        lna_idx = max_idx[1]
        k_idx = max_idx[2]
        best_k = k_array[k_idx]
        best_lna = lna_array[lna_idx]
        best_a = exp(best_lna)
        return best_a, best_k
    end

    function _generate_spiral_mask(best_a, best_k; width=1.0)
        s_array, phi_array = axes
        S,Φ = meshgrid(s_array, phi_array)
        spiral_phi = _logarithmic_spiral_ϕ.(S, a = best_a, k = best_k)
        Δϕ = @. _angular_distance(spiral_phi, Φ)
        Δϕ_max = @. width / (S * sqrt(1 + best_k^2))
        return @. Δϕ < Δϕ_max
    end

    function _filter_points_near_line(pointset_binary :: AbstractMatrix{Bool}; best_a, best_k, width=5.0)
        spiralbinary = _generate_spiral_mask(best_a, best_k, width=width)
        filtered_mask = pointset_binary .& .!spiralbinary
        return filtered_mask
    end
    normalized_weight = copy(weight_array)
    normalized_weight[weight_array .< (mean.(weight_array) .- 2 .* std(weight_array))] .= 0.0
    normalized_weight[weight_array .≥ (mean.(weight_array) .- 2 .* std(weight_array))] .= 1.0
    HoughTransform(binary) = Hough_transform_fitting_logarithmic_spiral(binary, axes, normalized_weight, a_range=a_range, k_range=k_range, num_a_bins=num_a_bins, num_k_bins=num_k_bins)

    # First fit SPIRAL 1
    accumulator, k_array, lna_array = HoughTransform(pointset_binary)
    best_a1, best_k1 = _find_best_line(accumulator, k_array, lna_array)

    # First fit SPIRAL 2 after filter S1  i.e  HT(U\S1)
    filtered_binary = _filter_points_near_line(pointset_binary, best_a=best_a1, best_k=best_k1,width=width)
    accumulator, _, _ = HoughTransform(filtered_binary)
    best_a2, best_k2 = _find_best_line(accumulator, k_array, lna_array)

    # Second fit SPIRAL 1 after filter S2 ∪ O   
    Noise_binary =  _filter_points_near_line(pointset_binary, best_a=best_a2, best_k=best_k2,width=width)                                # A = U\S2 = S1 ∪ O
    Noise_binary = _filter_points_near_line(Noise_binary, best_a=best_a1, best_k=best_k1,width=width)                                    # O = A\S1
    filtered_binary = pointset_binary .& .!Noise_binary                                                                                  # E = U\O = S1 ∪ S2
    filtered_binary = _filter_points_near_line(filtered_binary, best_a=best_a2, best_k=best_k2,width=width)                              # F = E\S2
    accumulator, _, _ = HoughTransform(filtered_binary)
    best_a1_refit, best_k1_refit = _find_best_line(accumulator, k_array, lna_array)

    # Second fit SPIRAL 2 after filter S1 ∪ O  
    Noise_binary =  _filter_points_near_line(pointset_binary, best_a=best_a1_refit, best_k=best_k1_refit,width=width)                    # A = U\S1 = S2 ∪ O
    Noise_binary = _filter_points_near_line(Noise_binary, best_a=best_a2, best_k=best_k2,width=width)                                    # O = A\S2
    filtered_binary = pointset_binary .& .!Noise_binary                                                                                  # E = U\O = S1 ∪ S2
    filtered_binary = _filter_points_near_line(filtered_binary, best_a=best_a1_refit, best_k=best_k1_refit,width=width)                  # F = E\S1
    accumulator, _, _ = HoughTransform(filtered_binary)
    best_a2_refit, best_k2_refit = _find_best_line(accumulator, k_array, lna_array)

    # Get spirals 
    Noise_binary =  _filter_points_near_line(pointset_binary, best_a=best_a1_refit, best_k=best_k1_refit,width=width)                    # A = U\S1 = S2 ∪ O
    Noise_binary = _filter_points_near_line(Noise_binary, best_a=best_a2_refit, best_k=best_k2_refit,width=width)                        # O = A\S2
    filtered_binary = pointset_binary .& .!Noise_binary   
    spiral1_binary =  _filter_points_near_line(filtered_binary, best_a=best_a2_refit, best_k=best_k2_refit,width=width)
    spiral2_binary =  _filter_points_near_line(filtered_binary, best_a=best_a1_refit, best_k=best_k1_refit,width=width)

    send = axes[1][end]
    ϕend1 = _logarithmic_spiral_ϕ(send, a = best_a1_refit, k = best_k1_refit)
    ϕend2 = _logarithmic_spiral_ϕ(send, a = best_a2_refit, k = best_k2_refit)
    Δϕ1 = _angular_distance(ϕend1, ϕend_spiral1)
    Δϕ2 = _angular_distance(ϕend2, ϕend_spiral1)
    if Δϕ1 > Δϕ2
        best_a1_refit, best_a2_refit = best_a2_refit, best_a1_refit
        best_k1_refit, best_k2_refit = best_k2_refit, best_k1_refit
        spiral1_binary, spiral2_binary = spiral2_binary, spiral1_binary
    end

    spiral1_params = Dict{String, Real}()
    spiral1_params["a"] = best_a1_refit
    spiral1_params["k"] = best_k1_refit
    spiral1_params["label"] = 1
    spiral1 = PatternResult(spiral1_binary, axes, spiral1_params)
    spiral2_params = Dict{String, Real}()
    spiral2_params["a"] = best_a2_refit
    spiral2_params["k"] = best_k2_refit
    spiral2_params["label"] = 2
    spiral2 = PatternResult(spiral2_binary, axes, spiral2_params)

    return spiral1, spiral2
end
