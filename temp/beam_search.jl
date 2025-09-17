"""
Beam search for determine best modes for spiral detection
    by Wei-Shan Su,
    June 10, 2025
"""

"""
    struct PeakCandidate

A container for a single peak candidate from the Hough-transform accumulator,
representing one potential logarithmic-spiral arm.

# Fields
- `a_idx :: Int64`  
  Row index in the accumulator (corresponds to `a_array[a_idx]`).
- `k_idx :: Int64`  
  Column index in the accumulator (corresponds to `k_array[k_idx]`).
- `votes :: Float64`  
  Accumulator value (log‐Hough votes) at `(a_idx, k_idx)`.
- `ϕ_end :: Float64`  
  Azimuth (rad) at the image’s maximum radius `s_max` for this spiral arm.
"""
struct PeakCandidate
    a_idx :: Int64          # row
    k_idx :: Int64          # col
    votes :: Float64      # log accumulator value
    ϕ_end :: Float64      # phi value at s_max of image (Used for symmetric panalty)
end

"""
    struct SpiralState

Represents one state in the beam‐search combining multiple spiral arms,
tracking their combined coverage, vote sum, and composite score.

# Fields
- `peaks :: Vector{PeakCandidate}`  
  The selected peak candidates for each spiral arm.
- `votes :: Float64`  
  Sum of `votes` from each `PeakCandidate` in `peaks`.
- `strengths_sum :: Float64`  
  Total strength (sum of pixel weights) covered by the combined spiral mask.
- `mask :: BitMatrix`  
  Boolean mask of all image pixels covered by these spirals.
- `score :: Float64`  
  Composite score combining coverage gain, angle‐spread, and overlap penalties.
"""
struct SpiralState
    peaks           :: Vector{PeakCandidate}      # Spiral properties (N spirals per assumption) (PeakCandidate array)
    votes           :: Float64                    # total log accumulator value of each peaks
    strengths_sum   :: Float64                    # How much detected points has been covered (scale with strength of each points)
    mask            :: BitMatrix                  # spiral mask of current state
    score           :: Float64                    # score
end

@inline function _calc_beam_size(N::Int; beam_ratio::Float64 = 0.1, α::Float64 = 5.0)
    B_by_ratio = ceil(Int, beam_ratio * N)
    B_by_root  = 1 + ceil(Int, α * sqrt(N))
    return max(min(B_by_ratio, B_by_root), 1)   
end

@inline function _generate_spiral_mask(a::Float64, k::Float64, S::Matrix{Float64}, Φ::Matrix{Float64}, width :: Vector{Float64})
    if length(width) != size(S)[1]
        error("DimensionMismatch: The size of width $(length(width)) should be identical as the first axis of `S` and `Φ` $(size(S))")
    end
    abs(k) < eps() && return falses(size(S)) 
    spiral_phi = @. mod((1/k)*(log(S/a)), 2π)
    Δϕ = @. angular_distance(spiral_phi, Φ)
    Δϕ_max = zeros(Float64, size(S)...)
    @inbounds for i in eachindex(width)
        Δϕ_max[i,:] .= width[i] ./ (S[i,:] .* sqrt(1 + k^2))
    end
    return @. Δϕ < Δϕ_max
end

@inline function _gain(mask::BitMatrix, weight::AbstractMatrix)   
    n_add = count(mask)
    n_add == 0 && return 0.0

    points = weight .> 0.0
    cover_c   = count(points[mask])
    total_c   = count(points)
    cover_frac = cover_c / (total_c + eps())    
    
    cover_w   = sum(weight[mask])
    total_w   = sum(weight)
    weight_frac = cover_w / (total_w + eps())  

    gain = (0.75 * cover_frac^3 + 0.25 * weight_frac^3)
    return gain 
end

@inline function _angle_spread_penalty(phis::Vector{Float64}) :: Float64
    N = length(phis)
    N ≤ 1 && return 0.0
    ph = sort(mod.(phis, 2π))
    Δ  = [ph[2:end] .- ph[1:end-1]; ph[1] + 2π - ph[end]]
    ideal = 2π / N
    return sum(abs.(Δ .- ideal)) / (2 * N * ideal)      # 0 (perfect) -> ~1
end

@inline function _spiral_overlap_penalty(mask_new::BitMatrix, mask_old::BitMatrix) :: Float64
    inter = count(mask_new .& mask_old)
    inter == 0 && return 0.0
    return inter / min(count(mask_new), count(mask_old))
end

@inline function _score_combination(gain::Float64, P_angle::Float64, P_overlap::Float64; w_angle::Float64, w_overlap ::Float64) :: Float64
    return gain - w_angle * P_angle - w_overlap * P_overlap
end

"""
    pickup_accumelator_peaks(
        accumulator::Matrix{Float64},
        smax::Float64,
        a_array::Vector{Float64},
        k_array::Vector{Float64};
        r::Int              = 1,
        threshold::Float64  = -Inf
    ) -> Vector{PeakCandidate}

Extracts local maxima (“peaks”) from a 2D Hough-space accumulator and
computes the corresponding spiral parameters `(a, k)` and end-azimuths.

# Parameters
- `accumulator::Matrix{Float64}`  
  The Hough transform vote histogram (ln a × k grid).
- `smax::Float64`  
  Maximum radial coordinate of the image (used to compute φ_end).
- `a_array::Vector{Float64}`  
  The scale-length values for each row of `accumulator`.
- `k_array::Vector{Float64}`  
  The pitch-parameter values for each column of `accumulator`.

# Keyword Arguments
| kw        | default | meaning                                                       |
|-------------|---------|---------------------------------------------------------------|
| `r`         | `1`     | Neighborhood radius for non‐maximum suppression (in bins).    |
| `threshold` | `-Inf`  | Minimum accumulator value to consider as a peak.              |

# Returns
- `Vector{PeakCandidate}`  
  A list of detected peaks, sorted by descending `votes`.  
  Each `PeakCandidate` contains:
  ```julia
  struct PeakCandidate
      a_idx::Int       # row index in accumulator
      k_idx::Int       # column index in accumulator
      votes::Float64   # accumulator value at that bin
      ϕ_end::Float64   # azimuth at s = smax (rad)
  end
  ````
"""
function pickup_accumelator_peaks(accumulator :: Matrix{Float64}, smax :: Float64, a_array::Vector{Float64}, k_array::Vector{Float64}; r :: Int64 = 1, threshold :: Float64 =-Inf)
    @info "Start picking up peaks."
    peaks = PeakCandidate[]
    if iszero(accumulator)
        @info "End picking up peaks. No peaks are found since the accumulator is an empty array."
        return peaks
    end

    nb        = trues(2r+1, 2r+1)
    local_max = dilate(accumulator, nb) .== accumulator
    mask = local_max .& (accumulator .>= threshold)
    labels = label_components(mask)

    for lbl in 1:maximum(labels)
        inds = findall(labels .== lbl)
        if isempty(inds) continue end
        vals = accumulator[inds]
        maxval, rel = findmax(vals)
        idx = inds[rel]  # CartesianIndex

        ap = a_array[idx[1]]
        kp = k_array[idx[2]]

        abs(kp) < eps() && continue

        ϕ_end = mod((1/kp)*(log(smax/ap)), 2π)
        push!(peaks, PeakCandidate(idx[1], idx[2], maxval, ϕ_end))
    end

    sort!(peaks, by = p -> -p.votes)
    @info "End picking up peaks."
    return peaks
end

"""
    Beam_search_logarithmic_spiral(
        peaks::Vector{PeakCandidate},
        a_array::Vector{Float64},
        k_array::Vector{Float64},
        S::Matrix{Float64},
        Φ::Matrix{Float64},
        strength::Matrix{Float64}
        best_t::Matrix{Float64};
        Nmax::Int             = 2,
        beam_ratio::Float64   = 0.2,
        score_gain_thr::Float64 = 0.003,
        λ_angle::Float64      = 0.6,
        λ_overlap::Float64    = 0.8
    ) -> SpiralState

Performs a beam‐search over Hough‐transform peaks to identify up to `Nmax` 
logarithmic spirals that best cover the ridge points while balancing coverage,
angle‐spread, and overlap penalties.

# Parameters
- `peaks::Vector{PeakCandidate}`  
  List of candidate peaks sorted by vote strength.
- `a_array::Vector{Float64}`  
  Scale‐length values corresponding to the rows of the Hough accumulator.
- `k_array::Vector{Float64}`  
  Pitch‐parameter values corresponding to the columns of the Hough accumulator.
- `S::Matrix{Float64}`  
  Radial coordinates grid (same dims as the ridge map).
- `Φ::Matrix{Float64}`  
  Angular coordinates grid (same dims as the ridge map).
- `strength::Matrix{Float64}`  
  Per‐pixel weight/strength map used for scoring coverage.
- `best_t::Matrix{Float64}`  
  Per‐pixel best t map from Ridge detection.

# Keyword Arguments
| kw | default | meaning |
|---|---|---|
| `Nmax`          | `2`     | Max number of spirals to return. |
| `beam_ratio`    | `0.1`   | Beam width = `beam_ratio*length(peaks)` (hard-capped internally). |
| `score_gain_thr`| `0.003` | Relative score gain below which the search stops early. |
| `λ_angle`       | `1.0`   | Weight of angle-spread penalty. |
| `λ_overlap`     | `1.0`   | Weight of inter-arm overlap penalty. |
| `λ_noncovered`  | `1.0`   | Weight of non covered points penalty. |

# Returns
- `SpiralState`  
  A struct containing the best combination:
  - `peaks::Vector{PeakCandidate}`       — Selected peak candidates for each arm.
  - `votes::Float64`                     — Sum of Hough votes for the chosen peaks.
  - `weighted_cover::Float64`            — Total pixel strength covered by these spirals.
  - `mask::BitMatrix`                    — Boolean mask of all covered ridge pixels.
  - `score::Float64`                     — Final composite score of the solution.

"""
function Beam_search_logarithmic_spiral(
    peaks            ::   Vector{PeakCandidate},
    a_array          ::   Vector{Float64},
    k_array          ::   Vector{Float64}, 
    S                ::   Matrix{Float64},
    Φ                ::   Matrix{Float64},
    strength         ::   Matrix{Float64},
    best_t           ::   Matrix{Float64};
    Nmax             ::   Int64       = 2,
    beam_ratio       ::   Float64     = 0.1,
    score_gain_thr   ::   Float64     = 0.003,
    λ_angle          ::   Float64     = 1.0,
    λ_overlap        ::   Float64     = 1.0,
)
    @info "Start Beam search."
    # Data validation
    if isempty(peaks)
        @info "End Beam search. No spiral has been selected! (peaks is empty)"
        return SpiralState(PeakCandidate[], 0.0, 0.0, falses(size(S)), 0.0)
    end

    Nmax ≤ 0 && error("Nmax must be positive, got $Nmax")
    beam_ratio ≤ 0 && error("beam_ratio must be > 0, got $beam_ratio")

    size(S) == size(Φ) || error("DimensionMismatch: size(S)=$(size(S)) vs size(Φ)=$(size(Φ))")
    size(S) == size(strength) || error("DimensionMismatch: strength matrix size $(size(strength)) inconsistent with S")


    @info "Find $(length(peaks)) peaks for clustering."
    # Calculate Beam size
    beam_size = _calc_beam_size(length(peaks); beam_ratio=beam_ratio, α=5.0)
    
    # Make azimuthal-averaged width array
    best_σ = sqrt.(best_t)
    dr = mean(diff(S[:,1]))
    width = [ mean(view(best_σ, i, :)[view(best_σ, i, :) .> 0 ]) for i in 1:size(best_σ,1) ]
    width .*= dr
    mean_width = mean(filter(!isnan, width))   
    width[isnan.(width)] .= mean_width 

    # Calculate pkmask
    pksmask = [_generate_spiral_mask(a_array[pk.a_idx], k_array[pk.k_idx], S, Φ, width) for pk in peaks]

    # Search tree
    root = SpiralState(PeakCandidate[], 0.0, 0.0, falses(size(S)), 0.0)
    open_set = [root]
    best_prev = open_set[1].score          # score in the previous depth

    # Start Beam search
    cand = SpiralState[]
    for depth in 1:Nmax
        seen = Set{UInt64}()      
        for st in open_set
            # Iterate thorugh all peaks
            for i in eachindex(peaks)
                pk = peaks[i]

                peaks_combination = Set{PeakCandidate}(st.peaks)
                push!(peaks_combination, pk)
                hashkey = hash(peaks_combination)
                if (hashkey in seen)
                    continue
                else
                    push!(seen, hashkey)
                end
                peaks_new = copy(st.peaks)
                push!(peaks_new, pk)

                # Get spiral mask by combine current state and pk                  
                pkmask = pksmask[i]
                new_mask = st.mask .|  pkmask               # Ptot =  {P in st} ∪ Pnew

                # Test increasing of total strength (If not high enough, ignore this peak)
                gain = _gain(new_mask, strength)

                # spiral overlap panalty
                Poverlap = _spiral_overlap_penalty(pkmask, st.mask)

                # angle spread penalty
                ϕends = [currentpk.ϕ_end for currentpk in st.peaks]
                push!(ϕends, pk.ϕ_end)
                Pang = _angle_spread_penalty(ϕends)

                # score this combination
                score_new  = _score_combination(gain, Pang, Poverlap ;w_angle=λ_angle, w_overlap=λ_overlap)

                if score_new < 0.0
                    continue
                end

                # new strength
                strengths_sum_new = sum(@view strength[new_mask])

                # new votes
                votes_new = 0.0
                for p in peaks_new
                    votes_new += p.votes
                end
                
                push!(cand, SpiralState(peaks_new, votes_new, strengths_sum_new, new_mask, score_new))
            end
        end
        if isempty(cand)
            break
        end 
        sort!(cand, by = s -> -s.score)
        open_set = cand[1:min(beam_size, length(cand))]
        
        best_now = open_set[1].score
        if (best_prev == 0.0)
            best_prev = best_now
            continue
        else
            gain = (best_now - best_prev) / (abs(best_prev) + 1e-9)
            if (gain > score_gain_thr)
                break
            end
            if (best_now > 0.75)
                break
            end
            best_prev = best_now 
        end         
    end
    topN = 5
    msg = ["Top $(min(topN, length(open_set))) high-score states:";         
        ["$(node.score), depth=$(length(node.peaks))"
            for node in open_set[1:min(topN, length(open_set))]]]       

    @info join(msg, "\n") 
    
    best = open_set[1]
    @info "Choosed: score: $(best.score)"
    @info "End Beam search. Find $(length(best.peaks)) spiral$(length(best.peaks) == 1 ? "" : "s")"
    return best
end

"""
    get_subpointsset(
        a::Float64,
        k::Float64,
        pointsset_binary::Matrix{Bool},
        S::Matrix{Float64},
        Φ::Matrix{Float64};
        width::Float64 = 4.0
    )

Extract the points within a specific logarithmic spiral arm from a binary ridge mask.

This function intersects the ridge binary map (`pointsset_binary`) with a spiral mask defined by parameters `a` and `k`, over the polar coordinate mesh (`S`, `Φ`). The spiral is treated as a finite-width region defined by `width`.

# Parameters
- `a::Float64`: Spiral scale length.
- `k::Float64`: Spiral pitch parameter.
- `pointsset_binary::Matrix{Bool}`: Binary mask representing ridge-detected points.
- `S::Matrix{Float64}`: Radial mesh (from `meshgrid`).
- `Φ::Matrix{Float64}`: Azimuthal mesh (from `meshgrid`).
- `width::Vector{Float64}`: Width of spiral as the function of `s`. Note that the length of this parameter should be identical as the first axis of size(S)

# Returns
- `Matrix{Bool}`: A binary matrix indicating points that lie within the defined spiral.
"""
function get_subpointsset(a :: Float64, k :: Float64, pointsset_binary :: Matrix{Bool}, S :: Matrix{Float64}, Φ :: Matrix{Float64}, best_t::Matrix{Float64};)
    size(pointsset_binary) == size(S) == size(Φ) == size(best_t) ||
        error("DimensionMismatch: sizes are ",
    size(pointsset_binary), " vs ", size(S), " vs ", size(Φ))
    best_σ = sqrt.(best_t)
    dr = mean(diff(S[:,1]))
    width = [ mean(view(best_σ, i, :)[view(best_σ, i, :) .> 0 ]) for i in 1:size(best_σ,1) ]
    width .*= dr
    mean_width = mean(filter(!isnan, width))   
    width[isnan.(width)] .= mean_width 

    spiralbinary = _generate_spiral_mask(a, k, S, Φ, width)
    return pointsset_binary .& spiralbinary
end

"""
    reorder_spirals(peaks::Vector{PeakCandidate}, ϕ1_end::Float64)

Reorders the list of spiral peaks such that the first one has azimuth `ϕ_end` closest to `ϕ1_end`,
and the remaining spirals are sorted in counter-clockwise direction based on their `ϕ_end`.

# Parameters
- `peaks::Vector{PeakCandidate}`: A list of detected spiral peak candidates.
- `ϕ1_end::Float64`: Reference azimuth (rad) to determine the primary spiral.

# Returns
- `Vector{PeakCandidate}`: Reordered list of peaks in counter-clockwise azimuthal order starting from the one closest to `ϕ1_end`.
"""
function reorder_spirals(peaks::Vector{PeakCandidate}, ϕ1_end::Float64)
    N = length(peaks)
    N == 0 && return NamedTuple[]        # No spiral

    # Find the spiral that has ϕ_end which is the cloest one to ϕ1_end
    dists = [angular_distance(pk.ϕ_end, ϕ1_end) for pk in peaks]
    start = argmin(dists)                # spiral 1

    # ordered as counter-clockwise
    ϕ_start = mod(peaks[start].ϕ_end, 2π)
    sortidx = sortperm(1:N, by = i -> mod(peaks[i].ϕ_end - ϕ_start, 2π))

    return peaks[sortidx]
end

"""
    reorder_spirals(spiralstate::SpiralState, ϕ1_end::Float64)

Reorders the peaks inside a `SpiralState` object based on the reference azimuth `ϕ1_end`,  
keeping all other attributes (`votes`, `strengths_sum`, `mask`, `score`) unchanged.

# Parameters
- `spiralstate::SpiralState`: The spiral state containing an unordered list of peaks.
- `ϕ1_end::Float64`: The target reference azimuth to align the first peak.

# Returns
- `SpiralState`: A new spiral state with reordered peaks.
"""
function reorder_spirals(spiralstate::SpiralState, ϕ1_end::Float64)
    new_peaks = reorder_spirals(spiralstate.peaks, ϕ1_end)
    return SpiralState(new_peaks, spiralstate.votes, spiralstate.strengths_sum, spiralstate.mask, spiralstate.score)
end