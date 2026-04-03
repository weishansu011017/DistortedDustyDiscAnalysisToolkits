using PhantomRevealer
using LaTeXStrings
using Makie
using CairoMakie
using Statistics

"""
Tracing N-spirals from a Face-on PhantomRevealer dusty-disc result 
    by Wei-Shan Su,
    June 17, 2025
"""

struct pitch_result
    time_array :: Vector{Float64}
    ϕ1s_end :: Vector{Float64}
    βg_array :: Matrix{Float64}
    βd_array :: Matrix{Float64}
end

function spirals_tracer(file :: String; index :: Int64, result :: pitch_result)
    # ------------------------------BRIEF INTRODUCTION------------------------------
    # The spiral detection follow the following process
    #
    #                    2D-density map
    #                        │
    #                        ▼
    #                [Ridge detection & automatic scale-selection]
    #                        ⇒ Lindeberg 1996, 1998
    #                        │  (scale-space γ-norm ridge + σ̂ selection)
    #                        │
    #                        ▼
    #                Detected ridge points  +  ridge strength  +  ridge width
    #                        │
    #                        ▼
    #                [Log-polar Hough transform for logarithmic spirals]
    #                        ⇒ Duda & Hart 1972   (Hough framework)
    #                        │
    #                        ▼
    #                Accumulator in (ln a, k) space
    #                        │
    #                        ▼
    #                [Local peak selection - Non-Maximum Suppression]
    #                        ⇒ Canny 1986   (NMS on gray-scale image)
    #                        │
    #                        ▼
    #                Potential spiral peaks  (aᵢ , kᵢ)
    #                        │
    #                        ▼
    #                [Gain-penalty Beam Search]
    #                        ⇒ Lowerre 1976 / Graves 2012  (beam-search strategy)
    #                        ⇒ Su, et al (in prep.)        (gain & penalty objective)
    #                        │
    #                        ▼
    #                Final best-fit spiral arm set
    #
    # Detail process is refered to: 
    #    [Ridge detection]: Lindeberg(1996)
    #    [Hough Transform]: Duda & Hart(1962)
    #    [Local Peak Selection]: Canny (1986)
    #    [Beam Search]: Lowerre(1976) & Graves(2012)
    #    [Spiral detection pipline]: Su, et al.(in prep.)
    #
    # ------------------------------PARAMETER SETTING------------------------------
    # General parameters for spiral detection
    Nmax :: Int64   = 2                                         # Maximum spiral in a single image
    slim :: Tuple{Float64, Float64} = (45.0,110.0)              # Range of spiral detection IN CODE UNITS               

    # Ridge Detection parameters
    width_pixel_range :: Tuple{Float64, Float64} = (4.0, 22.0)  # The range of width of ridge IN PIXEL.
    width_resolution :: Int64 = 36                              # The resolution of t-scaling for scale selection.              
    boxfactor :: Float64 = 8.0                                  # Size factor for the convolution kernel in ridge detection. Affects smoothing scale and edge sensitivity.

    # Hough Transform parameters
    a_range :: Tuple{Float64, Float64} = (30.0, 300.0)          # Range of spiral base radius `a` (in pixels) used in log-spiral model: r(θ) = a * exp(kθ).
    k_range :: Tuple{Float64, Float64} = (-0.25, -0.06)         # Range of spiral winding parameter `k` in the log-spiral model (controls tightness of winding).
    num_a_bins :: Int64 = 450                                   # Number of discretized bins for `a` in the Hough accumulator.
    num_k_bins :: Int64 = 80                                    # Number of discretized bins for `k` in the Hough accumulator.

    # Gain-penalty Beam Search parameters
    beam_ratio :: Float64 = 0.2                                 # Beam width ratio; limits the number of top candidate arms retained at each step to `ceil(ratio * n_peaks)`.
    score_gain_thr :: Float64 = 0.003                           # Relative score gain below which the search stops early. 
    λ_angle :: Float64 = 1.0                                    # Weight of angle-spread penalty.
    λ_overlap :: Float64 = 1.25                                 # Weight of inter-arm overlap penalty.
    # -----------------------------------------------------------------------------
    index_g = 2
    index_d = 3
    @info "-------------------------------------------------------"
    # Read data
    data = Read_HDF5(file, true)
    if data.params["Analysis_type"] != "Faceon_disk"
        error("InputError: The Analysis type of data needs to be `Faceon_disk`!")
    end
    transfer_cgs!(data)

    # Detected Gas spiral
    spiralg :: Vector{NamedTuple} = spirals_detection(data, index_g, result.ϕ1s_end[1],
                                                    slim = slim,
                                                    # Ridge Detection
                                                    width_pixel_range = width_pixel_range,
                                                    width_resolution = width_resolution,
                                                    boxfactor = boxfactor,
                                                    # Hough Transform
                                                    a_range = a_range,
                                                    k_range = k_range,
                                                    num_a_bins = num_a_bins,
                                                    num_k_bins = num_k_bins,
                                                    # Beam search
                                                    Nmax = Nmax,
                                                    beam_ratio = beam_ratio,
                                                    score_gain_thr = score_gain_thr,
                                                    λ_angle = λ_angle,
                                                    λ_overlap = λ_overlap)
    # Detected Dust spiral
    spirald :: Vector{NamedTuple} = spirals_detection(data, index_d, result.ϕ1s_end[2],
                                                    slim = slim,
                                                    # Ridge Detection
                                                    width_pixel_range = width_pixel_range,
                                                    width_resolution = width_resolution,
                                                    boxfactor = boxfactor,
                                                    # Hough Transform
                                                    a_range = a_range,
                                                    k_range = k_range,
                                                    num_a_bins = num_a_bins,
                                                    num_k_bins = num_k_bins,
                                                    # Beam search
                                                    Nmax = Nmax,
                                                    beam_ratio = beam_ratio,
                                                    score_gain_thr = score_gain_thr,
                                                    λ_angle = λ_angle,
                                                    λ_overlap = λ_overlap)

    # Transfering k to β (pitch angle)
    result.time_array[index] = data.time
    if length(spiralg) != 0
        result.βg_array[index, 1] = k2pitch(spiralg[1].k)
        if length(spiralg) > 1
            result.βg_array[index, 2] = k2pitch(spiralg[2].k)
            result.ϕ1s_end[1] = spiralg[1].ϕ_end
        else
            if angular_distance(result.ϕ1s_end[1], spiralg[1].ϕ_end) >= π/2
                result.βg_array[index, 2] = k2pitch(spiralg[1].k)
                result.βg_array[index, 1] = NaN
            else
                result.βg_array[index, 2] = NaN
                result.ϕ1s_end[1] = spiralg[1].ϕ_end
            end
    
        end
    else 
        result.βg_array[index, 1] = NaN
        result.βg_array[index, 2] = NaN
    end

    if length(spirald) != 0
        result.βd_array[index, 1] = k2pitch(spirald[1].k)
        if length(spirald) > 1
            result.βd_array[index, 2] = k2pitch(spirald[2].k)
            result.ϕ1s_end[2] = spirald[1].ϕ_end
        else
            if angular_distance(result.ϕ1s_end[2], spirald[1].ϕ_end) >= π/2
                result.βd_array[index, 2] = k2pitch(spirald[1].k)
                result.βd_array[index, 1] = NaN
            else
                result.βd_array[index, 2] = NaN
                result.ϕ1s_end[2] = spirald[1].ϕ_end
            end
    
        end
    else 
        result.βg_array[index, 1] = NaN
        result.βg_array[index, 2] = NaN
    end


    @info "-------------------------------------------------------"
end

function draw_pitch2t_diagram(result :: pitch_result)
    time_array = result.time_array
    βg_array = result.βg_array
    βd_array = result.βd_array
    @info "Draw result..."
    # Modify array
    βg_array_pos :: Matrix{Float64} = abs.(βg_array)
    βd_array_pos :: Matrix{Float64} = abs.(βd_array)
    time_array ./= 1e4

    # Initialized Grid
    colors = (:blue, :red)
    linestyles = (:solid, :dash)

    Fax = FigureAxes(1, 1, figsize=(8,8), sharex = true, sharey = true)
    set_xlabel!(Fax, L"t[$10^4$yr]")
    set_ylabel!(Fax, L"$\beta [^{\circ}]$")

    # Draw result
    lines!(Fax.axes[1,1], time_array, βg_array_pos[:,1], color=colors[1], linestyle=linestyles[1])
    lines!(Fax.axes[1,1], time_array, βg_array_pos[:,2], color=colors[1], linestyle=linestyles[2])
    lines!(Fax.axes[1,1], time_array, βd_array_pos[:,1], color=colors[2], linestyle=linestyles[1])
    lines!(Fax.axes[1,1], time_array, βd_array_pos[:,2], color=colors[2], linestyle=linestyles[2])

    set_xlim!(Fax, (1,1), (minimum(time_array), maximum(time_array)))
    set_ylim!(Fax, (1,1), (3.0, 13.0))
    save_Fig!(Fax, "Spiral_PitchAngle.pdf")
end

function main()
    ϕ1g_end :: Float64 = 3π/2                            # The approximate azimuthal position of gas spiral with label `1` on smax at the first snapshot (i.e. ϕ1(smax, t = 0))
    ϕ1d_end :: Float64 = 3π/2                            # The approximate azimuthal position of dust spiral with label `1` on smax at the first snapshot (i.e. ϕ1(smax, t = 0))

    # Analysis_file
    files = ARGS

    # Arrays
    result :: pitch_result = pitch_result(zeros(Float64, length(files)), Float64[ϕ1g_end, ϕ1d_end], zeros(Float64, length(files), 2), zeros(Float64, length(files), 2))

    for (i,file) in enumerate(files)
        @info "File: $file"
        spirals_tracer(file, index = i, result = result)
    end
    draw_pitch2t_diagram(result)
    
end

main()