"""
    The tool kits for analysis/visualizing the dumpfile from PhantomRevealer analysis
        by Wei-Shan Su,
        July 18, 2024
"""
initialization_modules()
const TRANSFER_DICT = Dict{String, LaTeXString}(
    "∇" => "\\nabla",
    "ϕ" => L"\\phi",
    "θ" => L"\\theta",
    "ρ" => L"\\rho",
    "Σ" => L"\\Sigma",
)
"""
    replace_trans_LaTeXStr(str::String)
Replace all the weird charter to the LaTeX format in `TRANSFER_DICT`, and change the string into latexstring.

# Parameters
- `str :: String`: The string that would be transform.

# Returns
- `LaTeXString`: The transformation results.
"""
function replace_trans_LaTeXStr(str::String)
    transfer_dict = TRANSFER_DICT
    for target in keys(transfer_dict)
        result = transfer_dict[target]
        str = LaTeXString(replace(str, target=>result))
    end
    return str
end

"""
    Faceon_polar_plot!(Disk2Ddata :: Analysis_result, array_index :: Int64;
        Fax::Union{FigureAxes, Nothing} = nothing, figsize :: Tuple = (8,6),
        colormap::String="plasma", cbar_log::Bool=false, 
        vlim :: Union{Nothing,Vector} = nothing, time_unit::String = "yr")

Draw a face-on polar plot from the Faceon disk data.

# Parameters
- `Disk2Ddata :: Analysis_result`: The analysis result from PhantomRevealer.
- `array_index :: Int64`: The column index of the data array.

# Keyword Arguments
- `Fax :: Union{FigureAxes, Nothing} = nothing`: The existing figure object. If `nothing`, a new figure will be created.
- `figsize :: Tuple = (8,6)`: The figure size.
- `colormap :: String = "plasma"`: The colormap used for visualization.
- `cbar_log :: Bool = false`: If `true`, the colorbar will be in logarithmic scale.
- `vlim :: Union{Nothing, Vector} = nothing`: The colorbar range. If `nothing`, it will be determined automatically.
- `slim :: Union{Nothing, Tuple} = nothing`: The range of radial direction.
- `time_unit :: String = "yr"`: The unit of time displayed in the plot.

# Returns
- `FigureAxes`: The figure object containing the polar plot.

# Example
```julia
data = Read_HDF5("PRdumpfile.h5")
fax = Faceon_polar_plot!(data, 2) 
# To plot on the same canvas, use the keyword argument:
fax = Faceon_polar_plot!(data, 3, Fax=fax)
```
"""
function Faceon_polar_plot!(Disk2Ddata :: Analysis_result, array_index :: Int64;
    Fax::Union{FigureAxes, Nothing} =nothing, figsize :: Tuple = (8,6),colormap::String="plasma",
    cbar_log::Bool=false, vlim :: Union{Nothing,Vector} = nothing, slim::Union{Nothing,Tuple}=nothing,
    time_unit::String = "yr")
    if Disk2Ddata.params["Analysis_type"] != "Faceon_disk"
        error("InputError: The Analysis type of data needs to be `Faceon_disk`!")
    end
    if isnothing(slim)
        srange = eachindex(Disk2Ddata.axes[1])
    else
        srange = value2closestvalueindex(Disk2Ddata.axes[1],slim[1]):value2closestvalueindex(Disk2Ddata.axes[1],slim[2])
    end
    s = Disk2Ddata.axes[1][srange]
    ϕ = Disk2Ddata.axes[2]
    z = Disk2Ddata.data_dict[array_index][srange,:]
    z_unit = ""
    if haskey(Disk2Ddata.params,"column_units")
        if haskey(Disk2Ddata.params["column_units"],array_index)
            z_unit = Disk2Ddata.params["column_units"][array_index]
        else
            z_unit = Disk2Ddata.column_names[array_index]
        end
    else
        z_unit = Disk2Ddata.column_names[array_index]
    end
    @info "Unit: $z_unit"
    time = Disk2Ddata.time
    label_left = latexstring(L"$t = ",Int64(round(time)), L"$", time_unit)
    label_right = Disk2Ddata.column_names[array_index]

    if isnothing(Fax)
        Fax = FigureAxes(1,1,figsize=figsize,polar_axis=fill(true,1,1))
    end
    if isnothing(vlim)
        entervlim = Get_vminmax(z)
    else
        entervlim = vlim
    end
    scale::Union{Function,ReversibleScale}=identity
    if cbar_log
        if entervlim[1] <= 0.0
            scale=Symlog10Scale(entervlim...)
        else
            scale=log10
        end
    end
    
    lazypcolor!(Fax,(1,1),s,ϕ,z,colormap=colormap, colorrange=entervlim, colorscale=scale)
    set_colorbar!(Fax,(1,1),clabel=z_unit)
    set_annotation!(Fax,(1,1),label_left,halign=:left, valign= :top)
    set_annotation!(Fax,(1,1),label_right,halign=:right, valign= :top)
    return Fax
end

"""
    Faceon_plot(data :: Analysis_result, array_index :: Int64,Log_flag::Bool=false,minzero::Bool=false, vlim :: Union{Nothing,Vector} = nothing,colormap::String="plasma", figzise :: Tuple = (8,6);fax::Union{PyObject, Nothing} =nothing, xunit::Union{Nothing,String,LaTeXString} = nothing,yunit::Union{Nothing,String,LaTeXString} = nothing,zlabel::Union{Nothing,String,LaTeXString} = nothing, time_unit::String = "yr")
Draw the face-on plot from the data.

#Parameters
- `data :: Analysis_result`: The analysis result from PhantomRevealer
- `array_index :: Int64`: The column index of array.
- `Log_flag :: Bool = true`: Change the colorbar to log scale
- `vlim :: Union{Nothing,Vector} = nothing`: The colorbar range
- `colormap :: String = "plasma"`: The colormap.
- `figzise :: Tuple = (10,6)` The size of figure.

# Keyword argument
- `fax :: Union{PyObject, Nothing} = nothing`: The existing object of canvas. If `nothing` will generate a new object.
- `xunit :: Union{Nothing,String, LaTeXString} = nothing`: The unit of x-axis.
- `yunit :: Union{Nothing,String, LaTeXString} = nothing`: The unit of y-axis.
- `zlabel :: Union{Nothing,String, LaTeXString} = nothing`: The label of colorbar
- `time_unit :: String = "yr"`: The unit of time.

# Returns
- `PyCall.PyObject <pyplot_backend.polar_plot>`: The object of plotting.

# Example
```julia
data = Read_HDF5("PRdumpfile.h5")
fax = Faceon_plot(data, 2) 
# In order to plot it on the same canvas, use the keyword argument
fax = Faceon_plot(fax=fax, data, 3)
````
"""
function Faceon_plot!(data :: Analysis_result, array_index :: Int64;
    Fax::Union{FigureAxes, Nothing} = nothing, figsize :: Tuple = (8,6), colormap::String="plasma",
    cbar_log::Bool=false, vlim :: Union{Nothing,Vector} = nothing,
    z_plane::Union{Nothing,Float64} = nothing,
    xunit::Union{Nothing,String,LaTeXString} = nothing, 
    yunit::Union{Nothing,String,LaTeXString} = nothing,
    zlabel::Union{Nothing,String,LaTeXString} = nothing, time_unit::String = "yr")

    if length(data.axes) !== 2
        if isnothing(z_plane)
            error("InputError: The Analysis type of data needs to be in 2D grid!")
        else
            z_index = value2closestvalueindex(data.axes[1],z_plane)
            z = data.data_dict[array_index][:,:,z_index]
        end
    else
        z = data.data_dict[array_index]
    end

    x = data.axes[1]
    xlabel = isnothing(xunit) ? latexstring(L"$x$") : latexstring(L"$x$ [", xunit, "]")
    y = data.axes[2]
    ylabel = isnothing(yunit) ? latexstring(L"$y$") : latexstring(L"$y$ [", yunit, "]")
    

    # 設定色條標籤
    if isnothing(zlabel)
        if haskey(data.params, "column_units") && haskey(data.params["column_units"], array_index)
            z_unit = data.params["column_units"][array_index]
        else
            z_unit = data.column_names[array_index]
        end
    else
        z_unit = zlabel
    end

    time = data.time
    label_left = latexstring(L"$t = ", Int64(round(time)), L"$", time_unit)

    activate_backend("GL")

    if isnothing(Fax)
        Fax = FigureAxes(1,1, figsize=figsize)
    end

    if isnothing(vlim)
        entervlim = Get_vminmax(z)
    else
        entervlim = vlim
    end

    scale::Union{Function,ReversibleScale} = identity
    if cbar_log
        if entervlim[1] <= 0.0
            scale = Symlog10Scale(entervlim...)
        else
            scale = log10
        end
    end

    lazypcolor!(Fax, (1,1), x, y, z, colormap=colormap, colorrange=entervlim, colorscale=scale)

    set_colorbar!(Fax, (1,1), clabel=z_unit)
    set_annotation!(Fax, (1,1), label_left, halign=:left, valign=:top)
    set_xlabel!(Fax,xlabel)
    set_ylabel!(Fax,ylabel)

    draw_Fig!(Fax)

    return Fax
end

"""
    Check_array_quantity(data :: Analysis_result, array_index :: Int64)
Print out the statistical properties of array with given array index.

# Parameters
- `data :: Analysis_result`: The analysis result from PhantomRevealer.
- `array_index :: Int64`: The column index of array.
"""
function Check_array_quantities(data :: Analysis_result, array_index :: Int64)
    if haskey(data.params, "_cgs")
        In_cgs = data.params["_cgs"]
    else
        In_cgs = false
    end
    nanmax(x) = maximum(filter(!isnan,x))
    nanmin(x) = minimum(filter(!isnan,x))
    nanmean(x) = mean(filter(!isnan,x))
    nanmedian(x) = median(filter(!isnan,x))
    nanstd(x) = std(filter(!isnan,x))
    array = data.data_dict[array_index]
    column_name = data.column_names[array_index]
    shape = size(array)
    max = nanmax(array)
    min = nanmin(array)
    average = nanmean(array)
    med = nanmedian(array)
    STD = nanstd(array)
    nanratio = round((count(isnan, array) / length(array))*100,sigdigits=2)
    println("--------------Properties of array $column_name--------------")
    println("Size: $shape")
    println("In cgs unit?: $In_cgs")
    println("Ratio of NaN: $nanratio %")
    println("Maximum: $max")
    println("Minimum: $min")
    println("Average: $average")
    println("Median: $med")
    println("STD: $STD")
    println("----------------------------------------------------------------")
end
"""
    spirals_detection(Disk2Ddata::Analysis_result, array_index::Int64, ϕend_spiral1 = 0.0;
                      Fax::Union{FigureAxes,Nothing}=nothing,
                      slim::Union{Nothing,Tuple{Float64,Float64}}=(50.0,100.0),
                      Faxacc::Union{FigureAxes,Nothing}=nothing,
                      width_pixel_range :: Tuple{Float64, Float64} = (8.0,12.0),
                      width_resolution :: Int64         = 24
                      boxfactor::Float64                = 8.0,
                      a_range::Tuple{Float64,Float64}   = (30.0,300.0),
                      k_range::Tuple{Float64,Float64}   = (-0.5,-0.06),
                      num_a_bins::Int                   = 800,
                      num_k_bins::Int                   = 200,
                      Nmax::Int                         = 2,
                      beam_ratio::Float64               = 0.2,
                      score_gain_thr::Float64           = 0.003,
                      λ_angle::Float64                  = 1.0,
                      λ_overlap::Float64                = 1.0)

Detects one-armed or multi-armed logarithmic spirals in a **face-on** disc snapshot,  
using the pipeline *ridge detection → Hough transform → beam search clustering*.  

The spiral detection follow the following process

                       2D-density map
                           │
                           ▼
                   [Ridge detection & automatic scale-selection]
                           ⇒ Lindeberg 1996, 1998
                           │  (scale-space γ-norm ridge + σ̂ selection)
                           │
                           ▼
                   Detected ridge points  +  ridge strength  +  ridge width
                           │
                           ▼
                   [Log-polar Hough transform for logarithmic spirals]
                           ⇒ Duda & Hart 1972   (Hough framework)
                           │
                           ▼
                   Accumulator in (ln a, k) space
                           │
                           ▼
                   [Local peak selection - Non-Maximum Suppression]
                           ⇒ Canny 1986   (NMS on gray-scale image)
                           │
                           ▼
                   Potential spiral peaks  (aᵢ , kᵢ)
                           │
                           ▼
                   [Coverage-penalty Beam Search]
                           ⇒ Lowerre 1976 / Graves 2012  (beam-search strategy)
                           ⇒ Su, et al (in prep.)         (gain & penalty objective)
                           │
                           ▼
                   Final best-fit spiral arm set

Optional plotting hooks allow the routine to update / reuse Makie figures in real-time.

# Positional Arguments
- `Disk2Ddata::Analysis_result` — A PhantomRevealer face-on data object (`Analysis_type = "Faceon_disk"`).
- `array_index::Int64`          — Column index of the physical quantity to analyse (e.g. Σ_d).
- `ϕend_spiral1` = 0.0          — Reference azimuth (rad) used to order the detected spirals.

# Keyword Arguments
### Global
| kw | default | meaning |
|---|---|---|
| `Fax`        | `nothing` | FigureAxes with `(nrow, ncol) = (1, 1)` (The return `Fax` from `Faceon_polar_plot!` is recommended) for the *polar map* (points + fitted spirals). If `nothing`, nothing is drawn. |
| `Faxacc`     | `nothing` | FigureAxes with `(nrow, ncol) = (1, 2)` that hosts *(1)* smoothed accumulator with peak markers, *(2)* strength histogram. (a Figure)|
| `slim`       | `(50,100)`| Radial range (same unit as `Disk2Ddata.axes[1]`) to search for ridges. `nothing` = full range. |
### Ridge detection
| kw | default | meaning |
|---|---|---|
| `width_pixel_range` | `(8.0, 12.0)`  | The range of width of ridge IN PIXEL.|
| `width_resolution`  | `24`  | The resolution of t-scaling for scale selection.. |
| `boxfactor`         | `8.0`  | Spatial box factor passed to ridge detection. |

### Hough transform
| kw | default | meaning |
|---|---|---|
| `a_range`    | `(30,300)` | Search range for the logarithmic-spiral scale length *a* (same unit as *s*). |
| `k_range`    | `(-0.5,-0.06)` | Search range for pitch parameter *k*. |
| `num_a_bins` | `800`  | Radial bins in Hough space. |
| `num_k_bins` | `200`  | Pitch-angle bins in Hough space. |

### Beam-search clustering
| kw | default | meaning |
|---|---|---|
| `Nmax`          | `2`     | Max number of spirals to return. |
| `beam_ratio`    | `0.1`   | Beam width = `beam_ratio*length(peaks)` (hard-capped internally). |
| `score_gain_thr`| `0.003` | Relative score gain below which the search stops early. |
| `λ_angle`       | `1.0`   | Weight of angle-spread penalty.|
| `λ_overlap`     | `1.0`   | Weight of inter-arm overlap penalty.|

# Returns
`Vector{NamedTuple}` — ordered list of detected spirals.  
Each tuple contains  
```julia
(a   = best_a,      # scale length
 k   = best_k,      # pitch parameter
 ϕ_end = phi_end,   # azimuth at s_max
 pointsset = Set{Tuple{Float64,Float64}}  # ridge pixels classified to this arm
)
```
"""
function spirals_detection(Disk2Ddata :: Analysis_result, array_index :: Int64, ϕend_spiral1 = 0.0;
    Fax::Union{FigureAxes, Nothing} = nothing, slim = (50.0,100.0), 
    width_pixel_range :: Tuple{Float64, Float64} = (8.0,12.0), width_resolution :: Int64 = 24,                                                                                                                                                            # Range of spiral detection
    Faxacc::Union{FigureAxes, Nothing} = nothing, boxfactor ::Float64 = 12.0, a_range::Tuple{Float64,Float64} = (30.0, 300.0), k_range::Tuple{Float64,Float64} = (-0.5, -0.06), num_a_bins::Int = 800, num_k_bins::Int = 200,                                                    # Parameters for Hough transform
    Nmax:: Int64 = 2, beam_ratio :: Float64 = 0.2, score_gain_thr :: Float64 = 0.003, λ_angle :: Float64 = 1.0, λ_overlap :: Float64 = 1.0  # Parameters of Beam search
)
    if Disk2Ddata.params["Analysis_type"] != "Faceon_disk"
        error("InputError: The Analysis type of data needs to be `Faceon_disk`!")
    end
    if isnothing(slim)
        srange = eachindex(Disk2Ddata.axes[1])
    else
        srange = value2closestvalueindex(Disk2Ddata.axes[1],slim[1]):value2closestvalueindex(Disk2Ddata.axes[1],slim[2])
    end
    if !isnothing(Fax)
        # Clear spiral
        for p in Fax.axes[1,1].scene.plots
            if occursin("spiral_detection_s", p.label[])
                p.converted[1][] = Point{2,Float64}.([], [])
            end
        end
    end

    s = Disk2Ddata.axes[1][srange]
    ϕ = Disk2Ddata.axes[2]
    z = Disk2Ddata.data_dict[array_index]
    
    # Assign mutithreading
    FFTW.set_num_threads(nthreads())

    # Process: Ridge detection -> Hough transform -> Beam search
    ## Ridge detection
    pointsset_binary_full, strength_array_full, best_t_full = ridge_detection_automatic_scale_selection(z,  padax1_mode=0.0 , padax2_mode=:circular,  width_pixel_range = width_pixel_range, width_resolution = width_resolution, boxfactor = boxfactor)
    pointsset_binary = pointsset_binary_full[srange,:]          # points set binary 
    strength_array = strength_array_full[srange,:]              # strength_array
    best_t = best_t_full[srange,:]

    if (sum(pointsset_binary) == 0)
        if !isnothing(Fax)
            # Clear points
            for p in Faxacc.axes[1,1].scene.plots
                if p.label[] == "PeakPoints"
                    peakpoint_plot = p
                    peakpoint_plot.converted[1][] = Point{2,Float64}.([],[])
                    break
                end
            end
            for p in Faxacc.axes[1,1].scene.plots
                if p.label[] == "BestPeak"
                    bestpeak_plot = p
                    bestpeak_plot.converted[1][] = Point{2,Float64}.([],[])
                    break
                end
            end
        end
        spirals = NamedTuple[]
        @info "End spirals detection. No points has detected in given region!"
        return spirals
    end

    weighted_array = log10.(strength_array)
    clamp!(weighted_array, 0.0, maximum(weighted_array))
    for (i, value) in enumerate(weighted_array)
        if value <= 0.0
            pointsset_binary[i] = false
        end
    end

    # Generate meshgrid
    S, Φ = meshgrid(s, ϕ)
    detected_s = S[pointsset_binary]
    detected_ϕ = Φ[pointsset_binary]
    detected_color = weighted_array[pointsset_binary]
    if !isnothing(Fax)
        @info "Start plotting ridge points (colored by log strength)"
        # Plot points
        points_sc = nothing
        for p in Fax.axes[1,1].scene.plots
            if p.label[] == "spiral_detection_points"
                points_sc = p
            end
        end
        if isnothing(points_sc)
            points_sc = scatter!(Fax.axes[1,1],detected_ϕ,detected_s ; markersize = 5, color=detected_color,colormap=:amp, colorrange=(0.0,maximum(detected_color)))
            points_sc.label = "spiral_detection_points"
        else
            points_sc.converted[1][] = Point{2,Float64}.(detected_ϕ, detected_s)
            points_sc.color[] = detected_color
        end
    end


    ## Hough transform
    accumulator, a_array, k_array = Hough_transform_logarithmic_spiral(pointsset_binary, (s, ϕ), weighted_array, a_range=a_range, k_range=k_range, num_a_bins=num_a_bins, num_k_bins=num_k_bins, npattern = Nmax)
    
    if (iszero(accumulator))
        spirals = NamedTuple[]
        @info "End spirals detection. No peaks has detected in given region!"
        return spirals
    end
    log_accumulator :: Matrix{Float64} = @. log10(accumulator + 1e-9)  
    log_accumulator_non0 = log_accumulator[log_accumulator .> 0.0]

    if !isnothing(Faxacc)
        @info "Start plotting Accumelator"
        # First panel: Accumelator
        lazypcolor!(Faxacc,(1,1),collect(log.(a_array)),collect(k_array),log_accumulator ,colormap=:binary, colorrange=(0.0, maximum(log_accumulator)))
        set_xlim!(Faxacc, (1,1), (minimum(log.(a_array)), maximum(log.(a_array))))
        set_ylim!(Faxacc, (1,1), (minimum(k_array), maximum(k_array)))
        Faxacc.axes[1,1].xlabel[] = L"$\ln a$"
        Faxacc.axes[1,1].ylabel[] = L"$k$"
        set_colorbar!(Faxacc,(1,1), clabel=L"$\log \sum_{\mathrm{bin}} \mathcal{N}_{\gamma\text{-norm}} L$")
        # Second panel: histogram of weighted_array
        vec_weighted = detected_color[detected_color .> 0.0]
        mean_weighted = mean(vec_weighted)
        median_weighted = median(vec_weighted)
        std_weighted = std(vec_weighted)
        binbox = LinRange(0.0, maximum(vec_weighted), 40)  # 40 centers
        binindex_array = AssignBinIndices(vec_weighted, binbox, mode = :nearest)

        valid_bins = binindex_array
        valid_weights = vec_weighted

        nbins = length(binbox)
        weighted_hist = zeros(Float64, nbins)

        for (bin, w) in zip(valid_bins, valid_weights)
            weighted_hist[bin] += 1
        end

        StrengthBar = nothing
        for p in Faxacc.axes[1,2].scene.plots
            if p.label[] == "StrengthBar"
                StrengthBar = p
            end
        end
        if isnothing(StrengthBar)
            barplot = barplot!(Faxacc.axes[1,2], binbox, weighted_hist)

            barplot.label = "StrengthBar"
        else
            StrengthBar.converted[1][] = Point{2,Float64}.(binbox, weighted_hist)
        end

        meanindicator = nothing
        for p in Faxacc.axes[1,2].scene.plots
            if p.label[] == "meanindicator"
                meanindicator = p
            end
        end
        if isnothing(meanindicator)
            vli = vlines!(Faxacc.axes[1,2], [mean_weighted - std_weighted, mean_weighted, mean_weighted + std_weighted])
            vli.color = :red
            vli.label = "meanindicator"
        else
            meanindicator.converted[1][] = [mean_weighted - std_weighted, mean_weighted, mean_weighted + std_weighted]
            meanindicator.color[] = :red
        end

        medianindicator = nothing
        for p in Faxacc.axes[1,2].scene.plots
            if p.label[] == "medianindicator"
                medianindicator = p
            end
        end
        if isnothing(medianindicator)
            vm = vlines!(Faxacc.axes[1,2], [median_weighted])
            vm.color = :green
            vm.label = "medianindicator"
        else
            medianindicator.converted[1][] =  [median_weighted]
            medianindicator.color[] = :green
        end

        Faxacc.axes[1,2].xlabel[] = L"$\log \mathcal{N}_{\gamma\text{-norm}} L$"
 
    end

    if isempty(log_accumulator_non0)
        if !isnothing(Fax)
            # Clear spiral
            for p in Fax.axes[1,1].scene.plots
                if occursin("spiral_detection_s", p.label[])
                    p.converted[1][] = Point{2,Float64}.([], [])
                end
            end
        end
        spirals = NamedTuple[]
        @info "End spirals detection. No reliable Hough peaks exist in accumulator!"
        return spirals
    end
    ## Beam search 
    peaks :: Vector{PeakCandidate} = pickup_accumelator_peaks(log_accumulator, s[end], a_array, k_array; r = 2, threshold = 0.0) 
        
    best_combination :: SpiralState = Beam_search_logarithmic_spiral(peaks, a_array, k_array, S, Φ, weighted_array, best_t; 
                                                                    Nmax = Nmax,
                                                                    beam_ratio = beam_ratio,
                                                                    score_gain_thr = score_gain_thr,
                                                                    λ_angle = λ_angle,
                                                                    λ_overlap = λ_overlap)

    reordered_best_combination = reorder_spirals(best_combination, ϕend_spiral1)
    spirals = NamedTuple[]
    for peak in reordered_best_combination.peaks
        apk = a_array[peak.a_idx]
        kpk = k_array[peak.k_idx]
        subpointsset_binary = get_subpointsset(apk, kpk, pointsset_binary, S, Φ, best_t)
        subpointsset = bitarray2pointsset(subpointsset_binary, (s, ϕ))

        spiral = (; :a => apk, :k => kpk, :pointsset => subpointsset, :ϕ_end => peak.ϕ_end)
        push!(spirals, spiral)
    end
    # Plot accumulator
    if !isnothing(Faxacc)
        # Draw peak points
        peak_lna = zeros(Float64, length(peaks)) 
        peak_k = zeros(Float64, length(peaks)) 
        for i in eachindex(peaks)
            peak_lna[i] = log(a_array[peaks[i].a_idx])
            peak_k[i] = k_array[peaks[i].k_idx]
        end
        peakpoint_plot = nothing 
        for p in Faxacc.axes[1,1].scene.plots
            if p.label[] == "PeakPoints"
                peakpoint_plot = p
                break
            end
        end
        if isnothing(peakpoint_plot)
            peakpoint_plot = scatter!(Faxacc.axes[1,1],peak_lna,peak_k , markersize = 3, color=:orange)
            peakpoint_plot.label = "PeakPoints"
        else
            peakpoint_plot.converted[1][] = Point{2,Float64}.(peak_lna,peak_k)
        end

        bestpeak_plot = nothing 
        for p in Faxacc.axes[1,1].scene.plots
            if p.label[] == "BestPeak"
                bestpeak_plot = p
                break
            end
        end
        if isnothing(bestpeak_plot)
            bestpeak_plot = scatter!(Faxacc.axes[1,1],[log(a_array[pk.a_idx]) for pk in best_combination.peaks],[k_array[pk.k_idx] for pk in best_combination.peaks] , markersize = 5, color=:cyan)
            bestpeak_plot.label = "BestPeak"
        else
            bestpeak_plot.converted[1][] = Point{2,Float64}.([log(a_array[pk.a_idx]) for pk in best_combination.peaks],[k_array[pk.k_idx] for pk in best_combination.peaks])
        end
    end
    # Plot points
    if !isnothing(Fax)
        if length(spirals) > 0
            # Plot spiral
            @info "Start plotting spirals"
            colormap = [:blue, :red, :green, :orange, :purple, :cyan, :magenta]
            color_of(i) = colormap[mod1(i, length(colormap))]

            ϕs = [_logarithmic_spiral_ϕ.(s, a = spiral.a, k = spiral.k) for spiral in spirals]
            for i in eachindex(ϕs)
                spiral_plot = nothing
                for p in Fax.axes[1,1].scene.plots
                    if p.label[] == "spiral_detection_s$i"
                        spiral_plot = p
                        break
                    end
                end
                if isnothing(spiral_plot)
                    spiral_plot = lines!(Fax.axes[1,1], ϕs[i],s , color=color_of(i))
                    spiral_plot.label = "spiral_detection_s$i"
                else
                    spiral_plot.converted[1][] = Point{2,Float64}.(ϕs[i], s)
                end
            end
            # Plot points
            points_sc = nothing
            for p in Fax.axes[1,1].scene.plots
                if p.label[] == "spiral_detection_points"
                    points_sc = p
                end
            end
            if isnothing(points_sc)
                points_sc = scatter!(Fax.axes[1,1],detected_ϕ,detected_s ; markersize = 5, color=detected_color,colormap=:amp, colorrange=(0.0,maximum(detected_color)))
                points_sc.label = "spiral_detection_points"
            else
                points_sc.converted[1][] = Point{2,Float64}.(detected_ϕ, detected_s)
                points_sc.color[] = detected_color
            end
        end
    end
    # Turn Back FFTW threads for safty
    FFTW.set_num_threads(1)
    return spirals
end


"""
    spirals_detection(Full_image :: A, s_array :: V, ϕ_array :: V, ϕend_spiral1 = 0.0;
                      Fax::Union{FigureAxes,Nothing}=nothing,
                      slim::Union{Nothing,Tuple{Float64,Float64}}=(50.0,100.0),
                      Faxacc::Union{FigureAxes,Nothing}=nothing,
                      width_pixel_range :: Tuple{Float64, Float64} = (8.0,12.0),
                      width_resolution :: Int64         = 24
                      boxfactor::Float64                = 8.0,
                      a_range::Tuple{Float64,Float64}   = (30.0,300.0),
                      k_range::Tuple{Float64,Float64}   = (-0.5,-0.06),
                      num_a_bins::Int                   = 800,
                      num_k_bins::Int                   = 200,
                      Nmax::Int                         = 2,
                      beam_ratio::Float64               = 0.2,
                      score_gain_thr::Float64           = 0.003,
                      λ_angle::Float64                  = 1.0,
                      λ_overlap::Float64                = 1.0) where {A <: AbstractArray, V <: AbstractVector}

Detects one-armed or multi-armed logarithmic spirals in a **face-on** disc snapshot,  
using the pipeline *ridge detection → Hough transform → beam search clustering*.  

The spiral detection follow the following process

                       2D-density map
                           │
                           ▼
                   [Ridge detection & automatic scale-selection]
                           ⇒ Lindeberg 1996, 1998
                           │  (scale-space γ-norm ridge + σ̂ selection)
                           │
                           ▼
                   Detected ridge points  +  ridge strength  +  ridge width
                           │
                           ▼
                   [Log-polar Hough transform for logarithmic spirals]
                           ⇒ Duda & Hart 1972   (Hough framework)
                           │
                           ▼
                   Accumulator in (ln a, k) space
                           │
                           ▼
                   [Local peak selection - Non-Maximum Suppression]
                           ⇒ Canny 1986   (NMS on gray-scale image)
                           │
                           ▼
                   Potential spiral peaks  (aᵢ , kᵢ)
                           │
                           ▼
                   [Coverage-penalty Beam Search]
                           ⇒ Lowerre 1976 / Graves 2012  (beam-search strategy)
                           ⇒ Su, et al (in prep.)         (gain & penalty objective)
                           │
                           ▼
                   Final best-fit spiral arm set

Optional plotting hooks allow the routine to update / reuse Makie figures in real-time.

# Positional Arguments
- `Full_image::A`               — 2D image of a physical quantity (e.g. Σ_d), shape = (s, ϕ).
- `s_array::V`                  — Radial grid coordinates corresponding to the first dimension of `Full_image`.
- `ϕ_array::V`                  — Azimuthal grid coordinates (in radians) corresponding to the second dimension of `Full_image`.
- `ϕend_spiral1` = 0.0          — Reference azimuth (rad) used to order the detected spirals.

# Keyword Arguments
### Global
| kw | default | meaning |
|---|---|---|
| `slim`       | `(50,100)`| Radial range (same unit as `Disk2Ddata.axes[1]`) to search for ridges. `nothing` = full range. |
### Ridge detection
| kw | default | meaning |
|---|---|---|
| `width_pixel_range` | `(8.0, 12.0)`  | The range of width of ridge IN PIXEL.|
| `width_resolution`  | `24`  | The resolution of t-scaling for scale selection.. |
| `boxfactor`         | `8.0`  | Spatial box factor passed to ridge detection. |

### Hough transform
| kw | default | meaning |
|---|---|---|
| `a_range`    | `(30,300)` | Search range for the logarithmic-spiral scale length *a* (same unit as *s*). |
| `k_range`    | `(-0.5,-0.06)` | Search range for pitch parameter *k*. |
| `num_a_bins` | `800`  | Radial bins in Hough space. |
| `num_k_bins` | `200`  | Pitch-angle bins in Hough space. |

### Beam-search clustering
| kw | default | meaning |
|---|---|---|
| `Nmax`          | `2`     | Max number of spirals to return. |
| `beam_ratio`    | `0.1`   | Beam width = `beam_ratio*length(peaks)` (hard-capped internally). |
| `score_gain_thr`| `0.003` | Relative score gain below which the search stops early. |
| `λ_angle`       | `1.0`   | Weight of angle-spread penalty.|
| `λ_overlap`     | `1.0`   | Weight of inter-arm overlap penalty.|

# Returns
`Vector{NamedTuple}` — ordered list of detected spirals.  
Each tuple contains  
```julia
(a   = best_a,      # scale length
 k   = best_k,      # pitch parameter
 ϕ_end = phi_end,   # azimuth at s_max
 pointsset = Set{Tuple{Float64,Float64}}  # ridge pixels classified to this arm
)
```
"""
function spirals_detection(Full_image :: A, s_array :: V, ϕ_array :: V, ϕend_spiral1 = 0.0;
    slim = (50.0,100.0), 
    width_pixel_range :: Tuple{Float64, Float64} = (8.0,12.0), width_resolution :: Int64 = 24,                                                                                                                                                            # Range of spiral detection
    boxfactor ::Float64 = 12.0, a_range::Tuple{Float64,Float64} = (30.0, 300.0), k_range::Tuple{Float64,Float64} = (-0.5, -0.06), num_a_bins::Int = 800, num_k_bins::Int = 200,                                                    # Parameters for Hough transform
    Nmax:: Int64 = 2, beam_ratio :: Float64 = 0.2, score_gain_thr :: Float64 = 0.003, λ_angle :: Float64 = 1.0, λ_overlap :: Float64 = 1.0  # Parameters of Beam search
) where {A <: AbstractArray, V <: AbstractVector}
    if isnothing(slim)
        srange = s_array[1]:s_array[end]
    else
        srange = value2closestvalueindex(s_array,slim[1]):value2closestvalueindex(s_array,slim[2])
    end

    s = s_array[srange]
    ϕ = ϕ_array
    z = Full_image
    
    # Assign mutithreading
    FFTW.set_num_threads(nthreads())

    # Process: Ridge detection -> Hough transform -> Beam search
    ## Ridge detection
    pointsset_binary_full, strength_array_full, best_t_full = ridge_detection_automatic_scale_selection(z,  padax1_mode=0.0 , padax2_mode=:circular,  width_pixel_range = width_pixel_range, width_resolution = width_resolution, boxfactor = boxfactor)
    pointsset_binary = pointsset_binary_full[srange,:]          # points set binary 
    strength_array = strength_array_full[srange,:]              # strength_array
    best_t = best_t_full[srange,:]

    if (sum(pointsset_binary) == 0)
        spirals = NamedTuple[]
        @info "End spirals detection. No points has detected in given region!"
        return spirals
    end

    weighted_array = log10.(strength_array)
    clamp!(weighted_array, 0.0, maximum(weighted_array))
    for (i, value) in enumerate(weighted_array)
        if value <= 0.0
            pointsset_binary[i] = false
        end
    end

    # Generate meshgrid
    S, Φ = meshgrid(s, ϕ)


    ## Hough transform
    accumulator, a_array, k_array = Hough_transform_logarithmic_spiral(pointsset_binary, (s, ϕ), weighted_array, a_range=a_range, k_range=k_range, num_a_bins=num_a_bins, num_k_bins=num_k_bins, npattern = Nmax)
    
    if (iszero(accumulator))
        spirals = NamedTuple[]
        @info "End spirals detection. No peaks has detected in given region!"
        return spirals
    end
    log_accumulator :: Matrix{Float64} = @. log10(accumulator + 1e-9)  

    ## Beam search 
    peaks :: Vector{PeakCandidate} = pickup_accumelator_peaks(log_accumulator, s[end], a_array, k_array; r = 2, threshold = 0.0) 
        
    best_combination :: SpiralState = Beam_search_logarithmic_spiral(peaks, a_array, k_array, S, Φ, weighted_array, best_t; 
                                                                    Nmax = Nmax,
                                                                    beam_ratio = beam_ratio,
                                                                    score_gain_thr = score_gain_thr,
                                                                    λ_angle = λ_angle,
                                                                    λ_overlap = λ_overlap)

    reordered_best_combination = reorder_spirals(best_combination, ϕend_spiral1)
    spirals = NamedTuple[]
    for peak in reordered_best_combination.peaks
        apk = a_array[peak.a_idx]
        kpk = k_array[peak.k_idx]
        subpointsset_binary = get_subpointsset(apk, kpk, pointsset_binary, S, Φ, best_t)
        subpointsset = bitarray2pointsset(subpointsset_binary, (s, ϕ))

        spiral = (; :a => apk, :k => kpk, :pointsset => subpointsset, :ϕ_end => peak.ϕ_end)
        push!(spirals, spiral)
    end

    # Turn Back FFTW threads for safty
    FFTW.set_num_threads(1)
    return spirals
end