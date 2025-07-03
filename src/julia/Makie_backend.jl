"""
Makie backend for PhantomRevealer plotting.
    by Wei-Shan Su,
    Febuary 12, 2025
"""

############# Default theme #############
const DEFAULT_THEME_SETUP :: Dict{String,Any} = Dict{String,Any}(
    "font_family" => "Times New Roman",
    "fontsize" => 18,
    "savefig_dpi" => 450,
    "savefig_format" => "eps"
)

"""
    rcParams_update(;font_family="Times New Roman", font_size=13, savefig_dpi=450,savefig_format="eps")
Update the font setting

# Keyword arguments
- `font_family`: The default font famaily.
- `font_size`: The default font size.
- `savefig_dpi`: The default DPI while saving figures
- `savefig_format`: The default format whilt saving figures
"""
function rcParams_update(;font_family="Times New Roman", font_size=13, savefig_dpi=450,savefig_format="eps")
    DEFAULT_THEME_SETUP["font_family"] = font_family
    DEFAULT_THEME_SETUP["font_size"] = font_size
    DEFAULT_THEME_SETUP["savefig_dpi"] = savefig_dpi
    DEFAULT_THEME_SETUP["savefig_format"] = savefig_format
end

function _constructing_fonts()
    return (; regular = DEFAULT_THEME_SETUP["font_family"])
end
############ Plotting object setup #############
"""
    mutable struct FigureAxes <: PhantomRevealerDataStructures

A data structure for storing the plotting figure and its associated axes.

This structure is designed to manage and store plotting-related components in a 
Makie.jl visualization environment. It provides a structured way to handle 
figures, axes, and additional metadata.

# Fields
- `fig` :: `Figure`
    - The main `Figure` object that contains the plots.
- `axes` :: `Matrix{Union{Nothing, Makie.Block}}`
    - A matrix representing the relative position of axis among all `Axis`-alike object. 
- `axes_type` :: `Matrix{String}`
    - A matrix that stores the type of each axis as a string. This can be used 
      to track different axis configurations (e.g., "Cartesian", "Polar", "3D").
- `screen` :: `Union{Nothing, Any}`
    - The `GLMakie.Screen` object associated with the figure, used when rendering 
      interactive visualizations. `Nothing` if no screen is assigned.
"""
mutable struct FigureAxes <: PhantomRevealerDataStructures
    fig :: Figure
    axes :: Matrix{Union{Nothing,Makie.Block}}
    axes_type :: Matrix{String}
    screen :: Union{Nothing, Any}
end

function _contructing_axes_from_axes_type(fig::Figure, axes_type::Matrix{String})
    nrows,ncols = size(axes_type)
    axes :: Matrix{Union{Nothing,Makie.Block}} = Matrix{Union{Nothing,Makie.Block}}(undef, nrows,ncols)
    for I in CartesianIndices(axes)
        axis_type = axes_type[I]
        ITuple = Tuple(I)
        if axis_type == "Cartesian"
            axes[I] = Axis(fig[ITuple...])
            # Clear all decoration
            axes[I].titlevisible = false
            hidedecorations!(axes[I])
            if ITuple[2] == 1
                axes[I].yticklabelsvisible = true
                axes[I].yticksvisible = true
                axes[I].ylabelvisible = true
            end
            if ITuple[1] == nrows
                axes[I].xticklabelsvisible = true
                axes[I].xticksvisible = true
                axes[I].xlabelvisible = true
            end

        elseif axis_type == "Polar"
            axes[I] = PolarAxis(fig[ITuple...],clip=false)
            axes[I].titlevisible = false
            hidedecorations!(axes[I])
        elseif axis_type == "3D"
            axes[I] = Axis3(fig[ITuple...])
            axes[I].titlevisible = false
        else
            error("InputError: Construction do not support the axis type `$axis_type`! ")
        end
        
    end
    return axes
end

function _contructing_axes_from_axes_type(Fax :: FigureAxes)
    return _contructing_axes_from_axes_type(Fax.fig,Fax.axes_type)
end

"""
    function FigureAxes(nrows::Int64,ncols::Int64;
        figsize::Tuple{Int64,Int64}=(8,6),
        sharex::Bool = true, sharey::Bool = true,
        polar_axis::Union{Nothing,Matrix{Bool}}=nothing,ThreeDim_axis::Union{Nothing,Matrix{Bool}}=nothing)

Create a `FigureAxes` object with a structured `Figure` containing different types of axes.

This function initializes a `Figure` with a grid of size `(nrows, ncols)`, where each cell in the grid contains either:
- A **Cartesian (2D) Axis** (default)
- A **Polar Axis** (if `polar_axis` is provided and `true` in certain positions)
- A **3D Axis** (if `ThreeDim_axis` is provided and `true` in certain positions)

It ensures that each grid cell is assigned to only one axis type.

# Parameters
- `nrows::Int`: The number of rows in the figure grid.
- `ncols::Int`: The number of columns in the figure grid.

# Keyword arguments
- `figsize::Tuple{Int, Int}=(8,6)`: The overall size of the figure in inches.
- `sharex::Bool`: If `true`, all x-axes in the `FigureAxes` object will be linked, ensuring that zooming or panning in one axis will update all other axes accordingly.
- `sharey::Bool`: If `true`, all y-axes in the `FigureAxes` object will be linked, ensuring that zooming or panning in one axis will update all other axes accordingly.
- `polar_axis::Union{Nothing,Matrix{Bool}}=nothing`: A `Bool` matrix of shape `(nrows, ncols)`, where `true` marks positions for **PolarAxes**.
- `ThreeDim_axis::Union{Nothing,Matrix{Bool}}=nothing`: A `Bool` matrix of shape `(nrows, ncols)`, where `true` marks positions for **3D Axes**.

# Returns
- A `FigureAxes` object containing:
  - `fig::Figure`: The `Makie.Figure` object.
  - `axes::Matrix{Union{Nothing,Block}}`: A matrix storing the created axes.
  - `axes_type::Matrix{String}`: A matrix indicating the type of each axis (`"Cartesian"`, `"Polar"`, or `"3D"`).

# Example Usage
```julia
nrows, ncols = 2, 3
polar_mask = [false true false; false false true]
ThreeDim_mask = [true false false; false true false]

fig_axes = FigureAxes(nrows, ncols; polar_axis=polar_mask, ThreeDim_axis=ThreeDim_mask)

fig_axes.fig  # Displays the figure
```
This creates a 2×3 grid where:
	•	The (1,2) and (2,3) positions have **PolarAxes**.
	•	The (1,1) and (2,2) positions have **3D Axes**.
	•	The remaining positions have **Cartesian Axes**.
"""
function FigureAxes(nrows::Int64,ncols::Int64;
    figsize::Tuple{Int64,Int64}=(8,6),
    sharex::Bool = true, sharey::Bool = true,
    polar_axis::Union{Nothing,Matrix{Bool}}=nothing,ThreeDim_axis::Union{Nothing,Matrix{Bool}}=nothing)

    fig :: Figure = Figure(size=Tuple([figsize...].*96),fontsize = DEFAULT_THEME_SETUP["fontsize"], fonts = _constructing_fonts())
    axes_type::Matrix{String} = fill("Cartesian",nrows,ncols)
    if !isnothing(polar_axis)
        if (nrows,ncols) != size(polar_axis)
            error(DimensionMismatch,": Size of the `polar_axis` $(size(polar_axis)) do not match with $((nrows,ncols))!")
        else
            for i in eachindex(polar_axis)
                if axes_type[i] != "Cartesian"
                    error("ValueError: The element $i in `axes_type` has already assigned with $(axes_type[i])!")
                else
                    axes_type[i] = polar_axis[i] ? "Polar" : "Cartesian"
                end
            end
        end
    elseif !isnothing(ThreeDim_axis)
        if (nrows,ncols) != size(ThreeDim_axis)
            error(DimensionMismatch,": Size of the `ThreeDim_axis` $(size(ThreeDim_axis)) do not match with $((nrows,ncols))!")
        else
            for i in eachindex(ThreeDim_axis)
                if axes_type[i] != "Cartesian"
                    error("ValueError: The element $i in `axes_type` has already assigned with $(axes_type[i])!")
                else
                    axes_type[i] = ThreeDim_axis[i] ? "3D" : "Cartesian"
                end
            end
        end
    end
    axes = _contructing_axes_from_axes_type(fig,axes_type)
    rowgap!(fig.layout, 0)  # 讓行之間沒有間距
    colgap!(fig.layout, 0)  # 讓列之間沒有間距
    if sharex && all(ax -> ax isa Axis, axes)
        linkxaxes!(axes...)
    end
    if sharey && all(ax -> ax isa Axis, axes)
        linkyaxes!(axes...)
    end
    return FigureAxes(fig, axes,axes_type,nothing)
end

"""
    current_backend()
Get the current activated backend

# Returns
- `String`: The name of backend.
"""
function current_backend()
    current = "$(Makie.current_backend())"
    if current == "GLMakie"
        return "GL"
    elseif current == "CairoMakie"
        return "Cairo"
    else
        return "UnknownBackend"
    end
end

"""
    activate_backend(backend::String)
Activate the specific backend for plotting.

# Parameters
- `backend :: String`: The backend. The avaliable backends are either `GL` or `Cairo`.
"""
function activate_backend(backend::String)
    if backend == "GL"
        ext = Base.get_extension(PhantomRevealer, :BackendExtra)
        if ext !== nothing
            ext.activate_backend()
        else
            error("MisloadingError: GLMakie is not installed or loaded. Please use `CairoMakie` instead!")
        end
    elseif backend == "Cairo"
        if isdefined(Main, :CairoMakie)
            CairoMakie.activate!()
        else
            error("MisloadingError: The module `CairoMakie` haven't loaded yet! Using `using CairoMakie` before activating backend!")
        end
    else
        error("ValueError: Unknown name of backend $backend !. The avaliable backends are either `GL` or `Cairo`.")
    end
end


"""
    draw_Fig!(Fax :: FigureAxes)
Draw the content that saves insinde the given `FigureAxes`.

# Parameters
- `Fax :: FigureAxes`: The `FigureAxes` object.
"""
function draw_Fig!(Fax :: FigureAxes)
    if current_backend() == "GL"
        if isnothing(Fax.screen)
            ext = Base.get_extension(PhantomRevealer, :BackendExtra)
            Fax.screen = ext.open_GLscreen()
        end
        display(Fax.screen, Fax.fig)
    end
end

function _DPI2PPU(dpi::Union{Float64,Int64})
    return dpi / 96
end

"""
    save_Fig!(Fax :: FigureAxes, filepath::String)
Save the content to the given file path

# Parameters
- `Fax :: FigureAxes`: The `FigureAxes` object.
- `filepath :: String`: The output filepath.
"""
function save_Fig!(Fax :: FigureAxes, filepath::String, dpi::Union{Nothing, Int64, Float64}=nothing)
    fig = Fax.fig
    ppu = _DPI2PPU(isnothing(dpi) ? DEFAULT_THEME_SETUP["savefig_dpi"] : dpi)
    extension = splitext(filepath)[end]
    cbackend = current_backend()
    if (extension == ".svg") || (extension == ".pdf")
        if !(cbackend == "Cairo")
            error("UnsuppotedFileExtension: The extension $(extension) is not supported in the backend $(cbackend)!")
        end
        save(filepath, fig)
    elseif (extension == ".png")
        if !((cbackend == "Cairo") || (cbackend == "GL"))
            error("UnsuppotedFileExtension: The extension $(extension) is not supported in the backend $(cbackend)!")
        end
        save(filepath, fig, px_per_unit=ppu)
    else
        error("UnsuppotedFileExtension: The extension $(extension) is not supported currently!")
    end
end

"""
    close_Fig!(Fax :: FigureAxes)
Close the `GLMakie.screen` window that correlates to the given `FigureAxes`.

# Parameters
- `Fax :: FigureAxes`: The `FigureAxes` object.
"""
function close_Fig!(Fax :: FigureAxes)
    if current_backend == "GL"
        ext = Base.get_extension(PhantomRevealer, :BackendExtra)
        if (!isnothing(Fax.screen)) || (ext !== nothing)
            ext.close_Fig!(Fax)
            Fax.screen = nothing
        end
    end
end

"""
    set_xlabel!(Fax :: FigureAxes, content::Union{AbstractString, Vector})
Set the label of the x-axis for each column in the `FigureAxes` object.

If a single string is provided, all columns will have the same x-axis label.  
If a vector of strings is provided, each column will receive a corresponding label.

# Parameters
- `Fax :: FigureAxes`: The `FigureAxes` object.
- `content :: Union{AbstractString, Vector}`: 
  - If a single string is provided, all x-axis labels will be the same.
  - If a vector is provided, each column gets a corresponding label. The vector **must** have the same length as the number of columns.
"""
function set_xlabel!(Fax :: FigureAxes, content::Union{AbstractString, Vector})
    axes = Fax.axes
    nrows, ncols = size(axes)
    if typeof(content) <: AbstractString
        for i in 1:ncols
            axes[nrows,i].xlabel = content
        end
    else
        if length(content) == ncols
            for i in 1:ncols
                axes[nrows,i].xlabel = content[i]
            end
        else
            error(DimensionMismatch,": The length of `content` $(length(content)) should be equal to number of columns $(ncols) if an array is provided!")
        end
    end
end

"""
    set_ylabel!(Fax :: FigureAxes, content::Union{AbstractString, Vector})
Set the label of the y-axis for each row in the `FigureAxes` object.

If a single string is provided, all rows will have the same x-axis label.  
If a vector of strings is provided, each row will receive a corresponding label.

# Parameters
- `Fax :: FigureAxes`: The `FigureAxes` object.
- `content :: Union{AbstractString, Vector}`: 
  - If a single string is provided, all x-axis labels will be the same.
  - If a vector is provided, each row gets a corresponding label. The vector **must** have the same length as the number of rows.
"""
function set_ylabel!(Fax :: FigureAxes, content::Union{AbstractString, Vector})
    axes = Fax.axes
    nrows, ncols = size(axes)
    if typeof(content) <: AbstractString
        for i in 1:nrows
            axes[i,1].ylabel = content
        end
    else
        if length(content) == nrows
            for i in 1:nrows
                axes[i,1].ylabel = content[i]
            end
        else
            error(DimensionMismatch,": The length of `content` $(length(content)) should be equal to number of rows $(nrows) if an array is provided!")
        end
    end
end

"""
    set_xlim!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, xlim::Tuple{Float64,Float64})

Sets the x-axis limits for a specified axis in the figure.

# Parameters
- `Fax::FigureAxes`: The figure container with axes.
- `axis_index::Tuple{Int64, Int64}`: The (row, column) position of the target axis in `Fax.axes`.
- `xlim::Tuple{Float64, Float64}`: The new limits for the x-axis in the form `(xmin, xmax)`.
"""
function set_xlim!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, xlim::Tuple{Float64,Float64})
    axis = Fax.axes[axis_index...]
    current_limits = axis.limits[]
    new_limits = (xlim, current_limits[2])
    axis.limits[] = new_limits
end

"""
    set_ylim!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, ylim::Tuple{Float64,Float64})

Sets the y-axis limits for a specified axis in the figure.

# Parameters
- `Fax::FigureAxes`: The figure container with axes.
- `axis_index::Tuple{Int64, Int64}`: The (row, column) position of the target axis in `Fax.axes`.
- `ylim::Tuple{Float64, Float64}`: The new limits for the y-axis in the form `(ymin, ymax)`.
"""
function set_ylim!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, ylim::Tuple{Float64,Float64})
    axis = Fax.axes[axis_index...]
    current_limits = axis.limits[]
    new_limits = (current_limits[1],ylim)
    axis.limits[] = new_limits
end

"""
    set_clim!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, clim::Tuple{Float64,Float64};
              target_hm_index::Int64=1)

Sets the color limits (`clim`) for a specified heatmap in the figure.

# Parameters
- `Fax::FigureAxes`: The figure container with axes.
- `axis_index::Tuple{Int64, Int64}`: The (row, column) position of the target axis in `Fax.axes`.
- `clim::Tuple{Float64, Float64}`: The new color range limits in the form `(cmin, cmax)`.

# Keyword Arguments
- `target_hm_index::Int64=1`: Specifies which heatmap to modify in case multiple heatmaps exist in the given axis. 
  If there is only one heatmap in the axis, this parameter can be ignored.
"""
function set_clim!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, clim::Tuple{Float64,Float64};
    target_hm_index::Int64=1)

    axis = Fax.axes[axis_index...]
    heatmaps = filter(p -> p isa Heatmap, axis.scene.plots)
    heatmap = heatmaps[target_hm_index]
    heatmap.colorrange[] = clim
end

"""
    set_xscale!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, scaling::Function)

Sets the x-axis scale for a specified axis in the figure.

# Parameters
- `Fax::FigureAxes`: The figure container with axes.
- `axis_index::Tuple{Int64, Int64}`: The (row, column) position of the target axis in `Fax.axes`.
- `scaling::Function`: The new function for scaling for the x-axis.
"""
function set_xscale!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, scaling::Function)
    axis = Fax.axes[axis_index...]
    axis.xscale[] = scaling
end

"""
    set_yscale!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, scaling::Function)

Sets the y-axis scale for a specified axis in the figure.

# Parameters
- `Fax::FigureAxes`: The figure container with axes.
- `axis_index::Tuple{Int64, Int64}`: The (row, column) position of the target axis in `Fax.axes`.
- `scaling::Function`: The new function for scaling for the x-axis.
"""
function set_yscale!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, scaling::Function)
    axis = Fax.axes[axis_index...]
    axis.yscale[] = scaling
end

function _get_axis_position(Fax::FigureAxes, target_axis_index::Tuple{Int64,Int64})
    fig = Fax.fig
    nrows, ncols = size(Fax.axes)

    if target_axis_index[1] > nrows || target_axis_index[2] > ncols || target_axis_index[1] < 1 || target_axis_index[2] < 1
        error("Index $(target_axis_index) is out of bounds for Fax.axes (valid range: (1:$nrows, 1:$ncols)).")
    end

    axis = Fax.axes[target_axis_index...]

    for gc in fig.layout.content
        if gc.content === axis
            return (gc.span.rows.start, gc.span.cols.start) 
        end
    end

    error("Axis at $(target_axis_index) was not found in the GridLayout. This may indicate that the Axis was not properly assigned to the Figure's layout.")
end

function _existcontents(layout::GridLayout, row::Int, col::Int)
    return any(gc -> (row in gc.span.rows && col in gc.span.cols), layout.content)
end

function _support_set_colorbar(plot)
    return any(ht -> typeof(plot) <: ht, HEATMAP)
end

"""
    colormap_with_base(cmap_name::Symbol; to_white::Bool=true, white_ratio::Float64=0.05)
Create a new colormap in Makie that smoothly transitions to white or black in the low-value region.

# Parameters:
- `cmap_name::Symbol`: Name of the built-in colormap in `ColorSchemes.jl`, e.g., `:viridis`, `:inferno`.
- `to_white::Bool`: If `true`, transitions to white; if `false`, transitions to black.
- `white_ratio::Float64`: Ratio of the white or black region in the colormap (0~1), default is `0.05` (5%).

# Returns:
- A colormap that smoothly transitions to white or black in the low-value region.
"""
function colormap_with_base(
    cmap_name::Union{Symbol,String};
    to_white::Bool = true,
    white_ratio::Float64 = 0.05
)
    if cmap_name isa String
        cmap_name = Symbol(cmap_name)
    end

    base_cmap = get(ColorSchemes.colorschemes, cmap_name, ColorSchemes.viridis)

    base_colors = [RGB(c) for c in base_cmap.colors]

    base_color = to_white ? RGB(1, 1, 1) : RGB(0, 0, 0)

    positions = Float64[0.0, white_ratio]
    colors = RGB[base_color, base_colors[1]]

    n = length(base_colors)
    for i in 2:n
        frac = (i-1)/(n-1) 
        domain_val = white_ratio + frac * (1 - white_ratio)
        push!(positions, domain_val)
        push!(colors, base_colors[i])
    end

    return cgrad(colors, positions, categorical=false)
end

"""
    set_colorbar!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64};
        clabel::AbstractString = "", 
        link_row_colormap=false, link_column_colormap=false, 
        mutiaxes_extend_range::Union{Nothing,UnitRange{Int64}}=nothing)
Sets and adjusts the colorbar for a specified heatmap in the figure. This function dynamically inserts a `Colorbar` next to the target axis, ensuring proper placement and spacing in the grid layout.
The function automatically adjusts the grid layout by inserting additional rows or columns if needed, ensuring the colormap placement does not overlap existing elements.

# Parameters
- `Fax::FigureAxes`: The figure container with axes.
- `axis_index::Tuple{Int64, Int64}`: The (row, column) position of the target axis in `Fax.axes`.

# Keyword Arguments
- `clabel::AbstractString = ""`: The label of colorbar.
- `link_row_colormap::Bool=false`: If `true`, extends the colormap across multiple columns.
- `link_column_colormap::Bool=false`: If `true`, extends the colormap across multiple rows.
- `mutiaxes_extend_range::Union{Nothing, UnitRange{Int64}}=nothing`: Specifies the range of axes to extend the colormap when working with multiple heatmaps.
"""
function set_colorbar!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64};
    clabel::AbstractString = "", 
    link_row_colormap=false, link_column_colormap=false, 
    mutiaxes_extend_range::Union{Nothing,UnitRange{Int64}}=nothing)

    if link_row_colormap && link_column_colormap
        error("ValueError: Row and column colormap cannot be linked simultaneously!")
    end

    heatmaps = filter(p -> _support_set_colorbar(p), Fax.axes[axis_index...].scene.plots)
    if isempty(heatmaps)
        error("EmptyHeatmapDetected: No Heatmap detected in the specified axis $(axis_index).")
    end
    targeted_heatmap = first(heatmaps)

    fig = Fax.fig
    layout = fig.layout

    nrows, ncols = size(layout)
    nrows_figindex, ncols_figindex = _get_axis_position(Fax,size(Fax.axes))
    current_axesrow_figindex, current_axescol_figindex  = _get_axis_position(Fax,axis_index)
    if isnothing(mutiaxes_extend_range)
        row_pos = link_column_colormap ? nrows_figindex + 1 : current_axesrow_figindex
        col_pos = link_row_colormap ? ncols_figindex : current_axescol_figindex
        if !link_column_colormap
            col_pos += 1 
        end
        vertical = !link_column_colormap

        row_range = row_pos
        col_range = col_pos
    else 
        if link_row_colormap
            row_range = mutiaxes_extend_range
            col_range = ncols_figindex + 1
            vertical = true  
        elseif link_column_colormap
            row_range = nrows_figindex + 1
            col_range = mutiaxes_extend_range
            vertical = false
        else
            row_range = mutiaxes_extend_range
            col_range = current_axescol_figindex + 1
            vertical = true
        end
    end

    if maximum(col_range) > ncols
        Makie.appendcols!(layout, 1)
    end
    if maximum(row_range) > nrows
        Makie.appendrows!(layout, 1)
    end
    for i in row_range
        for j in col_range
            content_exist = _existcontents(layout,i, j)
            if content_exist
                for block in contents(fig[i,j])
                    if block isa Colorbar
                        block.label = clabel
                        return
                    end
                end
                if link_column_colormap
                    Makie.insertrows!(layout, last(i), 1)
                    rowgap!(layout, last(i)-1, 3) 
                else
                    Makie.insertcols!(layout, last(j), 1)
                    colgap!(layout, last(j)-1, 16)
                end
            end
        end
    end
    hm_scale = String(Symbol(targeted_heatmap.colorscale[]))
    
    if hm_scale == "log10"
        matrix = targeted_heatmap[3][]
        if String(Symbol(targeted_heatmap.colorrange[])) == "MakieCore.Automatic()"
            vmin = minimum(filter(!iszero, matrix)) 
            vmax = maximum(matrix)
        else
            vmin, vmax = targeted_heatmap.colorrange[]
        end

        ticks, _, _ = _customLog10_ticks(matrix; clip=(vmin, vmax), minorticks=true)

        Colorbar(fig[row_range, col_range], targeted_heatmap, vertical=vertical,label=clabel,ticks=ticks,
                tickformat=_customLog10_formatter,
                minorticks=IntervalsBetween(9),
                minorticksvisible=true)
        return
    elseif hm_scale == "ReversibleScale(Symlog10)"
        if String(Symbol(targeted_heatmap.colorrange[])) == "MakieCore.Automatic()"
            matrix = targeted_heatmap[3][]
            vmin = minimum(matrix)
            vmax = maximum(matrix)
        elseif (targeted_heatmap.colorrange[] isa Tuple) || (targeted_heatmap.colorrange[] isa Vector)
            vmin,vmax = targeted_heatmap.colorrange[]
        else
            Colorbar(fig[row_range, col_range], targeted_heatmap, vertical=vertical,label=clabel)
            return
        end
        linthresh = 0.1 * min(abs(vmin),abs(vmax))
        major_ticks = _customSymlog10Ticks(linthresh, vmin, vmax)
        Colorbar(fig[row_range, col_range], targeted_heatmap, vertical=vertical,label=clabel,ticks=major_ticks,tickformat=_customSymlog10_formatter)
        
    else
        Colorbar(fig[row_range, col_range], targeted_heatmap, vertical=vertical,label=clabel)
        return
    end 
end

function _support_legend(plot)
    return any(sl -> typeof(plot) <: sl, SUPPORTLEGEND)
end

"""
    set_legend!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64})

Sets and adjusts the box of legend for a specified axis in the figure. This function dynamically inserts a `Legend` next to the target axis, ensuring proper placement and spacing in the grid layout.
# Parameters
- `Fax::FigureAxes`: The figure container with axes.
- `axis_index::Tuple{Int64, Int64}`: The (row, column) position of the target axis in `Fax.axes`.
"""
function set_legend!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64})
    fig = Fax.fig
    ax = Fax.axes[axis_index...]
    layout = fig.layout
    plots = filter(p -> _support_legend(p), ax.scene.plots)
    labels = Vector{AbstractString}(undef,length(plots))
    for (i,plot) in enumerate(plots)
        labels[i] = plot.label[]
    end

    nrows, ncols = size(layout)
    current_axesrow_figindex, current_axescol_figindex  = _get_axis_position(Fax,axis_index)
    Legend(Fax.fig[current_axesrow_figindex, ncols+1],plots,labels)
end

"""
    set_annotation!(Fax::FigureAxes, axis_index::Tuple{Int64,Int64}, text::AbstractString;
                  halign::Symbol=:left, valign::Symbol=:top)

Adds or updates an annotation box with a label at the specified axis position in the figure. The function ensures that the annotation is properly aligned within the `GridLayout`, and dynamically creates a new annotation box if one does not already exist.

If an annotation box already exists at the target axis, the text is updated instead of creating a new box.

# Parameters
- `Fax::FigureAxes`: The figure container with axes.
- `axis_index::Tuple{Int64, Int64}`: The (row, column) position of the target axis in `Fax.axes`.
- `text::AbstractString`: The annotation text to display.

# Keyword Arguments
- `halign::Symbol = :left`: Horizontal alignment of the annotation (`:left`, `:center`, or `:right`).
- `valign::Symbol = :top`: Vertical alignment of the annotation (`:top`, `:center`, or `:bottom`).
- `fontsize::Union{Nothing, Float64, Int64}=nothing`: The fontsize.
"""
function set_annotation!(Fax :: FigureAxes,  axis_index::Tuple{Int64,Int64}, text::AbstractString;
    halign::Symbol=:left, valign::Symbol= :top, fontsize::Union{Nothing, Float64,Int64}=nothing)
    cornerradius = 4
    axis_figindex = _get_axis_position(Fax,axis_index)
    current_content = contents(Fax.fig[axis_figindex...])
    if isnothing(fontsize)
        fontsize = DEFAULT_THEME_SETUP["fontsize"]
    end
    pad = 0.3 * fontsize
    gridlayouts = filter(x -> x isa GridLayout, current_content)
    for layout in gridlayouts
        content = contents(layout)
        exist_annotate = any(x -> x isa Box, content) && any(x -> x isa Label, content)
        if !isnothing(exist_annotate)
            lbl = content[2]
            lblha = lbl.halign[]
            lblva = lbl.valign[]
            if (lblha == halign) && (lblva == valign)
                lbl.text = text
                return
            end
        end
    end
    gl = GridLayout(Fax.fig[axis_figindex...], tellwidth=false, tellheight=false,halign=halign,valign=valign)
        Box(gl[1,1], color=:black, strokecolor=:black, strokewidth=0, cornerradius = cornerradius)
        Label(gl[1,1], text, 
            padding = (pad, pad, pad, pad), 
            halign = halign, valign = valign, 
            color = :white, fontsize = fontsize)
    exist_annotate = findfirst(gl -> begin
        content = contents(gl)
        any(x -> x isa Box, content) && any(x -> x isa Label, content)
        end, gridlayouts)
end

############# Polar heatmap integration #############
@recipe(PolarHeatmap, θ,r, matrix) do scene
    Attributes(
        # Keyword arguments
        alpha = 1.0,
        calculated_colors = nothing,
        colormap = :viridis,
        colorrange = MakieCore.Automatic(),
        colorscale = identity,
        fxaa = true,
        interpolate = false,
        nan_color = :white,
        overdraw = false,
        ssao = false,
        transparency = false,
        visible = true,
    )
end

function _center_to_edges(arr::AbstractVector)
    n = length(arr)
    edges = similar(arr, n+1)

    for i in 2:n
        edges[i] = (arr[i] + arr[i-1]) / 2
    end

    edges[1]  = arr[1]  - (edges[2]  - arr[1])
    edges[end] = arr[end] + (arr[end] - edges[n])

    return edges
end

function Makie.convert_arguments(::Type{<:PolarHeatmap},
    θcenter::AbstractVector,
    rcenter::AbstractVector,
    matrix::AbstractMatrix)
    nr, nc = size(matrix)
    lr = length(rcenter)
    lt = length(θcenter)
    if (lt,lr) == (nr,nc)
        new_matrix = matrix
    elseif (lt,lr) == (nc,nr)
        new_matrix = transpose(matrix)
    else
        error("Dimension of `matrix` must match with length(rcenter) and length(rcenter).")
    end

    rEdges = _center_to_edges(rcenter) 
    θEdges = _center_to_edges(θcenter)

    return θEdges,rEdges, new_matrix
end

function Makie.plot!(plot::PolarHeatmap)
    r = plot.r
    θ = plot.θ
    matrix = plot.matrix
    attr = plot.attributes

    cmin = attr.colorrange[][1]

    mesh_obj = Observable(GeometryBasics.Mesh(Point3f[], TriangleFace{Int}[]))
    color_obj = Observable(Float32[])

    function update_plot()
        nr = length(r[]) - 1
        nt = length(θ[]) - 1
        @assert size(matrix[]) == (nt, nr) "Dimension mismatch! With size of matrix: $(size(matrix[])) and the dimention of grid ($(nt), $(nr)))"

        nvertices = nr * nt * 4
        nfaces    = nr * nt * 2

        vertices  = Vector{Point3f}(undef, nvertices)
        faces     = Vector{GeometryBasics.TriangleFace{Int}}(undef, nfaces)
        colors    = Vector{Float32}(undef, nvertices)

        vertidx = 1
        faceidx = 1

        for i in 1:nr
            r1 = r[][i]
            r2 = r[][i+1]
            for j in 1:nt
                t1 = θ[][j]
                t2 = θ[][j+1]

                cellval = Float32(matrix[][j, i])
                if isnan(cellval)
                    cellval = cmin
                end

                vertices[vertidx] = Point3f(t1, r1, 0)
                colors[vertidx]   = cellval
                v1 = vertidx
                vertidx += 1

                vertices[vertidx] = Point3f(t1, r2, 0)
                colors[vertidx]   = cellval
                v2 = vertidx
                vertidx += 1

                vertices[vertidx] = Point3f(t2, r2, 0)
                colors[vertidx]   = cellval
                v3 = vertidx
                vertidx += 1

                vertices[vertidx] = Point3f(t2, r1, 0)
                colors[vertidx]   = cellval
                v4 = vertidx
                vertidx += 1


                faces[faceidx]   = TriangleFace(v1, v3, v2)
                faceidx += 1
                faces[faceidx]   = TriangleFace(v1, v4, v3)
                faceidx += 1
            end
        end

        mesh_obj[] = GeometryBasics.Mesh(vertices, faces)
        color_obj[] = colors

    end

    # **Observable dynamics update**
    on(r) do _ 
        update_plot()
    end
    on(θ) do _ 
        update_plot()
    end
    on(matrix) do _ 
        update_plot()
    end
    on(r) do _ 
        update_plot()
    end
    on(θ) do _ 
        update_plot()
    end
    on(attr.colormap) do _ 
        update_plot()
    end
    on(attr.colorrange) do _ 
        update_plot()
    end
    on(attr.colorscale) do _ 
        update_plot()
    end

    update_plot()
    mesh!(
        plot,
        mesh_obj;
        alpha       = attr.alpha,
        color       = color_obj,
        colormap    = attr.colormap,
        colorrange  = attr.colorrange,
        colorscale  = attr.colorscale,
        nan_color   = to_color(attr.nan_color[]),
        fxaa        = attr.fxaa,
        interpolate = attr.interpolate,
        overdraw    = attr.overdraw,
        shading     = NoShading,
        ssao        = attr.ssao,
        transparency= attr.transparency,
        visible     = attr.visible
    )

    return plot
end

############# Abstract type #############
const HEATMAP :: Vector{UnionAll} = [
    Heatmap,
    PolarHeatmap,
    Contourf,
    Voronoiplot,
    Poly,
    Contour
]
const SUPPORTLEGEND :: Vector{UnionAll} = [
    Lines,
    HLines,
    Scatter,
    ScatterLines,
    BarPlot,
    Contour
]

############# Symlog10Scale option #############
"""
    Symlog10Scale(cmin::AbstractFloat, cmax::AbstractFloat)

Creates a symmetrical logarithmic scale (`Symlog10`) for color mapping in visualizations. The function determines the linear threshold (`linsmth_boundary`) based on the smallest absolute value in the data range to ensure a smooth transition between the linear and logarithmic regions.

# Parameters
- `cmin::AbstractFloat`: The minimum value of the data range.
- `cmax::AbstractFloat`: The maximum value of the data range.

# Returns
- `Makie.Symlog10`: A symmetrical logarithmic scale object with the computed linear threshold.

# Notes
- The linear threshold (`linsmth_boundary`) is computed as `0.1 * min(abs(cmin), abs(cmax))`.
- The scale is symmetrical, meaning that values near zero are mapped linearly while larger values are mapped logarithmically.
"""
function Symlog10Scale(cmin::AbstractFloat, cmax::AbstractFloat)
    linsmth_boundary = 0.1 * min(abs(cmin),abs(cmax))
    return Makie.Symlog10(-linsmth_boundary,linsmth_boundary)
end

function _customSymlog10Ticks(linthresh::AbstractFloat, vmin::AbstractFloat, vmax::AbstractFloat)
    major_ticks = Float64[]
    stopping_threadshold = linthresh*0.6

    if stopping_threadshold <= 0
        error("stopping_threadshold must be greater than 0")
    end

    if vmin < -stopping_threadshold
        for i in ceil(Int, log10(abs(vmin))):-1:ceil(Int, log10(stopping_threadshold))
            push!(major_ticks, -10.0^i)
        end
    end

    push!(major_ticks, 0.0)  # 確保 0 是 major tick

    # 計算正數對數區間的 tick，從最大指數往下數
    if vmax > stopping_threadshold
        for i in ceil(Int, log10(vmax)):-1:ceil(Int, log10(stopping_threadshold))
            push!(major_ticks, 10.0^i)
        end
    end

    return major_ticks
end

function _customSymlog10_formatter(values)
    return map(v -> begin
        if abs(v) != 0.0  # log 區域
            latexstring((v>=0) ? "" : "-","10^{", string(round(Int, log10(abs(v)))), "}")
        else 
            string(0)
        end
    end, values)
end

function _customLog10_formatter(values)
    return map(v -> iszero(v) ? "0" : latexstring("10^{", string(round(Int, log10(abs(v)))), "}"), values)
end

function _customLog10_ticks(data; clip=nothing, min_exp=nothing, max_exp=nothing, minorticks::Bool=false)
    # Determine data range
    if clip !== nothing
        vmin, vmax = clip
    else
        vmin = minimum(filter(!iszero, data)) 
        vmax = maximum(data)
    end

    # Determine exponent range
    minexp = isnothing(min_exp) ? floor(Int, log10(vmin)) : min_exp
    maxexp = isnothing(max_exp) ? ceil(Int, log10(vmax)) : max_exp
    major_ticks = 10.0 .^ (minexp:maxexp)

    # Tick labels (log10 style)
    ticklabels = map(e -> latexstring("10^{", string(e), "}"), minexp:maxexp)

    # Minor ticks (between each 10^n and 10^{n+1})
    if minorticks
        minor_ticks = Float64[]
        for e in minexp:maxexp-1
            append!(minor_ticks, 10.0^e .* (2:9))
        end
        return major_ticks, ticklabels, minor_ticks
    else
        return major_ticks, ticklabels
    end
end

############# Pcolor plotting #############
function _generate_discrete_points_with_values(x::AbstractVector, y::AbstractVector, vals::AbstractMatrix)
    X,Y = meshgrid(x,y)
    points = [Point3f(x[i], y[j], vals[i,j]) for i in 1:m for j in 1:n]
    return points
end

"""
    lazypcolor!(Fax::FigureAxes, axis_index::Tuple{Int,Int}, x, y, matrix; kwargs...)

Efficiently updates or creates a pseudocolor plot (`heatmap` or `polarheatmap`) in a given axis of a `FigureAxes` structure.

This function checks if a heatmap-like plot already exists in the specified axis. If it does, it updates the existing plot's 
`x`, `y`, and `z` (stored in `converted[3]`) values to avoid re-plotting. If no such plot exists, it creates a new one.

It supports both **Cartesian (`heatmap!`)** and **Polar (`polarheatmap!`)** coordinate systems.

# Parameters
- `Fax::FigureAxes`: The `FigureAxes` structure that contains the figure and its axes.
- `axis_index::Tuple{Int,Int}`: The `(row, column)` index of the target axis in `Fax.axes`.
- `x, y`: The coordinate vectors defining the grid of the heatmap.
- `matrix`: The data matrix to be visualized.

# Keyword Arguments
Accepts all keyword arguments supported by `heatmap!` and `polarheatmap!`, such as:
- `colorrange`: Sets the color mapping range.
- `colorscale`: Defines the scaling function for colors (e.g., `log10` for logarithmic scaling).
- `colormap`: Specifies the colormap used in the plot.
- Other attributes related to appearance and scaling.
"""
function lazypcolor!(Fax::FigureAxes, axis_index::Tuple{Int,Int}, x, y, matrix; kwargs...)
    function update_necessary_attribute!(plot::Makie.ScenePlot; kwargs...)
        if haskey(kwargs,:colorrange)
            if (oldplot.colorrange[] != kwargs[:colorrange]) && (kwargs[:colorrange][2] > kwargs[:colorrange][1])
                oldplot.colorrange[] = kwargs[:colorrange]
            end
        end
        if haskey(kwargs,:colorscale)
            if oldplot.colorscale[] != kwargs[:colorscale]
                oldplot.colorscale[] = kwargs[:colorscale]
            end
        end
        if haskey(kwargs,:colormap)
            if oldplot.colormap[] != kwargs[:colormap]
                oldplot.colormap[] = kwargs[:colormap]
            end
        end
    end
    axis = Fax.axes[axis_index...]
    axis_type = Fax.axes_type[axis_index...]

    if axis_type == "Cartesian" || current_backend()=="Cairo"
        if axis_type == "Polar"
            x,y = y,x
            matrix = transpose(matrix)
        end
        oldplot = nothing
        for p in axis.scene.plots
            if p.label[] == "lazypcolor"
                oldplot = p
                break
            end
        end
        if isnothing(oldplot)
            hm = heatmap!(axis, x, y, matrix; kwargs...)
            hm.label = "lazypcolor"
        else
            oldplot.converted[1][] = _center_to_edges(x)        
            oldplot.converted[2][] = _center_to_edges(y)         
            oldplot.converted[3][] = matrix   
            update_necessary_attribute!(oldplot; kwargs...)
        end
    elseif axis_type == "Polar" 
        oldplot = nothing
        for p in axis.scene.plots
            if p.label[] == "lazypcolor"
                oldplot = p
                break
            end
        end
        if isnothing(oldplot)
            hm = polarheatmap!(axis, y, x, matrix; kwargs...)
            hm.label = "lazypcolor"
        else
            oldplot = axis.scene.plots[1]
            θedge, redge, newmatrix = Makie.convert_arguments(PolarHeatmap, y ,x, matrix)
            oldplot.θ[] = θedge
            oldplot.r[] = redge
            oldplot.matrix[] = newmatrix
            update_necessary_attribute!(oldplot; kwargs...)
        end
        rlims!(axis, x[1], x[end])
    else
        error("UnsupportedAxisType: $axis_type is not supported in `lazypcolor!`")
    end
end

############# Other tools for plotting #############
"""
    Get_vminmax(array::Array)

Computes the minimum (`vmin`) and maximum (`vmax`) values for setting the color scale in visualizations. The calculation is based on the median and standard deviation of the input array, excluding infinite and `NaN` values.

This function automatically adjusts `vmin` to avoid negative values if the input array contains only non-negative numbers.

# Parameters
- `arr::Array`: The input numerical array which may contain `NaN` or `Inf` values.

# Returns
- `(vmin::Float64, vmax::Float64)`: A tuple containing the computed minimum and maximum values.
"""
function Get_vminmax(arr::Array)
    array = copy(arr)
    array = replace_inf_with_nan!(array)
    med = median(filter(x -> !isnan(x), array)) 
    std_dev = std(filter(x -> !isnan(x), array))
    
    vmax = med + 3 * std_dev
    vmin = med - 3 * std_dev

    if minimum(filter(x -> !isnan(x), array)) >= 0.0 && vmin < 0.0
        if vmax < 1e-15
            vmin = 5e-18
        else
            vmin = 1e-15
        end
    end

    @warn "Warning: Automatically calculate (vmin, vmax) = ($(vmin), $(vmax))" 
    return (vmin, vmax)
end
