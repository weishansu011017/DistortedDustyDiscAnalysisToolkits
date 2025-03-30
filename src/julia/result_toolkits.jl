"""
    The tool kits for analysis/visualizing the dumpfile from PhantomRevealer analysis
        by Wei-Shan Su,
        July 18, 2024
"""
initialization_modules()
const TRANSFER_DICT = Dict{String, LaTeXString}(
    "∇" => L"$\nabla$",
    "ϕ" => L"$\phi$",
    "θ" => L"$\theta$",
    "ρ" => L"$\rho$",
    "Σ" => L"$\Sigma$",
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
    println(z_unit)
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
    draw_Fig!(Fax)
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

    # 檢查數據維度
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

    # 啟動 Makie
    activate_backend("GL")

    # 如果沒有提供 `Fax`，則創建一個新的 `FigureAxes`
    if isnothing(Fax)
        Fax = FigureAxes(1,1, figsize=figsize)
    end

    # 設定 colorbar 範圍
    if isnothing(vlim)
        entervlim = Get_vminmax(z)
    else
        entervlim = vlim
    end

    # 設定色彩縮放方式
    scale::Union{Function,ReversibleScale} = identity
    if cbar_log
        if entervlim[1] <= 0.0
            scale = Symlog10Scale(entervlim...)
        else
            scale = log10
        end
    end
    # 繪製 pcolor 圖
    lazypcolor!(Fax, (1,1), x, y, z, colormap=colormap, colorrange=entervlim, colorscale=scale)

    # 設定 colorbar 和標籤
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
