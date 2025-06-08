"""
Ridge detection with automatic scale selection from Lindeberg(1998)
    by Wei-Shan Su,
    May 10, 2025
"""

const _GAUSSIANNORMALIZED :: Dict{Symbol, Symbol} = Dict{Symbol, Symbol}(
    # (dx^-1 order, dy^-1 order, normalized?)
    :gaussian_kernel => :sum,
    :gaussian_kerneldx => :L2,
    :gaussian_kerneldy => :L2,
    :gaussian_kerneldx2 => :L2,
    :gaussian_kerneldy2 => :L2,
    :gaussian_kerneldxdy => :L2
)

@inline function _gaussian(x::Real; t::Real)
    inv(sqrt(2π*t)) * exp(-(x^2)/(2t))
end

@inline function gaussian_kernel(x::Real, y::Real; tx::Real = 1.0, ty::Real = 1.0)
    _gaussian(x,t=tx) * _gaussian(y,t=ty)
end

@inline function gaussian_kerneldx(x::Real, y::Real; tx::Real = 1.0, ty::Real = 1.0)
    (- x/tx) * gaussian_kernel(x,y,tx=tx,ty=ty)
end

@inline function gaussian_kerneldy(x::Real, y::Real; tx::Real = 1.0, ty::Real = 1.0)
    (- y/ty) * gaussian_kernel(x,y,tx=tx,ty=ty)
end

@inline function gaussian_kerneldx2(x::Real, y::Real; tx::Real = 1.0, ty::Real = 1.0)
    ((x^2 - tx)/tx^2) * gaussian_kernel(x,y,tx=tx,ty=ty)
end

@inline function gaussian_kerneldy2(x::Real, y::Real;tx::Real = 1.0, ty::Real = 1.0)
    ((y^2 - ty)/ty^2) * gaussian_kernel(x,y,tx=tx,ty=ty)
end

@inline function gaussian_kerneldxdy(x::Real, y::Real;tx::Real = 1.0, ty::Real = 1.0)
    ((x*y)/(tx*ty)) * gaussian_kernel(x,y,tx=tx,ty=ty)
end

function _gaussian_box(kernel::Function; tx = 1.0, ty = 1.0, boxfactor = 8.0)
    σx = sqrt(tx)
    σy = sqrt(ty)
    boxsizex = Int64(ceil(2*boxfactor*σx + 1))
    boxsizey = Int64(ceil(2*boxfactor*σy + 1))
    x = collect(LinRange(-boxfactor*σx, boxfactor*σx, boxsizex))
    y = collect(LinRange(-boxfactor*σy, boxfactor*σy, boxsizey))

    X, Y = meshgrid(x, y)
    box = kernel.(X,Y, tx = tx, ty=ty)
    return box
end

function _generate_equalsize_kernel(image :: AbstractMatrix, kernel_function :: Function; tx = 1.0, ty = 1.0, boxfactor = 8.0)
    kernelbox = _gaussian_box(kernel_function; tx=tx, ty=ty, boxfactor = boxfactor)
    n, m = size(kernelbox)
    N, M = size(image)
    p = (N-n) ÷ 2  # Rational division
    q = (M-m) ÷ 2  # Rational division

    kernel = zeros(Float64, N, M)
    kernel[p+1:p+n, q+1:q+m] .= kernelbox
    if _GAUSSIANNORMALIZED[Symbol(kernel_function)] == :sum
        kernel ./= sum(kernel) 
    elseif _GAUSSIANNORMALIZED[Symbol(kernel_function)] == :L2
        kernel ./= sqrt(sum(abs2, kernel))
    end
    kernel = ifftshift(kernel)
    return kernel
end

function _padding_matrix(image::AbstractMatrix, padax1_num :: Int64, padax2_num :: Int64; padax1_mode=:circular , padax2_mode=0.0)    
    padax1_spec = typeof(padax1_mode) <: Real ? Fill(padax1_mode, (padax1_num,0)) : Pad(padax1_mode, (padax1_num,0))
    padax2_spec = typeof(padax2_mode) <: Real ? Fill(padax2_mode, (0,padax2_num)) : Pad(padax2_mode, (0,padax2_num))
    padaxn_spec = (padax1_spec, padax2_spec)
    image_pad = padarray(image, padaxn_spec[2])
    image_pad = collect(padarray(image_pad, padaxn_spec[1]))
    return image_pad
end

"""
    function image_convolution(image::Array, kernel_function::Function; tx = 1.0, ty = 1.0, padax1_mode=:circular , padax2_mode=0.0, boxfactor = 8.0) 
Convolving the image to the given kernel function.

# Parameters
- `image :: AbstractMatrix`: The image.

# Keyword Arguments
- `tx = 1.0`: STD along first-axis for the gaussian kernel.
- `ty = 1.0`: STD along second-axis for the gaussian kernel.
- `padax1_mode = :circular`: The padding mode of first axis for FFT-convolution. Option: specific number(e.g. 0.0), :circular, :replicate, :symmetric, :reflect .(Note: for corners padding, it will follow this mode)
- `padax2_mode = 0.0`: The padding mode of second axis for FFT-convolution. Option: specific number(e.g. 0.0), :circular, :replicate, :symmetric, :reflect .
- `boxfactor = 8.0`: The size of kernel box for convolution.

# Returns
- `Matrix{Float64}`: The image after convolutiton.
"""
function image_convolution(image::Array, kernel_function::Function; tx = 1.0, ty = 1.0, padax1_mode=:circular , padax2_mode=0.0, boxfactor = 8.0) 
    N, M = size(image)
    padfactor1 = Int64(ceil(1 + 2*boxfactor/N))
    padfactor2 = Int64(ceil(1 + 2*boxfactor/M))
    pad_1 = (N * (padfactor1 - 1)) ÷ 2          # axis=1 (rows)
    pad_2 = (M * (padfactor2 - 1)) ÷ 2          # axis=2 (cols)

    img_pad  = _padding_matrix(image, pad_1, pad_2, padax1_mode=padax1_mode, padax2_mode=padax2_mode)
    kern_pad = _generate_equalsize_kernel(img_pad, kernel_function, tx=tx, ty=ty,boxfactor=boxfactor)

    F = fft(img_pad)
    K = fft(kern_pad)
    conv_pad = real.(ifft!(F .* K))
    return conv_pad[pad_1+1 : pad_1+N,  pad_2+1 : pad_2+M]
end

"""
    function ridge_selector(image::AbstractMatrix; gamma = 0.0, tx = 1.0, ty = 1.0, padax1_mode=:circular , padax2_mode=0.0, boxfactor = 8.0)

Ridge detection with given scale-t. Also calculate the ridge strengtth defined as

    N_{γ-norm}L = t^4γ * (Lxx + Lyy)^2 * ((Lxx - Lyy)^2 + 4 * Lxy^2)

# Parameters
- `image :: AbstractMatrix`: The image.

# Keyword arguments
- `gamma = 0.0`: The normalized consttant that would be used in calculating ridge strength.
- `tx = 1.0`: STD along first-axis for the gaussian kernel.
- `ty = 1.0`: STD along second-axis for the gaussian kernel.
- `padax1_mode = :circular`: The padding mode of first axis for FFT-convolution. Option: specific number(e.g. 0.0), :circular, :replicate, :symmetric, :reflect .(Note: for corners padding, it will follow this mode)
- `padax2_mode = 0.0`: The padding mode of second axis for FFT-convolution. Option: specific number(e.g. 0.0), :circular, :replicate, :symmetric, :reflect .
- `boxfactor = 8.0`: The size of kernel box for convolution.

# Returns
- `Matrix{Bool}`: The bit array which indicates the poisition of ridge candidate.
- `Matrix{Float64}`: The array of ridge strength.
"""
function ridge_selector(image::AbstractMatrix; gamma = 0.0, tx = 1.0, ty = 1.0, padax1_mode=:circular , padax2_mode=0.0, boxfactor = 8.0)
    # Image convolution derivative
    image_norm = (image .- minimum(image))./(maximum(image) .- minimum(image))
    ∇imagex = image_convolution(image_norm, gaussian_kerneldx ;tx = tx, ty=ty, padax1_mode=padax1_mode, padax2_mode=padax2_mode, boxfactor=boxfactor)
    ∇imagey = image_convolution(image_norm, gaussian_kerneldy ;tx = tx, ty=ty, padax1_mode=padax1_mode, padax2_mode=padax2_mode, boxfactor=boxfactor)
    ∇2imagexx = image_convolution(image_norm, gaussian_kerneldx2 ;tx = tx, ty=ty, padax1_mode=padax1_mode, padax2_mode=padax2_mode, boxfactor=boxfactor)
    ∇2imageyy = image_convolution(image_norm, gaussian_kerneldy2 ;tx = tx, ty=ty, padax1_mode=padax1_mode, padax2_mode=padax2_mode, boxfactor=boxfactor)
    ∇2imagexy = image_convolution(image_norm, gaussian_kerneldxdy ;tx = tx, ty=ty, padax1_mode=padax1_mode, padax2_mode=padax2_mode, boxfactor=boxfactor)

    mean_image_norm = mean(image_norm)
    std_image_norm = std(image_norm)
    image_mask = image_norm .≥ (mean_image_norm)


    # Masked array that would mask all the points except spiral
    mask = zeros(Bool, size(image_norm))
    # Array of Ridge strength (with N_{γ-norm}L)
    strength = zeros(Float64, size(image_norm))

    # Parameter of strength
    t = sqrt(tx*ty)
    t4gamma = t^(4*gamma)

    # Threadshold for L
    ε = 0.1 / sqrt(t)

    @threads for i in eachindex(mask)
        Lx = ∇imagex[i]
        Ly = ∇imagey[i]
        Lxx = ∇2imagexx[i]
        Lyy = ∇2imageyy[i]
        Lxy = ∇2imagexy[i]

        # rotational parameters
        # Q = (Lxx - Lyy)/sqrt((Lxx-Lyy)^2 + 4Lxy^2) = (Lxx - Lyy) / P
        P = sqrt((Lxx - Lyy)^2 + 4 * Lxy^2)
        if P <1e-12
            mask[i] = false
            continue
        end
        Q = (Lxx - Lyy)/P
        cosβ = sqrt(0.5 * (1 + Q))
        sinβ = sign(Lxy) * sqrt(0.5 * (1 - Q))

        # Rotating Lx Ly
        Lp = Lx * sinβ - Ly * cosβ
        Lq = Lx * cosβ + Ly * sinβ
        Lpp = Lxx * sinβ^2 - 2 * Lxy * sinβ * cosβ + Lyy * cosβ^2
        Lqq = Lxx * cosβ^2 + 2 * Lxy * sinβ * cosβ + Lyy * sinβ^2
        # Appling the requirement
        if image_mask[i] && (abs(Lp) <= ε) && (Lpp < 0) && (abs(Lpp) ≥ abs(Lqq)) || ((abs(Lq) <= ε) && (Lqq < 0) && (abs(Lqq) ≥ abs(Lpp)))   
            Ngammanorm = t4gamma * (Lxx + Lyy)^2 * P^2
            mask[i] = true
            strength[i] = Ngammanorm
        else
            mask[i] = false
        end
    end
    return mask, strength
end

"""
    function ridge_detection_automatic_scale_selection(image::AbstractMatrix; gamma = 0.75, width_pixel_range = (5.0, 15.0), width_resolution = 15, padax1_mode=:circular , padax2_mode=0.0, boxfactor = 8.0)

Ridge detection with automatic scale selection in Lindeberg(1998). Please check Lindeberg(1998) for further information.

# Parameters
- `image :: AbstractMatrix`: The image.

# Keyword arguments
- `gamma = 0.0`: The normalized consttant that would be used in calculating ridge strength.
- `width_pixel_range = (5.0, 15.0)`: The range of width of ridge IN PIXEL.
- `width_resolution = 15`: The resolution of t-scaling for scale selection.
- `tx = 1.0`: STD along first-axis for the gaussian kernel.
- `ty = 1.0`: STD along second-axis for the gaussian kernel.
- `padax1_mode = :circular`: The padding mode of first axis for FFT-convolution. Option: specific number(e.g. 0.0), :circular, :replicate, :symmetric, :reflect .(Note: for corners padding, it will follow this mode)
- `padax2_mode = 0.0`: The padding mode of second axis for FFT-convolution. Option: specific number(e.g. 0.0), :circular, :replicate, :symmetric, :reflect .
- `boxfactor = 8.0`: The size of kernel box for convolution.

# Returns
- `Matrix{Bool}`: The bit array which indicates the poisition of ridge candidate.
- `Matrix{Float64}`: The array of ridge strength.
- `Matrix{Float64}`: The array of best t value at each points.
"""
function ridge_detection_automatic_scale_selection(image::AbstractMatrix; gamma = 0.75, width_pixel_range = (5.0, 15.0), width_resolution = 15, padax1_mode=:circular , padax2_mode=0.0, boxfactor = 8.0)
    trange = LinRange(width_pixel_range[1]^2,width_pixel_range[2]^2, width_resolution)
    final_mask = zeros(Bool, size(image))
    largest_strength = zeros(Float64, size(image))
    best_t = zeros(Float64, size(image))
    for ttest in trange
        mask, strength = ridge_selector(image, gamma=gamma, tx=ttest, ty=ttest, padax1_mode=padax1_mode, padax2_mode=padax2_mode, boxfactor=boxfactor)
        @threads for j in eachindex(strength)
            current_strength = largest_strength[j]
            new_strength = strength[j]
            if (mask[j]) && (new_strength > current_strength)
                final_mask[j] = true
                largest_strength[j] = new_strength
                best_t[j] = ttest
            end
        end
    end
    return final_mask, largest_strength, best_t
end

"""
    function bitarray2pointsset(bitarray::AbstractArray{Bool}, axes::Union{Nothing, Tuple{AbstractArray, AbstractArray}} = nothing)

Convert a 2D bit array (`Bool` matrix) into a list of points where the value is `true`.

If `axes` is provided, the resulting coordinates are mapped to physical space using the corresponding axis values. Otherwise, the output points are in pixel coordinates (integer indices).

# Arguments
- `bitarray::AbstractArray{Bool}`: A 2D Boolean array representing the binary mask.
- `axes::Union{Nothing, Tuple{AbstractArray, AbstractArray}} = nothing`: 
  A tuple of physical axes `(x, y)` where:
  - `axes[1]` corresponds to the vertical axis (rows, length must be `size(bitarray, 1)`),
  - `axes[2]` corresponds to the horizontal axis (columns, length must be `size(bitarray, 2)`).

# Returns
- `Vector{Tuple{Float64, Float64}}` if `axes` is provided.
- `Vector{Tuple{Int64, Int64}}` if `axes == nothing`.
"""
function bitarray2pointsset(bitarray::AbstractArray{Bool}, axes::Union{Nothing, Tuple{AbstractArray,AbstractArray}}=nothing)
    N,M = size(bitarray)
    iarray = nothing
    jarray = nothing
    pointsset = nothing
    if !isnothing(axes)
        pointsset = Tuple{Float64, Float64}[]
        if (length(axes[1]) == N) && (length(axes[2]) == M)
            iarray = axes[1]
            jarray = axes[2]
        else
            error("DimentionMismatch: The dimention of axes $(axes) do not match with $((N,M)).")
        end
    else 
        pointsset = Tuple{Int64, Int64}[]
        iarray = collect(1:N)
        jarray = collect(1:M)
    end

    for i in eachindex(iarray), j in eachindex(jarray)
        id = iarray[i]
        jd = jarray[j]
        if bitarray[i,j]
            push!(pointsset, (id,jd))
        end
    end
    return pointsset
end