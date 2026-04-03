using Partia
using .Threads
using CUDA

function _divv!(grid :: GeneralGrid{D}, i :: Int, input :: ITPINPUT, LBVHConcise :: LinearBVHConcise, itp_strategy :: Type{ITPSTRATEGY}) where {D, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    point = grid.coor[i]
    # Particles searching
    ha_nearest_idx, _ = LBVH_find_nearest(LBVHConcise, point)
    ha = input.h[ha_nearest_idx]
    divvi = divergence_quantity_interpolate(input, point, ha, LBVHConcise, 1, 2, 3, itp_strategy)
    grid.grid[i] = divvi
    return nothing
end

@inline function _divv!(grid :: GeneralGrid{D}, input :: ITPINPUT, LBVHConcise :: LinearBVHConcise, itp_strategy :: Type{ITPSTRATEGY}) where {D, T <: AbstractFloat, V <: CuDeviceVector{T}, ITPINPUT <: InterpolationInput{T, V}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    tid    = Int(CUDA.threadIdx().x)
    bid    = Int(CUDA.blockIdx().x)
    bdim   = Int(CUDA.blockDim().x)
    gdim   = Int(CUDA.gridDim().x)

    gid    = (bid - 1) * bdim + tid
    stride = bdim * gdim

    npoints = length(grid)
    i = gid
    while i <= npoints
        point = grid.coor[i]
        # Particles searching
        ha_nearest_idx, _ = LBVH_find_nearest(LBVHConcise, point)
        ha = input.h[ha_nearest_idx]

        divvi = divergence_quantity_interpolate(input, point, ha, LBVHConcise, 1, 2, 3, itp_strategy)
        grid.grid[i] = divvi
        i += stride
    end
    return nothing
end

function add_divv!(data :: PartiaDataFrame, mass_from_params :: MassFromParams, :: CPUComputeBackend;  itp_strategy::Type{B} = itpSymmetric, smoothed_kernel :: Type{K} = M5_spline) where {B <: AbstractInterpolationStrategy, K <: AbstractSPHKernel}
    @info "Preparing data for ∇⋅v computation..."
    @time begin
    input, _ = build_input(data, mass_from_params, scalars = Symbol[], divergences = [:v], smoothed_kernel = smoothed_kernel);
    end
    dim = get_dim(data)
    @info "Building LinearBVH for divv computation..."
    @time begin
    LBVH = LinearBVH!(input, Val(dim), CodeType = UInt64)
    LBVHConcise = LinearBVHConcise(LBVH)
    end
    @info "Building grid for ∇⋅v computation..."
    @time begin
    grid = GeneralGrid(data.dfdata.x, data.dfdata.y, data.dfdata.z)
    end

    npoints = length(grid)
    @info "Computing ∇⋅v for $npoints grid points..."
    @time begin
    @inbounds @threads for i in 1:npoints
        _divv!(grid, i, input, LBVHConcise, itp_strategy)
    end
    end

    data[!, "∇⋅v"] = grid.grid
    return nothing
end

@inline function add_divv!(data :: PartiaDataFrame, mass_from_params :: MassFromParams;  itp_strategy::Type{B} = itpSymmetric, smoothed_kernel :: Type{K} = M5_spline) where {B <: AbstractInterpolationStrategy, K <: AbstractSPHKernel}
    add_divv!(data, mass_from_params, CPUComputeBackend(); itp_strategy=itp_strategy, smoothed_kernel=smoothed_kernel)
    return nothing
end

function add_divv!(data :: PartiaDataFrame, mass_from_params :: MassFromParams, :: CUDAComputeBackend;  itp_strategy::Type{B} = itpSymmetric, smoothed_kernel :: Type{K} = M5_spline) where {B <: AbstractInterpolationStrategy, K <: AbstractSPHKernel}
    @info "Preparing data for ∇⋅v computation..."
    @time begin
    input, _ = build_input(data, mass_from_params, scalars = Symbol[], divergences = [:v], smoothed_kernel = smoothed_kernel);
    end
    dim = get_dim(data)
    @info "Building LinearBVH for divv computation..."
    @time begin
    LBVH = LinearBVH!(input, Val(dim), CodeType = UInt64)
    LBVHConcise = LinearBVHConcise(LBVH)
    end
    @info "Building grid for ∇⋅v computation... On CUDA"
    @time begin
    grid_cu = GeneralGrid(data.dfdata.x, data.dfdata.y, data.dfdata.z, CUDAComputeBackend())
    end

    @info "Uploading input data to GPU..."
    @info "Converting input to CuVector..."
    @time begin
    input_cu = to_CuVector(input)
    end
    @info "Converting LBVH to CuVector..."
    @time begin
    LBVHConcise_cu = to_CuVector(LBVHConcise)
    end

    npoints = length(grid_cu)
    @info "Computing ∇⋅v for $npoints grid points..."
    @time begin
    @cuda threads=256 blocks=cld(npoints, 256) _divv!(grid_cu, input_cu, LBVHConcise_cu, itp_strategy)
    CUDA.synchronize()
    end

    # @info "Run again for device_code_warntype"
    # @time begin
    # @device_code_warntype @cuda threads=256 blocks=cld(npoints, 256) _divv!(grid_cu, input_cu, LBVH_cu, itp_strategy)
    # CUDA.synchronize()
    # end

    # @info "Investigating warp divergence"
    # @time begin
    # open("divv_ptx.txt", "w") do io
    #     redirect_stdout(io) do
    #         CUDA.@device_code_ptx @cuda threads=256 blocks=cld(npoints, 256) _divv!(grid_cu, input_cu, LBVH_cu, itp_strategy)
    #     end
    # end
    # open("divv_sass.txt", "w") do io
    #     redirect_stdout(io) do
    #         CUDA.@device_code_sass @cuda threads=256 blocks=cld(npoints, 256) _divv!(grid_cu, input_cu, LBVH_cu, itp_strategy)
    #     end
    # end
    # end

    @info "Downloading ∇⋅v results from GPU..."
    @time begin
    divv = Vector(grid_cu.grid)
    end
    data[!, "∇⋅v"] = divv
    return nothing
end


prdf_list = read_phantom("./test/testinput/testdumpfile_00000", separate_types=:all);
datag = prdf_list[1];

x = datag.dfdata.x
y = datag.dfdata.y
z = datag.dfdata.z

add_rho!(datag)

# @time add_divv!(datag, MassFromParams(:mass), smoothed_kernel=M6_spline, CUDAComputeBackend())
@time add_divv!(datag, MassFromParams(:mass), smoothed_kernel=M6_spline, CPUComputeBackend())
