"""
    LineSamples_interpolation(backend::CUDAComputeBackend, grid_template::LineSamples{3,TF},
                              input::InterpolationInput{3,TF}, catalog::InterpolationCatalog{3, N, 0, 0, 0, N},
                              itp_strategy::Type{ITPSTRATEGY}=itpScatter)

Performs line-integrated SPH interpolation on the GPU using CUDA.

This routine mirrors the CPU `LineSamples_interpolation` path: it only supports
scalar line-integrated quantities and only the `itpScatter` strategy.
"""
function Partia.LineSamples_interpolation(:: CUDAComputeBackend, grid_template::LineSamples{3, TF}, input::InterpolationInput{3, TF}, catalog::InterpolationCatalog{3, N, 0, 0, 0, N}, itp_strategy::Type{ITPSTRATEGY} = itpScatter) where {N, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    itp_strategy === itpScatter || throw(ArgumentError(
        "LineSamples_interpolation only supports itpScatter. " *
        "Line-integrated samples do not have a well-defined query smoothing length ha, " *
        "so itpGather and itpSymmetric are not supported."
    ))

    grids, LBVH, names, catalog_consice = Partia.initialize_interpolation(Partia.CPUComputeBackend(), grid_template, input, catalog)
    @info "     SPH Interpolation: Copying interpolated grids to device memory..."
    input_Cu = to_CuVector(input)
    grids_Cu = ntuple(i -> to_CuVector(grids[i]), Val(N))
    LBVH_Cu = to_CuVector(LBVH)
    @info "     SPH Interpolation: End copying interpolated grids to device memory."

    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @cuda threads=(256,) blocks=(cld(npoints, 256)) _line_samples_interpolation_kernel!(grids_Cu, input_Cu, catalog_consice, LBVH_Cu, itpScatter)
    CUDA.synchronize()
    @info "     SPH Interpolation: End interpolation."
    @info "     SPH Interpolation: Copying interpolated grids back to host memory..."
    grids_result = ntuple(i -> Partia.to_HostVector(grids_Cu[i]), Val(N))
    @info "     SPH Interpolation: End copying interpolated grids back to host memory."
    return GridBundle(grids_result, names)
end

@inline function _line_samples_interpolation_kernel!(
    grids::NTuple{N, LineSamples{3, TF}},
    input::InterpolationInput{3, TF},
    catalog_consice::InterpolationCatalogConcise{3, N, 0, 0, 0},
    LBVH::LinearBVH,
    ::Type{itpScatter},
) where {N, TF <: AbstractFloat}
    tid = Int(CUDA.threadIdx().x)
    bid = Int(CUDA.blockIdx().x)
    bdim = Int(CUDA.blockDim().x)
    gdim = Int(CUDA.gridDim().x)

    gid = (bid - 1) * bdim + tid
    stride = bdim * gdim

    npoints = length(grids[1])
    i = gid
    while i <= npoints
        @inbounds begin
            geometry = grids[1]

            xoa = geometry.origin[1][i]
            yoa = geometry.origin[2][i]
            zoa = geometry.origin[3][i]
            origin::NTuple{3, TF} = (xoa, yoa, zoa)

            xda = geometry.direction[1][i]
            yda = geometry.direction[2][i]
            zda = geometry.direction[3][i]
            direction::NTuple{3, TF} = (xda, yda, zda)

            scalar_slots::NTuple{N, Int} = catalog_consice.scalar_slots
            scalar_snormalization::NTuple{N, Bool} = catalog_consice.scalar_snormalization
            scalars::NTuple{N, TF} =
                Partia.KernelInterpolation._line_integrated_quantities_interpolate_kernel(
                    input,
                    origin,
                    direction,
                    LBVH,
                    scalar_slots,
                    scalar_snormalization,
                    itpScatter,
                )

            if N > 0
                for j in 1:N
                    grids[j].grid[i] = scalars[j]
                end
            end
        end

        i += stride
    end
    return nothing
end
