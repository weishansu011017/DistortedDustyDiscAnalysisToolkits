"""
    LineSamples_interpolation(backend::CUDAComputeBackend, grid_template::LineSamples{D},
                              input::ITPINPUT, catalog::InterpolationCatalog{N, 0, 0, 0, N},
                              itp_strategy::Type{ITPSTRATEGY}=itpScatter)

Performs line-integrated SPH interpolation on the GPU using CUDA.

This routine mirrors the CPU `LineSamples_interpolation` path: it only supports
scalar line-integrated quantities and only the `itpScatter` strategy.
"""
function PhantomRevealer.LineSamples_interpolation(:: CUDAComputeBackend, grid_template::LineSamples{D}, input::ITPINPUT, catalog::InterpolationCatalog{N, 0, 0, 0, N}, itp_strategy::Type{ITPSTRATEGY} = itpScatter) where {D, N, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    itp_strategy === itpScatter || throw(ArgumentError(
        "LineSamples_interpolation only supports itpScatter. " *
        "Line-integrated samples do not have a well-defined query smoothing length ha, " *
        "so itpGather and itpSymmetric are not supported."
    ))

    grids, LBVH, names, catalog_consice = PhantomRevealer.initialize_interpolation(PhantomRevealer.CPUComputeBackend(), grid_template, input, catalog)
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
    grids_result = ntuple(i -> PhantomRevealer.to_HostVector(grids_Cu[i]), Val(N))
    @info "     SPH Interpolation: End copying interpolated grids back to host memory."
    return GridBundle(grids_result, names)
end
