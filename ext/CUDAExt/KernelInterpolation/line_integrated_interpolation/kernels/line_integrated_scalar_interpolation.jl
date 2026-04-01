@inline function _line_samples_interpolation_kernel!(
    grids::NTuple{N, LineSamples{3}},
    input::ITPINPUT,
    catalog_consice::InterpolationCatalogConcise{N, 0, 0, 0},
    LBVH::LinearBVH,
    ::Type{itpScatter},
) where {N, TF <: AbstractFloat, ITPINPUT <: InterpolationInput{TF}}
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
                PhantomRevealer.KernelInterpolation._line_integrated_quantities_interpolate_kernel(
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
