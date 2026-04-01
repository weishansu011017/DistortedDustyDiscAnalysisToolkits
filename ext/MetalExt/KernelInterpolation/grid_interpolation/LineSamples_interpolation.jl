@inline function _line_integrated_tables_Mtl()
    Q32 = x -> MtlVector{Float32}(Vector{Float32}(x))
    return (
        Q32(PhantomRevealer.KernelInterpolation._M4_spline_Q32),
        Q32(PhantomRevealer.KernelInterpolation._M4_spline_Inorm32),
        Q32(PhantomRevealer.KernelInterpolation._M5_spline_Q32),
        Q32(PhantomRevealer.KernelInterpolation._M5_spline_Inorm32),
        Q32(PhantomRevealer.KernelInterpolation._M6_spline_Q32),
        Q32(PhantomRevealer.KernelInterpolation._M6_spline_Inorm32),
        Q32(PhantomRevealer.KernelInterpolation._C2_Wendland_Q32),
        Q32(PhantomRevealer.KernelInterpolation._C2_Wendland_Inorm32),
        Q32(PhantomRevealer.KernelInterpolation._C4_Wendland_Q32),
        Q32(PhantomRevealer.KernelInterpolation._C4_Wendland_Inorm32),
        Q32(PhantomRevealer.KernelInterpolation._C6_Wendland_Q32),
        Q32(PhantomRevealer.KernelInterpolation._C6_Wendland_Inorm32),
    )
end


@inline function _lin_lut_Mtl(q::TF, Q::VF, I::VF) where {TF <: Float32, VF <: MtlDeviceVector{TF}}
    dq = Q[2] - Q[1]
    idxf = q / dq + 1.0f0
    i = Int(clamp(Base.unsafe_trunc(Int32, idxf), Int32(1), Int32(length(Q) - 1)))
    t = idxf - Float32(i)
    return I[i] * (1.0f0 - t) + I[i + 1] * t
end


@inline function _lookup_line_integrated_kernel_Mtl(::Type{M4_spline}, q_perp::TF, tables) where {TF <: Float32}
    return _lin_lut_Mtl(q_perp, tables[1], tables[2])
end
@inline function _lookup_line_integrated_kernel_Mtl(::Type{M5_spline}, q_perp::TF, tables) where {TF <: Float32}
    return _lin_lut_Mtl(q_perp, tables[3], tables[4])
end
@inline function _lookup_line_integrated_kernel_Mtl(::Type{M6_spline}, q_perp::TF, tables) where {TF <: Float32}
    return _lin_lut_Mtl(q_perp, tables[5], tables[6])
end
@inline function _lookup_line_integrated_kernel_Mtl(::Type{C2_Wendland}, q_perp::TF, tables) where {TF <: Float32}
    return _lin_lut_Mtl(q_perp, tables[7], tables[8])
end
@inline function _lookup_line_integrated_kernel_Mtl(::Type{C4_Wendland}, q_perp::TF, tables) where {TF <: Float32}
    return _lin_lut_Mtl(q_perp, tables[9], tables[10])
end
@inline function _lookup_line_integrated_kernel_Mtl(::Type{C6_Wendland}, q_perp::TF, tables) where {TF <: Float32}
    return _lin_lut_Mtl(q_perp, tables[11], tables[12])
end


@inline function _line_integrated_kernel_function_dimensionless_Mtl(::Type{K}, q_perp::TF, tables) where {K <: AbstractSPHKernel, TF <: Float32}
    q_perp ≥ KernelFunctionValid(K, TF) && return zero(TF)
    return _lookup_line_integrated_kernel_Mtl(K, q_perp, tables)
end


@inline function _line_integrated_kernel_function_Mtl(::Type{K}, r::TF, h::TF, tables) where {K <: AbstractSPHKernel, TF <: Float32}
    invh = inv(h)
    q_perp = r * invh
    I_dimless = _line_integrated_kernel_function_dimensionless_Mtl(K, q_perp, tables)
    return invh * I_dimless
end


@inline function _line_integrated_quantities_interpolate_kernel_Mtl(input::InterpolationInput{TF, VF, K, NCOLUMN}, origin::NTuple{3, TF}, direction::NTuple{3, TF}, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, tables, :: Type{itpScatter}) where {TF <: Float32, VF <: MtlDeviceVector{TF}, K <: AbstractSPHKernel, NCOLUMN, M}
    Kvalid = KernelFunctionValid(K, TF)

    output :: MVector{M, TF} = zero(MVector{M, TF})
    S1 :: TF = zero(TF)

    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: TF  = zero(TF)
    hb          :: TF  = zero(TF)

    NeighborSearch.@LBVH_scatter_line_traversal LBVH origin direction Kvalid leaf_idx p2leaf_d2 hb begin
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            W = _line_integrated_kernel_function_Mtl(K, Δr, hb, tables)
            S1b = mb * W / ρb
            S1 += S1b

            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += Ab * S1b
            end
        end
    end

    if iszero(S1)
        return ntuple(_ -> TF(NaN32), Val(M))
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        end
    end

    return NTuple{M, TF}(output)
end


@inline function _line_samples_interpolation_kernel!(grids :: NTuple{N, LineSamples{3}}, input :: ITPINPUT, catalog_consice :: InterpolationCatalogConcise{N, 0, 0, 0}, LBVH :: LinearBVH, tables, ::Type{itpScatter}) where {N, TF <: Float32, VF <: MtlDeviceVector{TF}, ITPINPUT <: InterpolationInput{TF, VF}}
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    npoints = length(grids[1])
    i = tid
    while i <= npoints
        @inbounds begin
            geometry = grids[1]

            xoa = geometry.origin[1][i]
            yoa = geometry.origin[2][i]
            zoa = geometry.origin[3][i]
            origin :: NTuple{3, TF} = (xoa, yoa, zoa)

            xda = geometry.direction[1][i]
            yda = geometry.direction[2][i]
            zda = geometry.direction[3][i]
            direction :: NTuple{3, TF} = (xda, yda, zda)
        end

        scalar_slots :: NTuple{N, Int} = catalog_consice.scalar_slots
        scalar_snormalization :: NTuple{N, Bool} = catalog_consice.scalar_snormalization
        scalars :: NTuple{N, TF} = _line_integrated_quantities_interpolate_kernel_Mtl(input, origin, direction, LBVH, scalar_slots, scalar_snormalization, tables, itpScatter)

        if N > 0
            @inbounds for j in 1:N
                grids[j].grid[i] = scalars[j]
            end
        end
        i += stride
    end
    return nothing
end

"""
    LineSamples_interpolation(backend::MetalComputeBackend, grid_template::LineSamples{D},
                              input::ITPINPUT, catalog::InterpolationCatalog{N, 0, 0, 0, N},
                              itp_strategy::Type{ITPSTRATEGY}=itpScatter)

Performs line-integrated SPH interpolation on the GPU using Metal.

This routine mirrors the CPU `LineSamples_interpolation` path: it only supports
scalar line-integrated quantities and only the `itpScatter` strategy.
"""
function PhantomRevealer.LineSamples_interpolation(::MetalComputeBackend, grid_template::LineSamples{D}, input::ITPINPUT, catalog::InterpolationCatalog{N, 0, 0, 0, N}, itp_strategy::Type{ITPSTRATEGY} = itpScatter) where {D, N, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    itp_strategy === itpScatter || throw(ArgumentError(
        "LineSamples_interpolation only supports itpScatter. " *
        "Line-integrated samples do not have a well-defined query smoothing length ha, " *
        "so itpGather and itpSymmetric are not supported."
    ))

    grids, LBVH, names, catalog_consice = PhantomRevealer.initialize_interpolation(PhantomRevealer.CPUComputeBackend(), grid_template, input, catalog)
    @info "     SPH Interpolation: Copying interpolated grids to device memory..."
    input_Mtl = to_MtlVector(input)
    grids_Mtl = ntuple(i -> to_MtlVector(grids[i]), Val(N))
    LBVH_Mtl = to_MtlVector(LBVH)
    tables_Mtl = _line_integrated_tables_Mtl()
    @info "     SPH Interpolation: End copying interpolated grids to device memory."

    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @metal threads=(256,) groups=(cld(npoints, 256)) _line_samples_interpolation_kernel!(grids_Mtl, input_Mtl, catalog_consice, LBVH_Mtl, tables_Mtl, itpScatter)
    Metal.synchronize()
    @info "     SPH Interpolation: End interpolation."
    @info "     SPH Interpolation: Copying interpolated grids back to host memory..."
    grids_result = ntuple(i -> PhantomRevealer.to_HostVector(grids_Mtl[i]), Val(N))
    @info "     SPH Interpolation: End copying interpolated grids back to host memory."
    return GridBundle(grids_result, names)
end
