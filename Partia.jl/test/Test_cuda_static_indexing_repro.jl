using CUDA
using Partia
using StaticArrays
using Test

@inline function lin_lut_static(q::T, Q::SVector{N,T}, I::SVector{N,T}) where {N,T<:AbstractFloat}
    dq = Q[2] - Q[1]
    idxf = q / dq + one(T)
    i = Int(clamp(Base.unsafe_trunc(Int32, idxf), Int32(1), Int32(N - 1)))
    t = idxf - T(i)
    return I[i] * (one(T) - t) + I[i + 1] * t
end

function runtime_static_index_kernel!(
    out_svec_first,
    out_tuple_first,
    out_svec_runtime,
    out_tuple_runtime,
    out_svec_interp,
    out_tuple_interp,
    out_cu_interp,
    qs,
    q_dev,
    i_dev,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(qs)
        q = @inbounds qs[idx]

        q_svec = SVector(0.0f0, 0.5f0, 1.0f0, 1.5f0)
        i_svec = SVector(1.0f0, 0.7f0, 0.2f0, 0.0f0)
        i_tup = (1.0f0, 0.7f0, 0.2f0, 0.0f0)

        idxf = q / 0.5f0 + 1.0f0
        i = Int(clamp(Base.unsafe_trunc(Int32, idxf), Int32(1), Int32(3)))
        t = idxf - Float32(i)

        @inbounds begin
            out_svec_first[idx] = i_svec[1]
            out_tuple_first[idx] = i_tup[1]
            out_svec_runtime[idx] = i_svec[i]
            out_tuple_runtime[idx] = i_tup[i]
            out_svec_interp[idx] = lin_lut_static(q, q_svec, i_svec)
            out_tuple_interp[idx] = i_tup[i] * (1.0f0 - t) + i_tup[i + 1] * t
            out_cu_interp[idx] = i_dev[i] * (1.0f0 - t) + i_dev[i + 1] * t
        end
    end
    return nothing
end

function package_line_lut_kernel!(out, qs)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(qs)
        q = @inbounds qs[idx]
        @inbounds out[idx] =
            Partia.KernelInterpolation.line_integrated_kernel_function_dimensionless(
                Partia.M4_spline,
                q,
            )
    end
    return nothing
end

function expected_outputs(qs::Vector{Float32})
    svec_first = fill(1.0f0, length(qs))
    tuple_first = fill(1.0f0, length(qs))
    svec_runtime = similar(qs)
    tuple_runtime = similar(qs)
    svec_interp = similar(qs)
    tuple_interp = similar(qs)
    cu_interp = similar(qs)

    q_host = Float32[0.0f0, 0.5f0, 1.0f0, 1.5f0]
    i_host = Float32[1.0f0, 0.7f0, 0.2f0, 0.0f0]

    for n in eachindex(qs)
        q = qs[n]
        idxf = q / 0.5f0 + 1.0f0
        i = Int(clamp(trunc(Int32, idxf), Int32(1), Int32(3)))
        t = idxf - Float32(i)

        svec_runtime[n] = i_host[i]
        tuple_runtime[n] = i_host[i]
        svec_interp[n] = i_host[i] * (1.0f0 - t) + i_host[i + 1] * t
        tuple_interp[n] = svec_interp[n]
        cu_interp[n] = svec_interp[n]
    end

    return (
        svec_first,
        tuple_first,
        svec_runtime,
        tuple_runtime,
        svec_interp,
        tuple_interp,
        cu_interp,
    )
end

function main()
    CUDA.functional(true)

    qs = Float32[0.25f0, 0.75f0, 1.25f0]
    q_dev = CuArray(Float32[0.0f0, 0.5f0, 1.0f0, 1.5f0])
    i_dev = CuArray(Float32[1.0f0, 0.7f0, 0.2f0, 0.0f0])

    out_svec_first = CUDA.zeros(Float32, length(qs))
    out_tuple_first = CUDA.zeros(Float32, length(qs))
    out_svec_runtime = CUDA.zeros(Float32, length(qs))
    out_tuple_runtime = CUDA.zeros(Float32, length(qs))
    out_svec_interp = CUDA.zeros(Float32, length(qs))
    out_tuple_interp = CUDA.zeros(Float32, length(qs))
    out_cu_interp = CUDA.zeros(Float32, length(qs))
    out_pkg_lut = CUDA.zeros(Float32, length(qs))

    @cuda threads=length(qs) blocks=1 runtime_static_index_kernel!(
        out_svec_first,
        out_tuple_first,
        out_svec_runtime,
        out_tuple_runtime,
        out_svec_interp,
        out_tuple_interp,
        out_cu_interp,
        CuArray(qs),
        q_dev,
        i_dev,
    )
    synchronize()

    @cuda threads=length(qs) blocks=1 package_line_lut_kernel!(out_pkg_lut, CuArray(qs))
    synchronize()

    got = (
        Array(out_svec_first),
        Array(out_tuple_first),
        Array(out_svec_runtime),
        Array(out_tuple_runtime),
        Array(out_svec_interp),
        Array(out_tuple_interp),
        Array(out_cu_interp),
        Array(out_pkg_lut),
    )
    expected = expected_outputs(qs)
    expected_pkg_lut = Float32[
        Partia.KernelInterpolation.line_integrated_kernel_function_dimensionless(
            Partia.M4_spline,
            q,
        ) for q in qs
    ]

    println("CUDA runtime indexing repro")
    println("svec_first    = ", got[1])
    println("tuple_first   = ", got[2])
    println("svec_runtime  = ", got[3])
    println("tuple_runtime = ", got[4])
    println("svec_interp   = ", got[5])
    println("tuple_interp  = ", got[6])
    println("cu_interp     = ", got[7])
    println("pkg_lut       = ", got[8])

    @testset "CUDA static aggregate runtime indexing" begin
        @test got[1] == expected[1]
        @test got[2] == expected[2]
        @test got[3] ≈ expected[3] atol = 0.0f0 rtol = 0.0f0
        @test got[4] ≈ expected[4] atol = 0.0f0 rtol = 0.0f0
        @test got[5] ≈ expected[5] atol = 1.0f-6 rtol = 1.0f-6
        @test got[6] ≈ expected[6] atol = 1.0f-6 rtol = 1.0f-6
        @test got[7] ≈ expected[7] atol = 1.0f-6 rtol = 1.0f-6
        @test got[8] ≈ expected_pkg_lut atol = 1.0f-6 rtol = 1.0f-6
    end
end

main()
