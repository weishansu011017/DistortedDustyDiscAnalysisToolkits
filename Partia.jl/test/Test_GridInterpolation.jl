using DataFrames
using Statistics
using Partia
# using Metal
using CUDA

@inline ρ_sample(x :: T, y :: T, z :: T) where {T <:AbstractFloat} = 1.0 + 0.2 * sin(π * x) * cos(π * y)
@inline vx_sample(x :: T, y :: T, z :: T; a :: T = 1.0) where {T <:AbstractFloat} = a * x + a
@inline vy_sample(x :: T, y :: T, z :: T; b :: T = 2.0) where {T <:AbstractFloat} = b * y + b
@inline vz_sample(x :: T, y :: T, z :: T; c :: T = 3.0) where {T <:AbstractFloat} = c * z + c
@inline divv_sample(x :: T, y :: T, z :: T; a :: T = 1.0, b :: T = 2.0, c :: T = 3.0) where {T <:AbstractFloat} = a + b + c
function generate_dataframe(nx, ny, nz; hfact = 1.2)
    x = range(0, 1; length = nx)
    y = range(0, 1; length = ny)
    z = range(0, 1; length = nz)
    X = repeat(x, inner = ny*nz)
    Y = repeat(repeat(y, inner = nz), outer = nx)
    Z = repeat(z, outer = nx*ny)
    N = length(X)
    mass = 1/N
    # 建議：密度改成永遠 > 0
    rho = ρ_sample.(X, Y, Z)
    vx = vx_sample.(X, Y, Z, a = 1.0)
    vy = vy_sample.(X, Y, Z, b = 2.0)
    vz = vz_sample.(X, Y, Z, c = 3.0)
    divv = divv_sample.(X, Y, Z, a = 1.0, b = 2.0, c = 3.0)
    h = similar(rho)
    @inbounds for i in eachindex(rho)
        h[i] = hfact * (mass / rho[i])^(1/3)
    end
    df = DataFrame(
        x = X, y = Y, z = Z,
        rho = rho,
        vx = vx, vy = vy, vz = vz,
        divv = divv,
        h = h
    )
    return df, mass
end
function main(kernel :: Type{K}, backend :: B = CPUBackend(), n_red :: Int = 3) where {K <: AbstractSPHKernel, B <: AbstractExecutionBackend}
    # Test target: [vx, vy, vz, rho, divv]
    @info "-------------------------------- kernel: $(kernel) --------------------------------"
    hfact = 1.2
    df, mass = generate_dataframe(300, 300, 300, hfact = hfact)
    params = Dict{Symbol, Any}()
    params[:mass] = mass
    datag = PartiaDataFrame(df, params)
    input, catalog = build_input(datag, MassFromParams(:mass), scalars=[:rho, :vx, :vy, :vz], divergences=[:v], smoothed_kernel=kernel);
    testaxis = collect(LinRange(0.35, 0.65, 150))
    rho = ρ_sample.(testaxis, testaxis, testaxis)
    vx = vx_sample.(testaxis, testaxis, testaxis, a = 1.0)
    vy = vy_sample.(testaxis, testaxis, testaxis, b = 2.0)
    vz = vz_sample.(testaxis, testaxis, testaxis, c = 3.0)
    divv = divv_sample.(testaxis, testaxis, testaxis, a = 1.0, b = 2.0, c = 3.0)
    grid = GeneralGrid(testaxis, testaxis, testaxis);
    griditpresult = GeneralGrid_interpolation(backend, grid, input, catalog, itpGather, n_red = n_red);
    rho_err_abs = similar(rho)
    vx_err_abs = similar(vx)
    vy_err_abs = similar(vy)
    vz_err_abs = similar(vz)
    divv_err_abs = similar(divv)
    @inbounds @simd for i in eachindex(rho_err_abs, vx_err_abs, vy_err_abs, vz_err_abs, divv_err_abs)
        @inbounds begin
            rho_real = rho[i]
            vx_real = vx[i]
            vy_real = vy[i]
            vz_real = vz[i]
            divv_real = divv[i]
            rho_itp = griditpresult.grids[1].grid[i]
            vx_itp = griditpresult.grids[2].grid[i]
            vy_itp = griditpresult.grids[3].grid[i]
            vz_itp = griditpresult.grids[4].grid[i]
            divv_itp = griditpresult.grids[5].grid[i]
        end
        Δrho = abs((rho_real - rho_itp)/rho_real)
        Δvx = abs(sqrt((vx_real - vx_itp)^2)/vx_real)
        Δvy = abs(sqrt((vy_real - vy_itp)^2)/vy_real)
        Δvz = abs(sqrt((vz_real - vz_itp)^2)/vz_real)
        Δdivv = abs(sqrt((divv_real - divv_itp)^2)/divv_real)
        @inbounds begin
            rho_err_abs[i] = Δrho * 100
            vx_err_abs[i] = Δvx * 100
            vy_err_abs[i] = Δvy * 100
            vz_err_abs[i] = Δvz * 100
            divv_err_abs[i] = Δdivv * 100
        end
    end
    # print
    # @info "====================== ρ ========================"
    # @info "Maximum Error: $(maximum(rho_err_abs)) %"
    # @info "Minimum Error: $(minimum(rho_err_abs)) %"
    # @info "Mean Error: $(mean(rho_err_abs)) %"
    # @info "Median Error: $(median(rho_err_abs)) %"
    # @info "std Error: $(median(rho_err_abs)) %"
    # @info "====================== vx ======================="
    # @info "Maximum Error: $(maximum(vx_err_abs)) %"
    # @info "Minimum Error: $(minimum(vx_err_abs)) %"
    # @info "Mean Error: $(mean(vx_err_abs)) %"
    # @info "Median Error: $(median(vx_err_abs)) %"
    # @info "std Error: $(median(vx_err_abs)) %"
    # @info "====================== vy ======================="
    # @info "Maximum Error: $(maximum(vy_err_abs)) %"
    # @info "Minimum Error: $(minimum(vy_err_abs)) %"
    # @info "Mean Error: $(mean(vy_err_abs)) %"
    # @info "Median Error: $(median(vy_err_abs)) %"
    # @info "std Error: $(median(vy_err_abs)) %"
    # @info "====================== vz ======================="
    # @info "Maximum Error: $(maximum(vz_err_abs)) %"
    # @info "Minimum Error: $(minimum(vz_err_abs)) %"
    # @info "Mean Error: $(mean(vz_err_abs)) %"
    # @info "Median Error: $(median(vz_err_abs)) %"
    # @info "std Error: $(median(vz_err_abs)) %"
    # @info "===================== divv ======================="
    # @info "Maximum Error: $(maximum(divv_err_abs)) %"
    # @info "Minimum Error: $(minimum(divv_err_abs)) %"
    # @info "Mean Error: $(mean(divv_err_abs)) %"
    # @info "Median Error: $(median(divv_err_abs)) %"
    # @info "std Error: $(median(divv_err_abs)) %"
    # @info "=================================================="
    @info "-----------------------------------------------------------------------------------"
    return griditpresult
end
backend = CUDAComputeBackend()
main(M4_spline, backend, 0)
main(M4_spline, backend, 0)
main(M4_spline, backend, 1)
main(M4_spline, backend, 2)
main(M4_spline, backend, 3)
main(M4_spline, backend, 4)
main(M4_spline, backend, 5)
main(M4_spline, backend, 6)
main(M4_spline, backend, 7)
main(M4_spline, backend, 8)
main(M4_spline, backend, 9)
main(M4_spline, backend, 10)
main(M4_spline, backend, 11)
main(M4_spline, backend, 12)
main(M4_spline, backend, 13)
main(M4_spline, backend, 14)
main(M4_spline, backend, 15)
main(M4_spline, backend, 16)
main(M4_spline, backend, 32)
# main(M5_spline, backend)
# main(M6_spline, backend)
# main(C2_Wendland, backend)
# main(C4_Wendland, backend)
# main(C6_Wendland, backend)