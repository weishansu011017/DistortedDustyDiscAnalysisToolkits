# StreamingInstability.jl

`StreamingInstability.jl` implements a classical linear streaming-instability growth-rate solver for dust-gas mixtures in protoplanetary discs, following the method of [Chen & Lin (2020)](https://doi.org/10.3847/1538-4357/ab76ca).  The package supports both single-mode evaluation and growth-rate maps over a wavenumber grid, and uses [`TinyEigvals.jl`](https://github.com/weishansu011017/TinyEigvals.jl) for the small dense eigenvalue problems. Currently only the classical single-species linear growth-rate solver is implemented.



## Installation

`StreamingInstability.jl` is not registered in the General registry. If you want to install it directly from this repository, use

```julia
using Pkg
Pkg.add(url="https://github.com/weishansu011017/DistortedDustyDiscAnalysis.git", subdir="StreamingInstability.jl")
```

If you already have this repository locally and want a development checkout, use

```julia
using Pkg
Pkg.develop(path="path/to/DistortedDustyDiscAnalysis/StreamingInstability.jl")
```



## Growth rate in classical streaming instability

For the classical streaming instability described by [Youdin & Goodman (2005)](https://iopscience.iop.org/article/10.1086/426895), `StreamingInstability.jl` exposes the callable struct `ClassicalSIGrowthRateInput`. Its constructor takes the Stokes number `St`, the midplane gas and dust densities `ρg` and `ρd`, and the equilibrium gas and dust velocities `(vxlcs, vylcs, ωxlcs, ωylcs)`, all expressed in sound-speed units. Calling this object as `input(Κx, Κz)` evaluates the dimensionless growth rate for radial and vertical wavenumbers in scale-height units, `Κx = kx H` and `Κz = kz H`. Since many SI benchmark tables instead quote wavenumbers in `ηr` units, those values must first be converted using `ηvK/cs = η / (H/r)`, as shown in the example below.

```julia
using StreamingInstability

# Example: the linA benchmark in Youdin & Johansen (2007)
h_over_r = 0.05
eta = 0.0025
etavk_over_cs = eta / h_over_r

St = 0.1
eps = 3.0
rho_g = 1.0
rho_d = eps * rho_g

# Equilibrium velocities (Nakagawa, Sekiya, & Hayashi 1986; doi:10.1016/0019-1035(86)90121-1)
Δ = (1 + eps)^2 + St^2
vxlcs =  etavk_over_cs * (2 * eps * St) / Δ
vylcs = -etavk_over_cs * (1 + eps * St^2 / Δ) / (1 + eps)
wxlcs = -etavk_over_cs * (2 * St) / Δ
wylcs = -etavk_over_cs * (1 - St^2 / Δ) / (1 + eps)

# Build the linearised SI input
input = ClassicalSIGrowthRateInput(St, rho_g, rho_d, vxlcs, vylcs, wxlcs, wylcs)

# Single-mode growth rate
# Literature values: Kx = Kz = 30 in ηr
Kx_eta_r = 30.0
Kz_eta_r = 30.0
Κx = Kx_eta_r / etavk_over_cs                    # Convert from ηr units to scale-height units
Κz = Kz_eta_r / etavk_over_cs                    # Solver input expects Κ = kH
s = input(Κx, Κz)                                # Expected: s ≈ 0.4190204 for the linA benchmark

# Growth-rate map on a literature grid, converted from ηr units to scale-height units
Kx_eta_r_grid = range(10.0, 100.0, length = 32)
Kz_eta_r_grid = range(10.0, 100.0, length = 32)
Κxs = collect(Kx_eta_r_grid ./ etavk_over_cs)    # Convert from ηr units to scale-height units
Κzs = collect(Kz_eta_r_grid ./ etavk_over_cs)    # Convert from ηr units to scale-height units
growth = input(Κxs, Κzs)                         # Matrix of growth rates with size (length(Κxs), length(Κzs))
```



This implementation is tested against four published benchmark problems: linA and linB from [Youdin & Johansen (2007)](https://doi.org/10.1086/516729), Table 1, and linC and linD from [Bai & Stone (2010)](https://doi.org/10.1088/0067-0049/190/2/297), Table 2. For the linA example above, the expected result is `s ≈ 0.4190204`. In the literature sign convention, the reported growth rate is `Im(ω/Ω)`; in this package, the solver returns the equivalent quantity as the maximum real part of the eigenvalues. For a more complete set of benchmark cases and reference values, see [`test/growthrate_classicalSI.jl`](./test/growthrate_classicalSI.jl).



## GPU capability

The core eigensolver in [`TinyEigvals.jl`](https://github.com/weishansu011017/TinyEigvals.jl) is designed for small fixed-size complex matrices and works naturally with [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl). Since `StreamingInstability.jl` formulates the classical SI problem as a type-stable 8x8 eigenvalue problem, its core workflow is compatible with GPU-oriented use cases where the required inputs and small matrices are already available on device. See [`TinyEigvals.jl`](https://github.com/weishansu011017/TinyEigvals.jl) for further details on device-side usage and limitations.



## References

Bai X.-N., Stone J. M., 2010, ApJS, 190, 297, [doi:10.1088/0067-0049/190/2/297](https://doi.org/10.1088/0067-0049/190/2/297)

Chen K., Lin M.-K., 2020, ApJ, 891, 132, [doi:10.3847/1538-4357/ab76ca](https://doi.org/10.3847/1538-4357/ab76ca)

Nakagawa Y., Sekiya M., Hayashi C., 1986, Icarus, 67, 375, [doi:10.1016/0019-1035(86)90121-1](https://doi.org/10.1016/0019-1035(86)90121-1)

Youdin A. N., Goodman J., 2005, ApJ, 620, 459, [doi:10.1086/426895](https://doi.org/10.1086/426895)

Youdin A. N., Johansen A., 2007, ApJ, 662, 613, [doi:10.1086/516729](https://doi.org/10.1086/516729)
