# Partia.jl

`Partia.jl` is the Smoothed Particle Hydrodynamics (SPH) interpolation and grid-analysis package. It provides the kernel, neighbour-search, interpolation, grid, and structured-output machinery.



## Installation

`Partia.jl` is not registered in the General registry. If you want to install it directly from this repository, use

```julia
using Pkg
Pkg.add(url="https://github.com/weishansu011017/DistortedDustyDiscAnalysis.git", subdir="Partia.jl")
```

If you already have this repository locally and want a development checkout, use

```julia
using Pkg
Pkg.develop(path="path/to/DistortedDustyDiscAnalysis/Partia.jl")
```



## Functionality

`Partia.jl` provides SPH-based interpolation of particle-carried quantities onto arbitrary sampling points and grids, following the standard SPH formalism described in [Price 2007](https://doi.org/10.1071/AS07022) and [Price 2012](https://doi.org/10.1016/j.jcp.2010.12.011). In particular, the derivative operators use the error-reduced SPH difference form summarized in [Price 2012](https://doi.org/10.1016/j.jcp.2010.12.011).

For a sampling position $\mathbf{r}$, the density estimate is

$$
\rho(\mathbf{r})
=
\sum_b m_b \, W\!\left(\lvert \mathbf{r} - \mathbf{r}_b \rvert, h\right),
$$


and a particle-carried scalar quantity $A_b$ is interpolated as

$$
A(\mathbf{r})
=
\sum_b \frac{m_b}{\rho_b} A_b \,
W\!\left(\lvert \mathbf{r} - \mathbf{r}_b \rvert, h\right).
$$


When Shepard normalization is enabled, the interpolated value becomes

$$
\tilde{A}(\mathbf{r})
=
\frac{
\sum_b \frac{m_b}{\rho_b} A_b \,
W\!\left(\lvert \mathbf{r} - \mathbf{r}_b \rvert, h\right)
}{
\sum_b \frac{m_b}{\rho_b}
W\!\left(\lvert \mathbf{r} - \mathbf{r}_b \rvert, h\right)
}.
$$

For a vector field $\mathbf{A}$, the Shepard-normalized value $\tilde{\mathbf{A}}(\mathbf{r})$ is obtained by applying the same normalization component-wise.

The interpolation formula is the same in all cases; the only difference is how the smoothing length used in the kernel evaluation is chosen. `Partia.jl` supports gather, scatter, and symmetric evaluation:

$$
W_{\mathrm{gather}}(\mathbf{r}, \mathbf{r}_b)
=
W\!\left(\lvert \mathbf{r} - \mathbf{r}_b \rvert, h_{\mathrm{samp}}\right),
$$

$$
W_{\mathrm{scatter}}(\mathbf{r}, \mathbf{r}_b)
=
W\!\left(\lvert \mathbf{r} - \mathbf{r}_b \rvert, h_b\right),
$$

$$
W_{\mathrm{symmetric}}(\mathbf{r}, \mathbf{r}_b)
=
\frac{1}{2}
\left[
W\!\left(\lvert \mathbf{r} - \mathbf{r}_b \rvert, h_{\mathrm{samp}}\right)
+
W\!\left(\lvert \mathbf{r} - \mathbf{r}_b \rvert, h_b\right)
\right].
$$

Here $h_{\mathrm{samp}}$ is the smoothing length assigned to the sampling point and $h_b$ is the smoothing length carried by particle $b$.

Spatial derivatives are evaluated using the error-reduced SPH difference form of [Price 2012](https://doi.org/10.1016/j.jcp.2010.12.011). For a scalar field $A$,

$$
\nabla A(\mathbf{r})
=
\sum_b
\frac{m_b}{\rho_b}
\left(A_b - \tilde{A}(\mathbf{r})\right)
\nabla W\!\left(\mathbf{r} - \mathbf{r}_b, h\right).
$$

For density,

$$
\nabla \rho(\mathbf{r})
=
\sum_b m_b \nabla W\!\left(\mathbf{r} - \mathbf{r}_b, h\right)
-
\rho(\mathbf{r})
\sum_b \frac{m_b}{\rho_b}
\nabla W\!\left(\mathbf{r} - \mathbf{r}_b, h\right).
$$

For a vector field $\mathbf{A}$,

$$
\nabla \cdot \mathbf{A}(\mathbf{r})
=
\sum_b
\frac{m_b}{\rho_b}
\left(\mathbf{A}_b - \tilde{\mathbf{A}}(\mathbf{r})\right)
\cdot
\nabla W\!\left(\mathbf{r} - \mathbf{r}_b, h\right),
$$

$$
\nabla \times \mathbf{A}(\mathbf{r})
=
-
\sum_b
\frac{m_b}{\rho_b}
\left(\mathbf{A}_b - \tilde{\mathbf{A}}(\mathbf{r})\right)
\times
\nabla W\!\left(\mathbf{r} - \mathbf{r}_b, h\right).
$$

This derivative form reduces zeroth-order gradient errors and guarantees that the derivative vanishes for a constant field.

In addition to point sampling, `Partia.jl` also supports line-integrated interpolation through pre-tabulated line-integrated kernels, which is useful for projected quantities such as column density.

Currently the package provides:

- Point-sampled interpolation of scalar and vector quantities
- Shepard-normalized interpolation
- Gradient, divergence, and curl operators
- Structured-grid sampling
- Line-sampled / line-integrated interpolation

The current interpolation pipeline uses a **Linear Bounding Volume Hierarchy (LBVH)** acceleration structure, inspired by *SHAMROCK* ([David-Cleris et al. 2025](https://academic.oup.com/mnras/article/539/1/1/8085154)) (see [Lauterbach et al. (2009)](https://doi.org/10.1111/j.1467-8659.2009.01377.x) and [Karras (2012)](https://doi.org/10.2312/EGGH/HPG12/033-037) for more information). In particular, traversal of this LBVH follows the **stackless DFS traversal** presented by [Prokopenko & Lebrun-Grandié (2024)](https://doi.org/10.2172/2301619).



At present, the interpolation routines in `Partia.jl` are implemented for **3D data**. The structured-grid interpolation path therefore currently accepts `StructuredGrid{3}` templates only.



The computation proceeds through the following stages:

1. Materializing the particle-side interpolation input.

   Particle coordinates, smoothing lengths, densities, masses, and any extra quantity columns are packed into an `InterpolationInput`. The requested scalar, gradient, divergence, and curl outputs are then collected into an `InterpolationCatalog`, which records both column locations and output ordering. In practice, `build_input()` constructs both objects together.

2. Preparing the output containers.

   The chosen sampling geometry, such as `PointSamples`, `LineSamples`, or `StructuredGrid`, acts as the template for the output. During `initialize_interpolation`, one output container is allocated for each requested result field, and the full catalog is reduced to a concise execution-oriented form.

3. Constructing the LBVH over the particle distribution.

   The particle coordinates stored in the `InterpolationInput` are used to build a `LinearBVH`, which becomes the main neighbour-search structure for all subsequent interpolation calls.

4. Converting the sampling geometry into the execution form.

   `PointSamples` and `LineSamples` are used directly. `StructuredGrid` is first flattened into a `PointSamples` representation so that the same point-wise interpolation kernels can be reused, and is restored to structured form after interpolation finishes.

5. Determining the local interpolation geometry for each sample.

   For point samples, the code extracts the sample coordinates and, for gather or symmetric interpolation, estimates a query smoothing length through `LBVH_find_nearest_h`. For line samples, only `itpScatter` is supported, since there is no well-defined sample-side smoothing length for a line-integrated query.

6. Evaluating the single-sample interpolation kernel.

   Each sample is evaluated independently. Point samples dispatch to `_general_quantity_interpolate_kernel`, which accumulates the requested scalar, gradient, divergence, and curl quantities according to the concise catalog and its Shepard-normalization flags. Line samples dispatch to `_line_integrated_quantities_interpolate_kernel`, which evaluates line-integrated scalar quantities using particle-side smoothing lengths. The enclosing sample loop is then executed by the selected backend (`CPUComputeBackend`, `CUDAComputeBackend`, `MetalComputeBackend`).

7. Writing results back into the output grids.

   The interpolated values are stored into the preallocated output containers in the order prescribed by the catalog. For structured grids, the flattened outputs are finally reshaped back into `StructuredGrid` objects before the `GridBundle` is returned.



## Example

Assume the particle data are already available as arrays or in a dictionary-like container:

```julia
using Partia

particles = Dict(
    "x" => x,
    "y" => y,
    "z" => z,
    "m" => m,
    "h" => h,
    "ρ" => ρ,
    "u" => u,
    "vx" => vx,
    "vy" => vy,
    "vz" => vz,
)

backend = CPUComputeBackend()

x = particles["x"]
y = particles["y"]
z = particles["z"]
m = particles["m"]
h = particles["h"]
ρ = particles["ρ"]
u = particles["u"]
vx = particles["vx"]
vy = particles["vy"]
vz = particles["vz"]
```

To interpolate **internal energy**, the **velocity components**, the **density gradient**, the **divergence of velocity**, and the curl of velocity, first construct the interpolation input and catalog:

```julia
input, catalog = build_input(
    backend,
    x,
    y,
    z,
    m,
    h,
    ρ,
    (ρ, u, vx, vy, vz);
    column_names = (:ρ, :u, :vx, :vy, :vz),
    scalars = (:u, :vx, :vy, :vz),
    gradients = (:rho,),
    divergences = (:v,),
    curls = (:v,),
)
```

Here `:v` is resolved as the vector field `(:vx, :vy, :vz)`. The density column is also included in `quantity_columns` so that `:rho` can be requested as an interpolated output quantity and as a gradient target through the catalog interface.

For this particular catalog, the output order is fixed as:

1. `u`
2. `vx`
3. `vy`
4. `vz`
5. `∇ρˣ`
6. `∇ρʸ`
7. `∇ρᶻ`
8. `∇⋅v`
9. `(∇×v)ˣ`
10. `(∇×v)ʸ`
11. `(∇×v)ᶻ`

The returned `GridBundle` follows exactly this catalog order.

### Example 1 - Cartesian 3D grid

To sample the particle data onto a Cartesian structured grid:

```julia
grid_template = StructuredGrid(
    Cartesian,
    (-10.0, 10.0, 128),
    (-10.0, 10.0, 128),
    (-2.0, 2.0, 64),
)

result = StructuredGrid_interpolation(
    backend,
    Cartesian,
    grid_template,
    input,
    catalog,
    itpSymmetric,
)

u_grid = result.grids[1]
```

Here `result` is a `GridBundle`, and each entry of `result.grids` is a `StructuredGrid` with the same axes as `grid_template`.

### Example 2 - Cylindrical structured grid

For non-Cartesian 3D structured grids, pass the coordinate-system tag explicitly:

```julia
cyl_template = StructuredGrid(
    Cylindrical,
    (1.0, 50.0, 256),
    (0.0, 2π, 512),
    (-5.0, 5.0, 128),
)

cyl_result = StructuredGrid_interpolation(
    backend,
    Cylindrical,
    cyl_template,
    input,
    catalog,
    itpSymmetric,
)
```

In this case the cylindrical sample coordinates are converted to Cartesian positions before the SPH interpolation kernel is evaluated, while the output grids are restored on the original cylindrical axes.

### Example 3 - Particle-wise postprocessing

To evaluate interpolated quantities at an arbitrary set of sample points, use `PointSamples`:

```julia
sample_points = PointSamples(x, y, z)

point_result = PointSamples_interpolation(
    backend,
    sample_points,
    input,
    catalog,
    itpSymmetric,
)

u_at_points = point_result.grids[1].grid
```

This is useful for particle-wise postprocessing, field probes, or consistency checks against existing particle data.

### Example 4 - Column density and line-integrated quantities

Line-integrated interpolation currently supports scalar outputs only and uses `itpScatter`:

```julia
line_catalog = InterpolationCatalog(
    (:rho, :u), Val(3);
    scalars = (:rho, :u),
)

lines = LineSamples(
    x_origin,
    y_origin,
    z_origin,
    x_direction,
    y_direction,
    z_direction,
)

line_result = LineSamples_interpolation(
    backend,
    lines,
    input,
    line_catalog,
    itpScatter,
)

Sigma = line_result.grids[1].grid
u_column = line_result.grids[2].grid
```

Since `line_catalog` is constructed with `scalars = (:rho, :u)`, the output order is exactly `(:rho, :u)`, so `line_result.grids[1]` is the line-integrated density and `line_result.grids[2]` is the line-integrated internal energy.

This is the intended path for projected quantities such as column density or line-integrated scalar diagnostics.



## Grid IO

`Partia.jl` provides HDF5-based serialization for `GridBundle` and `GridDataset`.

The main entry points are:

- `write_GridBundle(gb, filename; code_units, operation_name="", params=nothing)`
- `write_GridDataset(gd, filename)`
- `read_GridDataset(filename)`

`write_GridBundle` is the usual high-level path. It wraps the `GridBundle` into a `GridDataset`, records metadata such as the floating-point type, grid type, code units, file identifier, and any extra user parameters, and then writes the dataset to disk.

`write_GridDataset` writes the dataset in HDF5 form. For `StructuredGrid` outputs, the shared axes are stored once under `data/axes/`, while for `PointSamples` outputs the shared coordinates are stored once under `data/coord/`. In both cases, the numerical grid arrays are written under `data/grids/`, and the quantity names are written under `data/names`.

All grids in a written dataset must share the same geometric vectors by identity:

- `StructuredGrid` datasets require all grids to share the same `axes`
- `PointSamples` datasets require all grids to share the same `coor`

If this sharing assumption is violated, `write_GridDataset` throws an `ArgumentError`.

A typical workflow is:

```julia
write_GridBundle(
    result,
    "partia_output.h5";
    code_units = Dict(
        :umass => 1.0,
        :udist => 1.0,
        :utime => 1.0,
        :umagfd => 1.0,
    ),
    operation_name = "demo_interpolation",
)

gd = read_GridDataset("partia_output.h5")

loaded_names = gd.data.names
loaded_grids = gd.data.grids
```

At present, `read_GridDataset` reconstructs `StructuredGrid` and `PointSamples` datasets from files written by the corresponding Partia writers.



## GPU capability

`Partia.jl` provides GPU execution through Julia package extensions for [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) and [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl). These are weak dependencies, so the corresponding extension is activated when the backend package is loaded.

The available execution backends are:

- `CPUComputeBackend()`
- `CUDAComputeBackend()`
- `MetalComputeBackend()`

The interpolation interface is unchanged across backends. For example, a CUDA run uses

```julia
using Partia
using CUDA

backend = CUDAComputeBackend()

result = PointSamples_interpolation(
    backend,
    sample_points,
    input,
    catalog,
    itpSymmetric,
)
```

At present, GPU execution is implemented for the 3D interpolation pipeline:

- `PointSamples_interpolation`
- `LineSamples_interpolation`
- `StructuredGrid_interpolation`, which flattens the structured grid to `PointSamples`, dispatches through the selected backend, and restores the structured output afterwards

The extension layers also provide explicit data-movement helpers:

- `to_CuVector`
- `to_MtlVector`
- `to_HostVector`

In the current implementation, GPU interpolation returns its results to host memory before assembling the final `GridBundle`. CUDA preserves the working floating-point type, while the Metal path currently materializes device-side data as `Float32` and therefore returns `Float32` host grids.



## References

David-Cleris T., Laibe G., Lapeyre Y., 2025, MNRAS, 539, 1, [doi:10.1093/mnras/staf444](https://doi.org/10.1093/mnras/staf444)

Karras T., 2012, in High Performance Graphics, p. 33, [doi:10.2312/EGGH/HPG12/033-037](https://doi.org/10.2312/EGGH/HPG12/033-037)

Lauterbach C., Garland M., Sengupta S., Luebke D., Manocha D., 2009, Comput. Graph. Forum, 28, 375, [doi:10.1111/j.1467-8659.2009.01377.x](https://doi.org/10.1111/j.1467-8659.2009.01377.x)

Price D. J., 2007, Publ. Astron. Soc. Aust., 24, 159, [doi:10.1071/AS07022](https://doi.org/10.1071/AS07022)

Price D. J., 2012, J. Comput. Phys., 231, 759, [doi:10.1016/j.jcp.2010.12.011](https://doi.org/10.1016/j.jcp.2010.12.011)

Prokopenko A., Lebrun-Grandie D., 2024, ORNL/TM-2024/3259, [doi:10.2172/2301619](https://doi.org/10.2172/2301619)
