module MetalExt
using Dates
using Metal
using Reexport
using Partia

# Adapt structure
include(joinpath(@__DIR__, "MetalExt", "AdaptStructure", "AdaptStructure.jl"))
@reexport using .AdaptStructure

# Grids
include(joinpath(@__DIR__, "MetalExt", "Grids", "Grids.jl"))
@reexport using .AdaptStructure

# Kernel interpolation
include(joinpath(@__DIR__, "MetalExt", "KernelInterpolation", "KernelInterpolation.jl"))
@reexport using .KernelInterpolation

"""
    Partia.Greeting_Metal()

A deliberately over-explained diagnostic utility that greets you from the Metal
extension layer and prints an exhaustive snapshot of your Metal environment.

# Overview
`Greeting_Metal()` is intentionally simple: it serves one purpose—to prove that
`Partia`’s Metal extension is alive and able to communicate with
`Metal.jl`.  
It does so by logging `"Hello from MetalExt!"` and then calling
`Metal.versioninfo()`, which dumps every detectable component of your Metal stack.

# What It Actually Does
1. Emits an `@info` message confirming that the extension is active.
2. Invokes `Metal.versioninfo()`, which queries:
   - Installed driver and runtime versions
   - Metal toolkit path and compatibility
   - GPU device inventory with model names and compute capabilities
   - Presence and versions of developer utilities such as `nvcc`, `nvlink`,
     and `ptxas`
   - Host compiler configuration and environment variables

# Why It Exists
This function is the software equivalent of shouting into the void and hearing
an echo.  If you see version output, your extension linkage works.  
If you do not, something in the dependency chain (`Metal.jl`, drivers, or
Partia’s weak-dep mechanism) is broken.

# Behaviour and Guarantees
- **Inputs:** none.  
- **Outputs:** none (side effects only).  
- **Side effects:** console log and environment report.  
- **Performance:** near-zero cost aside from context initialization.  
- **State changes:** none. The call is read-only with respect to both
  Partia and Metal contexts.

# When To Use
- After installing `Metal.jl`, to verify GPU accessibility.  
- During CI or cluster deployment, to record the Metal stack configuration.  
- When filing bug reports, to attach reproducible environment data.  
- When you just need reassurance that something, somewhere, still speaks Metal.

# Example
```julia
julia> using Partia, Metal
julia> Partia.Greeting_Metal()
[ Info: Hello from MetalExt!
macOS 14.5.0, Darwin 23.5.0
1 device:
- Apple M2 10 GPU cores (64.000 KiB allocated)
...
```
This function has no return value because knowledge is its only reward.

Overwriting it during precompilation will anger Julia; declare the name in
the main module with function `Greeting_Metal`` end to stay on her good side.

Calling it repeatedly will not speed up your GPU, but it will confirm that
logging still works, which is arguably more satisfying.
"""
function Partia.Greeting_Metal()
    t = hour(now())
    word = ""
    if 5 ≤ t < 12
        word = "Good morning"
    elseif 12 ≤ t < 18
        word = "Good afternoon"
    else
        word = "Good evening"
    end

    @info "$(word) from MetalExt!"
    Metal.versioninfo()
end

end