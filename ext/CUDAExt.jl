module CUDAExt
using Dates
using CUDA
using PhantomRevealer

# Adapt structure
include(joinpath(@__DIR__, "CUDAExt", "AdaptStructure", "AdaptStructure.jl"))


"""
    PhantomRevealer.Greeting_CUDA()

A deliberately over-explained diagnostic utility that greets you from the CUDA
extension layer and prints an exhaustive snapshot of your CUDA environment.

# Overview
`Greeting_CUDA()` is intentionally simple: it serves one purpose—to prove that
`PhantomRevealer`’s CUDA extension is alive and able to communicate with
`CUDA.jl`.  
It does so by logging `"Hello from CUDAExt!"` and then calling
`CUDA.versioninfo()`, which dumps every detectable component of your CUDA stack.

# What It Actually Does
1. Emits an `@info` message confirming that the extension is active.
2. Invokes `CUDA.versioninfo()`, which queries:
   - Installed driver and runtime versions
   - CUDA toolkit path and compatibility
   - GPU device inventory with model names and compute capabilities
   - Presence and versions of developer utilities such as `nvcc`, `nvlink`,
     and `ptxas`
   - Host compiler configuration and environment variables

# Why It Exists
This function is the software equivalent of shouting into the void and hearing
an echo.  If you see version output, your extension linkage works.  
If you do not, something in the dependency chain (`CUDA.jl`, drivers, or
PhantomRevealer’s weak-dep mechanism) is broken.

# Behaviour and Guarantees
- **Inputs:** none.  
- **Outputs:** none (side effects only).  
- **Side effects:** console log and environment report.  
- **Performance:** near-zero cost aside from context initialization.  
- **State changes:** none. The call is read-only with respect to both
  PhantomRevealer and CUDA contexts.

# When To Use
- After installing `CUDA.jl`, to verify GPU accessibility.  
- During CI or cluster deployment, to record the CUDA stack configuration.  
- When filing bug reports, to attach reproducible environment data.  
- When you just need reassurance that something, somewhere, still speaks CUDA.

# Example
```julia
julia> using PhantomRevealer, CUDA
julia> PhantomRevealer.Greeting_CUDA()
[ Info: Hello from CUDAExt!
CUDA runtime 12.3, driver 550.40.07
Device 0: NVIDIA RTX 4090 (cc 8.9, 24576 MB)
...
```
This function has no return value because knowledge is its only reward.

Overwriting it during precompilation will anger Julia; declare the name in
the main module with function `Greeting_CUDA`` end to stay on her good side.

Calling it repeatedly will not speed up your GPU, but it will confirm that
logging still works, which is arguably more satisfying.
"""
function PhantomRevealer.Greeting_CUDA()
    t = hour(now())
    word = ""
    if 5 ≤ t < 12
        word = "Good morning"
    elseif 12 ≤ t < 18
        word = "Good afternoon"
    else
        word = "Good evening"
    end
    @info "$(word) from CUDAExt!"
    CUDA.versioninfo()
end

end