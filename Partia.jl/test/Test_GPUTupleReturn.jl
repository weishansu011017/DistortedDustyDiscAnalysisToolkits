using CUDA

# device function：真正回傳 NTuple{N,Float64}
@generated function dev_make_ntuple(::Val{N}, v::Float64) where {N}
    quote
        return ntuple(_ -> v, N)
    end
end

# kernel：呼叫 device function，將結果寫到 output array
function kernel_make_ntuple(out, ::Val{N}) where {N}
    i = threadIdx().x
    v = Float64(i)
    tup = dev_make_ntuple(Val(N), v)

    @inbounds for k in 1:N
        out[k] = tup[k]
    end
    return
end

# host function：測試某 N 是否成功
function test_return(N)
    println("Testing NTuple{$N,Float64}, size = $(N*8) bytes...")

    d_out = CUDA.zeros(Float64, N)

    try
        @cuda threads=1 kernel_make_ntuple(d_out, Val(N))
        CUDA.synchronize()
        out = Array(d_out)

        if length(out) == N
            println("  SUCCESS: First element = ", out[1])
        else
            println("  FAILED: Wrong output size!")
        end

    catch e
        println("  FAILED: ", typeof(e))
    end
end

function test_return()
    
    
end

# Run test for N = 1~32
for N in 1:32
    test_return(N)
end
