

function GeneralGridInterpolation(grid :: GeneralGrid, input :: ITPINPUT, catalog :: InterpolationCatalog{N,G,Div,C,L}, itp_strategy :: InterpolationStrategy = itpSymmetric) where {N, G, Div, C, L, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}}
    # Counting total number of columns to interpolate
    order = catalog.ordered_names
    ncolumns = length(order)
    
    # Duplicate grid structure for storing each interpolated column
    grids = ntuple(_ -> similar(grid), Val(L))
    gridv = grid.coor

    # Constructing LBVH for neighbor search
    LBVH = LinearBVH!(input, CodeType=UInt64)

    # Kernel function
    multiplier = KernelFunctionValid(input.smoothed_kernel, T)
    hatemp = mean(input.h)
    radiustemp = multiplier * hatemp

    # Thread-local pools and stacks for LBVH queries
    pools = [zeros(Int, 2048) for _ in 1:nthreads()]
    stacks = [zeros(Int, 2048) for _ in 1:nthreads()]

    @threads for i in eachindex(gridv)
        # particle filtering and selection of smoothed radius
        point = gridv[i]
        thread_id = threadid()
        pool = pools[thread_id]
        stack = stacks[thread_id]
        if itp_strategy == itpSymmetric || itp_strategy == itpGather
            selection, ha = LBVH_query!(pool, stack, LBVH, point, multiplier, input.h)
        elseif itp_strategy == itpScatter
            selection = LBVH_query_scatter!(pool, stack, LBVH, point, radiustemp)
            ha = input.h[selection]
            radius = multiplier .* ha
            selection = LBVH_query_scatter!(pool, stack, LBVH, point, radius)
            ha = input.h[selection]
        else
            throw(ArgumentError("Unknown interpolation strategy: $(itp_strategy)"))
        end

        # SPH Interpolation
        ## scalar quantities


    end
        

    
    

    
end