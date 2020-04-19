# This version is written for a shared memory implementation.
# The matrix A is local to the first Worker, which allocates work to other Workers
# All updates to A are carried out by the first Worker. Thus A is not distributed

using LinearAlgebra

function hpl_shared(A::Matrix, b::Vector, blocksize::Integer, run_parallel::Bool)

    n = size(A,1)
    A = [A b]

    if blocksize < 1
       throw(ArgumentError("hpl_shared: invalid blocksize: $blocksize < 1"))
    end

    B_rows = collect(range(0, n, step=blocksize))
    B_rows[end] = n
    B_cols = [B_rows; [n+1]]
    nB = length(B_rows)
    depend =  Array{Any}(undef, nB, nB)

    ## Small matrix case
    if nB <= 1
        x = A[1:n, 1:n] \ A[:,n+1]
        return x
    end

    ## Add a ghost row of dependencies to boostrap the computation
    for j=1:nB; depend[1,j] = true; end
    for i=2:nB, j=1:nB; depend[i,j] = false; end

    for i=1:(nB-1)
        #println("A=$A") #####
        ## Threads for panel factorizations
        I = (B_rows[i]+1):B_rows[i+1]
        K = I[1]:n
        A_KI = @view A[K,I]
        panel_p = panel_factor!(A_KI, depend[i,i])

        ## Write the factorized panel back to A
        A[K,I] = A_KI

        ## Panel permutation
        panel_p = K[panel_p]
        depend[i+1,i] = true

        ## Apply permutation from pivoting
        J = (B_cols[i+1]+1):B_cols[nB+1]
        A[K, J] = @view A[panel_p, J]
        ## Threads for trailing updates
        #L_II = tril(A[I,I], -1) + LinearAlgebra.I
        L_II = UnitLowerTriangular(@view A[I,I])
        K = (I[length(I)]+1):n
        A_KI = @view A[K,I]

        for j=(i+1):nB
            J = (B_cols[j]+1):B_cols[j+1]

            ## Do the trailing update (Compute U, and DGEMM - all flops are here)
            A_IJ = @view A[I,J]
            A_KJ = @view A[K,J]
            if run_parallel
                depend[i+1,j] = Threads.@spawn trailing_update!(L_II, A_IJ, A_KI, A_KJ,
                                                                depend[i+1,i], depend[i,j])
            else
                depend[i+1,j] = trailing_update!(L_II, A_IJ, A_KI, A_KJ,
                                                 depend[i+1,i], depend[i,j])
            end
        end

        # Wait for all trailing updates to complete, and write back to A
        for j=(i+1):nB
            J = (B_cols[j]+1):B_cols[j+1]
            if run_parallel
                (A_IJ, A_KJ) = fetch(depend[i+1,j])
            else
                (A_IJ, A_KJ) = depend[i+1,j]
            end
            A[I,J] = A_IJ
            A[K,J] = A_KJ
            depend[i+1,j] = true
        end

    end

    ## Completion of the last diagonal block signals termination
    @assert depend[nB, nB]

    ## Solve the triangular system
    x = triu(A[1:n,1:n]) \ A[:,n+1]

    return x

end ## hpl()


### Panel factorization ###

function panel_factor!(A_KI, col_dep)

    @assert col_dep

    ## Factorize a panel
    panel_p = lu!(A_KI).p

end ## panel_factor_par()


### Trailing update ###

function trailing_update!(L_II, A_IJ, A_KI, A_KJ, row_dep, col_dep)

    @assert row_dep
    @assert col_dep

    ## Compute blocks of U
    A_IJ = L_II \ A_IJ

    ## Trailing submatrix update - All flops are here
    if !isempty(A_KJ)
        m, k = size(A_KI)
        n = size(A_IJ,2)
        BLAS.gemm!('N','N',-1.0,A_KI,A_IJ,1.0,A_KJ)
        #A_KJ = A_KJ - A_KI*A_IJ
    end

    return (A_IJ, A_KJ)
end ## trailing_update_par()

hpl_shared(A::Matrix, b::Vector) = hpl_shared(A, b, max(1, div(maximum(size(A)),4)), true)

hpl_shared(A::Matrix, b::Vector, bsize::Integer) = hpl_shared(A, b, bsize, true)

a = rand(1000,1000);
b = rand(1000);

x=hpl_shared(a, b, 128);
@time x=hpl_shared(a, b, 128);

@show norm(a*x-b)
