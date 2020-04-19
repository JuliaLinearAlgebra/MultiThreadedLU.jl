### using DArrays ###

function hpl_par2(A::Matrix, b::Vector)
    n = size(A,1)
    A = [A b]

    C = distribute(A, 2)
    nB = length(C.pmap)

    ## case if only one processor
    if nB <= 1
        x = A[1:n, 1:n] \ A[:,n+1]
        return x
    end

    depend = Array(RemoteRef, nB, nB)

    #pmap[i] is where block i's stuff is
    #block i is dist[i] to dist[i+1]-1
    for i = 1:nB
        #println("C=$(convert(Array, C))") #####
        ##panel factorization
        panel_p = remotecall_fetch(C.pmap[i], panel_factor_par2, C, i, n)

        ## Apply permutation from pivoting
        for j = (i+1):nB
           depend[i,j] = remotecall(C.pmap[j], permute, C, i, j, panel_p, n, false)
        end
        ## Special case for last column
        if i == nB
           depend[nB,nB] = remotecall(C.pmap[nB], permute, C, i, nB+1, panel_p, n, true)
        end

        ##Trailing updates
        (i == nB) ? (I = (C.dist[i]):n) :
                    (I = (C.dist[i]):(C.dist[i+1]-1))
        C_II = C[I,I]
        L_II = tril(C_II, -1) + eye(length(I))
        K = (I[length(I)]+1):n
        if length(K) > 0
            C_KI = C[K,I]
        else
            C_KI = zeros(0)
        end

        for j=(i+1):nB
            dep = depend[i,j]
            depend[j,i] = remotecall(C.pmap[j], trailing_update_par2, C, L_II, C_KI, i, j, n, false, dep)
        end

        ## Special case for last column
        if i == nB
            dep = depend[nB,nB]
            remotecall_fetch(C.pmap[nB], trailing_update_par2, C, L_II, C_KI, i, nB+1, n, true, dep)
        else
            #enforce dependencies for nonspecial case
            for j=(i+1):nB
                wait(depend[j,i])
            end
        end
    end

    A = convert(Array, C)
    x = triu(A[1:n,1:n]) \ A[:,n+1]
end ## hpl_par2()

function panel_factor_par2(C, i, n)
    (C.dist[i+1] == n+2) ? (I = (C.dist[i]):n) :
                           (I = (C.dist[i]):(C.dist[i+1]-1))
    K = I[1]:n
    C_KI = C[K,I]
    #(C_KI, panel_p) = lu!(C_KI) #economy mode
    panel_p = lu!(C_KI)[2]
    C[K,I] = C_KI

    return panel_p
end ##panel_factor_par2()

function permute(C, i, j, panel_p, n, flag)
    if flag
        K = (C.dist[i]):n
        J = (n+1):(n+1)
        C_KJ = C[K,J]

        C_KJ = C_KJ[panel_p,:]
        C[K,J] = C_KJ
    else
        K = (C.dist[i]):n
        J = (C.dist[j]):(C.dist[j+1]-1)
        C_KJ = C[K,J]

        C_KJ = C_KJ[panel_p,:]
        C[K,J] = C_KJ
    end
end ##permute()

function trailing_update_par2(C, L_II, C_KI, i, j, n, flag, dep)
    if isa(dep, RemoteRef); wait(dep); end
    if flag
        #(C.dist[i+1] == n+2) ? (I = (C.dist[i]):n) :
        #                       (I = (C.dist[i]):(C.dist[i+1]-1))
        I = C.dist[i]:n
        J = (n+1):(n+1)
        K = (I[length(I)]+1):n
        C_IJ = C[I,J]
        if length(K) > 0
            C_KJ = C[K,J]
        else
            C_KJ = zeros(0)
        end
        ## Compute blocks of U
        C_IJ = L_II \ C_IJ
        C[I,J] = C_IJ
    else
        #(C.dist[i+1] == n+2) ? (I = (C.dist[i]):n) :
        #                       (I = (C.dist[i]):(C.dist[i+1]-1))

        I = (C.dist[i]):(C.dist[i+1]-1)
        J = (C.dist[j]):(C.dist[j+1]-1)
        K = (I[length(I)]+1):n
        C_IJ = C[I,J]
        if length(K) > 0
            C_KJ = C[K,J]
        else
            C_KJ = zeros(0)
        end

        ## Compute blocks of U
        C_IJ = L_II \ C_IJ
        C[I,J] = C_IJ
        ## Trailing submatrix update - All flops are here
        if !isempty(C_KJ)
            cm, ck = size(C_KI)
            cn = size(C_IJ,2)
            blas_gemm('N','N',cm,cn,ck,-1.0,C_KI,cm,C_IJ,ck,1.0,C_KJ,cm)
            #C_KJ = C_KJ - C_KI*C_IJ
            C[K,J] = C_KJ
        end
    end
end ## trailing_update_par2()

## Test n*n matrix on np processors
## Prints 5 numbers that should be close to zero
function test(n, np)
    A = rand(n,n); b = rand(n);
    X = (@elapsed x = A \ b);
    Y = (@elapsed y = hpl_par(A,b, max(1,div(n,np))));
    Z = (@elapsed z = hpl_par2(A,b));
    for i=1:(min(5,n))
        print(z[i]-y[i], " ")
    end
    println()
    return (X,Y,Z)
end

## test k times and collect average
function test(n,np,k)
    sum1 = 0; sum2 = 0; sum3 = 0;
    for i = 1:k
        (X,Y,Z) = test(n,np)
        sum1 += X
        sum2 += Y
        sum3 += Z
    end
    return (sum1/k, sum2/k, sum3/k)
end
