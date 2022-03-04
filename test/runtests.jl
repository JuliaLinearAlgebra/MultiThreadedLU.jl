using MultiThreadedLU
using Test

@testset "MultiThreadedLU.jl" begin
    par = true
    n = 4096
    blocksize=64

    tseq=0.
    a = rand(n,n);
    b = rand(n);
    x = hpl_shared(a, b, blocksize, par);
    @printf "Total time = %4.2f sec\n" @elapsed x = hpl_shared(a, b, blocksize, par);
    @printf "  Seq time = %4.2f sec\n" tseq

    @test norm(a*x-b) < 1e-6
end
