using MultiThreadedLU
using Test, Random

n = 1024
a = rand(n,n);
b = rand(n);

@testset "hpl_seq" begin
    x = hpl_seq(a, b)
    @test a*x ≈ b
end

@testset "hpl_mt" begin
    x = hpl_mt(a, b)
    @test a*x ≈ b
end
