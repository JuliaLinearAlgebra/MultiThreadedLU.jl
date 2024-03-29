# MultiThreadedLU.jl

A multi-threaded LU implementation.

Install and test:
```
using Pkg
Pkg.add(url="https://github.com/JuliaLinearAlgebra/MultiThreadedLU.jl")
Pkg.test("MultiThreadedLU")
```

Example:
```
using MultiThreadedLU, Random, LinearAlgebra

n = 1024
a = rand(n,n);
b = rand(n);
x = hpl_mt(a, b)
norm(a*x-b)
```

## References

Husbands, Parry, and Katherine Yelick. ["Multi-threading and one-sided communication in parallel LU factorization."](https://upc.lbl.gov/publications/husbands-lu-sc07.pdf) In Proceedings of the 2007 ACM/IEEE Conference on Supercomputing, pp. 1-10. 2007.

## See also

[RecursiveFactorization.jl](https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl) is a better optimized package and much more well tested. This package is mainly a proof of concept for an eventual large-scale distributed implementation.
