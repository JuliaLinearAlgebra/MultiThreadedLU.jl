# MultiThreadedLU.jl

A multi-threaded LU implementation.

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
