using ExponentialUtilities, BenchmarkTools

d, q = 4, 3
L = rand(d, d);
out = ExponentialUtilities.phi(L, q);
caches = (zeros(d), zeros(d, q + 1), zeros(d + q, d + q));
expmethod = ExpMethodHigham2005()

expcache = ExponentialUtilities.alloc_mem(zeros(d + q, d + q), expmethod);
ExponentialUtilities.phi!(out, L, q; caches, expmethod, expcache);

@btime ExponentialUtilities.phi!(
    $out,
    $L,
    $q;
    caches=$caches,
    expmethod=$expmethod,
    expcache=$expcache,
);
# 21.998 μs (32 allocations: 10.00 KiB)

M = rand(d * (q + 1), d * (q + 1));
cache = ExponentialUtilities.alloc_mem(M, expmethod);
@btime ExponentialUtilities.exponential!(copy($M), $expmethod, $cache);
# 3.214 μs (2 allocations: 288 bytes)

expcache = ExponentialUtilities.alloc_mem(zeros(d + q, d + q), expmethod);
@btime ExponentialUtilities.phi!(
    $out,
    $L,
    $q;
    caches=$caches,
    expmethod=$expmethod,
    expcache=$expcache,
);
