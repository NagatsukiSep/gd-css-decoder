# ChannelPass GPU tuning notes for GF=256

This note summarizes which of the earlier optimization ideas typically help when `GF=256` (the common case in this project) without adding extra host processing or memory transfers.

## What the current kernel does
- `ChannelPassKernel` allocates three contiguous shared-memory regions: `input[GF]`, `output[GF]`, and a `scratch` buffer sized to the block dimension for reductions. With `GF=256` and a 256-thread block, the shared footprint is `(2*GF + block.x) * sizeof(double) ≈ 6 KB`, so shared memory is not the bottleneck. 【F:src/gd_css_patched.cc†L565-L629】【F:src/gd_css_patched.cc†L2396-L2399】
- The launch picks the smallest power-of-two thread count ≥ `GF` (capped at 1024). For `GF=256`, that means exactly 256 threads (8 warps), so no threads sit idle due to GF padding. 【F:src/gd_css_patched.cc†L2382-L2396】
- Grid sizing chooses `grid.x = min(N, 65535)` and `grid.y = ceil(N / grid.x)`, so `grid.x * grid.y ≥ N` for practical `N`. The kernel’s row loop therefore runs once per block in typical workloads, already giving “one block per row.” 【F:src/gd_css_patched.cc†L2389-L2395】【F:src/gd_css_patched.cc†L557-L630】
- The block-wide reductions for normalization use a shared-memory tree (`BlockReduceSum`) with multiple `__syncthreads`. 【F:src/gd_css_patched.cc†L540-L612】
- The channel matrix is flattened on the host to `GF*GF` doubles (`matrixFlat`), i.e., about 512 KB for `GF=256`, which is too large for constant memory and would not fit in shared memory per block. 【F:src/gd_css_patched.cc†L2295-L2400】

## Which tweaks pay off at GF=256
- **Prioritize warp-level reduction paths.** Replacing the block-wide tree reduction with warp shuffles (plus one shared-memory handoff across warps) would cut several `__syncthreads` while keeping the same data flow, which directly trims kernel time without extra transfers. This targets the current reduction hot spots. 【F:src/gd_css_patched.cc†L540-L612】
- **Keep the block size at 256.** The launch logic already lands on 256 threads, so manually shrinking or growing the block would either leave GF entries uncovered or add idle warps; there is no idle-thread waste to reclaim here. 【F:src/gd_css_patched.cc†L2382-L2396】
- **Grid restructuring brings little benefit.** Because the chosen grid covers all `N` rows, the kernel’s outer loop rarely iterates more than once. Changing grid dimensions alone won’t meaningfully reduce per-row work. 【F:src/gd_css_patched.cc†L557-L630】【F:src/gd_css_patched.cc†L2389-L2395】
- **Caching the matrix is not feasible.** At 512 KB, the channel matrix exceeds constant-memory capacity and would exhaust per-block shared memory, so caching it on-chip is impractical without changing data precision or tiling strategy. 【F:src/gd_css_patched.cc†L2295-L2400】

## Kernel bottleneck at GF=256
- The dominant per-row cost is the dense `GF×GF` matrix multiply inside the kernel: each thread computes one or more output entries and walks the full `matrixT` column while reading the normalized `input`, resulting in ~65K multiply-adds per row for GF=256. This arithmetic and the global-memory traffic for `matrixT` dwarf the normalization reductions. 【F:src/gd_css_patched.cc†L586-L610】
- Because `matrixT` lives in global memory and is accessed in a column-wise pattern (`matrixT[e * GF + d]`), the loads have limited cache reuse across threads. Without tiling `matrixT` into shared memory (which would exceed the available ~48–96 KB per block for doubles at GF=256), the matrix read bandwidth remains the main bottleneck. 【F:src/gd_css_patched.cc†L586-L611】【F:src/gd_css_patched.cc†L2396-L2399】

## Practical takeaway
For `GF=256`, the most impactful kernel-only change (no extra transfers or host work) is to introduce warp-level reductions for the two normalization steps. The existing block sizing and grid setup are already near-optimal, and matrix caching is constrained by size.
