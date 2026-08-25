#pragma once

#include <cfloat>
#include <cuda_runtime.h>

// Partial softmax statistics for a (possibly empty) subset of a row:
//   m  = max of the values seen so far
//   d  = sum_i exp(x_i - m)
//
// The empty pair is the identity of merge_pair: max = -inf, sum = 0.
// Merging two pairs is the associative operator from Milakov & Gimelshein
// (eq. 4): rescale both sums onto the larger max, then add.
struct SoftmaxPair {
    float m;
    float d;
};

__device__ __forceinline__ SoftmaxPair empty_pair() {
    return {-FLT_MAX, 0.0f};
}

__device__ __forceinline__ SoftmaxPair merge_pair(SoftmaxPair a, SoftmaxPair b) {
    const float m = fmaxf(a.m, b.m);
    // If a.m == m, exp(a.m - m) == 1 and a.d is unchanged.
    // If a.m  < m, a.d is multiplied by exp(old_max - new_max) so it
    // remains a valid sum of exp(x - new_max). Same for b.
    // empty_pair contributes 0 because exp(-FLT_MAX - m) underflows to 0.
    return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

// Tree reduction of 32 SoftmaxPairs inside a warp. After this call, lane 0
// holds the merged result for the warp; other lanes hold partial values.
__device__ __forceinline__ SoftmaxPair warp_reduce_pair(SoftmaxPair value) {
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const SoftmaxPair other = {
            __shfl_down_sync(mask, value.m, offset),
            __shfl_down_sync(mask, value.d, offset)};
        value = merge_pair(value, other);
    }
    return value;
}
