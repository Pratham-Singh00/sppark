// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_HPP__
#define __SPPARK_MSM_PIPPENGER_HPP__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <utility>
#include <util/thread_pool_t.hpp>
#include "glv.hpp"
#include "pippenger_common.hpp"

namespace pasta_msm {
size_t getwindow(const uint32_t k[4],size_t start_idx){
    uint32_t block=(start_idx)/32;
    uint32_t blockidx=start_idx%32;
    uint32_t mask=k[block];
    if(blockidx==16){return (mask>>16);}
    else {return uint16_t(mask);}
}
template <class bucket_t,
          class point_t,
          class scalar_t,
          class affine_t = typename bucket_t::affine_t>
void mult_pippenger_glv(
    point_t& ret,
    const affine_t pts[],
    size_t npoints,
    const scalar_t scalars[],
    bool mont,
    thread_pool_t* da_pool = nullptr
) {
    typedef typename scalar_t::pow_t pow_t;
    size_t ncpus = da_pool ? da_pool->size() : 0;
    // below is little-endian dependency, should it be removed?
    const pow_t* scalarsp = reinterpret_cast<decltype(scalarsp)>(scalars);
    std::unique_ptr<pow_t[]> store = nullptr;
    if (mont) {
        store = decltype(store)(new pow_t[npoints]);
        if (ncpus < 2 || npoints < 1024) {
            for (size_t i = 0; i < npoints; i++)
                scalars[i].to_scalar(store[i]);
        } else {
            da_pool->par_map(npoints, 512, [&](size_t i) {
                scalars[i].to_scalar(store[i]);
            });
        }
        scalarsp = &store[0];
    }
    constexpr size_t BITS_TO_PROCESS = 128; 
    constexpr size_t W = 16;
    constexpr size_t num_windows = (BITS_TO_PROCESS + W - 1) / W;
    size_t num_buckets = size_t(1) << W;
    std::vector<std::pair<DecomposedScalar, DecomposedScalar>> decomposed_scalars(npoints);
    for (size_t i = 0; i < npoints; ++i) {
        uint8_t tmp[32];
        for(int j=0;j<32;++j){
            tmp[j]=scalarsp[i][j];
        }
        decomposed_scalars[i] = glv_split(tmp);
    }
    ret.inf();
    std::unique_ptr<bucket_t[]> buckets(new bucket_t[num_buckets]);
    for (size_t win = num_windows; win-- > 0;) {
        if (win != num_windows - 1) {
            for (size_t d = 0; d < W; ++d) {
                ret.dbl();
            }
        }

        for (size_t b = 0; b < num_buckets; ++b) {
            buckets[b].inf();
        }
        
        size_t bit_offset = win * W;

        for (size_t i = 0; i < npoints; ++i) {
            const auto& decomp_pair = decomposed_scalars[i];
            const auto& d1 = decomp_pair.first;
            const auto& d2 = decomp_pair.second;

            size_t w1 = getwindow(d1.k, bit_offset);
            if (w1) {
                affine_t p1 = pts[i];
                p1.cneg(d1.is_negative);
                buckets[w1].add(p1);
            }

            size_t w2 = getwindow(d2.k, bit_offset);
            if (w2) {
                affine_t p2 = pts[i];
                affine_t p2_transformed;
                transform_point_glv<affine_t>(p2, p2_transformed);
                p2_transformed.cneg(d2.is_negative);
                buckets[w2].add(p2_transformed);
            }
        }

        point_t window_sum; window_sum.inf();
        point_t bucket_accumulator; bucket_accumulator.inf();
        for (size_t m = num_buckets - 1; m > 0; --m) {
            bucket_accumulator.add(point_t(buckets[m]));
            window_sum.add(bucket_accumulator);
        }
        
        ret.add(window_sum);
    }
}

} 

#endif