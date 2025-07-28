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
#include <array>
#include <util/thread_pool_t.hpp>
#include "glv.hpp"
#include "pippenger_common.hpp"

namespace pasta_msm {

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
    
    const pow_t* scalarsp = reinterpret_cast<decltype(scalarsp)>(scalars);
    std::unique_ptr<pow_t[]> store = nullptr;
    if (mont) {
        store = decltype(store)(new pow_t[npoints]);
        auto convert_scalars = [&](size_t i) {
            scalars[i].to_scalar(store[i]);
        };
        if (ncpus < 2 || npoints < 1024) {
            for (size_t i = 0; i < npoints; i++) convert_scalars(i);
        } else {
            da_pool->par_map(npoints, 512, convert_scalars);
        }
        scalarsp = &store[0];
    }
    
    constexpr size_t WINDOW_BITS = 16;
    constexpr size_t SCALAR_BITS = 128; 
    constexpr size_t NUM_WINDOWS = (SCALAR_BITS + WINDOW_BITS - 1) / WINDOW_BITS;
    constexpr size_t BUCKETS_PER_WINDOW = size_t(1) << (WINDOW_BITS - 1);
    
    using assignment_array_t = std::array<BucketAssignment, NUM_WINDOWS>;
    

    std::vector<assignment_array_t> assignments_k1(npoints);
    std::vector<assignment_array_t> assignments_k2(npoints);
    
    auto precompute_assignments = [&](size_t i) {
        uint8_t scalar_bytes[32];
        for (int j = 0; j < 32; ++j) {
            scalar_bytes[j] = scalarsp[i][j];
        }
        auto [k1, k2] = glv_split(scalar_bytes);
        assignments_k1[i] = recode_wnaf<WINDOW_BITS, NUM_WINDOWS>(k1);
        assignments_k2[i] = recode_wnaf<WINDOW_BITS, NUM_WINDOWS>(k2);
    };

    if (ncpus < 2 || npoints < 1024) {
        for (size_t i = 0; i < npoints; i++) precompute_assignments(i);
    } else {
        da_pool->par_map(npoints, 512, precompute_assignments);
    }
    
    ret.inf();
    std::unique_ptr<bucket_t[]> buckets(new bucket_t[BUCKETS_PER_WINDOW]);

    for (size_t win = NUM_WINDOWS; win-- > 0;) {
        if (win != NUM_WINDOWS - 1) {
            for (size_t d = 0; d < WINDOW_BITS; ++d) {
                ret.dbl();
            }
        }
        for (size_t b = 0; b < BUCKETS_PER_WINDOW; ++b) {
            buckets[b].inf();
        }
        for (size_t i = 0; i < npoints; ++i) {
            const auto& assignment1 = assignments_k1[i][win];
            if (assignment1.index != 0) {
                affine_t p1 = pts[i];
                p1.cneg(!assignment1.add);
                buckets[assignment1.index - 1].add(p1);
            }

            const auto& assignment2 = assignments_k2[i][win];
            if (assignment2.index != 0) {
                affine_t p2_transformed;
                transform_point_glv<affine_t>(pts[i], p2_transformed);
                p2_transformed.cneg(!assignment2.add);
                buckets[assignment2.index - 1].add(p2_transformed);
            }
        }
        
        point_t window_sum; window_sum.inf();
        point_t bucket_accumulator; bucket_accumulator.inf();
        for (size_t m = BUCKETS_PER_WINDOW; m-- > 0;) {
            bucket_accumulator.add(point_t(buckets[m]));
            window_sum.add(bucket_accumulator);
        }
        
        ret.add(window_sum);
    }
}

} 

#endif