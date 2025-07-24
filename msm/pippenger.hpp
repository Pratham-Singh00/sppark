// SPDX-License-Identifier: Apache-2.0
#ifndef __SPPARK_MSM_PIPPENGER_HPP__
#define __SPPARK_MSM_PIPPENGER_HPP__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <util/thread_pool_t.hpp>
#include "glv.hpp"   
#include "pippenger_common.hpp"

namespace pasta_msm {

template<typename BucketT, typename ScalarT>
void mult_pippenger_glv(
    typename BucketT::point_t      &ret,
    typename BucketT::affine_t const pts[],
    size_t                          npts,
    ScalarT const                   scalars[],
    bool                            mont = true,
    thread_pool_t                 *pool  = nullptr
) {
    using Affine = typename BucketT::affine_t;
    using PowT   = typename ScalarT::pow_t;

    constexpr size_t TOP = ScalarT::nbits;
    constexpr size_t W   = 16;
    size_t nw = (TOP + W - 1)/W;
    size_t nb = 1u<<W;

    std::unique_ptr<BucketT[]> buckets(new BucketT[nb]);
    ret.inf();

    const PowT *ks = reinterpret_cast<const PowT*>(scalars);
    std::unique_ptr<PowT[]> store;
    if (mont) {
        store.reset(new PowT[npts]);
        if (!pool||npts<1024) {
            for (size_t i=0;i<npts;i++) scalars[i].to_scalar(store[i]);
        } else {
            pool->par_map(npts,512,[&](size_t i){
                scalars[i].to_scalar(store[i]);
            });
        }
        ks = store.get();
    }

    for (size_t win=0; win<nw; ++win) {
        size_t bit0 = win*W;
        size_t take = win? W : ((TOP-1)%W+1);
        for (size_t b=0;b<nb;++b) buckets[b].inf();
        for (size_t i=0;i<npts;++i) {
            const uint32_t *limbs = (uint32_t*)&ks[i];
            uint32_t k1[8], k2[8];
            glv_split(limbs, k1, k2);

            size_t w1 = get_wval((unsigned char*)k1, bit0, take);
            if (w1) buckets[w1].add(pts[i]);

            size_t w2 = get_wval((unsigned char*)k2, bit0, take);
            if (w2) {
                Affine tmp;
                transform_point_glv<BucketT>(pts[i], tmp);
                buckets[w2].add(tmp);
            }
        }
        typename BucketT::point_t acc, sum;
        size_t m = nb;
        acc = buckets[--m];
        sum = buckets[m];
        buckets[m].inf();
        while (m--) {
            acc.add(buckets[m]);
            sum.add(acc);
            buckets[m].inf();
        }
        ret.add(sum);
        if (win+1<nw)
            for (size_t d=0; d<W; ++d)
                ret.dbl();
    }
}

} 

#endif // __SPPARK_MSM_PIPPENGER_HPP__