// SPDX-License-Identifier: Apache-2.0
#ifndef __SPPARK_MSM_PIPPENGER_HPP__
#define __SPPARK_MSM_PIPPENGER_HPP__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <util/thread_pool_t.hpp>
#include "glv.hpp"   

static size_t get_wval(const unsigned char *d, size_t off, size_t bits)
{
    size_t i, top = (off + bits - 1)/8;
    size_t ret, mask = (size_t)0 - 1;
    d   += off/8;     top -= off/8-1;
    for (ret=0, i=0; i<4;) {
        ret |= (*d & mask) << (8*i);
        mask = (size_t)0 - ((++i - top) >> (8*sizeof(top)-1));
        d += 1 & mask;
    }
    return ret >> (off%8);
}

static inline size_t window_size(size_t npoints)
{
    size_t w=0; for (; npoints>>=1; ++w);
    return w>12 ? w-3 : (w>4 ? w-2 : (w ? 2 : 1));
}

template<class P, class B>
static void integrate_buckets(P &out, B buckets[], size_t w)
{
    size_t top = 1u<<w;
    B acc = buckets[top-1], sum = acc;
    buckets[top-1].inf();
    for (size_t i=top-1; i-->0; ) {
        acc.add(buckets[i]);
        sum.add(acc);
        buckets[i].inf();
    }
    out = sum;
}

template<class B, class A>
static void bucket(B buckets[], size_t idx, size_t w, A const &p)
{
    idx &= (1u<<w)-1;
    if (idx--) buckets[idx].add(p);
}

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
