#ifndef __SPPARK_MSM_PIPPENGER_HPP__
#define __SPPARK_MSM_PIPPENGER_HPP__

#include <cstddef>
#include <cstdint>
#include <memory>
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
    size_t npts,
    const scalar_t scalars[],
    bool mont,
    thread_pool_t* pool = nullptr
) {
    using Affine = affine_t;
    using PowT   = typename scalar_t::pow_t;

    constexpr size_t TOP = scalar_t::nbits;
    constexpr size_t W   = 16;
    size_t nw = (TOP + W - 1) / W;
    size_t nb = size_t(1) << W;

    std::unique_ptr<bucket_t[]> buckets(new bucket_t[nb]);
    for (size_t i = 0; i < nb; ++i) {
        buckets[i].inf();
    }
    ret.inf();

    const PowT* ks = reinterpret_cast<const PowT*>(scalars);
    std::unique_ptr<PowT[]> store;
    if (mont) {
        store.reset(new PowT[npts]);
        if (!pool || npts < 1024) {
            for (size_t i = 0; i < npts; ++i) {
                scalars[i].to_scalar(store[i]);
            }
        } else {
            pool->par_map(npts, 512, [&](size_t i) {
                scalars[i].to_scalar(store[i]);
            });
        }
        ks = store.get();
    }

    for (size_t win = 0; win < nw; ++win) {
        size_t bit0 = win * W;
        size_t take = (win == 0 ? ((TOP - 1) % W + 1) : W);

        for (size_t b = 0; b < nb; ++b) {
            buckets[b].inf();
        }

        for (size_t i = 0; i < npts; ++i) {
            const uint32_t* limbs = reinterpret_cast<const uint32_t*>(&ks[i]);
            uint32_t k1[8], k2[8];
            glv_split(limbs, k1, k2);

            size_t w1 = get_wval(reinterpret_cast<const unsigned char*>(k1), bit0, take);
            if (w1) {
                buckets[w1].add(pts[i]);
            }

            size_t w2 = get_wval(reinterpret_cast<const unsigned char*>(k2), bit0, take);
            if (w2) {
                Affine tmp;
                transform_point_glv<Affine>(pts[i], tmp);
                buckets[w2].add(tmp);
            }
        }

        point_t acc = point_t(buckets[--nb]);
        point_t sum = acc;
        buckets[nb].inf();

        size_t m = nb;
        while (m--) {
            acc.add(point_t(buckets[m]));
            sum.add(acc);
            buckets[m].inf();
        }

        ret.add(sum);
        if (win + 1 < nw) {
            for (size_t d = 0; d < W; ++d) {
                ret.dbl();
            }
        }
    }
}

} // namespace pasta_msm

#endif // __SPPARK_MSM_PIPPENGER_HPP__