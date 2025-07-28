// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_COMMON_HPP__
#define __SPPARK_MSM_PIPPENGER_COMMON_HPP__

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <tuple>
#include <algorithm>

static size_t get_wval(const unsigned char *d, size_t off, size_t bits)
{
    size_t i, top = (off + bits - 1)/8;
    size_t ret, mask = (size_t)0 - 1;

    d   += off/8;
    top -= off/8-1;

    /* this is not about constant-time-ness, but branch optimization */
    for (ret=0, i=0; i<4;) {
        ret |= (*d & mask) << (8*i);
        mask = (size_t)0 - ((++i - top) >> (8*sizeof(top)-1));
        d += 1 & mask;
    }

    return ret >> (off%8);
}

static inline size_t window_size(size_t npoints)
{
    size_t wbits;

    for (wbits=0; npoints>>=1; wbits++) ;

    return wbits>12 ? wbits-3 : (wbits>4 ? wbits-2 : (wbits ? 2 : 1));
}

template<class point_t, class bucket_t>
static void integrate_buckets(point_t& out, bucket_t buckets[], size_t wbits)
{
    bucket_t ret, acc;
    size_t n = (size_t)1 << wbits;

    /* Calculate sum of x[i-1]*i for i=1 through 1<<|wbits|. */
    acc = buckets[--n];
    ret = buckets[n];
    buckets[n].inf();
    while (n--) {
        acc.add(buckets[n]);
        ret.add(acc);
        buckets[n].inf();
    }
    out = ret;
}

template<class bucket_t, class affine_t>
static void bucket(bucket_t buckets[], size_t booth_idx,
                   size_t wbits, const affine_t& p)
{
    booth_idx &= (1<<wbits) - 1;
    if (booth_idx--)
        buckets[booth_idx].add(p);
}

template<class bucket_t>
static void prefetch(const bucket_t buckets[], size_t booth_idx, size_t wbits)
{
#if 0
    booth_idx &= (1<<wbits) - 1;
    if (booth_idx--)
        vec_prefetch(&buckets[booth_idx], sizeof(buckets[booth_idx]));
#else
    (void)buckets;
    (void)booth_idx;
    (void)wbits;
#endif
}

template<class point_t, class affine_t, class bucket_t>
static void tile(point_t& ret, const affine_t points[], size_t npoints,
                 const unsigned char* scalars, size_t nbits,
                 bucket_t buckets[], size_t bit0, size_t wbits, size_t cbits)
{
    size_t wmask, wval, wnxt;
    size_t i, nbytes;

    nbytes = (nbits + 7)/8; /* convert |nbits| to bytes */
    wmask = ((size_t)1 << wbits) - 1;
    wval = get_wval(scalars, bit0, wbits) & wmask;
    scalars += nbytes;
    wnxt = get_wval(scalars, bit0, wbits) & wmask;
    npoints--;  /* account for prefetch */

    bucket(buckets, wval, cbits, points[0]);
    for (i = 1; i < npoints; i++) {
        wval = wnxt;
        scalars += nbytes;
        wnxt = get_wval(scalars, bit0, wbits) & wmask;
        prefetch(buckets, wnxt, cbits);
        bucket(buckets, wval, cbits, points[i]);
    }
    bucket(buckets, wnxt, cbits, points[i]);
    integrate_buckets(ret, buckets, cbits);
}

template<typename T>
static size_t num_bits(T l)
{
    const size_t T_BITS = 8*sizeof(T);
# define MSB(x) ((T)(x) >> (T_BITS-1))
    T x, mask;

    if ((T)-1 < 0) {    // handle signed T
        mask = MSB(l);
        l ^= mask;
        l += 1 & mask;
    }

    size_t bits = (((T)(~l & (l-1)) >> (T_BITS-1)) & 1) ^ 1;

    if (sizeof(T) > 4) {
        x = l >> (32 & (T_BITS-1));
        mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
        bits += 32 & mask;
        l ^= (x ^ l) & mask;
    }

    if (sizeof(T) > 2) {
        x = l >> 16;
        mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
        bits += 16 & mask;
        l ^= (x ^ l) & mask;
    }

    if (sizeof(T) > 1) {
        x = l >> 8;
        mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
        bits += 8 & mask;
        l ^= (x ^ l) & mask;
    }

    x = l >> 4;
    mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
    bits += 4 & mask;
    l ^= (x ^ l) & mask;

    x = l >> 2;
    mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
    bits += 2 & mask;
    l ^= (x ^ l) & mask;

    bits += l >> 1;

    return bits;
# undef MSB
}

std::tuple<size_t, size_t, size_t>
static breakdown(size_t nbits, size_t window, size_t ncpus)
{
    size_t nx, ny, wnd;

    if (nbits > window * ncpus) {
        nx = 1;
        if (window + (wnd = num_bits(ncpus / 4)) > 18) {
            wnd = window - wnd;
        } else {
            wnd = (nbits / window + ncpus - 1) / ncpus;
            if ((nbits / (window+1) + ncpus - 1) / ncpus < wnd)
                wnd = window + 1;
            else
                wnd = window;
        }
    } else {
        nx = 2;
        wnd = window - 2;
        while ((nbits / wnd + 1) * nx < ncpus) {
            nx += 1;
            wnd = window - num_bits(3 * nx / 2);
        }
        nx -= 1;
        wnd = window - num_bits(3 * nx / 2);
    }
    ny = nbits / wnd + 1;
    wnd = nbits / ny + 1;

    return std::make_tuple(nx, ny, wnd);
}

#endif // __SPPARK_MSM_PIPPENGER_COMMON_HPP__