// Copyright Supranational LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_COMMON_HPP__
#define __SPPARK_MSM_PIPPENGER_COMMON_HPP__

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <tuple>
#include <algorithm>

static inline size_t get_wval(const unsigned char *d, size_t off, size_t bits)
{
    size_t i, top = (off + bits - 1) / 8;
    size_t ret = 0, mask = static_cast<size_t>(-1);
    d += off/8;
    top -= off/8 - 1;
    for (i = 0; i < 4; ) {
        ret |= (*d & mask) << (8*i);
        mask = static_cast<size_t>(
          -(((long)++i - (long)top) >> (8*sizeof(top)-1))
        );
        d += mask & 1;
    }
    return ret >> (off % 8);
}

static inline size_t window_size(size_t npoints)
{
    size_t w = 0;
    size_t t = npoints;
    while (t >>= 1) ++w;
    if      (w > 12) return w - 3;
    else if (w >  4) return w - 2;
    else if (w >  0) return 2;
    else             return 1;
}

template<class P, class B>
static inline void integrate_buckets(P &out, B buckets[], size_t wbits)
{
    size_t n = static_cast<size_t>(1) << wbits;
    B acc = buckets[--n];
    B sum = acc;
    buckets[n].inf();
    while (n--) {
        acc .add(buckets[n]);
        sum .add(acc);
        buckets[n].inf();
    }
    out = sum;
}

template<class B, class A>
static inline void bucket(B buckets[], size_t idx, size_t wbits, A const &p)
{
    idx &= (static_cast<size_t>(1) << wbits) - 1;
    if (idx--) buckets[idx].add(p);
}

template<class B>
static inline void prefetch(const B[], size_t, size_t)
{
    /* nop */
}

template<class P, class A, class B>
static inline void tile(
    P &out,
    A const pts[], size_t n,
    const unsigned char *scalars, size_t nbits,
    B buckets[], size_t bit0, size_t wbits, size_t cbits
) {
    size_t byte_len = (nbits + 7)/8;
    size_t wmask    = (static_cast<size_t>(1) << wbits) - 1;
    size_t w0       = get_wval(scalars, bit0, wbits) & wmask;
    scalars += byte_len;
    size_t w1       = get_wval(scalars, bit0, wbits) & wmask;

    bucket(buckets, w0, cbits, pts[0]);
    for (size_t i = 1; i + 1 < n; ++i) {
        bucket(buckets, w1, cbits, pts[i]);
        scalars += byte_len;
        w1 = get_wval(scalars, bit0, wbits) & wmask;
        prefetch(buckets, w1, cbits);
    }
    bucket(buckets, w1, cbits, pts[n-1]);
    integrate_buckets(out, buckets, cbits);
}

template<typename T>
static inline size_t num_bits(T l)
{
    const size_t TB = 8 * sizeof(T);
#   define MSB(x) ((T)(x) >> (TB-1))
    T mask, x;
    if ((T)-1 < 0) {
        mask = MSB(l);
        l   ^= mask;
        l   += 1 & mask;
    }
    size_t bits = ((((~l) & (l-1)) >> (TB-1)) & 1) ^ 1;
    for (size_t shift : {32,16,8,4,2}) {
        if (TB > shift) {
            x    = l >> shift;
            mask = MSB(0 - x);
            if ((T)-1 > 0) mask = 0 - mask;
            bits += shift & mask;
            l    ^= (x ^ l) & mask;
        }
    }
    bits += l >> 1;
#   undef MSB
    return bits;
}

static inline std::tuple<size_t,size_t,size_t>
breakdown(size_t nbits, size_t window, size_t ncpu)
{
    size_t nx, ny, wnd;
    if (nbits > window*ncpu) {
        nx = 1;
        size_t t = num_bits(static_cast<size_t>(ncpu/4));
        if (window + t > 18) wnd = window - t;
        else                 wnd = std::min(window+1,(nbits+window-1)/window);
    } else {
        nx = 2;
        wnd = window - 2;
        while ((nbits/wnd + 1)*nx < ncpu) {
            ++nx;
            wnd = window - num_bits(static_cast<size_t>(3*nx/2));
        }
        --nx;
        wnd = window - num_bits(static_cast<size_t>(3*nx/2));
    }
    ny  = nbits / wnd + 1;
    wnd = nbits / ny + 1;
    return {nx,ny,wnd};
}

#endif // __SPPARK_MSM_PIPPENGER_COMMON_HPP__