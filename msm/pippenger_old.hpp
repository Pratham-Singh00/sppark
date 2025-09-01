// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_OLD_HPP__
#define __SPPARK_MSM_PIPPENGER_OLD_HPP__

#include "pippenger_common.hpp"

#include <vector>
#include <memory>
#include <tuple>

template <class point_t, class affine_t, typename pow_t>
static void mult(point_t& ret, const affine_t& point,
                 const pow_t scalar, size_t top)
{
    ret.inf();
    if (point.is_inf())
        return;

    struct is_bit {
        static bool set(const pow_t v, size_t i)
        {   return (v[i/8] >> (i%8)) & 1;   }
    };

    while (--top && !is_bit::set(scalar, top)) ;
    if (is_bit::set(scalar, top)) {
        ret = point;
        while (top--) {
            ret.dbl();
            if (is_bit::set(scalar, top))
                ret.add(point);
        }
    }
}

#include <util/thread_pool_t.hpp>

template <class bucket_t, class point_t, class scalar_t,
          class affine_t = class bucket_t::affine_t>
static void mult_pippenger(point_t& ret, const affine_t points[], size_t npoints,
                           const scalar_t _scalars[], bool mont,
                           thread_pool_t* da_pool = nullptr)
{
    typedef typename scalar_t::pow_t pow_t;
    size_t nbits = scalar_t::nbits;
    size_t window = window_size(npoints);
    size_t ncpus = da_pool ? da_pool->size() : 0;

    // below is little-endian dependency, should it be removed?
    const pow_t* scalars = reinterpret_cast<decltype(scalars)>(_scalars);
    std::unique_ptr<pow_t[]> store = nullptr;
    if (mont) {
        store = decltype(store)(new pow_t[npoints]);
        if (ncpus < 2 || npoints < 1024) {
            for (size_t i = 0; i < npoints; i++)
                _scalars[i].to_scalar(store[i]);
        } else {
            da_pool->par_map(npoints, 512, [&](size_t i) {
                _scalars[i].to_scalar(store[i]);
            });
        }
        scalars = &store[0];
    }

    if (ncpus < 2 || npoints < 32) {
        if (npoints == 1) { // for completeness
            mult(ret, points[0], scalars[0], nbits);
            return;
        }

        std::vector<bucket_t> buckets(1 << window); /* zeroed */

        point_t p;
        ret.inf();

        /* top excess bits modulo target window size */
        size_t wbits = nbits % window, /* yes, it may be zero */
               cbits = wbits + 1,
               bit0 = nbits;
        while (bit0 -= wbits) {
            tile(p, points, npoints, scalars[0], nbits,
                    &buckets[0], bit0, wbits, cbits);
            ret.add(p);
            for (size_t i = 0; i < window; i++)
                ret.dbl();
            cbits = wbits = window;
        }
        tile(p, points, npoints, scalars[0], nbits,
                &buckets[0], 0, wbits, cbits);
        ret.add(p);
        return;
    }

    size_t nx, ny;
    std::tie(nx, ny, window) = breakdown(nbits, window, ncpus);

    struct tile_t {
        size_t x, dx, y, dy;
        point_t p;
        tile_t() {}
    };
    std::vector<tile_t> grid(nx * ny);

    size_t dx = npoints / nx,
           y  = window * (ny - 1);

    size_t total = 0;
    while (total < nx) {
        grid[total].x  = total * dx;
        grid[total].dx = dx;
        grid[total].y  = y;
        grid[total].dy = nbits - y;
        total++;
    }
    grid[total - 1].dx = npoints - grid[total - 1].x;

    while (y) {
        y -= window;
        for (size_t i = 0; i < nx; i++, total++) {
            grid[total].x  = grid[i].x;
            grid[total].dx = grid[i].dx;
            grid[total].y  = y;
            grid[total].dy = window;
        }
    }

    std::vector<std::atomic<size_t>> row_sync(ny); /* zeroed */
    counter_t<size_t> counter(0);
    channel_t<size_t> ch;

    auto n_workers = std::min(ncpus, total);
    while (n_workers--) {
        da_pool->spawn([&, window, total, nbits, nx, counter]() {
            size_t work;
            if ((work = counter++) < total) {
                std::vector<bucket_t> buckets(1 << window); /* zeroed */

                do {
                    size_t x  = grid[work].x,
                           dx = grid[work].dx,
                           y  = grid[work].y,
                           dy = grid[work].dy;
                    tile(grid[work].p, &points[x], dx,
                                       scalars[x], nbits, &buckets[0],
                                       y, dy, dy + (dy < window));
                    if (++row_sync[y / window] == nx)
                        ch.send(y);
                } while ((work = counter++) < total);
            }
        });
    }

    ret.inf();
    size_t row = 0;
    while (ny--) {
        auto y = ch.recv();
        row_sync[y / window] = -1U;
        while (grid[row].y == y) {
            while (row < total && grid[row].y == y)
                ret.add(grid[row++].p);
            if (y == 0)
                break;
            for (size_t i = 0; i < window; i++)
                ret.dbl();
            y -= window;
            if (row_sync[y / window] != -1U)
                break;
        }
    }
}

template <class bucket_t, class point_t, class scalar_t,
          class affine_t = class bucket_t::affine_t>
static void mult_pippenger(point_t& ret, const std::vector<affine_t>& points,
                           const std::vector<scalar_t>& scalars, bool mont,
                           thread_pool_t* da_pool = nullptr)
{
    mult_pippenger<bucket_t>(ret, points.data(),
                                  std::min(points.size(), scalars.size()),
                                  scalars.data(), mont, da_pool);
}

#include <util/slice_t.hpp>

template <class bucket_t, class point_t, class scalar_t,
          class affine_t = class bucket_t::affine_t>
static void mult_pippenger(point_t& ret, slice_t<affine_t> points,
                           slice_t<scalar_t> scalars, bool mont,
                           thread_pool_t* da_pool = nullptr)
{
    mult_pippenger<bucket_t>(ret, points.data(),
                                  std::min(points.size(), scalars.size()),
                                  scalars.data(), mont, da_pool);
}
#endif
