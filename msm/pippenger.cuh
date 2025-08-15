// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_CUH__
#define __SPPARK_MSM_PIPPENGER_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <cassert>
#include <vector>

#include <util/vec2d_t.hpp>
#include <util/slice_t.hpp>
#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include "sort.cuh"
#include "batch_addition.cuh"
#include "glv.hpp"
#include "../ec/affine_t.hpp"


#ifndef WARP_SZ
# define WARP_SZ 32
#endif
#ifdef __GNUC__
# define asm __asm__ __volatile__
#else
# define asm asm volatile
#endif

/*
 * Break down |scalars| to signed |wbits|-wide digits.
 */
#ifdef __CUDA_ARCH__
// Transposed scalar_t
template<class scalar_t>
class scalar_T {
    uint32_t val[sizeof(scalar_t)/sizeof(uint32_t)][WARP_SZ];

public:
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }
    __device__ scalar_T& operator()(uint32_t laneid)
    {   return *reinterpret_cast<scalar_T*>(&val[0][laneid]);   }
    __device__ scalar_T& operator=(const scalar_t& rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

template<class scalar_t>
__device__ __forceinline__
static uint32_t get_wval(const scalar_T<scalar_t>& scalar, uint32_t off,
                         uint32_t top_i = (size_t(143) + 31) / 32 - 1)
{
    uint32_t i = off / 32;
    uint64_t ret = scalar[i];

    if (i < top_i)
        ret |= (uint64_t)scalar[i+1] << 32;

    return ret >> (off%32);
}

__device__ __forceinline__
static uint32_t booth_encode(uint32_t wval, uint32_t wmask, uint32_t wbits)
{
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0-wval : wval;
}
#endif

template<class scalar_t>
__launch_bounds__(1024) __global__
void breakdown(vec2d_t<uint32_t> digits, const scalar_t scalars[], size_t len,
               uint32_t nwins, uint32_t wbits, bool mont = true)
{
    assert(len <= (1U<<31) && wbits < 32);

#ifdef __CUDA_ARCH__
    extern __shared__ scalar_T<scalar_t> xchange[];
    const uint32_t tid = threadIdx.x;
    const uint32_t tix = threadIdx.x + blockIdx.x*blockDim.x;

    const uint32_t top_i = (size_t(143) + 31) / 32 - 1;
    const uint32_t wmask = 0xffffffffU >> (31-wbits); // (1U << (wbits+1)) - 1;

    auto& scalar = xchange[tid/WARP_SZ](tid%WARP_SZ);

    #pragma unroll 1
    for (uint32_t i = tix; i < (uint32_t)len; i += gridDim.x*blockDim.x) {
        auto s = scalars[i];


        // clear the most significant bit
        uint32_t msb = s.abs();
        msb <<= 31;

        scalar = s;
        #pragma unroll 1
        for (uint32_t bit0 = nwins*wbits - 1, win = nwins; --win;) {
            bit0 -= wbits;
            uint32_t wval = get_wval(scalar, bit0, top_i);
            wval = booth_encode(wval, wmask, wbits);
            if (wval) wval ^= msb;
            digits[win][i] = wval;
        }

        uint32_t wval = s[0] << 1;
        wval = booth_encode(wval, wmask, wbits);
        if (wval) wval ^= msb;
        digits[0][i] = wval;
    }
#endif
}

template<class scalar_t, class affine_t>
__launch_bounds__(256) __global__
void decompose(
    const affine_t* points_in,  
    const scalar_t* scalars_in,  
    affine_t* final_points_out,   
    scalar_t* final_scalars_out,     
    size_t npoints)
{
#ifdef __CUDA_ARCH__
    const size_t idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t idx = idx0; idx < npoints; idx += stride) {
        affine_t p = points_in[idx];
        affine_t p_glv;
        pasta_msm::transform_point_glv(p, p_glv);

        scalar_t s = scalars_in[idx];
        s.from();

        const uint8_t* byt = reinterpret_cast<const uint8_t*>(&s);

        pasta_msm::DecomposedScalar d1, d2;
        pasta_msm::glv_split(byt, idx, d1, d2);

        scalar_t k1, k2;
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); ++i) {
            k1[i] = 0;
            k2[i] = 0;
        }
        for (size_t i = 0; i < 4; ++i) {
            k1[i] = d1.k[i];
            k2[i] = d2.k[i];
        }

        p.cneg(d1.is_negative);
        p_glv.cneg(d2.is_negative);

        final_scalars_out[2*idx]     = k1;
        final_scalars_out[2*idx + 1] = k2;
        final_points_out[2*idx]      = p;
        final_points_out[2*idx + 1]  = p_glv;
    }
#endif
}

template<class affine_t>
__launch_bounds__(256) __global__
void transform(
    const affine_t* points_in,
    affine_t* prepared_points_out,
    size_t npoints)
{
#ifdef __CUDA_ARCH__
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < npoints; idx += gridDim.x * blockDim.x) {
        affine_t p1 = points_in[idx];
        affine_t p2;
        pasta_msm::transform_point_glv(p1, p2);
        prepared_points_out[2 * idx]     = p1;
        prepared_points_out[2 * idx + 1] = p2;
    }
#endif
}

template<class scalar_t, class affine_t>
__launch_bounds__(256) __global__
void glv_split(
    const scalar_t* scalars_in,
    const affine_t* prepared_points_in,
    scalar_t* final_scalars_out,
    affine_t* final_points_out,
    size_t npoints)
{
#ifdef __CUDA_ARCH__
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < npoints; idx += gridDim.x * blockDim.x) {
        auto tmp(scalars_in[idx]);
        tmp.from();

        const uint8_t* byt = reinterpret_cast<const uint8_t*>(&tmp);

        pasta_msm::DecomposedScalar d1, d2;
        pasta_msm::glv_split(byt, idx, d1, d2);

        scalar_t k1, k2;
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++) {
            k1[i] = 0; k2[i] = 0;
        }
        for (size_t i = 0; i < 4; i++) {
            k1[i] = d1.k[i];
            k2[i] = d2.k[i];
        }

        affine_t p1 = prepared_points_in[2 * idx];
        affine_t p2 = prepared_points_in[2 * idx + 1];

        p1.cneg(d1.is_negative);
        p2.cneg(d2.is_negative);

        final_scalars_out[2 * idx] = k1;
        final_scalars_out[2 * idx + 1] = k2;
        final_points_out[2 * idx] = p1;
        final_points_out[2 * idx + 1] = p2;
    }
#endif
}


#ifndef LARGE_L1_CODE_CACHE
# if __CUDA_ARCH__-0 >= 800
#  define LARGE_L1_CODE_CACHE 1
#  define ACCUMULATE_NTHREADS 512
# else
#  define LARGE_L1_CODE_CACHE 0
#  define ACCUMULATE_NTHREADS (bucket_t::degree == 1 ? 512 : 256)
# endif
#endif

#ifndef MSM_NTHREADS
# define MSM_NTHREADS 128
#endif
#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS-1)) != 0
# error "bad MSM_NTHREADS value"
#endif
#ifndef MSM_NSTREAMS
# define MSM_NSTREAMS 8
#elif MSM_NSTREAMS<2
# error "invalid MSM_NSTREAMS"
#endif
template<class bucket_t,
         class affine_h,
         class bucket_h = class bucket_t::mem_t,
         class affine_t = class bucket_t::affine_t>
__launch_bounds__(ACCUMULATE_NTHREADS) __global__
void accumulate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                /*const*/ affine_h points_[], const vec2d_t<uint32_t> digits,
                const vec2d_t<uint32_t> histogram, uint32_t sid = 0)
{
    vec2d_t<bucket_h> buckets{buckets_, 1U<<--wbits};
    const affine_h* points = points_;

    static __device__ uint32_t streams[MSM_NSTREAMS];
    uint32_t& current = streams[sid % MSM_NSTREAMS];
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t lane_id = laneid / degree;

    uint32_t x, y;
#if 1
    __shared__ uint32_t xchg;

    if (threadIdx.x == 0)
        xchg = atomicAdd(&current, blockDim.x/degree);
    __syncthreads();
    x = xchg + threadIdx.x/degree;
#else
    x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
    x = __shfl_sync(0xffffffff, x, 0) + lane_id;
#endif

    while (x < (nwins << wbits)) {
        y = x >> wbits;
        x &= (1U << wbits) - 1;
        const uint32_t* h = &histogram[y][x];

        uint32_t idx, len = h[0];

        asm("{ .reg.pred %did;"
            "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
            "  @!%did mov.b32 %0, 0;"
            "}" : "=r"(idx) : "r"(len), "r"(degree));

        if (lane_id == 0 && x != 0)
            idx = h[-1];

        if ((len -= idx) && !(x == 0 && y == 0)) {
            const uint32_t* digs_ptr = &digits[y][idx];
            uint32_t digit = *digs_ptr++;

            affine_t p = points[digit & 0x7fffffff];
            bucket_t bucket = p;
            bucket.cneg(digit >> 31);

            while (--len) {
                digit = *digs_ptr++;
                p = points[digit & 0x7fffffff];
                if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE)
                    bucket.add(p, digit >> 31);
                else
                    bucket.uadd(p, digit >> 31);
            }

            buckets[y][x] = bucket;
        } else {
            buckets[y][x].inf();
        }

        x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
        x = __shfl_sync(0xffffffff, x, 0) + lane_id;
    }

    cooperative_groups::this_grid().sync();

    if (threadIdx.x + blockIdx.x == 0)
        current = 0;
}

template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__
void integrate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits, uint32_t nbits)
{
    const uint32_t degree = bucket_t::degree;
    uint32_t Nthrbits = 31 - __clz(blockDim.x / degree);

    assert((blockDim.x & (blockDim.x-1)) == 0 && wbits-1 > Nthrbits);

    vec2d_t<bucket_h> buckets{buckets_, 1U<<(wbits-1)};
    extern __shared__ uint4 scratch_[];
    auto* scratch = reinterpret_cast<bucket_h*>(scratch_);
    const uint32_t tid = threadIdx.x / degree;
    const uint32_t bid = blockIdx.x;

    auto* row = &buckets[bid][0];
    uint32_t i = 1U << (wbits-1-Nthrbits);
    row += tid * i;

    uint32_t mask = 0;
    if ((bid+1)*wbits > nbits) {
        uint32_t lsbits = nbits - bid*wbits;
        mask = (1U << (wbits-lsbits)) - 1;
    }

    bucket_t res, acc = row[--i];

    if (i & mask) {
        if (sizeof(res) <= 128) res.inf();
        else                    scratch[tid].inf();
    } else {
        if (sizeof(res) <= 128) res = acc;
        else                    scratch[tid] = acc;
    }

    bucket_t p;

    #pragma unroll 1
    while (i--) {
        p = row[i];

        uint32_t pc = i & mask ? 2 : 0;
        #pragma unroll 1
        do {
            if (sizeof(bucket_t) <= 128) {
                p.add(acc);
                if (pc == 1) {
                    res = p;
                } else {
                    acc = p;
                    if (pc == 0) p = res;
                }
            } else {
                if (LARGE_L1_CODE_CACHE && degree == 1)
                    p.add(acc);
                else
                    p.uadd(acc);
                if (pc == 1) {
                    scratch[tid] = p;
                } else {
                    acc = p;
                    if (pc == 0) p = scratch[tid];
                }
            }
        } while (++pc < 2);
    }

    __syncthreads();

    buckets[bid][2*tid] = p;
    buckets[bid][2*tid+1] = acc;
}
#undef asm
#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
template __global__
void accumulate<bucket_t, affine_t::mem_t>(bucket_t::mem_t buckets_[],
                                           uint32_t nwins, uint32_t wbits,
                                           /*const*/ affine_t::mem_t points_[],
                                           const vec2d_t<uint32_t> digits,
                                           const vec2d_t<uint32_t> histogram,
                                           uint32_t sid);
template __global__
void batch_addition<bucket_t>(bucket_t::mem_t buckets[],
                              const affine_t::mem_t points[], size_t npoints,
                              const uint32_t digits[], const uint32_t& ndigits);
template __global__
void integrate<bucket_t>(bucket_t::mem_t buckets_[], uint32_t nwins,
                         uint32_t wbits, uint32_t nbits);
template __global__
void breakdown<scalar_t>(vec2d_t<uint32_t> digits, const scalar_t scalars[],
                         size_t len, uint32_t nwins, uint32_t wbits, bool mont);
template __global__
void transform<affine_t>(
    const affine_t* points_in,
    affine_t* prepared_points_out,
    size_t npoints);
template __global__
void glv_split<scalar_t, affine_t>(
    const scalar_t* scalars_in,
    const affine_t* prepared_points_in,
    scalar_t* final_scalars_out,
    affine_t* final_points_out,
    size_t npoints
);
template __global__
void decompose<scalar_t, affine_t>(
    const affine_t* points_in,
    const scalar_t* scalars_in,
    affine_t* final_points_out,
    scalar_t* final_scalars_out,
    size_t npoints);
#endif

#include <vector>
#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

template<class affine_h>
struct point_precomputation_cache {
    static affine_h* s_d_prepared_points;
    static size_t s_original_npoints;
};

template<class affine_h> affine_h* point_precomputation_cache<affine_h>::s_d_prepared_points = nullptr;
template<class affine_h> size_t point_precomputation_cache<affine_h>::s_original_npoints = 0;

template<class scalar_t, class affine_h>
struct pipeline_cache {
    static affine_h* s_d_final_points;
    static scalar_t* s_d_final_scalars;
    static size_t s_original_npoints;
};

template<class scalar_t, class affine_h> affine_h* pipeline_cache<scalar_t, affine_h>::s_d_final_points = nullptr;
template<class scalar_t, class affine_h> scalar_t* pipeline_cache<scalar_t, affine_h>::s_d_final_scalars = nullptr;
template<class scalar_t, class affine_h> size_t pipeline_cache<scalar_t, affine_h>::s_original_npoints = 0;


template<class bucket_t, class point_t, class affine_t, class scalar_t,
         class affine_h = class affine_t::mem_t,
         class bucket_h = class bucket_t::mem_t>
class msm_t {
    const gpu_t& gpu;
    uint32_t wbits, nwins;
    bucket_h *d_buckets;
    vec2d_t<uint32_t> d_hist;

    using point_cache_t = point_precomputation_cache<affine_h>;
    using pipeline_cache_t = pipeline_cache<scalar_t, affine_h>;

    template<typename T> using vec_t = slice_t<T>;

    class result_t {
        bucket_t ret[MSM_NTHREADS/bucket_t::degree][2];
    public:
        result_t() {}
        inline operator decltype(ret)&()                    { return ret;    }
        inline const bucket_t* operator[](size_t i) const   { return ret[i]; }
    };

    constexpr static int lg2(size_t n)
    {   int ret=0; while (n>>=1) ret++; return ret;   }

public:
     static void initialize_device_constants()
    {
        cudaError_t err;
        err = cudaMemcpyToSymbol(pasta_msm::g1, pasta_msm::GLVHostConstants::h_g1, sizeof(pasta_msm::g1));
        if (err != cudaSuccess) { printf("ERROR: cudaMemcpyToSymbol failed for g1\n"); }
        err = cudaMemcpyToSymbol(pasta_msm::g2, pasta_msm::GLVHostConstants::h_g2, sizeof(pasta_msm::g2));
        if (err != cudaSuccess) { printf("ERROR: cudaMemcpyToSymbol failed for g2\n"); }
        err = cudaMemcpyToSymbol(pasta_msm::Pallas_a1, pasta_msm::GLVHostConstants::h_Pallas_a1, sizeof(pasta_msm::Pallas_a1));
        if (err != cudaSuccess) { printf("ERROR: cudaMemcpyToSymbol failed for Pallas_a1\n"); }
        err = cudaMemcpyToSymbol(pasta_msm::Pallas_a2, pasta_msm::GLVHostConstants::h_Pallas_a2, sizeof(pasta_msm::Pallas_a2));
        if (err != cudaSuccess) { printf("ERROR: cudaMemcpyToSymbol failed for Pallas_a2\n"); }
        err = cudaMemcpyToSymbol(pasta_msm::Pallas_b1, pasta_msm::GLVHostConstants::h_Pallas_b1, sizeof(pasta_msm::Pallas_b1));
        if (err != cudaSuccess) { printf("ERROR: cudaMemcpyToSymbol failed for Pallas_b1\n"); }
        err = cudaMemcpyToSymbol(pasta_msm::Pallas_b2, pasta_msm::GLVHostConstants::h_Pallas_b2, sizeof(pasta_msm::Pallas_b2));
        if (err != cudaSuccess) { printf("ERROR: cudaMemcpyToSymbol failed for Pallas_b2\n"); }
        err = cudaMemcpyToSymbol(pasta_msm::beta2, pasta_msm::GLVHostConstants::h_beta2, sizeof(pasta_msm::beta2));
        if (err != cudaSuccess) { printf("ERROR: cudaMemcpyToSymbol failed for beta2\n"); }
    }

    static void cleanup_msm_caches() {
        if (point_cache_t::s_d_prepared_points != nullptr) {
            cudaFree(point_cache_t::s_d_prepared_points);
            point_cache_t::s_d_prepared_points = nullptr;
        }
        point_cache_t::s_original_npoints = 0;
        
        if (pipeline_cache_t::s_d_final_scalars != nullptr) {
            cudaFree(pipeline_cache_t::s_d_final_scalars);
            pipeline_cache_t::s_d_final_scalars = nullptr;
        }
        if (pipeline_cache_t::s_d_final_points != nullptr) {
            cudaFree(pipeline_cache_t::s_d_final_points);
            pipeline_cache_t::s_d_final_points = nullptr;
        }
        pipeline_cache_t::s_original_npoints = 0;
    }

    msm_t(const affine_t[] /*points*/, size_t /*np*/,
          size_t /*ffi_affine_sz*/, int device_id = -1)
        : gpu(select_gpu(device_id))
    {
        static bool constants_initialized = false;
        if (!constants_initialized) {
            initialize_device_constants();
            constants_initialized = true;
        }

        wbits = 16;
        nwins = 9;
        
        uint32_t row_sz = 1U << (wbits-1);
        size_t d_buckets_sz = (nwins * row_sz)
                            + (gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
        size_t d_blob_sz = (d_buckets_sz * sizeof(d_buckets[0]))
                         + (nwins * row_sz * sizeof(uint32_t));
        d_buckets = reinterpret_cast<decltype(d_buckets)>(gpu.Dmalloc(d_blob_sz));
        d_hist = vec2d_t<uint32_t>(&d_buckets[d_buckets_sz], row_sz);
    }
    inline msm_t(vec_t<affine_t> points, size_t ffi_affine_sz = sizeof(affine_t),
                 int device_id = -1)
        : msm_t(points.data(), points.size(), ffi_affine_sz, device_id) {};
    inline msm_t(int device_id = -1)
        : msm_t(nullptr, 0, 0, device_id) {};
    ~msm_t()
    {
        gpu.sync();
        if (d_buckets) gpu.Dfree(d_buckets);
    }

private:
    void digits(const scalar_t d_scalars[], size_t len,
                vec2d_t<uint32_t>& d_digits, vec2d_t<uint2>&d_temps, bool mont)
    {
        uint32_t grid_size = gpu.sm_count() / 3;
        while (grid_size & (grid_size - 1))
            grid_size -= (grid_size & (0 - grid_size));

        breakdown<<<2*grid_size, 512, sizeof(scalar_t)*1024, gpu[2]>>>(
            d_digits, d_scalars, len, nwins, wbits, mont
        );
        CUDA_OK(cudaGetLastError());
        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
        uint32_t top = size_t(143) - wbits * (nwins-1);
        uint32_t win;
        for (win = 0; win < nwins-1; win += 2) {
            gpu[2].launch_coop(sort, {{grid_size, 2}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, win == nwins-2 ? top-1 : wbits-1);
        }
        if (win < nwins) {
            gpu[2].launch_coop(sort, {{grid_size, 1}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
        }
    }
public:
RustError invoke(point_t& out, const affine_t* points_, size_t npoints_in,
                 const scalar_t* scalars, bool mont = true,
                 size_t ffi_affine_sz = sizeof(affine_t))
{
    try {
        if (npoints_in == 0) {
            out.inf();
            return RustError{cudaSuccess};
        }

        if (pipeline_cache_t::s_d_final_scalars == nullptr || pipeline_cache_t::s_original_npoints != npoints_in) {
            cleanup_msm_caches();

            CUDA_OK(cudaMalloc(&pipeline_cache_t::s_d_final_scalars, npoints_in * 2 * sizeof(scalar_t)));
            CUDA_OK(cudaMalloc(&pipeline_cache_t::s_d_final_points, npoints_in * 2 * sizeof(affine_h)));
            pipeline_cache_t::s_original_npoints = npoints_in;

            affine_t* d_input_points_tmp = nullptr;
            scalar_t* d_input_scalars_tmp = nullptr;
            CUDA_OK(cudaMalloc(&d_input_points_tmp, npoints_in * sizeof(affine_t)));
            CUDA_OK(cudaMalloc(&d_input_scalars_tmp, npoints_in * sizeof(scalar_t)));

            gpu[0].HtoD(d_input_points_tmp, points_, npoints_in, ffi_affine_sz);
            gpu[0].HtoD(d_input_scalars_tmp, scalars, npoints_in);
            const uint32_t block_size = 256;
            const uint32_t grid_size = (npoints_in + block_size - 1) / block_size;

            decompose<scalar_t, affine_t><<<grid_size, block_size, 0, gpu[0]>>>(
                d_input_points_tmp,
                d_input_scalars_tmp,
                (affine_t*)pipeline_cache_t::s_d_final_points,
                pipeline_cache_t::s_d_final_scalars,
                npoints_in
            );
            CUDA_OK(cudaGetLastError());
            gpu[0].sync();

            CUDA_OK(cudaFree(d_input_points_tmp));
            CUDA_OK(cudaFree(d_input_scalars_tmp));
        }

        size_t npoints = npoints_in * 2;
        scalar_t* d_glv_scalars = pipeline_cache_t::s_d_final_scalars;
        affine_h* d_glv_points = pipeline_cache_t::s_d_final_points;

        uint32_t lg_npoints = lg2(npoints + npoints / 2);
        size_t batch = 1 << (std::max(lg_npoints, wbits) - wbits);
        batch >>= 6;
        batch = batch ? batch : 1;
        uint32_t stride = (npoints + batch - 1) / batch;
        stride = (stride + WARP_SZ - 1) & ((size_t)0 - WARP_SZ);

        std::vector<result_t> res(nwins);
        std::vector<bucket_t> ones(gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);

        out.inf();
        point_t p;

        size_t temp_sz = stride * 2 * sizeof(uint2);
        size_t digits_sz = nwins * stride * sizeof(uint32_t);
        dev_ptr_t<uint8_t> d_temp{temp_sz + digits_sz, gpu[2]};

        vec2d_t<uint2> d_temps{&d_temp[0], stride};
        vec2d_t<uint32_t> d_digits{&d_temp[temp_sz], stride};
        
        size_t d_src_off = 0;
        size_t num = std::min(static_cast<size_t>(stride), npoints);

        event_t ev;
        event_t dtoh_done[2];

        digits(&d_glv_scalars[d_src_off], num, d_digits, d_temps, mont);
        gpu[2].record(ev);

        for (uint32_t i = 0; i < batch; i++) {
            int stream_idx = i & 1;

            if (i > 0) {
                cudaEventSynchronize(dtoh_done[(i - 1) & 1]);
                collect(p, res, ones);
                out.add(p);
            }
            
            gpu[stream_idx].wait(ev);

            affine_h* d_current_points = &d_glv_points[d_src_off];

            batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE, 0, gpu[stream_idx]>>>(
                &d_buckets[nwins << (wbits - 1)], d_current_points, num,
                &d_digits[0][0], d_hist[0][0]
            );
            CUDA_OK(cudaGetLastError());
            
            gpu[stream_idx].launch_coop(accumulate<bucket_t, affine_h>,
                launch_params_t(gpu.sm_count(), ACCUMULATE_NTHREADS),
                d_buckets, nwins, wbits, d_current_points, d_digits, d_hist, static_cast<uint32_t>(stream_idx)
            );

            integrate<bucket_t><<<nwins, MSM_NTHREADS,
                                  sizeof(bucket_t) * MSM_NTHREADS / bucket_t::degree,
                                  gpu[stream_idx]>>>(
                d_buckets, nwins, wbits, size_t(143)
            );
            CUDA_OK(cudaGetLastError());
            
            gpu[stream_idx].DtoH(ones, d_buckets + (nwins << (wbits - 1)));
            gpu[stream_idx].DtoH(res, d_buckets, sizeof(bucket_h) << (wbits - 1));
            gpu[stream_idx].record(dtoh_done[stream_idx]);
            
            if (i < batch - 1) {
                d_src_off += stride;
                num = std::min(static_cast<size_t>(stride), npoints - d_src_off);
                digits(&d_glv_scalars[d_src_off], num, d_digits, d_temps, mont);
                gpu[2].record(ev);
            }
        }

        cudaEventSynchronize(dtoh_done[(batch - 1) & 1]);
        collect(p, res, ones);
        out.add(p);

        return RustError{cudaSuccess};

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}


    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points.data(), points.size(), scalars, mont, ffi_affine_sz);   }

    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   vec_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points.data(), points.size(), scalars.data(), mont, ffi_affine_sz);   }

    RustError invoke(point_t& out, const std::vector<affine_t>& points,
                                   const std::vector<scalar_t>& scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        return invoke(out, points.data(),
                           std::min(points.size(), scalars.size()),
                           scalars.data(), mont, ffi_affine_sz);
    }

private:
    point_t integrate_row(const result_t& row, uint32_t lsbits)
    {
        const int NTHRBITS = lg2(MSM_NTHREADS/bucket_t::degree);
        assert(wbits-1 > NTHRBITS);
        size_t i = MSM_NTHREADS/bucket_t::degree - 1;

        if (lsbits-1 <= NTHRBITS) {
            size_t mask = (1U << (NTHRBITS-(lsbits-1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask)   res.inf();
            else        res = acc;

            while (i--) {
                acc.add(row[i][1]);
                if ((i & mask) == 0)
                    res.add(acc);
            }
            return res;
        }

        point_t  res = row[i][0];
        bucket_t acc = row[i][1];
        while (i--) {
            point_t raise = acc;
            for (size_t j = 0; j < lsbits-1-NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(point_t{row[i][0]});
            if (i)
                acc.add(row[i][1]);
        }
        return res;
    }

    void collect(point_t& out, const std::vector<result_t>& res,
                               const std::vector<bucket_t>& ones)
    {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);

        uint32_t y = nwins-1, total = 0;

        grid[0].x  = 0;
        grid[0].y  = y;
        grid[0].dy = size_t(143) - y*wbits;
        total++;

        while (y--) {
            grid[total].x  = grid[0].x;
            grid[total].y  = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins);
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--) {
            gpu.spawn([&, this, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res[y], item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones);

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        out.add(one);
    }
};

template<class bucket_t, class point_t, class affine_t, class scalar_t> static
RustError mult_pippenger(point_t *out, const affine_t points[], size_t npoints,
                                       const scalar_t scalars[], bool mont = true,
                                       size_t ffi_affine_sz = sizeof(affine_t))
{
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm(points, npoints, ffi_affine_sz);
        return msm.invoke(*out, points, npoints, scalars, mont, ffi_affine_sz);
    } catch (const cuda_error& e) {
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}
#endif