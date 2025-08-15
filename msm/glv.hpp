// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_GLV_HPP__
#define __SPPARK_MSM_GLV_HPP__

#include <cstdint>
#include <array>
#include <utility>
#include <vector>
#include "../ec/affine_t.hpp"
#include <iostream>

namespace pasta_msm {
__device__ __constant__ uint32_t g1[8];
__device__ __constant__ uint32_t g2[8];
__device__ __constant__ uint32_t Pallas_a1[8];
__device__ __constant__ uint32_t Pallas_a2[8];
__device__ __constant__ uint32_t Pallas_b1[8];
__device__ __constant__ uint32_t Pallas_b2[8];
__device__ __constant__ uint32_t beta2[8];

struct GLVHostConstants {
    static const uint32_t h_g1[8];
    static const uint32_t h_g2[8];
    static const uint32_t h_Pallas_a1[8];
    static const uint32_t h_Pallas_a2[8];
    static const uint32_t h_Pallas_b1[8];
    static const uint32_t h_Pallas_b2[8];
    static const uint32_t h_beta2[8];
};
const uint32_t GLVHostConstants::h_g1[8] = {
        0x111f6861, 0x086862e0, 0xc35fbd4d, 0x00000002,
        0x31f02568, 0x066389a4, 0x4f34e8b2, 0x00000002
    };
const uint32_t GLVHostConstants::h_g2[8] = {
        0x4a95a2d9, 0x8480fa55, 0x61afdea6, 0xffffffff,
        0x32c49e4b, 0x02a2654e, 0x279a7459, 0x00000001
    };
const uint32_t GLVHostConstants::h_Pallas_a1[8] = {
        0x00000001, 0x7fcae1c7, 0x40f04915, 0x49e69d16,
        0x00000000, 0x00000000, 0x00000000, 0x00000000
    };
const uint32_t GLVHostConstants::h_Pallas_a2[8] = {
        0x00000000, 0x8cb12793, 0x40a89953, 0x49e69d16,
        0x00000000, 0x00000000, 0x00000000, 0x00000000
    };
const uint32_t GLVHostConstants::h_Pallas_b1[8] = {
        0x00000000, 0x8cb12793, 0x40a89953, 0x49e69d16,
        0x00000000, 0x00000000, 0x00000000, 0x00000000
    };
const uint32_t GLVHostConstants::h_Pallas_b2[8] = {
        0x00000001, 0x0c7c095a, 0x8198e269, 0x93cd3a2c,
        0x00000000, 0x00000000, 0x00000000, 0x00000000
    };
const uint32_t GLVHostConstants::h_beta2[8] = {
        0x619a153d,0x02021cf6,0x4980b78e,0x9e8c2697,
        0xc87a4666,0x2a676d5c,0xa7a17876,0x15d8049d
    };


struct DecomposedScalar {
    uint32_t k[4];
    bool is_negative = false;
};

inline __host__ __device__
void glv_split(const uint8_t v[32], uint32_t thread_idx,
             DecomposedScalar& r1, DecomposedScalar& r2) {
#if !defined(__CUDA_ARCH__)
    (void)v;
    (void)thread_idx;
    (void)r1;
    (void)r2;
#else
    uint32_t kl[8];
    uint32_t k1_limbs[8] = {0}, k2_limbs[8] = {0};

    for(size_t i = 0; i < 8; i++){
        uint32_t currlimb=0;
        for(size_t j = 0; j < 4; j++){
            currlimb+=(((uint32_t)v[i*4+j])<<(8*j));
        }
        kl[i]=currlimb;
    }

    for(int i=0;i<8;++i){
        k1_limbs[i]=kl[i];
    }
    uint32_t tmp1[16] = {0}, tmp2[16] = {0};
    for(int i = 0; i < 8; i++) {
        uint64_t carry1 = 0, carry2 = 0;
        for(int j = 0; j < 8; j++) {
            uint64_t p1 = (uint64_t)kl[i] * g1[j] + tmp1[i+j] + carry1;
            uint64_t p2 = (uint64_t)kl[i] * g2[j] + tmp2[i+j] + carry2;
            tmp1[i+j] = (uint32_t)p1;
            tmp2[i+j] = (uint32_t)p2;
            carry1 = p1 >> 32;
            carry2 = p2 >> 32;
        }
        for (int k = i + 8; k < 16 && (carry1 > 0 || carry2 > 0); k++) {
            uint64_t sum1 = tmp1[k] + carry1;
            uint64_t sum2 = tmp2[k] + carry2;
            tmp1[k] = (uint32_t)sum1;
            tmp2[k] = (uint32_t)sum2;
            carry1 = sum1 >> 32;
            carry2 = sum2 >> 32;
        }
    }
    tmp1[11] += tmp1[10] >> 31;
    tmp2[11] += tmp2[10] >> 31;
    uint32_t c1_[4], c2_[4];
    for(int i = 0; i < 4; i++) {
        c1_[i] = tmp1[11 + i];
        c2_[i] = tmp2[11 + i];
    }
    uint32_t c1a1[8] = {0}, c2a2[8] = {0}, c1b1[8] = {0}, c2b2[8] = {0};
    for(int t = 0; t < 2; t++) {
        const uint32_t* C = (t == 0) ? c1_ : c2_;
        const uint32_t* A = (t == 0) ? Pallas_a1 : Pallas_a2;
        const uint32_t* B = (t == 0) ? Pallas_b1 : Pallas_b2;
        uint32_t* X = (t == 0) ? c1a1 : c2a2;
        uint32_t* Y = (t == 0) ? c1b1 : c2b2;
        for(int i = 0; i < 4; i++) {
            uint64_t carry1 = 0, carry2 = 0;
            for(int j = 0; j < 4; j++) {
                uint64_t v1 = (uint64_t)C[j] * A[i] + X[i+j] + carry1;
                uint64_t v2 = (uint64_t)C[j] * B[i] + Y[i+j] + carry2;
                X[i+j] = (uint32_t)v1;
                Y[i+j] = (uint32_t)v2;
                carry1 = v1 >> 32;
                carry2 = v2 >> 32;
            }
            X[i+4] = (uint32_t)carry1;
            Y[i+4] = (uint32_t)carry2;
        }
    }
    bool carry1 = true, carry2 = true, carry3 = true;
    for(int i = 0; i < 8; i++) {
        c1a1[i] = ~c1a1[i]; c2a2[i] = ~c2a2[i]; c2b2[i] = ~c2b2[i];
        if(carry1) { c1a1[i]++; carry1 = (c1a1[i] == 0); }
        if(carry2) { c2a2[i]++; carry2 = (c2a2[i] == 0); }
        if(carry3) { c2b2[i]++; carry3 = (c2b2[i] == 0); }
    }
    uint64_t carry_t1 = 0;
    uint64_t carry_t2 = 0;
    for(int i = 0; i < 8; i++) {
        uint64_t t1 = (uint64_t)k1_limbs[i] + c1a1[i] + c2a2[i] + carry_t1;
        uint64_t t2 = (uint64_t)k2_limbs[i] + c2b2[i] + c1b1[i] + carry_t2;
        k1_limbs[i] = (uint32_t)t1;
        k2_limbs[i] = (uint32_t)t2;
        carry_t1 = t1 >> 32;
        carry_t2 = t2 >> 32;
    }
    r1.is_negative = (k1_limbs[4]==0xffffffff);
    r2.is_negative = (k2_limbs[4]==0xffffffff);

    if (r1.is_negative) {
        uint64_t carry = 1;
        for (int i = 0; i < 4; i++) {
            uint64_t neg_k = (uint64_t)(~k1_limbs[i]) + carry;
            r1.k[i] = (uint32_t)neg_k;
            carry = neg_k >> 32;
        }
    } else {
        for (int i = 0; i < 4; i++) r1.k[i] = k1_limbs[i];
    }
    if (r2.is_negative) {
        uint64_t carry = 1;
        for (int i = 0; i < 4; i++) {
            uint64_t neg_k = (uint64_t)(~k2_limbs[i]) + carry;
            r2.k[i] = (uint32_t)neg_k;
            carry = neg_k >> 32;
        }
    } else {
        for (int i = 0; i < 4; i++) r2.k[i] = k2_limbs[i];
    }
#endif
}

template<typename PointT>
inline __host__ __device__
void transform_point_glv(const PointT& in, PointT& out) {
    out=in;
    decltype(in.X) beta_field;
    for(size_t i = 0; i < sizeof(beta2)/sizeof(uint32_t); i++) {
        beta_field[i] = beta2[i];
    }
    out.X *= beta_field;
}

}
#endif