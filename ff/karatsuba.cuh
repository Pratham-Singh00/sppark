#pragma once
#include <cstdint>
#include <cuda.h>

__device__ __forceinline__
void mul2(uint64_t out[4], const uint64_t a[2], const uint64_t b[2]) {
    unsigned __int128 t0 = (unsigned __int128)a[0] * b[0];
    unsigned __int128 t1 = (unsigned __int128)a[0] * b[1]
                         + (unsigned __int128)a[1] * b[0];
    unsigned __int128 t2 = (unsigned __int128)a[1] * b[1];
    out[0] = (uint64_t)t0;
    t1 += t0 >> 64;    out[1] = (uint64_t)t1;
    t2 += t1 >> 64;    out[2] = (uint64_t)t2;
    out[3] = (uint64_t)(t2 >> 64);
}

__device__ __forceinline__
void add2(uint64_t out[3], const uint64_t a[2], const uint64_t b[2]) {
    unsigned __int128 s0 = (unsigned __int128)a[0] + b[0];
    unsigned __int128 s1 = (unsigned __int128)a[1] + b[1] + (s0 >> 64);
    out[0] = (uint64_t)s0;
    out[1] = (uint64_t)s1;
    out[2] = (uint64_t)(s1 >> 64);
}

__device__ __forceinline__
void sub4(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
    unsigned __int128 d0 = (unsigned __int128)a[0] - b[0];
    unsigned __int128 d1 = (unsigned __int128)a[1] - b[1] - (uint64_t)(d0 >> 127);
    unsigned __int128 d2 = (unsigned __int128)a[2] - b[2] - (uint64_t)(d1 >> 127);
    unsigned __int128 d3 = (unsigned __int128)a[3] - b[3] - (uint64_t)(d2 >> 127);
    out[0] = (uint64_t)d0;
    out[1] = (uint64_t)d1;
    out[2] = (uint64_t)d2;
    out[3] = (uint64_t)d3;
}

__device__ __forceinline__
void karatsuba4(uint64_t out[8],
                const uint64_t A[4],
                const uint64_t B[4]) {
    uint64_t z0[4], z2[4], z1[4], sumA[3], sumB[3];
    mul2(z0, A+0, B+0);
    mul2(z2, A+2, B+2);
    add2(sumA, A+0, A+2);
    add2(sumB, B+0, B+2);
    mul2(z1, sumA, sumB);
    sub4(z1, z1, z0);
    sub4(z1, z1, z2);
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) out[i] += z0[i];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      unsigned __int128 t = (unsigned __int128)out[i+2] + z1[i];
      out[i+2] = (uint64_t)t;
      out[i+3] += (uint64_t)(t>>64);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      unsigned __int128 t = (unsigned __int128)out[i+4] + z2[i];
      out[i+4] = (uint64_t)t;
      if (i+5 < 8) out[i+5] += (uint64_t)(t>>64);
    }
}

