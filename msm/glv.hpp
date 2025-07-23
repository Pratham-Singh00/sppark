// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: psingh-7
// Date: 2025-07-22

#ifndef __SPPARK_MSM_GLV_HPP__
#define __SPPARK_MSM_GLV_HPP__

#include <cstdint>
#include <array>

namespace pasta_msm {

class GLVConstants {
public:
    // Constants for GLV decomposition in Pallas curve
    static constexpr uint32_t lambda[8] = {
        0x50aa0e4f, 0x2aa9d2e0, 0x47c033af, 0x0fed467d,
        0x1cf70f5a, 0x511db4d8, 0x283e528e, 0x06819a58
    };

    static constexpr uint32_t beta[8] = {
        0xfdfe4ab9, 0x1dad5ebd, 0x37ad3149, 0x1d1f8bd2,
        0x57aab1b0, 0x2caad5dc, 0x4acdba71, 0x12ccca83
    };

    static constexpr uint32_t Pallas_a1[8] = {
        0x00000001, 0x7fcae1c7, 0x40f04915, 0x49e69d16,
        0x00000000, 0x00000000, 0x00000000, 0x00000000
    };

    static constexpr uint32_t Pallas_a2[8] = {
        0x00000000, 0x8cb12793, 0x40a89953, 0x49e69d16,
        0x00000000, 0x00000000, 0x00000000, 0x00000000
    };

    static constexpr uint32_t Pallas_b1[8] = {
        0x00000000, 0x8cb12793, 0x40a89953, 0x49e69d16,
        0x00000000, 0x00000000, 0x00000000, 0x00000000
    };

    static constexpr uint32_t Pallas_b2[8] = {
        0x00000001, 0x0c7c095a, 0x8198e269, 0x93cd3a2c,
        0x00000000, 0x00000000, 0x00000000, 0x00000000
    };

    static constexpr uint32_t g1[8] = {
        0x111f6861, 0x086862e0, 0xc35fbd4d, 0x00000002,
        0x31f02568, 0x066389a4, 0x4f34e8b2, 0x00000002
    };

    static constexpr uint32_t g2[8] = {
        0x4a95a2d9, 0x8480fa55, 0x61afdea6, 0xffffffff,
        0x32c49e4b, 0x02a2654e, 0x279a7459, 0x00000001
    };
};

// GLV decomposition function
inline void glv_split(const uint32_t k[8], uint32_t k1[8], uint32_t k2[8]) {
    uint32_t tmp1[16]={0}, tmp2[16]={0};
    for(int i=0;i<8;++i){
        k1[i]=k[i];
    }
    for(int i=0;i<8;i++){
        uint64_t c1=0,c2=0;
        for(int j=0;j<8;j++){
            uint64_t p1 = uint64_t(k[i])*GLVConstants::g1[j] + tmp1[i+j] + c1;
            uint64_t p2 = uint64_t(k[i])*GLVConstants::g2[j] + tmp2[i+j] + c2;
            tmp1[i+j] = uint32_t(p1);
            tmp2[i+j] = uint32_t(p2);
            c1 = p1>>32;
            c2 = p2>>32;
        }
        tmp1[i+8] = uint32_t(c1);
        tmp2[i+8] = uint32_t(c2);
    }
    tmp1[11]+=((tmp1[10]&(1ULL<<31))>0);
    tmp2[11]+=((tmp2[10]&(1ULL<<31))>0);
    
    uint32_t c1[4]={0}, c2[4]={0};
    for(int i=0;i<4;i++){
        c1[i] = tmp1[11+i];
        c2[i] = tmp2[11+i];
    }
    uint32_t c1a1[8]={0},c2a2[8]={0},c1b1[8]={0},c2b2[8]={0};
    
    for(int t=0;t<2;t++){
        const uint32_t* C = (!t?c1:c2);
        const uint32_t* A = (!t?GLVConstants::Pallas_a1:GLVConstants::Pallas_a2);
        const uint32_t* B = (!t?GLVConstants::Pallas_b1:GLVConstants::Pallas_b2);
        for(int i=0;i<4;i++){
            uint64_t car1=0,car2=0;
            for(int j=0;j<4;j++){
                uint64_t v1=uint64_t(C[j])*A[i]+car1;
                uint64_t v2=uint64_t(C[j])*B[i]+car2;
                (!t?v1+=c1a1[i+j]:v1+=c2a2[i+j]);
                (!t?v2+=c1b1[i+j]:v2+=c2b2[i+j]);
                (!t?c1a1[i+j]=uint32_t(v1):c2a2[i+j]=uint32_t(v1));
                (!t?c1b1[i+j]=uint32_t(v2):c2b2[i+j]=uint32_t(v2));
                car1=v1>>32;
                car2=v2>>32;
            }
            (!t?c1a1[i+4]=car1:c2a2[i+4]=car1);
            (!t?c1b1[i+4]=car2:c2b2[i+4]=car2);
        }
    }
    
    bool car1=true,car2=true,car3=true;
    for(int i=0;i<8;++i){
        c1a1[i]=~c1a1[i];c2a2[i]=~c2a2[i];c2b2[i]=~c2b2[i];
        if(car1){c1a1[i]++;car1=(c1a1[i]==0);}
        if(car2){c2a2[i]++;car2=(c2a2[i]==0);}
        if(car3){c2b2[i]++;car3=(c2b2[i]==0);}
    }
    
    for(int i=0;i<8;++i){
        uint64_t tmp=(uint64_t(k1[i])+c1a1[i]+c2a2[i]);
        uint64_t tmp2=(uint64_t(k2[i])+c2b2[i]+c1b1[i]);
        k1[i]=uint32_t(tmp);
        k2[i]=uint32_t(tmp2);
        if(i+1<8){
            k1[i+1]+=((tmp&(3ULL<<32))>>32);
            k2[i+1]+=((tmp2&(1ULL<<32))>>32);
        }
    }
}
//k1 P
//k2 (beta*x,y)
//add 1 to scalar, add the first 4 limbs to bucket but add (x,-y)
template<typename PointT>
inline void transform_point_glv(const typename PointT::affine_t& in,
                              typename PointT::affine_t& out) {
    // Compute λP = (βx, y) efficiently
    out.Y = in.Y;  // y-coordinate remains unchanged
    
    // Load beta constant and compute βx
    typename PointT::field_t beta;
    for (size_t i = 0; i < 8; i++) {
        beta.val[i] = GLVConstants::beta[i];
    }
    out.X = in.X * beta;
}

} // namespace pasta_msm

#endif
