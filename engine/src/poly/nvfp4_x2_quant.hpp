// src/poly/nvfp4_x2_quant.hpp - the rounding of nvfp4_x2.hpp's activation rows, the same on the card and the host:
// CUDA's FP8 and FP4 conversions, which round to nearest even and saturate on both, and every other operation
// rounded once, as written (no contraction, no fast-math approximation).
//
// Built with -use_fast_math (as PR 202 built it; strata_poly.cmake does not), -ftz=true would flush subnormal
// inputs and results to zero (keeping their sign) where the host does not.  The two would still agree: a row whose
// largest |value| is under kTinyRow is written as zeros, and above it a step that could come out subnormal is one
// that a conversion rounds to zero anyway (a UE4M3 scale or an E2M1 code under half its smallest value) or a
// remainder a term then rounds to zero.  A code of zero is written +0: a tiny negative value is -0 on both sides,
// but its remainder is -0 - -0 = +0 once flushed.
//
// Non-finite values: a row with a NaN or an infinity anywhere is written as zeros with a NaN row scale, so its
// products are NaN, as they would be in floats.  Both sides find it with the same maximum (max_abs()).
//
// <cuda_fp4.h> came with CUDA 12.8; an older toolkit has neither it nor the 12xa target, so nothing here is built
// (STRATA_X2_HAS_FP4 is not defined) and nvfp4_x2_built() is false.
//
// Two-term FP4 activations and parts of the approach are informed by Basalt's src/prefill/moe_fp4.cu
// (github.com/jesdga95/basalt), which is under the MIT license:
// MIT License
// Copyright (c) 2026 Jesús Gulfo
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include "nvfp4_x2.hpp"

#include <cuda_runtime_api.h>

#include <cmath>

#if CUDART_VERSION >= 12080
#define STRATA_X2_HAS_FP4 1
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace strata::poly::x2 {

#if defined(__CUDA_ARCH__)
STRATA_X2_HD float mul_rn(float a, float b) { return __fmul_rn(a, b); }
STRATA_X2_HD float sub_rn(float a, float b) { return __fsub_rn(a, b); }
STRATA_X2_HD float div_rn(float a, float b) { return __fdiv_rn(a, b); }
#else
inline float mul_rn(float a, float b) { return a * b; }
inline float sub_rn(float a, float b) { return a - b; }
inline float div_rn(float a, float b) { return a / b; }
#endif

STRATA_X2_HD float abs_f(float a) { return a < 0.0f ? -a : a; }
/// The larger of m and |v|, NaN once either is NaN: the rows' and the sub-blocks' maximum on both sides.
STRATA_X2_HD float max_abs(float m, float v) {
    const float a = abs_f(v);
    return (a > m || a != a) ? a : m;
}

/// A row whose largest |value| is under this is written as zeros (2^-100: its scale and steps stay normal floats).
constexpr float kTinyRow = 7.88860905e-31f;

// 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (without a table: a table would be local memory on the card)
STRATA_X2_HD float e2m1_value(unsigned code) {
    const unsigned m = code & 7u;
    const float mag = m < 2u ? 0.5f * (float) m : (float) (2u + (m & 1u)) * (float) (1u << (m >> 1)) * 0.25f;
    return (code & 8u) ? -mag : mag;
}

STRATA_X2_HD float ue4m3_value(uint8_t s) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) s, __NV_E4M3)));
}

/// The row's scale: its largest |value| goes to the largest code times the largest block scale.  0 for a row under
/// kTinyRow, NaN for a row with a NaN or an infinity (m is max_abs() over the row): both are written as zeros.
STRATA_X2_HD float row_scale(float m) {
    if (!(m <= 3.40282347e38f)) return NAN;
    return m < kTinyRow ? 0.0f : div_rn(m, kTopCode * kTopScale);
}

/// The host's row_scale() of a row of cols values (the card takes the maximum across a warp, which is the same).
inline float host_row_scale(const float* x, int64_t cols) {
    float m = 0.0f;
    for (int64_t i = 0; i < cols; ++i) m = max_abs(m, x[i]);
    return row_scale(m);
}

/// One term of a 16-value sub-block of a row with scale g > 0: its UE4M3 scale byte and its 8 code bytes in
/// block_nvfp4's order, and v less what they encode.
STRATA_X2_HD void term16(float* v, float g, uint8_t& d, uint8_t* q) {
    float m = 0.0f;
    for (int j = 0; j < 16; ++j) m = max_abs(m, v[j]);
    d = (uint8_t) __nv_cvt_float_to_fp8(div_rn(m, mul_rn(kTopCode, g)), __NV_SATFINITE, __NV_E4M3);
    const float step = mul_rn(ue4m3_value(d), g);
    for (int j = 0; j < 8; ++j) {
        uint8_t b = 0;
        if (step > 0.0f) {
            float2 p;
            p.x = div_rn(v[j], step);
            p.y = div_rn(v[j + 8], step);
            b = (uint8_t) __nv_cvt_float2_to_fp4x2(p, __NV_E2M1, cudaRoundNearest);
            if ((b & 0x07u) == 0) b &= 0xf0u;   // a code of -0 as +0 (see the top)
            if ((b & 0x70u) == 0) b &= 0x0fu;
        }
        q[j] = b;
        v[j] = sub_rn(v[j], mul_rn(e2m1_value(b & 15u), step));
        v[j + 8] = sub_rn(v[j + 8], mul_rn(e2m1_value(b >> 4), step));
    }
}

/// Sub-block s (values 16 s .. 16 s + 15) of a row with scale g into the row's two blocks (nvfp4_x2.hpp's format).
STRATA_X2_HD void quantize16(const float* x, float g, uint8_t* row, int s) {
    float v[16];
    for (int j = 0; j < 16; ++j) v[j] = x[j];
    uint8_t* pair = row + (size_t) (s / 4) * kPairBytes;
    const int sub = s % 4;
    if (!(g > 0.0f)) {
        pair[sub] = pair[kBlockBytes + sub] = 0;
        for (int j = 0; j < 8; ++j) pair[4 + 8 * sub + j] = pair[kBlockBytes + 4 + 8 * sub + j] = 0;
        return;
    }
    term16(v, g, pair[sub], pair + 4 + 8 * sub);
    term16(v, g, pair[kBlockBytes + sub], pair + kBlockBytes + 4 + 8 * sub);
}

}  // namespace strata::poly::x2

#endif  // CUDART_VERSION >= 12080
