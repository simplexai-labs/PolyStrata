// src/poly/cpu_lowbit.cpp - see the header.  The kernels are compiled for AVX2 + FMA + F16C by target attribute
// and chosen at run time, so a build for older CPUs never runs them; a build for another architecture has none.
//
// IQ1_S: ggml-cpu's AVX2 vec_dot_iq1_s_q8_K (ggml/src/ggml-cpu/arch/x86/quants.c, MIT) with the same integer sums
// and float steps, but without BMI2's pdep, which Zen 1 and 2 run in microcode (~250 cycles each, two per 64
// values).  Q2_0: ggml's generic vec_dot_q2_0_q8_0 (ggml-cpu/quants.c) in AVX2.
#include "cpu_lowbit.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#define STRATA_LOWBIT_X86 1
#endif

#if STRATA_LOWBIT_X86
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOWBIT_FN __attribute__((target("avx2,fma,f16c")))
#else
#define LOWBIT_FN
#endif
#endif

namespace strata::poly {
#if STRATA_LOWBIT_X86
namespace {

#if defined(_MSC_VER) && !defined(__clang__)
// MSVC has no _cvtsh_ss: the same F16C conversion through the vector intrinsic.
inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
#else
LOWBIT_FN inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
#endif

LOWBIT_FN inline float hsum_float_8(const __m256 x) {   // ggml's
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

LOWBIT_FN inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {   // ggml's: pairs of x * y in int16
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

// ---- Q2_0 x Q8_0 -------------------------------------------------------------------------------------------------

// One Q8_0 block's 8 partial sums of (code - 1) * q, the block's 32 codes in the 8 bytes at `codes` (value
// 4i + m in byte i, bits 2m).  The activation's bytes go to "plane" order inside each 128-bit half (dword m holds
// values 4i + m, i = 0..3: one pshufb), where the codes of a plane are a byte shifted by 2m and masked: no
// interleave.  sum (c - 1) q = sum c q - sum q, both in int16 pairs (|pair| <= 6 * 127), then madd.
LOWBIT_FN inline __m256i q8_block_sums(const uint8_t* codes, const int8_t* q) {
    const __m256i plane = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                           0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i qp = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*) q), plane);
    uint64_t c8;
    std::memcpy(&c8, codes, 8);
    const __m256i cb = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_cvtsi64_si128((long long) c8)),
                                                   _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));
    const __m256i c = _mm256_and_si256(_mm256_srlv_epi32(cb, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)),
                                       _mm256_set1_epi8(3));
    const __m256i cq = _mm256_maddubs_epi16(c, qp);
    const __m256i sq = _mm256_maddubs_epi16(_mm256_set1_epi8(1), qp);
    return _mm256_madd_epi16(_mm256_sub_epi16(cq, sq), _mm256_set1_epi16(1));
}

// As ggml's AVX2 dot products of the other types do it: each Q8_0 block's 8 partial sums go into an 8-lane float
// accumulator with the block's two scales, fma(d_x * d_y, float(sums), acc), and the lanes are added once at the
// end.  The last bits differ from ggml's generic C (the sums go in another order).
LOWBIT_FN void vec_dot_q2_0_q8_0(int n, float* s, size_t, const void* vx, size_t, const void* vy, size_t, int) {
    const int nb = n / QK2_0;
    const block_q2_0* x = (const block_q2_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d0 = h2f(x[i].d);
        const __m256 s0 = _mm256_cvtepi32_ps(q8_block_sums(x[i].qs, y[2 * i].qs));
        const __m256 s1 = _mm256_cvtepi32_ps(q8_block_sums(x[i].qs + 8, y[2 * i + 1].qs));
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0 * h2f(y[2 * i].d)), s0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d0 * h2f(y[2 * i + 1].d)), s1, acc1);
    }
    *s = hsum_float_8(_mm256_add_ps(acc0, acc1));
}

// ---- IQ1_S x Q8_K ------------------------------------------------------------------------------------------------

// ggml's AVX2 kernel with its scalar parts done in vectors, the same integers and the same float steps:
//   * the 32 grid indices of a block, qs[k] | ((qh[k / 4] >> 3 (k % 4)) & 7) << 8, as 16-bit lanes at once (left
//     shifts by multiplies: qh << 8, << 5, << 2 by mullo, qh >> 1 by mulhi), then read out for the table loads (scalar
//     loads: AVX2's gathers took 1.25x as long on Zen 2);
//   * sumi1, the delta term: sum over the 8 groups of (bsums[2g] + bsums[2g + 1]) * (+-1) * ls, one madd of the
//     bsums, one product with the signed scales and a sum, an exact integer as ggml's scalar loop's.
LOWBIT_FN void vec_dot_iq1_s_q8_K(int n, float* s, size_t, const void* vx, size_t, const void* vy, size_t, int) {
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    const int nb = n / QK_K;
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    // a group's qh four times: 16-bit lane k of the 16 holds qh[k / 4] (one half of the block's 8 groups)
    const __m256i rep4 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
                                          4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7);
    const __m256i mul_lo = _mm256_setr_epi16(256, 32, 4, 0, 256, 32, 4, 0, 256, 32, 4, 0, 256, 32, 4, 0);
    const __m256i mul_hi = _mm256_setr_epi16(0, 0, 0, -32768, 0, 0, 0, -32768, 0, 0, 0, -32768, 0, 0, 0, -32768);
    const __m256i m700 = _mm256_set1_epi16(0x700);
    for (int i = 0; i < nb; ++i) {
        const int8_t* q8 = y[i].qs;
        const uint16_t* qh = x[i].qh;
        alignas(32) uint16_t idx[32];
        {
            const __m128i qh8 = _mm_loadu_si128((const __m128i*) qh);   // the 8 groups' qh
            const __m256i h0 = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(qh8), rep4);   // groups 0..3
            const __m256i h1 = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_srli_si128(qh8, 8)), rep4);
            const __m256i q0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) x[i].qs));
            const __m256i q1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (x[i].qs + 16)));
            const __m256i i0 = _mm256_or_si256(
                q0, _mm256_and_si256(_mm256_or_si256(_mm256_mullo_epi16(h0, mul_lo), _mm256_mulhi_epu16(h0, mul_hi)),
                                     m700));
            const __m256i i1 = _mm256_or_si256(
                q1, _mm256_and_si256(_mm256_or_si256(_mm256_mullo_epi16(h1, mul_lo), _mm256_mulhi_epu16(h1, mul_hi)),
                                     m700));
            _mm256_store_si256((__m256i*) idx, i0);
            _mm256_store_si256((__m256i*) (idx + 16), i1);
        }
        __m256i sumi = _mm256_setzero_si256();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 4
#endif
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const uint16_t* id = idx + 4 * ib;
            const __m256i q1b_1 =
                _mm256_set_epi64x(iq1s_grid[id[3]], iq1s_grid[id[2]], iq1s_grid[id[1]], iq1s_grid[id[0]]);
            const __m256i q1b_2 =
                _mm256_set_epi64x(iq1s_grid[id[7]], iq1s_grid[id[6]], iq1s_grid[id[5]], iq1s_grid[id[4]]);
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32 * ib));
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32 * ib + 32));
            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2 * ((qh[ib + 0] >> 12) & 7) + 1;
            const int16_t ls2 = 2 * ((qh[ib + 1] >> 12) & 7) + 1;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
        }
        int sumi1;
        {
            const __m256i gs = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) y[i].bsums), _mm256_set1_epi16(1));
            const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) qh));
            const __m256i ls = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 12),
                                                                                   _mm256_set1_epi32(7)), 1),
                                                _mm256_set1_epi32(1));
            const __m256i sl = _mm256_sign_epi32(ls, _mm256_sub_epi32(_mm256_set1_epi32(1),
                                                                     _mm256_slli_epi32(_mm256_srli_epi32(h, 15), 1)));
            const __m256i pr = _mm256_mullo_epi32(gs, sl);
            const __m128i t = _mm_add_epi32(_mm256_castsi256_si128(pr), _mm256_extracti128_si256(pr, 1));
            const __m128i u = _mm_add_epi32(t, _mm_unpackhi_epi64(t, t));
            sumi1 = _mm_cvtsi128_si32(_mm_add_epi32(u, _mm_shuffle_epi32(u, 1)));
        }
        const float d = y[i].d * h2f(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 = std::fmaf(d, (float) sumi1, accum1);
    }
    *s = std::fmaf(IQ1S_DELTA, accum1, hsum_float_8(accum));
}

bool cpu_ok() {
#if defined(__GNUC__) || defined(__clang__)
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                           __builtin_cpu_supports("f16c");
    return ok;
#else
    return false;
#endif
}

}  // namespace

ggml_vec_dot_t lowbit_vec_dot(int type) {
    if (!cpu_ok()) return nullptr;
    if (type == GGML_TYPE_Q2_0) return &vec_dot_q2_0_q8_0;
    if (type == GGML_TYPE_IQ1_S) return &vec_dot_iq1_s_q8_K;
    return nullptr;
}
#else
ggml_vec_dot_t lowbit_vec_dot(int) { return nullptr; }
#endif

}  // namespace strata::poly
