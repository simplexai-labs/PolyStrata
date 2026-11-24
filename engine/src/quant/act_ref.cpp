// src/quant/act_ref.cpp - the reference activation quantizers (Generic contract) and the Q4_0, Q5_0 and IQ4_NL
// requantizers.
//
// Derived from ggml's `quantize_row_q8_0_ref`, `quantize_row_q8_1_ref`, `quantize_row_q8_K_ref`, `quantize_row_q4_0_ref`,
// `quantize_row_q5_0_ref` and `quantize_row_iq4_nl_ref` (llama.cpp
// ggml/src/ggml-quants.c at 3cf03257f219afbe7334045ff7c6a06ac68c627d, Copyright (c) 2023-2026 The ggml authors, MIT).  Q8_0 / Q8_1 round half
// away from zero (roundf) here; ggml's AVX / AVX2 quantizers round half to even (x86/act_avx*.cpp).  Q8_K is ggml's scalar code on every x86
// flavour, and its scale is -127/max, not -128/max (the comment in the loop says why): do not "fix" that.
#define STRATA_QUANT_TABLES
#include "strata/quant/act.hpp"

#include <cassert>

#include "internal.hpp"
#include "registry.hpp"

#define GGML_UNUSED(x) (void) (x)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace strata::quant {
namespace {

void q8_0_ref(const float* STRATA_RESTRICT x, block_q8_0* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void q8_1_ref(const float* STRATA_RESTRICT x, block_q8_1* STRATA_RESTRICT y, int64_t k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_1; j++) {
            const float v = x[i*QK8_1 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        int sum = 0;

        for (int j = 0; j < QK8_1/2; ++j) {
            const float v0 = x[i*QK8_1           + j]*id;
            const float v1 = x[i*QK8_1 + QK8_1/2 + j]*id;

            y[i].qs[          j] = roundf(v0);
            y[i].qs[QK8_1/2 + j] = roundf(v1);

            sum += y[i].qs[          j];
            sum += y[i].qs[QK8_1/2 + j];
        }

        y[i].s = fp32_to_fp16(sum*d);
    }
}

void q8_K_ref(const float* STRATA_RESTRICT x, block_q8_K* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void q4_0_ref(const float* STRATA_RESTRICT x, block_q4_0* STRATA_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void q5_0_ref(const float* STRATA_RESTRICT x, block_q5_0* STRATA_RESTRICT y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j]  = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

int best_index_int8(int n, const int8_t* val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// quantize_row_iq4_nl_impl() for one block of 32 with no importance weights and ntry 7
void iq4_nl_ref(const float* STRATA_RESTRICT x, block_iq4_nl* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    const int8_t* values = kvalues_iq4nl;
    const int ntry = 7;
    for (int64_t ib = 0; ib < k / QK4_NL; ++ib) {
        const float* xb = x + ib*QK4_NL;
        float weight[QK4_NL];
        uint8_t L[QK4_NL];
        for (int j = 0; j < QK4_NL; ++j) weight[j] = xb[j]*xb[j];
        float amax = 0, max = 0;
        for (int j = 0; j < QK4_NL; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        float d = 0;
        if (amax >= 1e-15f) {
            d = -max/values[0];
            float id = 1/d;
            float sumqx = 0, sumq2 = 0;
            for (int j = 0; j < QK4_NL; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                L[j] = l;
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            d = sumqx/sumq2;
            float best = d*sumqx;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx = sumq2 = 0;
                for (int j = 0; j < QK4_NL; ++j) {
                    float al = id*xb[j];
                    int l = best_index_int8(16, values, al);
                    float q = values[l];
                    float w = weight[j];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    d = sumqx/sumq2; best = d * sumqx;
                }
            }
        }
        y[ib].d = fp32_to_fp16(d);
        const float id = d ? 1/d : 0;
        for (int j = 0; j < QK4_NL; ++j) L[j] = best_index_int8(16, values, id*xb[j]);
        for (int j = 0; j < 16; ++j) y[ib].qs[j] = L[j] | (L[16 + j] << 4);
    }
}

}  // namespace

void quantize_q8_0_ref(const float* x, void* y, int64_t n) { q8_0_ref(x, (block_q8_0*) y, n); }
void quantize_q8_1_ref(const float* x, void* y, int64_t n) { q8_1_ref(x, (block_q8_1*) y, n); }
void quantize_q8_K_ref(const float* x, void* y, int64_t n) { q8_K_ref(x, (block_q8_K*) y, n); }
void quantize_q4_0_ref(const float* x, void* y, int64_t n) { q4_0_ref(x, (block_q4_0*) y, n); }
void quantize_q5_0_ref(const float* x, void* y, int64_t n) { q5_0_ref(x, (block_q5_0*) y, n); }
void quantize_iq4_nl_ref(const float* x, void* y, int64_t n) { iq4_nl_ref(x, (block_iq4_nl*) y, n); }

namespace {
const ActKernel kGenericAct[] = {
    {quantize_q8_0_ref, Type::Q8_0, Arith::Generic, Isa::Scalar, "generic_q8_0"},
    {quantize_q8_1_ref, Type::Q8_1, Arith::Generic, Isa::Scalar, "generic_q8_1"},
    {quantize_q8_K_ref, Type::Q8_K, Arith::Generic, Isa::Scalar, "generic_q8_K"},
};
}  // namespace

ActTable generic_act_table() noexcept { return {kGenericAct, 3}; }

}  // namespace strata::quant
