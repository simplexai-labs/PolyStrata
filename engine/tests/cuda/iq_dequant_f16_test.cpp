// iq_dequant_f16 (dq_dispatch through dequant_flat_kernel) against ggml's own to_float, for every type
// iq_dequant_supported() takes that ggml can quantize: what strata-poly's prompt products read on a card without the
// int8 tiles (the BLAS path: Turing, HIP) and what gemm_iq_f16_grouped decodes on Volta and Turing.  The rows are
// ggml_quantize_chunk()'s blocks of normal values with an uneven importance, a few rows of large values among them.
// A value may differ from ggml's by its fp16 rounding only.  Exit 77 without a card.
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

float half_to_float(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {   // subnormal: normalise
            int k = -1;
            uint32_t mm = m;
            do { ++k; mm <<= 1; } while (!(mm & 0x400));
            bits = s | (uint32_t) (127 - 15 - k) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | m << 13;
    } else {
        bits = s | (e + 127 - 15) << 23 | m << 13;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

}  // namespace

int main() {
    int cards = 0;
    if (cudaGetDeviceCount(&cards) != cudaSuccess || cards == 0) {
        std::puts("iq_dequant_f16_test: no card, skipped");
        return 77;
    }
    const int types[] = {1, 30, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 42};
    constexpr int64_t kCols = 1024, kRows = 16;
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> w(kCols * kRows), imp(kCols);
    for (int64_t r = 0; r < kRows; ++r)
        for (int64_t c = 0; c < kCols; ++c) w[r * kCols + c] = normal(rng) * (r % 5 == 4 ? 40.f : 0.02f);
    for (auto& v : imp) v = 0.1f + std::fabs(normal(rng));
    int bad = 0;
    for (int t : types) {
        const auto ty = (ggml_type) t;
        if (!strata::kernels::iq_dequant_supported(t)) {
            std::printf("%-8s not taken by iq_dequant_f16\n", ggml_type_name(ty));
            ++bad;
            continue;
        }
        ggml_quantize_init(ty);
        const size_t bytes = ggml_row_size(ty, kCols) * kRows;
        std::vector<uint8_t> q(bytes);
        ggml_quantize_chunk(ty, w.data(), q.data(), 0, kRows, kCols, ggml_quantize_requires_imatrix(ty) || t >= 16 ? imp.data() : nullptr);
        std::vector<float> want(w.size());
        ggml_get_type_traits(ty)->to_float(q.data(), want.data(), (int64_t) want.size());
        void* dq = nullptr;
        void* dy = nullptr;
        std::vector<uint16_t> got(w.size());
        if (!ok(cudaMalloc(&dq, bytes), "cudaMalloc") || !ok(cudaMalloc(&dy, got.size() * 2), "cudaMalloc") ||
            !ok(cudaMemcpy(dq, q.data(), bytes, cudaMemcpyHostToDevice), "copy in"))
            return 1;
        strata::kernels::iq_dequant_f16(t, dq, (int64_t) w.size(), (uint16_t*) dy, nullptr);
        if (!ok(cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost), "copy out")) return 1;
        cudaFree(dq);
        cudaFree(dy);
        // each value against ggml's, allowed fp16's rounding of it (and of a subnormal: 2^-24)
        int64_t wrong = 0, first = -1;
        double worst = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            const float g = half_to_float(got[i]), e = want[i];
            const double room = std::fabs(e) * 1.0 / 1024 + 6e-8;
            const double d = std::fabs((double) g - e);
            if (!(d <= room)) {
                if (first < 0) first = (int64_t) i;
                ++wrong;
            }
            if (d > worst) worst = d;
        }
        std::printf("%-8s %s  worst |diff| %.3g", ggml_type_name(ty), wrong ? "WRONG" : "ok   ", worst);
        if (wrong)
            std::printf("  %lld of %zu values, first at %lld (row %lld, value %lld): %g, ggml %g", (long long) wrong, want.size(),
                        (long long) first, (long long) (first / kCols), (long long) (first % kCols),
                        half_to_float(got[first]), want[first]);
        std::printf("\n");
        bad += wrong != 0;
    }
    ggml_quantize_free();
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
