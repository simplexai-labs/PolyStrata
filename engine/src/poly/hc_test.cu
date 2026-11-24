// src/poly/hc_test.cu - hc_pre() and hc_head() for a batch of rows (hc_mix_rows_kernel) against the same rows one
// at a time (hc_mix_kernel): the mixes, the stream weights and the read rows, the same bits.
//
// usage: poly_hc_test   (exit 0: all within)
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kHc = 4, kEmbd = 4096, kMix = kHc * (2 + kHc), kRows = 40;

float* upload(const std::vector<float>& v) {
    float* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(float));
    cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

// the same bits, or the worst difference over its kind's largest value: a row's values are `cols` kinds
int compare(const char* what, const std::vector<float>& a, const std::vector<float>& b, size_t cols) {
    double worst = 0.0;
    for (size_t c = 0; c < cols; ++c) {
        double diff = 0.0, big = 0.0;
        for (size_t i = c; i < a.size(); i += cols) {
            diff = std::max(diff, std::abs((double) a[i] - b[i]));
            big = std::max(big, std::abs((double) b[i]));
        }
        worst = std::max(worst, diff / std::max(big, 1e-30));
    }
    const bool ok = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    std::printf("%-22s %zu values: %s (worst %.2g of its kind's largest)\n", what, a.size(), ok ? "same" : "DIFF",
                worst);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t width = size_t(kHc) * kEmbd;
    std::vector<float> r(kRows * width), fn(kMix * width), scale(3), base(kMix);
    for (float& x : r) x = nd(rng);
    for (float& x : fn) x = 0.02f * nd(rng);
    for (float& x : scale) x = 0.5f + 0.1f * nd(rng);
    for (float& x : base) x = 0.1f * nd(rng);
    float *d_r = upload(r), *d_fn = upload(fn), *d_scale = upload(scale), *d_base = upload(base);
    float *mix = nullptr, *hcw = nullptr, *cur = nullptr;
    cudaMalloc(&mix, size_t(kRows) * (kMix + 1) * sizeof(float));
    cudaMalloc(&hcw, size_t(kRows) * (kHc + kHc * kHc) * sizeof(float));
    cudaMalloc(&cur, size_t(kRows) * kEmbd * sizeof(float));
    int bad = 0;

    hc_pre(d_r, d_fn, d_scale, d_base, mix, hcw, cur, kEmbd, kRows, 1e-6f, 1e-6f, 20, nullptr);
    const auto mix_b = download(mix, size_t(kRows) * (kMix + 1));
    const auto hcw_b = download(hcw, size_t(kRows) * (kHc + kHc * kHc));
    const auto cur_b = download(cur, size_t(kRows) * kEmbd);
    for (int t = 0; t < kRows; ++t)
        hc_pre(d_r + t * width, d_fn, d_scale, d_base, mix + t * (kMix + 1), hcw + t * (kHc + kHc * kHc),
               cur + size_t(t) * kEmbd, kEmbd, 1, 1e-6f, 1e-6f, 20, nullptr);
    bad += compare("hc_pre mixes", mix_b, download(mix, mix_b.size()), kMix + 1);
    bad += compare("hc_pre stream weights", hcw_b, download(hcw, hcw_b.size()), kHc + kHc * kHc);
    bad += compare("hc_pre read rows", cur_b, download(cur, cur_b.size()), kEmbd);

    hc_head(d_r, d_fn, d_scale, d_base, mix, cur, kEmbd, kRows, 1e-6f, 1e-6f, nullptr);
    const auto head_mix = download(mix, size_t(kRows) * (kHc + 1)), head_b = download(cur, size_t(kRows) * kEmbd);
    for (int t = 0; t < kRows; ++t)
        hc_head(d_r + t * width, d_fn, d_scale, d_base, mix + t * (kHc + 1), cur + size_t(t) * kEmbd, kEmbd, 1, 1e-6f,
                1e-6f, nullptr);
    bad += compare("hc_head mixes", head_mix, download(mix, head_mix.size()), kHc + 1);
    bad += compare("hc_head rows", head_b, download(cur, head_b.size()), kEmbd);

    for (float* p : {d_r, d_fn, d_scale, d_base, mix, hcw, cur}) cudaFree(p);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
