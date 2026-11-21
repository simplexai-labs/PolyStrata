// src/poly/nvfp4_x2_gpu_test.cu - STRATA_PREFILL_NVFP4=w4a4x2's products on the card, at Inkling 2 Mini's expert
// shapes (gate/up [2304 x 2304], down [2304 x 1152]), against
//   * the host's arithmetic (nvfp4_x2_quant.hpp: the same activation bytes, and the product of ggml's dequantized
//     weights with the dequantized rows);
//   * the exact product (ggml's dequantized weights times the float rows), next to the default W4A8 MMQ's error;
// and the way through Mmq with STRATA_PREFILL_NVFP4=w4a4x2 set (the test sets it): a caller that allows the products
// (Mmq::nvfp4_x2, as Inkling 2's experts do) gets them in fits, q8_bytes, quantize, mul and split_exact, and one that
// does not gets the default W4A8 ones, bit for bit; on a card or build without them, both get W4A8.  Also the rows
// gathered by ids from rows ld floats apart, one K step, and rows with NaN, infinities and subnormals (the card's
// bytes against the host's, which flushes nothing).  With `bench` both modes' times for a prompt batch of 1,024
// tokens (128 experts, 64 rows each on average) and for 16 experts of 625 rows.
//
// usage: poly_nvfp4_x2_gpu_test [bench]   (exit 0: every check passes; 77: the gate checks pass, and this is not an
//        RTX 50 build and card)
#include "mmq.hpp"
#include "nvfp4_x2.hpp"
#include "nvfp4_x2_quant.hpp"

#include <cstdio>

#if !defined(STRATA_X2_HAS_FP4)
int main() {
    std::printf("CUDA %d has no FP4 conversions (12.8 and later): no two-term products\n", CUDART_VERSION);
    return 77;
}
#else

#include "ggml-quants.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata;
using namespace strata::poly::x2;
using poly::Mmq;

int failures = 0;
std::mt19937 rng(5);

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void check(bool ok, const std::string& what) {
    std::printf("%-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

double rel(const std::vector<double>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += (a[i] - b[i]) * (a[i] - b[i]);
        d += b[i] * b[i];
    }
    return std::sqrt(n / std::max(d, 1e-300));
}

struct Experts {
    int n, rows, cols;
    size_t row, stride;
    std::vector<uint8_t> w;
    std::vector<float> wf;   // ggml's dequantized weights
};

// `keep`: with the dequantized weights (the checks; the benchmark's 128 experts would take 2.7 GB of them)
Experts make_experts(int n, int rows, int cols, bool keep = true) {
    Experts x{n, rows, cols, ggml_row_size(GGML_TYPE_NVFP4, cols), 0, {}, {}};
    x.stride = x.row * rows;
    x.w.resize(x.stride * n);
    if (keep) x.wf.resize((size_t) n * rows * cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> f(cols);
    for (int64_t r = 0; r < (int64_t) n * rows; ++r) {
        for (float& v : f) v = nd(rng) * 0.02f;
        block_nvfp4* b = (block_nvfp4*) (x.w.data() + r * x.row);
        quantize_row_nvfp4_ref(f.data(), b, cols);
        if (keep) dequantize_row_nvfp4(b, x.wf.data() + r * cols, cols);
    }
    return x;
}

std::vector<float> make_rows(int rows, int cols) {
    std::vector<float> x((size_t) rows * cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) {
        const float a = std::max(nd(rng), 0.f);
        v = a * a * nd(rng) + 0.1f * nd(rng);
    }
    return x;
}

// the host's activation rows, as nvfp4_x2_test.cpp writes them: row r is row ids[r] of x (row r without ids), ld
// floats a row
std::vector<uint8_t> host_rows(const std::vector<float>& x, int rows, int cols, int64_t ld = 0,
                               const std::vector<int32_t>* ids = nullptr) {
    if (ld == 0) ld = cols;
    std::vector<uint8_t> xq(rows_bytes(rows, cols), 0);
    float* gs = (float*) (xq.data() + (size_t) rows * row_bytes(cols));
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + (size_t) (ids ? (*ids)[r] : r) * ld;
        gs[r] = host_row_scale(xr, cols);
        for (int s = 0; s < cols / 16; ++s) quantize16(xr + 16 * s, gs[r], xq.data() + (size_t) r * row_bytes(cols), s);
    }
    return xq;
}

std::vector<float> dequant_rows(const std::vector<uint8_t>& xq, int rows, int cols) {
    std::vector<float> out((size_t) rows * cols);
    const int nb = cols / kBlock;
    std::vector<block_nvfp4> t(nb);
    std::vector<float> f(cols);
    for (int r = 0; r < rows; ++r) {
        const float g = row_scales(xq.data(), rows, cols)[r];
        for (int k = 0; k < 2; ++k) {
            for (int b = 0; b < nb; ++b)
                std::memcpy(&t[b], xq.data() + (size_t) r * row_bytes(cols) + (size_t) b * kPairBytes + k * kBlockBytes,
                            kBlockBytes);
            dequantize_row_nvfp4(t.data(), f.data(), cols);
            for (int i = 0; i < cols; ++i) out[(size_t) r * cols + i] += g * f[i];
        }
    }
    return out;
}

// one launch over the experts of `counts`, both ways, against the host
void check_product(const Experts& E, const std::vector<int>& counts, poly::Mmq& mmq, cudaStream_t s) {
    const int n = E.n, cols = E.cols, wr = E.rows;
    std::vector<int32_t> bounds(n + 1, 0);
    for (int e = 0; e < n; ++e) bounds[e + 1] = bounds[e] + counts[e];
    const int total = bounds[n];
    const int most = *std::max_element(counts.begin(), counts.end());
    const std::vector<float> x = make_rows(total, cols);
    std::vector<int32_t> ids(total);
    for (int i = 0; i < total; ++i) ids[i] = (int32_t) ((i * 5 + 1) % total);   // rows to scattered places

    float* d_x = upload(x);
    int32_t* d_b = upload(bounds);
    int32_t* d_ids = upload(ids);
    void* d_xq = nullptr;
    ck(cudaMalloc(&d_xq, Mmq::q8_bytes(total, cols, Mmq::nvfp4_x2)), "malloc");
    float *d_y = nullptr, *d_y8 = nullptr;
    ck(cudaMalloc(&d_y, (size_t) total * wr * sizeof(float)), "malloc");
    ck(cudaMalloc(&d_y8, (size_t) total * wr * sizeof(float)), "malloc");
    uint8_t* d_w = upload(E.w);

    poly::nvfp4_x2_quantize(d_x, nullptr, d_xq, cols, cols, total, s);
    std::vector<uint8_t> xq(rows_bytes(total, cols));
    ck(cudaMemcpyAsync(xq.data(), d_xq, xq.size(), cudaMemcpyDeviceToHost, s), "download");
    poly::nvfp4_x2_mul(d_w, wr, cols, E.stride, n, d_xq, d_b, d_ids, total, most, d_y, wr, s);
    std::vector<float> y((size_t) total * wr), y8(y.size()), ym(y.size());
    ck(cudaMemcpyAsync(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "download");
    // through Mmq, allowed: the same launches
    Mmq::quantize(d_x, nullptr, d_xq, GGML_TYPE_NVFP4, cols, cols, total, s, Mmq::nvfp4_x2);
    mmq.mul(d_w, GGML_TYPE_NVFP4, wr, cols, E.stride, n, d_xq, d_b, d_ids, total, most, d_y8, wr, s, Mmq::nvfp4_x2);
    ck(cudaMemcpyAsync(ym.data(), d_y8, ym.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "download");
    // the default path: not allowed, with the variable set
    Mmq::quantize(d_x, nullptr, d_xq, GGML_TYPE_NVFP4, cols, cols, total, s);
    mmq.mul(d_w, GGML_TYPE_NVFP4, wr, cols, E.stride, n, d_xq, d_b, d_ids, total, most, d_y8, wr, s);
    ck(cudaMemcpyAsync(y8.data(), d_y8, y8.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "download");
    ck(cudaStreamSynchronize(s), "sync");

    const std::vector<uint8_t> hq = host_rows(x, total, cols);
    size_t diff = 0;
    for (size_t i = 0; i < hq.size(); ++i) diff += hq[i] != xq[i];
    char what[200];
    std::snprintf(what, sizeof what, "[%d x %d] x %d rows: the card's activation bytes are the host's (%zu differ)", wr,
                  cols, total, diff);
    check(diff == 0, what);
    check(std::memcmp(y.data(), ym.data(), y.size() * sizeof(float)) == 0, "  Mmq, allowed: the same products");
    check(std::memcmp(y.data(), y8.data(), y.size() * sizeof(float)) != 0, "  Mmq without it: other products (W4A8)");

    const std::vector<float> xd = dequant_rows(hq, total, cols);
    std::vector<double> got, got8, same, exact;
    for (int e = 0; e < n; ++e)
        for (int i = bounds[e]; i < bounds[e + 1]; ++i)
            for (int r = 0; r < wr; ++r) {
                const float* w = E.wf.data() + ((size_t) e * wr + r) * cols;
                double a = 0, b = 0;
                for (int k = 0; k < cols; ++k) {
                    a += (double) w[k] * xd[(size_t) i * cols + k];
                    b += (double) w[k] * x[(size_t) i * cols + k];
                }
                same.push_back(a);
                exact.push_back(b);
                got.push_back(y[(size_t) ids[i] * wr + r]);
                got8.push_back(y8[(size_t) ids[i] * wr + r]);
            }
    const double e_same = rel(got, same), e_x2 = rel(got, exact), e_w4a8 = rel(got8, exact);
    std::snprintf(what, sizeof what, "  products: vs the same arithmetic %.1e; vs floats %.3f%% (W4A8 MMQ %.3f%%)", e_same,
                  100 * e_x2, 100 * e_w4a8);
    check(e_same < 1e-5 && e_x2 < 1.5 * e_w4a8 + 0.002, what);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_ids); cudaFree(d_xq); cudaFree(d_y); cudaFree(d_y8); cudaFree(d_w);
}

// Mmq with the variable set: the caller's Mmq::Allow picks the products; without the card's kernels both are W4A8.
// Rows of 2304 and 1152 values (whole steps) and 192 (not: W4A8 whatever is allowed).
void check_gate(bool built, Mmq& mmq, cudaStream_t s) {
    const int rows = 70;
    bool ok = true;
    for (const int cols : {2304, 1152, 192}) {
        const bool x2 = built && cols % kK == 0;
        const size_t q8 = (size_t) rows * ((cols + 511) / 512 * 4) * 144 + 128 * 144;   // Q8_1's: 144-byte blocks
        ok = ok && Mmq::fits(GGML_TYPE_NVFP4, 2304, cols, Mmq::nvfp4_x2) ==
                       (x2 || Mmq::fits(GGML_TYPE_NVFP4, 2304, cols));
        ok = ok && Mmq::q8_bytes(rows, cols) == q8 &&
             Mmq::q8_bytes(rows, cols, Mmq::nvfp4_x2) == (x2 ? std::max(q8, rows_bytes(rows, cols)) : q8);
        ok = ok && Mmq::split_exact(GGML_TYPE_NVFP4, 2304) == Mmq::split_exact(GGML_TYPE_NVFP4, 2304, Mmq::q2_0) &&
             (!x2 || Mmq::split_exact(GGML_TYPE_NVFP4, 2304, Mmq::nvfp4_x2));
    }
    check(ok, std::string("fits, q8_bytes, split_exact: the two-term products only when allowed") +
                  (built ? "" : " (none on this card: never)"));

    // quantize and mul: allowed and not, the bytes and the products
    const int cols = 2304, wr = 256;
    const Experts E = make_experts(2, wr, cols, false);
    const std::vector<int32_t> bounds = {0, 40, rows};
    std::vector<int32_t> ids(rows);
    for (int i = 0; i < rows; ++i) ids[i] = i;
    float* d_x = upload(make_rows(rows, cols));
    int32_t* d_b = upload(bounds);
    int32_t* d_ids = upload(ids);
    uint8_t* d_w = upload(E.w);
    const size_t qb = Mmq::q8_bytes(rows, cols, Mmq::nvfp4_x2);
    void *d_q0 = nullptr, *d_q1 = nullptr;
    float *d_y0 = nullptr, *d_y1 = nullptr;
    ck(cudaMalloc(&d_q0, qb), "malloc");
    ck(cudaMalloc(&d_q1, qb), "malloc");
    ck(cudaMemsetAsync(d_q0, 0, qb, s), "memset");
    ck(cudaMemsetAsync(d_q1, 0, qb, s), "memset");
    ck(cudaMalloc(&d_y0, (size_t) rows * wr * sizeof(float)), "malloc");
    ck(cudaMalloc(&d_y1, (size_t) rows * wr * sizeof(float)), "malloc");
    Mmq::quantize(d_x, nullptr, d_q0, GGML_TYPE_NVFP4, cols, cols, rows, s);
    Mmq::quantize(d_x, nullptr, d_q1, GGML_TYPE_NVFP4, cols, cols, rows, s, Mmq::nvfp4_x2);
    mmq.mul(d_w, GGML_TYPE_NVFP4, wr, cols, E.stride, 2, d_q0, d_b, d_ids, rows, 40, d_y0, wr, s);
    mmq.mul(d_w, GGML_TYPE_NVFP4, wr, cols, E.stride, 2, d_q1, d_b, d_ids, rows, 40, d_y1, wr, s, Mmq::nvfp4_x2);
    std::vector<uint8_t> q0(qb), q1(qb);
    std::vector<float> y0((size_t) rows * wr), y1(y0.size());
    ck(cudaMemcpyAsync(q0.data(), d_q0, qb, cudaMemcpyDeviceToHost, s), "download");
    ck(cudaMemcpyAsync(q1.data(), d_q1, qb, cudaMemcpyDeviceToHost, s), "download");
    ck(cudaMemcpyAsync(y0.data(), d_y0, y0.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "download");
    ck(cudaMemcpyAsync(y1.data(), d_y1, y1.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "download");
    ck(cudaStreamSynchronize(s), "sync");
    const bool same_q = q0 == q1, same_y = std::memcmp(y0.data(), y1.data(), y0.size() * sizeof(float)) == 0;
    if (built) {
        const std::vector<float> x = [&] {
            std::vector<float> h((size_t) rows * cols);
            ck(cudaMemcpy(h.data(), d_x, h.size() * sizeof(float), cudaMemcpyDeviceToHost), "download");
            return h;
        }();
        const std::vector<uint8_t> hq = host_rows(x, rows, cols);
        check(std::equal(hq.begin(), hq.end(), q1.begin()) && !same_q && !same_y,
              "quantize and mul: two-term rows when allowed, W4A8 ones when not");
    } else {
        check(same_q && same_y, "quantize and mul: W4A8 bytes and products whether allowed or not");
    }
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_ids); cudaFree(d_w); cudaFree(d_q0); cudaFree(d_q1); cudaFree(d_y0);
    cudaFree(d_y1);
}

// The card's quantize against the host's: rows gathered by ids (repeats, any order) from rows ld floats apart, with
// NaN, infinities, rows under kTinyRow, subnormal values and blocks from 2^-150 to 2^10; and their products: NaN for
// a row with a NaN or an infinity, 0 for one of zeros.
void check_rows(Mmq& mmq, cudaStream_t s) {
    const int cols = 256, src = 24;
    const int64_t ld = cols + 12;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-150, 10);
    std::vector<float> x((size_t) src * ld, 12345.f);   // the gaps between rows are never read
    for (int r = 0; r < src; ++r)
        for (int i = 0; i < cols; ++i) {
            float v = nd(rng);
            switch (r) {
                case 0: v = i == 7 ? NAN : v; break;
                case 1: v = i == cols - 1 ? INFINITY : v; break;
                case 2: v = i == 0 ? -INFINITY : v; break;
                case 3: v *= 1e-40f; break;                       // subnormal values only
                case 4: v *= 1e-33f; break;                       // a row scale that would be subnormal
                case 5: v = i == 1 ? kTinyRow : v * 0.05f * kTinyRow; break;
                case 6: v = i % 3 == 0 ? (i % 2 ? -1e-39f : 1e-39f) : v; break;
                case 7: v = 0.f; break;
                default: break;
            }
            x[(size_t) r * ld + i] = v;
        }
    // rows 16..23 stay plain; 8..15 take a scale a 16-value block
    for (int r = 8; r < 16; ++r)
        for (int b = 0; b < cols / 16; ++b) {
            const float f = std::ldexp(1.f, ex(rng));
            for (int j = 0; j < 16; ++j) x[(size_t) r * ld + 16 * b + j] = nd(rng) * f;
        }
    const int rows = 40;
    std::vector<int32_t> ids(rows);
    for (int i = 0; i < rows; ++i) ids[i] = (int32_t) ((i * 7 + 5) % src);
    float* d_x = upload(x);
    int32_t* d_ids = upload(ids);
    void* d_xq = nullptr;
    ck(cudaMalloc(&d_xq, rows_bytes(rows, cols)), "malloc");
    Mmq::quantize(d_x, d_ids, d_xq, GGML_TYPE_NVFP4, cols, ld, rows, s, Mmq::nvfp4_x2);
    std::vector<uint8_t> xq(rows_bytes(rows, cols));
    ck(cudaMemcpyAsync(xq.data(), d_xq, xq.size(), cudaMemcpyDeviceToHost, s), "download");
    // one matrix, every row: NaN rows and zero rows come out as such
    const Experts E = make_experts(1, 128, cols, false);
    uint8_t* d_w = upload(E.w);
    const std::vector<int32_t> bounds = {0, rows}, iota = [&] {
        std::vector<int32_t> v(rows);
        for (int i = 0; i < rows; ++i) v[i] = i;
        return v;
    }();
    int32_t* d_b = upload(bounds);
    int32_t* d_io = upload(iota);
    float* d_y = nullptr;
    ck(cudaMalloc(&d_y, (size_t) rows * 128 * sizeof(float)), "malloc");
    mmq.mul(d_w, GGML_TYPE_NVFP4, 128, cols, E.stride, 1, d_xq, d_b, d_io, rows, rows, d_y, 128, s, Mmq::nvfp4_x2);
    std::vector<float> y((size_t) rows * 128);
    ck(cudaMemcpyAsync(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "download");
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<uint8_t> hq = host_rows(x, rows, cols, ld, &ids);
    size_t diff = 0;
    for (size_t i = 0; i < hq.size(); ++i) diff += hq[i] != xq[i];
    char what[200];
    std::snprintf(what, sizeof what,
                  "rows by ids from ld %lld: NaN, inf, tiny, subnormal: the card's bytes are the host's (%zu differ)",
                  (long long) ld, diff);
    check(diff == 0, what);
    bool out_ok = true;
    for (int i = 0; i < rows; ++i) {
        const int r = ids[i];
        for (int c = 0; c < 128; ++c) {
            const float v = y[(size_t) i * 128 + c];
            if (r <= 2) out_ok = out_ok && std::isnan(v);
            else if (r == 3 || r == 4 || r == 7) out_ok = out_ok && v == 0.f;
            else out_ok = out_ok && std::isfinite(v);
        }
    }
    check(out_ok, "  their products: NaN for a non-finite row, 0 for a row of zeros or under kTinyRow, else finite");
    cudaFree(d_x); cudaFree(d_ids); cudaFree(d_xq); cudaFree(d_w); cudaFree(d_b); cudaFree(d_io); cudaFree(d_y);
}

// ms a call of quantize + product, both ways, n experts of counts[e] rows
void bench(const char* name, const Experts& E, const std::vector<int>& counts, poly::Mmq& mmq, cudaStream_t s) {
    const int n = E.n, cols = E.cols, wr = E.rows;
    std::vector<int32_t> bounds(n + 1, 0);
    for (int e = 0; e < n; ++e) bounds[e + 1] = bounds[e] + counts[e];
    const int total = bounds[n], most = *std::max_element(counts.begin(), counts.end());
    std::vector<int32_t> ids(total);
    for (int i = 0; i < total; ++i) ids[i] = i;
    float* d_x = upload(make_rows(total, cols));
    int32_t* d_b = upload(bounds);
    int32_t* d_ids = upload(ids);
    uint8_t* d_w = upload(E.w);
    void* d_xq = nullptr;
    float* d_y = nullptr;
    ck(cudaMalloc(&d_xq, Mmq::q8_bytes(total, cols, Mmq::nvfp4_x2)), "malloc");
    ck(cudaMalloc(&d_y, (size_t) total * wr * sizeof(float)), "malloc");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int way = 0; way < 2; ++way) {
        const auto call = [&] {
            if (way == 0) {
                Mmq::quantize(d_x, nullptr, d_xq, GGML_TYPE_NVFP4, cols, cols, total, s);
                mmq.mul(d_w, GGML_TYPE_NVFP4, wr, cols, E.stride, n, d_xq, d_b, d_ids, total, most, d_y, wr, s);
            } else {
                poly::nvfp4_x2_quantize(d_x, nullptr, d_xq, cols, cols, total, s);
                poly::nvfp4_x2_mul(d_w, wr, cols, E.stride, n, d_xq, d_b, d_ids, total, most, d_y, wr, s);
            }
        };
        for (int i = 0; i < 3; ++i) call();
        constexpr int kRuns = 20;
        cudaEventRecord(a, s);
        for (int i = 0; i < kRuns; ++i) call();
        cudaEventRecord(b, s);
        ck(cudaEventSynchronize(b), "bench");
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double flop = 2.0 * total * wr * cols;
        std::printf("bench %-34s %-7s %8.3f ms  %7.1f TFLOP/s\n", name, way ? "w4a4x2" : "default", ms / kRuns,
                    flop / (ms / kRuns * 1e-3) / 1e12);
    }
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_ids); cudaFree(d_w); cudaFree(d_xq); cudaFree(d_y);
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    _putenv_s("STRATA_PREFILL_NVFP4", "w4a4x2");   // before Mmq reads it
#else
    setenv("STRATA_PREFILL_NVFP4", "w4a4x2", 1);
#endif
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("no card\n");
        return 77;
    }
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    Mmq mmq;
    const bool built = poly::nvfp4_x2_built();
    check(cudaPeekAtLastError() == cudaSuccess, "the probe leaves no error behind");
    check_gate(built, mmq, s);
    if (!built) {
        std::printf("the two-term FP4 products need a compute capability 12.x card and a build for it (12xa)\n");
        std::printf("%s\n", failures ? "FAILED" : "gate checks ok");
        return failures ? 1 : 77;
    }
    check_rows(mmq, s);
    const Experts gu = make_experts(6, 2304, 2304), down = make_experts(6, 2304, 1152);
    check_product(gu, {150, 0, 1, 64, 65, 7}, mmq, s);
    check_product(down, {33, 128, 0, 2, 200, 64}, mmq, s);
    const Experts tail = make_experts(2, 200, 384);   // rows ending inside a tile
    check_product(tail, {70, 3}, mmq, s);
    const Experts one = make_experts(3, 256, 128);   // one K step
    check_product(one, {5, 64, 65}, mmq, s);
    if (argc > 1 && std::string(argv[1]) == "bench") {
        std::vector<int> batch(128);
        std::exponential_distribution<double> ex(1.0 / 64);   // a prompt batch's routing: uneven
        for (int& c : batch) c = std::min(400, (int) ex(rng));
        const Experts gu128 = make_experts(128, 2304, 2304, false), down128 = make_experts(128, 2304, 1152, false);
        bench("gate/up, 128 experts, ~64 rows", gu128, batch, mmq, s);
        bench("down, 128 experts, ~64 rows", down128, batch, mmq, s);
        const Experts gu16 = make_experts(16, 2304, 2304, false), down16 = make_experts(16, 2304, 1152, false);
        bench("gate/up, 16 experts x 625 rows", gu16, std::vector<int>(16, 625), mmq, s);
        bench("down, 16 experts x 625 rows", down16, std::vector<int>(16, 625), mmq, s);
    }
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}

#endif  // STRATA_X2_HAS_FP4
