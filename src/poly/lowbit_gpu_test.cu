// src/poly/lowbit_gpu_test.cu - the card's products of Inkling 2's low-bit expert types:
//   * IQ1_S and Q2_0 in the decode products (native_mmvq, native_mmvq_slots) against ggml's dequantized rows times
//     the Q8_1 activation on the host;
//   * Q2_0 in the prompt products (poly::Mmq, llama.cpp's MMQ tiles) against the dequantized rows times the floats,
//     and against the same matrix written exactly as Q4_0 through Q4_0's tiles;
// and with `bench` the products' times.
//
// usage: poly_lowbit_gpu_test [bench]   (exit 0: every check passes; 77: no card)
#include "mmq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata;

int failures = 0;
std::mt19937 rng(11);
cudaStream_t ts;   // the checks' stream (native_mmvq takes no null stream)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void check(bool ok, const std::string& what) {
    std::printf("%-78s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

std::vector<uint8_t> random_rows(ggml_type t, int n, int rows) {
    const size_t rb = ggml_row_size(t, n);
    std::vector<uint8_t> w(rb * rows);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t bb = ggml_type_size(t);
    std::uniform_real_distribution<float> sc(0.002f, 0.05f);
    for (size_t off = 0; off + bb <= w.size(); off += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(&w[off], &d, 2);
    }
    return w;
}

// the card's copy, with zero bytes behind it (the MMQ tiles read past a matrix's last row)
void* upload(const void* p, size_t n) {
    void* d = nullptr;
    ck(cudaMalloc(&d, n + (1 << 20)), "malloc");
    ck(cudaMemset(d, 0, n + (1 << 20)), "memset");
    ck(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice), "upload");
    return d;
}

std::vector<float> dequant(ggml_type t, const std::vector<uint8_t>& w, int n, int rows) {
    std::vector<float> f((size_t) n * rows);
    ggml_get_type_traits(t)->to_float(w.data(), f.data(), (int64_t) n * rows);
    return f;
}

// Q2_0 as Q4_0, exactly: a block of 64 is (q - 1) * d with q in 0..3, two Q4_0 blocks of (q4 - 8) * d, q4 = q + 7
void q2_0_to_q4_0(const uint8_t* src, uint8_t* dst, int64_t n) {
    for (int64_t b = 0; b < n / 64; ++b, src += 18)
        for (int h = 0; h < 2; ++h, dst += 18) {
            std::memcpy(dst, src, 2);
            uint8_t* qs = dst + 2;
            for (int j = 0; j < 16; ++j) {
                const auto q = [&](int e) { return (src[2 + (32 * h + e) / 4] >> (2 * ((32 * h + e) % 4))) & 3; };
                qs[j] = (uint8_t) ((q(j) + 7) | ((q(j + 16) + 7) << 4));
            }
        }
}

// the Q8_1 activation as the decode products read it: 32 values a block, half2 (d, s) then 32 codes
std::vector<float> q8_1_values(const void* dq, int n) {
    std::vector<uint8_t> b((size_t) n / 32 * 36);
    ck(cudaMemcpy(b.data(), dq, b.size(), cudaMemcpyDeviceToHost), "download");
    std::vector<float> v(n);
    for (int k = 0; k < n / 32; ++k) {
        __half d;
        std::memcpy(&d, &b[(size_t) k * 36], 2);
        for (int i = 0; i < 32; ++i) v[k * 32 + i] = __half2float(d) * (float) (int8_t) b[(size_t) k * 36 + 4 + i];
    }
    return v;
}

// max over the rows of |got - ref| / sum |terms|
double row_err(const std::vector<float>& got, const std::vector<float>& wf, const std::vector<float>& xv, int n,
               int rows, size_t got_stride = 1, size_t got_off = 0) {
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
        double d = 0, mag = 0;
        for (int i = 0; i < n; ++i) {
            const double t = double(wf[(size_t) r * n + i]) * xv[i];
            d += t, mag += std::fabs(t);
        }
        worst = std::max(worst, std::fabs(got[got_off + (size_t) r * got_stride] - d) / std::max(mag, 1e-30));
    }
    return worst;
}

// The IQ1_S row dot as ggml-cuda's vec_dot_iq1_s_q8_1 defines it: a group of 32 is dl * (grid + delta), taken as
// dl * ((g + 1) . q * d + (delta - 1) * s), s the activation block's sum as Q8_1 keeps it (a half of the sum of the
// values before they were quantized).  In double otherwise.  qx: the Q8_1 bytes of the activation (36 a block of 32).
std::vector<double> iq1_s_ref(const std::vector<uint8_t>& w, const std::vector<uint8_t>& qx, int n_in, int n_out) {
    std::vector<float> wf = dequant(GGML_TYPE_IQ1_S, w, n_in, n_out);
    std::vector<double> y(n_out);
    const size_t rb = ggml_row_size(GGML_TYPE_IQ1_S, n_in);
    for (int r = 0; r < n_out; ++r) {
        double acc = 0;
        for (int b = 0; b < n_in / 256; ++b) {
            const uint8_t* blk = w.data() + r * rb + b * 50;
            ggml_fp16_t dh;
            std::memcpy(&dh, blk, 2);
            const double d = ggml_fp16_to_fp32(dh);
            for (int g = 0; g < 8; ++g) {
                uint16_t qh;
                std::memcpy(&qh, blk + 34 + 2 * g, 2);
                const double dl = d * (2 * ((qh >> 12) & 7) + 1), delta = qh & 0x8000 ? -0.125 : 0.125;
                const uint8_t* xb = qx.data() + (size_t) (b * 8 + g) * 36;
                __half hd, hs;
                std::memcpy(&hd, xb, 2), std::memcpy(&hs, xb + 2, 2);
                const double xd = __half2float(hd), xs = __half2float(hs);
                // the kernel's form: the grid as g + 1 (0..2) against the codes, and delta - 1 against the block's
                // sum s, which Q8_1 keeps as the sum of the unquantized values (llama.cpp's quantizer too): the two
                // agree with dl * (g + delta) . x only as far as d * sum q is s
                for (int i = 0; i < 32; ++i) {
                    const double g1 = (wf[(size_t) r * n_in + b * 256 + g * 32 + i] - dl * delta) + dl;   // dl (g + 1)
                    acc += g1 * xd * (int8_t) xb[4 + i];
                }
                acc += dl * (delta - 1.0) * xs;
            }
        }
        y[r] = acc;
    }
    return y;
}

std::vector<uint8_t> q8_1_bytes(const void* dq, int n) {
    std::vector<uint8_t> b((size_t) n / 32 * 36);
    ck(cudaMemcpy(b.data(), dq, b.size(), cudaMemcpyDeviceToHost), "download");
    return b;
}

// max over the rows of |got - ref| / sum |w * x|
double ref_err(const std::vector<float>& got, size_t off, const std::vector<double>& ref, const std::vector<float>& wf,
               const std::vector<float>& xv, int n, int rows) {
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
        double mag = 0;
        for (int i = 0; i < n; ++i) mag += std::fabs(double(wf[(size_t) r * n + i]) * xv[i]);
        worst = std::max(worst, std::fabs(got[off + r] - ref[r]) / std::max(mag, 1e-30));
    }
    return worst;
}

void test_iq1_s_decode() {
    const int n_in = 2304, n_out = 2304, count = 8;   // Inkling's gate|up rows, a token's 8 experts
    std::vector<std::vector<uint8_t>> w;
    std::vector<void*> dw;
    std::vector<std::vector<float>> xs;
    void* dx = nullptr;
    float* dy = nullptr;
    ck(cudaMalloc(&dx, (size_t) count * n_in / 32 * 36), "malloc");
    ck(cudaMalloc(&dy, (size_t) count * n_out * 4), "malloc");
    float* dxf = nullptr;
    ck(cudaMalloc(&dxf, (size_t) count * n_in * 4), "malloc");
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf((size_t) count * n_in);
    for (float& v : xf) v = nd(rng);
    ck(cudaMemcpy(dxf, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice), "upload");
    kernels::quantize_q8_1_rows(dxf, count, n_in, dx, ts);
    kernels::MmvqSlots slots{};
    for (int e = 0; e < count; ++e) {
        w.push_back(random_rows(GGML_TYPE_IQ1_S, n_in, n_out));
        dw.push_back(upload(w.back().data(), w.back().size()));
        slots.w[e] = dw.back();
        slots.x[e] = (const uint8_t*) dx + (size_t) e * n_in / 32 * 36;
        slots.y[e] = dy + (size_t) e * n_out;
    }
    check(kernels::native_mmvq_supported(GGML_TYPE_IQ1_S), "IQ1_S is a native MMVQ type");
    check(kernels::native_mmvq_slots(GGML_TYPE_IQ1_S, slots, count, n_in, n_out, ts), "native_mmvq_slots takes IQ1_S");
    ck(cudaStreamSynchronize(ts), "slots");
    std::vector<float> got((size_t) count * n_out);
    ck(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost), "download");
    double worst = 0;
    for (int e = 0; e < count; ++e) {
        const auto wf = dequant(GGML_TYPE_IQ1_S, w[e], n_in, n_out);
        const auto xv = q8_1_values(slots.x[e], n_in);
        const auto ref = iq1_s_ref(w[e], q8_1_bytes(slots.x[e], n_in), n_in, n_out);
        worst = std::max(worst, ref_err(got, (size_t) e * n_out, ref, wf, xv, n_in, n_out));
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "IQ1_S slots, 8 x [2304 x 2304]: vs ggml-cuda's dot in double %.2e",
                  worst);
    check(worst < 1e-5, buf);
    // native_mmvq with 1 to 3 columns of the first matrix
    for (int nc = 1; nc <= 3; ++nc) {
        kernels::native_mmvq(GGML_TYPE_IQ1_S, dw[0], dx, dy, n_in, n_out, nc, ts);
        ck(cudaStreamSynchronize(ts), "mmvq");
        std::vector<float> g((size_t) nc * n_out);
        ck(cudaMemcpy(g.data(), dy, g.size() * 4, cudaMemcpyDeviceToHost), "download");
        const auto wf = dequant(GGML_TYPE_IQ1_S, w[0], n_in, n_out);
        double wc = 0;
        for (int c = 0; c < nc; ++c) {
            const void* xc = (const uint8_t*) dx + (size_t) c * n_in / 32 * 36;
            wc = std::max(wc, ref_err(g, (size_t) c * n_out, iq1_s_ref(w[0], q8_1_bytes(xc, n_in), n_in, n_out), wf,
                                      q8_1_values(xc, n_in), n_in, n_out));
        }
        std::snprintf(buf, sizeof buf, "IQ1_S native_mmvq, %d column(s): max rel error %.2e", nc, wc);
        check(wc < 1e-5, buf);
    }
    for (void* p : dw) cudaFree(p);
    cudaFree(dx), cudaFree(dy), cudaFree(dxf);
}

// Q2_0 down in the decode products: a token's 8 experts
void test_q2_0_decode() {
    const int n_in = 1152, n_out = 2304, count = 8;
    void* dx = nullptr;
    float *dy = nullptr, *dxf = nullptr;
    ck(cudaMalloc(&dx, (size_t) count * n_in / 32 * 36), "malloc");
    ck(cudaMalloc(&dy, (size_t) count * n_out * 4), "malloc");
    ck(cudaMalloc(&dxf, (size_t) count * n_in * 4), "malloc");
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf((size_t) count * n_in);
    for (float& v : xf) v = nd(rng);
    ck(cudaMemcpy(dxf, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice), "upload");
    kernels::quantize_q8_1_rows(dxf, count, n_in, dx, ts);
    kernels::MmvqSlots slots{};
    std::vector<std::vector<uint8_t>> w;
    std::vector<void*> dw;
    for (int e = 0; e < count; ++e) {
        w.push_back(random_rows(GGML_TYPE_Q2_0, n_in, n_out));
        dw.push_back(upload(w.back().data(), w.back().size()));
        slots.w[e] = dw.back();
        slots.x[e] = (const uint8_t*) dx + (size_t) e * n_in / 32 * 36;
        slots.y[e] = dy + (size_t) e * n_out;
    }
    const bool took = kernels::native_mmvq_slots(GGML_TYPE_Q2_0, slots, count, n_in, n_out, ts);
    ck(cudaStreamSynchronize(ts), "slots");
    std::vector<float> got((size_t) count * n_out);
    ck(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost), "download");
    double worst = 0;
    for (int e = 0; e < count; ++e)
        worst = std::max(worst, row_err(got, dequant(GGML_TYPE_Q2_0, w[e], n_in, n_out), q8_1_values(slots.x[e], n_in),
                                        n_in, n_out, 1, (size_t) e * n_out));
    char buf[160];
    std::snprintf(buf, sizeof buf, "Q2_0 slots, 8 x [2304 x 1152]: max |y - dequantized dot| / sum|terms| %.2e", worst);
    check(took && worst < 1e-5, buf);
    for (void* p : dw) cudaFree(p);
    cudaFree(dx), cudaFree(dy), cudaFree(dxf);
}

void test_q2_0_mmq(int rows) {
    const int n_in = 1152, n_out = 2304;   // Inkling's down matrix
    check(poly::Mmq::fits(GGML_TYPE_Q2_0, n_out, n_in, poly::Mmq::q2_0) &&
              !poly::Mmq::fits(GGML_TYPE_Q2_0, n_out, n_in),
          "Mmq::fits(Q2_0) only when asked for");
    const auto w = random_rows(GGML_TYPE_Q2_0, n_in, n_out);
    std::vector<uint8_t> w4(ggml_row_size(GGML_TYPE_Q4_0, n_in) * n_out);
    q2_0_to_q4_0(w.data(), w4.data(), (int64_t) n_in * n_out);
    void *dw = upload(w.data(), w.size()), *dw4 = upload(w4.data(), w4.size());
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) rows * n_in);
    for (float& v : x) v = nd(rng);
    float *dxf, *d2, *d4;
    void* xq;
    ck(cudaMalloc(&dxf, x.size() * 4), "malloc");
    ck(cudaMemcpy(dxf, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMalloc(&xq, poly::Mmq::q8_bytes(rows, n_in)), "malloc");
    ck(cudaMalloc(&d2, (size_t) rows * n_out * 4), "malloc");
    ck(cudaMalloc(&d4, (size_t) rows * n_out * 4), "malloc");
    poly::Mmq mmq;
    poly::Mmq::quantize(dxf, nullptr, xq, GGML_TYPE_Q2_0, n_in, n_in, rows, ts);
    mmq.mul(dw, GGML_TYPE_Q2_0, n_out, n_in, 0, 1, xq, nullptr, nullptr, rows, rows, d2, n_out, ts);
    poly::Mmq::quantize(dxf, nullptr, xq, GGML_TYPE_Q4_0, n_in, n_in, rows, ts);
    mmq.mul(dw4, GGML_TYPE_Q4_0, n_out, n_in, 0, 1, xq, nullptr, nullptr, rows, rows, d4, n_out, ts);
    ck(cudaStreamSynchronize(ts), "mmq");
    std::vector<float> g2((size_t) rows * n_out), g4(g2.size());
    ck(cudaMemcpy(g2.data(), d2, g2.size() * 4, cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(g4.data(), d4, g4.size() * 4, cudaMemcpyDeviceToHost), "download");
    const auto wf = dequant(GGML_TYPE_Q2_0, w, n_in, n_out);
    double vs_ref = 0, vs_q4 = 0;
    for (int t = 0; t < rows; ++t) {
        std::vector<float> xt(x.begin() + (size_t) t * n_in, x.begin() + (size_t) (t + 1) * n_in);
        vs_ref = std::max(vs_ref, row_err(g2, wf, xt, n_in, n_out, 1, (size_t) t * n_out));
        for (int r = 0; r < n_out; ++r) {
            double mag = 0;
            for (int i = 0; i < n_in; ++i) mag += std::fabs(double(wf[(size_t) r * n_in + i]) * xt[i]);
            vs_q4 = std::max(vs_q4, std::fabs(double(g2[(size_t) t * n_out + r]) - g4[(size_t) t * n_out + r]) / mag);
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "Q2_0 MMQ, %d rows x [2304 x 1152]: vs float dot %.2e (activation's Q8_1), vs the same as Q4_0 %.2e",
                  rows, vs_ref, vs_q4);
    // Q2_0's tiles take the -1 offset from the activation's block sums, which the Q8_1 rows keep in halves (as
    // Q4_0's take its -8): the two differ by those roundings, a few 1e-5 of sum |terms|
    check(vs_ref < 2e-2 && vs_q4 < 2e-4, buf);
    cudaFree(dw), cudaFree(dw4), cudaFree(dxf), cudaFree(xq), cudaFree(d2), cudaFree(d4);
}

// ---- timings ------------------------------------------------------------------------------------------------------

cudaStream_t bs;
template <class F> float time_us(F&& f, int reps = 200) {
    f();
    ck(cudaStreamSynchronize(bs), "warm-up");
    cudaGraph_t g;
    cudaGraphExec_t ex;
    ck(cudaStreamBeginCapture(bs, cudaStreamCaptureModeRelaxed), "capture");
    for (int i = 0; i < reps; ++i) f();
    ck(cudaStreamEndCapture(bs, &g), "end capture");
    ck(cudaGraphInstantiate(&ex, g, 0), "instantiate");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    float best = 1e30f;
    for (int k = 0; k < 6; ++k) {
        cudaEventRecord(e0, bs);
        cudaGraphLaunch(ex, bs);
        cudaEventRecord(e1, bs);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        if (k) best = std::min(best, ms);
    }
    cudaGraphExecDestroy(ex), cudaGraphDestroy(g);
    return 1000.f * best / reps;
}

void bench() {
    cudaStreamCreateWithFlags(&bs, cudaStreamNonBlocking);
    const int count = 8;
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Mat {
        const char* name;
        ggml_type t;
        int n_in, n_out;
    };
    // a token's 8 routed experts: gate|up (IQ1_S) and down (Q2_0), and other types of the same shapes beside them
    const Mat mats[] = {{"gate|up IQ1_S", GGML_TYPE_IQ1_S, 2304, 2304}, {"gate|up Q5_0", GGML_TYPE_Q5_0, 2304, 2304},
                        {"gate|up IQ2_S", GGML_TYPE_IQ2_S, 2304, 2304}, {"down Q2_0", GGML_TYPE_Q2_0, 1152, 2304},
                        {"down Q4_0", GGML_TYPE_Q4_0, 1152, 2304}};
    void* dx;
    ck(cudaMalloc(&dx, (size_t) count * 2304 / 32 * 36), "malloc");
    float* dy;
    ck(cudaMalloc(&dy, (size_t) count * 2304 * 4), "malloc");
    std::printf("\ndecode: native_mmvq_slots, 8 experts' matrices in one launch (us a launch; GB/s of the weights)\n");
    for (const Mat& m : mats) {
        std::vector<void*> dw;
        kernels::MmvqSlots s{};
        size_t bytes = 0;
        for (int e = 0; e < count; ++e) {
            const auto w = random_rows(m.t, m.n_in, m.n_out);
            bytes += w.size();
            dw.push_back(upload(w.data(), w.size()));
            s.w[e] = dw.back();
            s.x[e] = (const uint8_t*) dx + (size_t) e * m.n_in / 32 * 36;
            s.y[e] = dy + (size_t) e * m.n_out;
        }
        const float us = time_us([&] { kernels::native_mmvq_slots(m.t, s, count, m.n_in, m.n_out, bs); });
        std::printf("bench %-32s %8.2f us  %7.1f GB/s  (%.2f MB)\n", m.name, us, bytes / (us * 1e3), bytes / 1e6);
        for (void* p : dw) cudaFree(p);
    }
    std::printf("\nprompt: Mmq::mul, one matrix, 512 rows (us a call)\n");
    const int rows = 512;
    float* dxf;
    ck(cudaMalloc(&dxf, (size_t) rows * 2304 * 4), "malloc");
    std::vector<float> xf((size_t) rows * 2304);
    for (float& v : xf) v = nd(rng);
    ck(cudaMemcpy(dxf, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice), "upload");
    void* xq;
    ck(cudaMalloc(&xq, poly::Mmq::q8_bytes(rows, 2304)), "malloc");
    float* dd;
    ck(cudaMalloc(&dd, (size_t) rows * 2304 * 4), "malloc");
    poly::Mmq mmq;
    const Mat pm[] = {{"down Q2_0 [2304 x 1152]", GGML_TYPE_Q2_0, 1152, 2304},
                      {"down Q4_0 [2304 x 1152]", GGML_TYPE_Q4_0, 1152, 2304},
                      {"gate|up IQ1_S [2304 x 2304]", GGML_TYPE_IQ1_S, 2304, 2304},
                      {"gate|up Q5_0 [2304 x 2304]", GGML_TYPE_Q5_0, 2304, 2304}};
    for (const Mat& m : pm) {
        const auto w = random_rows(m.t, m.n_in, m.n_out);
        void* dw = upload(w.data(), w.size());
        poly::Mmq::quantize(dxf, nullptr, xq, m.t, m.n_in, m.n_in, rows, bs);
        const float us = time_us([&] {
            mmq.mul(dw, m.t, m.n_out, m.n_in, 0, 1, xq, nullptr, nullptr, rows, rows, dd, m.n_out, bs);
        }, 50);
        std::printf("bench %-32s %8.2f us\n", m.name, us);
        cudaFree(dw);
    }
    cudaFree(dx), cudaFree(dy), cudaFree(dxf), cudaFree(xq), cudaFree(dd);
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no usable CUDA device: skipped\n");
        return 77;
    }
    cudaStreamCreateWithFlags(&ts, cudaStreamNonBlocking);
    test_iq1_s_decode();
    test_q2_0_decode();
    test_q2_0_mmq(16);
    test_q2_0_mmq(300);
    if (argc > 1 && std::string(argv[1]) == "bench") bench();
    std::printf("%s: %d failure%s\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
