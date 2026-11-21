// src/poly/cpu_lowbit_test.cpp - cpu_lowbit.cpp's Q2_0 and IQ1_S dot products against the dequantized rows in
// double precision; and, with `bench`, their speed against ggml-cpu's from memory (a buffer much larger than the
// caches), one thread and all of them, one token and four.
//
// usage: poly_cpu_lowbit_test [bench [threads...]]   (exit 0: every check passes; 77: no AVX2/FMA/F16C)
#include "cpu_lowbit.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace strata::poly;

int failures = 0;
std::mt19937 rng(7);

const ggml_type_traits_cpu* cpu(ggml_type t) { return ggml_get_type_traits_cpu(t); }

// random rows of a type: random code bytes, and block scales of a realistic size
std::vector<uint8_t> random_rows(ggml_type t, int n, int rows) {
    const size_t rb = ggml_row_size(t, n);
    std::vector<uint8_t> w(rb * rows);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t bb = ggml_type_size(t);
    std::uniform_real_distribution<float> sc(0.002f, 0.05f);
    for (size_t off = 0; off + bb <= w.size(); off += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(&w[off], &d, 2);   // d is the first field of q2_0, iq1_s, q4_0 and q5_0
    }
    return w;
}

std::vector<uint8_t> activation(ggml_type wt, const std::vector<float>& x) {
    const ggml_type at = cpu(wt)->vec_dot_type;
    std::vector<uint8_t> a(ggml_row_size(at, (int64_t) x.size()));
    cpu(at)->from_float(x.data(), a.data(), (int64_t) x.size());
    return a;
}

void parity(ggml_type t, int n, int rows) {
    const ggml_vec_dot_t mine = lowbit_vec_dot(t);
    const auto w = random_rows(t, n, rows);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (float& v : x) v = nd(rng);
    const auto a = activation(t, x);
    // the activation as the dot products see it
    std::vector<float> ax(n), wx(n);
    if (cpu(t)->vec_dot_type == GGML_TYPE_Q8_K) {   // ggml-base has no to_float for Q8_K: d (float), 256 codes, sums
        const size_t bb = ggml_type_size(GGML_TYPE_Q8_K);
        for (int b = 0; b < n / 256; ++b) {
            float d;
            std::memcpy(&d, a.data() + b * bb, 4);
            for (int i = 0; i < 256; ++i) ax[b * 256 + i] = d * (float) (int8_t) a[b * bb + 4 + i];
        }
    } else {
        ggml_get_type_traits(cpu(t)->vec_dot_type)->to_float(a.data(), ax.data(), n);
    }
    const size_t rb = ggml_row_size(t, n);
    double max_rel = 0;
    for (int r = 0; r < rows; ++r) {
        float got = 0;
        mine(n, &got, 0, w.data() + r * rb, 0, a.data(), 0, 1);
        ggml_get_type_traits(t)->to_float(w.data() + r * rb, wx.data(), n);
        double d = 0, mag = 0;
        for (int i = 0; i < n; ++i) d += double(wx[i]) * ax[i], mag += std::fabs(double(wx[i]) * ax[i]);
        max_rel = std::max(max_rel, std::fabs(got - d) / std::max(mag, 1e-30));
    }
    const bool ok = max_rel < 1e-5;
    if (!ok) ++failures;
    std::printf("%-8s n=%-5d %5d rows: max |dot - dequantized dot| / sum|terms| %.2e  %s\n", ggml_type_name(t), n,
                rows, max_rel, ok ? "ok" : "FAIL");
}

// GB/s of the weights' bytes: `threads` threads each walking its share of rows of a buffer of `bytes`, `tokens`
// activations a row (the rows a pool task takes, as cpu_experts does: a row, then every token of it)
double bench(ggml_type t, ggml_vec_dot_t dot, int n, size_t bytes, int threads, int tokens) {
    const size_t rb = ggml_row_size(t, n);
    const int rows = int(bytes / rb);
    static std::vector<uint8_t> w;
    w.assign(rb * rows, 0);
    {
        std::mt19937 g(3);
        for (size_t i = 0; i < w.size(); i += 4096) w[i] = (uint8_t) g();
        const size_t bb = ggml_type_size(t);
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f);
        for (size_t off = 0; off + bb <= w.size(); off += bb) std::memcpy(&w[off], &d, 2);
    }
    std::vector<float> x(n, 0.5f);
    const auto a = activation(t, x);
    std::vector<std::vector<uint8_t>> acts(tokens, a);
    auto run = [&](int th) {
        float sink = 0;
        // tasks of 32 rows, dealt out in turn (cpu_experts' kRowsPerTask): a row, then every token of it
        for (int r0 = 32 * th; r0 < rows; r0 += 32 * threads)
            for (int r = r0; r < std::min(rows, r0 + 32); ++r)
                for (int k = 0; k < tokens; ++k) {
                    float s;
                    dot(n, &s, 0, w.data() + (size_t) r * rb, 0, acts[k].data(), 0, 1);
                    sink += s;
                }
        volatile float keep = sink;
        (void) keep;
    };
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        for (int i = 1; i < threads; ++i) ts.emplace_back(run, i);
        run(0);
        for (auto& th : ts) th.join();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return double(rb) * rows / best / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    if (!lowbit_vec_dot(GGML_TYPE_Q2_0) || !lowbit_vec_dot(GGML_TYPE_IQ1_S)) {
        std::printf("this CPU has no AVX2/FMA/F16C: skipped\n");
        return 77;
    }
    parity(GGML_TYPE_Q2_0, 1152, 4000);   // Inkling's down rows
    parity(GGML_TYPE_Q2_0, 64, 2000);
    parity(GGML_TYPE_Q2_0, 4096, 500);
    parity(GGML_TYPE_IQ1_S, 2304, 4000);  // Inkling's gate|up rows
    parity(GGML_TYPE_IQ1_S, 256, 2000);
    parity(GGML_TYPE_IQ1_S, 4096, 500);

    if (argc > 1 && std::string(argv[1]) == "bench") {
        std::vector<int> threads;
        for (int i = 2; i < argc; ++i) threads.push_back(std::atoi(argv[i]));
        if (threads.empty()) threads = {1, (int) std::thread::hardware_concurrency()};
        const size_t bytes = size_t(256) << 20;   // 256 MiB of rows, 8 times the L3: from memory, as the experts are
        struct Case {
            const char* name;
            ggml_type t;
            ggml_vec_dot_t dot;
            int n;
        };
        const Case cases[] = {
            {"Q2_0  ggml", GGML_TYPE_Q2_0, cpu(GGML_TYPE_Q2_0)->vec_dot, 1152},
            {"Q2_0  this AVX2", GGML_TYPE_Q2_0, lowbit_vec_dot(GGML_TYPE_Q2_0), 1152},
            {"IQ1_S ggml", GGML_TYPE_IQ1_S, cpu(GGML_TYPE_IQ1_S)->vec_dot, 2304},
            {"IQ1_S this AVX2", GGML_TYPE_IQ1_S, lowbit_vec_dot(GGML_TYPE_IQ1_S), 2304},
            {"IQ4_NL ggml (for comparison)", GGML_TYPE_IQ4_NL, cpu(GGML_TYPE_IQ4_NL)->vec_dot, 2304},
        };
        std::printf("\nGB/s of each type's own bytes\n");
        for (int tokens : {1, 4})
            for (int th : threads)
                for (const Case& c : cases)
                    std::printf("%-32s threads %2d tokens %d: %7.2f GB/s\n", c.name, th, tokens,
                                bench(c.t, c.dot, c.n, bytes, th, tokens));
    }
    std::printf("%s: %d failure%s\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
