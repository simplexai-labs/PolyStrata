// src/poly/nvfp4_x2_test.cpp - STRATA_PREFILL_NVFP4=w4a4x2 on the host, no card needed:
//   * the two-term activation rows (nvfp4_x2_quant.hpp) read back through ggml's own NVFP4 rows
//     (dequantize_row_nvfp4) are the input to about 8 bits, and one term alone is not;
//   * the kernel's tiles (nvfp4_x2.hpp: the stage copies, the fragments' shared-memory reads, the accumulators'
//     places), run lane by lane with the MMA as the PTX ISA defines it, give the product of ggml's dequantized
//     weights with the dequantized rows, for matrices with tails in both directions, one K step, and experts of 0 to
//     150 rows;
//   * rows with a NaN or an infinity, rows too small for a normal row scale, and subnormal values: the rule of
//     nvfp4_x2_quant.hpp, and on x86 the same bytes with the CPU flushing subnormals as the card does.
//
// usage: poly_nvfp4_x2_test   (exit 0: every check passes; 77: a CUDA older than 12.8, without the FP4 conversions)
#include "nvfp4_x2.hpp"
#include "nvfp4_x2_quant.hpp"

#include <cstdio>

#if !defined(STRATA_X2_HAS_FP4)
int main() {
    std::printf("CUDA %d has no FP4 conversions (12.8 and later): nothing to test\n", CUDART_VERSION);
    return 77;
}
#else

#include "ggml-quants.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64)
#include <pmmintrin.h>
#include <xmmintrin.h>
#define STRATA_X2_TEST_FTZ 1
#endif

namespace {

using namespace strata::poly;
using namespace strata::poly::x2;

int failures = 0;
std::mt19937 rng(7);

void check(bool ok, const std::string& what) {
    std::printf("%-86s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

double rel(const std::vector<double>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += (a[i] - b[i]) * (a[i] - b[i]);
        d += b[i] * b[i];
    }
    return std::sqrt(n / std::max(d, 1e-300));
}

// rows of activations as a prompt has them: Gaussian, heavy-tailed, a few outliers, SwiGLU-like
std::vector<float> rows_of(int kind, int rows, int cols) {
    std::vector<float> x((size_t) rows * cols);
    std::normal_distribution<float> n(0.f, 1.f);
    std::student_t_distribution<float> t(2.f);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (auto& v : x) {
        switch (kind) {
            case 0: v = n(rng); break;
            case 1: v = t(rng); break;
            case 2: v = n(rng) * (u(rng) < 0.002f ? 100.f : 1.f); break;
            default: { const float a = std::max(n(rng), 0.f); v = a * a * n(rng); }
        }
    }
    return x;
}

// The host's quantize(): row r of x into its two terms, as the card's quantize_kernel writes it.
void quantize(const float* x, int rows, int cols, std::vector<uint8_t>& xq) {
    xq.assign(rows_bytes(rows, cols), 0);
    float* gs = (float*) (xq.data() + (size_t) rows * row_bytes(cols));
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t) r * cols;
        gs[r] = host_row_scale(xr, cols);
        for (int s = 0; s < cols / 16; ++s) quantize16(xr + 16 * s, gs[r], xq.data() + (size_t) r * row_bytes(cols), s);
    }
}

// Row r read back through ggml: g (term 1 + term 2), or the first term only.
std::vector<double> dequantize(const std::vector<uint8_t>& xq, int rows, int cols, int r, bool both) {
    const float g = row_scales(xq.data(), rows, cols)[r];
    const int nb = cols / kBlock;
    std::vector<block_nvfp4> t[2] = {std::vector<block_nvfp4>(nb), std::vector<block_nvfp4>(nb)};
    for (int b = 0; b < nb; ++b)
        for (int k = 0; k < 2; ++k)
            std::memcpy(&t[k][b], xq.data() + (size_t) r * row_bytes(cols) + (size_t) b * kPairBytes + k * kBlockBytes,
                        kBlockBytes);
    std::vector<float> f0(cols), f1(cols);
    dequantize_row_nvfp4(t[0].data(), f0.data(), cols);
    dequantize_row_nvfp4(t[1].data(), f1.data(), cols);
    std::vector<double> out(cols);
    for (int i = 0; i < cols; ++i) out[i] = (double) g * ((double) f0[i] + (both ? (double) f1[i] : 0.0));
    return out;
}

void check_rounding() {
    // E2M1 to nearest, ties to the even code, saturating: what the card's cvt.rn.satfinite.e2m1x2 does
    const float in[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 7.0f, 100.f, -0.25f, -5.0f, -2.6f};
    const float want[] = {0.f, 1.f, 1.f, 2.f, 2.f, 4.f, 4.f, 6.f, 6.f, -0.f, -4.f, -3.f};
    bool ok = true;
    for (size_t i = 0; i < sizeof(in) / sizeof(in[0]); ++i) {
        const unsigned c = __nv_cvt_float_to_fp4(in[i], __NV_E2M1, cudaRoundNearest) & 15u;
        ok = ok && e2m1_value(c) == want[i];
    }
    check(ok, "E2M1 rounding: nearest, ties to even, saturating");
    // UE4M3: the value of each byte the scales take is ggml's (which keeps them as twice the value)
    bool same = true;
    for (int b = 0; b < 0x7f; ++b) {
        block_nvfp4 blk = {};
        blk.d[0] = (uint8_t) b;
        blk.qs[0] = 2;   // value 0 is code 2: 1.0
        float f[64];
        dequantize_row_nvfp4(&blk, f, 64);
        same = same && f[0] == ue4m3_value((uint8_t) b);
    }
    check(same, "UE4M3 scale bytes decode to ggml's values");
}

void check_terms() {
    const char* kinds[] = {"Gaussian", "Student t(2)", "outliers x100", "SwiGLU-like"};
    for (int kind = 0; kind < 4; ++kind)
        for (const int cols : {2304, 1152}) {
            const int rows = 48;
            const std::vector<float> x = rows_of(kind, rows, cols);
            std::vector<uint8_t> xq;
            quantize(x.data(), rows, cols, xq);
            std::vector<double> ref, two, one;
            for (int r = 0; r < rows; ++r) {
                for (int i = 0; i < cols; ++i) ref.push_back(x[(size_t) r * cols + i]);
                const std::vector<double> a = dequantize(xq, rows, cols, r, true), b = dequantize(xq, rows, cols, r, false);
                two.insert(two.end(), a.begin(), a.end());
                one.insert(one.end(), b.begin(), b.end());
            }
            const double e2 = rel(two, ref), e1 = rel(one, ref);
            char what[160];
            std::snprintf(what, sizeof what, "two terms, %s rows of %d: %.3f%% (one term %.2f%%)", kinds[kind], cols,
                          100 * e2, 100 * e1);
            check(e2 < 0.012 && e2 < e1 / 6, what);
        }
    // a row of zeros, and one whose blocks are far below its largest value
    std::vector<float> x(2 * 256, 0.f);
    for (int i = 0; i < 256; ++i) x[256 + i] = i == 0 ? 1000.f : 1e-3f * (float) ((i % 7) - 3);
    std::vector<uint8_t> xq;
    quantize(x.data(), 2, 256, xq);
    const std::vector<double> z = dequantize(xq, 2, 256, 0, true), s = dequantize(xq, 2, 256, 1, true);
    bool ok = std::all_of(z.begin(), z.end(), [](double v) { return v == 0.0; }) && std::fabs(s[0] - 1000.0) < 1e-3;
    double worst = 0;   // the small blocks' scales are UE4M3 subnormals there, or 0 at worst
    for (int i = 1; i < 256; ++i) worst = std::max(worst, std::fabs(s[i] - x[256 + i]));
    ok = ok && worst <= 0.0031;
    std::printf("  large value %.6f, small values off by %.2e at most\n", s[0], worst);
    check(ok, "a row of zeros, and small blocks under a large value");
}

// Rows the rule of nvfp4_x2_quant.hpp is for: non-finite values, rows under kTinyRow, rows of subnormal values and
// rows whose blocks span the whole float range.
std::vector<float> edge_rows(int cols, int& rows) {
    std::vector<std::vector<float>> r;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-150, 10);
    const auto row = [&](float scale) {
        std::vector<float> v(cols);
        for (auto& x : v) x = nd(rng) * scale;
        return v;
    };
    r.push_back(row(1.f)); r.back()[cols / 3] = NAN;
    r.push_back(row(1.f)); r.back()[cols - 1] = INFINITY;
    r.push_back(row(1.f)); r.back()[0] = -INFINITY;
    r.push_back(row(1.f)); r.back()[5] = NAN; r.back()[6] = INFINITY;
    r.push_back(row(1e-40f));                                  // subnormal values only
    r.push_back(row(1e-33f));                                  // normal values, a row scale that would be subnormal
    r.push_back(row(0.05f * kTinyRow)); r.back()[1] = 0.99f * kTinyRow;   // just under kTinyRow
    r.push_back(row(0.05f * kTinyRow)); r.back()[1] = kTinyRow;           // its smallest row
    r.push_back(row(1e-29f));
    r.push_back(row(1.f));                                     // subnormal values among normal ones
    for (int i = 0; i < cols; i += 3) r.back()[i] = (i % 2 ? -1.f : 1.f) * 1e-39f;
    for (int k = 0; k < 6; ++k) {                              // every 16 values at a scale of its own, 2^-150 to 2^10
        r.push_back(row(1.f));
        for (int s = 0; s < cols / 16; ++s) {
            const float f = std::ldexp(1.f, ex(rng));
            for (int j = 0; j < 16; ++j) r.back()[16 * s + j] *= f;
        }
    }
    r.push_back(std::vector<float>(cols, -0.f));
    rows = (int) r.size();
    std::vector<float> x;
    for (auto& v : r) x.insert(x.end(), v.begin(), v.end());
    return x;
}

void check_edges() {
    const int cols = 256;
    int rows = 0;
    const std::vector<float> x = edge_rows(cols, rows);
    std::vector<uint8_t> xq;
    quantize(x.data(), rows, cols, xq);
    const float* gs = row_scales(xq.data(), rows, cols);
    const auto zeros = [&](int r) {
        const uint8_t* p = xq.data() + (size_t) r * row_bytes(cols);
        return std::all_of(p, p + row_bytes(cols), [](uint8_t b) { return b == 0; });
    };
    bool nan_ok = true;
    for (int r = 0; r < 4; ++r) nan_ok = nan_ok && std::isnan(gs[r]) && zeros(r);
    check(nan_ok, "a row with a NaN or an infinity: zeros and a NaN row scale");
    check(gs[4] == 0.f && zeros(4) && gs[5] == 0.f && zeros(5) && gs[6] == 0.f && zeros(6),
          "a row under kTinyRow (subnormal values; a subnormal row scale): zeros");
    bool normal = true;
    double worst = 0;
    for (int r = 7; r < rows - 1; ++r) {
        normal = normal && gs[r] >= FLT_MIN;
        std::vector<double> ref(x.begin() + (size_t) r * cols, x.begin() + (size_t) (r + 1) * cols);
        worst = std::max(worst, rel(dequantize(xq, rows, cols, r, true), ref));
    }
    char what[160];
    std::snprintf(what, sizeof what, "rows from kTinyRow up, subnormal values, 16-value blocks from 2^-150 to 2^10: "
                  "normal row scales, %.3f%% at worst", 100 * worst);
    check(normal && worst < 0.012, what);
    check(gs[rows - 1] == 0.f && zeros(rows - 1), "a row of -0: zeros");
#if defined(STRATA_X2_TEST_FTZ)
    // the card's unit flushes subnormal inputs and results to zero: the same bytes with the CPU doing so
    const unsigned csr = _mm_getcsr();
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    std::vector<uint8_t> ftz;
    quantize(x.data(), rows, cols, ftz);
    _mm_setcsr(csr);
    size_t diff = 0;
    for (size_t i = 0; i < xq.size(); ++i) diff += xq[i] != ftz[i];
    std::snprintf(what, sizeof what, "the same rows with subnormals flushed to zero, as on the card: %zu bytes differ",
                  diff);
    check(diff == 0, what);
#endif
}

// ---------------------------------------------------------------------------------- the kernel, lane by lane

// mma.sync m16n8k64 kind::mxf4nvf4 with scale_vec::4X, from the registers of the 32 lanes as the PTX ISA lays them
// out (independently of nvfp4_x2.hpp's helpers): A register i of lane l is row l/4 + 8 (i % 2), values
// 8 (l % 4) + 32 (i / 2) + 0..7, low nibble first; B register i is column l/4, values 8 (l % 4) + 32 i + 0..7; A's
// scale register from lanes with l % 4 = 0 (row l/4) and 1 (row l/4 + 8), B's from l % 4 = 0, byte k / 16 for value k;
// D register i of lane l is row l/4 + 8 (i / 2), column 2 (l % 4) + i % 2.
void mma_lanes(float d[32][4], const uint32_t a[32][4], const uint32_t b[32][2], const uint32_t sa[32],
               const uint32_t sb[32]) {
    uint8_t A[16][64], B[64][8];
    float SA[16][4], SB[8][4];
    for (int l = 0; l < 32; ++l) {
        for (int i = 0; i < 4; ++i)
            for (int n = 0; n < 8; ++n) A[l / 4 + 8 * (i % 2)][8 * (l % 4) + 32 * (i / 2) + n] = (a[l][i] >> (4 * n)) & 15;
        for (int i = 0; i < 2; ++i)
            for (int n = 0; n < 8; ++n) B[8 * (l % 4) + 32 * i + n][l / 4] = (b[l][i] >> (4 * n)) & 15;
        for (int k = 0; k < 4; ++k) {
            if (l % 4 == 0) SA[l / 4][k] = ue4m3_value((uint8_t) (sa[l] >> (8 * k)));
            if (l % 4 == 1) SA[l / 4 + 8][k] = ue4m3_value((uint8_t) (sa[l] >> (8 * k)));
            if (l % 4 == 0) SB[l / 4][k] = ue4m3_value((uint8_t) (sb[l] >> (8 * k)));
        }
    }
    for (int l = 0; l < 32; ++l)
        for (int i = 0; i < 4; ++i) {
            const int row = l / 4 + 8 * (i / 2), col = 2 * (l % 4) + i % 2;
            double s = 0;
            for (int k = 0; k < 64; ++k)
                s += (double) e2m1_value(A[row][k]) * SA[row][k / 16] * e2m1_value(B[k][col]) * SB[col][k / 16];
            d[l][i] += (float) s;
        }
}

uint32_t ld32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// mul_kernel's block (bx, by, e), the same steps on the host
void run_block(const uint8_t* w0, size_t stride, int w_rows, int w_cols, const uint8_t* xq, const int32_t* bounds,
               const int32_t* ids, int total_rows, float* dst, int64_t ld, int bx, int by, int e) {
    std::vector<uint8_t> sm(kSmemBytes);
    const int b0 = bounds[e], b1 = bounds[e + 1], t0 = b0 + by * kN, r0 = bx * kM;
    if (t0 >= b1) return;
    const uint8_t* w = w0 + (size_t) e * stride;
    const size_t w_row = (size_t) (w_cols / kBlock) * kBlockBytes, x_row = row_bytes(w_cols);
    const float* gs = row_scales(xq, total_rows, w_cols);
    const int steps = w_cols / kK;
    const auto load = [&](int stage, int step) {
        for (int i = 0; i < kWPieces; ++i) {
            int row, byte;
            w_piece(i, row, byte);
            const int wr = std::min(r0 + row, w_rows - 1);
            std::memcpy(&sm[w_smem(stage, row, 0) + byte], w + (size_t) wr * w_row + (size_t) step * (kBlocksPerStep * kBlockBytes) + byte, 8);
        }
        for (int i = 0; i < kTPieces; ++i) {
            int tok, byte;
            t_piece(i, tok, byte);
            const int xr = std::min(t0 + tok, b1 - 1);
            std::memcpy(&sm[t_smem(stage, tok, 0, 0) + byte], xq + (size_t) xr * x_row + (size_t) step * kTStride + byte, 16);
        }
    };
    static float acc[8][2][4][32][4];
    std::memset(acc, 0, sizeof acc);
    for (int step = 0; step < steps; ++step) {
        const int st = step % kStages;
        load(st, step);
        for (int warp = 0; warp < 8; ++warp) {
            const int wr0 = warp_rows(warp), wt0 = warp_toks(warp);
            for (int blk = 0; blk < kBlocksPerStep; ++blk) {
                uint32_t A[2][32][4], sa[2][32];
                for (int mt = 0; mt < 2; ++mt)
                    for (int lane = 0; lane < 32; ++lane) {
                        const uint8_t* p = &sm[w_smem(st, wr0 + 16 * mt, blk)];
                        for (int i = 0; i < 4; ++i) A[mt][lane][i] = ld32(p + a_row(lane, i) * kWStride + 4 * a_word(lane, i));
                        sa[mt][lane] = ld32(p + a_scale_row(lane) * kWStride);
                    }
                for (int term = 0; term < 2; ++term)
                    for (int nt = 0; nt < 4; ++nt) {
                        uint32_t B[32][2], sb[32];
                        for (int lane = 0; lane < 32; ++lane) {
                            const uint8_t* p = &sm[t_smem(st, wt0 + 8 * nt + lane / 4, blk, term)];
                            B[lane][0] = ld32(p + 4 * b_word(lane, 0));
                            B[lane][1] = ld32(p + 4 * b_word(lane, 1));
                            sb[lane] = ld32(p);
                        }
                        for (int mt = 0; mt < 2; ++mt) mma_lanes(acc[warp][mt][nt], A[mt], B, sa[mt], sb);
                    }
            }
        }
    }
    std::vector<float> out((size_t) kN * kOutStride);
    for (int warp = 0; warp < 8; ++warp)
        for (int mt = 0; mt < 2; ++mt)
            for (int nt = 0; nt < 4; ++nt)
                for (int lane = 0; lane < 32; ++lane)
                    for (int i = 0; i < 4; ++i)
                        out[(size_t) (warp_toks(warp) + 8 * nt + c_col(lane, i)) * kOutStride + warp_rows(warp) + 16 * mt +
                            c_row(lane, i)] = acc[warp][mt][nt][lane][i];
    const int cols = std::min(kM, w_rows - r0);
    for (int tok = 0; tok < kN && t0 + tok < b1; ++tok)
        for (int c = 0; c < cols; ++c)
            dst[(int64_t) ids[t0 + tok] * ld + r0 + c] = out[(size_t) tok * kOutStride + c] * gs[t0 + tok];
}

void check_kernel(int w_rows, int w_cols, const std::vector<int>& counts) {
    const int n = (int) counts.size();
    const int nb = w_cols / kBlock;
    const size_t w_row = (size_t) nb * kBlockBytes, stride = w_row * w_rows + 40;   // matrices 8-byte aligned apart
    std::vector<uint8_t> w(stride * n);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> wf(w_cols);
    std::vector<std::vector<double>> wd((size_t) n * w_rows);
    for (int e = 0; e < n; ++e)
        for (int r = 0; r < w_rows; ++r) {
            for (auto& v : wf) v = nd(rng) * 0.05f;
            block_nvfp4* b = (block_nvfp4*) (w.data() + e * stride + r * w_row);
            quantize_row_nvfp4_ref(wf.data(), b, w_cols);
            dequantize_row_nvfp4(b, wf.data(), w_cols);
            wd[(size_t) e * w_rows + r].assign(wf.begin(), wf.end());
        }
    std::vector<int32_t> bounds(n + 1, 0);
    for (int e = 0; e < n; ++e) bounds[e + 1] = bounds[e] + counts[e];
    const int total = bounds[n];
    int most = 1;
    for (int c : counts) most = std::max(most, c);
    const std::vector<float> x = rows_of(3, total, w_cols);
    std::vector<uint8_t> xq;
    quantize(x.data(), total, w_cols, xq);
    std::vector<int32_t> ids(total);   // row i to a scattered dst row, as the down product's places
    for (int i = 0; i < total; ++i) ids[i] = (int32_t) ((i * 7 + 3) % total);
    const int64_t ld = w_rows + 5;
    std::vector<float> dst((size_t) total * ld, NAN);
    for (int e = 0; e < n; ++e)
        for (int by = 0; by < (most + kN - 1) / kN; ++by)
            for (int bx = 0; bx < (w_rows + kM - 1) / kM; ++bx)
                run_block(w.data(), stride, w_rows, w_cols, xq.data(), bounds.data(), ids.data(), total, dst.data(), ld,
                          bx, by, e);
    std::vector<double> got, want, exact;
    bool untouched = true;
    for (int e = 0; e < n; ++e)
        for (int i = bounds[e]; i < bounds[e + 1]; ++i) {
            const std::vector<double> xd = dequantize(xq, total, w_cols, i, true);
            for (int r = 0; r < w_rows; ++r) {
                const std::vector<double>& wr = wd[(size_t) e * w_rows + r];
                double s = 0, s0 = 0;
                for (int k = 0; k < w_cols; ++k) {
                    s += wr[k] * xd[k];
                    s0 += wr[k] * x[(size_t) i * w_cols + k];
                }
                got.push_back(dst[(size_t) ids[i] * ld + r]);
                want.push_back(s);
                exact.push_back(s0);
            }
            for (int64_t c = w_rows; c < ld; ++c) untouched = untouched && std::isnan(dst[(size_t) ids[i] * ld + c]);
        }
    const double e_tile = rel(got, want), e_exact = rel(got, exact);
    char what[200];
    std::snprintf(what, sizeof what, "tiles, %d experts of [%d x %d], %d rows: vs the same arithmetic %.1e, vs floats %.3f%%",
                  n, w_rows, w_cols, total, e_tile, 100 * e_exact);
    check(e_tile < 1e-5 && e_exact < 0.015 && untouched, what);
}

}  // namespace

int main() {
    check_rounding();
    check_terms();
    check_edges();
    check_kernel(256, 256, {70, 0, 5, 64});
    check_kernel(256, 128, {3, 64, 65});        // one K step
    check_kernel(200, 384, {150, 1});           // a matrix's rows end inside a tile
    check_kernel(128, 1152, {33});              // Inkling 2's down product's K
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}

#endif  // STRATA_X2_HAS_FP4
