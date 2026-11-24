// tools/vision/own/vit_cpu.cpp - see vit_cpu.h
#include "vit_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace strata_vit {

namespace {

constexpr int kKC = 256;        // the k block of a matrix product: a packed weight panel is kKC x (2 tiles) floats
constexpr int kQB = 48;         // the unit of query rows of an attention task (a multiple of every tier's MR)
constexpr int kQBDefault = 96;  // query rows of an attention task: every task streams all of a head's K and V once, so 96 halves that traffic against 48
                                // (MSVC, Ryzen 5 7600, big1024: 0.988 of mtmd-CPU's time against 1.000 with 48 in the same session; RESULTS-OWN-3V.md D2)
constexpr int kTK = 256;        // keys per attention tile
constexpr int kRowsChunk = 4096;   // rows of the activations that are live at once outside the attention

struct Mat {                     // a weight matrix [rows = out][cols = in]
    std::vector<uint16_t> pk;    // BF16 / F16 matrices as 16-bit panels [ceil(rows / NR)][cols][NR] (the rows past `rows` zero), made at load; the stored copy is not kept
    std::vector<float> f32;      // anything else (F32, Q8_0 expanded)
    int kind = 0;                // 0 float32, 1 BF16, 2 F16
    int rows = 0, cols = 0;
    const void* data() const { return f32.data(); }       // kind 0 only (the 16-bit matrices live in pk)
};

struct Layer {
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, qkv_b, o_b, up_b, down_b;
    Mat qkv, o, up, down;
};

struct GemmaLayer {   // gemma4v: RMS norm weights, the per-head q / k norms, no biases
    std::vector<float> ln1, ln2, attn_post, ffn_post, q_norm, k_norm;
    Mat q, k, v, o, gate, up, down;
};

struct MiniLayer {    // minicpmv4_7: a SigLIP layer (LayerNorms, q / k / v / o with biases, a tanh-GELU MLP); the window merger's attention uses ln1, q..o
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, q_b, k_b, v_b, o_b, up_b, down_b;
    Mat q, k, v, o, up, down;
};

struct InkLayer {     // inkling2v: LayerNorms, q / k / v / o with biases, layer scales, an erf-GELU MLP with biases
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, q_b, k_b, v_b, o_b, ls1, ls2, up_b, down_b;
    Mat q, k, v, o, up, down;
};

// ggml's erf GELU (FFN_GELU_ERF): 0.5 x (1 + erf(x / sqrt 2))
void gelu_erf_rows(float* x, size_t ld, int rows, int n) {
    for (int r = 0; r < rows; ++r) {
        float* p = x + (size_t) r * ld;
        for (int i = 0; i < n; ++i) p[i] = 0.5f * p[i] * (1.0f + std::erf(p[i] * 0.70710678118654752f));
    }
}

// y = x / sqrt(mean(x^2) + eps) (* w), ggml_rms_norm's order: the sum in double, the scale in float
void rms_row(const float* x, float* y, int n, const float* w, float eps) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double) x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float) (s / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * scale;
    if (w)
        for (int i = 0; i < n; ++i) y[i] *= w[i];
}

}  // namespace

struct VitCpu::Impl {
    VitGeometry g;
    const CpuOps* ops = nullptr;
    int act_round = 0;                  // STRATA_VIT_REF_ROUND: 0 none, 1 BF16, 2 F16 (the weight's type)
    std::vector<float> patch_b, pos, post_w, post_b, mm0_b, mm2_b;
    Mat patch_w, mm0, mm2;
    std::vector<Layer> layers;
    std::vector<GemmaLayer> glayers;          // gemma4v
    std::vector<float> std_bias, std_scale;   // gemma4v, when the file has them
    Mat proj;                                 // gemma4v: mm.input_projection
    std::vector<MiniLayer> mlayers;           // minicpmv4_7
    MiniLayer wm;                             // minicpmv4_7: the window merger's attention
    std::vector<float> ds_ln_w, ds_ln_b, ds_up_b, ds_down_b, mmn_w, mmn_b, mm_up_b, mm_down_b;
    Mat ds_up, ds_down, mm_up, mm_down;
    std::vector<InkLayer> ilayers;            // inkling2v
    std::vector<float> cls, pre_w, pre_b;     // inkling2v: the CLS row (its embedding plus its position) and the first norm
    std::unique_ptr<Team> team;
    int team_threads = 0;

    bool read_f(const Gguf& gg, const std::string& name, std::vector<float>& dst, std::string& err) {
        const GgufTensorInfo* t = gg.tensor(name);
        if (!t) { err = "the mmproj file lacks the tensor " + name; return false; }
        dst.resize((size_t) t->nelem());
        return gg.read_f32(*t, dst.data(), err);
    }
    bool read_m(const Gguf& gg, const std::string& name, Mat& m, std::string& err) {
        const GgufTensorInfo* t = gg.tensor(name);
        if (!t || t->ne.size() != 2) { err = "the mmproj file lacks the matrix " + name; return false; }
        m.cols = (int) t->ne[0];
        m.rows = (int) t->ne[1];
        const size_t n = (size_t) t->nelem();
        if (t->type == GT_F16 || t->type == GT_BF16) {
            m.kind = t->type == GT_BF16 ? 1 : 2;
            std::vector<uint16_t> raw(n);
            if (!gg.read_raw(*t, raw.data(), err)) return false;
            // panels of NR rows, k-major inside a panel: a product task converts a [kc][NR] slice to float with one contiguous pass
            const int NR = ops->NR;
            const size_t ptiles = ((size_t) m.rows + NR - 1) / NR;
            m.pk.assign(ptiles * (size_t) m.cols * NR, 0);
            for (int r = 0; r < m.rows; ++r) {
                const uint16_t* src = &raw[(size_t) r * m.cols];
                uint16_t* dst = &m.pk[(size_t) (r / NR) * m.cols * NR + (size_t) (r % NR)];
                for (int k = 0; k < m.cols; ++k) dst[(size_t) k * NR] = src[k];
            }
            return true;
        }
        m.kind = 0;
        m.f32.resize(n);
        return gg.read_f32(*t, m.f32.data(), err);
    }

    template <class F>
    void run(int n, F f) { team->run(n, [&](int task, int thr) { f(task, thr); }); }

    // rows of simple per-row work, 16 rows per task
    template <class F>
    void rows_par(int n, F f) {
        const int per = 16, tasks = (n + per - 1) / per;
        team->run(tasks, [&](int task, int) {
            const int r1 = std::min(n, (task + 1) * per);
            for (int r = task * per; r < r1; ++r) f(r);
        });
    }

    void round_inplace(float* p, size_t n, int kind) {
        if (kind == 0) return;
        const size_t blk = 16384;
        const int tasks = (int) ((n + blk - 1) / blk);
        team->run(tasks, [&](int task, int) {
            const size_t i0 = (size_t) task * blk, i1 = std::min(n, i0 + blk);
            ops->round16(p + i0, i1 - i0, kind);
        });
    }

    // C[M][w.rows] (init-mode, see CpuOps::gemm) = A[M][w.cols] * w^T (+ bias), optionally GELU on the result
    void linear(const Mat& w, const float* A, size_t lda, int M, float* C, size_t ldc, int init, const float* bias, bool gelu, std::vector<std::vector<float>>& scratch) {
        const int NR = ops->NR, PW = 2 * NR, K = w.cols, N = w.rows;
        const int npan = (N + PW - 1) / PW;
        const int target = 4 * team->size();
        int mslabs = std::max(1, std::min((target + npan - 1) / npan, (M + 47) / 48));
        const int slab = (((M + mslabs - 1) / mslabs) + ops->MR - 1) / ops->MR * ops->MR;
        mslabs = (M + slab - 1) / slab;
        team->run(npan * mslabs, [&](int task, int thr) {
            const int p = task % npan, s = task / npan;
            const int m0 = s * slab, mr = std::min(slab, M - m0);
            const int n0 = p * PW, nv = std::min(PW, N - n0);
            float* buf = scratch[(size_t) thr].data();
            for (int k0 = 0; k0 < K; k0 += kKC) {
                const int kc = std::min(kKC, K - k0);
                for (int t = 0; t < 2; ++t) {
                    const int tv = std::max(0, std::min(NR, nv - t * NR));
                    if (tv <= 0) continue;
                    if (w.kind == 0) ops->pack_w(w.data(), 0, (size_t) K, n0 + t * NR, tv, k0, kc, buf + (size_t) t * kKC * NR);
                    else ops->conv(&w.pk[((size_t) (p * 2 + t) * K + (size_t) k0) * NR], w.kind, (size_t) kc * NR, buf + (size_t) t * kKC * NR);
                }
                for (int t = 0; t < 2; ++t) {
                    const int tv = std::max(0, std::min(NR, nv - t * NR));
                    if (tv <= 0) continue;
                    const int col = n0 + t * NR;
                    ops->gemm(A + (size_t) m0 * lda + k0, lda, mr, buf + (size_t) t * kKC * NR, (size_t) NR, kc, tv, C + (size_t) m0 * ldc + col, ldc,
                              k0 == 0 ? init : 1, bias ? bias + col : nullptr);
                }
            }
            if (gelu) ops->gelu(C + (size_t) m0 * ldc + n0, ldc, mr, nv);
        });
    }

    // softmax(q k^T) v of every head over N tokens, the scale already in q: qkv rows [q | k | v] of 3 D floats (and a tile of slack after the
    // last), ctx [N][D].  KT [H][HD][Np] and VP [H][N][HD] are scratch sized for N; sscr per thread.
    void attention(const float* qkv, int N, float* ctx, std::vector<float>& KT, std::vector<float>& VP, std::vector<std::vector<float>>& sscr) {
        const int D = g.width, H = g.heads, HD = g.head_dim, Np = (N + 63) / 64 * 64, OD = (HD + 15) / 16 * 16, QB = kQBDefault, tasks_q = (N + QB - 1) / QB;
        const size_t D3 = (size_t) 3 * D;
        const CpuOps& o = *ops;
        {   // K transposed per head (KT[h][d][j]) and V packed per head (VP[h][j][d])
            const int jb = 64, njb = (N + jb - 1) / jb;
            team->run(H * njb, [&](int task, int) {
                const int hh = task / njb, j0 = (task % njb) * jb, j1 = std::min(N, j0 + jb);
                for (int j = j0; j < j1; ++j) {
                    const float* kr = qkv + (size_t) j * D3 + D + (size_t) hh * HD;
                    float* kt = KT.data() + (size_t) hh * HD * Np + j;
                    for (int d = 0; d < HD; ++d) kt[(size_t) d * Np] = kr[d];
                    std::memcpy(VP.data() + ((size_t) hh * N + j) * HD, kr + D, sizeof(float) * (size_t) HD);
                }
            });
        }
        team->run(H * tasks_q, [&](int task, int thr) {
            const int hh = task / tasks_q, q0 = (task % tasks_q) * QB, qr = std::min(QB, N - q0);
            float* S = sscr[(size_t) thr].data();
            float* O = S + (size_t) QB * kTK;
            float* mx = O + (size_t) QB * OD;
            float* ls = mx + QB;
            std::memset(O, 0, sizeof(float) * (size_t) QB * OD);
            for (int r = 0; r < QB; ++r) { mx[r] = -3.0e38f; ls[r] = 0.0f; }
            const float* Q = qkv + (size_t) q0 * D3 + (size_t) hh * HD;
            const float* KTh = KT.data() + (size_t) hh * HD * Np;
            const float* Vh = VP.data() + (size_t) hh * N * HD;
            for (int j0 = 0; j0 < N; j0 += kTK) {
                const int tk = std::min(kTK, N - j0);
                o.gemm(Q, D3, qr, KTh + j0, (size_t) Np, HD, tk, S, (size_t) kTK, 0, nullptr);
                o.softmax_update(S, (size_t) kTK, qr, tk, mx, ls, O, (size_t) OD, HD);
                o.gemm(S, (size_t) kTK, qr, Vh + (size_t) j0 * HD, (size_t) HD, tk, HD, O, (size_t) OD, 1, nullptr);
            }
            for (int r = 0; r < qr; ++r) {
                const float inv = 1.0f / ls[r];
                float* dst = ctx + (size_t) (q0 + r) * D + (size_t) hh * HD;
                for (int d = 0; d < HD; ++d) dst[d] = O[(size_t) r * OD + d] * inv;
            }
        });
    }

    bool load_gemma4(const Gguf& gg, std::string& err);
    bool encode_gemma4(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err);
    bool load_minicpmv(const Gguf& gg, std::string& err);
    bool encode_minicpmv(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err);
    bool load_inkling(const Gguf& gg, std::string& err);
    bool encode_inkling(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err);
};

VitCpu::VitCpu() : impl_(new Impl()) {}
VitCpu::~VitCpu() = default;
const VitGeometry& VitCpu::geometry() const { return impl_->g; }
const char* VitCpu::isa_name() const { return impl_->ops ? impl_->ops->name : "none"; }

bool VitCpu::load(const Gguf& gg, const VitGeometry& g, std::string& err) {
    Impl& m = *impl_;
    std::string why;
    if (!vit_geometry_supported(g, why)) { err = "this tower is not supported: " + why; return false; }
    m.g = g;
    m.ops = &cpu_ops(quant::best_isa());
    if (g.projector == "gemma4v") return m.load_gemma4(gg, err);
    if (g.minicpmv()) return m.load_minicpmv(gg, err);
    if (g.inkling()) return m.load_inkling(gg, err);
    // the patch embedding: the two temporal taps summed (a still picture is fed to both), kept float32 (a small matrix)
    {
        const GgufTensorInfo* t0 = gg.tensor("v.patch_embd.weight");
        if (!t0) { err = "the mmproj file lacks v.patch_embd.weight"; return false; }
        std::vector<float> w((size_t) t0->nelem());
        if (!gg.read_f32(*t0, w.data(), err)) return false;
        if (w.size() != (size_t) g.width * 3 * g.patch * g.patch) { err = "the patch embedding has an unexpected size"; return false; }
        if (g.patch_taps == 2) {
            const GgufTensorInfo* t1 = gg.tensor("v.patch_embd.weight.1");
            std::vector<float> w1(w.size());
            if (!t1 || !gg.read_f32(*t1, w1.data(), err)) { if (err.empty()) err = "the mmproj file lacks v.patch_embd.weight.1"; return false; }
            for (size_t i = 0; i < w.size(); ++i) w[i] += w1[i];
        }
        m.patch_w.kind = 0;
        m.patch_w.rows = g.width;
        m.patch_w.cols = 3 * g.patch * g.patch;
        m.patch_w.f32 = std::move(w);
    }
    if (!m.read_f(gg, "v.patch_embd.bias", m.patch_b, err) || !m.read_f(gg, "v.position_embd.weight", m.pos, err) || !m.read_f(gg, "v.post_ln.weight", m.post_w, err) ||
        !m.read_f(gg, "v.post_ln.bias", m.post_b, err) || !m.read_m(gg, "mm.0.weight", m.mm0, err) || !m.read_f(gg, "mm.0.bias", m.mm0_b, err) ||
        !m.read_m(gg, "mm.2.weight", m.mm2, err) || !m.read_f(gg, "mm.2.bias", m.mm2_b, err))
        return false;
    m.layers.assign((size_t) g.layers, Layer{});
    for (int il = 0; il < g.layers; ++il) {
        Layer& L = m.layers[(size_t) il];
        const std::string p = "v.blk." + std::to_string(il) + ".";
        if (!m.read_f(gg, p + "ln1.weight", L.ln1_w, err) || !m.read_f(gg, p + "ln1.bias", L.ln1_b, err) || !m.read_f(gg, p + "ln2.weight", L.ln2_w, err) ||
            !m.read_f(gg, p + "ln2.bias", L.ln2_b, err) || !m.read_m(gg, p + "attn_qkv.weight", L.qkv, err) || !m.read_f(gg, p + "attn_qkv.bias", L.qkv_b, err) ||
            !m.read_m(gg, p + "attn_out.weight", L.o, err) || !m.read_f(gg, p + "attn_out.bias", L.o_b, err) || !m.read_m(gg, p + "ffn_up.weight", L.up, err) ||
            !m.read_f(gg, p + "ffn_up.bias", L.up_b, err) || !m.read_m(gg, p + "ffn_down.weight", L.down, err) || !m.read_f(gg, p + "ffn_down.bias", L.down_b, err))
            return false;
    }
    m.act_round = vit_cpu_round_kind(m.layers.empty() ? 0 : m.layers[0].qkv.kind, std::getenv("STRATA_VIT_REF_ROUND"));
    return true;
}

int vit_cpu_round_kind(int weight_kind, const char* env) {
    // No rounding by default, for every weight type: the tower then follows the float32 reference (VitRef).  Until 11-14 BF16 weights (Qwen3.8) rounded
    // the matrix inputs to BF16 by default, which matched llama.cpp's reader on code4217.png (0.9997) but not float32: 16 pictures, 12,308 tokens, 45 under
    // cosine 0.99 against VitRef with the rounding, 0 without (code4217 token 47: 0.627 -> 1.000).  STRATA_VIT_REF_ROUND=1 brings the rounding back.
    // float32 or Q8_0-expanded matrices have no 16-bit type to round to.
    if (weight_kind != 1 && weight_kind != 2) return 0;
    if (env && *env) return std::atoi(env) != 0 ? weight_kind : 0;
    return 0;
}

bool VitCpu::encode(const PreparedImage& img, int threads, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
    Impl& m = *impl_;
    const VitGeometry& g = m.g;
    if (g.projector == "gemma4v" || g.minicpmv() || g.inkling()) {
        if (m.glayers.empty() && m.mlayers.empty() && m.ilayers.empty()) { err = "the tower is not loaded"; return false; }
        const int nth = threads > 0 ? threads : (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
        if (!m.team || m.team_threads != nth) { m.team.reset(new Team(nth)); m.team_threads = nth; }
        if (g.inkling()) return m.encode_inkling(img, rows, nx, ny, err);
        return g.minicpmv() ? m.encode_minicpmv(img, rows, nx, ny, err) : m.encode_gemma4(img, rows, nx, ny, err);
    }
    if (m.layers.empty()) { err = "the tower is not loaded"; return false; }
    const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn;
    if (HD > 128 || HD % 4 != 0) { err = "this head size is not handled by the CPU tower"; return false; }
    if (img.w <= 0 || img.h <= 0 || img.w % g.align() != 0 || img.h % g.align() != 0) { err = "the picture size is not a multiple of patch * merge"; return false; }
    int nth = threads > 0 ? threads : (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
    if (!m.team || m.team_threads != nth) { m.team.reset(new Team(nth)); m.team_threads = nth; }
    const CpuOps& ops = *m.ops;
    // STRATA_VIT_PROF=1 (a diagnostic, stderr only, off by default): the time of each stage (every stage ends with its team run, so no extra wait)
    static const bool prof = std::getenv("STRATA_VIT_PROF") != nullptr;
    double pt[12] = {0};
    auto pclock = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double pmark = prof ? pclock() : 0.0;
    auto stage = [&](int s) {
        if (!prof) return;
        const double t = pclock();
        pt[s] += t - pmark;
        pmark = t;
    };
    const int pw = img.w / P, ph = img.h / P, N = pw * ph, nbx = pw / 2;
    nx = pw / 2;
    ny = ph / 2;
    const int R = std::min(N, kRowsChunk);

    std::vector<int> ty((size_t) N), tx((size_t) N);
    for (int t = 0; t < N; ++t) {
        const int b = t / 4, r = t % 4;
        ty[(size_t) t] = 2 * (b / nbx) + r / 2;
        tx[(size_t) t] = 2 * (b % nbx) + r % 2;
    }
    std::vector<std::vector<float>> scratch((size_t) nth, std::vector<float>((size_t) kKC * 2 * ops.NR));

    // ---- patch embedding: im2col of a chunk of rows in merge order, one product; then the position table (bilinear, align corners) ----
    const int K0 = 3 * P * P;
    std::vector<float> x((size_t) N * D);
    {
        std::vector<float> A((size_t) R * K0);
        const size_t plane = (size_t) img.w * img.h;
        for (int r0 = 0; r0 < N; r0 += R) {
            const int rn = std::min(R, N - r0);
            m.rows_par(rn, [&](int i) {
                const int t = r0 + i;
                float* a = &A[(size_t) i * K0];
                const int py = ty[(size_t) t], px = tx[(size_t) t];
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky)
                        std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
            });
            m.linear(m.patch_w, A.data(), (size_t) K0, rn, x.data() + (size_t) r0 * D, (size_t) D, 0, m.patch_b.data(), false, scratch);
        }
    }
    {
        const int G = g.pos_grid;
        float sfx = (float) pw / G, sfy = (float) ph / G;
        if (pw > 1 && G > 1) sfx = (float) (pw - 1) / (G - 1);
        if (ph > 1 && G > 1) sfy = (float) (ph - 1) / (G - 1);
        m.rows_par(N, [&](int t) {
            const float fy = (float) ty[(size_t) t] / sfy, fx = (float) tx[(size_t) t] / sfx;
            int y0 = (int) std::floor(fy), x0 = (int) std::floor(fx), y1 = y0 + 1, x1 = x0 + 1;
            y0 = std::max(0, std::min(y0, G - 1)); y1 = std::max(0, std::min(y1, G - 1));
            x0 = std::max(0, std::min(x0, G - 1)); x1 = std::max(0, std::min(x1, G - 1));
            const float dy = std::max(0.0f, std::min(fy - (float) y0, 1.0f)), dx = std::max(0.0f, std::min(fx - (float) x0, 1.0f));
            const float* a = &m.pos[((size_t) y0 * G + x0) * D];
            const float* b = &m.pos[((size_t) y0 * G + x1) * D];
            const float* c = &m.pos[((size_t) y1 * G + x0) * D];
            const float* d = &m.pos[((size_t) y1 * G + x1) * D];
            ops.pos_add(&x[(size_t) t * D], a, b, c, d, dx, dy, D);      // a tier kernel: its result depends on multiply-add contraction (cpu_tier.h)
        });
    }

    // ---- rotary tables: pair j of a head (elements j and j + HD/2), j < HD/4 turns with the row, the rest with the column ----
    const int half = HD / 2, quarter = HD / 4;
    std::vector<float> rc((size_t) ph * quarter), rs((size_t) ph * quarter), cc((size_t) pw * quarter), cs((size_t) pw * quarter);
    {
        const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) half);
        std::vector<float> inv((size_t) quarter);
        float th = 1.0f;
        for (int j = 0; j < quarter; ++j) { inv[(size_t) j] = th; th *= theta_scale; }
        for (int p = 0; p < ph; ++p)
            for (int j = 0; j < quarter; ++j) { const float a = (float) p * inv[(size_t) j]; rc[(size_t) p * quarter + j] = std::cos(a); rs[(size_t) p * quarter + j] = std::sin(a); }
        for (int p = 0; p < pw; ++p)
            for (int j = 0; j < quarter; ++j) { const float a = (float) p * inv[(size_t) j]; cc[(size_t) p * quarter + j] = std::cos(a); cs[(size_t) p * quarter + j] = std::sin(a); }
    }
    const float kq_scale = 1.0f / std::sqrt((float) HD);
    stage(0);

    // ---- the layers ----
    const size_t slack = 256;                                    // the product with V reads up to a tile past a head's columns
    std::vector<float> qkv((size_t) N * 3 * D + slack, 0.0f), ctx((size_t) N * D);
    const int Np = (N + 63) / 64 * 64;
    std::vector<float> KT((size_t) H * HD * Np, 0.0f);
    std::vector<float> hbuf((size_t) R * D), up((size_t) R * F);
    const int OD = (HD + 15) / 16 * 16;
    // Attention layout (RESULTS-OWN-3V.md D2): V packed per head (VP[h][j][d], rows of HD floats) instead of read in place from the qkv rows (a 3,456-float
    // stride: one cache line used per row and key); and queries per task. Both change only addresses and the grouping of query rows (every query row's
    // sums run in the same order), so the results are bit-identical.  STRATA_VIT_CPU_VPACK=0 / STRATA_VIT_CPU_QB=<multiple of 48>: diagnostics.
    static const bool vpack = [] { const char* e = std::getenv("STRATA_VIT_CPU_VPACK"); return !(e && std::atoi(e) == 0); }();
    static const int QB = [] { const char* e = std::getenv("STRATA_VIT_CPU_QB"); const int v = e ? std::atoi(e) : 0; return v >= kQB && v % kQB == 0 ? v : kQBDefault; }();
    const int attn_tasks_q = (N + QB - 1) / QB;
    std::vector<float> VP(vpack ? (size_t) H * N * HD + slack : 0);
    std::vector<std::vector<float>> sscr((size_t) nth, std::vector<float>((size_t) QB * kTK + (size_t) QB * OD + 2 * QB));
    const size_t D3 = (size_t) 3 * D;

    for (const Layer& L : m.layers) {
        // q, k, v of every token
        for (int r0 = 0; r0 < N; r0 += R) {
            const int rn = std::min(R, N - r0);
            m.rows_par(rn, [&](int i) { ops.layernorm(x.data() + (size_t) (r0 + i) * D, hbuf.data() + (size_t) i * D, D, L.ln1_w.data(), L.ln1_b.data(), g.eps); });
            m.round_inplace(hbuf.data(), (size_t) rn * D, m.act_round);
            m.linear(L.qkv, hbuf.data(), (size_t) D, rn, qkv.data() + (size_t) r0 * D3, D3, 0, L.qkv_b.data(), false, scratch);
        }
        stage(1);
        m.rows_par(N, [&](int t) {       // the rotary on q and k (with the attention's scale on q, after the optional 16-bit rounding)
            float* row = qkv.data() + (size_t) t * D3;
            const float* c1 = &rc[(size_t) ty[(size_t) t] * quarter];
            const float* s1 = &rs[(size_t) ty[(size_t) t] * quarter];
            const float* c2 = &cc[(size_t) tx[(size_t) t] * quarter];
            const float* s2 = &cs[(size_t) tx[(size_t) t] * quarter];
            ops.rope_row(row, (size_t) D, H, HD, c1, s1, c2, s2);       // a tier kernel, as pos_add
            if (m.act_round) ops.round16(row, D3, 2);
            for (int i = 0; i < D; ++i) row[i] *= kq_scale;
        });
        // K transposed per head: KT[h][d][j]
        {
            const int jb = 64, njb = (N + jb - 1) / jb;
            m.team->run(H * njb, [&](int task, int) {
                const int hh = task / njb, j0 = (task % njb) * jb, j1 = std::min(N, j0 + jb);
                for (int j = j0; j < j1; ++j) {
                    const float* kr = qkv.data() + (size_t) j * D3 + D + (size_t) hh * HD;
                    float* kt = KT.data() + (size_t) hh * HD * Np + j;
                    for (int d = 0; d < HD; ++d) kt[(size_t) d * Np] = kr[d];
                    if (vpack) std::memcpy(VP.data() + ((size_t) hh * N + j) * HD, kr + D, sizeof(float) * (size_t) HD);   // v follows k in the row
                }
            });
        }
        stage(2);
        // tiled attention with an online softmax: a task is a head and a block of kQB queries
        m.team->run(H * attn_tasks_q, [&](int task, int thr) {
            const int hh = task / attn_tasks_q, q0 = (task % attn_tasks_q) * QB, qr = std::min(QB, N - q0);
            float* S = sscr[(size_t) thr].data();
            float* O = S + (size_t) QB * kTK;
            float* mx = O + (size_t) QB * OD;
            float* ls = mx + QB;
            std::memset(O, 0, sizeof(float) * (size_t) QB * OD);
            for (int r = 0; r < QB; ++r) { mx[r] = -3.0e38f; ls[r] = 0.0f; }
            double ta = 0.0, tb = 0.0, tc = 0.0;
            const bool pa = prof && thr == 0;      // the split of the attention, sampled on the calling thread's tasks
            double t0 = pa ? pclock() : 0.0;
            const float* Q = qkv.data() + (size_t) q0 * D3 + (size_t) hh * HD;
            const float* KTh = KT.data() + (size_t) hh * HD * Np;
            const float* Vh = vpack ? VP.data() + (size_t) hh * N * HD : qkv.data() + 2 * (size_t) D + (size_t) hh * HD;
            const size_t ldv = vpack ? (size_t) HD : D3;
            for (int j0 = 0; j0 < N; j0 += kTK) {
                const int tk = std::min(kTK, N - j0);
                ops.gemm(Q, D3, qr, KTh + j0, (size_t) Np, HD, tk, S, (size_t) kTK, 0, nullptr);
                if (pa) { const double t = pclock(); ta += t - t0; t0 = t; }
                ops.softmax_update(S, (size_t) kTK, qr, tk, mx, ls, O, (size_t) OD, HD);
                if (pa) { const double t = pclock(); tb += t - t0; t0 = t; }
                ops.gemm(S, (size_t) kTK, qr, Vh + (size_t) j0 * ldv, ldv, tk, HD, O, (size_t) OD, 1, nullptr);
                if (pa) { const double t = pclock(); tc += t - t0; t0 = t; }
            }
            if (pa) { pt[8] += ta; pt[9] += tb; pt[10] += tc; pt[11] += 1.0; }
            for (int r = 0; r < qr; ++r) {
                const float inv = 1.0f / ls[r];
                float* dst = ctx.data() + (size_t) (q0 + r) * D + (size_t) hh * HD;
                for (int d = 0; d < HD; ++d) dst[d] = O[(size_t) r * OD + d] * inv;
            }
        });
        stage(3);
        m.round_inplace(ctx.data(), ctx.size(), m.act_round);
        for (int r0 = 0; r0 < N; r0 += R) {      // x += o(ctx)
            const int rn = std::min(R, N - r0);
            m.linear(L.o, ctx.data() + (size_t) r0 * D, (size_t) D, rn, x.data() + (size_t) r0 * D, (size_t) D, 2, L.o_b.data(), false, scratch);
        }
        stage(4);
        for (int r0 = 0; r0 < N; r0 += R) {      // x += down(gelu(up(LN2 x)))
            const int rn = std::min(R, N - r0);
            m.rows_par(rn, [&](int i) { ops.layernorm(x.data() + (size_t) (r0 + i) * D, hbuf.data() + (size_t) i * D, D, L.ln2_w.data(), L.ln2_b.data(), g.eps); });
            m.round_inplace(hbuf.data(), (size_t) rn * D, m.act_round);
            m.linear(L.up, hbuf.data(), (size_t) D, rn, up.data(), (size_t) F, 0, L.up_b.data(), true, scratch);
            m.round_inplace(up.data(), (size_t) rn * F, m.act_round);
            stage(5);
            m.linear(L.down, up.data(), (size_t) F, rn, x.data() + (size_t) r0 * D, (size_t) D, 2, L.down_b.data(), false, scratch);
            stage(6);
        }
    }

    // ---- post norm and the merger (4 consecutive tokens = one 4D row) ----
    std::vector<float>& h = ctx;       // the context is no longer needed: the post-norm output takes its place
    m.rows_par(N, [&](int t) { ops.layernorm(x.data() + (size_t) t * D, h.data() + (size_t) t * D, D, m.post_w.data(), m.post_b.data(), g.eps); });
    m.round_inplace(h.data(), h.size(), m.act_round);
    const int M = N / 4, Kin = 4 * D, Hid = g.merger_hidden, O = g.out_dim;
    rows.assign((size_t) M * O, 0.0f);
    const int RM = std::max(1, R / 4);
    std::vector<float> m1((size_t) std::min(M, RM) * Hid);
    for (int r0 = 0; r0 < M; r0 += RM) {
        const int rn = std::min(RM, M - r0);
        m.linear(m.mm0, h.data() + (size_t) r0 * Kin, (size_t) Kin, rn, m1.data(), (size_t) Hid, 0, m.mm0_b.data(), true, scratch);
        m.round_inplace(m1.data(), (size_t) rn * Hid, m.act_round);
        m.linear(m.mm2, m1.data(), (size_t) Hid, rn, rows.data() + (size_t) r0 * O, (size_t) O, 0, m.mm2_b.data(), false, scratch);
    }
    stage(7);
    if (prof)
        std::fprintf(stderr, "vit-cpu-prof N=%d threads=%d isa=%s: patch+pos %.1f, ln+qkv %.1f, rope+KT %.1f, attention %.1f (thread 0, %d tasks: QK %.1f, softmax %.1f, PV %.1f), "
                             "o-proj %.1f, ln+up+gelu %.1f, down %.1f, post+merger %.1f ms\n",
                     N, nth, ops.name, pt[0], pt[1], pt[2], pt[3], (int) pt[11], pt[8], pt[9], pt[10], pt[4], pt[5], pt[6], pt[7]);
    return true;
}

bool VitCpu::Impl::load_gemma4(const Gguf& gg, std::string& err) {
    {   // the patch embedding, float32 (a small matrix), no bias
        const GgufTensorInfo* t = gg.tensor("v.patch_embd.weight");
        if (!t) { err = "the mmproj file lacks v.patch_embd.weight"; return false; }
        patch_w.kind = 0;
        patch_w.rows = g.width;
        patch_w.cols = 3 * g.patch * g.patch;
        patch_w.f32.resize((size_t) t->nelem());
        if (!gg.read_f32(*t, patch_w.f32.data(), err)) return false;
    }
    if (!read_f(gg, "v.position_embd.weight", pos, err) || !read_m(gg, "mm.input_projection.weight", proj, err)) return false;
    if (g.std_norm && (!read_f(gg, "v.std_bias", std_bias, err) || !read_f(gg, "v.std_scale", std_scale, err))) return false;
    glayers.assign((size_t) g.layers, GemmaLayer{});
    for (int il = 0; il < g.layers; ++il) {
        GemmaLayer& L = glayers[(size_t) il];
        const std::string p = "v.blk." + std::to_string(il) + ".";
        if (!read_f(gg, p + "ln1.weight", L.ln1, err) || !read_f(gg, p + "ln2.weight", L.ln2, err) ||
            !read_f(gg, p + "attn_post_norm.weight", L.attn_post, err) || !read_f(gg, p + "ffn_post_norm.weight", L.ffn_post, err) ||
            !read_f(gg, p + "attn_q_norm.weight", L.q_norm, err) || !read_f(gg, p + "attn_k_norm.weight", L.k_norm, err) ||
            !read_m(gg, p + "attn_q.weight", L.q, err) || !read_m(gg, p + "attn_k.weight", L.k, err) || !read_m(gg, p + "attn_v.weight", L.v, err) ||
            !read_m(gg, p + "attn_out.weight", L.o, err) || !read_m(gg, p + "ffn_gate.weight", L.gate, err) ||
            !read_m(gg, p + "ffn_up.weight", L.up, err) || !read_m(gg, p + "ffn_down.weight", L.down, err))
            return false;
    }
    return true;
}

// Gemma 4's tower, mtmd's clip_graph_gemma4v step for step: tokens in reading order (row t is the patch at column t % pw, row t / pw), the
// pixels as 2 v - 1, the x and y position tables added, then per layer RMS norm, q / k / v, an RMS norm per head on q and k (weighted)
// and v (not), the rotary of the first half of a head by the column and of the second half by the row (NeoX pairs, i with i + HD / 4),
// attention without a scale, the output's RMS norm, the residual, then the same around a gated quick-GELU FFN.  After the layers: the
// average of every pool x pool square times sqrt(width), the standardisation, an RMS norm and the projection.  The rows come out in reading
// order of the pooled grid, and the grid is reported as one line (nx = rows, ny = 1): Gemma 4's image tokens take positions as text does.
bool VitCpu::Impl::encode_gemma4(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
    const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn, K = g.pool, T = g.pos_grid;
    if (HD > 128 || HD % 4 != 0) { err = "this head size is not handled by the CPU tower"; return false; }
    if (img.w <= 0 || img.h <= 0 || img.w % (P * K) != 0 || img.h % (P * K) != 0) { err = "the picture size is not a multiple of patch * pool"; return false; }
    const int pw = img.w / P, ph = img.h / P, N = pw * ph;
    if (pw > T || ph > T) { err = "the picture has more patches a side than the position tables"; return false; }
    const CpuOps& o = *ops;
    const int nth = team->size();
    std::vector<std::vector<float>> scratch((size_t) nth, std::vector<float>((size_t) kKC * 2 * o.NR));
    const int R = std::min(N, kRowsChunk);

    const int K0 = 3 * P * P;
    std::vector<float> x((size_t) N * D);
    {
        std::vector<float> A((size_t) R * K0);
        const size_t plane = (size_t) img.w * img.h;
        for (int r0 = 0; r0 < N; r0 += R) {
            const int rn = std::min(R, N - r0);
            rows_par(rn, [&](int i) {
                const int t = r0 + i, py = t / pw, px = t % pw;
                float* a = &A[(size_t) i * K0];
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky) {
                        const float* src = &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P];
                        float* dst = a + (size_t) c * P * P + (size_t) ky * P;
                        for (int kx = 0; kx < P; ++kx) dst[kx] = src[kx] * 2.0f - 1.0f;
                    }
            });
            linear(patch_w, A.data(), (size_t) K0, rn, x.data() + (size_t) r0 * D, (size_t) D, 0, nullptr, false, scratch);
        }
    }
    rows_par(N, [&](int t) {   // the x table's row of the column, then the y table's of the row: two adds, as the graph has them
        float* r = &x[(size_t) t * D];
        const float* tx = &pos[(size_t) (t % pw) * D];
        const float* ty = &pos[((size_t) T + (size_t) (t / pw)) * D];
        for (int d = 0; d < D; ++d) r[d] += tx[d];
        for (int d = 0; d < D; ++d) r[d] += ty[d];
    });

    // the rotary: pair i of a half head (i, i + HD / 4) turns by p * theta^(-4 i / HD), the angle grown by multiplication as ggml's cache is
    const int half = HD / 2, pairs = half / 2;
    const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) half);
    auto table = [&](int n, std::vector<float>& c, std::vector<float>& s) {
        c.resize((size_t) n * pairs);
        s.resize((size_t) n * pairs);
        for (int p = 0; p < n; ++p) {
            float th = (float) p;
            for (int i = 0; i < pairs; ++i) {
                c[(size_t) p * pairs + i] = std::cos(th);
                s[(size_t) p * pairs + i] = std::sin(th);
                th *= theta_scale;
            }
        }
    };
    std::vector<float> cx, sx, cy, sy;
    table(pw, cx, sx);
    table(ph, cy, sy);
    auto rope = [&](float* h, int px, int py) {
        for (int part = 0; part < 2; ++part) {
            const float* c = part ? &cy[(size_t) py * pairs] : &cx[(size_t) px * pairs];
            const float* s = part ? &sy[(size_t) py * pairs] : &sx[(size_t) px * pairs];
            float* q = h + part * half;
            for (int i = 0; i < pairs; ++i) {
                const float x0 = q[i], x1 = q[i + pairs];
                q[i] = x0 * c[i] - x1 * s[i];
                q[i + pairs] = x0 * s[i] + x1 * c[i];
            }
        }
    };

    const size_t slack = 256, D3 = (size_t) 3 * D;
    std::vector<float> qkv((size_t) N * D3 + slack, 0.0f), ctx((size_t) N * D), hbuf((size_t) R * D), tmp((size_t) R * D);
    std::vector<float> gb((size_t) R * F), ub((size_t) R * F);
    const int Np = (N + 63) / 64 * 64, OD = (HD + 15) / 16 * 16, QB = kQBDefault;
    std::vector<float> KT((size_t) H * HD * Np, 0.0f), VP((size_t) H * N * HD + slack, 0.0f);
    std::vector<std::vector<float>> sscr((size_t) nth, std::vector<float>((size_t) QB * kTK + (size_t) QB * OD + 2 * QB));
    const float eps = g.eps;
    for (const GemmaLayer& L : glayers) {
        for (int r0 = 0; r0 < N; r0 += R) {
            const int rn = std::min(R, N - r0);
            rows_par(rn, [&](int i) { rms_row(&x[(size_t) (r0 + i) * D], &hbuf[(size_t) i * D], D, L.ln1.data(), eps); });
            float* dst = qkv.data() + (size_t) r0 * D3;
            linear(L.q, hbuf.data(), (size_t) D, rn, dst, D3, 0, nullptr, false, scratch);
            linear(L.k, hbuf.data(), (size_t) D, rn, dst + D, D3, 0, nullptr, false, scratch);
            linear(L.v, hbuf.data(), (size_t) D, rn, dst + 2 * D, D3, 0, nullptr, false, scratch);
        }
        rows_par(N, [&](int t) {
            float* row = qkv.data() + (size_t) t * D3;
            for (int h = 0; h < H; ++h) {
                float* q = row + (size_t) h * HD;
                float* k = row + D + (size_t) h * HD;
                float* v = row + 2 * (size_t) D + (size_t) h * HD;
                rms_row(q, q, HD, L.q_norm.data(), eps);
                rope(q, t % pw, t / pw);
                rms_row(k, k, HD, L.k_norm.data(), eps);
                rope(k, t % pw, t / pw);
                rms_row(v, v, HD, nullptr, eps);
            }
        });
        attention(qkv.data(), N, ctx.data(), KT, VP, sscr);   // no scale: the graph's kq_scale is 1
        for (int r0 = 0; r0 < N; r0 += R) {      // x += norm(o(ctx)); x += norm(down(gelu_quick(gate) * up))
            const int rn = std::min(R, N - r0);
            linear(L.o, ctx.data() + (size_t) r0 * D, (size_t) D, rn, tmp.data(), (size_t) D, 0, nullptr, false, scratch);
            rows_par(rn, [&](int i) {
                float* t = &tmp[(size_t) i * D];
                float* xr = &x[(size_t) (r0 + i) * D];
                rms_row(t, t, D, L.attn_post.data(), eps);
                for (int d = 0; d < D; ++d) xr[d] += t[d];
                rms_row(xr, &hbuf[(size_t) i * D], D, L.ln2.data(), eps);
            });
            linear(L.gate, hbuf.data(), (size_t) D, rn, gb.data(), (size_t) F, 0, nullptr, false, scratch);
            linear(L.up, hbuf.data(), (size_t) D, rn, ub.data(), (size_t) F, 0, nullptr, false, scratch);
            rows_par(rn, [&](int i) {
                float* a = &gb[(size_t) i * F];
                const float* u = &ub[(size_t) i * F];
                for (int j = 0; j < F; ++j) a[j] = a[j] * (1.0f / (1.0f + std::exp(-1.702f * a[j]))) * u[j];
            });
            linear(L.down, gb.data(), (size_t) F, rn, tmp.data(), (size_t) D, 0, nullptr, false, scratch);
            rows_par(rn, [&](int i) {
                float* t = &tmp[(size_t) i * D];
                float* xr = &x[(size_t) (r0 + i) * D];
                rms_row(t, t, D, L.ffn_post.data(), eps);
                for (int d = 0; d < D; ++d) xr[d] += t[d];
            });
        }
    }

    const int ox = pw / K, oy = ph / K, M = ox * oy;
    const float sq = std::sqrt((float) D), area = (float) (K * K);
    std::vector<float> pooled((size_t) M * D);
    rows_par(M, [&](int t) {
        const int by = (t / ox) * K, bx = (t % ox) * K;
        float* dst = &pooled[(size_t) t * D];
        for (int d = 0; d < D; ++d) {
            float s = 0.0f;
            for (int ky = 0; ky < K; ++ky)
                for (int kx = 0; kx < K; ++kx) s += x[((size_t) (by + ky) * pw + (size_t) (bx + kx)) * D + d];
            dst[d] = s / area * sq;
        }
        if (g.std_norm)
            for (int d = 0; d < D; ++d) dst[d] = (dst[d] - std_bias[(size_t) d]) * std_scale[(size_t) d];
        rms_row(dst, dst, D, nullptr, eps);
    });
    rows.assign((size_t) M * g.out_dim, 0.0f);
    linear(proj, pooled.data(), (size_t) D, M, rows.data(), (size_t) g.out_dim, 0, nullptr, false, scratch);
    nx = M;
    ny = 1;
    return true;
}

bool VitCpu::Impl::load_minicpmv(const Gguf& gg, std::string& err) {
    {   // the patch embedding, float32 (a small matrix)
        const GgufTensorInfo* t = gg.tensor("v.patch_embd.weight");
        patch_w.kind = 0;
        patch_w.rows = g.width;
        patch_w.cols = 3 * g.patch * g.patch;
        patch_w.f32.resize((size_t) t->nelem());
        if (!gg.read_f32(*t, patch_w.f32.data(), err)) return false;
    }
    auto attn = [&](const std::string& p, MiniLayer& L) {
        return read_f(gg, p + "ln1.weight", L.ln1_w, err) && read_f(gg, p + "ln1.bias", L.ln1_b, err) && read_m(gg, p + "attn_q.weight", L.q, err) &&
               read_f(gg, p + "attn_q.bias", L.q_b, err) && read_m(gg, p + "attn_k.weight", L.k, err) && read_f(gg, p + "attn_k.bias", L.k_b, err) &&
               read_m(gg, p + "attn_v.weight", L.v, err) && read_f(gg, p + "attn_v.bias", L.v_b, err) && read_m(gg, p + "attn_out.weight", L.o, err) &&
               read_f(gg, p + "attn_out.bias", L.o_b, err);
    };
    if (!read_f(gg, "v.patch_embd.bias", patch_b, err) || !read_f(gg, "v.position_embd.weight", pos, err) || !read_f(gg, "v.post_ln.weight", post_w, err) ||
        !read_f(gg, "v.post_ln.bias", post_b, err) || !attn("v.vit_merger.", wm) || !read_f(gg, "v.vit_merger.ds_ln.weight", ds_ln_w, err) ||
        !read_f(gg, "v.vit_merger.ds_ln.bias", ds_ln_b, err) || !read_m(gg, "v.vit_merger.ds_ffn_up.weight", ds_up, err) ||
        !read_f(gg, "v.vit_merger.ds_ffn_up.bias", ds_up_b, err) || !read_m(gg, "v.vit_merger.ds_ffn_down.weight", ds_down, err) ||
        !read_f(gg, "v.vit_merger.ds_ffn_down.bias", ds_down_b, err) || !read_f(gg, "mm.input_norm.weight", mmn_w, err) ||
        !read_f(gg, "mm.input_norm.bias", mmn_b, err) || !read_m(gg, "mm.up.weight", mm_up, err) || !read_f(gg, "mm.up.bias", mm_up_b, err) ||
        !read_m(gg, "mm.down.weight", mm_down, err) || !read_f(gg, "mm.down.bias", mm_down_b, err))
        return false;
    mlayers.assign((size_t) g.layers, MiniLayer{});
    for (int il = 0; il < g.layers; ++il) {
        MiniLayer& L = mlayers[(size_t) il];
        const std::string p = "v.blk." + std::to_string(il) + ".";
        if (!attn(p, L) || !read_f(gg, p + "ln2.weight", L.ln2_w, err) || !read_f(gg, p + "ln2.bias", L.ln2_b, err) || !read_m(gg, p + "ffn_up.weight", L.up, err) ||
            !read_f(gg, p + "ffn_up.bias", L.up_b, err) || !read_m(gg, p + "ffn_down.weight", L.down, err) || !read_f(gg, p + "ffn_down.bias", L.down_b, err))
            return false;
    }
    return true;
}

// MiniCPM-V 4.7's tower on one tile, mtmd's clip_graph_minicpmv4_6 (n_merge 4) step for step.  Patches in reading order with the patch bias,
// plus the position table's row of each patch's bucket (pos_grid buckets a side).  Layers 0 .. insert_layer on every patch; then the window
// merger: attention inside each 2x2 window of patches (as the graph's window-major reorder and block mask give), its residual, and the 2x2
// downsample: the window's four rows side by side (top left, top right, bottom left, bottom right) through a LayerNorm and a tanh-GELU MLP,
// plus their mean.  The other layers on that half-sided grid, the post norm, and the final merger: the same 2x2 gather through a LayerNorm and
// an erf-GELU MLP.  The rows come out in reading order of a grid of (width / 4) x (height / 4) patches.
bool VitCpu::Impl::encode_minicpmv(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
    const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn, T = g.pos_grid, D4 = 4 * D;
    if (HD > 128) { err = "this head size is not handled by the CPU tower"; return false; }
    if (img.w <= 0 || img.h <= 0 || img.w % (P * 4) != 0 || img.h % (P * 4) != 0) { err = "the tile size is not a multiple of patch * 4"; return false; }
    const int pw = img.w / P, ph = img.h / P, N = pw * ph;
    const CpuOps& o = *ops;
    const int nth = team->size();
    std::vector<std::vector<float>> scratch((size_t) nth, std::vector<float>((size_t) kKC * 2 * o.NR));
    const int R = std::min(N, kRowsChunk);
    const float eps = g.eps, kq_scale = 1.0f / std::sqrt((float) HD);

    const int K0 = 3 * P * P;
    std::vector<float> x((size_t) N * D);
    {
        std::vector<float> A((size_t) R * K0);
        const size_t plane = (size_t) img.w * img.h;
        for (int r0 = 0; r0 < N; r0 += R) {
            const int rn = std::min(R, N - r0);
            rows_par(rn, [&](int i) {
                const int t = r0 + i, py = t / pw, px = t % pw;
                float* a = &A[(size_t) i * K0];
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky)
                        std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
            });
            linear(patch_w, A.data(), (size_t) K0, rn, x.data() + (size_t) r0 * D, (size_t) D, 0, patch_b.data(), false, scratch);
        }
    }
    rows_par(N, [&](int t) {   // SigLIP's position buckets: floor(T i / ph) * T + floor(T j / pw)
        const int bi = (int) std::floor((double) T * (t / pw) / ph), bj = (int) std::floor((double) T * (t % pw) / pw);
        const float* pr = &pos[((size_t) bi * T + (size_t) bj) * D];
        float* r = &x[(size_t) t * D];
        for (int d = 0; d < D; ++d) r[d] += pr[d];
    });

    const size_t slack = 256, D3 = (size_t) 3 * D;
    std::vector<float> qkv((size_t) N * D3 + slack, 0.0f), ctx((size_t) N * D), hbuf((size_t) R * D), up((size_t) R * F);
    const int OD = (HD + 15) / 16 * 16, QB = kQBDefault;
    std::vector<float> KT((size_t) H * HD * ((N + 63) / 64 * 64), 0.0f), VP((size_t) H * N * HD + slack, 0.0f);
    std::vector<std::vector<float>> sscr((size_t) nth, std::vector<float>((size_t) QB * kTK + (size_t) QB * OD + 2 * QB));
    // q, k, v (with the attention's scale on q) of n rows of x after the LayerNorm of L
    auto project = [&](const MiniLayer& L, int n) {
        for (int r0 = 0; r0 < n; r0 += R) {
            const int rn = std::min(R, n - r0);
            rows_par(rn, [&](int i) { o.layernorm(&x[(size_t) (r0 + i) * D], &hbuf[(size_t) i * D], D, L.ln1_w.data(), L.ln1_b.data(), eps); });
            float* dst = qkv.data() + (size_t) r0 * D3;
            linear(L.q, hbuf.data(), (size_t) D, rn, dst, D3, 0, L.q_b.data(), false, scratch);
            linear(L.k, hbuf.data(), (size_t) D, rn, dst + D, D3, 0, L.k_b.data(), false, scratch);
            linear(L.v, hbuf.data(), (size_t) D, rn, dst + 2 * D, D3, 0, L.v_b.data(), false, scratch);
        }
        rows_par(n, [&](int t) {
            float* q = qkv.data() + (size_t) t * D3;
            for (int i = 0; i < D; ++i) q[i] *= kq_scale;
        });
    };
    auto layer = [&](const MiniLayer& L, int n) {
        project(L, n);
        attention(qkv.data(), n, ctx.data(), KT, VP, sscr);
        for (int r0 = 0; r0 < n; r0 += R) {      // x += o(ctx); x += down(gelu(up(LN2 x)))
            const int rn = std::min(R, n - r0);
            linear(L.o, ctx.data() + (size_t) r0 * D, (size_t) D, rn, x.data() + (size_t) r0 * D, (size_t) D, 2, L.o_b.data(), false, scratch);
            rows_par(rn, [&](int i) { o.layernorm(&x[(size_t) (r0 + i) * D], &hbuf[(size_t) i * D], D, L.ln2_w.data(), L.ln2_b.data(), eps); });
            linear(L.up, hbuf.data(), (size_t) D, rn, up.data(), (size_t) F, 0, L.up_b.data(), true, scratch);
            linear(L.down, up.data(), (size_t) F, rn, x.data() + (size_t) r0 * D, (size_t) D, 2, L.down_b.data(), false, scratch);
        }
    };
    // the four rows of each 2x2 window of a gw x gh grid side by side, in the gather order of the graph's ds indices
    auto gather = [&](const float* src, int gw, int gh, std::vector<float>& cat) {
        const int qw = gw / 2, M = qw * (gh / 2);
        cat.resize((size_t) M * D4);
        rows_par(M, [&](int t) {
            const int i = t / qw, j = t % qw;
            const int at[4] = {(2 * i) * gw + 2 * j, (2 * i) * gw + 2 * j + 1, (2 * i + 1) * gw + 2 * j, (2 * i + 1) * gw + 2 * j + 1};
            for (int s = 0; s < 4; ++s) std::memcpy(&cat[(size_t) t * D4 + (size_t) s * D], src + (size_t) at[s] * D, sizeof(float) * (size_t) D);
        });
    };
    // LN(cat) through up (+bias), the activation and down (+bias) into out [M][od], in chunks of rows (the up product is wide)
    auto mlp = [&](const std::vector<float>& cat, int M, const std::vector<float>& lw, const std::vector<float>& lb, const Mat& wu, const std::vector<float>& bu,
                   const Mat& wd, const std::vector<float>& bd, bool erf, float* out, int od) {
        const int RC = std::max(1, std::min(M, (int) ((size_t) 64 << 20) / (int) (sizeof(float) * (size_t) wu.rows)));
        std::vector<float> nb((size_t) RC * D4), ub((size_t) RC * wu.rows);
        for (int r0 = 0; r0 < M; r0 += RC) {
            const int rn = std::min(RC, M - r0);
            rows_par(rn, [&](int i) { o.layernorm(&cat[(size_t) (r0 + i) * D4], &nb[(size_t) i * D4], D4, lw.data(), lb.data(), eps); });
            linear(wu, nb.data(), (size_t) D4, rn, ub.data(), (size_t) wu.rows, 0, bu.data(), !erf, scratch);
            if (erf)
                rows_par(rn, [&](int i) { gelu_erf_rows(&ub[(size_t) i * wu.rows], (size_t) wu.rows, 1, wu.rows); });
            linear(wd, ub.data(), (size_t) wu.rows, rn, out + (size_t) r0 * od, (size_t) od, 0, bd.data(), false, scratch);
        }
    };

    for (int il = 0; il <= g.insert_layer; ++il) layer(mlayers[(size_t) il], N);

    {   // the window merger: attention inside each 2x2 window of patches, then its residual
        project(wm, N);
        const int hw = pw / 2, wins = hw * (ph / 2);
        rows_par(wins, [&](int w) {
            const int i = w / hw, j = w % hw;
            const int at[4] = {(2 * i) * pw + 2 * j, (2 * i) * pw + 2 * j + 1, (2 * i + 1) * pw + 2 * j, (2 * i + 1) * pw + 2 * j + 1};
            for (int h = 0; h < H; ++h)
                for (int a = 0; a < 4; ++a) {
                    const float* q = qkv.data() + (size_t) at[a] * D3 + (size_t) h * HD;
                    float s[4], mx = -3.0e38f, sum = 0.0f;
                    for (int b = 0; b < 4; ++b) {
                        const float* k = qkv.data() + (size_t) at[b] * D3 + D + (size_t) h * HD;
                        float acc = 0.0f;
                        for (int d = 0; d < HD; ++d) acc += q[d] * k[d];
                        s[b] = acc;
                        mx = std::max(mx, acc);
                    }
                    for (int b = 0; b < 4; ++b) { s[b] = std::exp(s[b] - mx); sum += s[b]; }
                    float* dst = ctx.data() + (size_t) at[a] * D + (size_t) h * HD;
                    for (int d = 0; d < HD; ++d) {
                        float acc = 0.0f;
                        for (int b = 0; b < 4; ++b) acc += s[b] * qkv[(size_t) at[b] * D3 + 2 * (size_t) D + (size_t) h * HD + d];
                        dst[d] = acc / sum;
                    }
                }
        });
        for (int r0 = 0; r0 < N; r0 += R) {
            const int rn = std::min(R, N - r0);
            linear(wm.o, ctx.data() + (size_t) r0 * D, (size_t) D, rn, x.data() + (size_t) r0 * D, (size_t) D, 2, wm.o_b.data(), false, scratch);
        }
    }
    const int hw = pw / 2, hh = ph / 2, N2 = hw * hh;
    {   // the 2x2 downsample: MLP(LN(the window's four rows)) + their mean, on the half-sided grid
        std::vector<float> cat;
        gather(x.data(), pw, ph, cat);
        std::vector<float> x2((size_t) N2 * D);
        mlp(cat, N2, ds_ln_w, ds_ln_b, ds_up, ds_up_b, ds_down, ds_down_b, false, x2.data(), D);
        rows_par(N2, [&](int t) {
            const float* c = &cat[(size_t) t * D4];
            float* r = &x2[(size_t) t * D];
            for (int d = 0; d < D; ++d) r[d] += (((c[d] + c[D + d]) + c[2 * D + d]) + c[3 * D + d]) * 0.25f;
        });
        x.swap(x2);
        std::fill(KT.begin(), KT.end(), 0.0f);   // the quarter grid's rows of K are shorter: no stale keys past them
    }
    for (int il = g.insert_layer + 1; il < g.layers; ++il) layer(mlayers[(size_t) il], N2);

    std::vector<float>& hpost = ctx;       // the post norm's output in the context's place
    rows_par(N2, [&](int t) { o.layernorm(&x[(size_t) t * D], &hpost[(size_t) t * D], D, post_w.data(), post_b.data(), eps); });
    std::vector<float> cat;
    gather(hpost.data(), hw, hh, cat);
    const int M = (hw / 2) * (hh / 2);
    rows.assign((size_t) M * g.out_dim, 0.0f);
    mlp(cat, M, mmn_w, mmn_b, mm_up, mm_up_b, mm_down, mm_down_b, true, rows.data(), g.out_dim);
    nx = hw / 2;
    ny = hh / 2;
    return true;
}

}  // namespace strata_vit

namespace strata_vit {

bool VitCpu::Impl::load_inkling(const Gguf& gg, std::string& err) {
    {   // the patch embedding, float32 (a small matrix), no bias
        const GgufTensorInfo* t = gg.tensor("v.patch_embd.weight");
        if (!t) { err = "the mmproj file lacks v.patch_embd.weight"; return false; }
        patch_w.kind = 0;
        patch_w.rows = g.width;
        patch_w.cols = 3 * g.patch * g.patch;
        patch_w.f32.resize((size_t) t->nelem());
        if (!gg.read_f32(*t, patch_w.f32.data(), err)) return false;
    }
    std::vector<float> cls_pos;
    if (!read_f(gg, "v.position_embd.weight", pos, err) || !read_f(gg, "v.class_embd", cls, err) ||
        !read_f(gg, "v.class_position_embd.weight", cls_pos, err) || !read_f(gg, "v.pre_ln.weight", pre_w, err) ||
        !read_f(gg, "v.pre_ln.bias", pre_b, err) || !read_m(gg, "mm.1.weight", mm0, err) || !read_m(gg, "mm.2.weight", mm2, err))
        return false;
    for (size_t d = 0; d < cls.size(); ++d) cls[d] += cls_pos[d];   // ggml_add(class_embedding, class_position_embedding)
    if (g.post_ln && (!read_f(gg, "v.post_ln.weight", post_w, err) || !read_f(gg, "v.post_ln.bias", post_b, err))) return false;
    if (g.mm_bias && (!read_f(gg, "mm.1.bias", mm0_b, err) || !read_f(gg, "mm.2.bias", mm2_b, err))) return false;
    ilayers.assign((size_t) g.layers, InkLayer{});
    for (int il = 0; il < g.layers; ++il) {
        InkLayer& L = ilayers[(size_t) il];
        const std::string p = "v.blk." + std::to_string(il) + ".";
        if (!read_f(gg, p + "ln1.weight", L.ln1_w, err) || !read_f(gg, p + "ln1.bias", L.ln1_b, err) ||
            !read_f(gg, p + "ln2.weight", L.ln2_w, err) || !read_f(gg, p + "ln2.bias", L.ln2_b, err) ||
            !read_m(gg, p + "attn_q.weight", L.q, err) || !read_f(gg, p + "attn_q.bias", L.q_b, err) ||
            !read_m(gg, p + "attn_k.weight", L.k, err) || !read_f(gg, p + "attn_k.bias", L.k_b, err) ||
            !read_m(gg, p + "attn_v.weight", L.v, err) || !read_f(gg, p + "attn_v.bias", L.v_b, err) ||
            !read_m(gg, p + "attn_out.weight", L.o, err) || !read_f(gg, p + "attn_out.bias", L.o_b, err) ||
            !read_f(gg, p + "ls1.weight", L.ls1, err) || !read_f(gg, p + "ls2.weight", L.ls2, err) ||
            !read_m(gg, p + "ffn_up.weight", L.up, err) || !read_f(gg, p + "ffn_up.bias", L.up_b, err) ||
            !read_m(gg, p + "ffn_down.weight", L.down, err) || !read_f(gg, p + "ffn_down.bias", L.down_b, err))
            return false;
    }
    return true;
}

// Inkling 2 Mini's tower, llama.cpp's mtmd_graph_inkling2v step for step.  Patches in reading order (row t + 1 is the patch at column t % pw,
// row t / pw), the position table resized to pw x ph as ggml_interpolate's bilinear does it (half-pixel centres, edges clamped) and added; the
// CLS row in front; the first LayerNorm; then per layer LayerNorm, q / k / v, the rotary of q and k (the first half of a head by the column,
// the second by the row, adjacent pairs 2 i, 2 i + 1 turned by p * theta^(-2 i / half), positions from 1, the CLS row at 0), attention scaled
// by 1 / sqrt(head), o, times ls1, the residual; LayerNorm, up, the erf GELU, down, times ls2, the residual.  Then the CLS row is dropped, each
// 2x2 square of patches is one row of four (the left top, right top, left bottom, right bottom patch, as build_patch_merge_permute orders
// them), and mm.1, the erf GELU and mm.2 make the output rows, in reading order of the merged grid.
bool VitCpu::Impl::encode_inkling(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
    const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn, G = g.pos_grid, K = g.merge;
    if (HD > 128 || HD % 4 != 0) { err = "this head size is not handled by the CPU tower"; return false; }
    if (img.w <= 0 || img.h <= 0 || img.w % (P * K) != 0 || img.h % (P * K) != 0) { err = "the picture size is not a multiple of patch * merge"; return false; }
    const int pw = img.w / P, ph = img.h / P, N = pw * ph, N1 = N + 1;
    const CpuOps& o = *ops;
    const int nth = team->size();
    std::vector<std::vector<float>> scratch((size_t) nth, std::vector<float>((size_t) kKC * 2 * o.NR));
    const int R = std::min(N1, kRowsChunk);

    // the position table at the picture's grid (ggml_interpolate, GGML_SCALE_MODE_BILINEAR)
    std::vector<float> pt;
    const float* posg = pos.data();
    if (pw != G || ph != G) {
        pt.resize((size_t) N * D);
        const float sf0 = (float) pw / G, sf1 = (float) ph / G;
        rows_par(N, [&](int t) {
            const int i0 = t % pw, i1 = t / pw;
            const float y = ((float) i1 + 0.5f) / sf1 - 0.5f, x = ((float) i0 + 0.5f) / sf0 - 0.5f;
            int y0 = (int) std::floor(y), x0 = (int) std::floor(x), y1 = y0 + 1, x1 = x0 + 1;
            y0 = std::max(0, std::min(y0, G - 1));
            y1 = std::max(0, std::min(y1, G - 1));
            x0 = std::max(0, std::min(x0, G - 1));
            x1 = std::max(0, std::min(x1, G - 1));
            const float dy = std::max(0.0f, std::min(y - (float) y0, 1.0f)), dx = std::max(0.0f, std::min(x - (float) x0, 1.0f));
            const float* a = &pos[((size_t) y0 * G + x0) * D];
            const float* b = &pos[((size_t) y0 * G + x1) * D];
            const float* c = &pos[((size_t) y1 * G + x0) * D];
            const float* d = &pos[((size_t) y1 * G + x1) * D];
            float* dst = &pt[(size_t) t * D];
            for (int e = 0; e < D; ++e) dst[e] = a[e] * (1 - dx) * (1 - dy) + b[e] * dx * (1 - dy) + c[e] * (1 - dx) * dy + d[e] * dx * dy;
        });
        posg = pt.data();
    }

    // the patches (rows 1..N) and the CLS row (row 0)
    const int K0 = 3 * P * P;
    std::vector<float> x((size_t) N1 * D);
    std::memcpy(x.data(), cls.data(), sizeof(float) * (size_t) D);
    {
        const int RP = std::min(N, kRowsChunk);
        std::vector<float> A((size_t) RP * K0);
        const size_t plane = (size_t) img.w * img.h;
        for (int r0 = 0; r0 < N; r0 += RP) {
            const int rn = std::min(RP, N - r0);
            rows_par(rn, [&](int i) {
                const int t = r0 + i, py = t / pw, px = t % pw;
                float* a = &A[(size_t) i * K0];
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky)
                        std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P],
                                    (size_t) P * sizeof(float));
            });
            linear(patch_w, A.data(), (size_t) K0, rn, x.data() + (size_t) (1 + r0) * D, (size_t) D, 0, nullptr, false, scratch);
        }
        rows_par(N, [&](int t) {
            float* r = &x[(size_t) (1 + t) * D];
            const float* p = posg + (size_t) t * D;
            for (int d = 0; d < D; ++d) r[d] += p[d];
        });
    }
    rows_par(N1, [&](int t) {   // the first norm
        std::vector<float> row(x.begin() + (ptrdiff_t) t * D, x.begin() + (ptrdiff_t) (t + 1) * D);
        o.layernorm(row.data(), &x[(size_t) t * D], D, pre_w.data(), pre_b.data(), g.eps);
    });

    // the rotary: pair i of a half head (elements 2 i and 2 i + 1 of the half) turns by p * theta^(-2 i / half), grown by multiplication as
    // ggml's rope cache is; the first half by the column + 1, the second by the row + 1, both 0 for the CLS row
    const int half = HD / 2, pairs = half / 2;
    const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) half);
    auto table = [&](int n, std::vector<float>& c, std::vector<float>& s) {
        c.resize((size_t) n * pairs);
        s.resize((size_t) n * pairs);
        for (int p = 0; p < n; ++p) {
            float th = (float) p;
            for (int i = 0; i < pairs; ++i) {
                c[(size_t) p * pairs + i] = std::cos(th);
                s[(size_t) p * pairs + i] = std::sin(th);
                th *= theta_scale;
            }
        }
    };
    std::vector<float> cx, sx, cy, sy;
    table(pw + 1, cx, sx);
    table(ph + 1, cy, sy);
    auto rope = [&](float* h, int px, int py) {
        for (int part = 0; part < 2; ++part) {
            const float* c = part ? &cy[(size_t) py * pairs] : &cx[(size_t) px * pairs];
            const float* s = part ? &sy[(size_t) py * pairs] : &sx[(size_t) px * pairs];
            float* q = h + part * half;
            for (int i = 0; i < pairs; ++i) {
                const float x0 = q[2 * i], x1 = q[2 * i + 1];
                q[2 * i] = x0 * c[i] - x1 * s[i];
                q[2 * i + 1] = x0 * s[i] + x1 * c[i];
            }
        }
    };
    const float kq_scale = 1.0f / std::sqrt((float) HD);

    const size_t slack = 256, D3 = (size_t) 3 * D;
    std::vector<float> qkv((size_t) N1 * D3 + slack, 0.0f), ctx((size_t) N1 * D), hbuf((size_t) R * D), tmp((size_t) R * D), up((size_t) R * F);
    const int Np = (N1 + 63) / 64 * 64, OD = (HD + 15) / 16 * 16, QB = kQBDefault;
    std::vector<float> KT((size_t) H * HD * Np, 0.0f), VP((size_t) H * N1 * HD + slack, 0.0f);
    std::vector<std::vector<float>> sscr((size_t) nth, std::vector<float>((size_t) QB * kTK + (size_t) QB * OD + 2 * QB));
    const float eps = g.eps;
    for (const InkLayer& L : ilayers) {
        for (int r0 = 0; r0 < N1; r0 += R) {
            const int rn = std::min(R, N1 - r0);
            rows_par(rn, [&](int i) { o.layernorm(&x[(size_t) (r0 + i) * D], &hbuf[(size_t) i * D], D, L.ln1_w.data(), L.ln1_b.data(), eps); });
            float* dst = qkv.data() + (size_t) r0 * D3;
            linear(L.q, hbuf.data(), (size_t) D, rn, dst, D3, 0, L.q_b.data(), false, scratch);
            linear(L.k, hbuf.data(), (size_t) D, rn, dst + D, D3, 0, L.k_b.data(), false, scratch);
            linear(L.v, hbuf.data(), (size_t) D, rn, dst + 2 * D, D3, 0, L.v_b.data(), false, scratch);
        }
        rows_par(N1, [&](int t) {
            float* row = qkv.data() + (size_t) t * D3;
            const int px = t == 0 ? 0 : (t - 1) % pw + 1, py = t == 0 ? 0 : (t - 1) / pw + 1;
            for (int h = 0; h < H; ++h) {
                rope(row + (size_t) h * HD, px, py);
                rope(row + D + (size_t) h * HD, px, py);
            }
            for (int i = 0; i < D; ++i) row[i] *= kq_scale;
        });
        attention(qkv.data(), N1, ctx.data(), KT, VP, sscr);
        for (int r0 = 0; r0 < N1; r0 += R) {      // x += ls1 * o(ctx); x += ls2 * down(gelu_erf(up(LN2 x)))
            const int rn = std::min(R, N1 - r0);
            linear(L.o, ctx.data() + (size_t) r0 * D, (size_t) D, rn, tmp.data(), (size_t) D, 0, L.o_b.data(), false, scratch);
            rows_par(rn, [&](int i) {
                const float* t = &tmp[(size_t) i * D];
                float* xr = &x[(size_t) (r0 + i) * D];
                for (int d = 0; d < D; ++d) xr[d] += t[d] * L.ls1[(size_t) d];
                o.layernorm(xr, &hbuf[(size_t) i * D], D, L.ln2_w.data(), L.ln2_b.data(), eps);
            });
            linear(L.up, hbuf.data(), (size_t) D, rn, up.data(), (size_t) F, 0, L.up_b.data(), false, scratch);
            rows_par(rn, [&](int i) { gelu_erf_rows(&up[(size_t) i * F], (size_t) F, 1, F); });
            linear(L.down, up.data(), (size_t) F, rn, tmp.data(), (size_t) D, 0, L.down_b.data(), false, scratch);
            rows_par(rn, [&](int i) {
                const float* t = &tmp[(size_t) i * D];
                float* xr = &x[(size_t) (r0 + i) * D];
                for (int d = 0; d < D; ++d) xr[d] += t[d] * L.ls2[(size_t) d];
            });
        }
    }
    if (g.post_ln)
        rows_par(N1, [&](int t) {
            std::vector<float> row(x.begin() + (ptrdiff_t) t * D, x.begin() + (ptrdiff_t) (t + 1) * D);
            o.layernorm(row.data(), &x[(size_t) t * D], D, post_w.data(), post_b.data(), eps);
        });

    // the merge: merged row (bx, by) is the patches (2 bx, 2 by), (2 bx + 1, 2 by), (2 bx, 2 by + 1), (2 bx + 1, 2 by + 1) side by side (K = 2)
    const int ox = pw / K, oy = ph / K, M = ox * oy, Kin = g.merger_in, Hid = g.merger_hidden, O = g.out_dim;
    std::vector<float> merged((size_t) M * Kin);
    rows_par(M, [&](int t) {
        const int bx = t % ox, by = t / ox;
        float* dst = &merged[(size_t) t * Kin];
        for (int ky = 0; ky < K; ++ky)
            for (int kx = 0; kx < K; ++kx) {
                const int p = (by * K + ky) * pw + (bx * K + kx);
                std::memcpy(dst + (size_t) (ky * K + kx) * D, &x[(size_t) (1 + p) * D], sizeof(float) * (size_t) D);
            }
    });
    rows.assign((size_t) M * O, 0.0f);
    const int RM = std::max(1, std::min(M, kRowsChunk / 4));
    std::vector<float> m1((size_t) RM * Hid);
    for (int r0 = 0; r0 < M; r0 += RM) {
        const int rn = std::min(RM, M - r0);
        linear(mm0, merged.data() + (size_t) r0 * Kin, (size_t) Kin, rn, m1.data(), (size_t) Hid, 0, g.mm_bias ? mm0_b.data() : nullptr, false, scratch);
        rows_par(rn, [&](int i) { gelu_erf_rows(&m1[(size_t) i * Hid], (size_t) Hid, 1, Hid); });
        linear(mm2, m1.data(), (size_t) Hid, rn, rows.data() + (size_t) r0 * O, (size_t) O, 0, g.mm_bias ? mm2_b.data() : nullptr, false, scratch);
    }
    nx = ox;
    ny = oy;
    return true;
}

}  // namespace strata_vit
