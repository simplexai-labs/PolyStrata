// metal/src/poly/inkling.cpp - Inkling 2's kernels on Metal (inkling2_kernels.hpp, inkling.metal), and the shared
// entry points only its layers call here: rms_norm_many(), gemv_set() and situ_rows().  A row quantized in the launch
// that makes it (inkling2.cu's fusions) has the bytes native_quantize_q8_1() gives it here, and a fused norm is
// poly_norm's arithmetic, so that native_add_rms_norm_q8_1() and the launches it stands for agree.
#include "inkling2_kernels.hpp"

#include "launch.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

struct Args {
    const void* p[72]{};
    int64_t pos = 0;
    uint64_t stride = 0;
    int n[24]{};
    float f[16]{};
};
static_assert(sizeof(Args) == 752);

const std::string& source() {
    static const std::string s = metal_detail::read_source(STRATA_POLY_INKLING_SOURCE);
    return s;
}

template <class A>
void run(const char* name, const A& a, void* stream, gpu::metal::Grid groups, uint32_t threads, size_t shared = 0) {
    if (!groups.x || !groups.y || !groups.z) return;
    metal_detail::dispatch("poly_inkling", source(), name, a, stream, groups, threads, {}, shared);
}

uint32_t blocks(size_t n, uint32_t per) { return uint32_t((n + per - 1) / per); }

[[noreturn]] void unported(const char* what) { throw std::invalid_argument(std::string("Metal has no ") + what); }

void elem(Args& a, int kind, size_t n, void* stream) {
    if (!n) return;
    a.n[0] = kind;
    a.stride = n;
    run("ink_elem", a, stream, {blocks(n, 256), 1, 1}, 256);
}

}  // namespace

// ---- shared entry points ----

void rms_norm_many(const float* const* x, const float* const* w, float* const* y, int count, int n, int rows, float eps,
                   void* stream) {
    for (int j = 0; j < count; ++j) rms_norm(x[j], w[j], y[j], n, rows, eps, stream);
}

namespace {
void gemv_many(const void* const* w, const float* const* x, float* const* y, const int* n_out, int count, int n_in,
               int rows, bool bf16, void* stream) {
    if (count < 1 || count > 6) throw std::invalid_argument("gemv_set: 1 to 6 matrices");
    if (rows <= 0) return;
    Args a;
    int total = 0;
    for (int j = 0; j < count; ++j) {
        a.p[j] = w[j], a.p[6 + j] = x[j], a.p[12 + j] = y[j], a.n[j] = n_out[j];
        total += n_out[j];
    }
    a.n[7] = n_in, a.n[8] = rows, a.n[9] = bf16;
    run("ink_gemv_set", a, stream, {uint32_t(total), uint32_t(rows), 1}, n_in < 128 ? 32 : sum_threads(n_in) / 4);
}
}  // namespace

void gemv_set(const float* const* w, const float* const* x, float* const* y, const int* n_out, int count, int n_in,
              int rows, void* stream) {
    const void* m[6];
    std::copy(w, w + count, m);
    gemv_many(m, x, y, n_out, count, n_in, rows, false, stream);
}

void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream, void* q8,
               const ExpertScales* scales) {
    if (rows <= 0) return;
    if (q8 && n % 32 != 0) throw std::invalid_argument("situ_rows: quantized rows of a multiple of 32");
    struct SituArgs {
        const void* p[6]{};
        int n = 0, rows = 0;
        float beta = 0, lb = 0;
        int e[32]{};
    } a;
    static_assert(sizeof(SituArgs) == 192);
    a.p[0] = gate_up;
    a.p[1] = out;
    if (scales) {
        a.p[2] = scales->gate;
        a.p[3] = scales->up;
        a.p[4] = scales->ids;
        if ((scales->gate || scales->up) && !scales->ids && rows > 32)
            throw std::invalid_argument("situ_rows: more than 32 rows of experts named on the host");
        std::copy(scales->e, scales->e + 32, a.e);
    }
    a.p[5] = q8;   // the rows back to back, as one vector
    a.n = n;
    a.rows = rows;
    a.beta = beta;
    a.lb = lb;
    run("ink_situ", a, stream, {blocks(n, 256), uint32_t(rows), 1}, 256);
}

// ---- the mixes ----

namespace {
void mix(const InkStates& states, const float* const* query, int n_query, float* const* out, int jobs, int n_embd,
         int rows, const float* norm_w, float* norm_out, float eps, void* stream, void* q8 = nullptr) {
    if (states.n < 1 || states.n > InkStates::kMax || n_query < 1 || n_query > 4 || jobs < 1 || jobs > 12)
        throw std::invalid_argument("ink_mix: 1 to 6 states, 1 to 4 queries, 1 to 12 jobs");
    if (rows <= 0) return;
    if (n_embd > 4 * 1024) throw std::invalid_argument("ink_mix: rows of up to 4,096 values");
    Args a;
    for (int s = 0; s < states.n; ++s) a.p[s] = states.p[s];
    for (int k = 0; k < jobs; ++k) {
        a.p[6 + k] = query[k];
        for (int j = 0; j < n_query; ++j) a.p[18 + 4 * k + j] = out[k * n_query + j];
    }
    a.p[66] = norm_w;
    a.p[67] = norm_out;
    a.p[68] = q8;
    a.n[0] = states.n;
    a.n[1] = n_query;
    a.n[2] = n_embd;
    a.f[0] = eps;
    run("ink_mix", a, stream, {uint32_t(rows), uint32_t(jobs), 1}, 1024);
}
}  // namespace

void ink_mix(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
             void* stream) {
    mix(states, &query, n_query, out, 1, n_embd, rows, nullptr, nullptr, 0.f, stream);
}

void ink_mix_many(const InkStates& states, const float* const* query, int n_query, float* const* out, int jobs,
                  int n_embd, void* stream) {
    mix(states, query, n_query, out, jobs, n_embd, 1, nullptr, nullptr, 0.f, stream);
}

void ink_mix_norm(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
                  const float* norm_w, float* norm_out, float eps, void* stream, void* q8, bool ffn) {
    if ((q8 || ffn) && (rows != 1 || !norm_out || (ffn && (!q8 || !norm_w))))
        throw std::invalid_argument("ink_mix_norm: a quantized norm is one row's");
    mix(states, &query, n_query, out, 1, n_embd, rows, norm_w, norm_out, eps, stream, q8);
}

void ink_norm_many_q8(const float* const* x, const float* const* w, float* const* y, void* const* q8, int count, int n,
                      float eps, void* stream) {
    if (count < 1 || count > 3 || n % 32 != 0) throw std::invalid_argument("ink_norm_many_q8: 1 to 3 rows of 32k");
    Args a;
    for (int j = 0; j < count; ++j) a.p[j] = x[j], a.p[3 + j] = w[j], a.p[6 + j] = y[j], a.p[9 + j] = q8[j];
    a.n[0] = n;
    a.f[0] = eps;
    run("ink_norm_q8", a, stream, {1, uint32_t(count), 1}, sum_threads(n));
}

// ---- the loop controllers ----

void ink_cat_norm_set(const float* const* cur, const float* const* msg, float* const* out, int count, int n_embd,
                      int rows, void* stream) {
    if (count < 1 || count > 3) throw std::invalid_argument("ink_cat_norm_set: 1 to 3 pairs");
    if (rows <= 0) return;
    Args a;
    for (int j = 0; j < count; ++j) {
        a.p[j] = cur[j];
        a.p[3 + j] = msg[j];
        a.p[6 + j] = out[j];
    }
    a.n[0] = n_embd;
    run("ink_cat_norm", a, stream, {uint32_t(rows), uint32_t(count), 1}, 256);
}

void ink_cat_norm(const float* cur, const float* msg, float* out, int n_embd, int rows, void* stream) {
    ink_cat_norm_set(&cur, &msg, &out, 1, n_embd, rows, stream);
}

bool ink_ctrl_down(const float* const* cur, const float* const* msg, const void* const* down, float* const* ch,
                   int count, int n_embd, int rank, int rows, void* stream, bool bf16) {
    if (count < 1 || count > 3) throw std::invalid_argument("ink_ctrl_down: 1 to 3 controllers");
    static const int per_group = [] {   // STRATA_INK_DOWN_ROWS: outputs a threadgroup (1, 2, 4 or 8)
        const char* v = std::getenv("STRATA_INK_DOWN_ROWS");
        const int n = v ? std::atoi(v) : 2;
        return n == 1 || n == 2 || n == 4 || n == 8 ? n : 2;
    }();
    if (rank % per_group != 0 || 2 * n_embd > 24 * 256) return false;   // 24 values a thread
    if (rows <= 0) return true;
    Args a;
    for (int j = 0; j < count; ++j) a.p[j] = cur[j], a.p[3 + j] = msg[j], a.p[6 + j] = down[j], a.p[9 + j] = ch[j];
    a.n[0] = n_embd, a.n[1] = rank, a.n[2] = bf16, a.n[3] = per_group;
    run("ink_ctrl_down", a, stream, {uint32_t(count * rank / per_group), uint32_t(rows), 1}, 256);
    return true;
}

void ink_gate_set(const float* const* cur, const float* const* msg, const float* const* gain,
                  const float* const* forget, float* const* out, int count, size_t n, void* stream) {
    if (count < 1 || count > 3) throw std::invalid_argument("ink_gate_set: 1 to 3 controllers");
    if (!n) return;
    Args a;
    for (int j = 0; j < count; ++j) {
        a.p[j] = cur[j];
        a.p[3 + j] = msg[j];
        a.p[6 + j] = gain[j];
        a.p[9 + j] = forget[j];
        a.p[12 + j] = out[j];
    }
    a.n[0] = count;
    a.stride = n;
    run("ink_gate", a, stream, {blocks(n, 256), 1, 1}, 256);
}

void ink_gate(const float* cur, const float* msg, const float* gain, const float* forget, float* out, size_t n,
              void* stream) {
    ink_gate_set(&cur, &msg, &gain, &forget, &out, 1, n, stream);
}

void ink_gate_norm_q8(const float* cur, const float* msg, const float* gain, const float* forget, float* out,
                      const float* w, float* y, void* q8, int n, float eps, void* stream) {
    if (n % 32 != 0) throw std::invalid_argument("ink_gate_norm_q8: rows of 32k");
    Args a;
    a.p[0] = cur, a.p[3] = w, a.p[6] = y, a.p[9] = q8, a.p[12] = msg, a.p[13] = gain, a.p[14] = forget, a.p[15] = out;
    a.n[0] = n, a.n[1] = 1;
    a.f[0] = eps;
    run("ink_norm_q8", a, stream, {1, 1, 1}, sum_threads(n));
}

void ink_control3_gate(const float* const* cur, const float* const* msg, const float* const* ch,
                       const void* const* gain_w, const void* const* forget_w, float* const* out, int n, int rank,
                       void* stream, int count, bool bf16) {
    if (rank >= 128) throw std::invalid_argument("ink_control3_gate: gemv_set() takes products this wide otherwise");
    if (count < 1 || count > 3) throw std::invalid_argument("ink_control3_gate: 1 to 3 controllers");
    Args a;
    for (int j = 0; j < count; ++j) {
        a.p[j] = cur[j];
        a.p[3 + j] = msg[j];
        a.p[6 + j] = ch[j];
        a.p[9 + j] = gain_w[j];
        a.p[12 + j] = forget_w[j];
        a.p[15 + j] = out[j];
    }
    a.n[0] = n;
    a.n[1] = rank;
    a.n[2] = count;
    a.n[3] = bf16;
    run("ink_control3_gate", a, stream, {blocks((size_t) n, 8), 1, 1}, 256);
}

// ---- the convolutions ----

namespace {
float conv_norm(const float* dyn, int taps) { return 1.0f / std::sqrt((float) ((dyn ? 2 : 1) * taps + 1)); }
}  // namespace

void ink_conv_few(const InkConv& a, const InkConv& b, int channels, int heads, int taps, int rows, void* stream,
                  size_t snap_stride, unsigned snap_bits) {
    if (taps < 2 || taps > 8 || rows < 1 || rows > kConvFew)
        throw std::invalid_argument("ink_conv_few: 2 to 8 taps, 1 to 8 rows");
    Args g;
    const InkConv* jobs[2] = {&a, &b};
    for (int j = 0; j < 2; ++j) {
        const InkConv& c = *jobs[j];
        const void* p[8] = {c.x, c.state, c.w, c.dyn, c.out, snap_bits ? c.snap : nullptr, c.acc, c.keep};
        std::copy(p, p + 8, g.p + 8 * j);
        g.n[4 + j] = c.mode;
        g.f[j] = c.a;
        g.f[2 + j] = conv_norm(c.dyn, taps);
    }
    g.n[0] = channels;
    g.n[1] = heads;
    g.n[2] = taps;
    g.n[3] = rows;
    g.n[6] = (int) snap_bits;
    g.stride = snap_stride;
    run("ink_conv_few", g, stream, {blocks((size_t) channels, 256), b.x ? 2u : 1u, 1}, 256);
}

void ink_conv(const float* x, float* state, const float* w, const float* dyn, float* out, int channels, int heads,
              int taps, int rows, void* stream, float* snap, size_t snap_stride, unsigned snap_bits) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv: 2 to 8 taps");
    if (rows <= 0) return;
    if (rows <= kConvFew) {
        InkConv j;
        j.x = x, j.state = state, j.w = w, j.dyn = dyn, j.out = out, j.snap = snap_bits ? snap : nullptr;
        ink_conv_few(j, InkConv{}, channels, heads, taps, rows, stream, snap_stride, snap_bits);
        return;
    }
    Args a;
    a.p[0] = x, a.p[1] = state, a.p[2] = w, a.p[3] = dyn, a.p[4] = out;
    a.n[0] = channels, a.n[1] = heads, a.n[2] = taps, a.n[3] = rows;
    a.f[0] = conv_norm(dyn, taps);
    run("ink_conv_rows", a, stream, {blocks((size_t) rows * channels, 256), 1, 1}, 256);
    Args t;
    t.p[0] = x, t.p[1] = state, t.p[2] = snap_bits ? snap : nullptr;
    t.n[0] = channels, t.n[1] = taps, t.n[2] = rows, t.n[3] = (int) snap_bits;
    t.stride = snap_stride;
    run("ink_conv_tail", t, stream, {blocks((size_t) channels, 256), 1, 1}, 256);
}

void ink_conv_add(const float* x, float* state, const float* w, float* out, float* acc, float* keep, float a, int mode,
                  int channels, int taps, void* stream) {
    InkConv j;
    j.x = x, j.state = state, j.w = w, j.out = out, j.acc = acc, j.keep = keep, j.a = a, j.mode = mode;
    ink_conv_few(j, InkConv{}, channels, 1, taps, 1, stream, 0, 0);
}

void ink_moe_conv_add(const float* x, const float* w, const float* scale, const int* ids, int n_used, float* fo,
                      float* state, const float* cw, float* out, float* acc, float a, int channels, int taps,
                      void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_moe_conv_add: 2 to 8 taps");
    Args g;
    const void* p[9] = {x, w, scale, ids, fo, state, cw, out, acc};
    std::copy(p, p + 9, g.p);
    g.n[0] = n_used, g.n[1] = channels, g.n[2] = taps;
    g.f[0] = a, g.f[1] = conv_norm(nullptr, taps);
    run("ink_moe_conv_add", g, stream, {blocks((size_t) channels, 256), 1, 1}, 256);
}

// ---- the clips ----

void ink_clips(const InkClips& c, void* stream) {
    if (c.rows <= 0) return;
    if (c.kv8) unported("FP8 attention caches (--kv fp8)");
    Args a;
    int heads = 0;
    for (int j = 0; j < 4; ++j) {
        a.p[3 * j] = c.x[j];
        a.p[3 * j + 1] = c.scale[j];
        a.p[3 * j + 2] = c.cache[j];
        a.n[j] = c.dim[j];
        a.n[4 + j] = c.heads[j];
        a.n[8 + j] = c.rms[j];
        a.f[j] = c.thr[j];
        a.f[4 + j] = c.alpha[j];
        a.f[8 + j] = c.floor[j];
        heads += c.heads[j];
    }
    a.p[12] = c.at;
    a.n[12] = c.rows;
    a.n[13] = c.ring;
    a.f[12] = c.eps;
    a.pos = c.pos;
    run("ink_clips", a, stream, {uint32_t(c.rows * heads), 1, 1}, 128);
}

void ink_clip(float* x, const float* scale, float thr, int dim, int heads, int rows, int64_t pos, float alpha,
              float floor, void* stream, const long long* at) {
    InkClips c;
    c.x[0] = x, c.scale[0] = scale, c.thr[0] = thr, c.alpha[0] = alpha, c.floor[0] = floor, c.dim[0] = dim;
    c.heads[0] = heads;
    c.pos = pos, c.at = at, c.rows = rows;
    ink_clips(c, stream);
}

void ink_head_rms(float* x, int dim, int heads, int rows, float eps, void* stream) {
    InkClips c;
    c.x[0] = x, c.dim[0] = dim, c.heads[0] = heads, c.rms[0] = true, c.eps = eps, c.rows = rows;
    ink_clips(c, stream);
}

void ink_qkv(const InkQkv& q, void* stream) {
    if (q.dim > 128 || q.n_rel > 128 || q.taps < 2 || q.taps > 8 || q.n_kv < 1)
        throw std::invalid_argument("ink_qkv: heads of up to 128 values, 2 to 8 taps");
    if (q.kv8) unported("FP8 attention caches (--kv fp8)");
    Args a;
    const void* p[17] = {q.q,     q.q_scale, q.r,  q.r_scale, q.kraw, q.vraw, q.kstate, q.vstate, q.kw,
                         q.vw,    q.kdyn,    q.vdyn, q.k,     q.v,    q.keys, q.vals,   q.at};
    std::copy(p, p + 17, a.p);
    a.n[0] = q.n_head, a.n[1] = q.n_kv, a.n[2] = q.dim, a.n[3] = q.n_rel, a.n[4] = q.taps, a.n[5] = q.ring;
    a.f[0] = q.qk_thr, a.f[1] = q.rel_thr, a.f[2] = q.alpha, a.f[3] = q.floor, a.f[4] = q.eps;
    a.f[5] = conv_norm(q.kdyn, q.taps), a.f[6] = conv_norm(q.vdyn, q.taps);
    a.pos = q.pos;
    run("ink_qkv", a, stream, {uint32_t(2 * q.n_head + 2 * q.n_kv), 1, 1}, 128);
}

// ---- elementwise ----

void ink_mul_sigmoid(float* x, const float* gate, size_t n, void* stream) {
    Args a;
    a.p[0] = x, a.p[1] = gate;
    elem(a, 0, n, stream);
}

void ink_store_ring(const float* x, uint16_t* cache, int n, int64_t pos, int rows, int ring, void* stream) {
    if (rows <= 0) return;
    Args a;
    a.p[0] = x, a.p[1] = cache;
    a.n[1] = n, a.n[2] = ring;
    a.pos = pos;
    elem(a, 1, (size_t) rows * n, stream);
}

void ink_softcap(float* logits, int n_keep, int n_all, int rows, float cap, void* stream) {
    if (rows <= 0) return;
    Args a;
    a.p[0] = logits;
    a.n[1] = n_keep, a.n[2] = n_all;
    a.f[0] = cap;
    elem(a, 2, (size_t) rows * n_all, stream);
}

// ---- the attention ----

// online softmax: no scores are written out
size_t ink_attend_floats(int, int, int, int64_t) { return 1; }

void ink_attend(const InkAttn& c, void* stream) {
    if (c.rows <= 0) return;
    if (c.n_kv < 1 || c.n_head % c.n_kv != 0 || c.n_head / c.n_kv > 4 || (c.dim != 64 && c.dim != 128) || c.n_rel < 1 ||
        c.n_rel > 32)
        throw std::invalid_argument("ink_attend: heads of 64 or 128 values, up to 4 on a key head, up to 32 relative "
                                    "values");
    if (c.kv8) unported("FP8 attention caches (--kv fp8)");
    if (c.at && c.room < 1) throw std::invalid_argument("ink_attend: positions on the card need the rows to size for");
    Args a;
    const void* p[9] = {c.q, c.r, c.sink, c.keys, c.vals, c.rel_bias, c.gate, c.out, c.at};
    std::copy(p, p + 9, a.p);
    if (c.q8 && c.rows != 1) throw std::invalid_argument("ink_attend: a quantized output is one token's");
    a.p[10] = c.q8;
    a.n[0] = c.rows, a.n[1] = c.n_head, a.n[2] = c.n_kv, a.n[3] = c.dim, a.n[4] = c.n_rel, a.n[5] = c.extent;
    a.n[6] = c.window, a.n[7] = c.ring;
    a.f[0] = c.scale;
    a.pos = c.pos;
    // The keys a token of the call reads: up to its own, within the window; a recorded launch reads its position on
    // the card and is sized for the middle of the rows it serves.  A threadgroup to a key head and a token, its SIMD
    // groups (as many as its threadgroup memory takes, up to 16) each on every g-th key: one launch.  A few tokens
    // over more than 2,048 keys split them into runs of about 1,024 a threadgroup (up to 32 runs), which meet in a
    // scratch of its own (never freed: recorded launches keep its address) for ink_attn_join.
    int64_t keys = c.at ? std::max<int64_t>(c.room / 2, c.rows) : c.pos + c.rows;
    if (c.window > 0) keys = std::min<int64_t>(keys, c.window);
    static void* scratch = nullptr;
    static bool tried = false;
    constexpr size_t kRunFloats = size_t(4) << 20;
    const size_t per_run = (size_t) c.rows * c.n_head * (c.dim + 2);
    // A few tokens' keys go to runs of about 128 a threadgroup (up to 32 runs), a SIMD group walking 4 keys at a
    // time: 2 steps for a window's 1,024 keys, where one threadgroup a key head took 16.  On the M5 Ultra runs of 0
    // (none up to 2,048 keys), 128 and 256 keys, and 1, 2 or 8 down products a threadgroup in ink_ctrl_down, wrote a
    // short chat at 71.7 to 73.1 tok/s, within the machine's noise; 128 and 2 were the fastest.
    static const int64_t run_keys = [] {   // STRATA_INK_RUN_KEYS: keys a run (0: one threadgroup up to 2,048 keys)
        const char* v = std::getenv("STRATA_INK_RUN_KEYS");
        return v ? (int64_t) std::atoll(v) : (int64_t) 128;
    }();
    int splits = 1;
    if (c.rows < 32 && (run_keys > 0 ? keys > run_keys : keys > 2048)) {
        if (!tried) {
            tried = true;
            if (gpu::malloc_device_raw(&scratch, kRunFloats * 4) != gpu::kSuccess) scratch = nullptr;
        }
        const int64_t per = run_keys > 0 ? run_keys : 1024;
        if (scratch) splits = (int) std::min<size_t>({(size_t) ((keys + per - 1) / per), 32, kRunFloats / per_run});
    }
    a.n[8] = splits;
    a.p[9] = splits > 1 ? scratch : nullptr;
    const int per = c.n_head / c.n_kv;
    const int most = (int) std::min<size_t>(16, size_t(32768) / ((size_t) per * (c.dim + 2) * 4));
    const int groups = std::max(per, std::min<int>(most, (int) ((keys + splits * 16 - 1) / (splits * 16))));
    const size_t shared = ((size_t) groups * per * (c.dim + 2) * 4 + 15) / 16 * 16;
    run(c.dim == 128 ? "ink_attn_128" : "ink_attn_64", a, stream, {uint32_t(c.n_kv), uint32_t(c.rows), uint32_t(splits)},
        uint32_t(32 * groups), shared);
    if (splits > 1) run("ink_attn_join", a, stream, {uint32_t(c.n_head), uint32_t(c.rows), 1}, 32);
}

// A prompt batch: on the matrix units with --metal-optimize (inkling_flash.metal, Metal 4's matmul2d), for heads of
// 128 values and 16 relative values with the positions on the host; else ink_attend()'s kernel.  Neither writes a
// score out.
bool ink_attend_batch(const InkAttn& c, void* stream) {
    if (c.rows <= 0) return true;
    static const std::string flash = metal_detail::read_source(STRATA_POLY_INKLING_FLASH_SOURCE);
    static const bool ready = [] {
        std::string error;
        return gpu::metal::pipeline("poly_inkling_flash", flash.c_str(), "ink_flash_128", error) != nullptr;
    }();
    if (!(ready && gpu::metal::optimize() && c.dim == 128 && c.n_rel == 16 && !c.at && !c.q8 && !c.kv8 && c.n_kv > 0 &&
          c.n_head % c.n_kv == 0)) {
        ink_attend(c, stream);
        return true;
    }
    Args a;
    const void* p[8] = {c.q, c.r, c.sink, c.keys, c.vals, c.rel_bias, c.gate, c.out};
    std::copy(p, p + 8, a.p);
    a.n[0] = c.rows, a.n[1] = c.n_head, a.n[2] = c.n_kv, a.n[3] = c.dim, a.n[4] = c.n_rel, a.n[5] = c.extent;
    a.n[6] = c.window, a.n[7] = c.ring;
    a.f[0] = c.scale;
    a.pos = c.pos;
    // a ring: the call's keys and values copied in fp16 from the window's first, for matmul2d to read as tensors, into a
    // buffer grown to the largest call and kept
    if (c.ring) {
        static void* kv16 = nullptr;
        static size_t kv16_bytes = 0;
        const int64_t first = c.window > 0 ? std::max<int64_t>(0, c.pos - c.window + 1) : 0;
        const size_t n = size_t(c.pos + c.rows - first) * size_t(c.n_kv) * size_t(c.dim), need = 4 * n;
        if (need > kv16_bytes) {
            if (kv16) {
                gpu::stream_synchronize(static_cast<gpu::Stream>(stream));
                gpu::free_device(kv16);
            }
            kv16 = nullptr;
            kv16_bytes = 0;
            if (gpu::malloc_device_raw(&kv16, need) != gpu::kSuccess)
                throw std::runtime_error("Metal attention: the window's keys in fp16");
            kv16_bytes = need;
        }
        a.p[8] = kv16;
        a.p[9] = static_cast<uint16_t*>(kv16) + n;
        metal_detail::dispatch("poly_inkling_flash", flash, "ink_flash_kv", a, stream, {blocks(n, 256), 1, 1}, 256);
    }
    metal_detail::dispatch("poly_inkling_flash", flash, "ink_flash_128", a, stream,
                           {uint32_t(c.n_head), uint32_t((c.rows + 15) / 16), 1}, 64);
    return true;
}

bool ink_kv8_ok() { return false; }

// ---- the DSpark drafter (inkling.metal's ds_*): its pick's 2-bit screen (ds_screens()) has no Metal kernels ----

namespace {
constexpr int kDsRows = 24, kDsSplits = 64, kDsPart = 130, kDsPickParts = 128;
void ds_cap(Args& a, int64_t cap, int at) {
    a.n[at] = int(uint32_t(cap));
    a.n[at + 1] = int(uint32_t(uint64_t(cap) >> 32));
}
void ds_elem(Args& a, int kind, size_t n, void* stream) {
    if (!n) return;
    a.n[0] = kind;
    a.stride = n;
    run("ds_elem", a, stream, {blocks(n, 256), 1, 1}, 256);
}
}  // namespace

void ds_mix(const DsMix& m, int n, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    if (n > 4096) throw std::invalid_argument("ds_mix: rows of up to 4,096 values");
    Args a;
    a.p[0] = m.tap, a.p[4] = m.alpha, a.p[9] = m.out, a.p[10] = m.ring, a.p[11] = m.pos.at;
    for (int j = 0; j < 4; ++j) a.p[5 + j] = m.norm[j];
    a.n[0] = n, a.n[1] = rows, a.n[2] = m.ring_rows;
    a.f[0] = eps;
    a.pos = m.pos.pos0;
    a.stride = m.tap_stride;
    run("ds_mix", a, stream, {uint32_t(rows), 4, 1}, 256);
}

void ds_rope(float* x, int heads, int rows, const float* w, float eps, float theta, const DsPos& p, uint16_t* cache,
             int64_t cap, const float* v, uint16_t* vcache, void* stream, long long* tags) {
    if (rows <= 0) return;
    Args a;
    a.p[0] = x, a.p[1] = w, a.p[2] = cache, a.p[3] = v, a.p[4] = vcache, a.p[5] = tags, a.p[6] = p.at;
    a.n[0] = heads;
    ds_cap(a, cap, 1);
    a.f[0] = eps, a.f[1] = theta;
    a.pos = p.pos0;
    run("ds_rope", a, stream, {uint32_t(rows), uint32_t(heads), 1}, 128);
}

size_t ds_attend_floats(int kv_heads) { return (size_t) kv_heads * kDsSplits * kDsRows * kDsPart; }

void ds_attend(const float* q, const uint16_t* keys, const uint16_t* vals, const long long* tags, int64_t cap,
               const DsPos& p, int window, int rows, int heads, int kv_heads, float scale, float* part, float* out,
               void* stream) {
    if (heads % kv_heads || rows * (heads / kv_heads) > kDsRows)
        throw std::logic_error("ds_attend: more query rows a kv head than it holds");
    Args a;
    a.p[0] = q, a.p[1] = keys, a.p[2] = vals, a.p[3] = tags, a.p[4] = part, a.p[5] = p.at, a.p[6] = out;
    a.n[0] = rows, a.n[1] = window, a.n[2] = heads, a.n[3] = kv_heads;
    ds_cap(a, cap, 4);
    a.f[0] = scale;
    a.pos = p.pos0;
    run("ds_part", a, stream, {uint32_t(kv_heads), uint32_t(kDsSplits), 1}, 256);
    run("ds_join", a, stream, {uint32_t(rows), uint32_t(heads), 1}, 128);
}

void ds_swiglu(float* g, const float* u, size_t n, void* stream) {
    Args a;
    a.p[0] = g, a.p[1] = u;
    ds_elem(a, 0, n, stream);
}

void ds_block_in(float* x, const float* mask, int n, int rows, const DsPos& p, const long long* token, float* state,
                 int rank, int32_t* prev, const float* ring, int ring_rows, float* ctx, void* stream) {
    Args a;
    a.p[0] = x, a.p[1] = mask, a.p[2] = p.at, a.p[3] = token, a.p[4] = state, a.p[5] = prev, a.p[6] = ring, a.p[7] = ctx;
    a.n[1] = n, a.n[2] = rows, a.n[3] = rank, a.n[4] = ring_rows;
    a.pos = p.pos0;
    ds_elem(a, 1, (size_t) n, stream);
}

void ds_markov_in(const float* state, const uint16_t* w1, const int32_t* prev, float* in, int rank, void* stream) {
    Args a;
    a.p[0] = state, a.p[1] = w1, a.p[2] = prev, a.p[3] = in;
    a.n[1] = rank;
    ds_elem(a, 2, (size_t) rank, stream);
}

void ds_markov_step(const float* ctx_h, const float* ctx_f, const float* bias, const float* sp, float* state,
                    float* latent, int rank, void* stream) {
    Args a;
    a.p[0] = ctx_h, a.p[1] = ctx_f, a.p[2] = bias, a.p[3] = sp, a.p[4] = state, a.p[5] = latent;
    a.n[1] = rank;
    ds_elem(a, 3, (size_t) rank, stream);
}

size_t ds_pick_floats() { return kDsPickParts * 3; }

void ds_pick(const float* base, const float* fy, const float* fb, int n_keep, float cap, float* scratch, Likeliest* out,
             int32_t* prev, void* stream) {
    Args a;
    a.p[0] = base, a.p[1] = fy, a.p[2] = fb, a.p[3] = scratch, a.p[4] = out, a.p[5] = prev;
    a.n[0] = n_keep, a.n[1] = kDsPickParts;
    a.f[0] = cap;
    run("ds_pick_part", a, stream, {uint32_t(kDsPickParts), 1, 1}, 256);
    run("ds_pick_join", a, stream, {1, 1, 1}, 32);
}

void ds_fold_q8(const float* ft, int n_rows, int blocks_, int b, void* fold, void* stream) {
    Args a;
    a.p[0] = ft, a.p[1] = fold;
    a.n[0] = n_rows, a.n[1] = blocks_, a.n[2] = b;
    run("ds_fold_q8", a, stream, {blocks((size_t) n_rows, 256), 1, 1}, 256);
}

bool ds_screens(int, int) { return false; }
size_t ds_screen_bytes(int, int64_t) { return 0; }
void ds_screen_pack(const void*, int, int64_t, void*, void*) { unported("the drafter's 2-bit screen"); }
int64_t ds_screen_rows(int64_t n) { return n; }
void ds_screen_order(const float*, int64_t, float*, void*) { unported("the drafter's 2-bit screen"); }
size_t ds_screen_scratch_bytes(int64_t) { return 0; }
void ds_head_base(const DsScreen&, const void*, int, float*, void*) { unported("the drafter's 2-bit screen"); }
void ds_head_pick(const DsScreen&, const void*, const void*, const float*, int, void*, int32_t*, Likeliest*, void*) {
    unported("the drafter's 2-bit screen");
}

}  // namespace strata::poly

namespace strata::kernels {
void native_quantize_q8_1_many(const float* const* x, void* const* x_q8_1, int count, int n_in, void* stream) {
    for (int j = 0; j < count; ++j) native_quantize_q8_1(x[j], x_q8_1[j], n_in, 1, stream);
}
}  // namespace strata::kernels
