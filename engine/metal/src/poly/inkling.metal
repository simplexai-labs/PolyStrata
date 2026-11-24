// metal/src/poly/inkling.metal - Inkling 2's kernels on Metal (inkling2_kernels.hpp; inkling.cpp launches them).  The
// arithmetic is inkling2.cu's, kernel for kernel: the mixes over the residual streams, the loop controllers' gates,
// the short convolutions, the clips, the attention with its bias by distance and its sink, and SiTU.
#include <metal_stdlib>
using namespace metal;

// Mirrors Args in inkling.cpp; device pointers are Metal gpuAddress values.
struct Args {
    device void* p[72];
    long pos;
    ulong stride;
    int n[24];
    float f[16];
};
template <class T> inline device T* P(constant Args& a, int k) { return static_cast<device T*>(a.p[k]); }

// The threadgroup's sum of v in every thread, the SIMD groups' sums added in their order (basic.metal's block_sum).
inline float tg_sum(float v, threadgroup float* part, uint tid, uint nt) {
    for (uint k = 16; k; k >>= 1) v += simd_shuffle_xor(v, k);
    if ((tid & 31) == 0) part[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint k = 0; k < nt / 32; ++k) s += part[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// A controller's weight: F32, or BF16 (--loop-bf16) widened as bf16_to_f32() does it.
inline float wval(constant Args& a, int k, ulong i, bool bf16) {
    return bf16 ? as_type<float>(uint(P<ushort>(a, k)[i]) << 16) : P<float>(a, k)[i];
}

// Value i of a row into its q8_1 blocks as poly_quantize (quantized.metal) writes them: every lane of a SIMD group
// takes part, lane i % 32 of the block of 32 values the SIMD group holds.
inline void q8_put(device uchar* q, int i, float v) {
    float d = simd_max(abs(v)) / 127.0f, sum = v;
    for (uint k = 16; k; k >>= 1) sum += simd_shuffle_xor(sum, k);
    device uchar* dst = q + ulong(i / 32) * 36;
    reinterpret_cast<device char*>(dst + 4)[i % 32] = char(d ? round(v / d) : 0);
    if (i % 32 == 0) {
        reinterpret_cast<device half*>(dst)[0] = half(d);
        reinterpret_cast<device half*>(dst)[1] = half(sum);
    }
}

// rms_norm() of a row as poly_norm makes it (y = x * rsqrt(mean(x^2) + eps) * w), with q8 (null: none) its q8_1
// blocks; x may be y.  The whole threadgroup takes part; n a multiple of 32 where q8 is set.
inline void norm_row(device const float* x, device const float* w, device float* y, device uchar* q8, int n, float eps,
                     threadgroup float* part, uint tid, uint nt) {
    float ss = 0;
    for (int i = tid; i < n; i += nt) ss += x[i] * x[i];
    const float inv = rsqrt(tg_sum(ss, part, tid, nt) / n + eps);
    for (int i = tid; i < n; i += nt) {
        const float v = x[i] * inv * (w ? w[i] : 1.0f);
        y[i] = v;
        if (q8) q8_put(q8, i, v);
    }
}

// ---- the mixes over the residual streams (ink_mix(), ink_mix_many(), ink_mix_norm()) ----
// p[0..5] the states, p[6 + job] a job's queries (nq of n), p[18 + 4 job + j] its outputs, p[66] the norm's weight,
// p[67] its output (out[0]'s rms_norm()), p[68] that quantized (one row); n: states, queries, n_embd; f[0] the norm's
// eps.  A row and a job a threadgroup (grid rows x jobs), a row of up to 4 values a thread: each thread reads its
// values of the states once and keeps them, and the first output's, for the outputs and the norm.
kernel void ink_mix(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    constexpr int H = 4;
    const uint nt = shape.x;
    threadgroup float part[30][32];
    threadgroup float coef[6][4];
    threadgroup float npart[32];
    const int ns = a.n[0], nq = a.n[1], n = a.n[2];
    const ulong row = ulong(g.x) * n;
    device const float* q = P<float>(a, 6 + g.y);
    float x[H][6], sum[6][5];
    for (int s = 0; s < 6; ++s)
        for (int k = 0; k < 5; ++k) sum[s][k] = 0;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        float qv[4];
        for (int j = 0; j < 4; ++j) qv[j] = j < nq ? q[ulong(j) * n + i] : 0.0f;
        for (int s = 0; s < 6; ++s) {
            if (s < ns) {
                const float v = P<float>(a, s)[row + i];
                x[h][s] = v;
                sum[s][0] += v * v;
                for (int j = 0; j < 4; ++j) sum[s][1 + j] += v * qv[j];
            }
        }
    }
    for (int s = 0; s < 6; ++s)
        for (int k = 0; k < 5; ++k)
            if (s < ns && k <= nq) {
                const float w = simd_sum(sum[s][k]);
                if ((tid & 31) == 0) part[s * 5 + k][tid >> 5] = w;
            }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < nq) {
        const int j = tid;
        const float sc = 1.0f / sqrt(float(n));
        float v[6], inv[6], m = -INFINITY;
        for (int s = 0; s < ns; ++s) {
            float ss = 0, dot = 0;
            for (uint w = 0; w < nt / 32; ++w) {
                ss += part[s * 5][w];
                dot += part[s * 5 + 1 + j][w];
            }
            inv[s] = 1.0f / sqrt(ss / float(n) + 1e-6f);
            v[s] = dot * inv[s] * sc;
            m = max(m, v[s]);
        }
        float total = 0;
        for (int s = 0; s < ns; ++s) {
            v[s] = exp(v[s] - m);
            total += v[s];
        }
        for (int s = 0; s < ns; ++s) coef[s][j] = v[s] / total * inv[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o0[H], ss = 0;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        for (int j = 0; j < nq; ++j) {
            float acc = 0;
            for (int s = 0; s < ns; ++s) acc += coef[s][j] * x[h][s];
            P<float>(a, 18 + 4 * g.y + j)[row + i] = acc;
            if (j == 0) o0[h] = acc, ss += acc * acc;
        }
    }
    if (!a.p[67]) return;
    const float inv = rsqrt(tg_sum(ss, npart, tid, nt) / n + a.f[0]);
    device const float* w = P<float>(a, 66);
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        const float v = o0[h] * inv * (w ? w[i] : 1.0f);
        P<float>(a, 67)[row + i] = v;
        if (a.p[68]) q8_put(P<uchar>(a, 68), i, v);
    }
}

// ink_norm_many_q8(): job j (grid y) p[j] x, p[3 + j] w, p[6 + j] y, p[9 + j] q8; n[0] n; f[0] eps.  With n[1] the
// gate first (ink_gate_norm_q8(), one job): p[12] msg, p[13] gain, p[14] forget, p[15] the gated row, which p[0]
// (cur) may be.
kernel void ink_norm_q8(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[32];
    const int j = g.y, n = a.n[0];
    if (a.n[1]) {
        device const float* cur = P<float>(a, 0);
        device float* out = P<float>(a, 15);
        for (int i = tid; i < n; i += nt) {
            const float c = cur[i];
            out[i] = c - precise::tanh(P<float>(a, 14)[i]) * c + precise::tanh(P<float>(a, 13)[i]) * P<float>(a, 12)[i];
        }
        norm_row(out, P<float>(a, 3), P<float>(a, 6), P<uchar>(a, 9), n, a.f[0], part, tid, nt);
        return;
    }
    norm_row(P<float>(a, j), P<float>(a, 3 + j), P<float>(a, 6 + j), P<uchar>(a, 9 + j), n, a.f[0], part, tid, nt);
}

// ink_ctrl_down(): controller j's outputs o .. o + O - 1 (threadgroup x = (j * rank + o) / O, O = n[3], 8 at most) of
// row y: [rms(cur) | rms(msg)] (eps 1e-6, no weight) times its down matrix's rows (2 n values, F32 or BF16).  p[j]
// cur, p[3 + j] msg, p[6 + j] down, p[9 + j] ch; n: n_embd, rank, bf16, O.  Each thread keeps its normed inputs (up to
// 24) for the O rows.
kernel void ink_ctrl_down(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    constexpr int H = 24;
    const uint nt = shape.x;
    threadgroup float part[32], dots[8][32];
    const int n = a.n[0], rank = a.n[1], O = a.n[3], j = (g.x * O) / rank, o0 = (g.x * O) % rank;
    const bool bf = a.n[2] != 0;
    const ulong row = ulong(g.y) * n;
    device const float* cur = P<float>(a, j) + row;
    device const float* msg = P<float>(a, 3 + j) + row;
    float xs[H];
    float sa = 0, sb = 0;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= 2 * n) break;
        xs[h] = i < n ? cur[i] : msg[i - n];
        if (i < n) sa += xs[h] * xs[h];
        else sb += xs[h] * xs[h];
    }
    const float ia = 1.0f / sqrt(tg_sum(sa, part, tid, nt) / float(n) + 1e-6f);
    const float ib = 1.0f / sqrt(tg_sum(sb, part, tid, nt) / float(n) + 1e-6f);
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= 2 * n) break;
        xs[h] *= i < n ? ia : ib;
    }
    for (int k = 0; k < O; ++k) {
        if (o0 + k >= rank) break;
        const ulong w0 = ulong(o0 + k) * 2 * n;
        float acc = 0;
        for (int h = 0; h < H; ++h) {
            const int i = tid + h * nt;
            if (i >= 2 * n) break;
            acc += wval(a, 6 + j, w0 + i, bf) * xs[h];
        }
        acc = simd_sum(acc);
        if ((tid & 31) == 0) dots[k][tid >> 5] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < O && o0 + int(tid) < rank) {
        float acc = 0;
        for (uint w = 0; w < nt / 32; ++w) acc += dots[tid][w];
        P<float>(a, 9 + j)[ulong(g.y) * rank + o0 + tid] = acc;
    }
}

// gemv_set(): matrix j's p[j] (F32, or BF16 with n[9]) on input p[6 + j] into p[12 + j], n[j] outputs each, n[7]
// values a row, rows n[8]; the outputs of all of them in turn along the grid's x.  A threadgroup an output and a row.
kernel void ink_gemv_set(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[32];
    int o = g.x, j = 0;
    while (o >= a.n[j]) o -= a.n[j++];
    const int ni = a.n[7];
    const bool bf = a.n[9] != 0;
    device const float* x = P<float>(a, 6 + j) + ulong(g.y) * ni;
    float acc = 0;
    for (int i = tid; i < ni; i += nt) acc += wval(a, j, ulong(o) * ni + i, bf) * x[i];
    acc = tg_sum(acc, part, tid, nt);
    if (tid == 0) P<float>(a, 12 + j)[ulong(g.y) * a.n[j] + o] = acc;
}

// ---- the loop controllers ----
// ink_cat_norm_set(): p[j] cur, p[3 + j] msg, p[6 + j] out; n[0] n_embd.  A row and a pair a threadgroup.
kernel void ink_cat_norm(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[32];
    const int n = a.n[0];
    const ulong row = ulong(g.x) * n;
    device const float* cur = P<float>(a, g.y) + row;
    device const float* msg = P<float>(a, 3 + g.y) + row;
    device float* out = P<float>(a, 6 + g.y) + 2 * row;
    float sa = 0, sb = 0;
    for (int i = tid; i < n; i += nt) {
        sa += cur[i] * cur[i];
        sb += msg[i] * msg[i];
    }
    const float ia = 1.0f / sqrt(tg_sum(sa, part, tid, nt) / float(n) + 1e-6f);
    const float ib = 1.0f / sqrt(tg_sum(sb, part, tid, nt) / float(n) + 1e-6f);
    for (int i = tid; i < n; i += nt) {
        out[i] = cur[i] * ia;
        out[n + i] = msg[i] * ib;
    }
}

inline float gate_value(float c, float m, float g, float f) {
    return c - precise::tanh(f) * c + precise::tanh(g) * m;
}

// ink_gate_set(): p[j] cur, p[3 + j] msg, p[6 + j] gain, p[9 + j] forget, p[12 + j] out; n[0] the count; stride the
// values.  Every cur of a value is read before an out is written (an out may be a cur).
kernel void ink_gate(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.stride) return;
    const int count = a.n[0];
    float c[3];
    for (int j = 0; j < 3; ++j)
        if (j < count) c[j] = P<float>(a, j)[i];
    for (int j = 0; j < 3; ++j)
        if (j < count)
            P<float>(a, 12 + j)[i] = gate_value(c[j], P<float>(a, 3 + j)[i], P<float>(a, 6 + j)[i], P<float>(a, 9 + j)[i]);
}

// ink_control3_gate(): p[j] cur, p[3 + j] msg, p[6 + j] ch, p[9 + j] gain, p[12 + j] forget, p[15 + j] out; n: n,
// rank, count, bf16.  A SIMD group a value, 8 a threadgroup.
kernel void ink_control3_gate(constant Args& a [[buffer(0)]], uint grp [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int n = a.n[0], rank = a.n[1], count = a.n[2];
    const bool bf = a.n[3] != 0;
    const int i = int(grp) * 8 + int(sg);
    if (i >= n) return;
    float g[3], f[3];
    for (int j = 0; j < 3; ++j) {
        if (j >= count) break;
        device const float* ch = P<float>(a, 6 + j);
        float ga = 0, fa = 0;
        for (int k = lane; k < rank; k += 32) {
            ga += wval(a, 9 + j, ulong(i) * rank + k, bf) * ch[k];
            fa += wval(a, 12 + j, ulong(i) * rank + k, bf) * ch[k];
        }
        g[j] = simd_sum(ga);
        f[j] = simd_sum(fa);
    }
    if (lane) return;
    float c[3];
    for (int j = 0; j < 3; ++j)
        if (j < count) c[j] = P<float>(a, j)[i];
    for (int j = 0; j < 3; ++j)
        if (j < count) P<float>(a, 15 + j)[i] = gate_value(c[j], P<float>(a, 3 + j)[i], g[j], f[j]);
}

// ---- the short convolutions (ink_conv()) ----
// Token t's convolved value of channel c: past[k] is the input k tokens back, from x or, before the call, the state.
inline float conv_value(device const float* x, device const float* state, device const float* w,
                        device const float* dyn, int channels, int heads, int taps, int t, int c, float norm) {
    float past[8];
    for (int k = 0; k < taps; ++k) {
        const int u = t - k;
        past[k] = u >= 0 ? x[ulong(u) * channels + c] : state[ulong(taps - 1 + u) * channels + c];
    }
    float acc = 0;
    for (int j = 0; j < taps; ++j) acc += past[taps - 1 - j] * w[ulong(c) * taps + j];
    if (dyn) {
        const int head = c / (channels / heads);
        device const float* d = dyn + (ulong(t) * heads + head) * taps;
        for (int lag = 0; lag < taps; ++lag) acc += past[lag] * d[lag];
    }
    return (past[0] + acc) * norm;
}

// Channel c's state after the first `rows` tokens of a call, from the state before it into out (out may be state).
inline void conv_state_into(device const float* x, device const float* state, device float* out, int channels,
                            int taps, int rows, int c) {
    const int keep = taps - 1;
    for (int k = 0; k < keep; ++k) {
        const int u = rows - keep + k;
        out[ulong(k) * channels + c] = u >= 0 ? x[ulong(u) * channels + c] : state[ulong(keep + u) * channels + c];
    }
}

// The snapshots of a window (bit t of bits: the state after token t, the j-th at snap + j * stride), then the state.
inline void conv_states(device const float* x, device float* state, device float* snap, ulong stride, uint bits,
                        int channels, int taps, int rows, int c) {
    int k = 0;
    if (snap)
        for (uint b = bits; b; b &= b - 1, ++k)
            conv_state_into(x, state, snap + ulong(k) * stride, channels, taps, int(ctz(b)) + 1, c);
    conv_state_into(x, state, state, channels, taps, rows, c);
}

// ink_conv_few(): one or two jobs (grid y), a channel a thread, a call of a few tokens.  Job j: p[8j ..] x, state, w,
// dyn, out, snap, acc, keep; n: channels, heads, taps, rows, mode0, mode1, bits; f: a0, a1, norm0, norm1; stride.
kernel void ink_conv_few(constant Args& a [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    const int channels = a.n[0], heads = a.n[1], taps = a.n[2], rows = a.n[3], j = g.y, c = g.x;
    if (c >= channels) return;
    const int b = 8 * j, mode = a.n[4 + j];
    const float s = a.f[j], norm = a.f[2 + j];
    device const float* x = P<float>(a, b);
    device float* state = P<float>(a, b + 1);
    device float* out = P<float>(a, b + 4);
    device float* acc = P<float>(a, b + 6);
    device float* keep = P<float>(a, b + 7);
    for (int t = 0; t < rows; ++t) {
        const ulong i = ulong(t) * channels + c;
        const float y = conv_value(x, state, P<float>(a, b + 2), P<float>(a, b + 3), channels, heads, taps, t, c, norm);
        out[i] = y;
        if (!acc) continue;
        if (mode == 0) {
            acc[i] = acc[i] + y * s;
        } else {
            if (mode == 1) keep[i] = acc[i];
            acc[i] = y;
        }
    }
    conv_states(x, state, P<float>(a, b + 5), a.stride, uint(a.n[6]), channels, taps, rows, c);
}

// ink_conv() of many tokens: p x, state, w, dyn, out; n: channels, heads, taps, rows; f[0] norm.  A value a thread;
// ink_conv_tail then the snapshots and the state.
kernel void ink_conv_rows(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    const int channels = a.n[0];
    if (i >= uint(channels) * uint(a.n[3])) return;
    P<float>(a, 4)[i] = conv_value(P<float>(a, 0), P<float>(a, 1), P<float>(a, 2), P<float>(a, 3), channels, a.n[1],
                                   a.n[2], int(i / channels), int(i % channels), a.f[0]);
}

// p x, state, snap; n: channels, taps, rows, bits; stride.  A channel a thread.
kernel void ink_conv_tail(constant Args& a [[buffer(0)]], uint c [[thread_position_in_grid]]) {
    if (c >= uint(a.n[0])) return;
    conv_states(P<float>(a, 0), P<float>(a, 1), P<float>(a, 2), a.stride, uint(a.n[3]), a.n[0], a.n[1], a.n[2], c);
}

// ink_moe_conv_add(): p x, w, scale, ids, fo, state, cw, out, acc; n: n_used, channels, taps; f: a, norm.
kernel void ink_moe_conv_add(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    const int used = a.n[0], channels = a.n[1], taps = a.n[2];
    if (i >= uint(channels)) return;
    device const float* x = P<float>(a, 0);
    device const float* w = P<float>(a, 1);
    device const float* scale = P<float>(a, 2);
    device const int* ids = P<int>(a, 3);
    float s = 0;
    for (int k = 0; k < used; ++k) {
        const float wk = scale ? w[k] * scale[ids[k]] : w[k];
        s += wk * x[ulong(k) * channels + i];
    }
    device float* fo = P<float>(a, 4);
    device float* state = P<float>(a, 5);
    fo[i] = 0.0f + s;
    const float y = conv_value(fo, state, P<float>(a, 6), nullptr, channels, 1, taps, 0, int(i), a.f[1]);
    P<float>(a, 7)[i] = y;
    conv_state_into(fo, state, state, channels, taps, 1, int(i));
    device float* acc = P<float>(a, 8);
    acc[i] = acc[i] + y * a.f[0];
}

// ---- the clips and head norms ----
// One head's clip (ink_clip()): the threadgroup's threads over its values; p the token's position.
inline void clip_head(device float* v, device const float* sc, float thr, int dim, float alpha, float floor_, long p,
                      threadgroup float* part, uint tid, uint nt) {
    float ss = 0;
    for (int i = tid; i < dim; i += nt) {
        const float y = sc ? v[i] * sc[i] : v[i];
        ss += y * y;
    }
    float f = min(thr / (sqrt(tg_sum(ss, part, tid, nt)) + 1e-6f), 1.0f);
    if (alpha > 0.0f) f *= 1.0f + alpha * log(max(1.0f, float(p + 1) / floor_));
    for (int i = tid; i < dim; i += nt) v[i] = (sc ? v[i] * sc[i] : v[i]) * f;
}

inline void rms_head(device float* v, int dim, float eps, threadgroup float* part, uint tid, uint nt) {
    float ss = 0;
    for (int i = tid; i < dim; i += nt) ss += v[i] * v[i];
    const float f = 1.0f / sqrt(tg_sum(ss, part, tid, nt) / float(dim) + eps);
    for (int i = tid; i < dim; i += nt) v[i] *= f;
}

inline void store_head(device const float* v, device half* cache, long cell, int heads, int dim, int h, uint tid,
                       uint nt) {
    device half* to = cache + ulong(cell) * heads * dim + ulong(h) * dim;
    for (int i = tid; i < dim; i += nt) to[i] = half(v[i]);
}

// ink_clips(): job j's x p[3j], scale p[3j + 1], cache p[3j + 2]; p[12] at; n: dim[4], heads[4], rms[4], rows, ring;
// f: thr[4], alpha[4], floor[4], eps; pos.  A token's head of a job a threadgroup of 128.
kernel void ink_clips(constant Args& a [[buffer(0)]], uint grp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    const int rows = a.n[12];
    int b = int(grp), j = 0;
    while (j < 3 && b >= rows * a.n[4 + j]) b -= rows * a.n[4 + j++];
    const int heads = a.n[4 + j], dim = a.n[j], row = b / heads, head = b % heads;
    device const long* at = P<long>(a, 12);
    device float* v = P<float>(a, 3 * j) + (ulong(row) * heads + head) * dim;
    if (a.n[8 + j]) {
        rms_head(v, dim, a.f[12], part, tid, nt);
    } else {
        device const float* sc = P<float>(a, 3 * j + 1);
        clip_head(v, sc ? sc + ulong(head) * dim : nullptr, a.f[j], dim, a.f[4 + j], a.f[8 + j],
                  at ? at[2 * row] : a.pos + row, part, tid, nt);
    }
    if (!a.p[3 * j + 2]) return;
    threadgroup_barrier(mem_flags::mem_device);
    long cell = (at ? at[0] : a.pos) + row;
    if (a.n[13] > 0) cell %= a.n[13];
    store_head(v, P<half>(a, 3 * j + 2), cell, heads, dim, head, tid, nt);
}

// ink_qkv(): one token.  p: q, q_scale, r, r_scale, kraw, vraw, kstate, vstate, kw, vw, kdyn, vdyn, k, v, keys, vals,
// at; n: n_head, n_kv, dim, n_rel, taps, ring; f: qk_thr, rel_thr, alpha, floor, eps, k_norm, v_norm; pos.
// Threadgroups [0, n_head) Q's heads, then K's n_kv, V's n_kv, and the relative positions' n_head; 128 threads.
kernel void ink_qkv(constant Args& a [[buffer(0)]], uint grp [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    const int n_head = a.n[0], n_kv = a.n[1], dim = a.n[2], n_rel = a.n[3], taps = a.n[4], ring = a.n[5];
    const long p = a.p[16] ? P<long>(a, 16)[0] : a.pos;
    int b = int(grp);
    if (b < n_head) {
        device const float* sc = P<float>(a, 1);
        clip_head(P<float>(a, 0) + ulong(b) * dim, sc ? sc + ulong(b) * dim : nullptr, a.f[0], dim, a.f[2], a.f[3], p,
                  part, tid, nt);
        return;
    }
    b -= n_head;
    if (b < 2 * n_kv) {
        const bool is_v = b >= n_kv;
        const int h = is_v ? b - n_kv : b, channels = n_kv * dim;
        device const float* x = P<float>(a, is_v ? 5 : 4);
        device float* state = P<float>(a, is_v ? 7 : 6);
        device float* out = P<float>(a, is_v ? 13 : 12);
        if (int(tid) < dim) {
            const int c = h * dim + tid;
            out[c] = conv_value(x, state, P<float>(a, is_v ? 9 : 8), P<float>(a, is_v ? 11 : 10), channels, n_kv, taps,
                                0, c, a.f[is_v ? 6 : 5]);
            conv_state_into(x, state, state, channels, taps, 1, c);
        }
        threadgroup_barrier(mem_flags::mem_device);
        device float* head = out + ulong(h) * dim;
        if (is_v) rms_head(head, dim, a.f[4], part, tid, nt);
        else clip_head(head, nullptr, a.f[0], dim, 0.0f, 1.0f, p, part, tid, nt);
        threadgroup_barrier(mem_flags::mem_device);
        store_head(head, P<half>(a, is_v ? 15 : 14), ring > 0 ? p % ring : p, n_kv, dim, h, tid, nt);
        return;
    }
    b -= 2 * n_kv;
    device const float* rs = P<float>(a, 3);
    clip_head(P<float>(a, 2) + ulong(b) * n_rel, rs ? rs + ulong(b) * n_rel : nullptr, a.f[1], n_rel, a.f[2], a.f[3],
              p, part, tid, nt);
}

// ---- elementwise: n[0] the kind, stride the count ----
// 0: x *= sigmoid(gate) (p x, gate); 1: an fp16 ring's rows (p x, cache; n[1] n, n[2] ring; pos); 2: the soft cap
// (p logits; n[1] n_keep, n[2] n_all; f[0] cap)
kernel void ink_elem(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.stride) return;
    switch (a.n[0]) {
        case 0: P<float>(a, 0)[i] *= sigm(P<float>(a, 1)[i]); break;
        case 1: {
            const int n = a.n[1];
            P<half>(a, 1)[ulong((a.pos + long(i / n)) % a.n[2]) * n + i % n] = half(P<float>(a, 0)[i]);
            break;
        }
        case 2: {
            device float* l = P<float>(a, 0);
            l[i] = int(i % uint(a.n[2])) < a.n[1] ? a.f[0] * precise::tanh(l[i] / a.f[0]) : -INFINITY;
            break;
        }
    }
}

// ---- SiTU (situ_rows()) ----
struct SituArgs {
    device void* p[6];   // gate_up, out, gate scale, up scale, ids, q8 (the rows back to back as one vector)
    int n, rows;
    float beta, lb;
    int e[32];
};
// with q8 n is a multiple of 32, and a SIMD group's lanes a block of a row
kernel void ink_situ(constant SituArgs& a [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    const int i = g.x, r = g.y;
    if (i >= a.n) return;
    device const float* row = static_cast<device float*>(a.p[0]) + ulong(r) * 2 * a.n;
    float sg = 1, su = 1;
    if (a.p[2] || a.p[3]) {
        const int e = a.p[4] ? static_cast<device int*>(a.p[4])[r] : a.e[r];
        if (a.p[2]) sg = static_cast<device float*>(a.p[2])[e];
        if (a.p[3]) su = static_cast<device float*>(a.p[3])[e];
    }
    const float gv = row[i] * sg;
    float u = row[a.n + i] * su;
    if (a.lb > 0.0f) u = a.lb * precise::tanh(u / a.lb);
    const float h = a.beta * precise::tanh(gv / a.beta) * sigm(gv) * u;
    static_cast<device float*>(a.p[1])[ulong(r) * a.n + i] = h;
    if (a.p[5]) q8_put(static_cast<device uchar*>(a.p[5]), r * a.n + i, h);
}

// ---- the attention (ink_attend()) ----
// p: q, r, sink, keys, vals, rel_bias, gate, out, at, runs, q8 (one row: out quantized); n: rows, n_head, n_kv, dim, n_rel, extent, window, ring,
// splits; f[0] the scale; pos.  A threadgroup a key head, a token and a run of its keys (grid n_kv x rows x splits):
// its SIMD groups take every g-th key, each with a softmax of its own for each of the key head's query heads, lanes
// side by side on a key (PER values a lane).  A key at distance d scores scale * q.k + r . rel_bias[d] (0 from
// `extent` on) - sink.  The SIMD groups meet in threadgroup memory; with splits the runs go to `runs` for
// ink_attn_join, else the softmax's extra term of score 0 and the gate are taken here.
inline long ink_pos(constant Args& a, int t) { return a.p[8] ? P<long>(a, 8)[2 * t] : a.pos + t; }

template <int PER>   // a lane's values of a head: dim / 8
kernel void ink_attn(constant Args& a [[buffer(0)]], threadgroup float* sh [[threadgroup(0)]],
                     uint3 grp [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]]) {
    const int g = grp.x, t = grp.y, z = grp.z;
    const int nh = a.n[1], nkv = a.n[2], dim = a.n[3], nrel = a.n[4], extent = a.n[5], window = a.n[6],
              ring = a.n[7], splits = a.n[8];
    const int per = nh / nkv;
    const long p = ink_pos(a, t);
    long lo = window > 0 ? max(0L, p - window + 1) : 0, hi = p;
    if (splits > 1) {
        const long span = (hi - lo + splits) / splits, first = lo + long(z) * span;
        hi = min(hi, first + span - 1);
        lo = first;
    }
    const uint slot = lane >> 3, l8 = lane & 7;   // four keys at a time, 8 lanes side by side on a key
    const ulong head0 = ulong(t) * nh + ulong(g) * per;
    float q[4][PER], r[4][4], sink[4];
    for (int hp = 0; hp < 4; ++hp) {
        for (int c = 0; c < PER; ++c)
            q[hp][c] = hp < per ? P<float>(a, 0)[(head0 + hp) * dim + l8 * PER + c] * a.f[0] : 0;
        for (int c = 0; c < 4; ++c) {
            const int k = l8 * 4 + c;
            r[hp][c] = hp < per && k < nrel ? P<float>(a, 1)[(head0 + hp) * nrel + k] : 0;
        }
        sink[hp] = hp < per ? P<float>(a, 2)[head0 + hp] : 0;
    }
    float m[4], l[4], acc[4][PER];
    for (int hp = 0; hp < 4; ++hp) {
        m[hp] = -INFINITY;
        l[hp] = 0;
        for (int c = 0; c < PER; ++c) acc[hp][c] = 0;
    }
    device const half* keys = P<half>(a, 3);
    device const half* vals = P<half>(a, 4);
    device const float* rel = P<float>(a, 5);
    const long step = long(groups) * 4;
    for (long j0 = lo + long(sg) * 4; j0 <= hi; j0 += step) {   // the same steps for the SIMD group's lanes
        const long j = j0 + slot;
        const bool valid = j <= hi;
        float k[PER], v[PER], rb[4];
        const ulong at = (ulong(ring > 0 ? (valid ? j : hi) % ring : (valid ? j : hi)) * nkv + g) * dim + l8 * PER;
        for (int c = 0; c < PER; ++c) {
            k[c] = float(keys[at + c]);
            v[c] = float(vals[at + c]);
        }
        const long dist = p - j;
        for (int c = 0; c < 4; ++c) {
            const int i = l8 * 4 + c;
            rb[c] = valid && i < nrel && dist < extent ? rel[ulong(dist) * nrel + i] : 0.0f;
        }
        for (int hp = 0; hp < 4; ++hp) {
            if (hp >= per) break;
            float d = 0;
            for (int c = 0; c < 4; ++c) d += rb[c] * r[hp][c];
            for (int c = 0; c < PER; ++c) d += q[hp][c] * k[c];
            d += simd_shuffle_xor(d, 1);
            d += simd_shuffle_xor(d, 2);
            d += simd_shuffle_xor(d, 4);
            if (!valid) continue;
            d -= sink[hp];
            const float next = max(m[hp], d), keep = exp(m[hp] - next), w = exp(d - next);
            l[hp] = l[hp] * keep + w;
            m[hp] = next;
            for (int c = 0; c < PER; ++c) acc[hp][c] = acc[hp][c] * keep + w * v[c];
        }
    }
    // the four keys' softmaxes into one, the lanes of one value side by side across them
    for (int hp = 0; hp < 4; ++hp) {
        if (hp >= per) break;
        float top = max(m[hp], simd_shuffle_xor(m[hp], 8));
        top = max(top, simd_shuffle_xor(top, 16));
        const float f = m[hp] == -INFINITY ? 0.0f : exp(m[hp] - top);
        float lt = l[hp] * f;
        lt += simd_shuffle_xor(lt, 8);
        lt += simd_shuffle_xor(lt, 16);
        for (int c = 0; c < PER; ++c) {
            float x = acc[hp][c] * f;
            x += simd_shuffle_xor(x, 8);
            x += simd_shuffle_xor(x, 16);
            acc[hp][c] = x;
        }
        m[hp] = top;
        l[hp] = lt;
    }
    // each SIMD group's per heads: their maxima, their sums, then their dim sums
    const int stride = 2 + dim;
    for (int hp = 0; hp < per; ++hp) {
        threadgroup float* o = sh + (ulong(sg) * per + hp) * stride;
        if (lane == 0) {
            o[0] = m[hp];
            o[1] = l[hp];
        }
        if (slot == 0)
            for (int c = 0; c < PER; ++c) o[2 + l8 * PER + c] = acc[hp][c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg >= uint(per)) return;
    const int hp = sg;   // SIMD group hp joins head hp's
    float top = -INFINITY;
    for (uint s = 0; s < groups; ++s) top = max(top, sh[(ulong(s) * per + hp) * stride]);
    if (splits <= 1) top = max(top, 0.0f);   // the extra term of score 0
    float total = 0;
    for (uint s = 0; s < groups; ++s) {
        threadgroup float* o = sh + (ulong(s) * per + hp) * stride;
        if (o[1] > 0) total += o[1] * exp(o[0] - top);
    }
    const ulong head = head0 + hp;
    if (splits > 1) {
        device float* run = P<float>(a, 9) + (head * splits + z) * stride;
        if (lane == 0) {
            run[0] = top;
            run[1] = total;
        }
        for (int d = lane; d < dim; d += 32) {
            float sum = 0;
            for (uint s = 0; s < groups; ++s) {
                threadgroup float* o = sh + (ulong(s) * per + hp) * stride;
                if (o[1] > 0) sum += o[2 + d] * exp(o[0] - top);
            }
            run[2 + d] = sum;
        }
        return;
    }
    total += exp(-top);
    device const float* gate = P<float>(a, 6);
    for (int d = lane; d < dim; d += 32) {
        float sum = 0;
        for (uint s = 0; s < groups; ++s) {
            threadgroup float* o = sh + (ulong(s) * per + hp) * stride;
            if (o[1] > 0) sum += o[2 + d] * exp(o[0] - top);
        }
        float y = sum / total;
        if (gate) y *= sigm(gate[head * dim + d]);
        P<float>(a, 7)[head * dim + d] = y;
        if (a.p[10]) q8_put(P<uchar>(a, 10), int(head * dim) + d, y);
    }
}
typedef decltype(ink_attn<16>) ink_attn_t;
template [[host_name("ink_attn_128")]] kernel ink_attn_t ink_attn<16>;
template [[host_name("ink_attn_64")]] kernel ink_attn_t ink_attn<8>;

// The splits' runs of a token's head into its output, with the extra term of score 0 and the gate.  A SIMD group a
// token's head (grid n_head x rows).
kernel void ink_attn_join(constant Args& a [[buffer(0)]], uint2 grp [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const int nh = a.n[1], dim = a.n[3], splits = a.n[8], stride = 2 + dim;
    const ulong head = ulong(grp.y) * nh + grp.x;
    device const float* run = P<float>(a, 9) + head * splits * stride;
    float top = 0;   // the extra term's score
    for (int s = 0; s < splits; ++s) top = max(top, run[s * stride]);
    float total = exp(-top);
    for (int s = 0; s < splits; ++s)
        if (run[s * stride + 1] > 0) total += run[s * stride + 1] * exp(run[s * stride] - top);
    device const float* gate = P<float>(a, 6);
    for (int d = lane; d < dim; d += 32) {
        float sum = 0;
        for (int s = 0; s < splits; ++s)
            if (run[s * stride + 1] > 0) sum += run[s * stride + 2 + d] * exp(run[s * stride] - top);
        float y = sum / total;
        if (gate) y *= sigm(gate[head * dim + d]);
        P<float>(a, 7)[head * dim + d] = y;
        if (a.p[10]) q8_put(P<uchar>(a, 10), int(head * dim) + d, y);
    }
}

// ---- the DSpark drafter (Inkling2::guess_block(), ds_commit()): inkling2.cu's ds_* kernels ----
// A row's position: at[2 t] where the launch reads it on the card (at), else pos0 + t.
inline long ds_pos(device const long* at, long pos0, int t) { return at ? at[2 * t] : pos0 + t; }

// ds_mix(): row t (x), mix j (y).  p[0] the features (feature k at k * stride), p[4] alpha, p[5..8] norm[j], p[9]
// out, p[10] ring, p[11] at; n: n, rows, ring rows; f[0] eps; pos the first row's position.
kernel void ds_mix(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[32];
    threadgroup float v[4096];
    const int t = g.x, j = g.y, n = a.n[0], rows = a.n[1];
    device const float* al = P<float>(a, 4) + j * 4;
    device const float* tap = P<float>(a, 0);
    float ss = 0;
    for (int i = tid; i < n; i += nt) {
        float x = 0;
        for (int k = 0; k < 4; ++k) x += al[k] * tap[ulong(k) * a.stride + ulong(t) * n + i];
        v[i] = x;
        ss += x * x;
    }
    const float scale = 1.0f / sqrt(tg_sum(ss, part, tid, nt) / float(n) + a.f[0]);
    device float* out = j < 3 ? P<float>(a, 9) + (ulong(j) * rows + t) * n
                              : P<float>(a, 10) + ulong(ds_pos(P<long>(a, 11), a.pos, t) % a.n[2]) * n;
    device const float* w = P<float>(a, 5 + j);
    for (int i = tid; i < n; i += nt) out[i] = v[i] * scale * w[i];
}

// ds_rope(): row t (x), head h (y), a thread a value of 128.  p: x, w, cache, v, vcache, tags, at; n: heads, cap (low
// and high words); f: eps, theta; pos.
kernel void ds_rope(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                    uint i [[thread_index_in_threadgroup]]) {
    threadgroup float part[32];
    threadgroup float y[128];
    const int t = g.x, h = g.y, heads = a.n[0];
    device float* row = P<float>(a, 0) + (ulong(t) * heads + h) * 128;
    const float xi = row[i];
    const float scale = 1.0f / sqrt(tg_sum(xi * xi, part, i, 128) / 128.0f + a.f[0]);
    y[i] = xi * scale * P<float>(a, 1)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const long pos = ds_pos(P<long>(a, 6), a.pos, t);
    const int k = i % 64;
    const float freq = precise::pow(a.f[1], -2.0f * float(k) / 128.0f), ang = float(pos) * freq;
    const float c = precise::cos(ang), s = precise::sin(ang);
    const float out = i < 64 ? y[i] * c - y[i + 64] * s : y[i - 64] * s + y[i] * c;
    if (!a.p[2]) {
        row[i] = out;
        return;
    }
    const long cap = long(uint(a.n[1])) | (long(a.n[2]) << 32);
    const ulong cell = ulong(pos % cap) * heads * 128 + ulong(h) * 128 + i;
    P<half>(a, 2)[cell] = half(out);
    if (a.p[3]) P<half>(a, 4)[cell] = half(P<float>(a, 3)[(ulong(t) * heads + h) * 128 + i]);
    if (a.p[5] && h == 0 && i == 0) P<long>(a, 5)[pos % cap] = pos;
}

// ds_attend()'s part: kv head g (x), split s (y) of the keys [lo, lo + nk) (a window layer's window before the block
// and the block, a full layer's every position), each query row of the head's (block row t's head g * grp + k as
// row t * grp + k) a SIMD group's in turn, a key at a time with an online softmax; a cell whose tag is another
// position is left out.  p: q, keys, vals, tags, part, at, out; n: rows, window, heads, kv heads, cap (low, high);
// f[0] scale; pos the block's first position.  8 SIMD groups, rows r = sg, sg + 8, sg + 16.
constant constexpr int kDsRows = 24, kDsSplits = 64, kDsPart = 130;
inline void ds_span(constant Args& a, thread long& lo, thread long& nk, thread long& per) {
    const long p0 = ds_pos(P<long>(a, 5), a.pos, 0);
    lo = a.n[1] > 0 ? max(0L, p0 - a.n[1]) : 0L;
    nk = p0 + a.n[0] - lo;
    per = ((nk + kDsSplits - 1) / kDsSplits + 31) / 32 * 32;
}
kernel void ds_part(constant Args& a [[buffer(0)]], uint2 grp [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const int g = grp.x, s = grp.y, heads = a.n[2], kvh = a.n[3], gq = heads / kvh, R = a.n[0] * gq;
    const long cap = long(uint(a.n[4])) | (long(a.n[5]) << 32);
    long lo, nk, per;
    ds_span(a, lo, nk, per);
    const long k0 = long(s) * per;
    if (k0 >= nk) return;
    const long k1 = min(nk, k0 + per);
    device const half* keys = P<half>(a, 1);
    device const half* vals = P<half>(a, 2);
    device const long* tags = P<long>(a, 3);
    device float* out = P<float>(a, 4) + (ulong(g) * kDsSplits + s) * kDsRows * kDsPart;
    for (int r = sg; r < R; r += 8) {
        float q[4], acc[4] = {0, 0, 0, 0};
        device const float* qr = P<float>(a, 0) + (ulong(r / gq) * heads + g * gq + r % gq) * 128;
        for (int c = 0; c < 4; ++c) q[c] = qr[lane * 4 + c] * a.f[0];
        float m = -INFINITY, l = 0;
        for (long k = k0; k < k1; ++k) {
            const long at = lo + k, cell = at % cap;
            if (tags && tags[cell] != at) continue;
            const ulong o = (ulong(cell) * kvh + g) * 128 + lane * 4;
            float d = 0;
            for (int c = 0; c < 4; ++c) d += q[c] * float(keys[o + c]);
            d = simd_sum(d);
            const float next = max(m, d), keep = exp(m - next), w = exp(d - next);
            l = l * keep + w;
            m = next;
            for (int c = 0; c < 4; ++c) acc[c] = acc[c] * keep + w * float(vals[o + c]);
        }
        for (int c = 0; c < 4; ++c) out[r * kDsPart + lane * 4 + c] = acc[c];
        if (lane == 0) {
            out[r * kDsPart + 128] = m;
            out[r * kDsPart + 129] = l;
        }
    }
}

// ds_attend()'s join: query row t (x), head h (y), a thread a value
kernel void ds_join(constant Args& a [[buffer(0)]], uint2 grp [[threadgroup_position_in_grid]],
                    uint d [[thread_index_in_threadgroup]]) {
    const int t = grp.x, h = grp.y, heads = a.n[2], gq = heads / a.n[3];
    long lo, nk, per;
    ds_span(a, lo, nk, per);
    const int ns = int((nk + per - 1) / per);
    const ulong step = ulong(kDsRows) * kDsPart;
    device const float* in = P<float>(a, 4) + ulong(h / gq) * kDsSplits * step + ulong(t * gq + h % gq) * kDsPart;
    float m = -INFINITY;
    for (int s = 0; s < ns; ++s) m = max(m, in[s * step + 128]);
    float l = 0, acc = 0;
    for (int s = 0; s < ns; ++s) {
        const float ms = in[s * step + 128];
        const float w = ms == -INFINITY ? 0.0f : exp(ms - m);
        l += w * in[s * step + 129];
        acc += w * in[s * step + d];
    }
    P<float>(a, 6)[(ulong(t) * heads + h) * 128 + d] = acc / l;
}

// The drafter's elementwise steps, n[0] the kind; stride the count.
// 0 ds_swiglu: g = silu(g) * u (p g, u).
// 1 ds_block_in: x's rows 1 .. rows - 1 the mask, ctx the ring row before the block, state zeros, *prev = *token
//   (p x, mask, at, token, state, prev, ring, ctx; n: kind, n, rows, rank, ring rows; pos).
// 2 ds_markov_in: in = [state ; BF16 row *prev of w1] (p state, w1, prev, in; n[1] rank).
// 3 ds_markov_step (p ctx_h, ctx_f, bias, sp, state, latent; n[1] rank).
kernel void ds_elem(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.stride) return;
    switch (a.n[0]) {
        case 0: {
            device float* gv = P<float>(a, 0);
            gv[i] = gv[i] / (1.0f + exp(-gv[i])) * P<float>(a, 1)[i];
            break;
        }
        case 1: {
            const int n = a.n[1];
            for (int r = 1; r < a.n[2]; ++r) P<float>(a, 0)[ulong(r) * n + i] = P<float>(a, 1)[i];
            const long p0 = ds_pos(P<long>(a, 2), a.pos, 0);
            P<float>(a, 7)[i] = P<float>(a, 6)[ulong((p0 - 1) % a.n[4]) * n + i];
            if (int(i) < a.n[3]) P<float>(a, 4)[i] = 0.0f;
            if (i == 0) P<int>(a, 5)[0] = int(P<long>(a, 3)[0]);
            break;
        }
        case 2: {
            const int rank = a.n[1];
            P<float>(a, 3)[i] = P<float>(a, 0)[i];
            P<float>(a, 3)[rank + i] = as_type<float>(uint(P<ushort>(a, 1)[ulong(P<int>(a, 2)[0]) * rank + i]) << 16);
            break;
        }
        case 3: {
            const int rank = a.n[1];
            device const float* ch = P<float>(a, 0);
            device const float* cf = P<float>(a, 1);
            device const float* b = P<float>(a, 2);
            device const float* sp = P<float>(a, 3);
            const float gate = sigm(ch[i] + cf[i] + b[i] + sp[i]);
            const int u = rank + i, l = 2 * rank + i;
            device float* state = P<float>(a, 4);
            state[i] = gate * state[i] + (1.0f - gate) * precise::tanh(ch[u] + cf[u] + b[u] + sp[u]);
            P<float>(a, 5)[i] = precise::tanh(ch[l] + cf[l] + b[l] + sp[l]);
            break;
        }
    }
}

// ds_pick(): each threadgroup (of 256 threads) a slice of the vocabulary, its scores cap tanh((base + fy + fb) / cap):
// the largest (the lowest id of equals) and the softmax sum against it into part (m, s, id); then ds_pick_join.
// p: base, fy, fb, part, out, prev; n: n_keep, parts; f[0] cap.
struct DsPart {
    float m, s;
    int id;
};
inline void ds_join_part(thread float& m, thread float& s, thread int& id, float m2, float s2, int id2) {
    if (m2 > m || (m2 == m && id2 < id)) {
        s = (m == -INFINITY ? 0.0f : s * exp(m - m2)) + s2;
        m = m2;
        id = id2;
    } else if (m2 != -INFINITY) {
        s += s2 * exp(m2 - m);
    }
}
kernel void ds_pick_part(constant Args& a [[buffer(0)]], uint blk [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float bm[8], bs[8];
    threadgroup int bi[8];
    const int n_keep = a.n[0], parts = a.n[1], per = (n_keep + parts - 1) / parts, v0 = int(blk) * per,
              v1 = min(n_keep, v0 + per);
    const float cap = a.f[0];
    float m = -INFINITY, s = 0;
    int id = 0x7fffffff;
    for (int v = v0 + int(tid); v < v1; v += 256) {
        const float z = cap * precise::tanh((P<float>(a, 0)[v] + P<float>(a, 1)[v] + P<float>(a, 2)[v]) / cap);
        if (z > m) {
            s = s * exp(m - z) + 1.0f;
            m = z;
            id = v;
        } else {
            s += exp(z - m);
        }
    }
    for (ushort o = 16; o > 0; o >>= 1) {
        const float m2 = simd_shuffle_xor(m, o), s2 = simd_shuffle_xor(s, o);
        const int id2 = simd_shuffle_xor(id, o);
        ds_join_part(m, s, id, m2, s2, id2);
    }
    if ((tid & 31) == 0) bm[tid >> 5] = m, bs[tid >> 5] = s, bi[tid >> 5] = id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    for (int k = 1; k < 8; ++k) ds_join_part(m, s, id, bm[k], bs[k], bi[k]);
    device DsPart* part = reinterpret_cast<device DsPart*>(a.p[3]);
    part[blk].m = m;
    part[blk].s = s;
    part[blk].id = id;
}
kernel void ds_pick_join(constant Args& a [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    device const DsPart* part = reinterpret_cast<device const DsPart*>(a.p[3]);
    float m = -INFINITY, s = 0;
    int id = 0x7fffffff;
    for (int k = 0; k < a.n[1]; ++k) ds_join_part(m, s, id, part[k].m, part[k].s, part[k].id);
    P<int>(a, 4)[0] = id;
    P<float>(a, 4)[1] = 1.0f / s;
    P<int>(a, 5)[0] = id;
}

// ds_fold_q8(): a row a thread, Q8_0 block b of each row from the 32 rows of ft (ft[k][v]).  p: ft, fold; n: n_rows,
// blocks, b.
kernel void ds_fold_q8(constant Args& a [[buffer(0)]], uint v [[thread_position_in_grid]]) {
    const int n_rows = a.n[0];
    if (v >= uint(n_rows)) return;
    float x[32], amax = 0;
    for (int k = 0; k < 32; ++k) {
        x[k] = P<float>(a, 0)[ulong(k) * n_rows + v];
        amax = max(amax, abs(x[k]));
    }
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    device uchar* blk = P<uchar>(a, 1) + (ulong(v) * a.n[1] + a.n[2]) * 34;
    *reinterpret_cast<device half*>(blk) = half(d);
    for (int k = 0; k < 32; ++k) blk[2 + k] = uchar(char(round(x[k] * id)));
}
