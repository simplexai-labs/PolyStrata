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

// Broadcast rsqrt(sum(ss)/n + eps) to all threads with a single barrier (identical reduction order to tg_sum).
inline float tg_rsqrt(float ss, threadgroup float* part, threadgroup float& out_inv, int n, float eps, uint tid, uint nt) {
    for (uint k = 16; k; k >>= 1) ss += simd_shuffle_xor(ss, k);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint k = 0; k < nt / 32; ++k) s += part[k];
        out_inv = rsqrt(s / n + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return out_inv;
}

// rms_norm() of a row as poly_norm makes it (y = x * rsqrt(mean(x^2) + eps) * w), with q8 (null: none) its q8_1
// blocks; x may be y.  The whole threadgroup takes part; n a multiple of 32 where q8 is set.
inline void norm_row(device const float* x, device const float* w, device float* y, device uchar* q8, int n, float eps,
                     threadgroup float* part, threadgroup float& s_inv, uint tid, uint nt) {
    float ss = 0;
    for (int i = tid; i < n; i += nt) ss += x[i] * x[i];
    const float inv = tg_rsqrt(ss, part, s_inv, n, eps, tid, nt);
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
    if (tid < 32) {
        const uint lane = tid;
        const float sc = 1.0f / sqrt(float(n));
        float my_inv = 0.0f;
        if (int(lane) < ns) {
            float ss = 0.0f;
            for (uint w = 0; w < nt / 32; ++w) ss += part[lane * 5][w];
            my_inv = 1.0f / sqrt(ss / float(n) + 1e-6f);
        }
        const int sj_s = int(lane) % 6, sj_j = int(lane) / 6;
        float my_dot = 0.0f;
        if (sj_j < nq && sj_s < ns) {
            for (uint w = 0; w < nt / 32; ++w) my_dot += part[sj_s * 5 + 1 + sj_j][w];
        }
        float vs[6], invs[6];
        for (int s = 0; s < 6; ++s) {
            invs[s] = simd_shuffle(my_inv, ushort(s));
            const int src = (int(lane) < nq ? int(lane) * 6 : 0) + s;
            vs[s] = simd_shuffle(my_dot, ushort(src)) * invs[s] * sc;
        }
        if (int(lane) < nq) {
            const int j = int(lane);
            float m = -INFINITY;
            for (int s = 0; s < ns; ++s) m = max(m, vs[s]);
            float total = 0.0f;
            for (int s = 0; s < ns; ++s) {
                vs[s] = exp(vs[s] - m);
                total += vs[s];
            }
            for (int s = 0; s < ns; ++s) coef[s][j] = vs[s] / total * invs[s];
        }
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
    const float inv = tg_rsqrt(ss, npart, npart[0], n, a.f[0], tid, nt);
    device const float* w = P<float>(a, 66);
    device uchar* q8 = a.p[68] ? P<uchar>(a, 68) + ulong(g.x) * ulong(n / 32) * 36 : nullptr;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        const float v = o0[h] * inv * (w ? w[i] : 1.0f);
        P<float>(a, 67)[row + i] = v;
        if (q8) q8_put(q8, i, v);
    }
}

// ink_norm_many_q8_rows(): row (grid x), job j (grid y) p[j] x, p[3 + j] w, p[6 + j] y, p[9 + j] q8; n[0] n; f[0] eps.
// With n[1] == 1: ink_gate_norm_q8_rows() (row = grid x, job 0): p[12] msg, p[13] gain, p[14] forget, p[15] out.
kernel void ink_norm_q8(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[32];
    threadgroup float s_inv;
    const int j = g.y, n = a.n[0];
    const ulong off = ulong(g.x) * n;
    const ulong qoff = ulong(g.x) * ulong(n / 32) * 36;
    if (a.n[1] == 2) {
        device float* x = P<float>(a, 0) + off;
        device const float* add = P<float>(a, 12) + off;
        for (int i = tid; i < n; i += nt) x[i] += add[i];
        norm_row(x, P<float>(a, 3), P<float>(a, 6) + off, a.p[9] ? P<uchar>(a, 9) + qoff : nullptr, n, a.f[0], part,
                 s_inv, tid, nt);
        return;
    }
    if (a.n[1]) {
        device const float* cur = P<float>(a, 0) + off;
        device float* out = P<float>(a, 15) + off;
        device const float* msg = P<float>(a, 12) + off;
        device const float* gain = P<float>(a, 13) + off;
        device const float* forget = P<float>(a, 14) + off;
        for (int i = tid; i < n; i += nt) {
            const float c = cur[i];
            out[i] = c - precise::tanh(forget[i]) * c + precise::tanh(gain[i]) * msg[i];
        }
        norm_row(out, P<float>(a, 3), P<float>(a, 6) + off, a.p[9] ? P<uchar>(a, 9) + qoff : nullptr, n, a.f[0], part,
                 s_inv, tid, nt);
        return;
    }
    norm_row(P<float>(a, j) + off, P<float>(a, 3 + j), P<float>(a, 6 + j) + off,
             a.p[9 + j] ? P<uchar>(a, 9 + j) + qoff : nullptr, n, a.f[0], part, s_inv, tid, nt);
}

template <bool BF>
inline float wval_t(constant Args& a, int k, ulong i) {
    return BF ? as_type<float>(uint(P<ushort>(a, k)[i]) << 16) : P<float>(a, k)[i];
}

// ink_ctrl_down(): controller j's outputs o .. o + O - 1 (threadgroup x = (j * rank + o) / O, O = n[3], 8 at most) of
// row y: [rms(cur) | rms(msg)] (eps 1e-6, no weight) times its down matrix's rows (2 n values, F32 or BF16).  p[j]
// cur, p[3 + j] msg, p[6 + j] down, p[9 + j] ch; n: n_embd, rank, bf16, O.  Each thread keeps its normed inputs (up to
// 24) for the O rows.
template <bool BF>
kernel void ink_ctrl_down_t(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    constexpr int H = 24;
    const uint nt = shape.x;
    threadgroup float part[32], dots[8][32];
    const int n = a.n[0], rank = a.n[1], O = a.n[3], j = (g.x * O) / rank, o0 = (g.x * O) % rank;
    const ulong row = ulong(g.y) * n;
    device const float* cur = P<float>(a, j) + row;
    device const float* msg = P<float>(a, 3 + j) + row;
    if (n == 2304 && nt == 256) {
        float xs[18];
        float sa = 0, sb = 0;
        for (int h = 0; h < 9; ++h) {
            const float v = cur[tid + h * 256];
            xs[h] = v;
            sa += v * v;
        }
        for (int h = 0; h < 9; ++h) {
            const float v = msg[tid + h * 256];
            xs[9 + h] = v;
            sb += v * v;
        }
        const float ia = 1.0f / sqrt(tg_sum(sa, part, tid, 256) / 2304.0f + 1e-6f);
        const float ib = 1.0f / sqrt(tg_sum(sb, part, tid, 256) / 2304.0f + 1e-6f);
        for (int h = 0; h < 9; ++h) xs[h] *= ia;
        for (int h = 9; h < 18; ++h) xs[h] *= ib;
        for (int k = 0; k < O; ++k) {
            if (o0 + k >= rank) break;
            const ulong w0 = ulong(o0 + k) * 4608 + tid;
            float acc = 0;
            for (int h = 0; h < 18; ++h) acc += wval_t<BF>(a, 6 + j, w0 + h * 256) * xs[h];
            acc = simd_sum(acc);
            if ((tid & 31) == 0) dots[k][tid >> 5] = acc;
        }
    } else {
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
                acc += wval_t<BF>(a, 6 + j, w0 + i) * xs[h];
            }
            acc = simd_sum(acc);
            if ((tid & 31) == 0) dots[k][tid >> 5] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < O && o0 + int(tid) < rank) {
        float acc = 0;
        for (uint w = 0; w < nt / 32; ++w) acc += dots[tid][w];
        P<float>(a, 9 + j)[ulong(g.y) * rank + o0 + tid] = acc;
    }
}
template [[host_name("ink_ctrl_down_f32")]] kernel decltype(ink_ctrl_down_t<false>) ink_ctrl_down_t<false>;
template [[host_name("ink_ctrl_down_bf16")]] kernel decltype(ink_ctrl_down_t<true>) ink_ctrl_down_t<true>;

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
// rank, count, bf16.  A SIMD group a value, 8 a threadgroup, one row per grid.y.
// count == 1 uses xor-butterfly reduction to match ink_gemv_set()'s tg_sum() bit-for-bit; count > 1 uses simd_sum()
// to match the existing control3 path bit-for-bit.
template <bool BF>
kernel void ink_control3_gate_t(constant Args& a [[buffer(0)]], uint2 grp [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int n = a.n[0], rank = a.n[1], count = a.n[2];
    const ulong roff = ulong(grp.y) * n, rch = ulong(grp.y) * rank;
    threadgroup float sch[3][128];
    if (int(tid) < rank) {
        for (int j = 0; j < count; ++j) sch[j][tid] = P<float>(a, 6 + j)[rch + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int i = int(grp.x) * 8 + int(sg);
    if (i >= n) return;
    float g[3], f[3];
    for (int j = 0; j < 3; ++j) {
        if (j >= count) break;
        float ga = 0, fa = 0;
        for (int k = lane; k < rank; k += 32) {
            const float ck = sch[j][k];
            ga += wval_t<BF>(a, 9 + j, ulong(i) * rank + k) * ck;
            fa += wval_t<BF>(a, 12 + j, ulong(i) * rank + k) * ck;
        }
        if (count == 1) {
            for (uint o = 16; o; o >>= 1) {
                ga += simd_shuffle_xor(ga, o);
                fa += simd_shuffle_xor(fa, o);
            }
            g[0] = ga;
            f[0] = fa;
        } else {
            g[j] = simd_sum(ga);
            f[j] = simd_sum(fa);
        }
    }
    if (lane) return;
    float c[3];
    for (int j = 0; j < 3; ++j)
        if (j < count) c[j] = P<float>(a, j)[roff + i];
    for (int j = 0; j < 3; ++j)
        if (j < count) P<float>(a, 15 + j)[roff + i] = gate_value(c[j], P<float>(a, 3 + j)[roff + i], g[j], f[j]);
}
template [[host_name("ink_control3_gate_f32")]] kernel decltype(ink_control3_gate_t<false>) ink_control3_gate_t<false>;
template [[host_name("ink_control3_gate_bf16")]] kernel decltype(ink_control3_gate_t<true>) ink_control3_gate_t<true>;

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

// Job j (g.y = 0 or 1): p[3*j + 0..2] x, state, snap; n: channels, taps, rows0, bits, rows1; stride.
kernel void ink_conv_tail(constant Args& a [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= uint(a.n[0])) return;
    const int j = int(g.y);
    const int rows = j == 0 ? a.n[2] : a.n[4];
    conv_states(P<float>(a, 3 * j), P<float>(a, 3 * j + 1), P<float>(a, 3 * j + 2), a.stride, uint(a.n[3]), a.n[0],
                a.n[1], rows, int(g.x));
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

// Multi-row (1..8) ink_moe_conv_add with optional conv snapshots (eliminates memset blit + moe_sum + ink_conv_few).
kernel void ink_moe_conv_add_rows(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    const int used = a.n[0], channels = a.n[1], taps = a.n[2], rows = a.n[3];
    if (i >= uint(channels)) return;
    device const float* x = P<float>(a, 0);
    device const float* w = P<float>(a, 1);
    device const float* scale = P<float>(a, 2);
    device const int* ids = P<int>(a, 3);
    device float* fo = P<float>(a, 4);
    device float* state = P<float>(a, 5);
    device const float* cw = P<float>(a, 6);
    device float* out = P<float>(a, 7);
    device float* acc = P<float>(a, 8);
    for (int t = 0; t < rows; ++t) {
        float s = 0;
        const int base = t * used;
        for (int k = 0; k < used; ++k) {
            const float wk = scale ? w[base + k] * scale[ids[base + k]] : w[base + k];
            s += wk * x[ulong(base + k) * channels + i];
        }
        const ulong idx = ulong(t) * channels + i;
        fo[idx] = 0.0f + s;
        const float y = conv_value(fo, state, cw, nullptr, channels, 1, taps, t, int(i), a.f[1]);
        out[idx] = y;
        acc[idx] = acc[idx] + y * a.f[0];
    }
    conv_states(fo, state, P<float>(a, 9), a.stride, uint(a.n[4]), channels, taps, rows, int(i));
}

// Multi-row (1..8) MoE weighted sum into fo (grid {blocks(rows * channels, 64), 1, 1}, 64 threads).
kernel void ink_moe_sum_rows(constant Args& a [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
    const int used = a.n[0], channels = a.n[1], rows = a.n[2];
    if (idx >= uint(rows * channels)) return;
    const int t = int(idx / uint(channels)), i = int(idx % uint(channels));
    device const float* x = P<float>(a, 0);
    device const float* w = P<float>(a, 1);
    device const float* scale = P<float>(a, 2);
    device const int* ids = P<int>(a, 3);
    float s = 0.0f;
    const int base = t * used;
    for (int k = 0; k < used; ++k) {
        const float wk = scale ? w[base + k] * scale[ids[base + k]] : w[base + k];
        s += wk * x[ulong(base + k) * channels + i];
    }
    P<float>(a, 4)[idx] = 0.0f + s;
}

// Shared finish for ink_conv_mix, ink_conv_mix_rows and ink_moe_conv_mix (identical arithmetic to ink_mix).
template <int H, int MQ>
inline void ink_mix_finish(constant Args& a, thread float (&x)[H][6], thread float (&sum)[6][MQ + 1],
                           threadgroup float (&part)[6 * (MQ + 1)][32], threadgroup float (&coef)[6][MQ],
                           threadgroup float (&npart)[32], int ns, int nq, int n, int t, uint tid, uint nt) {
    constexpr int K = MQ + 1;
    const ulong row = ulong(t) * n;
    for (int s = 0; s < 6; ++s)
        for (int k = 0; k < K; ++k)
            if (s < ns && k <= nq) {
                const float w = simd_sum(sum[s][k]);
                if ((tid & 31) == 0) part[s * K + k][tid >> 5] = w;
            }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        const uint lane = tid;
        const float sc = 1.0f / sqrt(float(n));
        float my_inv = 0.0f;
        if (int(lane) < ns) {
            float ss = 0.0f;
            for (uint w = 0; w < nt / 32; ++w) ss += part[lane * K][w];
            my_inv = 1.0f / sqrt(ss / float(n) + 1e-6f);
        }
        const int sj_s = int(lane) % 6, sj_j = int(lane) / 6;
        float my_dot = 0.0f;
        if (sj_j < nq && sj_s < ns) {
            for (uint w = 0; w < nt / 32; ++w) my_dot += part[sj_s * K + 1 + sj_j][w];
        }
        float vs[6], invs[6];
        for (int s = 0; s < 6; ++s) {
            invs[s] = simd_shuffle(my_inv, ushort(s));
            const int src = (int(lane) < nq ? int(lane) * 6 : 0) + s;
            vs[s] = simd_shuffle(my_dot, ushort(src)) * invs[s] * sc;
        }
        if (int(lane) < nq) {
            const int j = int(lane);
            float m = -INFINITY;
            for (int s = 0; s < ns; ++s) m = max(m, vs[s]);
            float total = 0.0f;
            for (int s = 0; s < ns; ++s) {
                vs[s] = exp(vs[s] - m);
                total += vs[s];
            }
            for (int s = 0; s < ns; ++s) coef[s][j] = vs[s] / total * invs[s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o0[H], ss = 0;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        for (int j = 0; j < nq; ++j) {
            float acc = 0;
            for (int s = 0; s < ns; ++s) acc += coef[s][j] * x[h][s];
            P<float>(a, 7 + j)[row + i] = acc;
            if (j == 0) o0[h] = acc, ss += acc * acc;
        }
    }
    if (!a.p[12]) return;
    const float inv = tg_rsqrt(ss, npart, npart[0], n, a.f[0], tid, nt);
    device const float* w = P<float>(a, 11);
    device uchar* q8 = a.p[13] ? P<uchar>(a, 13) + ulong(t) * ulong(n / 32) * 36 : nullptr;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        const float v = o0[h] * inv * (w ? w[i] : 1.0f);
        P<float>(a, 12)[row + i] = v;
        if (q8) q8_put(q8, i, v);
    }
}

// ink_conv_mix(): one token's ink_conv_add() + ink_mix_norm(), bit-identical, specialized on (H, MQ) to avoid register spill.
template <int H, int MQ>
kernel void ink_conv_mix_t(constant Args& a [[buffer(0)]], uint tid [[thread_index_in_threadgroup]],
                           uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[6 * (MQ + 1)][32];
    threadgroup float coef[6][MQ];
    threadgroup float npart[32];
    const int ns = a.n[0], nq = a.n[1], n = a.n[2], taps = a.n[3], mode = a.n[4];
    const float s_scale = a.f[1], cnorm = a.f[2];
    device const float* q = P<float>(a, 6);
    device const float* cx = P<float>(a, 14);
    device float* state = P<float>(a, 15);
    device const float* cw = P<float>(a, 16);
    device float* ao = P<float>(a, 17);
    device float* acc = P<float>(a, 18);
    device float* keep = P<float>(a, 19);
    float x[H][6], sum[6][MQ + 1];
    for (int s = 0; s < 6; ++s)
        for (int k = 0; k < MQ + 1; ++k) sum[s][k] = 0;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        const float x0 = cx[i];
        float cacc = 0;
        for (int j = 0; j < taps - 1; ++j) cacc += state[ulong(j) * n + i] * cw[ulong(i) * taps + j];
        cacc += x0 * cw[ulong(i) * taps + (taps - 1)];
        const float y = (x0 + cacc) * cnorm;
        ao[i] = y;
        for (int k = 0; k < taps - 2; ++k) state[ulong(k) * n + i] = state[ulong(k + 1) * n + i];
        if (taps >= 2) state[ulong(taps - 2) * n + i] = x0;
        const float old_r = acc[i];
        float new_r;
        if (mode == 0) {
            new_r = old_r + y * s_scale;
        } else {
            if (mode == 1) keep[i] = old_r;
            new_r = y;
        }
        acc[i] = new_r;
        float qv[MQ];
        for (int j = 0; j < MQ; ++j) qv[j] = j < nq ? q[ulong(j) * n + i] : 0.0f;
        for (int s = 0; s < 6; ++s) {
            if (s < ns) {
                const float v = (s == ns - 1) ? new_r : (mode == 1 && s == ns - 2) ? old_r : P<float>(a, s)[i];
                x[h][s] = v;
                sum[s][0] += v * v;
                for (int j = 0; j < MQ; ++j) sum[s][1 + j] += v * qv[j];
            }
        }
    }
    ink_mix_finish<H, MQ>(a, x, sum, part, coef, npart, ns, nq, n, 0, tid, nt);
}
template [[host_name("ink_conv_mix_3_1")]] kernel decltype(ink_conv_mix_t<3, 1>) ink_conv_mix_t<3, 1>;
template [[host_name("ink_conv_mix_3_4")]] kernel decltype(ink_conv_mix_t<3, 4>) ink_conv_mix_t<3, 4>;
template [[host_name("ink_conv_mix_4_4")]] kernel decltype(ink_conv_mix_t<4, 4>) ink_conv_mix_t<4, 4>;

// Multi-row ink_conv_mix (grid {rows, 1, 1}): reads state read-only so all rows execute concurrently without race;
// caller updates state + snapshots afterwards via ink_conv_state_update (overlapped with router GEMV).
template <int H, int MQ>
kernel void ink_conv_mix_rows_t(constant Args& a [[buffer(0)]], uint grp [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    const int t = int(grp);
    threadgroup float part[6 * (MQ + 1)][32];
    threadgroup float coef[6][MQ];
    threadgroup float npart[32];
    const int ns = a.n[0], nq = a.n[1], n = a.n[2], taps = a.n[3], mode = a.n[4];
    const float s_scale = a.f[1], cnorm = a.f[2];
    const ulong row = ulong(t) * n;
    device const float* q = P<float>(a, 6);
    device const float* cx = P<float>(a, 14);
    device const float* state = P<float>(a, 15);
    device const float* cw = P<float>(a, 16);
    device float* ao = P<float>(a, 17);
    device float* acc = P<float>(a, 18);
    device float* keep = P<float>(a, 19);
    float x[H][6], sum[6][MQ + 1];
    for (int s = 0; s < 6; ++s)
        for (int k = 0; k < MQ + 1; ++k) sum[s][k] = 0;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        const float y = conv_value(cx, state, cw, nullptr, n, 1, taps, t, i, cnorm);
        ao[row + i] = y;
        const float old_r = acc[row + i];
        float new_r;
        if (mode == 0) {
            new_r = old_r + y * s_scale;
        } else {
            if (mode == 1) keep[row + i] = old_r;
            new_r = y;
        }
        acc[row + i] = new_r;
        float qv[MQ];
        for (int j = 0; j < MQ; ++j) qv[j] = j < nq ? q[ulong(j) * n + i] : 0.0f;
        for (int s = 0; s < 6; ++s) {
            if (s < ns) {
                const float v = (s == ns - 1) ? new_r : (mode == 1 && s == ns - 2) ? old_r : P<float>(a, s)[row + i];
                x[h][s] = v;
                sum[s][0] += v * v;
                for (int j = 0; j < MQ; ++j) sum[s][1 + j] += v * qv[j];
            }
        }
    }
    ink_mix_finish<H, MQ>(a, x, sum, part, coef, npart, ns, nq, n, t, tid, nt);
}
template [[host_name("ink_conv_mix_rows_3_1")]] kernel decltype(ink_conv_mix_rows_t<3, 1>) ink_conv_mix_rows_t<3, 1>;
template [[host_name("ink_conv_mix_rows_3_4")]] kernel decltype(ink_conv_mix_rows_t<3, 4>) ink_conv_mix_rows_t<3, 4>;
template [[host_name("ink_conv_mix_rows_4_4")]] kernel decltype(ink_conv_mix_rows_t<4, 4>) ink_conv_mix_rows_t<4, 4>;

// ink_moe_conv_mix(): one token's ink_moe_conv_add() + next layer's ink_mix_norm(), bit-identical.
template <int H, int MQ>
kernel void ink_moe_conv_mix_t(constant Args& a [[buffer(0)]], uint tid [[thread_index_in_threadgroup]],
                               uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[6 * (MQ + 1)][32];
    threadgroup float coef[6][MQ];
    threadgroup float npart[32];
    const int ns = a.n[0], nq = a.n[1], n = a.n[2], taps = a.n[3], used = a.n[4];
    const float s_scale = a.f[1], cnorm = a.f[2];
    device const float* q = P<float>(a, 6);
    device const float* mx = P<float>(a, 14);
    device const float* mw = P<float>(a, 15);
    device const float* mscale = P<float>(a, 16);
    device const int* mids = P<int>(a, 17);
    device float* fo = P<float>(a, 18);
    device float* state = P<float>(a, 19);
    device const float* cw = P<float>(a, 20);
    device float* fc = P<float>(a, 21);
    device float* acc = P<float>(a, 22);
    float x[H][6], sum[6][MQ + 1];
    for (int s = 0; s < 6; ++s)
        for (int k = 0; k < MQ + 1; ++k) sum[s][k] = 0;
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * nt;
        if (i >= n) break;
        float ms = 0;
        for (int k = 0; k < used; ++k) {
            const float wk = mscale ? mw[k] * mscale[mids[k]] : mw[k];
            ms += wk * mx[ulong(k) * n + i];
        }
        const float f0 = 0.0f + ms;
        fo[i] = f0;
        float cacc = 0;
        for (int j = 0; j < taps - 1; ++j) cacc += state[ulong(j) * n + i] * cw[ulong(i) * taps + j];
        cacc += f0 * cw[ulong(i) * taps + (taps - 1)];
        const float y = (f0 + cacc) * cnorm;
        fc[i] = y;
        for (int k = 0; k < taps - 2; ++k) state[ulong(k) * n + i] = state[ulong(k + 1) * n + i];
        if (taps >= 2) state[ulong(taps - 2) * n + i] = f0;
        const float new_r = acc[i] + y * s_scale;
        acc[i] = new_r;
        float qv[MQ];
        for (int j = 0; j < MQ; ++j) qv[j] = j < nq ? q[ulong(j) * n + i] : 0.0f;
        for (int s = 0; s < 6; ++s) {
            if (s < ns) {
                const float v = (s == ns - 1) ? new_r : P<float>(a, s)[i];
                x[h][s] = v;
                sum[s][0] += v * v;
                for (int j = 0; j < MQ; ++j) sum[s][1 + j] += v * qv[j];
            }
        }
    }
    ink_mix_finish<H, MQ>(a, x, sum, part, coef, npart, ns, nq, n, 0, tid, nt);
}
template [[host_name("ink_moe_conv_mix_3_1")]] kernel decltype(ink_moe_conv_mix_t<3, 1>) ink_moe_conv_mix_t<3, 1>;
template [[host_name("ink_moe_conv_mix_3_4")]] kernel decltype(ink_moe_conv_mix_t<3, 4>) ink_moe_conv_mix_t<3, 4>;
template [[host_name("ink_moe_conv_mix_4_4")]] kernel decltype(ink_moe_conv_mix_t<4, 4>) ink_moe_conv_mix_t<4, 4>;

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

// ink_qkv_win(): fused window (2..8 rows of one sequence) K/V conv + snapshots + Q/K/V/R clips + KV cache store.
// Grid: {2 * rows * n_head + 2 * n_kv, 1, 1}, threads 128.  p[0..16] match ink_qkv, p[17] ksnap, p[18] vsnap;
// n[6] rows, n[7] snap_bits; stride snap_stride.  Bit-identical to ink_conv_few + ink_clips.
kernel void ink_qkv_win(constant Args& a [[buffer(0)]], uint grp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    const int n_head = a.n[0], n_kv = a.n[1], dim = a.n[2], n_rel = a.n[3], taps = a.n[4], ring = a.n[5],
              rows = a.n[6];
    device const long* at = P<long>(a, 16);
    int b = int(grp);
    if (b < rows * n_head) {
        const int row = b / n_head, head = b % n_head;
        const long p = at ? at[2 * row] : a.pos + row;
        device const float* sc = P<float>(a, 1);
        clip_head(P<float>(a, 0) + (ulong(row) * n_head + head) * dim, sc ? sc + ulong(head) * dim : nullptr, a.f[0],
                  dim, a.f[2], a.f[3], p, part, tid, nt);
        return;
    }
    b -= rows * n_head;
    if (b < 2 * n_kv) {
        const bool is_v = b >= n_kv;
        const int h = is_v ? b - n_kv : b, channels = n_kv * dim;
        device const float* x = P<float>(a, is_v ? 5 : 4);
        device float* state = P<float>(a, is_v ? 7 : 6);
        device const float* cw = P<float>(a, is_v ? 9 : 8);
        device const float* dyn = P<float>(a, is_v ? 11 : 10);
        device float* out = P<float>(a, is_v ? 13 : 12);
        device half* cache = P<half>(a, is_v ? 15 : 14);
        device float* snap = P<float>(a, is_v ? 18 : 17);
        const float norm = a.f[is_v ? 6 : 5];
        const int c = h * dim + int(tid);
        for (int t = 0; t < rows; ++t) {
            float y = 0.0f;
            if (int(tid) < dim) y = conv_value(x, state, cw, dyn, channels, n_kv, taps, t, c, norm);
            const float ss = int(tid) < dim ? y * y : 0.0f;
            float f;
            if (is_v) f = 1.0f / sqrt(tg_sum(ss, part, tid, nt) / float(dim) + a.f[4]);
            else f = min(a.f[0] / (sqrt(tg_sum(ss, part, tid, nt)) + 1e-6f), 1.0f);
            if (int(tid) < dim) {
                const float v = y * f;
                out[ulong(t) * channels + c] = v;
                long cell = (at ? at[0] : a.pos) + t;
                if (ring > 0) cell %= ring;
                cache[(ulong(cell) * n_kv + h) * dim + tid] = half(v);
            }
        }
        if (int(tid) < dim) conv_states(x, state, snap, a.stride, uint(a.n[7]), channels, taps, rows, c);
        return;
    }
    b -= 2 * n_kv;
    const int row = b / n_head, head = b % n_head;
    const long p = at ? at[2 * row] : a.pos + row;
    device const float* rs = P<float>(a, 3);
    clip_head(P<float>(a, 2) + (ulong(row) * n_head + head) * n_rel, rs ? rs + ulong(head) * n_rel : nullptr, a.f[1],
              n_rel, a.f[2], a.f[3], p, part, tid, nt);
}

// ---- elementwise: n[0] the kind, stride the count ----
// 0: x *= sigmoid(gate) (p x, gate); 1: an fp16 ring's rows (p x, cache; n[1] n, n[2] ring; pos); 2: the soft cap
// (p logits; n[1] n_keep, n[2] n_all; f[0] cap); 3: LoopCD-Hidden extrapolation (p dst, ref; f[0] omega)
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
        case 3: {
            device float* d = P<float>(a, 0);
            const float v = d[i];
            d[i] = v + a.f[0] * (v - P<float>(a, 1)[i]);
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

// Single- and multi-token (up to 8 rows) fused attention + split join for dim=128 when splits * sub_groups <= 32:
// one threadgroup per (query head, row) (grid n_head x rows), bit-identical to ink_attn_128 + ink_attn_join.
kernel void ink_attn_128_fused(constant Args& a [[buffer(0)]], threadgroup float* sh [[threadgroup(0)]],
                               uint2 grp [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int PER = 16;
    const int h = int(grp.x), t = int(grp.y);
    const int nh = a.n[1], nkv = a.n[2], dim = a.n[3], nrel = a.n[4], extent = a.n[5], window = a.n[6],
              ring = a.n[7], splits = a.n[8], sub_groups = a.n[9];
    const int per = nh / nkv;
    const int g = h / per;
    const ulong head = ulong(t) * nh + ulong(h);
    const int z = int(sg) / sub_groups, sub_sg = int(sg) % sub_groups;
    const long p = ink_pos(a, t);
    long lo = window > 0 ? max(0L, p - window + 1) : 0, hi = p;
    const long span = (hi - lo + splits) / splits, first = lo + long(z) * span;
    hi = min(hi, first + span - 1);
    lo = first;
    const uint slot = lane >> 3, l8 = lane & 7;
    float q[PER], r[4];
    for (int c = 0; c < PER; ++c) q[c] = P<float>(a, 0)[head * dim + l8 * PER + c] * a.f[0];
    for (int c = 0; c < 4; ++c) {
        const int k = l8 * 4 + c;
        r[c] = k < nrel ? P<float>(a, 1)[head * nrel + k] : 0;
    }
    const float sink = P<float>(a, 2)[head];
    float m = -INFINITY, l = 0, acc[PER];
    for (int c = 0; c < PER; ++c) acc[c] = 0;
    device const half* keys = P<half>(a, 3);
    device const half* vals = P<half>(a, 4);
    device const float* rel = P<float>(a, 5);
    const long step = long(sub_groups) * 4;
    for (long j0 = lo + long(sub_sg) * 4; j0 <= hi; j0 += step) {
        const long j = j0 + slot;
        const bool valid = j <= hi;
        float k[PER], v[PER], rb[4];
        const ulong at = (ulong(ring > 0 ? (valid ? j : hi) % ring : (valid ? j : hi)) * nkv + g) * dim + l8 * PER;
        device const half4* k4 = reinterpret_cast<device const half4*>(keys + at);
        device const half4* v4 = reinterpret_cast<device const half4*>(vals + at);
        for (int c = 0; c < 4; ++c) {
            const float4 kf = float4(k4[c]), vf = float4(v4[c]);
            k[4 * c] = kf.x; k[4 * c + 1] = kf.y; k[4 * c + 2] = kf.z; k[4 * c + 3] = kf.w;
            v[4 * c] = vf.x; v[4 * c + 1] = vf.y; v[4 * c + 2] = vf.z; v[4 * c + 3] = vf.w;
        }
        const long dist = p - j;
        for (int c = 0; c < 4; ++c) {
            const int i = l8 * 4 + c;
            rb[c] = valid && i < nrel && dist < extent ? rel[ulong(dist) * nrel + i] : 0.0f;
        }
        float d = 0;
        for (int c = 0; c < 4; ++c) d += rb[c] * r[c];
        for (int c = 0; c < PER; ++c) d += q[c] * k[c];
        d += simd_shuffle_xor(d, 1);
        d += simd_shuffle_xor(d, 2);
        d += simd_shuffle_xor(d, 4);
        if (!valid) continue;
        d -= sink;
        const float next = max(m, d), keep = exp(m - next), w = exp(d - next);
        l = l * keep + w;
        m = next;
        for (int c = 0; c < PER; ++c) acc[c] = acc[c] * keep + w * v[c];
    }
    float top = max(m, simd_shuffle_xor(m, 8));
    top = max(top, simd_shuffle_xor(top, 16));
    const float f = m == -INFINITY ? 0.0f : exp(m - top);
    float lt = l * f;
    lt += simd_shuffle_xor(lt, 8);
    lt += simd_shuffle_xor(lt, 16);
    const int stride = 2 + dim;
    threadgroup float* o = sh + ulong(sg) * stride;
    if (lane == 0) {
        o[0] = top;
        o[1] = lt;
    }
    if (lt > 0.0f) {
        for (int c = 0; c < PER; ++c) {
            float x = acc[c] * f;
            x += simd_shuffle_xor(x, 8);
            x += simd_shuffle_xor(x, 16);
            if (slot == 0) o[2 + l8 * PER + c] = x;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* runs = sh + ulong(splits * sub_groups) * stride;
    if (int(sg) < splits) {
        const int sz = int(sg);
        float stop = -INFINITY;
        for (int s = 0; s < sub_groups; ++s) stop = max(stop, sh[ulong(sz * sub_groups + s) * stride]);
        float stotal = 0;
        for (int s = 0; s < sub_groups; ++s) {
            threadgroup float* os = sh + ulong(sz * sub_groups + s) * stride;
            if (os[1] > 0) stotal += os[1] * exp(os[0] - stop);
        }
        threadgroup float* run = runs + ulong(sz) * stride;
        if (lane == 0) {
            run[0] = stop;
            run[1] = stotal;
        }
        for (int d = lane; d < dim; d += 32) {
            float sum = 0;
            for (int s = 0; s < sub_groups; ++s) {
                threadgroup float* os = sh + ulong(sz * sub_groups + s) * stride;
                if (os[1] > 0) sum += os[2 + d] * exp(os[0] - stop);
            }
            run[2 + d] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float ftop = 0;
    for (int s = 0; s < splits; ++s) ftop = max(ftop, runs[ulong(s) * stride]);
    float ftotal = exp(-ftop);
    for (int s = 0; s < splits; ++s)
        if (runs[ulong(s) * stride + 1] > 0) ftotal += runs[ulong(s) * stride + 1] * exp(runs[ulong(s) * stride] - ftop);
    device const float* gate = P<float>(a, 6);
    for (int d = lane; d < dim; d += 32) {
        float sum = 0;
        for (int s = 0; s < splits; ++s)
            if (runs[ulong(s) * stride + 1] > 0) sum += runs[ulong(s) * stride + 2 + d] * exp(runs[ulong(s) * stride] - ftop);
        float y = sum / ftotal;
        if (gate) y *= sigm(gate[head * dim + d]);
        P<float>(a, 7)[head * dim + d] = y;
        if (a.p[10]) q8_put(P<uchar>(a, 10), int(head * dim) + d, y);
    }
}

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

// Long-context & SWA-1024 zero-spill vectorized attention with in-TG 8-subgroup reduction:
// grid {n_kv, rows, splits * n_chunks}, threads = 32 * per * 8 (24 SIMD groups for per=3), sh = per * 8 * 132 * 4 B.
// Each TG reduces 8 sub_sgs in threadgroup memory before writing 1 record per (head, row, split, chunk) -> 8x less
// global scratch traffic than ink_attn_128_sg while keeping full 80-core occupancy and branchless online softmax.
kernel void ink_attn_128_sg8(constant Args& a [[buffer(0)]], threadgroup float* sh [[threadgroup(0)]],
                             uint3 grp [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int PER = 16;
    constexpr int kSgStride = 132;
    const int g = int(grp.x), t = int(grp.y);
    const int nh = a.n[1], nkv = a.n[2], nrel = a.n[4], extent = a.n[5], window = a.n[6], ring = a.n[7],
              splits = a.n[8], groups = a.n[9];
    const int n_chunks = groups >> 3;
    const int z = int(grp.z) / n_chunks, chunk = int(grp.z) % n_chunks;
    const int per = nh / nkv;
    const int hp = int(sg) % per, ls = int(sg) / per;
    const int sub_sg = chunk * 8 + ls;
    const ulong head = ulong(t) * nh + ulong(g * per + hp);
    const long p = ink_pos(a, t);
    long lo = window > 0 ? max(0L, p - window + 1) : 0, hi = p;
    if (splits > 1) {
        const long span = (hi - lo + splits) / splits, first = lo + long(z) * span;
        hi = min(hi, first + span - 1);
        lo = first;
    }
    const uint slot = lane >> 3, l8 = lane & 7;
    float q[PER], r[4];
    for (int c = 0; c < PER; ++c) q[c] = P<float>(a, 0)[head * 128 + l8 * PER + c] * a.f[0];
    for (int c = 0; c < 4; ++c) {
        const int k = int(l8) * 4 + c;
        r[c] = k < nrel ? P<float>(a, 1)[head * nrel + k] : 0.0f;
    }
    const float sink = P<float>(a, 2)[head];
    float m = -INFINITY, l = 0.0f, acc[PER];
    for (int c = 0; c < PER; ++c) acc[c] = 0.0f;
    device const half* keys = P<half>(a, 3);
    device const half* vals = P<half>(a, 4);
    device const float* rel = P<float>(a, 5);
    const bool has_rel_lane = int(l8) * 4 < nrel;
    const long step = long(groups) * 4;
    for (long j0 = lo + long(sub_sg) * 4; j0 <= hi; j0 += step) {
        const long j = j0 + slot;
        const bool valid = j <= hi;
        const ulong at = (ulong(ring > 0 ? (valid ? j : hi) % ring : (valid ? j : hi)) * nkv + g) * 128 + l8 * PER;
        device const half4* k4 = reinterpret_cast<device const half4*>(keys + at);
        device const half4* v4 = reinterpret_cast<device const half4*>(vals + at);
        float d = 0.0f;
        const long dist = p - j;
        if (valid && has_rel_lane && dist < extent) {
            for (int c = 0; c < 4; ++c) {
                const int i = int(l8) * 4 + c;
                if (i < nrel) d += rel[ulong(dist) * nrel + i] * r[c];
            }
        }
        for (int c = 0; c < 4; ++c) {
            const float4 kf = float4(k4[c]);
            d += q[4 * c] * kf.x;
            d += q[4 * c + 1] * kf.y;
            d += q[4 * c + 2] * kf.z;
            d += q[4 * c + 3] * kf.w;
        }
        d += simd_shuffle_xor(d, 1);
        d += simd_shuffle_xor(d, 2);
        d += simd_shuffle_xor(d, 4);
        if (!valid) continue;
        d -= sink;
        const float next = max(m, d), keep = exp(m - next), w = exp(d - next);
        l = l * keep + w;
        m = next;
        for (int c = 0; c < 4; ++c) {
            const float4 vf = float4(v4[c]);
            acc[4 * c] = acc[4 * c] * keep + w * vf.x;
            acc[4 * c + 1] = acc[4 * c + 1] * keep + w * vf.y;
            acc[4 * c + 2] = acc[4 * c + 2] * keep + w * vf.z;
            acc[4 * c + 3] = acc[4 * c + 3] * keep + w * vf.w;
        }
    }
    float top = max(m, simd_shuffle_xor(m, 8));
    top = max(top, simd_shuffle_xor(top, 16));
    const float f = m == -INFINITY ? 0.0f : exp(m - top);
    float lt = l * f;
    lt += simd_shuffle_xor(lt, 8);
    lt += simd_shuffle_xor(lt, 16);
    threadgroup float* o = sh + ulong(hp * 8 + ls) * ulong(kSgStride);
    if (lane == 0) {
        o[0] = top;
        o[1] = lt;
    }
    if (lt > 0.0f) {
        for (int c = 0; c < 4; ++c) {
            float4 xv;
            for (int b = 0; b < 4; ++b) {
                float x = acc[4 * c + b] * f;
                x += simd_shuffle_xor(x, 8);
                x += simd_shuffle_xor(x, 16);
                xv[b] = x;
            }
            if (slot == 0) *reinterpret_cast<threadgroup float4*>(o + 4 + l8 * PER + 4 * c) = xv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(sg) >= per) return;
    const int rhp = int(sg);
    const ulong rhead = ulong(t) * nh + ulong(g * per + rhp);
    threadgroup const float* base = sh + ulong(rhp * 8) * ulong(kSgStride);
    float ctop = -INFINITY;
    for (int s = 0; s < 8; ++s) ctop = max(ctop, base[ulong(s) * kSgStride]);
    float ctotal = 0.0f;
    float ew[8];
    for (int s = 0; s < 8; ++s) {
        const float ls_v = base[ulong(s) * kSgStride + 1];
        const float ws = ls_v > 0.0f ? exp(base[ulong(s) * kSgStride] - ctop) : 0.0f;
        ew[s] = ws;
        if (ls_v > 0.0f) ctotal += ls_v * ws;
    }
    device float* dst = P<float>(a, 9) + ((rhead * ulong(splits) + ulong(z)) * ulong(n_chunks) + ulong(chunk)) * ulong(kSgStride);
    if (lane == 0) {
        dst[0] = ctop;
        dst[1] = ctotal;
    }
    if (ctotal > 0.0f) {
        float4 sum4 = float4(0.0f);
        for (int s = 0; s < 8; ++s)
            if (base[ulong(s) * kSgStride + 1] > 0.0f)
                sum4 += *reinterpret_cast<threadgroup const float4*>(base + ulong(s) * kSgStride + 4 + lane * 4) * ew[s];
        *reinterpret_cast<device float4*>(dst + 4 + lane * 4) = sum4;
    }
}

// Fast 1-SIMD-group join for ink_attn_128_sg8 (grid {n_head, rows, 1}, threads = 32, 0 TG mem, 0 barriers):
// joins splits * n_chunks (8 for SWA-1024, 64 for full 6K-64K) float4-aligned records directly into out (+ optional q8).
kernel void ink_attn_join_sg8(constant Args& a [[buffer(0)]], uint2 grp [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    constexpr int stride = 132;
    const int nh = a.n[1], splits = a.n[8], n_chunks = a.n[9] >> 3;
    const int nrec = splits * n_chunks;
    const ulong head = ulong(grp.y) * nh + grp.x;
    device const float* src = P<float>(a, 9) + head * ulong(nrec) * stride;
    float top = 0.0f;
    for (int s = 0; s < nrec; ++s) top = max(top, src[ulong(s) * stride]);
    float total = exp(-top);
    float4 sum4 = float4(0.0f);
    for (int s = 0; s < nrec; ++s) {
        device const float* os = src + ulong(s) * stride;
        const float ls = os[1];
        if (ls > 0.0f) {
            const float ws = exp(os[0] - top);
            total += ls * ws;
            sum4 += *reinterpret_cast<device const float4*>(os + 4 + lane * 4) * ws;
        }
    }
    float4 y4 = sum4 / total;
    device const float* gate = P<float>(a, 6);
    const ulong base = head * 128 + lane * 4;
    if (gate) {
        const float4 g4 = *reinterpret_cast<device const float4*>(gate + base);
        y4 *= float4(sigm(g4.x), sigm(g4.y), sigm(g4.z), sigm(g4.w));
    }
    *reinterpret_cast<device float4*>(P<float>(a, 7) + base) = y4;
    if (a.p[10]) {
        simdgroup_barrier(mem_flags::mem_device);
        device float* out_f = P<float>(a, 7) + head * 128;
        for (int b = 0; b < 4; ++b) {
            const int d = b * 32 + int(lane);
            q8_put(P<uchar>(a, 10), int(head * 128) + d, out_f[d]);
        }
    }
}

// Multi-row long-context (>2K, rows >= 2) zero-spill attention with all rows + per query heads in one TG (1x DRAM
// KV read across all rows when KV cache exceeds SLC), 0 threadgroup memory and 0 barriers.
// Grid: {n_kv, splits, groups}, threads = 32 * per * rows.
kernel void ink_attn_128_sg_rows(constant Args& a [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int PER = 16;
    constexpr int kSgStride = 132;
    const int g = int(grp.x), z = int(grp.y), sub_sg = int(grp.z);
    const int nh = a.n[1], nkv = a.n[2], nrel = a.n[4], extent = a.n[5], window = a.n[6], ring = a.n[7],
              splits = a.n[8], groups = a.n[9];
    const int per = nh / nkv;
    const int t = int(sg) / per, hp = int(sg) % per;
    const ulong head = ulong(t) * nh + ulong(g * per + hp);
    const long p = ink_pos(a, t);
    long lo = window > 0 ? max(0L, p - window + 1) : 0, hi = p;
    if (splits > 1) {
        const long span = (hi - lo + splits) / splits, first = lo + long(z) * span;
        hi = min(hi, first + span - 1);
        lo = first;
    }
    const uint slot = lane >> 3, l8 = lane & 7;
    float q[PER], r[4];
    for (int c = 0; c < PER; ++c) q[c] = P<float>(a, 0)[head * 128 + l8 * PER + c] * a.f[0];
    for (int c = 0; c < 4; ++c) {
        const int k = int(l8) * 4 + c;
        r[c] = k < nrel ? P<float>(a, 1)[head * nrel + k] : 0.0f;
    }
    const float sink = P<float>(a, 2)[head];
    float m = -INFINITY, l = 0.0f, acc[PER];
    for (int c = 0; c < PER; ++c) acc[c] = 0.0f;
    device const half* keys = P<half>(a, 3);
    device const half* vals = P<half>(a, 4);
    device const float* rel = P<float>(a, 5);
    const bool has_rel_lane = int(l8) * 4 < nrel;
    const long step = long(groups) * 4;
    for (long j0 = lo + long(sub_sg) * 4; j0 <= hi; j0 += step) {
        const long j = j0 + slot;
        const bool valid = j <= hi;
        const ulong at = (ulong(ring > 0 ? (valid ? j : hi) % ring : (valid ? j : hi)) * nkv + g) * 128 + l8 * PER;
        device const half4* k4 = reinterpret_cast<device const half4*>(keys + at);
        device const half4* v4 = reinterpret_cast<device const half4*>(vals + at);
        float d = 0.0f;
        const long dist = p - j;
        if (valid && has_rel_lane && dist < extent) {
            for (int c = 0; c < 4; ++c) {
                const int i = int(l8) * 4 + c;
                if (i < nrel) d += rel[ulong(dist) * nrel + i] * r[c];
            }
        }
        for (int c = 0; c < 4; ++c) {
            const float4 kf = float4(k4[c]);
            d += q[4 * c] * kf.x;
            d += q[4 * c + 1] * kf.y;
            d += q[4 * c + 2] * kf.z;
            d += q[4 * c + 3] * kf.w;
        }
        d += simd_shuffle_xor(d, 1);
        d += simd_shuffle_xor(d, 2);
        d += simd_shuffle_xor(d, 4);
        if (!valid) continue;
        d -= sink;
        const float next = max(m, d), keep = exp(m - next), w = exp(d - next);
        l = l * keep + w;
        m = next;
        for (int c = 0; c < 4; ++c) {
            const float4 vf = float4(v4[c]);
            acc[4 * c] = acc[4 * c] * keep + w * vf.x;
            acc[4 * c + 1] = acc[4 * c + 1] * keep + w * vf.y;
            acc[4 * c + 2] = acc[4 * c + 2] * keep + w * vf.z;
            acc[4 * c + 3] = acc[4 * c + 3] * keep + w * vf.w;
        }
    }
    float top = max(m, simd_shuffle_xor(m, 8));
    top = max(top, simd_shuffle_xor(top, 16));
    const float f = m == -INFINITY ? 0.0f : exp(m - top);
    float lt = l * f;
    lt += simd_shuffle_xor(lt, 8);
    lt += simd_shuffle_xor(lt, 16);
    device float* o = P<float>(a, 9) + ((head * ulong(splits) + ulong(z)) * ulong(groups) + ulong(sub_sg)) * ulong(kSgStride);
    if (lane == 0) {
        o[0] = top;
        o[1] = lt;
    }
    if (lt > 0.0f) {
        for (int c = 0; c < 4; ++c) {
            float4 xv;
            for (int b = 0; b < 4; ++b) {
                float x = acc[4 * c + b] * f;
                x += simd_shuffle_xor(x, 8);
                x += simd_shuffle_xor(x, 16);
                xv[b] = x;
            }
            if (slot == 0) *reinterpret_cast<device float4*>(o + 4 + l8 * PER + 4 * c) = xv;
        }
    }
}

// Two-stage float4 join for ink_attn_128_sg_rows (grid {n_head, rows, 1}, threads = 32 * splits, sh = splits * 132 * 4 B).
kernel void ink_attn_join_sg_rows(constant Args& a [[buffer(0)]], threadgroup float* sh [[threadgroup(0)]],
                                  uint2 grp [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int stride = 132;
    const int nh = a.n[1], splits = a.n[8], n_chunks = a.n[9];
    const ulong head = ulong(grp.y) * nh + grp.x;
    const int z = int(sg);
    device const float* src = P<float>(a, 9) + (head * ulong(splits) + ulong(z)) * ulong(n_chunks) * stride;
    float stop = -INFINITY;
    for (int s = 0; s < n_chunks; ++s) stop = max(stop, src[ulong(s) * stride]);
    float stotal = 0.0f;
    float4 sum4 = float4(0.0f);
    for (int s = 0; s < n_chunks; ++s) {
        device const float* os = src + ulong(s) * stride;
        const float ls = os[1];
        if (ls > 0.0f) {
            const float ws = exp(os[0] - stop);
            stotal += ls * ws;
            sum4 += *reinterpret_cast<device const float4*>(os + 4 + lane * 4) * ws;
        }
    }
    threadgroup float* run = sh + ulong(z) * stride;
    if (lane == 0) {
        run[0] = stop;
        run[1] = stotal;
    }
    *reinterpret_cast<threadgroup float4*>(run + 4 + lane * 4) = sum4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (z != 0) return;
    float top = 0.0f;
    for (int s = 0; s < splits; ++s) top = max(top, sh[ulong(s) * stride]);
    float total = exp(-top);
    float4 out4 = float4(0.0f);
    for (int s = 0; s < splits; ++s) {
        threadgroup const float* os = sh + ulong(s) * stride;
        const float ls = os[1];
        if (ls > 0.0f) {
            const float ws = exp(os[0] - top);
            total += ls * ws;
            out4 += *reinterpret_cast<threadgroup const float4*>(os + 4 + lane * 4) * ws;
        }
    }
    float4 y4 = out4 / total;
    device const float* gate = P<float>(a, 6);
    const ulong base = head * 128 + lane * 4;
    if (gate) {
        const float4 g4 = *reinterpret_cast<device const float4*>(gate + base);
        y4 *= float4(sigm(g4.x), sigm(g4.y), sigm(g4.z), sigm(g4.w));
    }
    *reinterpret_cast<device float4*>(P<float>(a, 7) + base) = y4;
    if (a.p[10]) {
        simdgroup_barrier(mem_flags::mem_device);
        device float* out_f = P<float>(a, 7) + head * 128;
        for (int b = 0; b < 4; ++b) {
            const int d = b * 32 + int(lane);
            q8_put(P<uchar>(a, 10), int(head * 128) + d, out_f[d]);
        }
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
inline float4 ds_load_q(device const float* qbase, int r, int g, int gq, int heads, uint lane, float scale) {
    device const float* qr = qbase + (ulong(r / gq) * heads + g * gq + r % gq) * 128 + lane * 4;
    return float4(qr[0], qr[1], qr[2], qr[3]) * scale;
}
inline void ds_store_r(device float* out, int r, uint lane, float4 acc, float m, float l) {
    out[r * kDsPart + lane * 4 + 0] = acc.x;
    out[r * kDsPart + lane * 4 + 1] = acc.y;
    out[r * kDsPart + lane * 4 + 2] = acc.z;
    out[r * kDsPart + lane * 4 + 3] = acc.w;
    if (lane == 0) {
        out[r * kDsPart + 128] = m;
        out[r * kDsPart + 129] = l;
    }
}
kernel void ds_part(constant Args& a [[buffer(0)]], uint2 grp [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const int g = grp.x, s = grp.y, heads = a.n[2], kvh = a.n[3], gq = heads / kvh, R = a.n[0] * gq;
    const int r = int(sg);
    if (r >= R) return;
    const long cap = long(uint(a.n[4])) | (long(a.n[5]) << 32);
    long lo, nk, per;
    ds_span(a, lo, nk, per);
    const long k0 = long(s) * per;
    if (k0 >= nk) return;
    const long k1 = min(nk, k0 + per);
    const float4 q0 = ds_load_q(P<float>(a, 0), r, g, gq, heads, lane, a.f[0]);
    float4 acc0 = float4(0.0f);
    float m0 = -INFINITY, l0 = 0.0f;
    device const half* keys = P<half>(a, 1);
    device const half* vals = P<half>(a, 2);
    device const long* tags = P<long>(a, 3);
    long at = lo + k0, cell = at % cap;
    for (long k = k0; k < k1; ++k, ++at, cell = (cell + 1 == cap ? 0L : cell + 1)) {
        if (tags && tags[cell] != at) continue;
        const ulong o = (ulong(cell) * kvh + g) * 128 + lane * 4;
        const float4 kv = float4(*reinterpret_cast<device const half4*>(keys + o));
        const float4 vv = float4(*reinterpret_cast<device const half4*>(vals + o));
        float d = q0.x * kv.x + q0.y * kv.y + q0.z * kv.z + q0.w * kv.w;
        d = simd_sum(d);
        const float next = max(m0, d), keep = exp(m0 - next), w = exp(d - next);
        l0 = l0 * keep + w;
        m0 = next;
        acc0 = acc0 * keep + w * vv;
    }
    device float* out = P<float>(a, 4) + (ulong(g) * kDsSplits + s) * kDsRows * kDsPart;
    ds_store_r(out, r, lane, acc0, m0, l0);
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
    const float y = acc / l;
    const int idx = (t * heads + h) * 128 + int(d);
    P<float>(a, 6)[idx] = y;
    if (a.p[7]) q8_put(P<uchar>(a, 7), idx, y);
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

// ---- the Markov head's 2-bit proxy screen (ds_screen_pack, ds_screen_order, ds_head_base, ds_head_pick) ----
constant constexpr int kScrPer = 4;
constant constexpr int kScrMaxCand = 64;
constant constexpr float kScrDelta = 5.0f;
constant constexpr int kDsDraftsMax = 7;
constant constexpr int kScrBaseTiles = 4;
constant constexpr int kScrTilesPerBlock = 4;

inline ulong scr_token(ulong r, ulong tiles) { return (r & 31ul) * tiles + (r >> 5); }
inline int scr_level(float w, float sh) { return min(3, max(0, int(floor(w / sh + 2.0f)))); }

inline uchar2 scr_k4_sc(int j, int k, device const uchar* q) {
    return j < 4 ? uchar2(q[j + k] & 63, q[j + 4 + k] & 63)
                 : uchar2((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2),
                          (q[j + 4 + k] >> 4) | ((q[j + k] & 0xc0) >> 2));
}

inline half scr_fit(thread const float* w, thread float& sh_out) {
    float amax = 1e-9f;
    for (int l = 0; l < 32; ++l) amax = max(amax, abs(w[l]));
    float best_s = amax / 1.5f, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = amax / 1.5f * (1.0f - 0.1f * float(t));
        const float sh = float(half(s));
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = (float(scr_level(w[l], sh)) - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) best_e = e, best_s = s;
    }
    half hs = half(best_s);
    const float sh = float(hs);
    sh_out = sh;
    float ww = 0.0f, wr = 0.0f;
    for (int l = 0; l < 32; ++l) {
        const float r = (float(scr_level(w[l], sh)) - 1.5f) * sh;
        ww += w[l] * w[l];
        wr += w[l] * r;
    }
    if (wr > 0.0f) hs = half(sh * sqrt(ww / wr));
    return hs;
}

// p[0] weights (Q8_0 or Q5_K), p[1] wq, p[2] sc; n[0] n_in, n[1] n_rows, n[2] tiles, n[3] type (8 or 13).
kernel void ds_scr_pack(constant Args& a [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
    const int n_in = a.n[0], n_rows = a.n[1], tiles = a.n[2], type = a.n[3];
    const int nb = n_in / 32, nw4 = n_in / 64;
    const ulong total = ulong(tiles) * 32ul * ulong(nb);
    if (ulong(idx) >= total) return;
    const ulong row = ulong(idx) / ulong(nb);
    const int g = int(ulong(idx) % ulong(nb));
    const ulong tok = scr_token(row, ulong(tiles));
    const ulong tile = row >> 5, lane = row & 31ul;
    const int w4 = g >> 1, h = g & 1;
    device uint* wd = P<uint>(a, 1) + (((tile * ulong(nw4) + ulong(w4)) * 32ul + lane) * 4ul + ulong(2 * h));
    device half* sd = P<half>(a, 2) + (((tile * ulong(nw4) + ulong(w4)) * 32ul + lane) * 2ul + ulong(h));
    if (tok >= ulong(n_rows)) {
        wd[0] = 0u;
        wd[1] = 0u;
        sd[0] = half(0.0f);
        return;
    }
    float w[32];
    if (type == 13) {
        const int blk_idx = g >> 3, sb = g & 7;
        device const uchar* blk = P<uchar>(a, 0) + (tok * ulong(n_in / 256) + ulong(blk_idx)) * 176ul;
        const float d = float(*reinterpret_cast<device const half*>(blk));
        const float dmin = float(*reinterpret_cast<device const half*>(blk + 2));
        const uchar2 sc = scr_k4_sc((sb / 2) * 2, sb & 1, blk + 4);
        const float dl = d * float(sc.x), ml = dmin * float(sc.y);
        device const uchar* qh = blk + 16;
        device const uchar* qs = blk + 48 + 32 * (sb / 2);
        const uchar ul = uchar(1u << sb);
        for (int l = 0; l < 32; ++l) {
            const int qv = ((sb & 1) ? (qs[l] >> 4) : (qs[l] & 0x0F)) + ((qh[l] & ul) ? 16 : 0);
            w[l] = dl * float(qv) - ml;
        }
    } else {
        device const uchar* blk = P<uchar>(a, 0) + (tok * ulong(nb) + ulong(g)) * 34ul;
        const float d = float(*reinterpret_cast<device const half*>(blk));
        for (int l = 0; l < 32; ++l) w[l] = d * float(as_type<char>(blk[2 + l]));
    }
    float sh;
    const half hs = scr_fit(w, sh);
    uint w0 = 0u, w1 = 0u;
    for (int l = 0; l < 16; ++l) {
        w0 |= uint(scr_level(w[l], sh)) << (2 * l);
        w1 |= uint(scr_level(w[16 + l], sh)) << (2 * l);
    }
    wd[0] = w0;
    wd[1] = w1;
    sd[0] = hs;
}

// p[0] by_token, p[1] by_row; n[0] n_rows, n[1] tiles.
kernel void ds_scr_order(constant Args& a [[buffer(0)]], uint r [[thread_position_in_grid]]) {
    const ulong tiles = ulong(a.n[1]);
    if (ulong(r) >= tiles * 32ul) return;
    const ulong tok = scr_token(ulong(r), tiles);
    P<float>(a, 1)[r] = tok < ulong(a.n[0]) ? P<float>(a, 0)[tok] : 0.0f;
}

inline void scr_unpack16_f(uint v, thread float4& w0, thread float4& w1, thread float4& w2, thread float4& w3) {
    uint b = v & 0xFFu;
    w0 = float4(b & 3u, (b >> 2) & 3u, (b >> 4) & 3u, (b >> 6) & 3u);
    b = (v >> 8) & 0xFFu;
    w1 = float4(b & 3u, (b >> 2) & 3u, (b >> 4) & 3u, (b >> 6) & 3u);
    b = (v >> 16) & 0xFFu;
    w2 = float4(b & 3u, (b >> 2) & 3u, (b >> 4) & 3u, (b >> 6) & 3u);
    b = (v >> 24) & 0xFFu;
    w3 = float4(b & 3u, (b >> 2) & 3u, (b >> 4) & 3u, (b >> 6) & 3u);
}

inline float scr_dot16_u(float4 w0, float4 w1, float4 w2, float4 w3, device const char4* q4) {
    return dot(w0, float4(q4[0])) + dot(w1, float4(q4[1])) + dot(w2, float4(q4[2])) + dot(w3, float4(q4[3]));
}

inline float scr_dot16_f(uint v, device const char4* q4) {
    float4 w0, w1, w2, w3;
    scr_unpack16_f(v, w0, w1, w2, w3);
    return scr_dot16_u(w0, w1, w2, w3, q4);
}

// p[0] wq, p[1] sc, p[2] x_q8_1, p[3] approx; n[0] rows, n[1] n_tiles, n[2] n_pad.  n_in = 2304 (NB=72, nw4=36).
// Zero threadgroup memory and zero barriers: unpacks each 2-bit weight word ONCE into exact float4s and dots across all rows.
kernel void ds_scr_base(constant Args& a [[buffer(0)]], uint blk [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int NB = 72, kNw4 = 36;
    const int rows = a.n[0], n_tiles = a.n[1], n_pad = a.n[2];
    const int tile = int(blk) * kScrBaseTiles + int(sg);
    if (tile >= n_tiles) return;
    device const uint4* wq = P<uint4>(a, 0);
    device const half2* sc = P<half2>(a, 1);
    device const uchar* x0 = P<uchar>(a, 2);
    device float* approx = P<float>(a, 3);
    float acc[kDsDraftsMax];
    for (int r = 0; r < kDsDraftsMax; ++r) acc[r] = 0.0f;
    const ulong tile_base = ulong(tile) * ulong(kNw4) * 32ul + ulong(lane);
    for (int w4 = 0; w4 < kNw4; ++w4) {
        const uint4 wj = wq[tile_base + ulong(w4) * 32ul];
        const half2 sj = sc[tile_base + ulong(w4) * 32ul];
        const float s0 = float(sj.x), s1 = float(sj.y);
        const ulong off0 = ulong(2 * w4) * 36ul;
        const ulong off1 = off0 + 36ul;
        float4 w0, w1, w2, w3;
        float p0[kDsDraftsMax];
        scr_unpack16_f(wj.x, w0, w1, w2, w3);
        for (int r = 0; r < kDsDraftsMax; ++r) {
            if (r >= rows) break;
            device const uchar* b0 = x0 + ulong(r) * ulong(NB) * 36ul + off0;
            p0[r] = scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(b0 + 4));
        }
        scr_unpack16_f(wj.y, w0, w1, w2, w3);
        for (int r = 0; r < kDsDraftsMax; ++r) {
            if (r >= rows) break;
            device const uchar* b0 = x0 + ulong(r) * ulong(NB) * 36ul + off0;
            device const half* d0 = reinterpret_cast<device const half*>(b0);
            const float pr = p0[r] + scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(b0 + 20));
            acc[r] += s0 * (float(d0[0]) * pr - 1.5f * float(d0[1]));
        }
        scr_unpack16_f(wj.z, w0, w1, w2, w3);
        for (int r = 0; r < kDsDraftsMax; ++r) {
            if (r >= rows) break;
            device const uchar* b1 = x0 + ulong(r) * ulong(NB) * 36ul + off1;
            p0[r] = scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(b1 + 4));
        }
        scr_unpack16_f(wj.w, w0, w1, w2, w3);
        for (int r = 0; r < kDsDraftsMax; ++r) {
            if (r >= rows) break;
            device const uchar* b1 = x0 + ulong(r) * ulong(NB) * 36ul + off1;
            device const half* d1 = reinterpret_cast<device const half*>(b1);
            const float pr = p0[r] + scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(b1 + 20));
            acc[r] += s1 * (float(d1[0]) * pr - 1.5f * float(d1[1]));
        }
    }
    const ulong row = ulong(tile) * 32ul + ulong(lane);
    for (int r = 0; r < kDsDraftsMax; ++r)
        if (r < rows) approx[ulong(r) * ulong(n_pad) + row] = acc[r];
}

// Compile-time rows == 3 specialization of ds_scr_base (default Metal DSpark block size nb = 4): scalar accumulators
// and branchless inner loop maximize GPU occupancy toward the 2-bit weight bandwidth ceiling.  Bit-identical (0 ULP).
kernel void ds_scr_base_3(constant Args& a [[buffer(0)]], uint blk [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int NB = 72, kNw4 = 36;
    const int n_tiles = a.n[1], n_pad = a.n[2];
    const int tile = int(blk) * kScrBaseTiles + int(sg);
    if (tile >= n_tiles) return;
    device const uint4* wq = P<uint4>(a, 0);
    device const half2* sc = P<half2>(a, 1);
    device const uchar* r0 = P<uchar>(a, 2);
    device const uchar* r1 = r0 + ulong(NB) * 36ul;
    device const uchar* r2 = r1 + ulong(NB) * 36ul;
    device float* approx = P<float>(a, 3);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    const ulong tile_base = ulong(tile) * ulong(kNw4) * 32ul + ulong(lane);
    for (int w4 = 0; w4 < kNw4; ++w4) {
        const uint4 wj = wq[tile_base + ulong(w4) * 32ul];
        const half2 sj = sc[tile_base + ulong(w4) * 32ul];
        const float s0 = float(sj.x), s1 = float(sj.y);
        const ulong off0 = ulong(2 * w4) * 36ul, off1 = off0 + 36ul;
        float4 w0, w1, w2, w3;
        scr_unpack16_f(wj.x, w0, w1, w2, w3);
        float p0 = scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r0 + off0 + 4));
        float p1 = scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r1 + off0 + 4));
        float p2 = scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r2 + off0 + 4));
        scr_unpack16_f(wj.y, w0, w1, w2, w3);
        device const half* d00 = reinterpret_cast<device const half*>(r0 + off0);
        device const half* d10 = reinterpret_cast<device const half*>(r1 + off0);
        device const half* d20 = reinterpret_cast<device const half*>(r2 + off0);
        acc0 += s0 * (float(d00[0]) * (p0 + scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r0 + off0 + 20))) - 1.5f * float(d00[1]));
        acc1 += s0 * (float(d10[0]) * (p1 + scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r1 + off0 + 20))) - 1.5f * float(d10[1]));
        acc2 += s0 * (float(d20[0]) * (p2 + scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r2 + off0 + 20))) - 1.5f * float(d20[1]));
        scr_unpack16_f(wj.z, w0, w1, w2, w3);
        p0 = scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r0 + off1 + 4));
        p1 = scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r1 + off1 + 4));
        p2 = scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r2 + off1 + 4));
        scr_unpack16_f(wj.w, w0, w1, w2, w3);
        device const half* d01 = reinterpret_cast<device const half*>(r0 + off1);
        device const half* d11 = reinterpret_cast<device const half*>(r1 + off1);
        device const half* d21 = reinterpret_cast<device const half*>(r2 + off1);
        acc0 += s1 * (float(d01[0]) * (p0 + scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r0 + off1 + 20))) - 1.5f * float(d01[1]));
        acc1 += s1 * (float(d11[0]) * (p1 + scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r1 + off1 + 20))) - 1.5f * float(d11[1]));
        acc2 += s1 * (float(d21[0]) * (p2 + scr_dot16_u(w0, w1, w2, w3, reinterpret_cast<device const char4*>(r2 + off1 + 20))) - 1.5f * float(d21[1]));
    }
    const ulong row = ulong(tile) * 32ul + ulong(lane);
    approx[row] = acc0;
    approx[ulong(n_pad) + row] = acc1;
    approx[2ul * ulong(n_pad) + row] = acc2;
}

// p[0] wq, p[1] sc, p[2] lat_q8_1, p[3] approx_i, p[4] fold_bp, p[5] cand, p[6] part;
// n[0] n_rows, n[1] n_tiles; f[0] cap.  rank = 256 (NB=8, nw4=4).  Zero threadgroup memory, L1 broadcast lat.
kernel void ds_scr_step(constant Args& a [[buffer(0)]], uint blk [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int kNw4 = 4;
    const int n_rows = a.n[0], n_tiles = a.n[1];
    const float cap = a.f[0];
    const int tile = int(blk) * kScrTilesPerBlock + int(sg);
    if (tile >= n_tiles) return;
    const ulong row = ulong(tile) * 32ul + ulong(lane);
    const bool real = scr_token(row, ulong(n_tiles)) < ulong(n_rows);
    device const uint4* wq = P<uint4>(a, 0);
    device const half2* sc = P<half2>(a, 1);
    device const uchar* lat = P<uchar>(a, 2);
    float acc = 0.0f;
    if (real) {
        const ulong tile_base = ulong(tile) * ulong(kNw4) * 32ul + ulong(lane);
        for (int w4 = 0; w4 < kNw4; ++w4) {
            const uint4 wj = wq[tile_base + ulong(w4) * 32ul];
            const half2 sj = sc[tile_base + ulong(w4) * 32ul];
            const int g0 = 2 * w4, g1 = g0 + 1;
            device const uchar* b0 = lat + ulong(g0) * 36ul;
            device const half* d0 = reinterpret_cast<device const half*>(b0);
            const float p0 = scr_dot16_f(wj.x, reinterpret_cast<device const char4*>(b0 + 4))
                           + scr_dot16_f(wj.y, reinterpret_cast<device const char4*>(b0 + 20));
            acc += float(sj.x) * (float(d0[0]) * p0 - 1.5f * float(d0[1]));
            device const uchar* b1 = lat + ulong(g1) * 36ul;
            device const half* d1 = reinterpret_cast<device const half*>(b1);
            const float p1 = scr_dot16_f(wj.z, reinterpret_cast<device const char4*>(b1 + 4))
                           + scr_dot16_f(wj.w, reinterpret_cast<device const char4*>(b1 + 20));
            acc += float(sj.y) * (float(d1[0]) * p1 - 1.5f * float(d1[1]));
        }
    }
    const float fb = P<float>(a, 4)[row];
    float v = real ? cap * precise::tanh((P<float>(a, 3)[row] + acc + fb) / cap) : -INFINITY;
    const float m = simd_max(v);
    const float e = simd_sum(v > -INFINITY ? exp(v - m) : 0.0f);
    if (lane == 0) P<float2>(a, 6)[tile] = float2(m, e);
    for (int k = 0; k < kScrPer; ++k) {
        float bv = v;
        int bi = int(lane);
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (lane == 0) P<float2>(a, 5)[tile * kScrPer + k] = float2(bv, as_type<float>(tile * 32 + bi));
        if (int(lane) == bi) v = -INFINITY;
    }
}

inline ulong scr_key(float2 c) {
    const uint u = as_type<uint>(c.x);
    const uint o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (ulong(o) << 32) | ulong(0xffffffffu - as_type<uint>(c.y));
}

inline float scr_dot_q8(device const uchar* w, threadgroup const uchar* x, int nb, uint lane) {
    const int k = int(lane & 3u) * 8;
    float s = 0.0f;
    for (int b = int(lane >> 2); b < nb; b += 8) {
        device const uchar* blk = w + ulong(b) * 34ul;
        const float dw = float(*reinterpret_cast<device const half*>(blk));
        const float dx = float(*reinterpret_cast<threadgroup const half*>(x + b * 36));
        int sumi = 0;
        for (int i = 0; i < 8; ++i) sumi += int(as_type<char>(blk[2 + k + i])) * int(as_type<char>(x[b * 36 + 4 + k + i]));
        s += dw * dx * float(sumi);
    }
    return simd_sum(s);
}

inline float scr_dot_q5k(device const uchar* w, threadgroup const uchar* x, int nb, uint lane) {
    float s = 0.0f;
    for (int g = int(lane); g < nb; g += 32) {
        const int blk_idx = g >> 3, sb = g & 7;
        device const uchar* blk = w + ulong(blk_idx) * 176ul;
        const float d = float(*reinterpret_cast<device const half*>(blk));
        const float dmin = float(*reinterpret_cast<device const half*>(blk + 2));
        const uchar2 sc = scr_k4_sc((sb / 2) * 2, sb & 1, blk + 4);
        const float dl = d * float(sc.x), ml = dmin * float(sc.y);
        device const uchar* qh = blk + 16;
        device const uchar* qs = blk + 48 + 32 * (sb / 2);
        const uchar ul = uchar(1u << sb);
        const float dx = float(*reinterpret_cast<threadgroup const half*>(x + g * 36));
        const float sx = float(*reinterpret_cast<threadgroup const half*>(x + g * 36 + 2));
        int sumi = 0;
        for (int l = 0; l < 32; ++l) {
            const int qv = ((sb & 1) ? (qs[l] >> 4) : (qs[l] & 0x0F)) + ((qh[l] & ul) ? 16 : 0);
            sumi += qv * int(as_type<char>(x[g * 36 + 4 + l]));
        }
        s += dl * dx * float(sumi) - ml * sx;
    }
    return simd_sum(s);
}

// p[0] head, p[1] fold, p[2] fold_b, p[3] xh_q8_1, p[4] xl_q8_1, p[5] cand, p[6] part, p[7] prev, p[8] out;
// n[0] n_tiles, n[1] head_type (8 or 13); f[0] cap.  1 TG x 512 threads.
kernel void ds_scr_final(constant Args& a [[buffer(0)]], uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int NE = 72, NR = 8;
    threadgroup uchar xh[NE * 36], xl[NR * 36];
    threadgroup float wmax[16];
    threadgroup int list[kScrMaxCand];
    threadgroup float lp[kScrMaxCand], ex[kScrMaxCand];
    threadgroup atomic_int cnt;
    threadgroup atomic_uint hist[256];
    threadgroup ulong want_s;
    threadgroup int need_s;
    const int n_tiles = a.n[0], head_type = a.n[1];
    const float cap = a.f[0];
    if (tid == 0) atomic_store_explicit(&cnt, 0, memory_order_relaxed);
    device const uint* xhg = P<uint>(a, 3);
    device const uint* xlg = P<uint>(a, 4);
    for (int k = int(tid); k < (NE + NR) * 9; k += 512) {
        const int b = k / 9, j = k % 9;
        if (b < NE) reinterpret_cast<threadgroup uint*>(xh + b * 36)[j] = xhg[b * 9 + j];
        else reinterpret_cast<threadgroup uint*>(xl + (b - NE) * 36)[j] = xlg[(b - NE) * 9 + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float2* cand = P<float2>(a, 5);
    device const float2* part = P<float2>(a, 6);
    float m = -INFINITY;
    for (int t = int(tid); t < n_tiles; t += 512) m = max(m, part[t].x);
    m = simd_max(m);
    if (lane == 0) wmax[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = wmax[0];
    for (int w = 1; w < 16; ++w) m = max(m, wmax[w]);
    if (m == -INFINITY) {
        if (tid == 0) {
            P<int>(a, 8)[0] = P<int>(a, 7)[0];
            P<float>(a, 8)[1] = 0.0f;
        }
        return;
    }
    for (int t = int(tid); t < n_tiles; t += 512) {
        const float tm = part[t].x;
        if (!(tm > -INFINITY && tm >= m - kScrDelta)) continue;
        for (int k = 0; k < kScrPer; ++k) {
            const float2 c = cand[t * kScrPer + k];
            if (c.x > -INFINITY && c.x >= m - kScrDelta) {
                const int p = atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
                if (p < kScrMaxCand) list[p] = as_type<int>(c.y), lp[p] = c.x;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int total = atomic_load_explicit(&cnt, memory_order_relaxed);
    if (total > kScrMaxCand) {
        const int nc = n_tiles * kScrPer;
        if (tid == 0) want_s = 0ul, need_s = kScrMaxCand;
        ulong mask = 0ul;
        for (int shift = 56; shift >= 0; shift -= 8) {
            if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const ulong want = want_s;
            for (int i = int(tid); i < nc; i += 512) {
                const float2 c = cand[i];
                if (!(c.x > -INFINITY && c.x >= m - kScrDelta)) continue;
                const ulong k = scr_key(c);
                if ((k & mask) == want)
                    atomic_fetch_add_explicit(&hist[int((k >> shift) & 255ul)], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                int b[8], s = 0;
                for (int j = 0; j < 8; ++j) {
                    b[j] = int(atomic_load_explicit(&hist[255 - 8 * int(lane) - j], memory_order_relaxed));
                    s += b[j];
                }
                const int incl = simd_prefix_inclusive_sum(s);
                const int need = need_s, excl = incl - s;
                if (excl < need && incl >= need) {
                    int acc = excl, j = 0;
                    while (acc + b[j] < need) acc += b[j++];
                    want_s = want | (ulong(255 - 8 * int(lane) - j) << shift);
                    need_s = need - acc;
                }
            }
            mask |= 255ul << shift;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const ulong kth = want_s;
        if (tid == 0) atomic_store_explicit(&cnt, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < nc; i += 512) {
            const float2 c = cand[i];
            if (c.x > -INFINITY && c.x >= m - kScrDelta && scr_key(c) >= kth) {
                const int p = atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
                if (p < kScrMaxCand) list[p] = as_type<int>(c.y), lp[p] = c.x;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int n = min(total, kScrMaxCand);
    int row = 0, rank = 0;
    float lpv = 0.0f;
    if (int(tid) < n) {
        row = list[tid];
        lpv = lp[tid];
        for (int j = 0; j < n; ++j) rank += list[j] < row;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < n) list[rank] = row, lp[rank] = lpv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar* head_w = P<uchar>(a, 0);
    device const uchar* fold_w = P<uchar>(a, 1);
    device const float* fold_b = P<float>(a, 2);
    for (int c = int(sg); c < n; c += 16) {
        const ulong tok = scr_token(ulong(list[c]), ulong(n_tiles));
        const float ha = head_type == 13 ? scr_dot_q5k(head_w + tok * ulong(NE / 8) * 176ul, xh, NE, lane)
                                         : scr_dot_q8(head_w + tok * ulong(NE) * 34ul, xh, NE, lane);
        const float fb_dot = scr_dot_q8(fold_w + tok * ulong(NR) * 34ul, xl, NR, lane);
        if (lane == 0) ex[c] = cap * precise::tanh((ha + fb_dot + fold_b[tok]) / cap), list[c] = int(tok);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int c = int(lane); c < n; c += 32)
            if (ex[c] > bv || (ex[c] == bv && list[c] < bi)) bv = ex[c], bi = list[c];
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if (lane == 0) wmax[0] = bv, need_s = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float bv = wmax[0];
    const int bi = need_s;
    float sum = 0.0f;
    for (int t = int(tid); t < n_tiles; t += 512) {
        const float2 p = part[t];
        if (p.y > 0.0f) sum += p.y * exp(p.x - bv);
    }
    for (int c = int(tid); c < n; c += 512) sum += exp(ex[c] - bv) - exp(lp[c] - bv);
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) wmax[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        sum = 0.0f;
        for (int w = 0; w < 16; ++w) sum += wmax[w];
        P<int>(a, 8)[0] = bi;
        P<float>(a, 8)[1] = 1.0f / max(sum, 1.0f);
        P<int>(a, 7)[0] = bi;
    }
}
