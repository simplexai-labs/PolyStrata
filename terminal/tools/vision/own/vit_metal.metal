// tools/vision/own/vit_metal.metal - the vision towers' kernels on Metal (own-stack phase 3; Qwen, Gemma 4, MiniCPM-V 4.7).  The counterparts of vit_gpu.cu's kernels, one for one,
// with the same arithmetic: float32 activations, 16-bit GEMM inputs of the weight's own type (F16 or BF16), q / k / v and the attention scores in F16,
// float32 accumulation.  Embedded into strata-vision at build time and compiled when the reader starts (vit_metal.mm); the tensor-op products are in
// vit_metal_mpp.metal (Metal 4 only).
//
// Every kernel takes its buffers at [[buffer(0..)]] and its sizes in a small constant struct at [[buffer(15)]].
#include <metal_stdlib>
using namespace metal;

// the 16-bit kinds (vit_metal.mm's Kind): 0 float32, 1 F16, 2 BF16, 3 a BF16 matrix held as F16 with its inputs written as F16 (clamped to +-65,504)
constant int K_F32 = 0;
constant int K_F16 = 1;
constant int K_HF = 3;

inline ushort f_to_bf16(float f) {       // round to nearest even; NaN stays NaN
    uint u = as_type<uint>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return ushort(u >> 16);
}
inline float bf16_to_f(ushort b) { return as_type<float>(uint(b) << 16); }
inline ushort f_to_f16bits(float f) { return as_type<ushort>(half(f)); }       // float -> half rounds to nearest even
inline float f16bits_to_f(ushort h) { return float(as_type<half>(h)); }

inline void store_kind(device void* out, ulong i, float v, int kind) {
    if (kind == K_F32) ((device float*) out)[i] = v;
    else if (kind == K_F16) ((device ushort*) out)[i] = f_to_f16bits(v);
    else if (kind == K_HF) ((device ushort*) out)[i] = f_to_f16bits(v > 65504.0f ? 65504.0f : v < -65504.0f ? -65504.0f : v);   // NaN stays NaN (vit_f16.h f16_clamp)
    else ((device ushort*) out)[i] = f_to_bf16(v);
}

// threadgroup-wide sum / max of one float per thread (threads a multiple of 32, at most 1024)
inline float tg_sum(float v, threadgroup float* red, ushort tid, ushort nt) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31) == 0) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort nw = (nt + 31) / 32;
    v = (tid & 31) < nw ? red[tid & 31] : 0.0f;
    return simd_sum(v);
}
inline float tg_max(float v, threadgroup float* red, ushort tid, ushort nt) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31) == 0) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort nw = (nt + 31) / 32;
    v = (tid & 31) < nw ? red[tid & 31] : -FLT_MAX;
    return simd_max(v);
}

struct PatchPostArgs { int G, pw, ph, nbx, D; };
// x[t] += patch bias + the position table resized (bilinear, align corners: ggml's maths, in float) to the token's grid position.  One group a token.
kernel void vk_patch_post(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]], device const float* pos [[buffer(2)]],
                          constant PatchPostArgs& a [[buffer(15)]], uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort nt [[threads_per_threadgroup]]) {
    const int G = a.G, pw = a.pw, ph = a.ph, D = a.D;
    const int b = int(t) / 4, r = int(t) % 4;
    const int ty = 2 * (b / a.nbx) + r / 2, tx = 2 * (b % a.nbx) + r % 2;
    float sfx = (float) pw / G, sfy = (float) ph / G;
    if (pw > 1 && G > 1) sfx = (float) (pw - 1) / (G - 1);
    if (ph > 1 && G > 1) sfy = (float) (ph - 1) / (G - 1);
    const float fy = (float) ty / sfy, fx = (float) tx / sfx;
    int y0 = (int) floor(fy), x0 = (int) floor(fx), y1 = y0 + 1, x1 = x0 + 1;
    y0 = max(0, min(y0, G - 1)); y1 = max(0, min(y1, G - 1));
    x0 = max(0, min(x0, G - 1)); x1 = max(0, min(x1, G - 1));
    const float dy = fmax(0.0f, fmin(fy - (float) y0, 1.0f)), dx = fmax(0.0f, fmin(fx - (float) x0, 1.0f));
    device const float* pa = pos + ((ulong) y0 * G + x0) * D;
    device const float* pb = pos + ((ulong) y0 * G + x1) * D;
    device const float* pc = pos + ((ulong) y1 * G + x0) * D;
    device const float* pd = pos + ((ulong) y1 * G + x1) * D;
    device float* xr = x + (ulong) t * D;
    for (int i = tid; i < D; i += nt)
        xr[i] += bias[i] + (pa[i] * (1 - dx) * (1 - dy) + pb[i] * dx * (1 - dy) + pc[i] * (1 - dx) * dy + pd[i] * dx * dy);
}

struct NormArgs { int D, kind; float eps; };
// out = LayerNorm(x) * w + b, into `kind`.  One group a row.
kernel void vk_layernorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* b [[buffer(2)]],
                         device void* out [[buffer(3)]], constant NormArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    const int D = a.D;
    device const float* xr = x + (ulong) row * D;
    float s = 0.0f;
    for (int i = tid; i < D; i += nt) s += xr[i];
    const float mean = tg_sum(s, red, tid, nt) / (float) D;
    float v = 0.0f;
    for (int i = tid; i < D; i += nt) { const float d = xr[i] - mean; v += d * d; }
    const float scale = rsqrt(tg_sum(v, red, tid, nt) / (float) D + a.eps);
    for (int i = tid; i < D; i += nt) store_kind(out, (ulong) row * D + i, (xr[i] - mean) * scale * w[i] + b[i], a.kind);
}

struct RopeArgs { int row0, nbx, heads, hd; float theta_scale, qscale; };
// y [rows][3D] (the fused qkv product of the tokens row0 ..) + bias -> q (scaled), k (rotary applied), v, each [N][D] in F16.  Pair j of a head is
// (j, j + hd/2); j < hd/4 turns with the token's row, the others with its column.  One group a row.
kernel void vk_qkv_rope(device const float* y [[buffer(0)]], device const float* bias [[buffer(1)]], device ushort* q [[buffer(2)]],
                        device ushort* k [[buffer(3)]], device ushort* v [[buffer(4)]], constant RopeArgs& a [[buffer(15)]],
                        uint gr [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int t = a.row0 + int(gr), heads = a.heads, hd = a.hd, D = heads * hd, half_ = hd / 2, quarter = hd / 4;
    const int b = t / 4, r = t % 4;
    const int ty = 2 * (b / a.nbx) + r / 2, tx = 2 * (b % a.nbx) + r % 2;
    device const float* yr = y + (ulong) gr * 3 * D;
    for (int i = tid; i < 2 * heads * half_; i += nt) {
        const int part = i / (heads * half_), rem = i % (heads * half_), h = rem / half_, j = rem % half_;
        const int base = part * D + h * hd;
        const float av = yr[base + j] + bias[base + j], cv = yr[base + j + half_] + bias[base + j + half_];
        const float freq = precise::pow(a.theta_scale, (float) (j % quarter));
        float cs;
        const float sn = precise::sincos((float) (j < quarter ? ty : tx) * freq, cs);
        device ushort* dst = (part == 0 ? q : k) + (ulong) t * D + h * hd;
        const float sc = part == 0 ? a.qscale : 1.0f;
        dst[j] = f_to_f16bits((av * cs - cv * sn) * sc);
        dst[j + half_] = f_to_f16bits((av * sn + cv * cs) * sc);
    }
    for (int e = tid; e < D; e += nt) v[(ulong) t * D + e] = f_to_f16bits(yr[2 * D + e] + bias[2 * D + e]);
}

struct SoftmaxArgs { int n, ld; };
// In place: a row of n F16 scores becomes the row of probabilities, F16.  One group a row; PER scores a thread held in registers (threads * PER >= n).
template <int PER>
kernel void vk_softmax_rows(device ushort* S [[buffer(0)]], constant SoftmaxArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device ushort* s = S + (ulong) row * a.ld;
    const int n = a.n;
    float v[PER];
    float m = -FLT_MAX;
    for (int k = 0; k < PER; ++k) {
        const int j = int(tid) + k * int(nt);
        v[k] = j < n ? f16bits_to_f(s[j]) : -FLT_MAX;
        m = fmax(m, v[k]);
    }
    m = tg_max(m, red, tid, nt);
    float sum = 0.0f;
    for (int k = 0; k < PER; ++k) {
        const int j = int(tid) + k * int(nt);
        v[k] = j < n ? precise::exp(v[k] - m) : 0.0f;
        sum += v[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / tg_sum(sum, red, tid, nt);
    for (int k = 0; k < PER; ++k) {
        const int j = int(tid) + k * int(nt);
        if (j < n) s[j] = f_to_f16bits(v[k] * inv);
    }
}
template [[host_name("vk_softmax_rows_2")]] kernel decltype(vk_softmax_rows<2>) vk_softmax_rows<2>;
template [[host_name("vk_softmax_rows_8")]] kernel decltype(vk_softmax_rows<8>) vk_softmax_rows<8>;
template [[host_name("vk_softmax_rows_16")]] kernel decltype(vk_softmax_rows<16>) vk_softmax_rows<16>;
template [[host_name("vk_softmax_rows_64")]] kernel decltype(vk_softmax_rows<64>) vk_softmax_rows<64>;

// The same for a row too long to hold in registers: three passes over the row in memory.
kernel void vk_softmax_rows_long(device ushort* S [[buffer(0)]], constant SoftmaxArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device ushort* s = S + (ulong) row * a.ld;
    const int n = a.n;
    float m = -FLT_MAX;
    for (int j = tid; j < n; j += nt) m = fmax(m, f16bits_to_f(s[j]));
    m = tg_max(m, red, tid, nt);
    float sum = 0.0f;
    for (int j = tid; j < n; j += nt) sum += precise::exp(f16bits_to_f(s[j]) - m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / tg_sum(sum, red, tid, nt);
    for (int j = tid; j < n; j += nt) s[j] = f_to_f16bits(precise::exp(f16bits_to_f(s[j]) - m) * inv);
}

struct EltArgs { ulong n; int kind; int D; };
// n floats to the 16-bit kind (grid-stride)
kernel void vk_cast(device const float* in [[buffer(0)]], device void* out [[buffer(1)]], constant EltArgs& a [[buffer(15)]],
                    uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    for (ulong i = gid; i < a.n; i += gsz) store_kind(out, i, in[i], a.kind);
}
// y[i] rounded to the 16-bit kind and back (STRATA_VIT_REF_ROUND, a diagnostic)
kernel void vk_round(device float* y [[buffer(0)]], constant EltArgs& a [[buffer(15)]], uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    for (ulong i = gid; i < a.n; i += gsz) {
        const float v = y[i];
        y[i] = a.kind == K_F16 ? float(half(v)) : bf16_to_f(f_to_bf16(v));
    }
}
// x[r] += y[r] + bias.  One group a row.
kernel void vk_add_bias_res(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* bias [[buffer(2)]],
                            constant EltArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    const ulong base = (ulong) row * a.D;
    for (int i = tid; i < a.D; i += nt) x[base + i] += y[base + i] + bias[i];
}
// out = gelu_tanh(y + bias) into `kind`.  One group a row of D.
kernel void vk_bias_gelu_cast(device const float* y [[buffer(0)]], device const float* bias [[buffer(1)]], device void* out [[buffer(2)]],
                              constant EltArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                              ushort nt [[threads_per_threadgroup]]) {
    const ulong base = (ulong) row * a.D;
    for (int i = tid; i < a.D; i += nt) {
        const float x = y[base + i] + bias[i];
        const float g = 0.5f * x * (1.0f + precise::tanh(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
        store_kind(out, base + i, g, a.kind);
    }
}
// y[r] += bias.  One group a row of D.
kernel void vk_add_bias(device float* y [[buffer(0)]], device const float* bias [[buffer(1)]], constant EltArgs& a [[buffer(15)]],
                        uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const ulong base = (ulong) row * a.D;
    for (int i = tid; i < a.D; i += nt) y[base + i] += bias[i];
}

// ---- the plain GEMM (any Apple GPU): Y[z][r][o] = sum_k X[z][r][k] * W[z](o, k), float32 accumulation.
//   X: [rows][in], row stride ldx.   W: WN false: [out][in], row stride ldw (k contiguous); WN true: [in][out], row stride ldw (o contiguous).
//   Y: [rows][out], row stride ldy.  Batch z (grid z): the bases move by sx, sw, sy elements.
// A 64 x 64 tile a group (4 simdgroups of 32 x 32), k in steps of 32, operands staged as float in threadgroup memory (exact for 16-bit inputs).
struct GemmArgs { uint rows, in, out, ldx, ldw, ldy; ulong sx, sw, sy; };

template <typename T> inline float ld_e(device const T* p, ulong i);
template <> inline float ld_e<float>(device const float* p, ulong i) { return p[i]; }
template <> inline float ld_e<half>(device const half* p, ulong i) { return float(p[i]); }
template <> inline float ld_e<ushort>(device const ushort* p, ulong i) { return bf16_to_f(p[i]); }      // ushort = BF16 here
template <typename T> inline void st_e(device T* p, ulong i, float v);
template <> inline void st_e<float>(device float* p, ulong i, float v) { p[i] = v; }
template <> inline void st_e<half>(device half* p, ulong i, float v) { p[i] = half(v); }

template <typename TI, typename TO, bool WN>
kernel void vk_gemm(device const TI* X0 [[buffer(0)]], device const TI* W0 [[buffer(1)]], device TO* Y0 [[buffer(2)]], constant GemmArgs& a [[buffer(15)]],
                    uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[4096];   // As [64][32] then Bs [32][64]; reused for the result tile [64][64]
    threadgroup float* As = smem;
    threadgroup float* Bs = smem + 2048;
    device const TI* X = X0 + a.sx * tgp.z;
    device const TI* W = W0 + a.sw * tgp.z;
    device TO* Y = Y0 + a.sy * tgp.z;
    const uint o0 = tgp.x * 64u, r0 = tgp.y * 64u;
    const uint sr = (uint(sg) >> 1) * 32u, so = (uint(sg) & 1u) * 32u;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < a.in; k0 += 32u) {
        for (uint idx = tid; idx < 2048u; idx += 128u) {
            const uint r = idx >> 5, kk = idx & 31u;
            const uint gr = r0 + r, gk = k0 + kk;
            As[idx] = (gr < a.rows && gk < a.in) ? ld_e<TI>(X, ulong(gr) * a.ldx + gk) : 0.0f;
        }
        if (!WN) {
            for (uint idx = tid; idx < 2048u; idx += 128u) {
                const uint o = idx >> 5, kk = idx & 31u;
                const uint go = o0 + o, gk = k0 + kk;
                Bs[kk * 64u + o] = (go < a.out && gk < a.in) ? ld_e<TI>(W, ulong(go) * a.ldw + gk) : 0.0f;
            }
        } else {
            for (uint idx = tid; idx < 2048u; idx += 128u) {
                const uint kk = idx >> 6, o = idx & 63u;
                const uint go = o0 + o, gk = k0 + kk;
                Bs[idx] = (go < a.out && gk < a.in) ? ld_e<TI>(W, ulong(gk) * a.ldw + go) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 4u; ++k8) {
            simdgroup_float8x8 fa[4], fb[4];
            for (uint i = 0; i < 4u; ++i) simdgroup_load(fa[i], As + (sr + i * 8u) * 32u + k8 * 8u, 32);
            for (uint j = 0; j < 4u; ++j) simdgroup_load(fb[j], Bs + k8 * 8u * 64u + so + j * 8u, 64);
            for (uint i = 0; i < 4u; ++i)
                for (uint j = 0; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4u; ++i)
        for (uint j = 0; j < 4u; ++j) simdgroup_store(acc[i][j], smem + (sr + i * 8u) * 64u + so + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 4096u; idx += 128u) {
        const uint r = idx >> 6, o = idx & 63u;
        const uint gr = r0 + r, go = o0 + o;
        if (gr >= a.rows || go >= a.out) continue;
        st_e<TO>(Y, ulong(gr) * a.ldy + go, smem[idx]);
    }
}
// The float32 patch product (7 GFLOP for a 1,024-token picture): one output a thread, a plain sequential FMA chain over k.  Not the tiled kernel above:
// the first run of vk_gemm costs the process about 136 MB of GPU-owned memory on an M5 Max (RESULTS-OWN-3V.md D5), which a Mac with the tensor-op
// products then never needs to pay.  Grid (ceil(out / 64), rows), 64 threads.
kernel void vk_gemm_seq_f32(device const float* X [[buffer(0)]], device const float* W [[buffer(1)]], device float* Y [[buffer(2)]],
                            constant GemmArgs& a [[buffer(15)]], uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint o = tgp.x * 64u + uint(tid), r = tgp.y;
    if (o >= a.out || r >= a.rows) return;
    device const float* x = X + ulong(r) * a.ldx;
    device const float* w = W + ulong(o) * a.ldw;
    float s = 0.0f;
    for (uint k = 0; k < a.in; ++k) s = fma(x[k], w[k], s);
    Y[ulong(r) * a.ldy + o] = s;
}

// weight products: 16-bit (or float32) X and W, float32 Y
template [[host_name("vk_gemm_f32")]] kernel decltype(vk_gemm<float, float, false>) vk_gemm<float, float, false>;
template [[host_name("vk_gemm_f16")]] kernel decltype(vk_gemm<half, float, false>) vk_gemm<half, float, false>;
template [[host_name("vk_gemm_bf16")]] kernel decltype(vk_gemm<ushort, float, false>) vk_gemm<ushort, float, false>;
// attention: S = Q K^T (F16 in, F16 out) and context = P V (F16 in, V o-contiguous, float32 out)
template [[host_name("vk_gemm_s")]] kernel decltype(vk_gemm<half, half, false>) vk_gemm<half, half, false>;
template [[host_name("vk_gemm_pv")]] kernel decltype(vk_gemm<half, float, true>) vk_gemm<half, float, true>;

// ---- Gemma 4's tower (vit_gpu.cu's k_gemma_* kernels, one for one; VitCpu::encode_gemma4 has the maths step by step) ----
struct RmsArgs { int D, kind, has_w; float eps; };
// out = x / sqrt(mean(x^2) + eps) (* w when has_w), into `kind`.  One group a row.
kernel void vk_rms_cast(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device void* out [[buffer(2)]],
                        constant RmsArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    const int D = a.D;
    device const float* xr = x + (ulong) row * D;
    float s = 0.0f;
    for (int i = tid; i < D; i += nt) s += xr[i] * xr[i];
    const float scale = rsqrt(tg_sum(s, red, tid, nt) / (float) D + a.eps);
    for (int i = tid; i < D; i += nt) store_kind(out, (ulong) row * D + i, a.has_w ? xr[i] * scale * w[i] : xr[i] * scale, a.kind);
}
// x[r] += RMS norm of y[r] * w (the post norms).  One group a row.
kernel void vk_rms_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* w [[buffer(2)]],
                       constant RmsArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    const int D = a.D;
    device const float* yr = y + (ulong) row * D;
    device float* xr = x + (ulong) row * D;
    float s = 0.0f;
    for (int i = tid; i < D; i += nt) s += yr[i] * yr[i];
    const float scale = rsqrt(tg_sum(s, red, tid, nt) / (float) D + a.eps);
    for (int i = tid; i < D; i += nt) xr[i] += yr[i] * scale * w[i];
}
struct GemmaPosArgs { int T, pw, D; };
// x[t] += the x table's row of the token's column, then the y table's row of its row (T positions a table); tokens in reading order.  One group a token.
kernel void vk_gemma_pos(device float* x [[buffer(0)]], device const float* pos [[buffer(1)]], constant GemmaPosArgs& a [[buffer(15)]],
                         uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int D = a.D, ti = int(t);
    device const float* tx = pos + (ulong) (ti % a.pw) * D;
    device const float* ty = pos + (ulong) (a.T + ti / a.pw) * D;
    device float* xr = x + (ulong) t * D;
    for (int i = tid; i < D; i += nt) {
        const float v = xr[i] + tx[i];
        xr[i] = v + ty[i];
    }
}
struct GemmaQkvArgs { int row0, pw, heads, hd; float theta_scale, eps; };
// The q, k, v products of the tokens row0 .. (three [rows][D] float32 blocks: yq, yk, yv) -> q, k in QT (float or half) and v F16, each [N][D].  Per head: an
// RMS norm (q and k weighted by qn and kn, v not), then the rotary on q and k: the first half of a head turns by the token's column, the second half by its row
// (NeoX pairs i, i + hd/4).  One group a row; H <= 32.
template <typename QT>
kernel void vk_gemma_qkv(device const float* yq [[buffer(0)]], device const float* yk [[buffer(1)]], device const float* yv [[buffer(2)]],
                         device const float* qn [[buffer(3)]], device const float* kn [[buffer(4)]], device QT* q [[buffer(5)]],
                         device QT* k [[buffer(6)]], device ushort* v [[buffer(7)]], constant GemmaQkvArgs& a [[buffer(15)]],
                         uint gr [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float sc[3][32];
    const int H = a.heads, hd = a.hd, D = H * hd, half_ = hd / 2, pairs = half_ / 2;
    const int t = a.row0 + int(gr), col = t % a.pw, rowi = t / a.pw;
    device const float* qr = yq + (ulong) gr * D;
    device const float* kr = yk + (ulong) gr * D;
    device const float* vr = yv + (ulong) gr * D;
    for (int h = tid; h < H; h += nt) {
        float sq = 0.0f, sk = 0.0f, sv = 0.0f;
        for (int i = 0; i < hd; ++i) {
            const float a0 = qr[h * hd + i], a1 = kr[h * hd + i], a2 = vr[h * hd + i];
            sq += a0 * a0;
            sk += a1 * a1;
            sv += a2 * a2;
        }
        sc[0][h] = rsqrt(sq / (float) hd + a.eps);
        sc[1][h] = rsqrt(sk / (float) hd + a.eps);
        sc[2][h] = rsqrt(sv / (float) hd + a.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < D; e += nt) v[(ulong) t * D + e] = f_to_f16bits(vr[e] * sc[2][e / hd]);
    for (int p = tid; p < 2 * H * pairs; p += nt) {
        const int part = p / (H * pairs), rem = p % (H * pairs), h = rem / pairs, i = rem % pairs;
        const int A = part * half_ + i, B = A + pairs, base = h * hd;
        float ang = float(part ? rowi : col);      // p theta^i grown by multiplication, as the CPU tower's and ggml's tables are
        for (int j = 0; j < i; ++j) ang *= a.theta_scale;
        const float sn = precise::sin(ang), cs = precise::cos(ang);
        const float q0 = qr[base + A] * sc[0][h] * qn[A], q1 = qr[base + B] * sc[0][h] * qn[B];
        const float k0 = kr[base + A] * sc[1][h] * kn[A], k1 = kr[base + B] * sc[1][h] * kn[B];
        q[(ulong) t * D + base + A] = QT(q0 * cs - q1 * sn);
        q[(ulong) t * D + base + B] = QT(q0 * sn + q1 * cs);
        k[(ulong) t * D + base + A] = QT(k0 * cs - k1 * sn);
        k[(ulong) t * D + base + B] = QT(k0 * sn + k1 * cs);
    }
}
template [[host_name("vk_gemma_qkv")]] kernel decltype(vk_gemma_qkv<half>) vk_gemma_qkv<half>;
template [[host_name("vk_gemma_qkv32")]] kernel decltype(vk_gemma_qkv<float>) vk_gemma_qkv<float>;
// out = gate * sigmoid(1.702 gate) * up into `kind` (ggml's geglu_quick).  Grid-stride over n.
kernel void vk_geglu_quick_cast(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]], device void* out [[buffer(2)]],
                                constant EltArgs& a [[buffer(15)]], uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    for (ulong i = gid; i < a.n; i += gsz) {
        const float x = gate[i];
        store_kind(out, i, x * (1.0f / (1.0f + exp(-1.702f * x))) * up[i], a.kind);
    }
}
struct GemmaPoolArgs { int pw, pool, ox, t0, D, has_std, kind; float sq, eps; };
// pooled token t0 + group: the mean of its pool x pool square of x, times sq, (- std_bias) * std_scale when has_std, then an RMS norm (no weight), into `kind`.
// One group a pooled token; its D values wait in threadgroup memory (D <= 4096).
kernel void vk_gemma_pool(device const float* x [[buffer(0)]], device const float* sb [[buffer(1)]], device const float* ss [[buffer(2)]],
                          device void* out [[buffer(3)]], constant GemmaPoolArgs& a [[buffer(15)]], uint g [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float pv[4096];
    threadgroup float red[32];
    const int D = a.D, t = a.t0 + int(g), by = (t / a.ox) * a.pool, bx = (t % a.ox) * a.pool;
    const float area = float(a.pool * a.pool);
    float s2 = 0.0f;
    for (int d = tid; d < D; d += nt) {
        float s = 0.0f;
        for (int ky = 0; ky < a.pool; ++ky)
            for (int kx = 0; kx < a.pool; ++kx) s += x[((ulong) (by + ky) * a.pw + (ulong) (bx + kx)) * D + d];
        float p = s / area * a.sq;
        if (a.has_std) p = (p - sb[d]) * ss[d];
        pv[d] = p;
        s2 += p * p;
    }
    const float scale = rsqrt(tg_sum(s2, red, tid, nt) / (float) D + a.eps);
    for (int d = tid; d < D; d += nt) store_kind(out, (ulong) g * D + d, pv[d] * scale, a.kind);
}
struct SoftmaxF32Args { int n, ld_in, ld_out; };
// float32 scores S (row ld_in) -> probabilities into F16 P (row ld_out), exact float32 softmax (Gemma 4: no 1/sqrt(d) scale, so the scores are large and
// F16 scores lose the softmax's precision).  One group a row, 256 threads; the row's three passes find it in the cache after the first.
kernel void vk_softmax_f32_to16(device const float* S [[buffer(0)]], device ushort* P [[buffer(1)]], constant SoftmaxF32Args& a [[buffer(15)]],
                                uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device const float* s = S + (ulong) row * a.ld_in;
    device ushort* p = P + (ulong) row * a.ld_out;
    const int n = a.n;
    float m = -FLT_MAX;
    for (int j = tid; j < n; j += nt) m = fmax(m, s[j]);
    m = tg_max(m, red, tid, nt);
    float sum = 0.0f;
    for (int j = tid; j < n; j += nt) sum += precise::exp(s[j] - m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / tg_sum(sum, red, tid, nt);
    for (int j = tid; j < n; j += nt) p[j] = f_to_f16bits(precise::exp(s[j] - m) * inv);
}

// ---- MiniCPM-V 4.7's tower (vit_gpu.cu's k_mini_* and window kernels, one for one; VitCpu::encode_minicpmv has the maths step by step) ----
struct MiniPosArgs { int T, pw, ph, D; };
// x[t] = (x[t] + patch bias) + the position table's row of the patch's bucket: floor(T i / ph) * T + floor(T j / pw) (integer division: the same floor
// as the CPU tower's in double).  Patches in reading order.  One group a patch.
kernel void vk_mini_pos(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]], device const float* pos [[buffer(2)]],
                        constant MiniPosArgs& a [[buffer(15)]], uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort nt [[threads_per_threadgroup]]) {
    const int ti = int(t), bi = a.T * (ti / a.pw) / a.ph, bj = a.T * (ti % a.pw) / a.pw;
    device const float* pr = pos + ((ulong) bi * a.T + (ulong) bj) * a.D;
    device float* xr = x + (ulong) t * a.D;
    for (int i = tid; i < a.D; i += nt) xr[i] = (xr[i] + bias[i]) + pr[i];
}
struct MiniQkvArgs { int row0, D; float qscale; };
// The joined q, k, v product of the patches row0 .. (a row [q | k | v], 3D) plus its joined bias -> F16 [N][D] each, q times qscale.  One group a row.
kernel void vk_mini_qkv(device const float* y [[buffer(0)]], device const float* bias [[buffer(1)]], device ushort* q [[buffer(2)]],
                        device ushort* k [[buffer(3)]], device ushort* v [[buffer(4)]], constant MiniQkvArgs& a [[buffer(15)]],
                        uint gr [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int D = a.D;
    device const float* yr = y + (ulong) gr * 3 * D;
    const ulong out = (ulong) (a.row0 + int(gr)) * D;
    for (int i = tid; i < D; i += nt) {
        q[out + i] = f_to_f16bits((yr[i] + bias[i]) * a.qscale);
        k[out + i] = f_to_f16bits(yr[D + i] + bias[D + i]);
        v[out + i] = f_to_f16bits(yr[2 * D + i] + bias[2 * D + i]);
    }
}
struct WindowArgs { int pw, heads, hd; };
// The window merger's attention: softmax(q k^T) v inside each 2x2 window of a pw-wide patch grid (q scaled), F16 [N][D] in, float32 ctx [N][D] out.
// A group a window, a simdgroup a head (32 * heads threads).
kernel void vk_window_attn(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]], device const half* v [[buffer(2)]],
                           device float* ctx [[buffer(3)]], constant WindowArgs& a [[buffer(15)]], uint w [[threadgroup_position_in_grid]],
                           ushort h [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if (int(h) >= a.heads) return;
    const int hw = a.pw / 2, wi = int(w) / hw, wj = int(w) % hw, D = a.heads * a.hd;
    const ulong at[4] = {(ulong) (2 * wi) * a.pw + 2 * wj, (ulong) (2 * wi) * a.pw + 2 * wj + 1, (ulong) (2 * wi + 1) * a.pw + 2 * wj,
                         (ulong) (2 * wi + 1) * a.pw + 2 * wj + 1};
    const ulong col = (ulong) h * a.hd;
    for (int r = 0; r < 4; ++r) {
        float s[4];
        for (int b = 0; b < 4; ++b) {
            float acc = 0.0f;
            for (int d = lane; d < a.hd; d += 32) acc += float(q[at[r] * D + col + d]) * float(k[at[b] * D + col + d]);
            s[b] = simd_sum(acc);
        }
        const float mx = fmax(fmax(s[0], s[1]), fmax(s[2], s[3]));
        float sum = 0.0f;
        for (int b = 0; b < 4; ++b) {
            s[b] = precise::exp(s[b] - mx);
            sum += s[b];
        }
        for (int d = lane; d < a.hd; d += 32) {
            float acc = 0.0f;
            for (int b = 0; b < 4; ++b) acc += s[b] * float(v[at[b] * D + col + d]);
            ctx[at[r] * D + col + d] = acc / sum;
        }
    }
}
struct Gather4Args { int gw, D; };
// The four rows of each 2x2 window of a gw-wide grid side by side (top left, top right, bottom left, bottom right): out [M][4D], window t in reading
// order.  One group a window.
kernel void vk_gather4(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], constant Gather4Args& a [[buffer(15)]],
                       uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int qw = a.gw / 2, i = int(t) / qw, j = int(t) % qw, D = a.D;
    const ulong at[4] = {(ulong) (2 * i) * a.gw + 2 * j, (ulong) (2 * i) * a.gw + 2 * j + 1, (ulong) (2 * i + 1) * a.gw + 2 * j,
                         (ulong) (2 * i + 1) * a.gw + 2 * j + 1};
    for (int e = tid; e < 4 * D; e += nt) out[(ulong) t * 4 * D + e] = x[at[e / D] * D + e % D];
}
// The downsample's output: x[r] = (y[r] + bias) + the mean of the four rows of cat[r] (((p0 + p1) + p2) + p3, then a quarter, as the graph adds them).
// One group a row of D.
kernel void vk_ds_out(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* bias [[buffer(2)]],
                      device const float* cat [[buffer(3)]], constant EltArgs& a [[buffer(15)]], uint r [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int D = a.D;
    device const float* c = cat + (ulong) r * 4 * D;
    for (int i = tid; i < D; i += nt) x[(ulong) r * D + i] = (y[(ulong) r * D + i] + bias[i]) + (((c[i] + c[D + i]) + c[2 * D + i]) + c[3 * D + i]) * 0.25f;
}
// erf for float, Abramowitz and Stegun 7.1.26 (absolute error under 1.5e-7, ggml-metal's erf): the MSL standard library has none
inline float erf_as(float x) {
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * precise::exp(-x * x);
    return copysign(y, x);
}
// out = gelu_erf(y + bias) into `kind` (ggml's FFN_GELU_ERF: 0.5 x (1 + erf(x / sqrt 2))).  One group a row of D.
kernel void vk_bias_gelu_erf_cast(device const float* y [[buffer(0)]], device const float* bias [[buffer(1)]], device void* out [[buffer(2)]],
                                  constant EltArgs& a [[buffer(15)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                                  ushort nt [[threads_per_threadgroup]]) {
    const ulong base = (ulong) row * a.D;
    for (int i = tid; i < a.D; i += nt) {
        const float x = y[base + i] + bias[i];
        store_kind(out, base + i, 0.5f * x * (1.0f + erf_as(x * 0.70710678118654752f)), a.kind);
    }
}
