// poly_tests/inkling_parity.cpp - Inkling 2's Metal kernels (src/poly/inkling.metal) against CPU references written from
// inkling2.cu's arithmetic: the mixes, the attention with its bias by distance and sink (a token, a window's ring, a
// prompt batch, positions on the card), the one-token Q/K/V launch, the convolutions and clips of a call, SiTU and the
// loop controllers' gates.
#include "test_utils.hpp"
#include "inkling2_kernels.hpp"
#include "strata/gpu/metal.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

using namespace strata::poly;
namespace kernels = strata::kernels;

namespace {

float sigm(float x) { return 1 / (1 + std::exp(-x)); }

uint16_t to_half(float f) {   // round to nearest even, the halves the caches hold
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int e = int((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e >= 31) return uint16_t(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);
        m |= 0x800000;
        const uint32_t shift = uint32_t(14 - e);
        uint32_t h = m >> shift;
        if ((m >> (shift - 1)) & 1 && ((m & ((1u << (shift - 1)) - 1)) || (h & 1))) ++h;
        return uint16_t(sign | h);
    }
    uint32_t h = uint32_t(e << 10) | (m >> 13);
    if ((m & 0x1000) && ((m & 0xfff) || (h & 1))) ++h;
    return uint16_t(sign | h);
}

float from_half(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    int e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff, x;
    if (e == 0) {
        if (!m) x = sign;
        else {
            e = 1;
            while (!(m & 0x400)) m <<= 1, --e;
            x = sign | uint32_t(e - 15 + 127) << 23 | (m & 0x3ff) << 13;
        }
    } else if (e == 31) x = sign | 0x7f800000 | m << 13;
    else x = sign | uint32_t(e - 15 + 127) << 23 | m << 13;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---- the mixes ----
void mix_ref(const std::vector<std::vector<float>>& s, const std::vector<float>& q, int nq, int n, int rows,
             std::vector<std::vector<float>>& out) {
    out.assign((size_t) nq, std::vector<float>((size_t) rows * n));
    for (int r = 0; r < rows; ++r) {
        std::vector<float> inv(s.size());
        std::vector<std::vector<float>> dot(s.size(), std::vector<float>((size_t) nq));
        for (size_t k = 0; k < s.size(); ++k) {
            double ss = 0;
            for (int i = 0; i < n; ++i) {
                const float v = s[k][(size_t) r * n + i];
                ss += v * v;
                for (int j = 0; j < nq; ++j) dot[k][(size_t) j] += v * q[(size_t) j * n + i];
            }
            inv[k] = 1 / std::sqrt(float(ss) / n + 1e-6f);
        }
        for (int j = 0; j < nq; ++j) {
            std::vector<float> v(s.size());
            float m = -INFINITY, total = 0;
            for (size_t k = 0; k < s.size(); ++k) m = std::max(m, v[k] = dot[k][(size_t) j] * inv[k] / std::sqrt(float(n)));
            for (size_t k = 0; k < s.size(); ++k) total += v[k] = std::exp(v[k] - m);
            for (int i = 0; i < n; ++i) {
                float acc = 0;
                for (size_t k = 0; k < s.size(); ++k) acc += v[k] / total * inv[k] * s[k][(size_t) r * n + i];
                out[(size_t) j][(size_t) r * n + i] = acc;
            }
        }
    }
}

void norm_ref(const float* x, const float* w, float* y, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1 / std::sqrt(float(ss) / n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (w ? w[i] : 1.f);
}

int mixes() {
    const int n = 2304, rows = 2, ns = 3, nq = 3;
    std::vector<std::vector<float>> s;
    for (int k = 0; k < ns; ++k) {
        auto v = values((size_t) rows * n, 1.f + k);
        for (size_t i = 0; i < v.size(); ++i) v[i] += 0.3f * std::cos(float(i * (k + 2)) * 0.011f);
        s.push_back(v);
    }
    auto q = values((size_t) nq * n, .05f), w = values(n, .9f);
    std::vector<std::vector<float>> ref;
    mix_ref(s, q, nq, n, rows, ref);
    Buffer S0(s[0]), S1(s[1]), S2(s[2]), Q(q), W(w), O0(rows * n * 4), O1(rows * n * 4), O2(rows * n * 4),
        N(rows * n * 4), Q8(kernels::native_q8_1_bytes(n));
    InkStates st;
    st.p[0] = S0.as(), st.p[1] = S1.as(), st.p[2] = S2.as(), st.n = 3;
    float* out[3] = {O0.as(), O1.as(), O2.as()};
    ink_mix_norm(st, Q.as(), nq, out, n, rows, W.as(), N.as(), 1e-6f, nullptr);
    int bad = compare(O0.read(rows * n), ref[0], "ink_mix out 0") + compare(O1.read(rows * n), ref[1], "ink_mix out 1") +
              compare(O2.read(rows * n), ref[2], "ink_mix out 2");
    std::vector<float> nref((size_t) rows * n);
    for (int r = 0; r < rows; ++r) norm_ref(ref[0].data() + r * n, w.data(), nref.data() + r * n, n, 1e-6f);
    bad += compare(N.read(rows * n), nref, "ink_mix_norm");
    // one row with its q8: the norm's rows quantized as native_quantize_q8_1() does it
    ink_mix_norm(st, Q.as(), 1, out, n, 1, W.as(), N.as(), 1e-6f, nullptr, Q8.as(), true);
    Buffer Q8b(kernels::native_q8_1_bytes(n));
    kernels::native_quantize_q8_1(N.as(), Q8b.as(), n, 1, nullptr);
    const auto a = Q8.read<uint8_t>(kernels::native_q8_1_bytes(n)), b = Q8b.read<uint8_t>(kernels::native_q8_1_bytes(n));
    const int q8bad = a != b;
    std::printf("ink_mix_norm q8: %s\n", q8bad ? "DIFFERS from native_quantize_q8_1" : "the same bytes");
    return bad + q8bad;
}

// ---- the attention ----
struct AttnCase {
    const char* name;
    int rows, pos, window, ring, extent;
    bool at;
    bool batch = false;   ///< ink_attend_batch(): the matrix units' kernel, q and the weights in fp16
};

int attention(const AttnCase& c) {
    const int nh = 6, nkv = 2, dim = 128, nrel = 16, per = nh / nkv;
    const int cells = c.ring ? c.ring : c.pos + c.rows;
    const float scale = 1 / std::sqrt(float(dim));
    auto q = values((size_t) c.rows * nh * dim, .6f), r = values((size_t) c.rows * nh * nrel, .4f),
         sink = values((size_t) c.rows * nh, 1.5f), bias = values((size_t) nrel * c.extent, .3f),
         gate = values((size_t) c.rows * nh * dim, 2.f);
    auto kf = values((size_t) cells * nkv * dim, .8f), vf = values((size_t) cells * nkv * dim, 1.1f);
    for (size_t i = 0; i < kf.size(); ++i) kf[i] += 0.2f * std::cos(float(i) * 0.013f);
    std::vector<uint16_t> kh(kf.size()), vh(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) kh[i] = to_half(kf[i]), vh[i] = to_half(vf[i]);
    std::vector<float> ref((size_t) c.rows * nh * dim);
    for (int t = 0; t < c.rows; ++t) {
        const long p = c.pos + t, lo = c.window > 0 ? std::max<long>(0, p - c.window + 1) : 0;
        for (int h = 0; h < nh; ++h) {
            const int g = h / per;
            std::vector<double> sc;
            double m = 0;   // the extra term's score
            for (long j = lo; j <= p; ++j) {
                const size_t cell = size_t(c.ring ? j % c.ring : j);
                double d = 0;
                for (int i = 0; i < dim; ++i)
                    d += double(q[((size_t) t * nh + h) * dim + i]) * scale * from_half(kh[(cell * nkv + g) * dim + i]);
                const long dist = p - j;
                if (dist < c.extent)
                    for (int i = 0; i < nrel; ++i) d += double(r[((size_t) t * nh + h) * nrel + i]) * bias[dist * nrel + i];
                d -= sink[(size_t) t * nh + h];
                sc.push_back(d);
                m = std::max(m, d);
            }
            double total = std::exp(-m);
            std::vector<double> acc(dim);
            for (long j = lo; j <= p; ++j) {
                const size_t cell = size_t(c.ring ? j % c.ring : j);
                const double w = std::exp(sc[size_t(j - lo)] - m);
                total += w;
                for (int i = 0; i < dim; ++i) acc[i] += w * from_half(vh[(cell * nkv + g) * dim + i]);
            }
            for (int i = 0; i < dim; ++i) {
                const size_t at = ((size_t) t * nh + h) * dim + i;
                ref[at] = float(acc[i] / total) * sigm(gate[at]);
            }
        }
    }
    Buffer Q(q), R(r), SK(sink), B(bias), G(gate), K(kh.size() * 2), V(vh.size() * 2), O(ref.size() * 4),
        SC(4 * 1024), AT(2 * c.rows * 8), Q8(kernels::native_q8_1_bytes(nh * dim));
    K.put(kh.data(), kh.size() * 2);
    V.put(vh.data(), vh.size() * 2);
    std::vector<long long> at(2 * (size_t) c.rows);
    for (int t = 0; t < c.rows; ++t) at[2 * t] = c.pos + t;
    AT.put(at.data(), at.size() * 8);
    InkAttn a;
    a.q = Q.as(), a.r = R.as(), a.sink = SK.as(), a.keys = K.as<uint16_t>(), a.vals = V.as<uint16_t>();
    a.rel_bias = B.as(), a.gate = G.as(), a.out = O.as(), a.score = SC.as(), a.score_n = 1024;
    a.pos = c.at ? 0 : c.pos, a.at = c.at ? AT.as<long long>() : nullptr, a.room = c.at ? c.pos + c.rows : 0;
    a.rows = c.rows, a.n_head = nh, a.n_kv = nkv, a.dim = dim, a.n_rel = nrel, a.extent = c.extent;
    a.window = c.window, a.ring = c.ring, a.scale = scale;
    if (c.rows == 1) a.q8 = Q8.as();
    if (c.batch) {
        strata::gpu::metal::set_optimize(true);
        ink_attend_batch(a, nullptr);
        strata::gpu::metal::set_optimize(false);
    } else {
        ink_attend(a, nullptr);
    }
    int bad = compare(O.read(ref.size()), ref, c.name, c.batch ? 5e-3f : 2e-4f);
    if (c.rows == 1) {
        Buffer Q8b(kernels::native_q8_1_bytes(nh * dim));
        kernels::native_quantize_q8_1(O.as(), Q8b.as(), nh * dim, 1, nullptr);
        const size_t nb = kernels::native_q8_1_bytes(nh * dim);
        const bool same = Q8.read<uint8_t>(nb) == Q8b.read<uint8_t>(nb);
        std::printf("%s q8: %s\n", c.name, same ? "the same bytes" : "DIFFERS from native_quantize_q8_1");
        bad += !same;
    }
    return bad;
}

// ---- the convolutions ----
float conv_ref(const std::vector<float>& x, const std::vector<float>& state, const std::vector<float>& w, const float* dyn,
               int ch, int heads, int taps, int t, int c, float norm) {
    float past[8];
    for (int k = 0; k < taps; ++k) {
        const int u = t - k;
        past[k] = u >= 0 ? x[(size_t) u * ch + c] : state[(size_t) (taps - 1 + u) * ch + c];
    }
    float acc = 0;
    for (int j = 0; j < taps; ++j) acc += past[taps - 1 - j] * w[(size_t) c * taps + j];
    if (dyn) {
        const int head = c / (ch / heads);
        for (int lag = 0; lag < taps; ++lag) acc += past[lag] * dyn[((size_t) t * heads + head) * taps + lag];
    }
    return (past[0] + acc) * norm;
}

void state_ref(const std::vector<float>& x, std::vector<float>& state, int ch, int taps, int rows) {
    const int keep = taps - 1;
    std::vector<float> s = state;
    for (int k = 0; k < keep; ++k)
        for (int c = 0; c < ch; ++c) {
            const int u = rows - keep + k;
            s[(size_t) k * ch + c] = u >= 0 ? x[(size_t) u * ch + c] : state[(size_t) (keep + u) * ch + c];
        }
    state = s;
}

int convolutions(int rows) {
    const int heads = 8, dim = 128, ch = heads * dim, taps = 4;
    auto x = values((size_t) rows * ch, .9f), state = values((size_t) (taps - 1) * ch, .7f), w = values((size_t) ch * taps, .3f),
         dyn = values((size_t) rows * heads * taps, .2f);
    std::vector<float> out((size_t) rows * ch);
    const float norm = 1 / std::sqrt(float(2 * taps + 1));
    for (int t = 0; t < rows; ++t)
        for (int c = 0; c < ch; ++c) out[(size_t) t * ch + c] = conv_ref(x, state, w, dyn.data(), ch, heads, taps, t, c, norm);
    // snapshots after tokens 0 and 2
    std::vector<float> snap(2 * state.size());
    for (int k = 0; k < 2; ++k) {
        std::vector<float> s = state;
        state_ref(x, s, ch, taps, k == 0 ? 1 : 3);
        std::copy(s.begin(), s.end(), snap.begin() + (long) k * state.size());
    }
    std::vector<float> after = state;
    state_ref(x, after, ch, taps, rows);
    Buffer X(x), ST(state), W(w), D(dyn), O(out.size() * 4), SN(snap.size() * 4);
    ink_conv(X.as(), ST.as(), W.as(), D.as(), O.as(), ch, heads, taps, rows, nullptr, SN.as(), state.size(), 0b101u);
    char label[64];
    std::snprintf(label, sizeof label, "ink_conv %d rows", rows);
    int bad = compare(O.read(out.size()), out, label);
    std::snprintf(label, sizeof label, "ink_conv %d rows: state", rows);
    bad += compare(ST.read(after.size()), after, label);
    std::snprintf(label, sizeof label, "ink_conv %d rows: snapshots", rows);
    return bad + compare(SN.read(snap.size()), snap, label);
}

// ---- a convolution's add fused with the next mix (ink_conv_mix(), ink_conv_mix_rows()) ----
// Inkling 2 Mini's width; the launch fits the pipeline's threadgroup limit (832 on an M1 Max, 1,024 elsewhere).
int conv_mixes(int rows, int nq) {
    const int n = 2304, taps = 4, ns = 3;
    const float a = .5f, eps = 1e-6f, cnorm = 1 / std::sqrt(float(taps + 1));
    auto x = values((size_t) rows * n, .9f), state = values((size_t) (taps - 1) * n, .7f), w = values((size_t) n * taps, .3f),
         acc = values((size_t) rows * n, 1.2f), s0 = values((size_t) rows * n, 1.f), s1 = values((size_t) rows * n, 1.5f),
         q = values((size_t) nq * n, .05f), nw = values(n, .9f);
    std::vector<float> ao((size_t) rows * n), acc2 = acc;
    for (int t = 0; t < rows; ++t)
        for (int i = 0; i < n; ++i) {
            const size_t k = (size_t) t * n + i;
            ao[k] = conv_ref(x, state, w, nullptr, n, 1, taps, t, i, cnorm);
            acc2[k] = acc[k] + ao[k] * a;
        }
    std::vector<std::vector<float>> ref;
    mix_ref({s0, s1, acc2}, q, nq, n, rows, ref);
    std::vector<float> nref((size_t) rows * n);
    for (int r = 0; r < rows; ++r) norm_ref(ref[0].data() + (size_t) r * n, nw.data(), nref.data() + (size_t) r * n, n, eps);
    const size_t bytes = (size_t) rows * n * 4;
    Buffer X(x), ST(state), W(w), AO(bytes), AC(acc), S0(s0), S1(s1), Q(q), NW(nw), N(bytes), O0(bytes), O1(bytes), O2(bytes);
    InkStates st;
    st.p[0] = S0.as(), st.p[1] = S1.as(), st.p[2] = AC.as(), st.n = ns;
    float* out[3] = {O0.as(), O1.as(), O2.as()};
    char label[64];
    const char* what = rows == 1 ? "ink_conv_mix" : "ink_conv_mix_rows";
    if (rows == 1)
        ink_conv_mix(X.as(), ST.as(), W.as(), AO.as(), AC.as(), nullptr, a, 0, taps, st, Q.as(), nq, out, n, NW.as(),
                     N.as(), eps, nullptr);
    else
        ink_conv_mix_rows(X.as(), ST.as(), W.as(), AO.as(), AC.as(), nullptr, a, 0, taps, st, Q.as(), nq, out, n, rows,
                          NW.as(), N.as(), eps, nullptr);
    std::snprintf(label, sizeof label, "%s %d rows, %d queries: conv", what, rows, nq);
    int bad = compare(AO.read(ao.size()), ao, label);
    std::snprintf(label, sizeof label, "%s %d rows, %d queries: acc", what, rows, nq);
    bad += compare(AC.read(acc2.size()), acc2, label);
    Buffer* outs[3] = {&O0, &O1, &O2};
    for (int j = 0; j < nq; ++j) {
        std::snprintf(label, sizeof label, "%s %d rows, %d queries: out %d", what, rows, nq, j);
        bad += compare(outs[j]->read(rows * n), ref[(size_t) j], label);
    }
    std::snprintf(label, sizeof label, "%s %d rows, %d queries: norm", what, rows, nq);
    bad += compare(N.read(nref.size()), nref, label);
    if (rows == 1) {
        std::vector<float> after = state;
        state_ref(x, after, n, taps, 1);
        std::snprintf(label, sizeof label, "%s: state", what);
        bad += compare(ST.read(after.size()), after, label);
    }
    return bad;
}

// ---- one token's Q, K and V (ink_qkv()) and a call's clips (ink_clips()) ----
void clip_ref(float* v, const float* sc, float thr, int dim, float alpha, float floor_, long p) {
    double ss = 0;
    for (int i = 0; i < dim; ++i) {
        const float y = sc ? v[i] * sc[i] : v[i];
        ss += y * y;
    }
    float f = std::min(thr / (std::sqrt(float(ss)) + 1e-6f), 1.f);
    if (alpha > 0) f *= 1 + alpha * std::log(std::max(1.f, float(p + 1) / floor_));
    for (int i = 0; i < dim; ++i) v[i] = (sc ? v[i] * sc[i] : v[i]) * f;
}

void rms_ref(float* v, int dim, float eps) {
    double ss = 0;
    for (int i = 0; i < dim; ++i) ss += v[i] * v[i];
    const float f = 1 / std::sqrt(float(ss) / dim + eps);
    for (int i = 0; i < dim; ++i) v[i] *= f;
}

int qkv() {
    const int nh = 6, nkv = 2, dim = 128, nrel = 16, taps = 4, ch = nkv * dim, ring = 40;
    const long pos = 9000;
    const float thr = 3.f, rthr = 2.f, alpha = .1f, floor_ = 8192.f, eps = 1e-6f;
    auto q = values((size_t) nh * dim, 2.f), qs = values((size_t) nh * dim, 1.f), r = values((size_t) nh * nrel, 3.f),
         rs = values((size_t) nh * nrel, .8f), kraw = values(ch, 1.3f), vraw = values(ch, .7f),
         kst = values((size_t) (taps - 1) * ch, .5f), vst = values((size_t) (taps - 1) * ch, .6f), kw = values((size_t) ch * taps, .4f),
         vw = values((size_t) ch * taps, .35f), kd = values((size_t) nkv * taps, .2f), vd = values((size_t) nkv * taps, .25f);
    std::vector<float> rq = q, rr = r, rk(ch), rv(ch), rkst = kst, rvst = vst;
    for (int h = 0; h < nh; ++h) clip_ref(rq.data() + h * dim, qs.data() + h * dim, thr, dim, alpha, floor_, pos);
    for (int h = 0; h < nh; ++h) clip_ref(rr.data() + h * nrel, rs.data() + h * nrel, rthr, nrel, alpha, floor_, pos);
    const float norm = 1 / std::sqrt(float(2 * taps + 1));
    for (int c = 0; c < ch; ++c) {
        rk[c] = conv_ref(kraw, kst, kw, kd.data(), ch, nkv, taps, 0, c, norm);
        rv[c] = conv_ref(vraw, vst, vw, vd.data(), ch, nkv, taps, 0, c, norm);
    }
    state_ref(kraw, rkst, ch, taps, 1);
    state_ref(vraw, rvst, ch, taps, 1);
    for (int h = 0; h < nkv; ++h) {
        clip_ref(rk.data() + h * dim, nullptr, thr, dim, 0, 1, pos);
        rms_ref(rv.data() + h * dim, dim, eps);
    }
    std::vector<float> keys((size_t) ring * ch), vals = keys;
    const size_t cell = size_t(pos % ring);
    for (int i = 0; i < ch; ++i) keys[cell * ch + i] = from_half(to_half(rk[i])), vals[cell * ch + i] = from_half(to_half(rv[i]));
    Buffer Q(q), QS(qs), R(r), RS(rs), KR(kraw), VR(vraw), KS(kst), VS(vst), KW(kw), VW(vw), KD(kd), VD(vd), K(ch * 4),
        V(ch * 4), KC((size_t) ring * ch * 2), VC((size_t) ring * ch * 2);
    std::vector<uint16_t> zeros((size_t) ring * ch);
    KC.put(zeros.data(), zeros.size() * 2);
    VC.put(zeros.data(), zeros.size() * 2);
    InkQkv a;
    a.q = Q.as(), a.q_scale = QS.as(), a.r = R.as(), a.r_scale = RS.as(), a.kraw = KR.as(), a.vraw = VR.as();
    a.kstate = KS.as(), a.vstate = VS.as(), a.kw = KW.as(), a.vw = VW.as(), a.kdyn = KD.as(), a.vdyn = VD.as();
    a.k = K.as(), a.v = V.as(), a.keys = KC.as<uint16_t>(), a.vals = VC.as<uint16_t>(), a.pos = pos;
    a.qk_thr = thr, a.rel_thr = rthr, a.alpha = alpha, a.floor = floor_, a.eps = eps;
    a.n_head = nh, a.n_kv = nkv, a.dim = dim, a.n_rel = nrel, a.taps = taps, a.ring = ring;
    ink_qkv(a, nullptr);
    int bad = compare(Q.read(rq.size()), rq, "ink_qkv q") + compare(R.read(rr.size()), rr, "ink_qkv r") +
              compare(K.read(ch), rk, "ink_qkv k") + compare(V.read(ch), rv, "ink_qkv v") +
              compare(KS.read(rkst.size()), rkst, "ink_qkv k state") + compare(VS.read(rvst.size()), rvst, "ink_qkv v state");
    const auto half_rows = [](const Buffer& b, size_t n) {
        std::vector<float> f(n);
        const auto h = b.read<uint16_t>(n);
        for (size_t i = 0; i < n; ++i) f[i] = from_half(h[i]);
        return f;
    };
    bad += compare(half_rows(KC, keys.size()), keys, "ink_qkv key cache") + compare(half_rows(VC, vals.size()), vals, "ink_qkv value cache");

    // the same clips for a call of 5 tokens (ink_clips()), positions on the card, the caches a ring
    const int rows = 5;
    auto q5 = values((size_t) rows * nh * dim, 2.f), k5 = values((size_t) rows * ch, 1.5f), v5 = values((size_t) rows * ch, .9f),
         r5 = values((size_t) rows * nh * nrel, 3.f);
    std::vector<float> eq = q5, ek = k5, ev = v5, er = r5, ekeys((size_t) ring * ch), evals = ekeys;
    const long p0 = 77;
    for (int t = 0; t < rows; ++t) {
        for (int h = 0; h < nh; ++h) {
            clip_ref(eq.data() + ((size_t) t * nh + h) * dim, qs.data() + h * dim, thr, dim, alpha, floor_, p0 + t);
            clip_ref(er.data() + ((size_t) t * nh + h) * nrel, rs.data() + h * nrel, rthr, nrel, alpha, floor_, p0 + t);
        }
        for (int h = 0; h < nkv; ++h) {
            clip_ref(ek.data() + ((size_t) t * nkv + h) * dim, nullptr, thr, dim, 0, 1, p0 + t);
            rms_ref(ev.data() + ((size_t) t * nkv + h) * dim, dim, eps);
        }
        const size_t c5 = size_t((p0 + t) % ring);
        for (int i = 0; i < ch; ++i) {
            ekeys[c5 * ch + i] = from_half(to_half(ek[(size_t) t * ch + i]));
            evals[c5 * ch + i] = from_half(to_half(ev[(size_t) t * ch + i]));
        }
    }
    Buffer Q5(q5), K5(k5), V5(v5), R5(r5), AT(2 * rows * 8);
    std::vector<long long> at(2 * rows);
    for (int t = 0; t < rows; ++t) at[2 * t] = p0 + t;
    AT.put(at.data(), at.size() * 8);
    KC.put(zeros.data(), zeros.size() * 2);
    VC.put(zeros.data(), zeros.size() * 2);
    InkClips c;
    float* xs[4] = {Q5.as(), K5.as(), V5.as(), R5.as()};
    const float* scs[4] = {QS.as(), nullptr, nullptr, RS.as()};
    const float thrs[4] = {thr, thr, 0, rthr}, als[4] = {alpha, 0, 0, alpha}, fls[4] = {floor_, 1, 1, floor_};
    const int dims[4] = {dim, dim, dim, nrel}, hs[4] = {nh, nkv, nkv, nh};
    for (int j = 0; j < 4; ++j) {
        c.x[j] = xs[j], c.scale[j] = scs[j], c.thr[j] = thrs[j], c.alpha[j] = als[j], c.floor[j] = fls[j];
        c.dim[j] = dims[j], c.heads[j] = hs[j];
    }
    c.rms[2] = true, c.eps = eps, c.at = AT.as<long long>(), c.rows = rows, c.cache[1] = KC.as<uint16_t>();
    c.cache[2] = VC.as<uint16_t>(), c.ring = ring;
    ink_clips(c, nullptr);
    bad += compare(Q5.read(eq.size()), eq, "ink_clips q") + compare(K5.read(ek.size()), ek, "ink_clips k") +
           compare(V5.read(ev.size()), ev, "ink_clips v") + compare(R5.read(er.size()), er, "ink_clips r") +
           compare(half_rows(KC, ekeys.size()), ekeys, "ink_clips key ring") +
           compare(half_rows(VC, evals.size()), evals, "ink_clips value ring");
    return bad;
}

// ---- SiTU and the gates ----
int situ_gates() {
    const int n = 1152, rows = 3, experts = 5, rank = 64, ne = 2304;
    const float beta = 4, lb = 25;
    auto gu = values((size_t) rows * 2 * n, 9.f), gs = values(experts, .5f), us = values(experts, .7f);
    for (float& v : gs) v += 1.f;
    for (float& v : us) v += 1.f;
    const int ids[rows] = {4, 0, 2};
    std::vector<float> ref((size_t) rows * n);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n; ++i) {
            const float g = gu[(size_t) r * 2 * n + i] * gs[ids[r]];
            const float u = lb * std::tanh(gu[(size_t) r * 2 * n + n + i] * us[ids[r]] / lb);
            ref[(size_t) r * n + i] = beta * std::tanh(g / beta) * sigm(g) * u;
        }
    Buffer GU(gu), GS(gs), US(us), ID(sizeof ids), O(ref.size() * 4), Q8(kernels::native_q8_1_bytes(n, rows)),
        Q8b(kernels::native_q8_1_bytes(n, rows));
    ID.put(ids, sizeof ids);
    ExpertScales es;
    es.gate = GS.as(), es.up = US.as(), es.ids = ID.as<int32_t>();
    situ_rows(GU.as(), O.as(), n, rows, beta, lb, nullptr, Q8.as(), &es);
    int bad = compare(O.read(ref.size()), ref, "situ_rows", 1e-4f);
    kernels::native_quantize_q8_1(O.as(), Q8b.as(), n * rows, 1, nullptr);
    const size_t nb = kernels::native_q8_1_bytes(n, rows);
    const bool same = Q8.read<uint8_t>(nb) == Q8b.read<uint8_t>(nb);
    std::printf("situ_rows q8: %s\n", same ? "the same bytes" : "DIFFERS from native_quantize_q8_1");
    bad += !same;

    // three controllers' gain and forget products and gates (ink_control3_gate()), one cur shared as out[0]
    auto cur = values(ne, 1.f), m0 = values(ne, .8f), m1 = values(ne, .9f), m2 = values(ne, 1.1f);
    std::vector<std::vector<float>> ch, gw, fw;
    for (int j = 0; j < 3; ++j) {
        ch.push_back(values(rank, .3f + .1f * j));
        gw.push_back(values((size_t) ne * rank, .05f * (j + 1)));
        fw.push_back(values((size_t) ne * rank, .04f * (j + 1)));
    }
    const std::vector<float>* msg[3] = {&m0, &m1, &m2};
    std::vector<std::vector<float>> want(3, std::vector<float>(ne));
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < ne; ++i) {
            float g = 0, f = 0;
            for (int k = 0; k < rank; ++k) {
                g += gw[j][(size_t) i * rank + k] * ch[j][k];
                f += fw[j][(size_t) i * rank + k] * ch[j][k];
            }
            want[j][i] = cur[i] - std::tanh(f) * cur[i] + std::tanh(g) * (*msg[j])[i];
        }
    Buffer C(cur), M0(m0), M1(m1), M2(m2), C0(ch[0]), C1(ch[1]), C2(ch[2]), G0(gw[0]), G1(gw[1]), G2(gw[2]), F0(fw[0]),
        F1(fw[1]), F2(fw[2]), O1(ne * 4), O2(ne * 4);
    const float* cs[3] = {C.as(), C.as(), C.as()};
    const float* ms[3] = {M0.as(), M1.as(), M2.as()};
    const float* chs[3] = {C0.as(), C1.as(), C2.as()};
    const void* gws[3] = {G0.as(), G1.as(), G2.as()};
    const void* fws[3] = {F0.as(), F1.as(), F2.as()};
    float* outs[3] = {C.as(), O1.as(), O2.as()};
    ink_control3_gate(cs, ms, chs, gws, fws, outs, ne, rank, nullptr, 3, false);
    bad += compare(C.read(ne), want[0], "ink_control3_gate 0", 1e-4f) + compare(O1.read(ne), want[1], "ink_control3_gate 1", 1e-4f) +
           compare(O2.read(ne), want[2], "ink_control3_gate 2", 1e-4f);

    // the controllers' norms and down products (ink_ctrl_down()), 2 rows, F32; then gemv_set() of their gain and
    // forget products
    const int rows2 = 2;
    auto c2 = values((size_t) rows2 * ne, 1.3f), g2 = values((size_t) rows2 * ne, .7f);
    std::vector<std::vector<float>> down, dref;
    for (int j = 0; j < 3; ++j) down.push_back(values((size_t) rank * 2 * ne, .02f * (j + 1)));
    for (int j = 0; j < 3; ++j) {
        std::vector<float> want2((size_t) rows2 * rank);
        for (int r = 0; r < rows2; ++r) {
            std::vector<float> cat(2 * (size_t) ne);
            norm_ref(c2.data() + (size_t) r * ne, nullptr, cat.data(), ne, 1e-6f);
            norm_ref(g2.data() + (size_t) r * ne, nullptr, cat.data() + ne, ne, 1e-6f);
            for (int o = 0; o < rank; ++o) {
                double acc = 0;
                for (int i = 0; i < 2 * ne; ++i) acc += down[j][(size_t) o * 2 * ne + i] * cat[i];
                want2[(size_t) r * rank + o] = float(acc);
            }
        }
        dref.push_back(want2);
    }
    Buffer CC(c2), GG(g2), D0(down[0]), D1(down[1]), D2(down[2]), H0(rows2 * rank * 4), H1(rows2 * rank * 4),
        H2(rows2 * rank * 4);
    const float* cc[3] = {CC.as(), CC.as(), CC.as()};
    const float* gg[3] = {GG.as(), GG.as(), GG.as()};
    const void* dd[3] = {D0.as(), D1.as(), D2.as()};
    float* hh[3] = {H0.as(), H1.as(), H2.as()};
    if (!ink_ctrl_down(cc, gg, dd, hh, 3, ne, rank, rows2, nullptr, false)) {
        std::printf("ink_ctrl_down: not taken\n");
        return bad + 1;
    }
    bad += compare(H0.read(rows2 * rank), dref[0], "ink_ctrl_down 0", 1e-4f) +
           compare(H1.read(rows2 * rank), dref[1], "ink_ctrl_down 1", 1e-4f) +
           compare(H2.read(rows2 * rank), dref[2], "ink_ctrl_down 2", 1e-4f);
    std::vector<float> gref((size_t) rows2 * ne);
    for (int r = 0; r < rows2; ++r)
        for (int i = 0; i < ne; ++i) {
            double acc = 0;
            for (int k = 0; k < rank; ++k) acc += gw[0][(size_t) i * rank + k] * dref[0][(size_t) r * rank + k];
            gref[(size_t) r * ne + i] = float(acc);
        }
    Buffer GO(rows2 * ne * 4), FO(rows2 * ne * 4);
    const float* sw[2] = {G0.as(), F0.as()};
    const float* sx[2] = {H0.as(), H0.as()};
    float* sy[2] = {GO.as(), FO.as()};
    const int so[2] = {ne, ne};
    gemv_set(sw, sx, sy, so, 2, rank, rows2, nullptr);
    bad += compare(GO.read(gref.size()), gref, "gemv_set", 1e-4f);
    return bad;
}

int loopcd_extrap() {
    const size_t n = 3 * 2304 + 17;
    const auto dst0 = values(n, 3.0f), ref0 = values(n, 2.5f);
    Buffer D(dst0), R(ref0);
    ink_loopcd_extrap(D.as(), R.as(), 0.0f, n, nullptr);
    int bad = compare(D.read(n), dst0, "ink_loopcd_extrap (omega 0 no-op)", 0.0f);
    const float omega = 0.2f;
    ink_loopcd_extrap(D.as(), R.as(), omega, n, nullptr);
    std::vector<float> want(n);
    for (size_t i = 0; i < n; ++i) want[i] = dst0[i] + omega * (dst0[i] - ref0[i]);
    bad += compare(D.read(n), want, "ink_loopcd_extrap (omega 0.2)", 1e-5f);
    return bad;
}

}  // namespace

int main() {
    int bad = mixes();
    // a token over the whole context, which ink_attend() splits into runs; a window's ring; a prompt batch; a recorded
    // call (positions on the card) over a ring; a short context
    bad += attention({"ink_attend: a token, 3,000 keys", 1, 2999, 0, 0, 1024, false});
    bad += attention({"ink_attend: a token, a window of 256 in a ring", 1, 700, 256, 260, 256, false});
    bad += attention({"ink_attend: a token, 20 keys", 1, 19, 0, 0, 1024, false});
    bad += attention({"ink_attend: a prompt batch of 40", 40, 100, 0, 0, 64, false});
    bad += attention({"ink_attend: 4 tokens, positions on the card, a ring", 4, 500, 128, 136, 128, true});
    bad += attention({"ink_attend_batch: 100 tokens after 300", 100, 300, 0, 0, 64, false, true});
    bad += attention({"ink_attend_batch: 70 tokens, a window of 128 in a ring", 70, 500, 128, 198, 128, false, true});
    bad += convolutions(5) + convolutions(40);
    bad += conv_mixes(1, 1) + conv_mixes(1, 3) + conv_mixes(4, 1) + conv_mixes(4, 3);
    bad += qkv();
    bad += situ_gates();
    bad += loopcd_extrap();
    std::printf("%s\n", bad ? "FAILED" : "all passed");
    return bad != 0;
}
