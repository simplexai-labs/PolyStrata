// metal/src/poly/inkling_flash.metal - ink_attend_batch() on the matrix units (Metal 4's matmul2d): a prompt batch's
// attention with Inkling 2's bias by distance and sink, no score written out.  flash.metal's poly_flash with the
// scores ink_attend() makes: scale * q.k (q rounded to fp16, the product summed in floats), plus r . rel_bias[d] for a
// key at distance d under `extent`, less the sink; a softmax with one more term of score 0 (its running maximum and
// sum start at 0 and 1: the term of a key whose value is 0), P in fp16, then the output gate.  A threadgroup takes QB
// of the call's tokens for one query head; SIMD group g owns head values 64 g .. 64 g + 63.  K and V are fp16 tensors
// matmul2d reads where they lie: a full layer's cache, or (a ring) ink_flash_kv's copy of the call's keys, key 0 the
// call's first.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// inkling.metal's Args.  p: q, r, sink, keys, vals, rel_bias, gate, out, the keys' copy, the values'; n: rows, n_head,
// n_kv, dim, n_rel, extent, window, ring; f[0] scale; pos.
struct Args {
    device void* p[72];
    long pos;
    ulong stride;
    int n[24];
    float f[16];
};
constant constexpr int QB = 16, KC = 64, NR = 16;   // tokens a threadgroup, keys a step, relative values

inline long call_first(constant Args& a) { return a.n[6] > 0 ? max(0L, a.pos - a.n[6] + 1) : 0; }

// The call's keys and values from call_first() to its last token's, fp16 [key][kv head][value] (p[8], p[9])
kernel void ink_flash_kv(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    const ulong row = ulong(a.n[2]) * a.n[3], keys = ulong(a.pos + a.n[0] - call_first(a));
    if (i >= keys * row) return;
    const long at = call_first(a) + long(i / row);
    const ulong src = ulong(a.n[7] ? at % a.n[7] : at) * row + i % row;
    static_cast<device half*>(a.p[8])[i] = static_cast<device half*>(a.p[3])[src];
    static_cast<device half*>(a.p[9])[i] = static_cast<device half*>(a.p[4])[src];
}

kernel void ink_flash_128(constant Args& a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int D = 128, NSG = D / 64, NTH = NSG * 32;
    threadgroup half qs[QB * D];          // the tokens' queries: [token][value]
    threadgroup float sp[NSG][QB * KC];   // each group's share of S: [token][key]
    threadgroup half p[QB * KC];          // the weights
    threadgroup float fac[QB], lsum[QB], mrun[QB], sink[QB], rs[QB * NR];
    const int rows = a.n[0], heads = a.n[1], nkv = a.n[2], extent = a.n[5], window = a.n[6];
    const int h = int(tg.x), t0 = int(tg.y) * QB, kh = h / (heads / nkv), n = min(QB, rows - t0);
    const long start = a.pos, last = start + t0 + n - 1;
    const long first = window > 0 ? max(0L, start + t0 - window + 1) : 0;
    device const float* q = static_cast<device const float*>(a.p[0]);
    device const float* r = static_cast<device const float*>(a.p[1]);
    device const float* rel = static_cast<device const float*>(a.p[5]);
    for (int i = tid; i < QB * D; i += NTH) {
        const int t = i / D;
        qs[i] = t < n ? half(q[(ulong(t0 + t) * heads + h) * D + i % D]) : half(0.0h);
    }
    for (int i = tid; i < QB * NR; i += NTH) {
        const int t = i / NR;
        rs[i] = t < n ? r[(ulong(t0 + t) * heads + h) * NR + i % NR] : 0.0f;
    }
    if (tid < QB) {   // the extra term of score 0
        mrun[tid] = 0.0f;
        lsum[tid] = 1.0f;
        sink[tid] = tid < n ? static_cast<device const float*>(a.p[2])[ulong(t0 + tid) * heads + h] : 0.0f;
    }
    constexpr auto dS = matmul2d_descriptor(QB, KC, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO = matmul2d_descriptor(QB, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tQ(qs, dextents<int32_t, 2>(D, QB));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(p, dextents<int32_t, 2>(KC, QB));
    const long kfirst = call_first(a);
    const bool copied = a.p[8] != nullptr;
    device half* kb = copied ? static_cast<device half*>(a.p[8]) : static_cast<device half*>(a.p[3]) + ulong(kfirst) * nkv * D;
    device half* vb = copied ? static_cast<device half*>(a.p[9]) : static_cast<device half*>(a.p[4]) + ulong(kfirst) * nkv * D;
    const int nkeys = int(start + rows - kfirst);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(kb + kh * D, dextents<int32_t, 2>(D, nkeys),
                                                               array<int32_t, 2>({1, nkv * D}));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(vb + kh * D, dextents<int32_t, 2>(D, nkeys),
                                                               array<int32_t, 2>({1, nkv * D}));
    auto tPh = tP.template slice<32, QB>(0, 0);
    auto rv = tV.slice(0, 0);
    using PType = __tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>;
    using VType = __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>;
    auto acc = opO.template get_destination_cooperative_tensor<PType, VType, float>();
    auto tot = opO.template get_destination_cooperative_tensor<PType, VType, float>();
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) tot[i] = 0.0f;
    const int d0 = sg * 64;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (long c0 = first; c0 <= last; c0 += KC) {
        const int nc = int(min(long(KC), last - c0 + 1)), at = int(c0 - kfirst);
        {
            auto rk = tK.slice(d0, at);
            auto tA = tQ.template slice<32, QB>(d0, 0);
            using AType = __tensor_ops_detail::__remove_addrspace_t<decltype(tA)>;
            using KType = __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>;
            auto s = opS.template get_destination_cooperative_tensor<AType, KType, float>();
            for (uint16_t i = 0; i < s.get_capacity(); ++i)
                if (s.is_valid_element(i)) s[i] = 0.0f;
            for (int k0 = d0; k0 < d0 + 64; k0 += 32) {
                tA = tQ.template slice<32, QB>(k0, 0);
                rk = tK.slice(k0, at);
                opS.run(tA, rk, s);
            }
            tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sp[sg], dextents<int32_t, 2>(KC, QB));
            s.store(tS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the softmax, TPR lanes of one SIMD group a token, KPT keys a lane: only the keys the token reads (up to its
        // own, within the window) weigh.  The maximum is taken across the lanes; the sum stays one lane's.
        {
            constexpr int TPR = NTH / QB, KPT = KC / TPR;
            const int t = tid / TPR, j = tid % TPR, k1 = j * KPT;
            int c1 = 0, c2 = 0;
            float m0 = 0.0f, m = 0.0f;
            if (t < n) {
                const long own = start + t0 + t, lo = window > 0 ? max(0L, own - window + 1) : 0;
                c1 = int(clamp(lo - c0, 0L, long(nc)));
                c2 = int(clamp(own - c0 + 1, 0L, long(nc)));
                m0 = mrun[t];
                m = m0;
                for (int c = max(c1, k1); c < min(c2, k1 + KPT); ++c) {
                    float v = 0.0f;
                    for (int g = 0; g < NSG; ++g) v += sp[g][t * KC + c];
                    v = v * a.f[0] - sink[t];
                    const long dist = own - (c0 + c);
                    if (dist < extent)
                        for (int k = 0; k < NR; ++k) v += rs[t * NR + k] * rel[ulong(dist) * NR + k];
                    sp[0][t * KC + c] = v;
                    m = max(m, v);
                }
                for (int o = 1; o < TPR; o <<= 1) m = max(m, simd_shuffle_xor(m, ushort(o)));
                for (int c = k1; c < k1 + KPT; ++c) {
                    const float e = c >= c1 && c < c2 ? exp(sp[0][t * KC + c] - m) : 0.0f;
                    sp[0][t * KC + c] = e;
                    p[t * KC + c] = half(e);
                }
            } else {
                for (int c = k1; c < k1 + KPT; ++c) p[t * KC + c] = half(0.0h);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);   // a token's lanes are one SIMD group's
            if (j == 0) {
                if (t < n) {
                    const float f = exp(m0 - m);
                    float l = lsum[t] * f;
                    for (int c = c1; c < c2; ++c) l += sp[0][t * KC + c];
                    mrun[t] = m;
                    lsum[t] = l;
                    fac[t] = f;
                } else
                    fac[t] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) acc[i] = 0.0f;
        for (int k0 = 0; k0 < KC; k0 += 32) {
            tPh = tP.template slice<32, QB>(k0, 0);
            rv = tV.slice(d0, at + k0);
            opO.run(tPh, rv, acc);
        }
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) tot[i] = fma(tot[i], fac[acc.get_multidimensional_index(i)[1]], acc[i]);
    }
    device float* out = static_cast<device float*>(a.p[7]);
    device const float* gate = static_cast<device const float*>(a.p[6]);
    for (uint16_t i = 0; i < tot.get_capacity(); ++i)
        if (tot.is_valid_element(i)) {
            const auto ix = tot.get_multidimensional_index(i);   // (value, token)
            const int t = ix[1];
            if (t < n) {
                const ulong o = (ulong(t0 + t) * heads + h) * D + d0 + ix[0];
                float y = tot[i] / lsum[t];
                if (gate) y *= 1.0f / (1.0f + exp(-gate[o]));
                out[o] = y;
            }
        }
}
