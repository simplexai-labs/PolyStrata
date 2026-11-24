// src/quant/x86/dot_avxvnni.cpp - the Avx2 contract on CPUs with AVX-VNNI (VEX: Alder Lake and later Intel, Zen 5): the same ported ggml code as
// dot_avx2.cpp, built so that its integer dot products of four bytes take VPDPBUSD, one instruction for maddubs + madd (ggml's __AVXVNNI__
// branch of mul_sum_us8_pairs_float, and mul_sum_i8_quads for the 4-bit grid formats).  The integer sums are exact either way and the float
// order is the AVX2 one, so the bits are the AVX2 tier's.  Only called when caps() says AVX-VNNI (Isa::AvxVnni); only the formats whose dot
// changes are registered.  The idea is Strata 0.2.18's (Niko1221/Strata, MIT, commit bde1682d1 "hy4: add AVX-VNNI + S8_K fast CPU kernels").
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__AVXVNNI__)
#define __AVXVNNI__ 1   // MSVC has the intrinsics under /arch:AVX2 but no macro for them
#endif
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

#if !defined(__AVXVNNI__) || !defined(__AVX2__)
#error "dot_avxvnni.cpp needs AVX2 and AVX-VNNI"
#endif

namespace strata::quant {
namespace avxvnni {

#include "x86_helpers.inc"
#include "x86_dots.inc"

}  // namespace avxvnni

namespace {
#define ENTRY(T, ACT, NAME) {avxvnni::dot_##NAME, Type::T, Type::ACT, Arith::Avx2, Isa::AvxVnni, "avxvnni_" #NAME}
const DotKernel kTable[] = {
    ENTRY(Q4_0, Q8_0, q4_0_q8_0), ENTRY(Q4_1, Q8_1, q4_1_q8_1),     ENTRY(Q5_0, Q8_0, q5_0_q8_0),   ENTRY(Q5_1, Q8_1, q5_1_q8_1),
    ENTRY(Q8_0, Q8_0, q8_0_q8_0), ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0), ENTRY(NVFP4, Q8_0, nvfp4_q8_0),
};
#undef ENTRY
}  // namespace

DotTable avxvnni_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
