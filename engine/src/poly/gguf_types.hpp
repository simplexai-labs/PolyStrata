// src/poly/gguf_types.hpp - what the engine asks of a GGUF tensor type, without ggml: a block's size, a row as
// floats, a float as BF16.  The quantized types are Strata's quant layer's (strata/quant, which gives ggml's values);
// the float types are here.  Type ids are the GGUF file format's.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "strata/quant/act.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/fp16.hpp"
#include "strata/quant/types.hpp"

namespace strata::poly {

constexpr uint32_t kTypeF16 = 1, kTypeQ4_0 = 2, kTypeNVFP4 = 40;   // (kTypeF32 and kTypeBF16 are runtime.hpp's)

/// Bytes of a block of the type and the values it holds; 0 for a type the table does not know.
inline size_t type_block_bytes(uint32_t type) {
    const strata::quant::TypeInfo* t = strata::quant::type_info((int) type);
    return t ? t->bytes : 0;
}
inline int type_block_values(uint32_t type) {
    const strata::quant::TypeInfo* t = strata::quant::type_info((int) type);
    return t ? t->blck : 0;
}

/// The type's values can be converted to floats by type_to_floats(): F16, BF16 and the quantized types.  Not F32, whose
/// values are floats already (ggml has no conversion for it either, and the callers copy it).
inline bool type_converts(uint32_t type) {
    if (type == kTypeF16 || type == 30) return true;
    const strata::quant::TypeInfo* t = strata::quant::type_info((int) type);
    return t != nullptr && strata::quant::can_dequant(t->type);
}

/// n values (whole blocks) at src as floats: a type type_converts() takes, or F32 itself.
inline void type_to_floats(uint32_t type, const void* src, float* dst, int64_t n) {
    if (type == 0) {
        std::memcpy(dst, src, (size_t) n * sizeof(float));
    } else if (type == kTypeF16) {
        const uint16_t* h = (const uint16_t*) src;
        for (int64_t i = 0; i < n; ++i) dst[i] = strata::quant::fp16_to_fp32(h[i]);
    } else if (type == 30) {   // BF16: a float's upper half
        const uint16_t* h = (const uint16_t*) src;
        for (int64_t i = 0; i < n; ++i) dst[i] = strata::quant::f32_from_bits((uint32_t) h[i] << 16);
    } else {
        strata::quant::dequant_row((strata::quant::Type) type, src, dst, n);
    }
}

/// A float as BF16, as ggml writes it: rounded to the nearest, ties to even; a NaN is made quiet.
inline uint16_t bf16_of(float f) {
    const uint32_t u = strata::quant::f32_to_bits(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64u);
    return (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

}  // namespace strata::poly
