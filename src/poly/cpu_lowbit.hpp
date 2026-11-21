// src/poly/cpu_lowbit.hpp - AVX2 dot products for two expert types whose ggml-cpu ones are slow on x86:
//   Q2_0 x Q8_0: ggml-cpu has only the generic C one;
//   IQ1_S x Q8_K: ggml-cpu's AVX2 one takes BMI2's pdep, which Zen 1 and 2 run in microcode.
// Both take ggml's own blocks and its activation formats (vec_dot_type Q8_0 / Q8_K, quantized by ggml), and have
// ggml's vec_dot signature, so cpu_experts.cpp calls them where it called ggml's.
#pragma once

#include "ggml-cpu.h"

namespace strata::poly {

/// This file's dot product for ggml type `type` (GGML_TYPE_Q2_0 or GGML_TYPE_IQ1_S), or null: for any other type,
/// and on a CPU without AVX2, FMA and F16C (or not x86).
ggml_vec_dot_t lowbit_vec_dot(int type);

}  // namespace strata::poly
