// src/poly/nvfp4_x2.hpp - NVFP4 prompt products with two-term FP4 activations on RTX 50's FP4 tensor cores
// (STRATA_PREFILL_NVFP4=w4a4x2, mmq_nvfp4_x2.cu).
//
// Each activation row is written as two NVFP4 rows under one float scale for the row: its FP4 rounding and the FP4
// rounding of what that leaves, so that their sum is the row to about 8 bits (two-term FP4 activations as in Basalt,
// below).  A product multiplies the weights' E2M1 codes with both terms on the block-scaled MMA (mma.sync
// kind::mxf4nvf4, a UE4M3 scale per 16 values on either side) into one sum, and the row's float scale goes on it.
// The terms are in ggml's own block_nvfp4 layout (4 scale bytes, then 32 bytes of codes: value j of a 16-value
// sub-block in the low nibble of its byte j, value j + 8 in the high one), the layout the weights have in the GGUF:
// the MMA takes both operands' bytes as they are, since the order of the values inside a 16-value block is the same
// on both sides and does not change the sum.
//
// The format and the kernel's shared-memory layout are here; mmq_own.cu takes the products through Mmq, and
// mmq_paths_test.cu checks them there.
//
// Two-term FP4 activations and parts of the approach are informed by Basalt's src/prefill/moe_fp4.cu
// (github.com/jesdga95/basalt), which is under the MIT license:
// MIT License
// Copyright (c) 2026 Jesús Gulfo
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define STRATA_X2_HD __host__ __device__ __forceinline__
#else
#define STRATA_X2_HD inline
#endif

namespace strata::poly::x2 {

// ---------------------------------------------------------------------------------------------- the rows' format
//
// Row r of n rows of `cols` values (cols a multiple of kK): cols / 64 pairs of blocks at r * row_bytes(cols), each
// pair the first term's block_nvfp4 then the second's (72 bytes); the rows' float scales behind all the rows, at
// n * row_bytes(cols).  Value v of the row is g (e2m1(q1) d1 + e2m1(q2) d2), d1 and d2 its sub-block's UE4M3 scales.

constexpr int kBlock = 64;                 // values in a block_nvfp4
constexpr int kBlockBytes = 36;            // its bytes: 4 UE4M3 scales, 32 bytes of codes
constexpr int kPairBytes = 2 * kBlockBytes;
constexpr float kTopScale = 448.0f;        // the largest UE4M3 value
constexpr float kTopCode = 6.0f;           // the largest E2M1 value

STRATA_X2_HD size_t row_bytes(int64_t cols) { return (size_t) (cols / kBlock) * kPairBytes; }
STRATA_X2_HD size_t rows_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * row_bytes(cols) + (size_t) rows * sizeof(float);
}
STRATA_X2_HD const float* row_scales(const void* xq, int64_t rows, int64_t cols) {
    return (const float*) ((const char*) xq + (size_t) rows * row_bytes(cols));
}

// --------------------------------------------------------------------------------------------------- the kernel
//
// A block: kM weight rows (output values) of one expert times kN of its rows (tokens), K in steps of kK values
// through a kStages-deep cp.async pipeline.  Eight warps, 4 along the weights x 2 along the tokens: a warp has
// 32 weight rows (two m16 tiles: the weights are the MMA's A) times 32 tokens (four n8 tiles: B), 16 MMAs (m16n8k64)
// a 64-value block, 8 of them with each term.
//
// Shared memory, a stage: the weights' kM rows of kK / 64 blocks (72 bytes) at a stride of kWStride, then the
// tokens' kN rows of kK / 64 pairs (144 bytes, kTStride).  With these strides the code fragments' 32-bit reads of a
// warp fall in 32 different banks: 20 words a weight row (rows g = 0..7 at 20 g mod 32 = 0, 20, 8, 28, 16, 4, 24, 12,
// and four words each), 36 a token row (4 g), and so do B's scale reads (8 rows, word 0).  A's scale read, rows g and
// g + 8 at word 0, is a two-way conflict (20 x 8 = 160 = 0 mod 32): one read in five of A's, and no stride keeps
// both it and the code reads apart.  57 KiB (58,368 bytes) with 3 stages, one block a multiprocessor on sm_120
// (99 KiB at most for a block, 100 KiB a multiprocessor).

constexpr int kM = 128, kN = 64, kK = 128, kStages = 3, kThreads = 256;
constexpr int kBlocksPerStep = kK / kBlock;
constexpr int kWStride = 80, kTStride = kBlocksPerStep * kPairBytes;
constexpr int kWStage = kM * kWStride, kTStage = kN * kTStride;
constexpr int kStageBytes = kWStage + kTStage;
constexpr int kSmemBytes = kStages * kStageBytes;
// the output tile, tokens by weight rows, staged in the pipeline's memory for whole-row stores
constexpr int kOutStride = kM + 4;
static_assert(kN * kOutStride * (int) sizeof(float) <= kSmemBytes, "the output tile fits the pipeline's memory");
static_assert(kStageBytes % 16 == 0 && kWStage % 16 == 0, "cp.async destinations stay 16-byte aligned");

STRATA_X2_HD int w_smem(int stage, int row, int blk) {
    return stage * kStageBytes + row * kWStride + blk * kBlockBytes;
}
STRATA_X2_HD int t_smem(int stage, int tok, int blk, int term) {
    return stage * kStageBytes + kWStage + tok * kTStride + blk * kPairBytes + term * kBlockBytes;
}

// A stage's copies: the weights in 8-byte pieces (a row is 36 (cols / 64) bytes, 8-byte aligned when cols / 64 is
// even, as kK makes it), the tokens in 16-byte ones (144 bytes a step, 16-byte aligned rows).  Piece i of a stage:
constexpr int kWPieces = kM * kBlocksPerStep * kBlockBytes / 8;   // 9 a row
constexpr int kTPieces = kN * kTStride / 16;                       // 9 a row
constexpr int kWRowPieces = kBlocksPerStep * kBlockBytes / 8, kTRowPieces = kTStride / 16;
STRATA_X2_HD void w_piece(int i, int& row, int& byte) {
    row = i / kWRowPieces;
    byte = 8 * (i % kWRowPieces);
}
STRATA_X2_HD void t_piece(int i, int& tok, int& byte) {
    tok = i / kTRowPieces;
    byte = 16 * (i % kTRowPieces);
}

// The warp's tiles and a lane's place in a fragment (PTX ISA, "Matrix fragments for mma.m16n8k64" with .e2m1, and
// "warp-level block scaling"): A (16 x 64) register i holds row g + 8 (i & 1), values 8 t + 32 (i >> 1) .. + 7;
// B (64 x 8) register i holds column g, values 8 t + 32 i .. + 7; with eight E2M1 values a 32-bit register, which
// are four bytes of a block's codes (values 8 t .. 8 t + 7 of the MMA are code bytes 4 t .. 4 t + 3, sub-block t / 2,
// in both operands' order).  The scale registers are four UE4M3 bytes, one per 16 values: A's from lanes with
// t = 0 (row g) and t = 1 (row g + 8), B's from t = 0 (column g).  g = lane / 4, t = lane % 4.
STRATA_X2_HD int warp_rows(int warp) { return (warp % 4) * 32; }
STRATA_X2_HD int warp_toks(int warp) { return (warp / 4) * 32; }
STRATA_X2_HD int a_word(int lane, int i) { return 1 + (lane % 4) + 4 * (i >> 1); }   // in 32-bit words of a block
STRATA_X2_HD int a_row(int lane, int i) { return lane / 4 + 8 * (i & 1); }
STRATA_X2_HD int b_word(int lane, int i) { return 1 + (lane % 4) + 4 * i; }
STRATA_X2_HD int a_scale_row(int lane) { return lane / 4 + 8 * (lane & 1); }
// The accumulator (16 x 8 floats): register i is row g + 8 (i >> 1), column 2 t + (i & 1).
STRATA_X2_HD int c_row(int lane, int i) { return lane / 4 + 8 * (i >> 1); }
STRATA_X2_HD int c_col(int lane, int i) { return 2 * (lane % 4) + (i & 1); }

}  // namespace strata::poly::x2

namespace strata::poly {

/// The card can take the two-term FP4 products: a compute capability 12.x card, and this build has the kernels'
/// image for it (the arch-specific 12xa target).
bool nvfp4_x2_built();
/// `rows` rows of x (row i is row ids[i], or row i without ids; ld floats a row, 16-byte aligned) into xq in the
/// format above; cols a multiple of x2::kK.
void nvfp4_x2_quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows,
                       void* stream);
/// Mmq::mul()'s product for NVFP4 matrices from rows nvfp4_x2_quantize() wrote (total_rows of them); with `from`,
/// row i of the product is row from[i] of xq.
void nvfp4_x2_mul(const void* w, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
                  const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
                  int64_t ld_dst, void* stream, const int32_t* from);

}  // namespace strata::poly
