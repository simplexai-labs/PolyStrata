// src/poly/mmq_nvfp4_x2.cu - see nvfp4_x2.hpp.  Built for the arch-specific 12xa targets (cmake/strata_poly.cmake):
// the block-scaled FP4 MMA and the E2M1 conversion exist only there.  For any other target, and
// with a CUDA older than 12.8, the kernels are empty and nvfp4_x2_built() says so, so Mmq never takes them.
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
#include "nvfp4_x2.hpp"
#include "nvfp4_x2_quant.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <stdexcept>
#include <string>

#if defined(STRATA_X2_HAS_FP4) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define STRATA_X2_FP4 1
#endif

namespace strata::poly {
namespace {

using namespace x2;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("nvfp4 x2: ") + what + ": " + cudaGetErrorString(e));
}

#ifdef STRATA_X2_HAS_FP4
__global__ void probe_kernel(int* has) {
#ifdef STRATA_X2_FP4
    *has = 1;
#else
    *has = 0;
#endif
}
#endif

// One warp a row: the row's scale from its largest |value|, then each 16 values' two terms, a lane's sub-blocks
// lane, lane + 32, ...
__global__ void quantize_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids, uint8_t* __restrict__ xq,
                                int cols, int64_t ld, int64_t rows) {
#ifdef STRATA_X2_FP4
    const int lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (r >= rows) return;
    const float* xr = x + (int64_t) (ids ? ids[r] : r) * ld;
    float m = 0.0f;   // max_abs(), as the host takes it: a NaN anywhere in the row is the row's
    for (int i = lane; i < cols / 4; i += 32) {
        const float4 v = ((const float4*) xr)[i];
        m = max_abs(max_abs(max_abs(max_abs(m, v.x), v.y), v.z), v.w);
    }
    for (int o = 16; o > 0; o /= 2) m = max_abs(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float g = row_scale(m);
    uint8_t* row = xq + (size_t) r * row_bytes(cols);
    for (int s = lane; s < cols / 16; s += 32) {
        float v[16];
        for (int q = 0; q < 4; ++q) {
            const float4 t = ((const float4*) (xr + 16 * s))[q];
            v[4 * q] = t.x;
            v[4 * q + 1] = t.y;
            v[4 * q + 2] = t.z;
            v[4 * q + 3] = t.w;
        }
        uint8_t* pair = row + (size_t) (s / 4) * kPairBytes;
        const int sub = s % 4;
        for (int term = 0; term < 2; ++term) {
            uint8_t d = 0;
            uint32_t q[2] = {0, 0};
            if (g > 0.0f) term16(v, g, d, (uint8_t*) q);
            uint8_t* blk = pair + term * kBlockBytes;
            blk[sub] = d;
            ((uint32_t*) (blk + 4 + 8 * sub))[0] = q[0];   // a block's codes are 4-byte aligned
            ((uint32_t*) (blk + 4 + 8 * sub))[1] = q[1];
        }
    }
    if (lane == 0) ((float*) (xq + (size_t) rows * row_bytes(cols)))[r] = g;
#endif
}

#ifdef STRATA_X2_FP4
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void copy8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void copy16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait_groups() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return *(const uint32_t*) p; }

// d += A B: A 16 weight rows x 64 values, B 64 values x 8 tokens, both E2M1 with a UE4M3 scale per 16 values
// (the scale registers' first byte and lane: selectors 0)
__device__ __forceinline__ void mma(float* d, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
#endif

struct MulArgs {
    const uint8_t* w;      // matrix e at w + e stride
    size_t stride;
    size_t w_row;          // bytes of a weight row
    int w_rows;
    int steps;             // kK-value steps along a row
    const uint8_t* xq;
    size_t x_row;          // bytes of an activation row (both terms)
    const float* gs;       // the rows' scales
    const int32_t* bounds;
    const int32_t* ids;
    const int32_t* from;   // row i reads row from[i] of xq, or row i without it
    int total_rows;
    float* dst;
    int64_t ld;
    bool vec;              // dst rows take float4 stores
};

// One block: weight rows [kM x, + kM) of matrix z times its rows [kN y, + kN).  A row past the matrix's last (the
// tile's tail, or the matrix's last weight rows) reads that last one again and is not stored.
__global__ __launch_bounds__(kThreads, 1) void mul_kernel(const MulArgs a) {
#ifdef STRATA_X2_FP4
    extern __shared__ __align__(16) uint8_t sm[];
    const int e = blockIdx.z;
    const int b0 = a.bounds ? a.bounds[e] : 0, b1 = a.bounds ? a.bounds[e + 1] : a.total_rows;
    const int t0 = b0 + (int) blockIdx.y * kN;
    if (t0 >= b1) return;
    const int r0 = (int) blockIdx.x * kM;
    const uint8_t* w = a.w + (size_t) e * a.stride;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;

    const auto load = [&](int stage, int step) {
        for (int i = tid; i < kWPieces; i += kThreads) {
            int row, byte;
            w_piece(i, row, byte);
            const int wr = min(r0 + row, a.w_rows - 1);
            copy8(sm + w_smem(stage, row, 0) + byte,
                  w + (size_t) wr * a.w_row + (size_t) step * (kBlocksPerStep * kBlockBytes) + byte);
        }
        for (int i = tid; i < kTPieces; i += kThreads) {
            int tok, byte;
            t_piece(i, tok, byte);
            int xr = min(t0 + tok, b1 - 1);
            if (a.from) xr = a.from[xr];
            copy16(sm + t_smem(stage, tok, 0, 0) + byte, a.xq + (size_t) xr * a.x_row + (size_t) step * kTStride + byte);
        }
    };

    float acc[2][4][4] = {};
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < a.steps) load(s, s);
        commit();
    }
    const int wr0 = warp_rows(warp), wt0 = warp_toks(warp);
    for (int step = 0; step < a.steps; ++step) {
        wait_groups<kStages - 2>();
        __syncthreads();   // the stage is in, and every warp is done with the one the next copy overwrites
        if (step + kStages - 1 < a.steps) load((step + kStages - 1) % kStages, step + kStages - 1);
        commit();
        const int st = step % kStages;
#pragma unroll
        for (int blk = 0; blk < kBlocksPerStep; ++blk) {
            uint32_t A[2][4], sa[2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const uint8_t* p = sm + w_smem(st, wr0 + 16 * mt, blk);
#pragma unroll
                for (int i = 0; i < 4; ++i) A[mt][i] = ld32(p + a_row(lane, i) * kWStride + 4 * a_word(lane, i));
                sa[mt] = ld32(p + a_scale_row(lane) * kWStride);
            }
#pragma unroll
            for (int term = 0; term < 2; ++term)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const uint8_t* p = sm + t_smem(st, wt0 + 8 * nt + lane / 4, blk, term);
                    const uint32_t B0 = ld32(p + 4 * b_word(lane, 0)), B1 = ld32(p + 4 * b_word(lane, 1)), sb = ld32(p);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) mma(acc[mt][nt], A[mt], B0, B1, sa[mt], sb);
                }
        }
    }
    wait_groups<0>();
    __syncthreads();

    // the tile through shared memory, tokens by weight rows, so that a token's row leaves in whole stores
    float* out = (float*) sm;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out[(wt0 + 8 * nt + c_col(lane, i)) * kOutStride + wr0 + 16 * mt + c_row(lane, i)] = acc[mt][nt][i];
    __syncthreads();
    const int cols = min(kM, a.w_rows - r0);
    for (int tok = warp; tok < kN && t0 + tok < b1; tok += kThreads / 32) {
        const int r = t0 + tok;
        const float g = a.gs[a.from ? a.from[r] : r];
        float* d = a.dst + (int64_t) (a.ids ? a.ids[r] : r) * a.ld + r0;
        const float* o = out + tok * kOutStride;
        if (a.vec && cols == kM) {
            const float4 v = ((const float4*) o)[lane];
            ((float4*) d)[lane] = make_float4(v.x * g, v.y * g, v.z * g, v.w * g);
        } else {
            for (int c = lane; c < cols; c += 32) d[c] = o[c] * g;
        }
    }
#endif
}

}  // namespace

// Asked once a device, and only for a matrix mmq_own.cu's x2_rows() lets through (an NVFP4 matrix, and the variable
// asks for the products).  Without the 12xa image the probe itself fails or reports 0; that error is cleared, and an
// error another call left pending is not: then nothing is cleared.
bool nvfp4_x2_built() {
#ifndef STRATA_X2_HAS_FP4
    return false;
#else
    static std::atomic<int> known[64];   // per device: 0 not yet asked, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (known[dev].load() == 0) {
        const bool pending = cudaPeekAtLastError() != cudaSuccess;
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        bool has = false;
        cudaFuncAttributes fa;
        if (major == 12 && cudaFuncGetAttributes(&fa, (const void*) mul_kernel) == cudaSuccess) {
            int* d = nullptr;
            int h = 0;
            if (cudaMalloc(&d, sizeof(int)) == cudaSuccess) {
                probe_kernel<<<1, 1>>>(d);
                has = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess && h == 1;
                cudaFree(d);
            }
            has = has && cudaFuncSetAttribute((const void*) mul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              kSmemBytes) == cudaSuccess;
        }
        if (!pending) cudaGetLastError();
        known[dev].store(has ? 2 : 1);
    }
    return known[dev].load() == 2;
#endif
}

void nvfp4_x2_quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows,
                       void* stream) {
    if (rows <= 0) return;
    if (cols % kK != 0) throw std::runtime_error("nvfp4 x2: rows of " + std::to_string(cols) + " values");
    constexpr int kRowsPerBlock = 8;
    quantize_kernel<<<(unsigned) ((rows + kRowsPerBlock - 1) / kRowsPerBlock), 32 * kRowsPerBlock, 0,
                      (cudaStream_t) stream>>>(x, ids, (uint8_t*) xq, (int) cols, ld, rows);
    ck(cudaGetLastError(), "quantize");
}

void nvfp4_x2_mul(const void* w, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
                  const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
                  int64_t ld_dst, void* stream, const int32_t* from) {
    if (n <= 0 || max_rows <= 0 || w_rows <= 0) return;
    const size_t w_row = (size_t) (w_cols / kBlock) * kBlockBytes;
    if (w_cols % kK != 0 || (uintptr_t) w % 8 != 0 || (n > 1 && stride % 8 != 0))
        throw std::runtime_error("nvfp4 x2: a matrix of " + std::to_string(w_cols) + " columns not 8-byte aligned");
    MulArgs a;
    a.w = (const uint8_t*) w;
    a.stride = stride;
    a.w_row = w_row;
    a.w_rows = (int) w_rows;
    a.steps = (int) (w_cols / kK);
    a.xq = (const uint8_t*) xq;
    a.x_row = row_bytes(w_cols);
    a.gs = row_scales(xq, total_rows, w_cols);
    a.bounds = bounds;
    a.ids = ids;
    a.from = from;
    a.total_rows = (int) total_rows;
    a.dst = dst;
    a.ld = ld_dst;
    a.vec = (uintptr_t) dst % 16 == 0 && ld_dst % 4 == 0;
    const dim3 grid((unsigned) ((w_rows + kM - 1) / kM), (unsigned) ((max_rows + kN - 1) / kN), (unsigned) n);
    mul_kernel<<<grid, kThreads, kSmemBytes, (cudaStream_t) stream>>>(a);
    ck(cudaGetLastError(), "product");
}

}  // namespace strata::poly
