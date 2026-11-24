// src/kernels/cuda/gdn_mma_tiles.cuh - the part of llama.cpp's ggml-cuda/mma.cuh that gdn_mma.cu uses, for CUDA:
// register tiles in the layouts of mma.sync m16n8k16, their element coordinates, and the BF16 product into FP32.
// Written from mma.cuh at llama.cpp 7182ef71e6d5dd0c8ea76486fac134fcd25025d4 (MIT; see third_party/ggml/LICENSE),
// NVIDIA paths only, so that the GDN kernel builds without a llama.cpp checkout.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define TURING_MMA_AVAILABLE
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define AMPERE_MMA_AVAILABLE
#endif
// Volta arm of gdn_mma.cu (STRATA_EXPERIMENTAL_SM60 builds only; the host gate is gdn_mma_policy.hpp)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
#define VOLTA_MMA_AVAILABLE
#endif

#define GGML_CUDA_MAX_DEVICES 16

template <typename... T>
static __host__ __device__ __forceinline__ void ggml_unused_vars(const T&...) {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars(__VA_ARGS__)

#if defined(__CUDA_ARCH__)
#define NO_DEVICE_CODE __trap()
#else
#define NO_DEVICE_CODE
#endif

static constexpr __host__ __device__ int ggml_cuda_get_physical_warp_size() { return 32; }

namespace ggml_cuda_mma {

enum data_layout {
    DATA_LAYOUT_I_MAJOR,
    DATA_LAYOUT_J_MAJOR,
    DATA_LAYOUT_I_MAJOR_MIRRORED,
    DATA_LAYOUT_J_MAJOR_MIRRORED,
};

// A warp's I x J tile, ne values a thread.  Float tiles are the m16n8 accumulators: a thread holds rows lane / 4 and
// lane / 4 + 8, columns 2 (lane % 4) and the one after.
template <int I_, int J_, typename T, data_layout dl_ = DATA_LAYOUT_I_MAJOR>
struct tile {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr data_layout dl = dl_;
    static constexpr int ne = I * J / 32;
    T x[ne] = {};

    static __device__ __forceinline__ int get_i(const int l) {
        static_assert(I == 16 && J == 8, "the float tiles here are the m16n8 accumulators");
        return (l / 2) * 8 + int(threadIdx.x) / 4;
    }
    static __device__ __forceinline__ int get_j(const int l) { return 2 * (int(threadIdx.x) % 4) + l % 2; }
};

// Pairs of 16-bit values (half2, nv_bfloat162), J counted in pairs: the A operand of m16n8k16 is a 16 x 8 tile of pairs
// (a thread: rows lane / 4 and + 8, pairs lane % 4 and + 4), the B operand an 8 x 8 one (row lane / 4, pairs lane % 4
// and + 4).
template <int I_, int J_, data_layout dl_>
struct tile<I_, J_, half2, dl_> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr data_layout dl = dl_;
    static constexpr int ne = I * J / 32;
    half2 x[ne] = {};

    static __device__ __forceinline__ int get_i(const int l) {
        if constexpr (I == 8) return int(threadIdx.x) / 4;
        else return (l % 2) * 8 + int(threadIdx.x) / 4;
    }
    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 8) return l * 4 + int(threadIdx.x) % 4;
        else return (l / 2) * 4 + int(threadIdx.x) % 4;
    }
};

template <int I_, int J_, data_layout dl_>
struct tile<I_, J_, nv_bfloat162, dl_> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr data_layout dl = dl_;
    static constexpr int ne = I * J / 32;
    nv_bfloat162 x[ne] = {};

    static __device__ __forceinline__ int get_i(const int l) {
        if constexpr (I == 8) return int(threadIdx.x) / 4;
        else return (l % 2) * 8 + int(threadIdx.x) / 4;
    }
    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 8) return l * 4 + int(threadIdx.x) % 4;
        else return (l / 2) * 4 + int(threadIdx.x) % 4;
    }
};

// D += A B on the tensor cores: A 16 x 16 BF16 (row), B 16 x 8 BF16 given as its transpose (col), D FP32.
static __device__ __forceinline__ void mma(tile<16, 8, float>& D, const tile<16, 8, nv_bfloat162>& A,
                                           const tile<8, 8, nv_bfloat162>& B) {
#if defined(AMPERE_MMA_AVAILABLE)
    const int* a = reinterpret_cast<const int*>(A.x);
    const int* b = reinterpret_cast<const int*>(B.x);
    int* d = reinterpret_cast<int*>(D.x);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    GGML_UNUSED_VARS(D, A, B);
    NO_DEVICE_CODE;
#endif
}

}  // namespace ggml_cuda_mma
