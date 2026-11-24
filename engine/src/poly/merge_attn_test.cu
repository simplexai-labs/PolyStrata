// src/poly/merge_attn_test.cu - the attention of a streamed cache (--kv-resident, Qwen35::attn_stream()): a cache cut in
// three pieces, each attended on its own with its (max, sum) pairs out of ml_out and merged by merge_attn_piece(),
// against one call over the whole cache, with attend() (its generic and its split kernels) and attend_batch().  A piece
// with no row a token attends leaves the merge as it was, and a first piece is taken whole.
//
// usage: poly_merge_attn_test   (exit 0: the pieces within the bounds below of the whole call)
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

// LFM2's heads, as poly_attend_test: 64 values in groups of 4, which attend()'s split kernels take
constexpr int kDim = 64, kHeads = 32, kKvHeads = 8, kChunk = 8, kRows = 3000, kTokens = 40;
constexpr float kScale = 0.125f;
constexpr int kCut[] = {0, 1000, 2048, kRows};   // the pieces: two before the tokens, the last with them

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

struct Buffers {
    float *q, *score, *work, *out, *ml, *piece, *pml;
    uint16_t *k, *v;
    size_t score_n;
    void* blas;
};

enum class Kind { generic, split, batch };

// one call: n tokens from pos over `rows` rows of the cache from row `first`
void call(const Buffers& b, Kind kind, int first, int64_t pos, int n, bool whole, float* out, float* ml) {
    const size_t at = size_t(first) * kKvHeads * kDim;
    if (kind == Kind::batch)
        attend_batch(b.blas, b.q, b.k + at, b.v + at, pos, n, b.score, b.score_n, b.work, out, kDim, kHeads, kKvHeads,
                     kScale, nullptr, 0, 0, whole, {}, {}, nullptr, nullptr, 0, nullptr, ml);
    else
        attend(b.q, b.k + at, b.v + at, pos, n, b.score, kChunk, out, kDim, kHeads, kKvHeads, kScale, nullptr, 0, 0,
               whole, nullptr, 0, kind == Kind::split ? b.score_n : 0, {}, {}, nullptr, 0, ml);
}

// the last n rows of the cache as tokens: the whole cache in one call, then piece by piece as attn_stream() does it
int check(const Buffers& b, Kind kind, int n, float bound) {
    const int64_t pos = kRows - n;
    const size_t vals = size_t(n) * kHeads * kDim;
    call(b, kind, 0, pos, n, false, b.out, b.ml);   // ml asked for, so attend_batch() takes the kernels the pieces do
    const std::vector<float> whole = download(b.out, vals);
    for (int p = 0; p + 1 < int(std::size(kCut)); ++p) {
        const int r0 = kCut[p], m = kCut[p + 1] - kCut[p];
        const bool before = r0 + m <= pos;   // a piece before the tokens: each sees all of it
        float* out = p == 0 ? b.out : b.piece;
        float* ml = p == 0 ? b.ml : b.pml;
        call(b, kind, r0, before ? m - n : pos - r0, n, before, out, ml);
        if (p > 0) merge_attn_piece(b.out, b.ml, b.piece, b.pml, n, kHeads, kDim, nullptr);
    }
    const std::vector<float> merged = download(b.out, vals);
    float most = 0.f;
    for (size_t i = 0; i < vals; ++i) most = std::max(most, std::fabs(merged[i] - whole[i]));
    const bool ok = most <= bound;
    std::printf("%s %-7s %2d tokens over %d rows in %d pieces: %.2e from one call (bound %.0e)\n", ok ? "ok  " : "FAIL",
                kind == Kind::batch ? "batch" : kind == Kind::split ? "split" : "generic", n, kRows,
                int(std::size(kCut)) - 1, most, bound);
    return ok ? 0 : 1;
}

// merge_attn_piece()'s two edges: a piece no token attended (max -inf, sum 0) changes nothing; an accumulator with
// nothing yet takes the piece as it is
int check_edges(const Buffers& b) {
    const int n = 3;
    const size_t vals = size_t(n) * kHeads * kDim, pairs = size_t(2) * n * kHeads;
    call(b, Kind::generic, 0, kRows - n, n, false, b.out, b.ml);
    const std::vector<float> out = download(b.out, vals), ml = download(b.ml, pairs);
    std::vector<float> none(pairs);
    for (size_t i = 0; i < pairs; i += 2) none[i] = -INFINITY, none[i + 1] = 0.f;
    cudaMemcpy(b.pml, none.data(), pairs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(b.piece, 0, vals * sizeof(float));
    merge_attn_piece(b.out, b.ml, b.piece, b.pml, n, kHeads, kDim, nullptr);
    const bool kept = download(b.out, vals) == out && download(b.ml, pairs) == ml;
    // now the other way round: the accumulator empty, the call's result the piece
    cudaMemcpy(b.piece, b.out, vals * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(b.pml, b.ml, pairs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(b.ml, none.data(), pairs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(b.out, 0, vals * sizeof(float));
    merge_attn_piece(b.out, b.ml, b.piece, b.pml, n, kHeads, kDim, nullptr);
    const bool taken = download(b.out, vals) == out && download(b.ml, pairs) == ml;
    std::printf("%s an empty piece leaves the merge %s, an empty merge takes the piece %s\n", kept && taken ? "ok  " : "FAIL",
                kept ? "as it was" : "CHANGED", taken ? "as it is" : "CHANGED");
    return kept && taken ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(kTokens) * kHeads * kDim);
    for (float& x : q) x = 2.f * nd(rng);   // spread scores, as poly_attend_test
    std::vector<uint16_t> k(size_t(kRows) * kKvHeads * kDim), v(k.size());
    for (uint16_t& x : k) x = __half_as_ushort(__float2half(nd(rng)));
    for (uint16_t& x : v) x = __half_as_ushort(__float2half(nd(rng)));
    Buffers b{};
    b.q = upload(q), b.k = upload(k), b.v = upload(v);
    const size_t vals = q.size(), pairs = size_t(2) * kTokens * kHeads;
    b.score_n = std::max(size_t(kChunk) * kHeads * kRows, attend_split_floats(kChunk, kHeads, kRows, kDim));
    cudaMalloc(&b.score, b.score_n * sizeof(float));
    cudaMalloc(&b.work, (vals + vals / 2 + 1 + size_t(3) * kTokens * kHeads) * sizeof(float));   // as Qwen35's awork_
    for (float** p : {&b.out, &b.piece}) cudaMalloc(p, vals * sizeof(float));
    for (float** p : {&b.ml, &b.pml}) cudaMalloc(p, pairs * sizeof(float));
    b.blas = blas_create(nullptr);
    int bad = 0;
    // attend()'s kernels sum in floats: the pieces are the one call's result to rounding.  attend_batch() gives the
    // softmax's weights as halves, each piece against its own largest score, so its pieces part from the one call in
    // the fourth or fifth digit.
    for (int n : {1, 2, 4, 8}) {
        bad += check(b, Kind::generic, n, 1e-5f);
        bad += check(b, Kind::split, n, 1e-5f);
    }
    for (int n : {kTokens, 33}) bad += check(b, Kind::batch, n, 1e-3f);
    bad += check_edges(b);
    cudaFree(b.q), cudaFree(b.k), cudaFree(b.v), cudaFree(b.score), cudaFree(b.work);
    cudaFree(b.out), cudaFree(b.piece), cudaFree(b.ml), cudaFree(b.pml);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
