// src/poly/expert_card.cpp - the second card's routed experts (expert_card.hpp) and Engine::fill_expert_card().
#include "expert_card.hpp"

#include "engine.hpp"
#include "gguf_types.hpp"
#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>

namespace strata::poly {

namespace {
// The calls below run on the expert card and leave the caller's card current again.
struct OnDevice {
    int home = 0;
    explicit OnDevice(int dev) {
        ck(cudaGetDevice(&home), "--expert-gpu");
        ck(cudaSetDevice(dev), "--expert-gpu");
    }
    ~OnDevice() { cudaSetDevice(home); }
};
}  // namespace

ExpertCard::ExpertCard(int device, int n_layer, int n_expert, int n_embd, int n_ff)
    : device_(device), n_layer_(n_layer), n_expert_(n_expert), n_embd_(n_embd), n_ff_(n_ff) {
    slots_.assign((size_t) n_layer * (size_t) n_expert, nullptr);
    s_gate_.assign((size_t) n_layer, nullptr);
    s_up_.assign((size_t) n_layer, nullptr);
    OnDevice on(device_);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "--expert-gpu stream");
    stream_ = s;
    const size_t ff = (size_t) n_ff, row = (size_t) n_embd;
    ck(cudaMalloc((void**) &xf_, (size_t) kCpuRows * row * sizeof(float)), "--expert-gpu scratch");
    ck(cudaMalloc(&xq_, kernels::native_q8_1_bytes(n_embd, kCpuRows)), "--expert-gpu scratch");
    ck(cudaMalloc((void**) &g_, (size_t) kSlots * 2 * ff * sizeof(float)), "--expert-gpu scratch");
    ck(cudaMalloc((void**) &h_, (size_t) kSlots * ff * sizeof(float)), "--expert-gpu scratch");
    ck(cudaMalloc(&q8_, kernels::native_q8_1_bytes(n_ff, kSlots)), "--expert-gpu scratch");
    ck(cudaMalloc((void**) &d_, (size_t) kSumRows * row * sizeof(float)), "--expert-gpu scratch");
    ck(cudaMalloc((void**) &acc_, (size_t) kCpuRows * row * sizeof(float)), "--expert-gpu scratch");
    ck(cudaMallocHost((void**) &out_, (size_t) kCpuRows * row * sizeof(float)), "--expert-gpu scratch");
}

ExpertCard::~ExpertCard() {
    int home = 0;
    if (cudaGetDevice(&home) != cudaSuccess) return;
    cudaSetDevice(device_);
    if (stream_) cudaStreamSynchronize((cudaStream_t) stream_);
    for (void* p : {(void*) xf_, xq_, (void*) g_, (void*) h_, q8_, (void*) d_, (void*) acc_, (void*) arena_})
        if (p) cudaFree(p);
    for (float* p : s_gate_)
        if (p) cudaFree(p);
    for (float* p : s_up_)
        if (p) cudaFree(p);
    if (out_) cudaFreeHost(out_);
    if (stream_) cudaStreamDestroy((cudaStream_t) stream_);
    cudaSetDevice(home);
}

size_t ExpertCard::claim(int reserve_mib, size_t most) {
    OnDevice on(device_);
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "--expert-gpu");
    const size_t reserve = (size_t) std::max(reserve_mib, 0) << 20;
    if (free_b <= reserve + kTail || most == 0) return 0;
    arena_bytes_ = std::min(free_b - reserve - kTail, most);
    ck(cudaMalloc((void**) &arena_, arena_bytes_ + kTail), "--expert-gpu experts");
    ck(cudaMemset(arena_, 0, arena_bytes_ + kTail), "--expert-gpu experts");   // the slots' tails stay zero (kTail)
    return arena_bytes_;
}

bool ExpertCard::put(const Layer& L, int il, int e, size_t slot_bytes) {
    if (used_ + slot_bytes > arena_bytes_) return false;
    OnDevice on(device_);
    uint8_t* s = arena_ + used_;
    const size_t x = (size_t) e;
    ck(cudaMemcpy(s, L.exp_gate + x * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice), "--expert-gpu experts");
    ck(cudaMemcpy(s + L.gu_bytes, L.exp_up + x * L.u_stride, L.u_bytes, cudaMemcpyHostToDevice), "--expert-gpu experts");
    ck(cudaMemcpy(s + L.gu2(), L.exp_down + x * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice), "--expert-gpu experts");
    used_ += slot_bytes;
    slots_[(size_t) il * (size_t) n_expert_ + x] = s;
    ++held_;
    return true;
}

void ExpertCard::scales(int il, const std::vector<float>& gate, const std::vector<float>& up) {
    OnDevice on(device_);
    const auto copy = [&](const std::vector<float>& v, float*& to) {
        if (v.empty() || to) return;
        ck(cudaMalloc((void**) &to, v.size() * sizeof(float)), "--expert-gpu scales");
        ck(cudaMemcpy(to, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "--expert-gpu scales");
    };
    copy(gate, s_gate_[(size_t) il]);
    copy(up, s_up_[(size_t) il]);
}

// What Engine::moe()'s launch does with the first card's cached experts, on this card: the rows' inputs quantized,
// gate and up, the activation, its quantization, down, then each output times its weight into its token's row.
void ExpertCard::start(const Layer& L, int il, const float* x, int m, const Job* jobs, int n, float limit, bool gelu,
                       float situ_beta, float situ_lb) {
    OnDevice on(device_);
    if (pending_) {   // a call that ended in an exception (the CPU's experts) left its work: it is dropped
        ck(cudaStreamSynchronize((cudaStream_t) stream_), "--expert-gpu experts");
        pending_ = 0;
    }
    cudaStream_t st = (cudaStream_t) stream_;
    const size_t row = (size_t) n_embd_, ff = (size_t) n_ff_;
    const size_t xq_row = kernels::native_q8_1_bytes(n_embd_), q8_row = kernels::native_q8_1_bytes(n_ff_);
    ck(cudaMemcpyAsync(xf_, x, (size_t) m * row * sizeof(float), cudaMemcpyHostToDevice, st), "--expert-gpu input");
    ck(cudaMemsetAsync(acc_, 0, (size_t) m * row * sizeof(float), st), "--expert-gpu input");
    kernels::native_quantize_q8_1(xf_, xq_, n_embd_, m, st);
    const int per = kernels::native_mmvq_slots_wide() ? kSlots : 8;
    const int gu_rows = L.mixed() ? n_ff_ : 2 * n_ff_;
    ScaledRows sum{};
    for (int i0 = 0, k; i0 < n; i0 += k) {
        k = std::min({per, n - i0, kSumRows - i0 % kSumRows});
        kernels::MmvqSlots gate_up{}, up{}, down{};
        for (int j = 0; j < k; ++j) {
            const Job& c = jobs[i0 + j];
            gate_up.w[j] = c.slot;
            gate_up.x[j] = (const uint8_t*) xq_ + (size_t) c.row * xq_row;
            gate_up.y[j] = g_ + (size_t) j * 2 * ff;
            up.w[j] = c.slot + L.gu_bytes;
            up.x[j] = gate_up.x[j];
            up.y[j] = gate_up.y[j] + ff;
            down.w[j] = c.slot + L.gu2();
            down.x[j] = (const uint8_t*) q8_ + (size_t) j * q8_row;
            down.y[j] = d_ + (size_t) ((i0 + j) % kSumRows) * row;
        }
        if (!(k > 1 && kernels::native_mmvq_slots((int) L.gu_type, gate_up, k, n_embd_, gu_rows, st)))
            for (int j = 0; j < k; ++j)
                kernels::native_mmvq((int) L.gu_type, gate_up.w[j], gate_up.x[j], gate_up.y[j], n_embd_, gu_rows, 1, st);
        if (L.mixed() && !(k > 1 && kernels::native_mmvq_slots((int) L.u_type, up, k, n_embd_, n_ff_, st)))
            for (int j = 0; j < k; ++j)
                kernels::native_mmvq((int) L.u_type, up.w[j], up.x[j], up.y[j], n_embd_, n_ff_, 1, st);
        ExpertScales es;
        es.gate = s_gate_[(size_t) il];
        es.up = s_up_[(size_t) il];
        for (int j = 0; j < k && (es.gate || es.up); ++j) es.e[j] = jobs[i0 + j].e;
        const ExpertScales* esp = es.gate || es.up ? &es : nullptr;
        if (situ_beta > 0) situ_rows(g_, h_, n_ff_, k, situ_beta, situ_lb, st, nullptr, esp);
        else if (gelu) geglu_rows(g_, h_, n_ff_, k, st, esp);
        else swiglu_clamp_rows(g_, h_, n_ff_, k, limit, st, esp);
        kernels::native_quantize_q8_1(h_, q8_, n_ff_, k, st);
        if (!(k > 1 && kernels::native_mmvq_slots((int) L.d_type, down, k, n_ff_, n_embd_, st)))
            for (int j = 0; j < k; ++j)
                kernels::native_mmvq((int) L.d_type, down.w[j], down.x[j], down.y[j], n_ff_, n_embd_, 1, st);
        for (int j = 0; j < k; ++j) {
            const Job& c = jobs[i0 + j];
            const int q = (i0 + j) % kSumRows;
            sum.x[q] = down.y[j];
            sum.acc[q] = acc_ + (size_t) c.row * row;
            sum.w[q] = c.w;
            if (q == kSumRows - 1 || i0 + j == n - 1) add_scaled(sum, q + 1, n_embd_, st);
        }
    }
    ck(cudaMemcpyAsync(out_, acc_, (size_t) m * row * sizeof(float), cudaMemcpyDeviceToHost, st), "--expert-gpu output");
    pending_ = m;
}

void ExpertCard::finish_into(float* rows) {
    if (!pending_) return;
    {
        OnDevice on(device_);
        ck(cudaStreamSynchronize((cudaStream_t) stream_), "--expert-gpu experts");
    }
    const size_t count = (size_t) pending_ * (size_t) n_embd_;
    pending_ = 0;
    for (size_t i = 0; i < count; ++i) rows[i] += out_[i];
}

// The experts the first card's cache did not take go to the second one, in the order the cache chose its own: the
// most used for their bytes first.  ponytail: placed once at load; a later resize of the first card's cache
// (resize_cache()) leaves these where they are, so an expert may then be on both cards (the first one's is used) or
// on neither (the CPU's).  Following the cache would move experts between the cards while it serves.
void Engine::fill_expert_card() {
    if (opt_.expert_gpu < 0 || n_moe_ == 0) return;
    int here = 0, cards = 0;
    ck(cudaGetDevice(&here), "--expert-gpu");
    ck(cudaGetDeviceCount(&cards), "--expert-gpu");
    if (opt_.expert_gpu >= cards || opt_.expert_gpu == here)
        throw std::runtime_error("--expert-gpu " + std::to_string(opt_.expert_gpu) + ": this process sees " +
                                 std::to_string(cards) + " CUDA device(s) and runs the model on device " +
                                 std::to_string(here) + "; name another one");
    ecard_ = std::make_unique<ExpertCard>(opt_.expert_gpu, n_layer_, n_expert_, n_embd_, n_ff_exp_);
    struct Cand {
        int il, e;
        size_t bytes;
        double value;
    };
    std::vector<Cand> cands;
    size_t lacked = 0;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe || !kernels::native_mmvq_supported((int) L.gu_type) ||
            !kernels::native_mmvq_supported((int) L.u_type) || !kernels::native_mmvq_supported((int) L.d_type))
            continue;
        // a slot as fill_expert_cache() sizes it: whole blocks of each type, and the zero tail the kernels may read
        const size_t unit = std::lcm(std::lcm(std::lcm(type_block_bytes(L.gu_type), type_block_bytes(L.u_type)),
                                              type_block_bytes(L.d_type)),
                                     size_t(8));
        const size_t bytes = (L.gu2() + L.d_bytes + kTail + unit - 1) / unit * unit;
        for (int e = 0; e < n_expert_; ++e) {
            if (L.slot[(size_t) e]) continue;
            const size_t i = (size_t) il * n_expert_ + e;
            const double used = (double) ((prior_.empty() ? 0 : prior_[i]) + use_[i]);
            cands.push_back({il, e, bytes, used / (double) bytes});
            lacked += bytes;
        }
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });
    const int reserve = opt_.expert_gpu_reserve_mib >= 0 ? opt_.expert_gpu_reserve_mib : 512;
    ecard_->claim(reserve, lacked);
    for (const Cand& c : cands) ecard_->put(layers_[(size_t) c.il], c.il, c.e, c.bytes);   // a smaller one may still fit
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.moe && (!L.h_s_gate.empty() || !L.h_s_up.empty())) ecard_->scales(il, L.h_s_gate, L.h_s_up);
    }
    std::fprintf(stderr,
                 "strata-poly: --expert-gpu %d: %d of the %zu routed experts the first card does not hold are on it "
                 "(%zu MiB, %d MiB left free)\n",
                 opt_.expert_gpu, ecard_->held(), cands.size(), ecard_->used() >> 20, reserve);
}

}  // namespace strata::poly
