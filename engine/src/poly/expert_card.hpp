// src/poly/expert_card.hpp - a second CUDA card for the routed experts the first one has no room for (--expert-gpu).
//
// The idea is the Strata project's two-card Inkling 2 engine (engine/src/inkling2/ep_dual_gpu.cu at v0.2.19, MIT):
// each routed expert has one card that owns it, the host routes, hands every card the rows of the tokens that chose its
// experts and adds up what comes back.  Here the first card keeps its expert cache as it is; the second one holds, at
// load, the experts the cache did not take, the most used first, as many as its VRAM has room for.  A call of a few
// tokens (Engine::moe()) gives this card the experts it holds and the CPU the ones neither card has; this card's sums
// are added to the CPU's rows before they go to the first card (Engine::moe_add()).  A prompt batch (moe_rows())
// stays on the first card, which stages what it lacks, as without this card.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::poly {

struct Layer;

class ExpertCard {
public:
    /// A routed expert of one token: its slot on this card (gate | up | down, as Layer::slot), its weight, the token's
    /// row among the call's rows, the expert.
    struct Job {
        const uint8_t* slot;
        float w;
        int row;
        int e;
    };

    /// device: the CUDA device (this process's number) the experts go to.  It is made current only inside the calls
    /// below, which leave the caller's device current again.
    ExpertCard(int device, int n_layer, int n_expert, int n_embd, int n_ff);
    ~ExpertCard();
    ExpertCard(const ExpertCard&) = delete;
    ExpertCard& operator=(const ExpertCard&) = delete;

    int device() const { return device_; }
    /// Claims the VRAM the experts may take: what the card has free less reserve_mib, and no more than `most`
    /// bytes.  Called once, before put().
    size_t claim(int reserve_mib, size_t most);
    /// Copies expert e of layer il into the next slot_bytes of the claim (false: no room left) and keeps it there.
    bool put(const Layer& L, int il, int e, size_t slot_bytes);
    /// Layer il's NVFP4 global scales of gate and up, one an expert, as Layer::h_s_gate and h_s_up (empty: none).
    void scales(int il, const std::vector<float>& gate, const std::vector<float>& up);
    /// Expert e of layer il on this card, or null.
    const uint8_t* slot(int il, int e) const {
        return slots_.empty() ? nullptr : slots_[(size_t) il * (size_t) n_expert_ + (size_t) e];
    }
    int held() const { return held_; }
    size_t used() const { return used_; }

    /// Starts n jobs of layer il for m rows: x is the rows' expert input (n_embd floats a row, pinned host memory, as
    /// the CPU's experts read it).  The activation is the layer's: SiTU when situ_beta > 0, else GELU when gelu, else
    /// SwiGLU clamped at limit.  finish_into() waits for it.
    void start(const Layer& L, int il, const float* x, int m, const Job* jobs, int n, float limit, bool gelu,
               float situ_beta, float situ_lb);
    /// Waits for start()'s work and adds its sums to `rows` (m rows of n_embd floats, host memory).
    void finish_into(float* rows);

private:
    int device_ = 0, n_layer_ = 0, n_expert_ = 0, n_embd_ = 0, n_ff_ = 0;
    std::vector<uint8_t*> slots_;      ///< per (layer, expert): its slot here, or null
    std::vector<float*> s_gate_, s_up_;   ///< per layer: the NVFP4 scales here, or null
    uint8_t* arena_ = nullptr;
    size_t arena_bytes_ = 0, used_ = 0;
    int held_ = 0;
    void* stream_ = nullptr;
    // scratch: the rows' inputs, as floats and quantized; gate|up, the activation and its quantization, and down's
    // output of kSlots experts at a time; the rows' sums here and on the host
    float *xf_ = nullptr, *g_ = nullptr, *h_ = nullptr, *d_ = nullptr, *acc_ = nullptr, *out_ = nullptr;
    void *xq_ = nullptr, *q8_ = nullptr;
    int pending_ = 0;   ///< the rows of the call under way; 0: none
};

}  // namespace strata::poly
