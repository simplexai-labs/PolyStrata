// src/poly/laguna.hpp - poolside's Laguna (llama.cpp architecture "laguna"): S 2.1 is 118B-A8B.
//
// One residual stream and attention in every layer, with normed queries and keys.  Three layers in four attend
// only the last tokens (a sliding window, keys and values in a ring); the fourth, from the first layer on, attends
// every token so far.  Every layer turns its queries and keys by the position, the two kinds of layer differently:
// a window layer turns the whole head with plain RoPE, a full layer half of it with YaRN.  The window layers have
// more query heads than the full ones; every layer has the same key and value heads.  The attention's output is
// gated by the softplus of a projection of the layer's input, one gate a head (or one a value), before its output
// matrix.  The first layer's FFN is dense; the others' are routed experts with sigmoid gates, a selection bias and
// weights made to sum to one and then scaled, beside an ungated shared expert.  The output matrix is a tensor of
// its own.
#pragma once

#include "engine.hpp"
#include "strata/poly/kernels.hpp"

namespace strata::poly {

class Laguna final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return max_head_ * head_dim_; }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        bool swa = false;      ///< attends a sliding window; its caches are rings
        int n_head = 0;        ///< its query heads
        int n_gate = 0;        ///< its gate's outputs a token: n_head (a gate a head) or n_head * head_dim (a value)
        int n_rot = 0;         ///< the values of a head it turns
        float theta = 0;       ///< a rotary pair turns by position * theta^pair
        bool yarn = false;     ///< and by YaRN (yarn_), a full layer of a file that scales its positions
        Dev wq, wk, wv, wo, wg, q_norm, k_norm;
        uint16_t* keys = nullptr;   ///< rows of n_kv * head_dim, fp16: max_context of them, or ring_ in a window layer
        uint16_t* vals = nullptr;
        int8_t* k_codes = nullptr;  ///< --kv q8_0: a row is n_kv * head_dim codes and n_kv * head_dim / 32 scales
        uint16_t* k_scales = nullptr;
        int8_t* v_codes = nullptr;
        uint16_t* v_scales = nullptr;
        Dev ffn_gate, ffn_up, ffn_down;   ///< a dense layer's FFN
        std::vector<float> probs_b;       ///< the bias the routed experts' pick adds to the gates (zeros without one)
        Dev probs_b_dev;
    };

    void attn(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void rope_heads(const Block& B, const float* x, const Dev& norm, float* y, int heads, int64_t pos, int rows,
              const long long* at);
    void gate(const Block& B, int n);
    void store(const float* x, const Block& B, int64_t pos, int n, bool value);
    void dense_ffn(Block& B, int il, int n);
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<uint8_t> swa_;    ///< per layer: a window layer
    std::vector<int> heads_;      ///< per layer: its query heads
    int max_head_ = 0, max_gate_ = 0, n_kv_ = 0, head_dim_ = 0, n_dense_ = 0;
    int rot_full_ = 0, rot_swa_ = 0;          ///< the values of a head each kind of layer turns
    float theta_full_ = 0, theta_swa_ = 0;    ///< and its rotary base, as theta_scale
    bool yarn_on_ = false;                    ///< the full layers turn by YaRN
    RopeYarn yarn_;
    bool w_norm_ = false;     ///< the picked experts' weights are made to sum to one
    float w_scale_ = 0;       ///< and then scaled by this, when it is not 0
    int window_ = 0;          ///< the tokens a window layer attends, the token itself among them; 0: no window layers
    int ring_ = 0;            ///< rows of a window layer's caches: the window and a call's tokens
    int keep_ = 0;            ///< rows of a window layer a sequence's state keeps
    int chunk_ = 1;           ///< tokens the attention takes at a time

    // device scratch
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *gate_ = nullptr, *score_ = nullptr;
    size_t score_n_ = 0;       ///< the floats of score_
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
    float* q8w_ = nullptr;     ///< a --kv q8_0 prompt batch's K and V, dequantized to fp16 (kwork | vwork)
    size_t work_halves_ = 0;   ///< the halves of each of q8w_'s two halves
};

}  // namespace strata::poly
