// src/poly/ling3.hpp - Ling 3.0 Flash and Tiny (llama.cpp architecture "bailingmoe3").
//
// Flash: 42 layers and one block that guesses a token ahead; Tiny: 24 layers, no such block.  Some layers' attention
// is MLA over cached latents (a 512-value latent and 64 rotary values a row, every key attended; Tiny's query is
// low rank); the others are KDA, GLM 5.3 Flash's gated delta rule with its gates full rank.  One residual stream.
// The FFN is dense in the first layers (two in Flash, one in Tiny), then routed experts (Flash 512, Tiny 128; 8 used,
// from the best 4 of 8 groups) with a shared one after them.
#pragma once

#include "engine.hpp"
#include "strata/poly/kernels.hpp"

namespace strata::poly {

class Ling3 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    size_t recurrent_floats() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    void moe_shared(const Layer& L, int il, int n) override;
    void block_attend(int64_t pos, int n) override;
    void block_rows(int64_t pos, int n) override;
    const Names* tensor_alias(const char* name) const override;

private:
    struct Block {
        bool mla = false;
        Dev wo;
        // KDA
        Dev wq, wk, wv, conv_q, conv_k, conv_v, f_a, dt_b, beta, a, g_a, o_norm;
        float* conv = nullptr;  ///< 3 * d_inner * (d_conv - 1)
        float* s = nullptr;     ///< n_head * head_dim * head_dim
        // MLA
        Dev q, kv_a, kv_a_norm, k_b, v_b, gate;
        Dev q_a, q_a_norm, q_b;   ///< a low-rank query (Ling 3.0 tiny): q_b(rms_norm(q_a x)) in place of q
        uint16_t* kv = nullptr;   ///< max_context * (kv_rank + n_rot), fp16: the normed latent, then the turned key
        // dense FFN
        Dev ffn_gate, ffn_up, ffn_down;
        std::vector<float> probs_b;   ///< what the routing adds to each expert's gate before it picks
    };

    void kda(Block& B, int n);
    void mla(Block& B, int64_t pos, int n);
    void mla_store(Block& B, int64_t pos, int n);
    void dense_ffn(Block& B, int il, int n);

    std::vector<Block> blocks_;

    int n_head_ = 0, kda_dim_ = 0, d_inner_ = 0, n_kda_ = 0, n_mla_ = 0;
    int kv_rank_ = 0, mla_k_ = 0, mla_v_ = 0, n_rot_ = 0, nope_ = 0, row_ = 0;   ///< row_: kv_rank + n_rot
    int n_group_ = 1, group_used_ = 1;
    int q_lora_ = 0;   ///< attention.q_lora_rank, 0 for a full-rank query
    int chunk_ = 1;
    float kda_lb_ = 0, w_scale_ = 1;
    bool w_norm_ = false;
    Rope rope_;

    // device scratch
    float *proj_ = nullptr, *qkv_ = nullptr, *g1_ = nullptr, *g2_ = nullptr, *beta_ = nullptr, *gated_ = nullptr;
    float *q_ = nullptr, *qa_ = nullptr, *qan_ = nullptr, *qn_ = nullptr, *kva_ = nullptr, *lat_ = nullptr, *qabs_ = nullptr, *qf_ = nullptr,
          *att_ = nullptr, *kqv_ = nullptr, *vout_ = nullptr, *agate_ = nullptr, *score_ = nullptr, *awork_ = nullptr;
};

}  // namespace strata::poly
