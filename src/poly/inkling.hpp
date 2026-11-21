// src/poly/inkling.hpp - Thinking Machines' Inkling 2 Mini (llama.cpp architecture "inkling2_loop"): 128 experts,
// 8 a token, 24 layers of which the middle twelve run three times.
//
// No rotary embedding: a layer's attention adds a bias by the distance between query and key, a learned table of
// 16 per distance that the query's own 16 coefficients a head weigh, and each query has a sink logit of its own.
// Five layers in six attend a window of 1,024 positions; every sixth attends everything, with its queries scaled up
// slowly past 8,192 positions.  Keys, values, the attention's output and the experts' sum each go through a short
// causal convolution of four taps.
//
// The residual stream starts afresh every six layers ("attention residuals"): what a block of six started from is
// kept, and the input of an attention, an FFN or the output is a learned mix of those and of the stream.
//
// Layers 6 to 17 are the core, run three times in a row ("passes").  Each pass starts from what layer 5 gave; from
// the second on, every core layer also gets a message from the pass before (a mix of what that pass left), which
// a small gate of its own blends into the attention's and the FFN's inputs.  Every run of a layer has its own
// attention cache and convolution state.
//
// In the engine a run of a layer is a "visit": 48 of them in the order they run (layers_, n_layer_), each with the
// weights of its physical layer.  The reference for the math is llama.cpp's src/models/inkling2.cpp and
// inkling2-loop.cpp (29b52d398b).
#pragma once

#include "engine.hpp"

#include <algorithm>

namespace strata::poly {

class Inkling final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    void alloc_caches() override;
    void assign_scratch() override;
    bool sees() const override { return true; }   // pictures (strata-vision's rows) in place of tokens
    /// step()'s rows in one head: the products, mixes and controllers for all of them, and each row's caches,
    /// convolution states and position from its own sequence (attn(), post_conv()).  The rows' positions go to the
    /// kernels by value, so a step is not recorded: Inkling sets no Layer::fixed, and step() records only those.
    bool batch_heads() const override { return true; }
    int widest() const override { return std::max(n_head_ * head_dim_, 2 * n_embd_); }
    int streams() const override { return kStreams; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void router_scores(const Layer& L, int n) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void finish_logits(float* logits, int rows) override;
    void shift_rows(ptrdiff_t by) override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    const Names* tensor_alias(const char* name) const override;
    const Names* meta_alias(const char* key) const override;

private:
    /// The residual streams of a token in r_, n_embd_ floats each: the stream itself, the four blocks' starts
    /// (the normed embedding, layer 5's output, layer 11's of this pass, layer 17's of the last pass), and what
    /// the pass before left for the messages (its layer 11's and 17's outputs).
    enum Stream { kCur = 0, kBlock0 = 1, kBank = 5, kBankOut = 6, kStreams = 7 };

    /// A visit's weights (its physical layer's: a later visit has the first one's pointers) and its caches.
    struct Block {
        int phys = 0;     ///< the physical layer
        int pass = 0;     ///< 0: the prelude, the epilogue and the core's first pass; 1, 2: the core's later passes
        bool swa = false;
        int extent = 0;   ///< the distances its bias table has
        Dev wq, wk, wv, wo, wgate, wrel, wsink, rel_bias, q_clip, rel_clip;
        Dev k_conv, v_conv, post_conv, ffn_post_conv;
        Dev q_norm, k_norm, v_norm;       ///< a full layer's three input norms (a window layer's is Layer::attn_norm)
        Dev k_conv_proj, v_conv_proj;     ///< a full layer's input-dependent taps
        Dev attn_query, ffn_query;        ///< the depth mix's queries: 1 (window) or 3 (full) for the attention
        Dev msg_query;                    ///< a core layer's: the messages' 4 queries
        Dev ctrl_down[4], ctrl_gain[4], ctrl_forget[4];   ///< a core layer's gates: q, k, v, ffn
        std::vector<float> probs_b;       ///< the bias the routed experts' pick adds to the gates
        Dev probs_b_dev;
        float* conv = nullptr;            ///< in state_: K, V, attention-out, FFN-out convolutions' 3 past inputs
        /// The caches, a row a position (max_context of them, or ring_ in a window visit): fp16 keys and values
        /// (planes 0, 1), or under --kv q8_0 the keys' codes and scales and the values' codes and scales (0 to 3), a
        /// Q8_0 block's fp16 scale for every 32 codes (store_q8()).
        uint8_t* plane[4] = {};
    };

    bool q8() const { return opt_.kv_format == KvFmt::kQ8_0; }
    int planes() const { return q8() ? 4 : 2; }
    size_t plane_row(int i) const;   ///< a row's bytes in plane i
    int phys_of(int v) const;
    int pass_of(int v) const;
    /// The blocks a layer's mix has before its attention (before) or its FFN (after its block is pushed).
    static int blocks_before(int ip) { return ip == 0 ? 0 : (ip - 1) / kBlockSize + 1; }
    static int blocks_at(int ip) { return ip / kBlockSize + 1; }
    static constexpr int kBlockSize = 6;
    /// loop.count's bound: the released model runs 3 passes.  Nothing else limits it (each pass more is 12 visits'
    /// caches and state); a larger value is taken for a bad file.
    static constexpr int kMaxPasses = 8;

    float* stream(int k) const { return r_ + (size_t) k * n_embd_; }
    /// Where a layer works on the stream of a call of n tokens: r_ itself for one, else res_ (gathered by the head,
    /// scattered back by the tail).
    float* work_stream(int n) const { return n == 1 ? stream(kCur) : res_; }
    void gather(int k, float* dst, int n, int first = 0);
    void scatter(int k, const float* src, int n);
    void copy_stream(int from, int to, int n);
    void attn(Block& B, int il, const float* xq, const float* xk, const float* xv, const void* const* q8, int n);
    int64_t attn_room(const Block& B, int n) const;
    void post_conv(const float* x, float* state, const Dev& w, float* acc, int n, bool fresh, const Row* rows);
    void conv_snaps(const float* x, const float* state, int channels, int n);
    void dump(const char* name, int il, const float* dev, int width, int n);
    int slot_of(int v) const;   ///< llama.cpp's layer index of visit v (its cache slot), for the dumps
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a visit, as layers_
    int n_phys_ = 0, core_start_ = 0, core_end_ = 0, n_passes_ = 0, rank_ = 0;
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rel_ = 0, extent_ = 0, n_conv_ = 4;
    int n_vocab_real_ = 0;
    float qk_clip_ = 0, rel_clip_ = 0, softcap_ = 0, w_scale_ = 0;
    float norm_eps_ = 1e-6f;      ///< the depth mix's and the controls' (the weighted norms take rms_eps_)
    float tau_alpha_ = 0;
    int tau_floor_ = 0;
    int window_ = 0;              ///< the tokens a window layer attends, the token itself among them
    int ring_ = 0;                ///< rows of a window layer's caches: the window and a call's tokens
    int keep_ = 0;                ///< rows of a window layer a sequence's state keeps
    Dev tok_norm_, out_query_;
    size_t conv_floats_ = 0;      ///< one visit's convolution state

    // device scratch, rows_ tokens each
    float *res_ = nullptr;                 ///< the stream (kCur) from a layer's head to its tail
    const Row* tail_rows_ = nullptr;       ///< step()'s rows (rows_of_) from a layer's head to its tail
    float *blk_[4] = {};                   ///< the blocks, gathered from r_
    float *bank_ = nullptr, *bank_out_ = nullptr;
    float *msg_[4] = {};                   ///< a core layer's messages: q, k, v, ffn
    float *mix_[3] = {};                   ///< the attention's inputs out of the depth mix
    float *in_[3] = {};                    ///< the normed (and controlled) q, k, v inputs
    float *ctl_[3] = {}, *ctl_work_ = nullptr;
    void* inq_[3] = {};                    ///< in_'s Q8_1 copies
    float* neg_inf_ = nullptr;             ///< -inf for the padded vocabulary's scores
    float *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *vraw_ = nullptr, *v_ = nullptr, *rel_ = nullptr,
          *sink_ = nullptr, *gate_ = nullptr, *ktap_ = nullptr, *vtap_ = nullptr, *att_ = nullptr, *o_ = nullptr;
    float* awork_ = nullptr;
    size_t awork_n_ = 0;
    /// unsloth's layout: architecture "inkling2", block_count the visits, loop_{attn,k,v,mlp}_*
    bool unsloth_ = false;
    void* msg_mem_ = nullptr;              ///< that layout's message queries, a core layer's [2304, 4] each
    static constexpr int kFusedRows = 16;  ///< the most tokens the controllers' and the router's fused kernels take
};

}  // namespace strata::poly
