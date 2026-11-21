// src/poly/inkling.cpp - the layers of Inkling 2 Mini.  The reference for the math is llama.cpp's
// src/models/inkling2.cpp and inkling2-loop.cpp (29b52d398b); the kernels' header has it written out
// (include/strata/poly/inkling_kernels.hpp).
#include "inkling.hpp"

#include "runtime.hpp"
#include "strata/poly/inkling_kernels.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

const char* const kCtrl[4] = {"q", "k", "v", "ffn"};

// Unsloth's files (architecture "inkling2", the loop in its metadata) name the loop's parts otherwise; byte for byte
// they are the ggml-org file's: attn = q, mlp = ffn.
const Names kTensorNames[] = {
    {"loop_q_ctrl_down.weight", {"loop_attn_down.weight", nullptr}},
    {"loop_q_ctrl_gain.weight", {"loop_attn_gain.weight", nullptr}},
    {"loop_q_ctrl_forget.weight", {"loop_attn_forget.weight", nullptr}},
    {"loop_k_ctrl_down.weight", {"loop_k_down.weight", nullptr}},
    {"loop_k_ctrl_gain.weight", {"loop_k_gain.weight", nullptr}},
    {"loop_k_ctrl_forget.weight", {"loop_k_forget.weight", nullptr}},
    {"loop_v_ctrl_down.weight", {"loop_v_down.weight", nullptr}},
    {"loop_v_ctrl_gain.weight", {"loop_v_gain.weight", nullptr}},
    {"loop_v_ctrl_forget.weight", {"loop_v_forget.weight", nullptr}},
    {"loop_ffn_ctrl_down.weight", {"loop_mlp_down.weight", nullptr}},
    {"loop_ffn_ctrl_gain.weight", {"loop_mlp_gain.weight", nullptr}},
    {"loop_ffn_ctrl_forget.weight", {"loop_mlp_forget.weight", nullptr}},
};
const Names kMetaNames[] = {
    {"loop.start", {"loop.core_start", nullptr}},
    {"loop.end", {"loop.core_end", nullptr}},
    {"loop.count", {"num_loops", nullptr}},
};

// Where a visit's keys and values go, in its planes pl: fp16 keys and values, or q8_0 codes and scales.
InklingKvDst kv_dst(bool q8, uint8_t* const* pl, int ring) {
    InklingKvDst dst;
    if (q8) {
        dst.kc = (int8_t*) pl[0], dst.ks = (uint16_t*) pl[1];
        dst.vc = (int8_t*) pl[2], dst.vs = (uint16_t*) pl[3];
    } else {
        dst.k = (uint16_t*) pl[0], dst.v = (uint16_t*) pl[1];
    }
    dst.ring = ring;
    return dst;
}

// The caches the attention of a token at `pos` reads, in the planes pl.
InklingAttnSeq attn_seq(bool q8, uint8_t* const* pl, int64_t pos, int64_t room) {
    InklingAttnSeq c;
    if (q8) {
        c.kq8 = Kvq8{(const int8_t*) pl[0], (const uint16_t*) pl[1]};
        c.vq8 = Kvq8{(const int8_t*) pl[2], (const uint16_t*) pl[3]};
    } else {
        c.kcache = (const uint16_t*) pl[0];
        c.vcache = (const uint16_t*) pl[1];
    }
    c.pos = pos;
    c.room = room;
    return c;
}

}  // namespace

// ---- visits ----------------------------------------------------------------------------------------------------

int Inkling::phys_of(int v) const {
    const int core = core_end_ - core_start_, end = core_start_ + n_passes_ * core;
    if (v < core_start_) return v;
    if (v < end) return core_start_ + (v - core_start_) % core;
    return v - (n_passes_ - 1) * core;
}

int Inkling::pass_of(int v) const {
    const int core = core_end_ - core_start_, end = core_start_ + n_passes_ * core;
    return v < core_start_ || v >= end ? 0 : (v - core_start_) / core;
}

// llama.cpp's cache slot of a visit: the physical layer for the first pass, then n_phys + the later passes' runs.
int Inkling::slot_of(int v) const {
    const int p = phys_of(v), q = pass_of(v);
    return q == 0 ? p : n_phys_ + (q - 1) * (core_end_ - core_start_) + (p - core_start_);
}

void Inkling::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

const Names* Inkling::tensor_alias(const char* name) const {
    for (const Names& n : kTensorNames)
        if (std::strcmp(n.name, name) == 0) return &n;
    return nullptr;
}

const Names* Inkling::meta_alias(const char* key) const {
    for (const Names& n : kMetaNames)
        if (std::strcmp(n.name, key) == 0) return &n;
    return nullptr;
}

void Inkling::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };
    unsloth_ = gg_->get("general.architecture")->s == "inkling2";
    core_start_ = shape_i("loop.start");
    core_end_ = shape_i("loop.end");
    n_passes_ = shape_i("loop.count");
    // ggml-org's block_count is the physical layers, unsloth's every visit
    n_phys_ = unsloth_ ? n_layer_ - (n_passes_ - 1) * (core_end_ - core_start_) : n_layer_;
    rank_ = shape_i("loop.gate_rank");
    const MetaValue* varying = find("loop.varying");
    need(!varying || varying->u == 0, "a loop whose passes vary (inkling2_loop.loop.varying)");
    need(shape_i("attn_res.block_size") == kBlockSize, "attn_res.block_size is not 6");
    // the streams in r_ hold four blocks and one block of the pass before: Inkling 2 Mini's shape
    need(n_phys_ == 24 && core_start_ == 6 && core_end_ == 18 && n_passes_ >= 2 && n_passes_ <= kMaxPasses,
         "a loop other than layers 6 to 17 of 24, or a bad loop.count");
    // the controllers' decode kernel holds the rank's values in a warp's registers (inkling_control_fused())
    if (rank_ <= 0 || rank_ > kInklingMaxRank)
        throw std::runtime_error("unsupported model: loop.gate_rank is " + std::to_string(rank_) + ", 1 to " +
                                 std::to_string(kInklingMaxRank) + " is supported");
    n_layer_ = n_phys_ + (n_passes_ - 1) * (core_end_ - core_start_);

    const MetaValue& heads = must("attention.head_count");
    const MetaValue& kv = must("attention.head_count_kv");
    n_head_ = (int) of_layer(heads, 0);
    n_kv_ = (int) of_layer(kv, 0);
    for (int il = 0; il < n_phys_; ++il)
        need(of_layer(heads, il) == n_head_ && of_layer(kv, il) == n_kv_, "layers with different head counts");
    head_dim_ = shape_i("attention.key_length");
    need(shape_i("attention.value_length") == head_dim_, "the keys and values have different widths");
    need(head_dim_ == 128 && n_kv_ > 0 && n_head_ == 3 * n_kv_, "attention other than 3 query heads of 128 a key");
    n_rel_ = shape_i("attention.relative_length");
    extent_ = shape_i("attention.relative_extent");
    window_ = shape_i("attention.sliding_window");
    need(n_rel_ > 0 && n_rel_ <= 32 && extent_ > 0 && window_ > 0, "bad relative-position or window sizes");
    n_conv_ = shape_i("shortconv.l_cache");
    need(n_conv_ == 4, "shortconv.l_cache is not 4");
    const MetaValue& swa = must("attention.sliding_window_pattern");
    for (int il = 0; il < n_phys_; ++il) of_layer(swa, il);
    qk_clip_ = shape_f("attention.qk_clip");
    rel_clip_ = shape_f("attention.relative_clip");
    tau_floor_ = shape_i("attention.log_scale_floor");
    tau_alpha_ = shape_f("attention.log_scale_alpha");
    need(tau_floor_ > 0 && qk_clip_ > 0.f && rel_clip_ > 0.f, "bad clips or log scale");
    const MetaValue* cap = find("final_logit_softcapping");
    softcap_ = cap ? (float) cap->f : 0.f;
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;
    reserve_mmq_ = true;   // MMQ's scratch before the expert cache takes the VRAM
    lowbit_experts_ = true;   // Unsloth's files: IQ1_S gate_up and Q2_0 down experts
#if !defined(STRATA_USE_HIP)
    lane_slots_ = true;       // the cached experts' launches through MmvqSlots::lanes' kernels (measured on CUDA)
#endif
    situ_ = Situ{shape_f("activation.situ_beta"), shape_f("activation.situ_linear_beta")};
    need(situ_->beta > 0.f, "bad activation.situ_beta");
    const MetaValue* real = find("vocab_size_unpadded");
    n_vocab_real_ = real ? (int) real->u : n_vocab_;
    need(n_vocab_real_ > 0 && n_vocab_real_ <= n_vocab_, "bad vocab_size_unpadded");
    conv_floats_ = (size_t) (n_conv_ - 1) * (2 * (size_t) n_kv_ * head_dim_ + 2 * (size_t) n_embd_);

    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;

    // Not supported yet: a draft model and a split over cards.
    if (!opt_.draft_model.empty()) throw std::runtime_error("Inkling 2: --draft-model is not supported yet");
    // step()'s rows are a sequence's each, and the _seqs kernels take kInklingSeqRows of them a launch
    if (opt_.batch + 1 > kInklingSeqRows)
        throw std::runtime_error("Inkling 2: --batch takes at most " + std::to_string(kInklingSeqRows - 1));
    if (opt_.kv_format != KvFmt::kFp16 && opt_.kv_format != KvFmt::kQ8_0)
        throw std::runtime_error("Inkling 2: the attention cache is fp16 or q8_0");
    if (!opt_.layer_split.empty()) throw std::runtime_error("Inkling 2: --layer-split is not supported yet");
    // The file has no block that guesses ahead, so a window's guesses come from the conversation's repeats
    // (--lookup) only: opt-in with --window, a token at a time by default.
    if (opt_.window == 0) opt_.window = 1;
    opt_.draft = false;
    std::fprintf(stderr, "strata-poly: %s: %d layers, %d to %d run %d times: %d visits\n",
                 unsloth_ ? "inkling2 (looped)" : "inkling2_loop", n_phys_, core_start_, core_end_ - 1, n_passes_,
                 n_layer_);
}

void Inkling::load_layer(int v) {
    Layer& L = layers_[(size_t) v];
    Block& B = blocks_[(size_t) v];
    const int p = phys_of(v);
    const int first = p < core_end_ ? p : p + (n_passes_ - 1) * (core_end_ - core_start_);
    if (v != first) {   // a later visit: the first one's weights, and caches of its own
        const Block& F = blocks_[(size_t) first];
        B = F;
        B.pass = pass_of(v);
        B.conv = nullptr;
        for (uint8_t*& q : B.plane) q = nullptr;
        L = layers_[(size_t) first];
        L.moe = false;   // the first visit's experts: one copy of each in the cache (Layer::alias)
        L.alias = first;
        L.slot.clear();
        return;
    }
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(p, n), blk(p, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(p, n), blk(p, n), a, b, kAnyMat); };
    const int64_t E = n_embd_, hq = (int64_t) n_head_ * head_dim_, hk = (int64_t) n_kv_ * head_dim_;

    if (v == 0) {   // the model's own small tensors, beside the first layer's
        const TensorInfo* tn = gg_->find("token_embd_norm.weight");
        tok_norm_ = upload(tn, "token_embd_norm.weight", E, 1, kF32);
        out_query_ = upload(gg_->find("output_res_query.weight"), "output_res_query.weight", E, 1, kF32);
    }
    B.phys = p;
    B.pass = 0;
    B.swa = of_layer(*meta_in(*gg_, "attention.sliding_window_pattern"), p) != 0;
    B.wq = mat("attn_q.weight", E, hq);
    B.wk = mat("attn_k.weight", E, hk);
    B.wv = mat("attn_v.weight", E, hk);
    B.wo = mat("attn_output.weight", hq, E);
    B.wgate = mat("attn_gate.weight", E, hq);
    B.wrel = mat("attn_rel.weight", E, (int64_t) n_rel_ * n_head_);
    B.wsink = mat("attn_sink_proj.weight", E, n_head_);
    const TensorInfo* rb = tensor(p, "attn_rel_bias.weight");
    if (!rb || rb->shape.size() != 2 || (int) rb->shape[0] != n_rel_)
        throw std::runtime_error(blk(p, "attn_rel_bias.weight") + " is missing or not [" + std::to_string(n_rel_) +
                                 ", extent]");
    B.extent = (int) rb->shape[1];
    if (B.extent != (B.swa ? window_ : extent_))
        throw std::runtime_error(blk(p, "attn_rel_bias.weight") + " has " + std::to_string(B.extent) +
                                 " distances, expected " + std::to_string(B.swa ? window_ : extent_));
    B.rel_bias = f32("attn_rel_bias.weight", n_rel_, B.extent);
    B.q_clip = f32("attn_q_clip.weight", head_dim_, n_head_);
    B.rel_clip = f32("attn_rel_clip.weight", n_rel_, n_head_);
    B.k_conv = f32("attn_k_conv.weight", n_conv_, hk);
    B.v_conv = f32("attn_v_conv.weight", n_conv_, hk);
    B.post_conv = f32("attn_post_conv.weight", n_conv_, E);
    B.ffn_post_conv = f32("ffn_post_conv.weight", n_conv_, E);
    if (B.swa) {
        L.attn_norm = f32("attn_norm.weight", E);
    } else {
        B.q_norm = f32("attn_q_inp_norm.weight", E);
        B.k_norm = f32("attn_k_inp_norm.weight", E);
        B.v_norm = f32("attn_v_inp_norm.weight", E);
        B.k_conv_proj = mat("attn_k_conv_proj.weight", E, (int64_t) n_conv_ * n_kv_);
        B.v_conv_proj = mat("attn_v_conv_proj.weight", E, (int64_t) n_conv_ * n_kv_);
    }
    if (p > 0) B.attn_query = f32("attn_res_query.weight", E, B.swa ? 1 : 3);
    B.ffn_query = f32("ffn_res_query.weight", E);
    L.ffn_norm = f32("ffn_norm.weight", E);
    if (p >= core_start_ && p < core_end_) {
        if (unsloth_ && !tensor(p, "loop_msg_query.weight")) {
            // unsloth's queries are two tensors of the model, [2304, 24] each: row 2i and 2i + 1 of loop_res_query are
            // core layer start + i's q and ffn queries, of loop_kv_res_query its k and v ones
            const TensorInfo* rq = gg_->find("loop_res_query.weight");
            const TensorInfo* kq = gg_->find("loop_kv_res_query.weight");
            const int nc = core_end_ - core_start_;
            for (const TensorInfo* t : {rq, kq})
                if (!t || t->type != kTypeF32 || (int64_t) t->elements() != 2 * nc * E)
                    throw std::runtime_error("inkling2: loop_res_query / loop_kv_res_query missing or not F32 [2304, " +
                                             std::to_string(2 * nc) + "]");
            if (dense_) {   // the second pass of the load: VRAM of its own beside the arena
                if (!msg_mem_) msg_mem_ = dmalloc((size_t) nc * 4 * E * sizeof(float), "the loop's queries");
                const int i = p - core_start_;
                const float* r = (const float*) home(*rq).data(*rq);
                const float* k = (const float*) home(*kq).data(*kq);
                std::vector<float> q((size_t) 4 * E);
                std::memcpy(q.data(), r + (size_t) (2 * i) * E, E * sizeof(float));
                std::memcpy(q.data() + E, k + (size_t) (2 * i) * E, E * sizeof(float));
                std::memcpy(q.data() + 2 * E, k + (size_t) (2 * i + 1) * E, E * sizeof(float));
                std::memcpy(q.data() + 3 * E, r + (size_t) (2 * i + 1) * E, E * sizeof(float));
                float* d = (float*) msg_mem_ + (size_t) i * 4 * E;
                h2d(d, q.data(), q.size() * sizeof(float));
                B.msg_query = Dev{d, kTypeF32, (int) E, 4};
            }
        } else {
            B.msg_query = f32("loop_msg_query.weight", E, 4);
        }
        for (int c = 0; c < 4; ++c) {
            const std::string pre = std::string("loop_") + kCtrl[c] + "_ctrl_";
            B.ctrl_down[c] = f32((pre + "down.weight").c_str(), 2 * E, rank_);
            B.ctrl_gain[c] = f32((pre + "gain.weight").c_str(), rank_, E);
            B.ctrl_forget[c] = f32((pre + "forget.weight").c_str(), rank_, E);
        }
    }
    load_moe(v, false, p, true);   // Unsloth's files scale every expert's gate_up and down, whatever their type
    if (L.gate_inp.type != kTypeF32)   // router_scores() takes it as floats
        throw std::runtime_error(blk(p, "ffn_gate_inp.weight") + " is not F32");
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(p, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(p, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(p, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

size_t Inkling::recurrent_floats() const { return (size_t) n_layer_ * conv_floats_; }

// Each visit's convolution state in state_; a window visit's caches are a ring of the rows a call can read, a full
// one's a row a position.
void Inkling::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    size_t one = 0;   // a position's bytes in a visit's planes together
    for (int i = 0; i < planes(); ++i) one += plane_row(i);
    if (q8()) std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", one / 2);
    size_t rings = 0;
    for (int v = 0; v < (int) blocks_.size(); ++v)
        if (runs(v) && blocks_[(size_t) v].swa) rings += (size_t) ring_ * one;
    uint8_t* kv = rings ? (uint8_t*) seq_alloc(rings, "the attention cache") : nullptr;
    float* st = state_;
    for (int v = 0; v < (int) blocks_.size(); ++v) {
        Block& B = blocks_[(size_t) v];
        if (!runs(v)) continue;
        B.conv = st;
        st += conv_floats_;
        for (int i = 0; i < planes(); ++i) {
            if (B.swa) {
                B.plane[i] = kv;
                kv += (size_t) ring_ * plane_row(i);
            } else {
                B.plane[i] = (uint8_t*) kv_alloc(plane_row(i), 1);
            }
        }
    }
    // the padded vocabulary's scores, which finish_logits() puts out of reach
    if (!neg_inf_ && n_vocab_ > n_vocab_real_) {
        const size_t pad_n = (size_t) (n_vocab_ - n_vocab_real_);
        neg_inf_ = (float*) dmalloc(pad_n * sizeof(float), "scratch");
        std::vector<float> h(pad_n, -INFINITY);
        ck(cudaMemcpy(neg_inf_, h.data(), pad_n * sizeof(float), cudaMemcpyHostToDevice), "scratch");
    }
}

// Per-token buffers hold rows_ tokens, row after row.
void Inkling::assign_scratch() {
    const size_t n = (size_t) rows_, e = n * n_embd_, hq = n * n_head_ * head_dim_, hk = n * n_kv_ * head_dim_;
    res_ = fbuf(e);
    for (float*& b : blk_) b = fbuf(e);
    bank_ = fbuf(e);
    bank_out_ = fbuf(e);
    for (float*& m : msg_) m = fbuf(e);
    for (float*& m : mix_) m = fbuf(e);
    for (float*& m : in_) m = fbuf(e);
    for (float*& m : ctl_) m = fbuf(e);
    const size_t q8 = (n * kernels::native_q8_1_bytes(n_embd_) + sizeof(float) - 1) / sizeof(float);
    for (void*& m : inq_) m = fbuf(q8);
    ctl_work_ = fbuf(n * (4 * (size_t) n_embd_ + rank_));
    q_ = fbuf(hq);
    gate_ = fbuf(hq);
    att_ = fbuf(hq);
    kraw_ = fbuf(hk);
    k_ = fbuf(hk);
    vraw_ = fbuf(hk);
    v_ = fbuf(hk);
    rel_ = fbuf(n * n_head_ * n_rel_);
    sink_ = fbuf(n * n_head_);
    ktap_ = fbuf(n * n_kv_ * n_conv_);
    vtap_ = fbuf(n * n_kv_ * n_conv_);
    o_ = fbuf(e);
    awork_n_ = inkling_attend_work_floats(n_head_, head_dim_);
    awork_ = awork_n_ ? fbuf(awork_n_) : nullptr;
}

// ---- the streams in r_ -----------------------------------------------------------------------------------------

void Inkling::gather(int k, float* dst, int n, int first) {
    const size_t e = (size_t) n_embd_ * sizeof(float), rs = e * kStreams;
    ck(cudaMemcpy2DAsync(dst, e, r_ + (size_t) first * kStreams * n_embd_ + (size_t) k * n_embd_, rs, e, (size_t) n,
                         cudaMemcpyDeviceToDevice, cs(st_)),
       "streams");
}

void Inkling::scatter(int k, const float* src, int n) {
    const size_t e = (size_t) n_embd_ * sizeof(float), rs = e * kStreams;
    ck(cudaMemcpy2DAsync(stream(k), rs, src, e, e, (size_t) n, cudaMemcpyDeviceToDevice, cs(st_)), "streams");
}

void Inkling::copy_stream(int from, int to, int n) {
    const size_t e = (size_t) n_embd_ * sizeof(float), rs = e * kStreams;
    ck(cudaMemcpy2DAsync(stream(to), rs, stream(from), rs, e, (size_t) n, cudaMemcpyDeviceToDevice, cs(st_)),
       "streams");
}

// --dump-layers (Engine::dbg()): a stage of n tokens, by llama.cpp's index of the visit (its cache slot)
void Inkling::dump(const char* name, int v, const float* dev, int width, int n) {
    dbg(name, v >= 0 ? slot_of(v) : -1, dev, width * n);
}

// ---- a layer ---------------------------------------------------------------------------------------------------

// The rows the attention of n tokens from pos_ is sized for: a full visit's in room()'s steps rather than pos + n,
// so that its splits, and so a token's bits, do not change with every position.  A window visit's serves every
// position: sized for the context, it gets the window's splits and the one more a window not starting on a split's
// edge spans (window + n - 1 rows would leave that one out).
int64_t Inkling::attn_room(const Block& B, int n) const {
    return B.swa ? (int64_t) opt_.max_context : Engine::room(n);
}

// A post-convolution of n rows into acc (written when `fresh`, else added), its state at `state` in the bound
// sequence's state_, or with step()'s rows at the same place in each row's own sequence's.
void Inkling::post_conv(const float* x, float* state, const Dev& w, float* acc, int n, bool fresh, const Row* rows) {
    if (rows) {
        float* states[kInklingSeqRows];
        for (int i = 0; i < n; ++i) states[i] = seq_state(rows[i].seq, state);
        if (fresh) inkling_conv_seqs(x, states, fp(w), acc, n_embd_, n, st_);
        else inkling_conv_acc_seqs(x, states, fp(w), acc, n_embd_, n, st_);
    } else {
        conv_snaps(x, state, n_embd_, n);
        if (fresh) inkling_conv(x, state, fp(w), nullptr, 0, acc, n_embd_, n, st_);
        else inkling_conv_acc(x, state, fp(w), nullptr, 0, acc, n_embd_, n, st_);
    }
}

// A window's snapshots of a convolution's state (Engine::window()), before the call's n rows of x advance it.
void Inkling::conv_snaps(const float* x, const float* state, int channels, int n) {
    if (snap_bits_) inkling_conv_snaps(x, state, channels, n, snap_bits_, snap_at_ + (state - state_), rec_n_, st_);
}

// The attention from its three normed inputs (and their Q8_1 copies) to ao_, through the output's convolution.
void Inkling::attn(Block& B, int v, const float* xq, const float* xk, const float* xv, const void* const* xq8, int n) {
    const int hk = n_kv_ * head_dim_, hq = n_head_ * head_dim_;
    // the projections of the three inputs, those of one format in one launch
    const LinJob proj[8] = {{&B.wq, xq, q_, xq8[0]},     {&B.wgate, xq, gate_, xq8[0]},
                            {&B.wrel, xq, rel_, xq8[0]}, {&B.wsink, xq, sink_, xq8[0]},
                            {&B.wk, xk, kraw_, xq8[1]},  {&B.wv, xv, vraw_, xq8[2]},
                            {&B.k_conv_proj, xk, ktap_, xq8[1]}, {&B.v_conv_proj, xv, vtap_, xq8[2]}};
    lin_jobs(proj, B.swa ? 6 : 8, n);
    const size_t v_at = (size_t) (n_conv_ - 1) * hk;   // the V convolution's state, after K's
    const InklingTau tau = B.swa ? InklingTau{} : InklingTau{tau_alpha_, tau_floor_};
    InklingClipJob jobs[2];
    jobs[0] = {q_, 0, fp(B.q_clip), q_, head_dim_, n_head_, qk_clip_, tau};
    jobs[1] = {rel_, 0, fp(B.rel_clip), rel_, n_rel_, n_head_, rel_clip_, tau};
    InklingRel rel;
    rel.r = rel_;
    rel.table = fp(B.rel_bias);
    rel.n_rel = n_rel_;
    rel.extent = B.extent;
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    // K and V from their projections into the cache in one launch, Q's and R's clips in another, then the attention
    if (rows_of_) {   // step(): each row a token of its own sequence, with its caches, convolution state and position
        InklingKvSeq kv[kInklingSeqRows];
        InklingAttnSeq at[kInklingSeqRows];
        int64_t pos[kInklingSeqRows];
        for (int i = 0; i < n; ++i) {
            const Row& r = rows_of_[i];
            uint8_t* pl[4] = {};
            for (int j = 0; j < planes(); ++j) pl[j] = (uint8_t*) seq_buf(r.seq, B.plane[j]);
            float* const conv = seq_state(r.seq, B.conv);
            kv[i].dst = kv_dst(q8(), pl, ring);
            kv[i].k_state = conv;
            kv[i].v_state = conv + v_at;
            kv[i].pos = pos[i] = r.pos;
            const int64_t keep = pos_;   // its room as alone (room() reads pos_)
            pos_ = r.pos;
            at[i] = attn_seq(q8(), pl, r.pos, attn_room(B, 1));
            pos_ = keep;
        }
        inkling_kv_store_seqs(kraw_, vraw_, kv, fp(B.k_conv), fp(B.v_conv), B.swa ? nullptr : ktap_,
                              B.swa ? nullptr : vtap_, n_kv_, head_dim_, n, qk_clip_, rms_eps_, k_, v_, st_);
        inkling_clip_many_seqs(jobs, 2, n, pos, st_);
        inkling_attend_seqs(q_, at, n, rel, sink_, att_, head_dim_, n_head_, n_kv_, scale, window, ring, awork_,
                            awork_n_, st_);
    } else {
        conv_snaps(kraw_, B.conv, hk, n);
        conv_snaps(vraw_, B.conv + v_at, hk, n);
        inkling_kv_store(kraw_, vraw_, B.conv, B.conv + v_at, fp(B.k_conv), fp(B.v_conv), B.swa ? nullptr : ktap_,
                         B.swa ? nullptr : vtap_, n_kv_, head_dim_, n, qk_clip_, rms_eps_, kv_dst(q8(), B.plane, ring),
                         pos_, k_, v_, st_);
        inkling_clip_many(jobs, 2, n, pos_, st_);
        dump("Qcur", v, q_, hq, n);
        dump("Kcur", v, k_, hk, n);
        dump("Vcur", v, v_, hk, n);
        dump("Rcur", v, rel_, n_head_ * n_rel_, n);
        const InklingAttnSeq c = attn_seq(q8(), B.plane, pos_, attn_room(B, n));
        inkling_attend(q_, c.kcache, c.vcache, c.kq8, c.vq8, rel, sink_, pos_, n, att_, head_dim_, n_head_, n_kv_,
                       scale, window, ring, awork_, awork_n_, st_, c.room);
    }
    dump("kqv_out", v, att_, hq, n);
    mul_sigmoid(att_, gate_, hq, n, hq, false, st_);
    lin(B.wo, att_, o_, n);   // the post-convolution: layer_head(), with the residual
}

// A visit from its input to its router's copy.  The streams: see inkling.hpp.
void Inkling::layer_head(Layer& L, int v, int n) {
    Block& B = blocks_[(size_t) v];
    const int ip = B.phys;
    const size_t e = (size_t) n * n_embd_;
    // One token's streams lie one after the other in r_, as a buffer of one row each: the mixes read them there and
    // the stream is worked on in place.  More tokens' are gathered, a buffer a stream, and the stream goes back in the
    // tail.
    const bool direct = n == 1;
    float* const cres = work_stream(n);
    tail_rows_ = rows_of_;   // step()'s rows, for the tail's convolution
    if (v == 0) {   // the embedding, normed, is the first block's start
        gather(kCur, o_, n);
        rms_norm(o_, fp(tok_norm_), res_, n_embd_, n, rms_eps_, st_);
        // a picture's rows go in as the encoder made them: llama.cpp norms token embeddings only (inkling2.cpp,
        // `if (ubatch.token)`).  A call can hold text and picture rows, so the raw rows are put back row by row.
        // step()'s rows have none (Engine::step()).
        if (!rows_of_)
            for (const Image& im : images_) {
                const int64_t a = std::max<int64_t>(im.at, pos_), b = std::min<int64_t>(im.at + im.n, pos_ + n);
                if (a < b)
                    ck(cudaMemcpyAsync(res_ + (size_t) (a - pos_) * n_embd_, o_ + (size_t) (a - pos_) * n_embd_,
                                       (size_t) (b - a) * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
                       "picture rows");
            }
        if (direct) ck(cudaMemcpyAsync(cres, res_, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "streams");
        else scatter(kCur, res_, n);
    } else {
        // a later pass of the core starts from layer 5's output, with what the pass before left for its messages:
        // the start of the block in the core's middle (its layer 11's output) and the stream (its layer 17's)
        if (B.pass > 0 && ip == core_start_) {
            const int core_in = kBlock0 + core_start_ / kBlockSize;
            copy_stream(core_in + 1, kBank, n);
            copy_stream(kCur, kBankOut, n);
            copy_stream(core_in, kCur, n);
        }
        if (!direct) gather(kCur, res_, n);
    }
    dump("l_in", v, cres, n_embd_, n);
    const int nb = blocks_before(ip);
    const float* bp[4];   // the blocks' starts
    for (int k = 0; k < nb; ++k) {
        if (direct) {
            bp[k] = stream(kBlock0 + k);
        } else {
            gather(kBlock0 + k, blk_[k], n);
            bp[k] = blk_[k];
        }
    }

    // the attention's inputs: the stream, or its mix with the blocks (one for a window layer, Q, K and V apart
    // for a full one)
    const float* in[3] = {cres, cres, cres};
    if (B.attn_query.p) {
        InklingMix m;
        for (int k = 0; k < nb; ++k) m.state[k] = bp[k];
        m.state[nb] = cres;
        m.n_state = nb + 1;
        m.query = fp(B.attn_query);
        m.n_query = B.swa ? 1 : 3;
        for (int j = 0; j < m.n_query; ++j) m.out[j] = mix_[j];
        inkling_mix(m, n_embd_, n, norm_eps_, st_);
        for (int j = 0; j < 3; ++j) in[j] = mix_[B.swa ? 0 : j];
    }
    // a later pass's messages: the pass before's mix, gated into each input
    if (B.pass > 0) {
        if (!direct) {
            gather(kBank, bank_, n);
            gather(kBankOut, bank_out_, n);
        }
        InklingMix m;
        m.state[0] = direct ? stream(kBank) : bank_;
        m.state[1] = direct ? stream(kBankOut) : bank_out_;
        m.n_state = 2;
        m.query = fp(B.msg_query);
        m.n_query = 4;
        for (int j = 0; j < 4; ++j) m.out[j] = msg_[j];
        inkling_mix(m, n_embd_, n, norm_eps_, st_);
        if (n <= kFusedRows) {   // the three controllers in two launches; each out its own buffer
            InklingCtrl c[3];
            for (int j = 0; j < 3; ++j)
                c[j] = {in[j], msg_[j], fp(B.ctrl_down[j]), fp(B.ctrl_gain[j]), fp(B.ctrl_forget[j]), ctl_[j]};
            inkling_control_fused(c, 3, n_embd_, rank_, n, norm_eps_, ctl_work_, st_);
            for (int j = 0; j < 3; ++j) in[j] = ctl_[j];
        } else {
            for (int j = 0; j < 3; ++j) {
                inkling_control(blas_, in[j], msg_[j], fp(B.ctrl_down[j]), fp(B.ctrl_gain[j]), fp(B.ctrl_forget[j]),
                                ctl_work_, ctl_[j], n_embd_, rank_, n, norm_eps_, st_);
                in[j] = ctl_[j];
            }
        }
    }
    const int threads = sum_threads(n_embd_);
    const void* q8[3];
    const auto norm = [&](int j, const float* x, const Dev& w) {
        kernels::native_add_rms_norm_q8_1(const_cast<float*>(x), nullptr, fp(w), in_[j], inq_[j], n_embd_, n,
                                          rms_eps_, threads, st_);
        q8[j] = inq_[j];
    };
    const float* x[3];
    if (B.swa && B.pass == 0) {
        norm(0, in[0], L.attn_norm);
        x[0] = x[1] = x[2] = in_[0];
        q8[1] = q8[2] = q8[0];
    } else {
        const Dev* w[3] = {&L.attn_norm, &L.attn_norm, &L.attn_norm};
        if (!B.swa) w[0] = &B.q_norm, w[1] = &B.k_norm, w[2] = &B.v_norm;
        for (int j = 0; j < 3; ++j) {
            norm(j, in[j], *w[j]);
            x[j] = in_[j];
        }
    }
    attn(B, v, x[0], x[1], x[2], q8, n);

    // the first layer of a block starts the stream afresh, and what it started from is the block's
    float* const post = B.conv + (size_t) 2 * (n_conv_ - 1) * n_kv_ * head_dim_;   // the attention-out's state
    if (ip % kBlockSize == 0) {
        const int k = ip / kBlockSize;
        if (direct) {
            ck(cudaMemcpyAsync(stream(kBlock0 + k), cres, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
               "streams");
            bp[k] = stream(kBlock0 + k);
        } else {
            scatter(kBlock0 + k, res_, n);
            ck(cudaMemcpyAsync(blk_[k], res_, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "streams");
            bp[k] = blk_[k];
        }
        post_conv(o_, post, B.post_conv, cres, n, true, rows_of_);
    } else {
        post_conv(o_, post, B.post_conv, cres, n, false, rows_of_);
    }
    dump("ffn_inp", v, cres, n_embd_, n);

    // the FFN's input: the stream mixed with the blocks, gated with its message in a later pass
    const int nf = blocks_at(ip);
    InklingMix m;
    for (int k = 0; k < nf; ++k) m.state[k] = bp[k];
    m.state[nf] = cres;
    m.n_state = nf + 1;
    m.query = fp(B.ffn_query);
    m.n_query = 1;
    m.out[0] = mix_[0];
    inkling_mix(m, n_embd_, n, norm_eps_, st_);
    float* f = mix_[0];
    if (B.pass > 0) {
        if (n <= kFusedRows) {
            const InklingCtrl c{f, msg_[3], fp(B.ctrl_down[3]), fp(B.ctrl_gain[3]), fp(B.ctrl_forget[3]), ctl_[0]};
            inkling_control_fused(&c, 1, n_embd_, rank_, n, norm_eps_, ctl_work_, st_);
        } else {
            inkling_control(blas_, f, msg_[3], fp(B.ctrl_down[3]), fp(B.ctrl_gain[3]), fp(B.ctrl_forget[3]),
                            ctl_work_, ctl_[0], n_embd_, rank_, n, norm_eps_, st_);
        }
        f = ctl_[0];
    }
    kernels::native_add_rms_norm_q8_1(f, nullptr, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dump("ffn_in", v, xn_, n_embd_, n);
    if (!batched(layers_[(size_t) home_of(v)], n)) moe_router(L, n);   // the experts' layer decides (Layer::alias)
}

// The routed experts' sum through its convolution, into the stream.
void Inkling::layer_tail(Layer&, int v, int n) {
    Block& B = blocks_[(size_t) v];
    const size_t hk = (size_t) n_kv_ * head_dim_;
    dump("ffn_moe_out", v, fo_, n_embd_, n);
    float* const state = B.conv + (size_t) (n_conv_ - 1) * (2 * hk + n_embd_);
    // the stream at this half's rows: a window's halves (shift_rows()) run the other half's head in between
    float* const cres = work_stream(n);
    post_conv(fo_, state, B.ffn_post_conv, cres, n, false, tail_rows_);
    tail_rows_ = nullptr;
    if (cres == res_) scatter(kCur, res_, n);   // several tokens' stream back to r_ (one token's is worked on there)
    dump("l_out", v, cres, n_embd_, n);
}

// A half of a window at its rows: the engine's buffers and the stream a layer's head leaves for its tail.
void Inkling::shift_rows(ptrdiff_t by) {
    Engine::shift_rows(by);
    res_ += by * n_embd_;
}

// The output's input: the stream mixed with the four blocks, normed.
void Inkling::output_rows(int first, int rows, float* h) {
    const int nb = blocks_at(n_phys_ - 1);
    InklingMix m;
    for (int k = 0; k < nb; ++k) {
        gather(kBlock0 + k, blk_[k], rows, first);
        m.state[k] = blk_[k];
    }
    gather(kCur, mix_[1], rows, first);
    m.state[nb] = mix_[1];
    m.n_state = nb + 1;
    m.query = fp(out_query_);
    m.n_query = 1;
    m.out[0] = mix_[0];
    inkling_mix(m, n_embd_, rows, norm_eps_, st_);
    rms_norm(mix_[0], fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
    dump("result_norm", -1, h, n_embd_, rows);
}

// The final softcap, and the padded vocabulary's rows out of reach.
void Inkling::finish_logits(float* logits, int rows) {
    if (softcap_ > 0.f) softcap(logits, (size_t) rows * n_vocab_, softcap_, st_);
    if (!neg_inf_) return;
    const size_t pad_n = (size_t) (n_vocab_ - n_vocab_real_);
    for (int r = 0; r < rows; ++r)
        ck(cudaMemcpyAsync(logits + (size_t) r * n_vocab_ + n_vocab_real_, neg_inf_, pad_n * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(st_)),
           "logits");
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, weights from the
// bare gates divided by their sum (+ 1e-20, expert_weights_norm_eps) and times expert_weights_scale.
void Inkling::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    sum += 1e-20f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= sum;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

// The router's F32 product: a decode step's few tokens with inkling_gemv(), a prompt batch as the engine does it.
void Inkling::router_scores(const Layer& L, int n) {
    if (n <= kFusedRows) inkling_gemv(fp(L.gate_inp), xn_, rl_, n_embd_, n_expert_, n, st_);
    else Engine::router_scores(L, n);
}

bool Inkling::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, B.probs_b_dev.p ? fp(B.probs_b_dev) : nullptr, nullptr, true,
               w_scale_, ids, w, st_);
    return true;
}

// No shared expert: the routed experts' sum starts from zero.
void Inkling::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

// ---- a sequence's state and rows -------------------------------------------------------------------------------

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).
template <class F>
void Inkling::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

size_t Inkling::plane_row(int i) const {
    const size_t width = (size_t) n_kv_ * head_dim_;
    if (!q8()) return width * sizeof(uint16_t);
    return i % 2 == 0 ? width : width / 32 * sizeof(uint16_t);
}

// A sequence's rows are the full visits'; a window visit's are part of its state (save_layer()).  A visit's part of
// a row is its planes one after the other (fp16 keys, values; or q8_0 key codes, key scales, value codes, value
// scales, as Flint's), each plane's rows together.
size_t Inkling::card_row_bytes() const {
    size_t one = 0;
    for (int i = 0; i < planes(); ++i) one += plane_row(i);
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.plane[0] && !B.swa) ++layers;
    return layers * one;
}

void Inkling::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    for (const Block& B : blocks_) {
        if (B.swa || n == 0 || !B.plane[0]) continue;
        for (int i = 0; i < planes(); ++i) {
            const size_t row = plane_row(i);
            ck(cudaMemcpy(dst, B.plane[i] + (size_t) first * row, (size_t) n * row, cudaMemcpyDeviceToHost),
               "save rows");
            dst += (size_t) n * row;
        }
    }
}

void Inkling::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    for (const Block& B : blocks_) {
        if (B.swa || n == 0 || !B.plane[0]) continue;
        for (int i = 0; i < planes(); ++i) {
            const size_t row = plane_row(i);
            h2d(B.plane[i] + (size_t) first * row, src + (size_t) (first - from) * row, (size_t) n * row);
            src += (size_t) count * row;
        }
    }
}

// What a window visit has of the sequence: the rows the next token attends, its planes one after the other.
size_t Inkling::extra_bytes() const {
    size_t one = 0;
    for (int i = 0; i < planes(); ++i) one += plane_row(i);
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.swa) ++layers;
    return layers * (size_t) keep_ * one;
}

// A visit's convolution state at its place in state_, and a window visit's rows behind the window visits before it.
void Inkling::save_layer(int v, int64_t pos, uint8_t* dst) {
    const Block& B = blocks_[(size_t) v];
    if (B.conv)
        ck(cudaMemcpy(dst + (size_t) (B.conv - state_) * sizeof(float), B.conv, conv_floats_ * sizeof(float),
                      cudaMemcpyDeviceToHost),
           "save state");
    if (!B.swa) return;
    size_t whole = 0;
    for (int i = 0; i < planes(); ++i) whole += (size_t) keep_ * plane_row(i);
    dst += state_n_ * sizeof(float);
    for (int j = 0; j < v; ++j)
        if (blocks_[(size_t) j].swa) dst += whole;
    for (int i = 0; i < planes(); ++i) {
        const size_t one_row = plane_row(i);
        std::memset(dst, 0, (size_t) keep_ * one_row);   // the places of positions before the sequence's start
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy(dst + slot * one_row, B.plane[i] + at * one_row, m * one_row, cudaMemcpyDeviceToHost),
               "save state");
        });
        dst += (size_t) keep_ * one_row;
    }
}

void Inkling::load_extra(const uint8_t* src, int64_t pos) {
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        for (int i = 0; i < planes(); ++i) {
            const size_t one_row = plane_row(i);
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                h2d(B.plane[i] + at * one_row, src + slot * one_row, m * one_row);
            });
            src += (size_t) keep_ * one_row;
        }
    }
}

}  // namespace strata::poly
