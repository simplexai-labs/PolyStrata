// src/poly/qwen35.cpp - the layers of Qwen3.5 and Qwen3.6 with routed experts.  The reference for the math is
// llama.cpp's qwen35moe graph.
#include "qwen35.hpp"

#include "q35_scratch_arena.hpp"

#include "fattn.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#ifdef STRATA_POLY_GDN_MMA
#include "gdn_mma.hpp"
#endif
#if defined(STRATA_USE_HIP)
#include "hip_value_split_policy.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace strata::poly {
namespace {

constexpr int kConv = 4;   // what conv4_silu is written for; checked at load
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
// A q8 prompt batch's product reads a block of keys at a time; before each block its K and V rows are dequantized
// into an fp16 workspace of this many rows.  Bounded, so it does not grow with the context: the natural block is
// about 16/3 * head_dim rows, well under this, so real prompts take the same products the fp16 path takes.
constexpr size_t kQ8WorkRows = 2048;

// n rows of `width` bytes, dpitch apart at dst and spitch apart at src (the Metal shim has no cudaMemcpy2D, and
// there the rows of a cache are always back to back)
void copy_rows(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t n, cudaMemcpyKind kind,
               const char* what) {
#if !defined(STRATA_GPU_METAL)
    if (dpitch != width || spitch != width) return ck(cudaMemcpy2D(dst, dpitch, src, spitch, width, n, kind), what);
#endif
    ck(cudaMemcpy(dst, src, n * width, kind), what);
}

}  // namespace

void Qwen35::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_gdn_ = n_attn_ = 0;
}

Qwen35::~Qwen35() {
    if (kv_stream_) cudaStreamSynchronize(cs(kv_stream_));
    for (int i = 0; i < 2; ++i) {
        if (kv_ev_[i]) cudaEventDestroy((cudaEvent_t) kv_ev_[i]);
        if (kv_use_ev_[i]) cudaEventDestroy((cudaEvent_t) kv_use_ev_[i]);
    }
    if (kv_store_ev_) cudaEventDestroy((cudaEvent_t) kv_store_ev_);
    if (kv_stream_) cudaStreamDestroy(cs(kv_stream_));
}

void Qwen35::check_draft_block() {
    if (blocks_[(size_t) n_layer_].gdn)
        throw std::runtime_error("the block that guesses ahead is not one this engine runs; --no-draft loads "
                                 "the model without it");
}

void Qwen35::load_geometry() {
    gdn_mma_.capture_request();
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    if (codes_read() && (n_kv_ * head_dim_) % 32 != 0)
        throw std::runtime_error("--kv " + std::string(opt_.kv_format == KvFmt::kQ4_0 ? "q4_0" : "q8_0") +
                                 " needs n_kv*head_dim to be a multiple of 32 (ggml QK8_0 / QK4_0); this model "
                                 "has " + std::to_string(n_kv_ * head_dim_));
    const MetaValue* rot = find("rope.dimension_count");
    n_rot_ = rot ? (int) rot->u : head_dim_;
    s_dim_ = shape_i("ssm.state_size");
    k_heads_ = shape_i("ssm.group_count");
    v_heads_ = shape_i("ssm.time_step_rank");
    d_inner_ = shape_i("ssm.inner_size");
    conv_dim_ = 2 * k_heads_ * s_dim_ + d_inner_;

    need(n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention.head_count");
    need(head_dim_ > 0 && shape_i("attention.value_length") == head_dim_,
         "the attention's keys and values differ in length");
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && n_rot_ <= head_dim_, "bad rope.dimension_count");
    need(shape_i("ssm.conv_kernel") == kConv, "ssm.conv_kernel is not 4");
    need(s_dim_ > 0 && s_dim_ % 32 == 0 && s_dim_ <= 256, "ssm.state_size is not a multiple of 32 up to 256");
    need(k_heads_ > 0 && v_heads_ % k_heads_ == 0 && d_inner_ == v_heads_ * s_dim_, "bad ssm head counts");

    // The rotary embedding has an axis for time, two for an image and a spare one, and its sections say which
    // pairs turn by which.  A token of text has its position on the first three and none on the spare one, so
    // the kernel turns every pair by the position: that is wrong for a pair the sections leave to the spare axis.
    // Clover reads no pictures and its file has no sections: every pair turns by the position, as a text token's do.
    clover_ = gg_->get("general.architecture")->s == "clover_moe";
    if (clover_) {
        // the token its block of guesses is filled with, and the share of the block's routing its experts are
        // chosen from
        const MetaValue* mask = gg_->get("tokenizer.ggml.mask_token_id");
        mask_ = mask && mask->is_num() && mask->u < (uint64_t) n_vocab_ ? (int) mask->u : -1;
        const MetaValue* on = find("moe_block_coreset_enabled");
        const MetaValue* share = find("moe_block_coreset_top_p");
        coreset_ = on && on->u && share && share->is_num() ? (float) share->num() : 0.f;
    } else {
        const MetaValue* sec = find("rope.dimension_sections");
        need(sec && sec->type == MetaType::ARRAY && sec->items.size() >= 3, "no rope.dimension_sections");
        int sections[4] = {}, covered = 0;
        for (size_t i = 0; i < 4 && i < sec->items.size(); ++i) covered += sections[i] = (int) sec->items[i].u;
        need(covered > 0, "empty rope.dimension_sections");
        for (int pair = 0; pair < n_rot_ / 2; ++pair) {
            const int s = pair % covered;
            need(s < 3 * sections[s % 3], "the rotary embedding leaves pairs to an axis text has no position on");
        }
        sections_ = covered;
    }
    theta_scale_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) n_rot_);

    // this architecture puts no limit on a SwiGLU's inputs
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Qwen35::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("post_attention_norm.weight", n_embd_);
    B.gdn = tensor(il, "attn_qkv.weight") != nullptr;
    // the attention's launches read the position on the card when they are recorded; a streamed layer's are not
    // recorded (its copies from host RAM are issued from the host, in prefetch_layer() and attn_stream())
    L.fixed = B.gdn || kv_ring() == 0;
    if (B.gdn) {
        B.wqkv = mat("attn_qkv.weight", n_embd_, conv_dim_);
        B.wz = mat("attn_gate.weight", n_embd_, d_inner_);
        B.conv = f32("ssm_conv1d.weight", kConv, conv_dim_);
        B.dt = f32("ssm_dt.bias", v_heads_);
        // a decay rate: llama.cpp's converter stores -exp(A_log), Clover's A_log itself
        B.a = clover_ ? upload(tensor(il, "ssm_a"), blk(il, "ssm_a"), v_heads_, 1, kF32,
                               [](float a) { return -std::exp(a); })
                      : f32("ssm_a", v_heads_);
        B.alpha = mat("ssm_alpha.weight", n_embd_, v_heads_);
        B.beta = mat("ssm_beta.weight", n_embd_, v_heads_);
        B.o_norm = f32("ssm_norm.weight", s_dim_);
        B.wo = mat("ssm_out.weight", d_inner_, n_embd_);
        if (mask_ >= 0) {
            B.conv_right = f32("ssm_conv1d.weight_right", kConv - 1, conv_dim_);
            B.bd_scale = f32("ssm_bd_logit_scale.weight", v_heads_);
            B.bd_hist_b = f32("ssm_bd_history.bias", v_heads_);
            B.bd_hist_c = f32("ssm_bd_history.weight", v_heads_);
        }
        ++n_gdn_;
    } else {
        // the query projection gives every head its query and then the gate of its output
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) 2 * n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.q_norm = f32("attn_q_norm.weight", head_dim_);
        B.k_norm = f32("attn_k_norm.weight", head_dim_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        ++n_attn_;
    }
    load_moe(il);
    B.sh_gate_inp = f32("ffn_gate_inp_shexp.weight", n_embd_);
}

size_t Qwen35::recurrent_floats() const {
    return size_t(n_gdn_) * (size_t(3) * conv_dim_ + size_t(v_heads_) * s_dim_ * s_dim_);
}

int Qwen35::widest() const { return std::max(d_inner_, n_head_ * head_dim_); }

// The attention layers' caches, and each recurrent layer's convolution and delta-rule state in state_.
void Qwen35::alloc_caches() {
    const int ring = kv_ring();
    if (ring > 0 && !axes_)
        std::fprintf(stderr,
                     "strata-poly: KV resident: %d cells in VRAM, %d in pinned RAM (ping-pong streaming + GDN prefetch)\n",
                     ring, opt_.max_context);
    if (opt_.kv_format == KvFmt::kQ8_0 && !axes_) {
        const size_t row_bytes = (size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", row_bytes);
    }
    if (opt_.kv_format == KvFmt::kQ4_0 && !axes_) {
        const size_t q4_row_bytes = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(block_q4_0);
        std::fprintf(stderr, "strata-poly: q4_0 KV cache: %zu bytes a row (%zu K + %zu V)\n", 2 * q4_row_bytes,
                     q4_row_bytes, q4_row_bytes);
    }
    const size_t past_n = size_t(3) * conv_dim_, s_n = size_t(v_heads_) * s_dim_ * s_dim_;
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    const size_t width = (size_t) n_kv_ * head_dim_;   // a key's or a value's codes a row
    const size_t codes_row = width;                                                  // int8
    const size_t scales_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);  // fp16
    const size_t q4_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(block_q4_0);    // Q4_0
    int8_t* codes = nullptr;
    uint16_t* scales = nullptr;
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    // q8_0 on CUDA: a row of every attention layer's keys and values side by side, in one array of codes and one of
    // scales a sequence.  The cache grows kKvGrow rows at a time in whole 2 MB pieces of each array, and 44 arrays
    // of Qwen3.6's took 132 MB a sequence for its first rows where these two take 48 MB.  A streamed cache (ring)
    // keeps each layer's rows apart: its ring rows are not the sequence's rows, so there is no shared array for it.
    if (opt_.kv_format == KvFmt::kQ8_0 && ring == 0) {
        int attn = 0;
        for (int il = 0; il < (int) blocks_.size(); ++il) attn += runs(il) && !blocks_[(size_t) il].gdn;
        kv_rs_ = size_t(attn) * 2 * width;
        if (kv_rs_) {
            codes = (int8_t*) kv_alloc(kv_rs_, 1);
            scales = (uint16_t*) kv_alloc(kv_rs_ / 32 * sizeof(uint16_t), 1);
        }
    }
#endif
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.gdn) {
            B.past = st;
            st += past_n;
            B.s = st;
            st += s_n;
        } else if (codes) {
            B.k_codes = codes;
            B.v_codes = codes + width;
            B.k_scales = scales;
            B.v_scales = scales + width / 32;
            codes += 2 * width;
            scales += 2 * width / 32;
        } else if (opt_.kv_format == KvFmt::kQ4_0) {
            // the Q4_0 rows (the whole cache unless streamed) and the Q8 codes the attention reads, dequantized from them
            if (ring == 0) {
                B.k_q4 = (block_q4_0*) kv_alloc(q4_row, 1);
                B.v_q4 = (block_q4_0*) kv_alloc(q4_row, 1);
            }
            B.k_codes = (int8_t*) kv_alloc(codes_row, 1);
            B.k_scales = (uint16_t*) kv_alloc(scales_row, 1);
            B.v_codes = (int8_t*) kv_alloc(codes_row, 1);
            B.v_scales = (uint16_t*) kv_alloc(scales_row, 1);
            if (ring > 0 && !axes_) {
                const size_t full = (size_t) opt_.max_context;
                B.h_k_q4 = (block_q4_0*) hmalloc(full * q4_row, "pinned RAM attention k_q4");
                B.h_v_q4 = (block_q4_0*) hmalloc(full * q4_row, "pinned RAM attention v_q4");
            }
        } else if (opt_.kv_format == KvFmt::kQ8_0) {
            B.k_codes = (int8_t*) kv_alloc(codes_row, 1);
            B.k_scales = (uint16_t*) kv_alloc(scales_row, 1);
            B.v_codes = (int8_t*) kv_alloc(codes_row, 1);
            B.v_scales = (uint16_t*) kv_alloc(scales_row, 1);
            if (ring > 0 && !axes_) {
                const size_t full = (size_t) opt_.max_context;
                B.h_k_codes = (int8_t*) hmalloc(full * codes_row, "pinned RAM attention k_codes");
                B.h_k_scales = (uint16_t*) hmalloc(full * scales_row, "pinned RAM attention k_scales");
                B.h_v_codes = (int8_t*) hmalloc(full * codes_row, "pinned RAM attention v_codes");
                B.h_v_scales = (uint16_t*) hmalloc(full * scales_row, "pinned RAM attention v_scales");
            }
        } else {
            B.keys = (uint16_t*) kv_alloc(row, 1);
            B.vals = (uint16_t*) kv_alloc(row, 1);
            if (ring > 0 && !axes_) {
                const size_t full = (size_t) opt_.max_context;
                B.h_keys = (uint16_t*) hmalloc(full * row, "pinned RAM attention keys");
                B.h_vals = (uint16_t*) hmalloc(full * row, "pinned RAM attention vals");
            }
        }
    }
    // a block of guesses attends in one launch (attn_block_pass())
    const size_t fewest = mask_ >= 0 ? std::min(opt_.window, rows_) : 1;
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), fewest, (size_t) rows_);
    if (axes_) return;   // bind(): a call's scratch is every sequence's
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    // The attention's scores, sized for the context and not for a call's tokens: apart from the scratch, so that its
    // two layouts (scratch_layouts()) share them.  In each they were 64 MB of Qwen3.6's at 131,072 tokens.
    score_ = (float*) dmalloc((size_t) chunk_ * n_head_ * opt_.max_context * sizeof(float), "the attention's scores");
#endif
    if (ring > 0) {
        // --kv-resident: the staging buffers for the cold rows (two of each, ping-pong), the q4_0 write stage, a
        // stream for the copies in, and the events that order them against the compute and the copies out
        const size_t srows = (size_t) kv_stage_rows();
        q4_stage_srows_ = std::max((int) srows, opt_.prompt_batch > 0 ? opt_.prompt_batch : 2048);
        if (opt_.kv_format == KvFmt::kQ4_0) {
            q4_stage_k_ = (block_q4_0*) dmalloc((size_t) q4_stage_srows_ * q4_row, "KV Q4 write stage K");
            q4_stage_v_ = (block_q4_0*) dmalloc((size_t) q4_stage_srows_ * q4_row, "KV Q4 write stage V");
        }
        for (int b = 0; b < 2; ++b) {
            KvStage& S = kv_stage_[b];
            if (opt_.kv_format == KvFmt::kQ4_0) {
                S.k_q4 = (block_q4_0*) dmalloc(srows * q4_row, "KV ping-pong k_q4");
                S.v_q4 = (block_q4_0*) dmalloc(srows * q4_row, "KV ping-pong v_q4");
                S.k_codes = (int8_t*) dmalloc(srows * codes_row, "KV ping-pong k_codes");
                S.k_scales = (uint16_t*) dmalloc(srows * scales_row, "KV ping-pong k_scales");
                S.v_codes = (int8_t*) dmalloc(srows * codes_row, "KV ping-pong v_codes");
                S.v_scales = (uint16_t*) dmalloc(srows * scales_row, "KV ping-pong v_scales");
            } else if (opt_.kv_format == KvFmt::kQ8_0) {
                S.k_codes = (int8_t*) dmalloc(srows * codes_row, "KV ping-pong k_codes");
                S.k_scales = (uint16_t*) dmalloc(srows * scales_row, "KV ping-pong k_scales");
                S.v_codes = (int8_t*) dmalloc(srows * codes_row, "KV ping-pong v_codes");
                S.v_scales = (uint16_t*) dmalloc(srows * scales_row, "KV ping-pong v_scales");
            } else {
                S.keys = (uint16_t*) dmalloc(srows * row, "KV ping-pong keys");
                S.vals = (uint16_t*) dmalloc(srows * row, "KV ping-pong vals");
            }
        }
        cudaStream_t kst = nullptr;
        ck(cudaStreamCreateWithFlags(&kst, cudaStreamNonBlocking), "kv_stream");
        kv_stream_ = kst;
        for (int b = 0; b < 2; ++b) {
            cudaEvent_t ev = nullptr, uev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "kv_ev");
            ck(cudaEventCreateWithFlags(&uev, cudaEventDisableTiming), "kv_use_ev");
            kv_ev_[b] = ev;
            kv_use_ev_[b] = uev;
        }
        cudaEvent_t sev = nullptr;
        ck(cudaEventCreateWithFlags(&sev, cudaEventDisableTiming), "kv_store_ev");
        kv_store_ev_ = sev;
        // each layer names the next attention layer, whose first cold chunk a GDN layer before it may prefetch
        const int limit = draft_ ? n_layer_ + 1 : n_layer_;
        for (int il = 0; il < (int) blocks_.size(); ++il) {
            if (!runs(il)) continue;
            int next = -1;
            for (int j = il + 1; j < limit; ++j) {
                if (runs(j) && !blocks_[(size_t) j].gdn) {
                    next = j;
                    break;
                }
            }
            blocks_[(size_t) il].next_attn = next;
        }
    }
    axes_ = (int32_t*) dmalloc((size_t) rows_ * 3 * sizeof(int32_t), "a call's rotary positions");
    h_axes_.resize((size_t) rows_ * 3);
}

// Per-token buffers hold buf_rows_ tokens, row after row.
void Qwen35::assign_scratch() {
    const size_t n = (size_t) buf_rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    // Recurrent and full-attention layers consume these temporary families in
    // separate phases on the same compute stream. Keep members within a family
    // disjoint, and keep score_/q8w_/persistent state outside this arena.
    // Metal retains its existing balanced allocation policy.
#if defined(STRATA_GPU_METAL)
    constexpr bool scratch_alias = false;
#else
    constexpr bool scratch_alias = true;
#endif
    if (scratch_alias) {
        const std::array<size_t, 6> recurrent = {
            n * conv_dim_, n * conv_dim_, n * d_inner_, n * v_heads_, n * v_heads_, n * d_inner_};
        const std::array<size_t, 7> attention = {
            2 * heads, heads, kv, kv, kv, heads,
            buf_rows_ >= kStageRows ? heads + heads / 2 + 1 + 3 * n * n_head_ : 0};
        const auto layout = poly_scratch::plan(recurrent, attention);
        float* arena = fbuf(layout.total);
        auto at = [&](size_t offset) { return arena != nullptr ? arena + offset : nullptr; };
        proj_ = at(layout.a[0]);
        qkv_ = at(layout.a[1]);
        z_ = at(layout.a[2]);
        alpha_ = at(layout.a[3]);
        beta_ = at(layout.a[4]);
        gated_ = at(layout.a[5]);
        qg_ = at(layout.b[0]);
        q_ = at(layout.b[1]);
        kraw_ = at(layout.b[2]);
        k_ = at(layout.b[3]);
        v_ = at(layout.b[4]);
        att_ = at(layout.b[5]);
        if (buf_rows_ >= kStageRows) awork_ = at(layout.b[6]);
    } else {
        proj_ = fbuf(n * conv_dim_);
        qkv_ = fbuf(n * conv_dim_);
        z_ = fbuf(n * d_inner_);
        alpha_ = fbuf(n * v_heads_);
        beta_ = fbuf(n * v_heads_);
        gated_ = fbuf(n * d_inner_);

        qg_ = fbuf(2 * heads);
        q_ = fbuf(heads);
        kraw_ = fbuf(kv);
        k_ = fbuf(kv);
        v_ = fbuf(kv);
        att_ = fbuf(heads);
    }
#if defined(STRATA_GPU_METAL)
    if (!opt_.metal_balanced)
#endif
    {
#if defined(STRATA_GPU_METAL) || defined(STRATA_USE_HIP)
        score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);   // else alloc_caches()'s
#endif
        if (!scratch_alias && buf_rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
        if (buf_rows_ >= kStageRows && codes_read()) {
            // the K and V blocks a q8 prompt batch dequantizes, side by side; work_halves_ halves each
            work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
            q8w_ = fbuf((2 * work_halves_ + 1) / 2);
        }
        if (kv_ring() > 0) {
            // a streamed layer's attention pieces: each piece's output and its (max, sum) pairs (attn_stream())
            part_out_ = fbuf(heads);
            part_ml_ = fbuf(4 * n * n_head_);
        }
    }
    if (gdn_mma_.initialize([&] {
#ifdef STRATA_POLY_GDN_MMA
            return gdn_mma_available(s_dim_, v_heads_, k_heads_);
#else
            return false;
#endif
        })) {
        std::fprintf(stderr, "strata-poly: STRATA_GDN_MMA: %s\n",
                     gdn_mma_.enabled() ? "prefill through chunked GDN MMA; decode and snapshots unchanged"
#ifdef STRATA_POLY_GDN_MMA
                                        : "unchanged (unsupported shape, architecture or shared memory)"
#else
                                        : "unchanged (needs a CUDA build)"
#endif
        );
    }
    // A prompt batch through llama.cpp's flash-attention kernel, in score_ (fattn.hpp), wherever it fits (Ampere or
    // later, heads of 256): on unless STRATA_FATTN=0.  At 100K of context on an RTX PRO 6000 held to 12 GB, Qwen3.6
    // read 8,210 tokens/s instead of 5,837; the answers part from the old path's at near-ties.  A card it does not fit
    // keeps the old path without a word, unless STRATA_FATTN=1 asked for it.
    const char* fa = std::getenv("STRATA_FATTN");
    if (buf_rows_ >= kStageRows && !(fa && std::atoi(fa) == 0) && !fattn_) {   // assign_scratch() runs twice: say it once
        fattn_ = fattn_fits(head_dim_, n_head_, n_kv_, buf_rows_, (size_t) chunk_ * n_head_ * opt_.max_context);
        if (fattn_ || (fa && std::atoi(fa) == 1))
            std::fprintf(stderr, "strata-poly: STRATA_FATTN: prompt attention %s\n",
                         fattn_ ? "through llama.cpp's flash-attention kernel (STRATA_FATTN=0 turns it off)"
                                : "unchanged (no kernel for it on this card, or --prompt-batch too large for the scratch)");
    }
    // A few tokens' attention through fdec_q8() / fdec_f16(), in score_ (fattn.hpp), wherever it fits: on unless
    // STRATA_FDEC=0, from 4,096 positions of context, below which its kernel was the slower one (docs/DETAILS.md);
    // STRATA_FDEC=1 from the first position, as the opt-in was.  At 100K Qwen3.6 wrote 230 tokens/s instead of 106.
    const char* fd = std::getenv("STRATA_FDEC");
    if (!(fd && std::atoi(fd) == 0) && !fdec_) {
        fdec_ = fdec_fits(head_dim_, n_head_, n_kv_, (int) opt_.max_context, (size_t) chunk_ * n_head_ * opt_.max_context);
        fdec_from_ = fd && std::atoi(fd) == 1 ? 0 : 4096;
        if (fdec_ || (fd && std::atoi(fd) == 1))
            std::fprintf(stderr, "strata-poly: STRATA_FDEC: decode attention %s\n",
                         !fdec_ ? "unchanged (no kernel for it on this card)"
                         : fdec_from_ ? "through the tensor-core decode kernel from 4,096 tokens of context (STRATA_FDEC=0 "
                                        "turns it off)"
                                      : "through the tensor-core decode kernel");
    }
#if defined(STRATA_USE_HIP)
    // STRATA_HIP_VALUE_SPLIT=1: split the long-context weighted-value reduction in HIP's attend().
    if (hip_value_split::requested(std::getenv("STRATA_HIP_VALUE_SPLIT")) && !hip_value_split_logged_) {
        const bool ready = hip_value_split::fits(
            head_dim_, n_head_, n_kv_, opt_.max_context, chunk_, 0, 0, false,
            (size_t) chunk_ * n_head_ * opt_.max_context);
        std::fprintf(stderr, "strata-poly: STRATA_HIP_VALUE_SPLIT: %s\n",
                     ready ? "split value reduction on supported decode calls from 12,288 keys"
                           : "unchanged (unsupported head shape, context or scratch)");
        hip_value_split_logged_ = true;
    }
#endif
    sg_ = fbuf(n);
}

// A recurrent layer's convolution and delta-rule state lie one behind the other in state_ (alloc_caches()).
void Qwen35::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.past) return;
    const size_t n = size_t(3) * conv_dim_ + size_t(v_heads_) * s_dim_ * s_dim_;
    ck(cudaMemcpyAsync(dst + (size_t) (B.past - state_) * sizeof(float), B.past, n * sizeof(float),
                       cudaMemcpyDeviceToHost, cs(st_)),
       "save state");
}

size_t Qwen35::card_row_bytes() const {
    if (opt_.kv_format == KvFmt::kQ4_0) return (size_t) n_attn_ * 2 * (size_t) n_kv_ * (head_dim_ / 32) * sizeof(block_q4_0);
    if (opt_.kv_format == KvFmt::kQ8_0)
        return (size_t) n_attn_ * 2 *
               ((size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t));
    return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t);
}

void Qwen35::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    if (kv_stream_) ck(cudaStreamSynchronize(cs(kv_stream_)), "save rows kv_stream");
    const int ring = kv_ring();   // a streamed cache's rows are in its host copy
    if (opt_.kv_format == KvFmt::kQ4_0) {
        const size_t q4_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(block_q4_0);
        const size_t at = (size_t) first * q4_row, bytes = (size_t) n * q4_row;
        for (const Block& B : blocks_) {
            if ((!B.k_q4 && !B.h_k_q4) || n == 0) continue;
            if (ring > 0) {
                std::memcpy(dst, (const uint8_t*) B.h_k_q4 + at, bytes);
                dst += bytes;
                std::memcpy(dst, (const uint8_t*) B.h_v_q4 + at, bytes);
                dst += bytes;
                continue;
            }
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_q4 + at, bytes, cudaMemcpyDeviceToHost), "save rows k_q4");
            dst += bytes;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_q4 + at, bytes, cudaMemcpyDeviceToHost), "save rows v_q4");
            dst += bytes;
        }
        return;
    }
    if (opt_.kv_format == KvFmt::kQ8_0) {
        // q8_0: each attention block writes k_codes, k_scales, v_codes, v_scales, in that order, so that a load
        // can walk the same runs back.  Same layout as card_row_bytes() sums to (n * card_row_bytes() / n_attn_).
        const size_t crow = (size_t) n_kv_ * head_dim_,
                     srow = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        // a cache's rows lie pc and ps bytes apart (kv_rs_: every layer's side by side)
        const size_t pc = kv_rs_ ? kv_rs_ : crow, ps = kv_rs_ ? kv_rs_ / 32 * sizeof(uint16_t) : srow;
        const size_t at_c = (size_t) first * pc, at_s = (size_t) first * ps;
        // a streamed cache's rows come from its host copy, which holds the whole context at the same offsets
        const auto save = [&](const void* cache, bool host, size_t at, size_t pitch, size_t width) {
            if (host) {
                std::memcpy(dst, (const uint8_t*) cache + at, (size_t) n * width);
            } else {
                copy_rows(dst, width, (const uint8_t*) cache + at, pitch, width, (size_t) n, cudaMemcpyDeviceToHost,
                          "save rows");
            }
            dst += (size_t) n * width;
        };
        for (const Block& B : blocks_) {
            if (!B.k_codes || n == 0) continue;
            const bool h = ring > 0;
            save(h ? (const void*) B.h_k_codes : (const void*) B.k_codes, h, at_c, pc, crow);
            save(h ? (const void*) B.h_k_scales : (const void*) B.k_scales, h, at_s, ps, srow);
            save(h ? (const void*) B.h_v_codes : (const void*) B.v_codes, h, at_c, pc, crow);
            save(h ? (const void*) B.h_v_scales : (const void*) B.v_scales, h, at_s, ps, srow);
        }
        return;
    }
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        if (ring > 0) {
            std::memcpy(dst, (const uint8_t*) B.h_keys + at, bytes);
            dst += bytes;
            std::memcpy(dst, (const uint8_t*) B.h_vals + at, bytes);
            dst += bytes;
            continue;
        }
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

// --kv-resident: the ring's last kv_ring() rows, from the host copy of the whole context (after a load, or a state
// brought back).  Cells not yet written past pos are left alone.
void Qwen35::sync_hot_ring(int64_t pos) {
    const int ring = kv_ring();
    if (ring <= 0) return;
    if (kv_stream_) ck(cudaStreamSynchronize(cs(kv_stream_)), "sync_hot_ring kv_stream");
    prefetched_il_ = -1;
    if (pos <= 0) return;
    cover(pos);
    const int64_t hot_lo = std::max<int64_t>(0, pos - ring);
    if (opt_.kv_format == KvFmt::kQ4_0) {
        const size_t q4_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(block_q4_0);
        const int kv_stride = n_kv_ * head_dim_;
        const int nb = kv_stride / 32;
        for (const Block& B : blocks_) {
            if (!B.h_k_q4) continue;
            for (int64_t r = hot_lo; r < pos; ) {
                const int p0 = (int) (r % ring);
                const int m = (int) std::min<int64_t>(pos - r, ring - p0);
                const int chunk = std::min(m, q4_stage_srows_);
                ck(cudaMemcpy(q4_stage_k_, (const uint8_t*) B.h_k_q4 + (size_t) r * q4_row,
                              (size_t) chunk * q4_row, cudaMemcpyHostToDevice), "sync_hot_ring k_q4");
                ck(cudaMemcpy(q4_stage_v_, (const uint8_t*) B.h_v_q4 + (size_t) r * q4_row,
                              (size_t) chunk * q4_row, cudaMemcpyHostToDevice), "sync_hot_ring v_q4");
                dequant_q4_to_q8(q4_stage_k_, B.k_codes + (size_t) p0 * kv_stride,
                                 B.k_scales + (size_t) p0 * nb, kv_stride, chunk, st_);
                dequant_q4_to_q8(q4_stage_v_, B.v_codes + (size_t) p0 * kv_stride,
                                 B.v_scales + (size_t) p0 * nb, kv_stride, chunk, st_);
                r += chunk;
            }
        }
        ck(cudaStreamSynchronize(cs(st_)), "sync_hot_ring unpack");
        return;
    }
    if (opt_.kv_format == KvFmt::kQ8_0) {
        const size_t crow = (size_t) n_kv_ * head_dim_,
                     srow = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        for (const Block& B : blocks_) {
            if (!B.k_codes || !B.h_k_codes) continue;
            for (int64_t r = hot_lo; r < pos; ) {
                const int p0 = (int) (r % ring);
                const int m = (int) std::min<int64_t>(pos - r, ring - p0);
                ck(cudaMemcpy((uint8_t*) B.k_codes + (size_t) p0 * crow, (const uint8_t*) B.h_k_codes + (size_t) r * crow,
                              (size_t) m * crow, cudaMemcpyHostToDevice), "sync_hot_ring");
                ck(cudaMemcpy((uint8_t*) B.k_scales + (size_t) p0 * srow, (const uint8_t*) B.h_k_scales + (size_t) r * srow,
                              (size_t) m * srow, cudaMemcpyHostToDevice), "sync_hot_ring");
                ck(cudaMemcpy((uint8_t*) B.v_codes + (size_t) p0 * crow, (const uint8_t*) B.h_v_codes + (size_t) r * crow,
                              (size_t) m * crow, cudaMemcpyHostToDevice), "sync_hot_ring");
                ck(cudaMemcpy((uint8_t*) B.v_scales + (size_t) p0 * srow, (const uint8_t*) B.h_v_scales + (size_t) r * srow,
                              (size_t) m * srow, cudaMemcpyHostToDevice), "sync_hot_ring");
                r += m;
            }
        }
        return;
    }
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    for (const Block& B : blocks_) {
        if (!B.keys || !B.h_keys) continue;
        for (int64_t r = hot_lo; r < pos; ) {
            const int p0 = (int) (r % ring);
            const int m = (int) std::min<int64_t>(pos - r, ring - p0);
            ck(cudaMemcpy((uint8_t*) B.keys + (size_t) p0 * row, (const uint8_t*) B.h_keys + (size_t) r * row,
                          (size_t) m * row, cudaMemcpyHostToDevice), "sync_hot_ring");
            ck(cudaMemcpy((uint8_t*) B.vals + (size_t) p0 * row, (const uint8_t*) B.h_vals + (size_t) r * row,
                          (size_t) m * row, cudaMemcpyHostToDevice), "sync_hot_ring");
            r += m;
        }
    }
}

// The engine's state for a position is brought back: the ring follows it.
void Qwen35::load_extra(const uint8_t*, int64_t pos) { sync_hot_ring(pos); }

void Qwen35::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    if (kv_stream_) ck(cudaStreamSynchronize(cs(kv_stream_)), "load rows kv_stream");
    prefetched_il_ = -1;
    const int ring = kv_ring();
    if (opt_.kv_format == KvFmt::kQ4_0) {
        const size_t q4_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(block_q4_0);
        const size_t at = (size_t) first * q4_row, bytes = (size_t) n * q4_row;
        const size_t sk = (size_t) (first - from) * q4_row, all = (size_t) count * q4_row;
        for (const Block& B : blocks_) {
            if ((!B.k_q4 && !B.h_k_q4) || n == 0) continue;
            if (ring > 0) {
                std::memcpy((uint8_t*) B.h_k_q4 + at, src + sk, bytes);
                src += all;
                std::memcpy((uint8_t*) B.h_v_q4 + at, src + sk, bytes);
                src += all;
                continue;
            }
            ck(cudaMemcpy((uint8_t*) B.k_q4 + at, src + sk, bytes, cudaMemcpyHostToDevice), "load rows k_q4");
            src += all;
            ck(cudaMemcpy((uint8_t*) B.v_q4 + at, src + sk, bytes, cudaMemcpyHostToDevice), "load rows v_q4");
            src += all;
        }
        if (ring > 0 && n > 0) sync_hot_ring(first + n);
        return;
    }
    if (opt_.kv_format == KvFmt::kQ8_0) {
        // q8_0: mirror save_card_rows run for run, advancing by each run's own saved size (count * row) so
        // multi-layer blobs line up; the rows wanted start `first - from` rows into the run.
        const size_t crow = (size_t) n_kv_ * head_dim_,
                     srow = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        const size_t pc = kv_rs_ ? kv_rs_ : crow, ps = kv_rs_ ? kv_rs_ / 32 * sizeof(uint16_t) : srow;
        const size_t at_c = (size_t) first * pc, at_s = (size_t) first * ps;
        // a streamed cache's rows go to its host copy, which holds the whole context at the same offsets
        const auto load = [&](void* cache, bool host, size_t at, size_t pitch, size_t width) {
            if (host) {
                std::memcpy((uint8_t*) cache + at, src + (size_t) (first - from) * width, (size_t) n * width);
            } else {
                copy_rows((uint8_t*) cache + at, pitch, src + (size_t) (first - from) * width, width, width, (size_t) n,
                          cudaMemcpyHostToDevice, "load rows");
            }
            src += (size_t) count * width;
        };
        for (const Block& B : blocks_) {
            if (!B.k_codes || n == 0) continue;
            const bool h = ring > 0;
            load(h ? (void*) B.h_k_codes : (void*) B.k_codes, h, at_c, pc, crow);
            load(h ? (void*) B.h_k_scales : (void*) B.k_scales, h, at_s, ps, srow);
            load(h ? (void*) B.h_v_codes : (void*) B.v_codes, h, at_c, pc, crow);
            load(h ? (void*) B.h_v_scales : (void*) B.v_scales, h, at_s, ps, srow);
        }
        if (ring > 0 && n > 0) sync_hot_ring(first + n);
        return;
    }
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        if (ring > 0) {
            std::memcpy((uint8_t*) B.h_keys + at, src + skip, bytes);
            src += all;
            std::memcpy((uint8_t*) B.h_vals + at, src + skip, bytes);
            src += all;
            continue;
        }
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
    if (ring > 0 && bytes > 0) sync_hot_ring(first + n);
}

// The gated delta net.  One projection gives the queries, keys and values, which a convolution over the last
// four tokens mixes; another gives the gate of the output.
void Qwen35::gdn(Block& B, int n) {
    lin_pair(B.wqkv, B.wz, xn_, proj_, z_, n, xq_);
    lin_pair(B.alpha, B.beta, xn_, alpha_, beta_, n, xq_);
#if !defined(STRATA_GPU_METAL)
    if (rows_of_) {   // step(): each sequence's rows with its own convolution and state, all sequences a launch
        // a token and its guess (step() with guesses) are a sequence's two rows, leaving the state after the first
        // at its snap for keep_step()
        SeqStates conv, delta;
        for (int i = 0, m; i < n && conv.n < SeqStates::kMax; i += m) {
            const int s = rows_of_[i].seq;
            m = i + 1 < n && rows_of_[i + 1].seq == s ? 2 : 1;
            float* snap = rows_of_[i].snap;
            for (SeqStates* q : {&conv, &delta}) {
                q->first[q->n] = i;
                q->rows[q->n] = m;
            }
            conv.state[conv.n] = seq_state(s, B.past);
            conv.snap[conv.n++] = snap ? snap + (B.past - state_) : nullptr;
            delta.state[delta.n] = seq_state(s, B.s);
            delta.snap[delta.n++] = snap ? snap + (B.s - state_) : nullptr;
        }
        if (n <= SeqStates::kMax) {
            conv4_silu_seqs(proj_, conv, fp(B.conv), qkv_, conv_dim_, st_);
            gdn_step_seqs(qkv_, alpha_, beta_, fp(B.dt), fp(B.a), delta, z_, fp(B.o_norm), gated_, s_dim_, v_heads_,
                          k_heads_, n, rms_eps_, st_);
            lin(B.wo, gated_, ao_, n);
            return;
        }
    }
#endif
    if (rows_of_) {   // step(): each sequence's rows with its own convolution and state
        for (int i = 0, m; i < n; i += m) {
            // a token and its guess (step() with guesses) go in one call, a window of two, leaving the state after
            // the first at its snap for keep_step()
            const int s = rows_of_[i].seq;
            m = i + 1 < n && rows_of_[i + 1].seq == s ? 2 : 1;
            float* snap = rows_of_[i].snap;
            const size_t c = (size_t) i * conv_dim_, h = (size_t) i * v_heads_, d = (size_t) i * d_inner_;
            conv4_silu(proj_ + c, seq_state(s, B.past), fp(B.conv), qkv_ + c, conv_dim_, m, st_,
                       snap ? snap + (B.past - state_) : nullptr, rec_n_, snap ? 1u : 0u);
            gdn_step(qkv_ + c, alpha_ + h, beta_ + h, fp(B.dt), fp(B.a), seq_state(s, B.s), z_ + d, fp(B.o_norm),
                     gated_ + d, s_dim_, v_heads_, k_heads_, m, rms_eps_, st_, snap ? snap + (B.s - state_) : nullptr,
                     rec_n_, snap ? 1u : 0u);
        }
    } else {
        conv4_silu(proj_, B.past, fp(B.conv), qkv_, conv_dim_, n, st_,
                   snap_bits_ ? snap_at_ + (B.past - state_) : nullptr, rec_n_, snap_bits_);
#ifdef STRATA_POLY_GDN_MMA
        if (!(gdn_mma_.enabled() && !recording_ && gdn_mma_step(qkv_, alpha_, beta_, fp(B.dt), fp(B.a), B.s, z_, fp(B.o_norm), gated_,
                                          s_dim_, v_heads_, k_heads_, n, rms_eps_, st_, snap_bits_ != 0)))
#endif
        gdn_step(qkv_, alpha_, beta_, fp(B.dt), fp(B.a), B.s, z_, fp(B.o_norm), gated_, s_dim_, v_heads_, k_heads_, n,
                 rms_eps_, st_, snap_bits_ ? snap_at_ + (B.s - state_) : nullptr, rec_n_, snap_bits_);
    }
    lin(B.wo, gated_, ao_, n);
}

// gdn() for a block of guesses (Engine::guess_block()): the block's tokens see each other and the state is only
// read.  The scratch of the attention's scores holds the block's weights: no attention runs meanwhile.
void Qwen35::gdn_block_pass(Block& B, int n) {
    lin(B.wqkv, xn_, proj_, n, xq_);
    lin(B.wz, xn_, z_, n, xq_);
    lin(B.alpha, xn_, alpha_, n, xq_);
    conv4_block_silu(proj_, B.past, fp(B.conv), fp(B.conv_right), qkv_, conv_dim_, n, st_);
    gdn_block(qkv_, alpha_, beta_, fp(B.dt), fp(B.a), B.s, z_, fp(B.o_norm), fp(B.bd_scale), fp(B.bd_hist_b),
              fp(B.bd_hist_c), gated_, score_, s_dim_, v_heads_, k_heads_, n, rms_eps_, st_);
    lin(B.wo, gated_, ao_, n);
}

// A row's position on the rotary embedding's three axes, as llama.cpp's mtmd counts them.  Text has the same on
// all three and counts up.  A picture's rows share the position the picture starts at on the first axis, and add
// their line to it on the second and their column on the third; the text after a picture goes on as many
// positions later as the picture has lines or columns, whichever is more.
void Qwen35::turn_of(int64_t row, int32_t* thw) const {
    int64_t skipped = 0;   // the rows before this one that took no position of their own
    for (const Image& im : images_) {
        if (row < im.at) break;
        if (row < im.at + im.n) {
            const int64_t p = im.at - skipped, j = row - im.at;
            thw[0] = (int32_t) p;
            thw[1] = (int32_t) (p + j / im.nx);
            thw[2] = (int32_t) (p + j % im.nx);
            return;
        }
        skipped += im.n - std::max(im.nx, im.n / im.nx);
    }
    thw[0] = thw[1] = thw[2] = (int32_t) (row - skipped);
}

// The rotary positions of n rows from row `pos`.  Rows of text alone count up from *first, and the result is
// null.  With a picture's rows among them the result is the positions of each row on the three axes, on the card.
const int32_t* Qwen35::turns(int64_t pos, int n, int64_t* first) {
    int32_t thw[3];
    turn_of(pos, thw);
    *first = thw[0];
    const bool pictures = std::any_of(images_.begin(), images_.end(), [&](const Image& im) {
        return im.at < pos + n && im.at + im.n > pos;
    });
    if (!pictures) return nullptr;
    if (axes_pos_ != pos || axes_n_ != n || axes_set_ != images_set_) {
        for (int t = 0; t < n; ++t) turn_of(pos + t, h_axes_.data() + (size_t) t * 3);
        // the launches that read the positions of the call before this one have to be through
        ck(cudaStreamSynchronize(cs(st_)), "rotary positions");
        ck(cudaMemcpy(axes_, h_axes_.data(), (size_t) n * 3 * sizeof(int32_t), cudaMemcpyHostToDevice),
           "rotary positions");
        axes_pos_ = pos;
        axes_n_ = n;
        axes_set_ = images_set_;
    }
    return axes_;
}

// The rows n tokens from position `pos` leave in an attention layer's caches: each head group's key, normed and
// turned by the position, and its value.  xn_ and xq_ hold the layer's input.
void Qwen35::attn_store(Block& B, int64_t pos, int n) {
    int64_t first;
    const int32_t* axes = turns(pos, n, &first);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    lin_pair(B.wk, B.wv, xn_, kraw_, v_, n, xq_);
    head_norm_rope(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, n_rot_, theta_scale_, first, n, rms_eps_, st_,
                   nullptr, axes, sections_, at ? at + 1 : nullptr);
    if (kv_ring() > 0) {
        store_ring(B, pos, n, k_, v_);
        return;
    }
    const int kv_stride = n_kv_ * head_dim_;
    if (opt_.kv_format == KvFmt::kQ4_0) {
        // the rows as Q4_0 (the cache's own), then the Q8 codes the attention reads, dequantized from them
        const int nb = kv_stride / 32;
        if (at) {
            store_q4_at(k_, B.k_q4, kv_stride, at, n, 0, st_);
            store_q4_at(v_, B.v_q4, kv_stride, at, n, 0, st_);
            dequant_q4_to_q8(B.k_q4, B.k_codes, B.k_scales, kv_stride, n, st_, at);
            dequant_q4_to_q8(B.v_q4, B.v_codes, B.v_scales, kv_stride, n, st_, at);
        } else {
            store_q4(k_, B.k_q4, kv_stride, pos, n, st_);
            store_q4(v_, B.v_q4, kv_stride, pos, n, st_);
            dequant_q4_to_q8(B.k_q4 + (size_t) pos * nb, B.k_codes + (size_t) pos * kv_stride,
                             B.k_scales + (size_t) pos * nb, kv_stride, n, st_);
            dequant_q4_to_q8(B.v_q4 + (size_t) pos * nb, B.v_codes + (size_t) pos * kv_stride,
                             B.v_scales + (size_t) pos * nb, kv_stride, n, st_);
        }
    } else if (opt_.kv_format == KvFmt::kQ8_0) {
        if (at) {
            store_q8_at(k_, B.k_codes, B.k_scales, n_kv_ * head_dim_, at, n, 0, st_, kv_rs_);
            store_q8_at(v_, B.v_codes, B.v_scales, n_kv_ * head_dim_, at, n, 0, st_, kv_rs_);
        } else {
            store_q8(k_, B.k_codes, B.k_scales, n_kv_ * head_dim_, pos, n, st_, kv_rs_);
            store_q8(v_, B.v_codes, B.v_scales, n_kv_ * head_dim_, pos, n, st_, kv_rs_);
        }
    } else if (at) {
        store_f16_at(k_, B.keys, n_kv_ * head_dim_, at, n, 0, st_);
        store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_);
    } else {
        store_f16(k_, B.keys, n_kv_ * head_dim_, pos, n, st_);
        store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_);
    }
}

// --kv-resident: n rows from position pos leave a streamed layer.  The ring takes them at slot p % ring, a run that
// wraps the ring going in two parts, and the host copy of the whole context takes the same rows at their own offsets,
// copied from the ring slots in the same parts.  ks and vs are the rows' keys (normed and turned) and values.
void Qwen35::store_ring(Block& B, int64_t pos, int n, const float* ks, const float* vs) {
    const int ring = kv_ring();
    const int kv_stride = n_kv_ * head_dim_;
    if (pos < prefetched_m_) prefetched_il_ = -1;   // a chunk staged before these rows is stale
    if (opt_.kv_format == KvFmt::kQ4_0) {
        const int nb = kv_stride / 32;
        const size_t q4_row = (size_t) nb * sizeof(block_q4_0);
        // a step of at most the ring and the write stage: Q4_0 into the stage, then dequantized into the ring's slots
        for (int s0 = 0, step = 0; s0 < n; s0 += step) {
            step = std::min({ring, n - s0, q4_stage_srows_});
            const int64_t r0 = pos + s0;
            const int p0 = (int) (r0 % ring);
            const float* kf = ks + (size_t) s0 * kv_stride;
            const float* vf = vs + (size_t) s0 * kv_stride;
            store_q4(kf, q4_stage_k_, kv_stride, 0, step, st_);
            store_q4(vf, q4_stage_v_, kv_stride, 0, step, st_);
            if (p0 + step <= ring) {
                dequant_q4_to_q8(q4_stage_k_, B.k_codes + (size_t) p0 * kv_stride, B.k_scales + (size_t) p0 * nb,
                                 kv_stride, step, st_);
                dequant_q4_to_q8(q4_stage_v_, B.v_codes + (size_t) p0 * kv_stride, B.v_scales + (size_t) p0 * nb,
                                 kv_stride, step, st_);
            } else {
                const int n1 = ring - p0;
                const int n2 = step - n1;
                dequant_q4_to_q8(q4_stage_k_, B.k_codes + (size_t) p0 * kv_stride, B.k_scales + (size_t) p0 * nb,
                                 kv_stride, n1, st_);
                dequant_q4_to_q8(q4_stage_v_, B.v_codes + (size_t) p0 * kv_stride, B.v_scales + (size_t) p0 * nb,
                                 kv_stride, n1, st_);
                dequant_q4_to_q8(q4_stage_k_ + (size_t) n1 * nb, B.k_codes, B.k_scales, kv_stride, n2, st_);
                dequant_q4_to_q8(q4_stage_v_ + (size_t) n1 * nb, B.v_codes, B.v_scales, kv_stride, n2, st_);
            }
            if (B.h_k_q4) {
                ck(cudaMemcpyAsync((uint8_t*) B.h_k_q4 + (size_t) r0 * q4_row, (const uint8_t*) q4_stage_k_,
                                   (size_t) step * q4_row, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h k_q4");
                ck(cudaMemcpyAsync((uint8_t*) B.h_v_q4 + (size_t) r0 * q4_row, (const uint8_t*) q4_stage_v_,
                                   (size_t) step * q4_row, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h v_q4");
            }
        }
    } else if (opt_.kv_format == KvFmt::kQ8_0) {
        const size_t crow = (size_t) kv_stride;
        const size_t srow = (size_t) (kv_stride / 32) * sizeof(uint16_t);
        for (int s0 = 0, step = 0; s0 < n; s0 += step) {
            step = std::min(ring, n - s0);
            const int64_t r0 = pos + s0;
            const int p0 = (int) (r0 % ring);
            const float* kf = ks + (size_t) s0 * kv_stride;
            const float* vf = vs + (size_t) s0 * kv_stride;
            if (p0 + step <= ring) {
                store_q8(kf, B.k_codes, B.k_scales, kv_stride, p0, step, st_);
                store_q8(vf, B.v_codes, B.v_scales, kv_stride, p0, step, st_);
                if (B.h_k_codes) {
                    ck(cudaMemcpyAsync((uint8_t*) B.h_k_codes + (size_t) r0 * crow,
                                       (const uint8_t*) B.k_codes + (size_t) p0 * crow,
                                       (size_t) step * crow, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h k_codes");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_k_scales + (size_t) r0 * srow,
                                       (const uint8_t*) B.k_scales + (size_t) p0 * srow,
                                       (size_t) step * srow, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h k_scales");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_v_codes + (size_t) r0 * crow,
                                       (const uint8_t*) B.v_codes + (size_t) p0 * crow,
                                       (size_t) step * crow, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h v_codes");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_v_scales + (size_t) r0 * srow,
                                       (const uint8_t*) B.v_scales + (size_t) p0 * srow,
                                       (size_t) step * srow, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h v_scales");
                }
            } else {
                const int n1 = ring - p0;
                const int n2 = step - n1;
                store_q8(kf, B.k_codes, B.k_scales, kv_stride, p0, n1, st_);
                store_q8(vf, B.v_codes, B.v_scales, kv_stride, p0, n1, st_);
                store_q8(kf + (size_t) n1 * kv_stride, B.k_codes, B.k_scales, kv_stride, 0, n2, st_);
                store_q8(vf + (size_t) n1 * kv_stride, B.v_codes, B.v_scales, kv_stride, 0, n2, st_);
                if (B.h_k_codes) {
                    ck(cudaMemcpyAsync((uint8_t*) B.h_k_codes + (size_t) r0 * crow,
                                       (const uint8_t*) B.k_codes + (size_t) p0 * crow,
                                       (size_t) n1 * crow, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h k_codes 1");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_k_codes + (size_t) (r0 + n1) * crow,
                                       (const uint8_t*) B.k_codes, (size_t) n2 * crow, cudaMemcpyDeviceToHost, cs(st_)),
                       "kv d2h k_codes 2");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_k_scales + (size_t) r0 * srow,
                                       (const uint8_t*) B.k_scales + (size_t) p0 * srow,
                                       (size_t) n1 * srow, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h k_scales 1");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_k_scales + (size_t) (r0 + n1) * srow,
                                       (const uint8_t*) B.k_scales, (size_t) n2 * srow, cudaMemcpyDeviceToHost, cs(st_)),
                       "kv d2h k_scales 2");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_v_codes + (size_t) r0 * crow,
                                       (const uint8_t*) B.v_codes + (size_t) p0 * crow,
                                       (size_t) n1 * crow, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h v_codes 1");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_v_codes + (size_t) (r0 + n1) * crow,
                                       (const uint8_t*) B.v_codes, (size_t) n2 * crow, cudaMemcpyDeviceToHost, cs(st_)),
                       "kv d2h v_codes 2");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_v_scales + (size_t) r0 * srow,
                                       (const uint8_t*) B.v_scales + (size_t) p0 * srow,
                                       (size_t) n1 * srow, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h v_scales 1");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_v_scales + (size_t) (r0 + n1) * srow,
                                       (const uint8_t*) B.v_scales, (size_t) n2 * srow, cudaMemcpyDeviceToHost, cs(st_)),
                       "kv d2h v_scales 2");
                }
            }
        }
    } else {
        const size_t row = (size_t) kv_stride * sizeof(uint16_t);
        for (int s0 = 0, step = 0; s0 < n; s0 += step) {
            step = std::min(ring, n - s0);
            const int64_t r0 = pos + s0;
            const int p0 = (int) (r0 % ring);
            const float* kf = ks + (size_t) s0 * kv_stride;
            const float* vf = vs + (size_t) s0 * kv_stride;
            if (p0 + step <= ring) {
                store_f16(kf, B.keys, kv_stride, p0, step, st_);
                store_f16(vf, B.vals, kv_stride, p0, step, st_);
                if (B.h_keys) {
                    ck(cudaMemcpyAsync((uint8_t*) B.h_keys + (size_t) r0 * row, (const uint8_t*) B.keys + (size_t) p0 * row,
                                       (size_t) step * row, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h keys");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_vals + (size_t) r0 * row, (const uint8_t*) B.vals + (size_t) p0 * row,
                                       (size_t) step * row, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h vals");
                }
            } else {
                const int n1 = ring - p0;
                const int n2 = step - n1;
                store_f16(kf, B.keys, kv_stride, p0, n1, st_);
                store_f16(vf, B.vals, kv_stride, p0, n1, st_);
                store_f16(kf + (size_t) n1 * kv_stride, B.keys, kv_stride, 0, n2, st_);
                store_f16(vf + (size_t) n1 * kv_stride, B.vals, kv_stride, 0, n2, st_);
                if (B.h_keys) {
                    ck(cudaMemcpyAsync((uint8_t*) B.h_keys + (size_t) r0 * row, (const uint8_t*) B.keys + (size_t) p0 * row,
                                       (size_t) n1 * row, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h keys 1");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_keys + (size_t) (r0 + n1) * row, (const uint8_t*) B.keys,
                                       (size_t) n2 * row, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h keys 2");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_vals + (size_t) r0 * row, (const uint8_t*) B.vals + (size_t) p0 * row,
                                       (size_t) n1 * row, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h vals 1");
                    ck(cudaMemcpyAsync((uint8_t*) B.h_vals + (size_t) (r0 + n1) * row, (const uint8_t*) B.vals,
                                       (size_t) n2 * row, cudaMemcpyDeviceToHost, cs(st_)), "kv d2h vals 2");
                }
            }
        }
    }
    if (kv_store_ev_) ck(cudaEventRecord((cudaEvent_t) kv_store_ev_, cs(st_)), "kv_store_ev");
}

// A streamed layer's attention: its rows are in attn_store()'s place already.  The gate and the output projection
// follow, as attn() has them.  whole: the block of guesses' (attn_block_pass()).
void Qwen35::attn_run(Block& B, int il, int64_t pos, int n, bool whole) {
    if (recording_) throw std::logic_error("a streamed attention layer is not recorded: its copies come from the host");
    int64_t first;
    const int32_t* axes = turns(pos, n, &first);
    lin(B.wq, xn_, qg_, n, xq_);
    head_norm_rope(qg_, 2 * head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, first, n, rms_eps_,
                   st_, nullptr, axes, sections_, nullptr);
    attn_store(B, pos, n);
    attn_stream(B, il, pos, n, whole, q_, att_);
    mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
    lin(B.wo, att_, ao_, n);
}

// The attention of n tokens from position pos, from q (n_head * head_dim floats a token) into out.  Whatever rows fit
// the ring are attended in their slots, as the caches lie.  The rows before the ring are cold: they come from the host
// copy through the two staging buffers, a chunk at a time, and each chunk's result is merged into out by its (max,
// sum) pairs, an exact online softmax.  The hot rows, the ring's, come last.  whole: each token sees the call's rows.
void Qwen35::attn_stream(Block& B, int il, int64_t pos, int n, bool whole, const float* q, float* out) {
    const int ring = kv_ring();
    const bool q8 = codes_read();
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const size_t score_floats = (size_t) chunk_ * n_head_ * opt_.max_context;
    const bool fd = fdec_ && pos >= fdec_from_;   // HEAD's rule for the decode kernels: from fdec_from_ positions
    if (pos + n <= ring) {
        const Kvq8 kq8{B.k_codes, B.k_scales}, vq8{B.v_codes, B.v_scales};
        if (whole) {
            attend(q, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, chunk_, out, head_dim_, n_head_, n_kv_,
                   scale, st_, 0, 0, true, nullptr, 0, 0, q8 ? kq8 : Kvq8{}, q8 ? vq8 : Kvq8{});
        } else if (n >= kStageRows) {
            const uint16_t* kc = q8 ? nullptr : B.keys;
            const uint16_t* vc = q8 ? nullptr : B.vals;
            uint16_t* kw = q8 ? (uint16_t*) q8w_ : nullptr;
            uint16_t* vw = q8 ? kw + work_halves_ : nullptr;
            if (!(fattn_ && (q8 ? fattn_q8(q, kq8, vq8, pos, n, out, head_dim_, n_head_, n_kv_, scale, score_, score_floats, st_)
                             : fattn_f16(q, B.keys, B.vals, pos, n, out, head_dim_, n_head_, n_kv_, scale, score_, score_floats,
                                         st_))))
                attend_batch(blas_, q, kc, vc, pos, n, score_, score_floats, awork_, out, head_dim_, n_head_, n_kv_, scale,
                             st_, 0, 0, false, q8 ? kq8 : Kvq8{}, q8 ? vq8 : Kvq8{}, kw, vw, q8 ? work_halves_ : 0);
        } else if (!(fd && (q8 ? fdec_q8(q, kq8, vq8, pos, n, nullptr, 0, out, head_dim_, n_head_, n_kv_, scale, score_,
                                         score_floats, st_)
                            : fdec_f16(q, B.keys, B.vals, pos, n, nullptr, 0, out, head_dim_, n_head_, n_kv_, scale, score_,
                                       score_floats, st_)))) {
            attend(q, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, chunk_, out, head_dim_, n_head_, n_kv_,
                   scale, st_, 0, 0, false, nullptr, 0, score_floats, q8 ? kq8 : Kvq8{}, q8 ? vq8 : Kvq8{});
        }
        return;
    }

    const int64_t total = pos + n;
    const int64_t cold = total - ring;
    const int stage_rows = kv_stage_rows();
    const size_t kv_stride = (size_t) n_kv_ * head_dim_;
    const size_t s_stride = kv_stride / 32;
    const size_t hq = (size_t) n_head_ * head_dim_;
    float* acc_ml = part_ml_;                               // the running (max, sum) pairs
    float* piece_ml = part_ml_ + 2 * (size_t) n * n_head_;  // the piece's

    // one piece of rows, [r0, r0 + m): the tokens that see all of it, and those inside it, which see up to their own
    const auto eval_piece = [&](const uint16_t* kc, const uint16_t* vc, Kvq8 kp, Kvq8 vp, int64_t r0, int m,
                                bool first_piece) {
        const int t0 = whole ? 0 : (int) std::clamp<int64_t>(r0 - pos, 0, n);
        const int t1 = whole ? 0 : (int) std::clamp<int64_t>(r0 + m - pos, 0, n);
        const int n_causal = t1 - t0;
        const int n_full = n - t1;
        if (n_causal > 0) {
            const float* q_sub = q + (size_t) t0 * hq;
            float* dst_out = first_piece ? (out + (size_t) t0 * hq) : part_out_;
            float* dst_ml = first_piece ? (acc_ml + 2 * (size_t) t0 * n_head_) : piece_ml;
            attn_piece(q_sub, kc, vc, kp, vp, (pos + t0) - r0, n_causal, false, dst_out, dst_ml, fd);
            if (!first_piece)
                merge_attn_piece(out + (size_t) t0 * hq, acc_ml + 2 * (size_t) t0 * n_head_, part_out_, piece_ml, n_causal,
                                 n_head_, head_dim_, st_);
        }
        if (n_full > 0) {
            const float* q_sub = q + (size_t) t1 * hq;
            float* dst_out = first_piece ? (out + (size_t) t1 * hq) : part_out_;
            float* dst_ml = first_piece ? (acc_ml + 2 * (size_t) t1 * n_head_) : piece_ml;
            attn_piece(q_sub, kc, vc, kp, vp, (int64_t) m - n_full, n_full, true, dst_out, dst_ml, fd);
            if (!first_piece)
                merge_attn_piece(out + (size_t) t1 * hq, acc_ml + 2 * (size_t) t1 * n_head_, part_out_, piece_ml, n_full,
                                 n_head_, head_dim_, st_);
        }
    };

    // The cold rows, in chunks of the staging buffers: the next chunk's copy goes on while this one is computed.
    const int m0 = (int) std::min<int64_t>(cold, stage_rows);
    const bool have_prefetch = prefetched_il_ == il && prefetched_m_ == m0 && m0 <= pos && !shared_ahead_;
    prefetched_il_ = -1;
    if (!have_prefetch) start_kv_stage_dma(B, 0, 0, m0, m0 > pos || shared_ahead_);
    for (int64_t c0 = 0, step = 0, step_idx = 0; c0 < cold; c0 += step, ++step_idx) {
        const int buf = (int) (step_idx & 1);
        step = std::min<int64_t>(cold - c0, stage_rows);
        if (c0 + step < cold) {
            const int64_t next_c0 = c0 + step;
            const int next_m = (int) std::min<int64_t>(cold - next_c0, stage_rows);
            start_kv_stage_dma(B, buf ^ 1, next_c0, next_m, next_c0 + next_m > pos || shared_ahead_);
        }
        ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) kv_ev_[buf], 0), "wait kv stage");
        const KvStage& S = kv_stage_[buf];
        if (opt_.kv_format == KvFmt::kQ4_0) {
            dequant_q4_to_q8(S.k_q4, S.k_codes, S.k_scales, (int) kv_stride, (int) step, st_);
            dequant_q4_to_q8(S.v_q4, S.v_codes, S.v_scales, (int) kv_stride, (int) step, st_);
        }
        eval_piece(q8 ? nullptr : S.keys, q8 ? nullptr : S.vals, q8 ? Kvq8{S.k_codes, S.k_scales} : Kvq8{},
                   q8 ? Kvq8{S.v_codes, S.v_scales} : Kvq8{}, c0, (int) step, c0 == 0);
        ck(cudaEventRecord((cudaEvent_t) kv_use_ev_[buf], cs(st_)), "kv stage used");
    }

    // The next attention layer's first cold chunk, copied while the MoE and the GDN layers between run (prefetch_layer()
    // does it for the layers that come before).
    if (B.next_attn > il && !shared_ahead_ && !(windowed_ && !all_cached() && halves_)) {
        if (m0 <= pos) {
            start_kv_stage_dma(blocks_[(size_t) B.next_attn], 0, 0, m0, false);
            prefetched_il_ = B.next_attn;
            prefetched_m_ = m0;
        }
    }

    // The hot rows [cold, total): the ring's last `ring` rows, in one or two slices
    const int p0 = (int) (cold % ring);
    const int m1 = ring - p0;
    const int m2 = p0;
    eval_piece(q8 ? nullptr : B.keys + (size_t) p0 * kv_stride, q8 ? nullptr : B.vals + (size_t) p0 * kv_stride,
               q8 ? Kvq8{B.k_codes + (size_t) p0 * kv_stride, B.k_scales + (size_t) p0 * s_stride} : Kvq8{},
               q8 ? Kvq8{B.v_codes + (size_t) p0 * kv_stride, B.v_scales + (size_t) p0 * s_stride} : Kvq8{}, cold, m1,
               false);
    if (m2 > 0)
        eval_piece(q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{},
                   q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{}, cold + m1, m2, false);
}

// attend() of one piece of a streamed layer: the piece's rows are its caches from 0, pos is the position in them.
void Qwen35::attn_piece(const float* q, const uint16_t* kc, const uint16_t* vc, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n,
                        bool whole, float* out, float* ml_out, bool fd) {
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const bool q8 = codes_read();
    const size_t score_floats = (size_t) chunk_ * n_head_ * opt_.max_context;
    if (n >= kStageRows) {
        uint16_t* kw = q8 ? (uint16_t*) q8w_ : nullptr;
        uint16_t* vw = q8 ? kw + work_halves_ : nullptr;
        attend_batch(blas_, q, kc, vc, pos, n, score_, score_floats, awork_, out, head_dim_, n_head_, n_kv_, scale, st_, 0,
                     0, whole, kq8, vq8, kw, vw, q8 ? work_halves_ : 0, nullptr, ml_out);
        return;
    }
    if (!(fd && (q8 ? fdec_q8(q, kq8, vq8, pos, n, nullptr, 0, out, head_dim_, n_head_, n_kv_, scale, score_, score_floats,
                              st_, whole, ml_out)
                    : fdec_f16(q, kc, vc, pos, n, nullptr, 0, out, head_dim_, n_head_, n_kv_, scale, score_, score_floats,
                               st_, whole, ml_out))))
        attend(q, kc, vc, pos, n, score_, chunk_, out, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, whole, nullptr, 0,
               score_floats, kq8, vq8, nullptr, score_floats, ml_out);
}

// The cold rows [row0, row0 + m) of a streamed layer copied from its host copy into staging buffer buf, on the copy
// stream, once the buffer's last reader is through (and, with wait_store, the rows stored just before are in the copy).
void Qwen35::start_kv_stage_dma(const Block& B, int buf, int64_t row0, int m, bool wait_store) {
    if (m <= 0) return;
    if (wait_store && kv_store_ev_) ck(cudaStreamWaitEvent(cs(kv_stream_), (cudaEvent_t) kv_store_ev_, 0), "wait kv store");
    ck(cudaStreamWaitEvent(cs(kv_stream_), (cudaEvent_t) kv_use_ev_[buf], 0), "wait kv stage free");
    KvStage& S = kv_stage_[buf];
    if (opt_.kv_format == KvFmt::kQ4_0) {
        const size_t q4_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(block_q4_0);
        ck(cudaMemcpyAsync(S.k_q4, (const uint8_t*) B.h_k_q4 + (size_t) row0 * q4_row, (size_t) m * q4_row,
                           cudaMemcpyHostToDevice, cs(kv_stream_)), "kv stage k_q4");
        ck(cudaMemcpyAsync(S.v_q4, (const uint8_t*) B.h_v_q4 + (size_t) row0 * q4_row, (size_t) m * q4_row,
                           cudaMemcpyHostToDevice, cs(kv_stream_)), "kv stage v_q4");
    } else if (opt_.kv_format == KvFmt::kQ8_0) {
        const size_t crow = (size_t) n_kv_ * head_dim_;
        const size_t srow = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        ck(cudaMemcpyAsync(S.k_codes, (const uint8_t*) B.h_k_codes + (size_t) row0 * crow, (size_t) m * crow,
                           cudaMemcpyHostToDevice, cs(kv_stream_)), "kv stage k_codes");
        ck(cudaMemcpyAsync(S.k_scales, (const uint8_t*) B.h_k_scales + (size_t) row0 * srow, (size_t) m * srow,
                           cudaMemcpyHostToDevice, cs(kv_stream_)), "kv stage k_scales");
        ck(cudaMemcpyAsync(S.v_codes, (const uint8_t*) B.h_v_codes + (size_t) row0 * crow, (size_t) m * crow,
                           cudaMemcpyHostToDevice, cs(kv_stream_)), "kv stage v_codes");
        ck(cudaMemcpyAsync(S.v_scales, (const uint8_t*) B.h_v_scales + (size_t) row0 * srow, (size_t) m * srow,
                           cudaMemcpyHostToDevice, cs(kv_stream_)), "kv stage v_scales");
    } else {
        const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
        ck(cudaMemcpyAsync(S.keys, (const uint8_t*) B.h_keys + (size_t) row0 * row, (size_t) m * row,
                           cudaMemcpyHostToDevice, cs(kv_stream_)), "kv stage keys");
        ck(cudaMemcpyAsync(S.vals, (const uint8_t*) B.h_vals + (size_t) row0 * row, (size_t) m * row,
                           cudaMemcpyHostToDevice, cs(kv_stream_)), "kv stage vals");
    }
    ck(cudaEventRecord((cudaEvent_t) kv_ev_[buf], cs(kv_stream_)), "kv stage ready");
}

// Before a GDN layer, the next attention layer's first cold chunk is copied (the copy overlaps this layer and the
// MoE): run() calls this before each layer, graph-replayed or not.
void Qwen35::prefetch_layer(int il, int n) {
    const int ring = kv_ring();
    if (ring <= 0 || rows_of_ || recording_ || shared_ahead_) return;
    const Block& B = blocks_[(size_t) il];
    if (!B.gdn || B.next_attn < 0 || pos_ + n <= ring) return;
    const int64_t cold = pos_ + n - ring;
    const int m0 = (int) std::min<int64_t>(cold, kv_stage_rows());
    if (prefetched_il_ == B.next_attn && prefetched_m_ == m0) return;
    if (m0 <= pos_) {
        start_kv_stage_dma(blocks_[(size_t) B.next_attn], 0, 0, m0, false);
        prefetched_il_ = B.next_attn;
        prefetched_m_ = m0;
    }
}

// The recordings of an attention layer's head are numbered by the rows they are sized for.
int Qwen35::head_key(int il, int n) const { return blocks_[(size_t) il].gdn ? 0 : room_key(n); }

int64_t Qwen35::turn(int64_t row) const {
    int32_t thw[3];
    turn_of(row, thw);
    return thw[0];
}

void Qwen35::attn(Block& B, int il, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, il, n);
    if (kv_ring() > 0) return attn_run(B, il, pos, n, false);
    int64_t first;
    const int32_t* axes = turns(pos, n, &first);
    const long long* at = recording_ ? at_ : nullptr;
    lin(B.wq, xn_, qg_, n, xq_);
    head_norm_rope(qg_, 2 * head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, first, n, rms_eps_,
                   st_, nullptr, axes, sections_, at ? at + 1 : nullptr);
    attn_store(B, pos, n);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const bool q8 = codes_read();
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company
    if (n >= kStageRows) {
        // a prompt batch: the fp16 path passes the caches where they lie (null caches and no workspace under q8);
        // under q8 each block of K and V the products read is dequantized into q8w_ (kwork then vwork)
        const uint16_t* kc = q8 ? nullptr : B.keys;
        const uint16_t* vc = q8 ? nullptr : B.vals;
        uint16_t* kw = q8 ? (uint16_t*) q8w_ : nullptr;
        uint16_t* vw = q8 ? kw + work_halves_ : nullptr;
        const size_t score_floats = (size_t) chunk_ * n_head_ * opt_.max_context;
        if (!(fattn_ && (q8 ? fattn_q8(q_, Kvq8{B.k_codes, B.k_scales, kv_rs_}, Kvq8{B.v_codes, B.v_scales, kv_rs_}, pos, n, att_,
                                       head_dim_, n_head_, n_kv_, scale, score_, score_floats, st_)
                            : fattn_f16(q_, B.keys, B.vals, pos, n, att_, head_dim_, n_head_, n_kv_, scale, score_,
                                        score_floats, st_))))
            attend_batch(blas_, q_, kc, vc, pos, n, score_, score_floats, awork_, att_, head_dim_, n_head_, n_kv_, scale,
                         st_, 0, 0, false, q8 ? Kvq8{B.k_codes, B.k_scales, kv_rs_} : Kvq8{},
                         q8 ? Kvq8{B.v_codes, B.v_scales, kv_rs_} : Kvq8{}, kw, vw, q8 ? work_halves_ : 0);
    } else if (!(fdec_ && pos >= fdec_from_ &&
                 (q8 ? fdec_q8(q_, Kvq8{B.k_codes, B.k_scales, kv_rs_}, Kvq8{B.v_codes, B.v_scales, kv_rs_}, pos, n, at,
                                        at ? room(n) : 0, att_, head_dim_, n_head_, n_kv_, scale, score_,
                                        (size_t) chunk_ * n_head_ * opt_.max_context, st_)
                              : fdec_f16(q_, B.keys, B.vals, pos, n, at, at ? room(n) : 0, att_, head_dim_, n_head_,
                                         n_kv_, scale, score_, (size_t) chunk_ * n_head_ * opt_.max_context, st_))))
        attend(q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_,
               n_kv_, scale, st_, 0, 0, false, at, at ? room(n) : 0, /*score_floats*/ 0,
               q8 ? Kvq8{B.k_codes, B.k_scales, kv_rs_} : Kvq8{}, q8 ? Kvq8{B.v_codes, B.v_scales, kv_rs_} : Kvq8{}, nullptr,
               (size_t) chunk_ * n_head_ * opt_.max_context /*part_floats: STRATA_HIP_SPLIT_ATTEND's partial sums*/);
    mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
    lin(B.wo, att_, ao_, n);
}

// attn() for a block of guesses: the block's tokens see each other, each attending to the rows up to the block's last.
// One launch (chunk_ holds a window), so a recording reads the position on the card as attn()'s do.
void Qwen35::attn_block_pass(Block& B, int il, int64_t pos, int n) {
    if (kv_ring() > 0) return attn_run(B, il, pos, n, true);
    int64_t first;
    const int32_t* axes = turns(pos, n, &first);
    const long long* at = recording_ ? at_ : nullptr;
    lin(B.wq, xn_, qg_, n, xq_);
    head_norm_rope(qg_, 2 * head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, first, n, rms_eps_,
                   st_, nullptr, axes, sections_, at ? at + 1 : nullptr);
    attn_store(B, pos, n);
    const bool q8 = codes_read();
    attend(q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_,
           1.f / std::sqrt((float) head_dim_), st_, 0, 0, true, at, at ? room(n) : 0, 0,
           q8 ? Kvq8{B.k_codes, B.k_scales, kv_rs_} : Kvq8{}, q8 ? Kvq8{B.v_codes, B.v_scales, kv_rs_} : Kvq8{});
    mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
    lin(B.wo, att_, ao_, n);
}

// step()'s rows, each of its own sequence: the products for all of them at once, each row's position from at_, and
// each row's keys and values stored and attended to in its own sequence's caches.
void Qwen35::attn_rows(Block& B, int il, int n) {
    const long long* at = at_;
    lin(B.wq, xn_, qg_, n, xq_);
    lin_pair(B.wk, B.wv, xn_, kraw_, v_, n, xq_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const bool q4 = opt_.kv_format == KvFmt::kQ4_0;
    const bool codes = q8 || q4;   // the attention reads Q8_0 codes (q4_0's are dequantized into them)
    const int ring = kv_ring();
    const size_t kv = (size_t) n_kv_ * head_dim_, hq = (size_t) n_head_ * head_dim_;
#if !defined(STRATA_GPU_METAL)
    // Under a q8_0 cache the turns and the stores are a row's alone: every row of every sequence in one launch
    // each, a row's own turn and cache from at_ and seq_buf().  Then the attention, a sequence at a time.  Not for a
    // streamed cache: its rows are the ring's, not the sequence's.
    if (q8 && ring == 0 && n <= RowCaches::kMax) {
        head_norm_rope_rows(qg_, 2 * head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, n,
                            rms_eps_, st_, sections_, at + 1);
        head_norm_rope_rows(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, n_rot_, theta_scale_, n, rms_eps_,
                            st_, sections_, at + 1);
        RowCaches keys, vals;
        keys.rs = vals.rs = kv_rs_;
        for (int r = 0; r < n; ++r) {
            keys.codes[r] = (int8_t*) seq_buf(rows_of_[r].seq, B.k_codes);
            keys.scales[r] = (uint16_t*) seq_buf(rows_of_[r].seq, B.k_scales);
            vals.codes[r] = (int8_t*) seq_buf(rows_of_[r].seq, B.v_codes);
            vals.scales[r] = (uint16_t*) seq_buf(rows_of_[r].seq, B.v_scales);
        }
        store_q8_rows(k_, keys, (int) kv, at, n, st_);
        store_q8_rows(v_, vals, (int) kv, at, n, st_);
        // the attention for every sequence in one launch a kernel, where attend() would run it for each of them
        AttnSeqs seqs;
        seqs.rs = kv_rs_;
        bool all = true;
        for (int i = 0, m; i < n && all; i += m) {
            const Row& r = rows_of_[i];
            m = i + 1 < n && rows_of_[i + 1].seq == r.seq ? 2 : 1;
            if (fdec_ && r.pos >= fdec_from_) all = false;   // the decode kernels of fattn.cu: a sequence at a time
            const int64_t keep = pos_;
            pos_ = r.pos;
            const int sized = room(m);
            pos_ = keep;
            const int k = seqs.n++;
            seqs.q[k] = q_ + i * hq;
            seqs.out[k] = att_ + i * hq;
            seqs.kc[k] = keys.codes[i];
            seqs.ks[k] = keys.scales[i];
            seqs.vc[k] = vals.codes[i];
            seqs.vs[k] = vals.scales[i];
            seqs.at[k] = at + 2 * i;
            seqs.keys[k] = sized;
            seqs.tokens[k] = m;
            all = seqs.n < AttnSeqs::kMax || i + m >= n;
        }
        if (all && attend_seqs_plan(seqs, score_, (size_t) chunk_ * n_head_ * opt_.max_context, chunk_, head_dim_,
                                    n_head_, n_kv_)) {
            attend_seqs(seqs, head_dim_, n_head_, n_kv_, scale, st_);
            mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
            lin(B.wo, att_, ao_, n);
            return;
        }
        for (int i = 0, m; i < n; i += m) {
            const Row& r = rows_of_[i];
            const long long* ai = at + 2 * i;
            m = i + 1 < n && rows_of_[i + 1].seq == r.seq ? 2 : 1;
            const Kvq8 kq{keys.codes[i], keys.scales[i], kv_rs_}, vq{vals.codes[i], vals.scales[i], kv_rs_};
            const int64_t keep = pos_;
            pos_ = r.pos;
            const int sized = room(m);
            pos_ = keep;
            const size_t sf = (size_t) chunk_ * n_head_ * opt_.max_context;
            if (!(fdec_ && r.pos >= fdec_from_ &&
                  fdec_q8(q_ + i * hq, kq, vq, r.pos, m, ai, sized, att_ + i * hq, head_dim_, n_head_, n_kv_, scale,
                          score_, sf, st_)))
                attend(q_ + i * hq, nullptr, nullptr, r.pos, m, score_, chunk_, att_ + i * hq, head_dim_, n_head_,
                       n_kv_, scale, st_, 0, 0, false, ai, sized, 0, kq, vq);
        }
        mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
        lin(B.wo, att_, ao_, n);
        return;
    }
    // The same for an fp16 cache, when attend_seqs() has the launches attend() would run for every sequence.
    // Otherwise the loop below, a sequence at a time.
    if (!codes && ring == 0 && n <= RowHalves::kMax) {
        RowHalves keys, vals;
        for (int r = 0; r < n; ++r) {
            keys.rows[r] = (uint16_t*) seq_buf(rows_of_[r].seq, B.keys);
            vals.rows[r] = (uint16_t*) seq_buf(rows_of_[r].seq, B.vals);
        }
        AttnSeqs seqs;
        seqs.f16 = true;
        bool all = true;
        for (int i = 0, m; i < n && all; i += m) {
            const Row& r = rows_of_[i];
            m = i + 1 < n && rows_of_[i + 1].seq == r.seq ? 2 : 1;
            if (fdec_ && r.pos >= fdec_from_) all = false;
            const int64_t keep = pos_;
            pos_ = r.pos;
            const int sized = room(m);
            pos_ = keep;
            const int k = seqs.n++;
            seqs.q[k] = q_ + i * hq;
            seqs.out[k] = att_ + i * hq;
            seqs.kc[k] = (const int8_t*) keys.rows[i];
            seqs.vc[k] = (const int8_t*) vals.rows[i];
            seqs.at[k] = at + 2 * i;
            seqs.keys[k] = sized;
            seqs.tokens[k] = m;
            all = seqs.n < AttnSeqs::kMax || i + m >= n;
        }
        if (all && attend_seqs_plan(seqs, score_, (size_t) chunk_ * n_head_ * opt_.max_context, chunk_, head_dim_,
                                    n_head_, n_kv_)) {
            head_norm_rope_rows(qg_, 2 * head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, n,
                                rms_eps_, st_, sections_, at + 1);
            head_norm_rope_rows(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, n_rot_, theta_scale_, n,
                                rms_eps_, st_, sections_, at + 1);
            // the stores a sequence at a time, as the loop below: store_f16_rows(), a row each from its own
            // position, left the draft block's caches different from run to run under --no-graphs
            for (int i = 0, m; i < n; i += m) {
                m = i + 1 < n && rows_of_[i + 1].seq == rows_of_[i].seq ? 2 : 1;
                store_f16_at(k_ + i * kv, keys.rows[i], (int) kv, at + 2 * i, m, 0, st_);
                store_f16_at(v_ + i * kv, vals.rows[i], (int) kv, at + 2 * i, m, 0, st_);
            }
            attend_seqs(seqs, head_dim_, n_head_, n_kv_, scale, st_);
            mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
            lin(B.wo, att_, ao_, n);
            return;
        }
    }
#endif
    for (int i = 0, m; i < n; i += m) {
        const Row& r = rows_of_[i];
        const long long* ai = at + 2 * i;
        // a sequence's rows at a time: one, or a token and its guess (step() with guesses), a window of two.  The
        // turn takes one position and counts on from it, which suits a window's rows only.
        m = i + 1 < n && rows_of_[i + 1].seq == r.seq ? 2 : 1;
        head_norm_rope(qg_ + 2 * i * hq, 2 * head_dim_, fp(B.q_norm), q_ + i * hq, head_dim_, n_head_, n_rot_,
                       theta_scale_, 0, m, rms_eps_, st_, nullptr, nullptr, sections_, ai + 1);
        head_norm_rope(kraw_ + i * kv, head_dim_, fp(B.k_norm), k_ + i * kv, head_dim_, n_kv_, n_rot_, theta_scale_, 0, m,
                       rms_eps_, st_, nullptr, nullptr, sections_, ai + 1);
        if (ring > 0) {
            // a streamed layer: the rows go to the ring and the host copy, and attend from there (attn_stream())
            store_ring(B, r.pos, m, k_ + i * kv, v_ + i * kv);
            attn_stream(B, il, r.pos, m, false, q_ + i * hq, att_ + i * hq);
            continue;
        }
        Kvq8 kq{}, vq{};
        uint16_t *kc = nullptr, *vc = nullptr;
        if (q4) {
            // the rows as Q4_0 at their positions, then the Q8 codes the attention reads, dequantized from them
            kq = {(int8_t*) seq_buf(r.seq, B.k_codes), (uint16_t*) seq_buf(r.seq, B.k_scales), 0};
            vq = {(int8_t*) seq_buf(r.seq, B.v_codes), (uint16_t*) seq_buf(r.seq, B.v_scales), 0};
            block_q4_0* kb = (block_q4_0*) seq_buf(r.seq, B.k_q4);
            block_q4_0* vb = (block_q4_0*) seq_buf(r.seq, B.v_q4);
            store_q4_at(k_ + i * kv, kb, (int) kv, ai, m, 0, st_);
            store_q4_at(v_ + i * kv, vb, (int) kv, ai, m, 0, st_);
            dequant_q4_to_q8(kb, (int8_t*) kq.codes, (uint16_t*) kq.scales, (int) kv, m, st_, ai);
            dequant_q4_to_q8(vb, (int8_t*) vq.codes, (uint16_t*) vq.scales, (int) kv, m, st_, ai);
        } else if (q8) {
            kq = {(int8_t*) seq_buf(r.seq, B.k_codes), (uint16_t*) seq_buf(r.seq, B.k_scales), kv_rs_};
            vq = {(int8_t*) seq_buf(r.seq, B.v_codes), (uint16_t*) seq_buf(r.seq, B.v_scales), kv_rs_};
            store_q8_at(k_ + i * kv, (int8_t*) kq.codes, (uint16_t*) kq.scales, (int) kv, ai, m, 0, st_, kv_rs_);
            store_q8_at(v_ + i * kv, (int8_t*) vq.codes, (uint16_t*) vq.scales, (int) kv, ai, m, 0, st_, kv_rs_);
        } else {
            kc = (uint16_t*) seq_buf(r.seq, B.keys);
            vc = (uint16_t*) seq_buf(r.seq, B.vals);
            store_f16_at(k_ + i * kv, kc, (int) kv, ai, m, 0, st_);
            store_f16_at(v_ + i * kv, vc, (int) kv, ai, m, 0, st_);
        }
        const int64_t keep = pos_;
        pos_ = r.pos;
        const int sized = room(m);
        pos_ = keep;
        const size_t sf = (size_t) chunk_ * n_head_ * opt_.max_context;
        if (!(fdec_ && r.pos >= fdec_from_ &&
              (codes ? fdec_q8(q_ + i * hq, kq, vq, r.pos, m, ai, sized, att_ + i * hq, head_dim_, n_head_, n_kv_,
                                     scale, score_, sf, st_)
                           : fdec_f16(q_ + i * hq, kc, vc, r.pos, m, ai, sized, att_ + i * hq, head_dim_, n_head_,
                                      n_kv_, scale, score_, sf, st_))))
            attend(q_ + i * hq, kc, vc, r.pos, m, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_,
                   0, 0, false, ai, sized, 0, kq, vq);
    }
    mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
    lin(B.wo, att_, ao_, n);
}

// A layer from its input to its router's copy.  A prompt batch's MoE (moe_rows) does its own routing.
// A norm and its quantization are one launch, and so are the residual's sum and the next norm
// (native_add_rms_norm_q8_1(), the same bytes as the launches it stands for: poly_norm_q8_test).  Metal keeps them
// apart.
void Qwen35::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
#if defined(STRATA_GPU_METAL)
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
#else
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
#endif
    dbg("attn_in", il, xn_, n_embd_);
    if (block_pass_) {
        if (B.gdn) gdn_block_pass(B, n);
        else attn_block_pass(B, il, pos_, n);
    } else if (B.gdn) {
        gdn(B, n);
    } else {
        attn(B, il, pos_, n);
    }
    dbg("attn", il, ao_, n_embd_);
#if defined(STRATA_GPU_METAL)
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
#else
    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
#endif
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

void Qwen35::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Qwen35::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: a softmax over every expert's score, the n_used most likely, and
// their weights made to sum to one.
void Qwen35::route(int, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    float mx = logits[0];
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e]);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs[e] = std::exp(logits[e] - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = probs[best] * inv;
        sum += w[k];
        probs[best] = -1.f;
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) w[k] /= denom;
}

// Clover's block of guesses chooses its experts among few: each token's n_used likeliest, their probabilities added
// up over the block, and the experts with the most of that sum, in order, while the ones before them hold less than
// coreset_ of it.  The others' scores become -inf for every token of the block.  Ties go to the lower expert.
void Qwen35::route_block(int, float* scores, int n) {
    if (coreset_ <= 0.f) return;
    std::vector<float> mass((size_t) n_expert_, 0.f), p((size_t) n_expert_);
    std::vector<int> order((size_t) n_expert_);
    const auto likelier = [&](const std::vector<float>& v) {
        return [&v](int a, int b) { return v[(size_t) a] > v[(size_t) b] || (v[(size_t) a] == v[(size_t) b] && a < b); };
    };
    for (int t = 0; t < n; ++t) {
        const float* s = scores + (size_t) t * n_expert_;
        const float mx = *std::max_element(s, s + n_expert_);
        double all = 0.0;
        for (int e = 0; e < n_expert_; ++e) all += p[(size_t) e] = std::exp(s[e] - mx);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + n_used_, order.end(), likelier(p));
        for (int k = 0; k < n_used_; ++k) mass[(size_t) order[(size_t) k]] += (float) (p[(size_t) order[(size_t) k]] / all);
    }
    double total = 0.0;
    for (float m : mass) total += m;
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), likelier(mass));
    std::vector<char> keep((size_t) n_expert_, 0);
    double before = 0.0;
    for (int k = 0; k < n_expert_; ++k) {
        if (k > 0 && before >= coreset_ * total) break;
        keep[(size_t) order[(size_t) k]] = 1;
        before += mass[(size_t) order[(size_t) k]];
    }
    for (int t = 0; t < n; ++t)
        for (int e = 0; e < n_expert_; ++e)
            if (!keep[(size_t) e]) scores[(size_t) t * n_expert_ + e] = -INFINITY;
}

bool Qwen35::route_card(int, int n, int32_t* ids, float* w) {
    if (block_pass_ && coreset_ > 0.f) coreset_mask(rl_, n_expert_, n_used_, n, coreset_, st_);   // route_block()'s
    route_rows(rl_, n_expert_, n_used_, n, 0, 1.f, nullptr, nullptr, true, 0.f, ids, w, st_);
    return true;
}

// The shared expert, times the sigmoid of its own gate.  It runs on the GPU while the host routes.
void Qwen35::moe_shared(const Layer& L, int il, int n) {
    lin_pair(L.sh_gate, L.sh_up, xn_, g_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
    lin(blocks_[(size_t) il].sh_gate_inp, xn_, sg_, n);
    mul_sigmoid(fo_, sg_, n_embd_, n, 1, true, st_);
}

// The draft block is one more attention layer.
void Qwen35::block_attend(int64_t pos, int n) { attn(blocks_[(size_t) n_layer_], n_layer_, pos, n); }
void Qwen35::block_rows(int64_t pos, int n) { attn_store(blocks_[(size_t) n_layer_], pos, n); }

}  // namespace strata::poly
