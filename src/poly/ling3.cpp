// src/poly/ling3.cpp - Ling 3.0 Flash's layers.  The reference for the math is llama.cpp's bailingmoe3 graph
// (src/models/bailingmoe3.cpp); the KDA layers are GLM 5.3 Flash's (glm5next.cpp) with full-rank gates.
//
// MLA, as llama.cpp runs it: a row of the cache is the latent, RMS-normed (kv_rank values), then the key's rotary
// part turned by the position (n_rot values), and it is the key and the value both.  A head's query is its
// non-rotary part through attn_k_b (to kv_rank values) then its rotary part turned; the attention's output for a
// head is a latent, whose first kv_rank values go through attn_v_b.  One key head, so attend() reads a row once
// for the 32 heads.
#include "ling3.hpp"

#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kConv = 4;
constexpr size_t kScoreFloats = size_t(16) << 20;   // the attention's scores of the tokens it takes at a time
constexpr float kMinWeightSum = 6.103515625e-5f;    // the floor ggml's MoE graph puts under the gate sum

constexpr Names kTensorNames[] = {
    {"nextn.shared_head_norm.weight", {"layer_output_norm.weight"}},
};

// rows x width floats from src (src_stride apart) to dst (dst_stride apart), on the stream
void copy_rows(float* dst, size_t dst_stride, const float* src, size_t src_stride, size_t width, size_t rows,
               void* stream) {
    ck(cudaMemcpy2DAsync(dst, dst_stride * sizeof(float), src, src_stride * sizeof(float), width * sizeof(float), rows,
                         cudaMemcpyDeviceToDevice, cs(stream)),
       "ling3 copy");
}

}  // namespace

const Names* Ling3::tensor_alias(const char* name) const { return names_of(kTensorNames, name); }

void Ling3::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_mla_ = n_kda_ = 0;
}

void Ling3::check_draft_block() {
    if (!blocks_[(size_t) n_layer_].mla || !layers_[(size_t) n_layer_].moe)
        throw std::runtime_error("the block that guesses ahead is not one this engine runs; --no-draft loads "
                                 "the model without it");
}

void Ling3::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    n_ff_ = shape_i("feed_forward_length");
    n_head_ = shape_i("attention.head_count");
    w_scale_ = shape_f("expert_weights_scale");
    w_norm_ = shape_i("expert_weights_norm") != 0;
    kda_dim_ = shape_i("kda.head_dim");
    kda_lb_ = shape_f("kda.gate_lower_bound");
    kv_rank_ = shape_i("attention.kv_lora_rank");
    mla_k_ = shape_i("attention.key_length_mla");
    mla_v_ = shape_i("attention.value_length_mla");
    n_rot_ = shape_i("rope.dimension_count");
    nope_ = mla_k_ - n_rot_;
    row_ = kv_rank_ + n_rot_;
    d_inner_ = n_head_ * kda_dim_;
    if (const MetaValue* g = find("expert_group_count")) n_group_ = (int) g->u;
    if (const MetaValue* g = find("expert_group_used_count")) group_used_ = (int) g->u;
    rope_.theta_scale = std::pow(shape_f("rope.freq_base"), -2.0f / (float) n_rot_);

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(shape_i("ssm.conv_kernel") == kConv, "ssm.conv_kernel is not 4");
    if (const MetaValue* q = find("attention.q_lora_rank")) q_lora_ = (int) q->u;
    need(!find("rope.dimension_sections"), "rope sections (a vision file)");
    const MetaValue* gating = find("expert_gating_func");
    need(!gating || gating->u == 2, "the expert gate is not a sigmoid");
    need(shape_i("expert_shared_count") == 1, "expert_shared_count is not 1");
    need(kda_dim_ > 0 && kda_dim_ % 32 == 0 && kda_dim_ <= 256, "kda.head_dim is not a multiple of 32 up to 256");
    need(nope_ > 0 && n_rot_ % 2 == 0 && row_ % 4 == 0, "bad key_length_mla or rope.dimension_count");
    need(n_group_ >= 1 && n_expert_ % n_group_ == 0 && group_used_ >= 1 && group_used_ <= n_group_ &&
             n_expert_ / n_group_ >= 2,
         "bad expert_group_count");
    need(opt_.kv_format != KvFmt::kQ8_0, "--kv q8_0 (this engine keeps its MLA cache in fp16)");
    clamp_exp_ = clamp_limits(find("swiglu_clamp_exp"), n_layer_ + 1);
    clamp_shexp_ = clamp_limits(find("swiglu_clamp_shexp"), n_layer_ + 1);
}

void Ling3::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b, Want w = kAnyMat) {
        return upload(tensor(il, n), blk(il, n), a, b, w);
    };
    auto has = [&](const char* n) { return tensor(il, n) != nullptr; };

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.mla = has("attn_kv_a_mqa.weight");
    L.fixed = !B.mla;   // the attention's launches take the position
    if (B.mla) {
        if (q_lora_ > 0) {
            B.q_a = mat("attn_q_a.weight", n_embd_, q_lora_);
            B.q_a_norm = f32("attn_q_a_norm.weight", q_lora_);
            B.q_b = mat("attn_q_b.weight", q_lora_, (int64_t) n_head_ * mla_k_);
        } else {
            B.q = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * mla_k_);
        }
        B.kv_a = mat("attn_kv_a_mqa.weight", n_embd_, row_);
        B.kv_a_norm = f32("attn_kv_a_norm.weight", kv_rank_);
        B.k_b = mat("attn_k_b.weight", nope_, (int64_t) kv_rank_ * n_head_, kFloatMat);
        B.v_b = mat("attn_v_b.weight", kv_rank_, (int64_t) mla_v_ * n_head_, kFloatMat);
        B.gate = mat("attn_gate.weight", n_embd_, n_head_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * mla_v_, n_embd_);
        ++n_mla_;
    } else {
        B.wq = mat("attn_q.weight", n_embd_, d_inner_);
        B.wk = mat("attn_k.weight", n_embd_, d_inner_);
        B.wv = mat("attn_v.weight", n_embd_, d_inner_);
        B.wo = mat("attn_output.weight", d_inner_, n_embd_);
        B.conv_q = f32("ssm_conv1d_q.weight", kConv, d_inner_);
        B.conv_k = f32("ssm_conv1d_k.weight", kConv, d_inner_);
        B.conv_v = f32("ssm_conv1d_v.weight", kConv, d_inner_);
        B.f_a = mat("ssm_f_a.weight", n_embd_, d_inner_);
        B.dt_b = f32("ssm_dt.bias", d_inner_);
        B.beta = mat("ssm_beta.weight", n_embd_, n_head_);
        // llama.cpp: decay = lower_bound * sigmoid((f_a x + dt_b) * ssm_a); kda_step() makes it of
        // sigmoid(-(g1 * a)), so a goes up negated
        B.a = upload(tensor(il, "ssm_a"), blk(il, "ssm_a"), 1, n_head_, kF32, [](float v) { return -v; });
        B.g_a = mat("ssm_g_a.weight", n_embd_, d_inner_);
        B.o_norm = f32("ssm_norm.weight", kda_dim_);
        ++n_kda_;
    }

    if (!has("ffn_gate_inp.weight")) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(b->name + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
    }
}

void Ling3::assign_scratch() {
    const size_t n = (size_t) rows_, d = n * d_inner_, h = n * n_head_;
    proj_ = fbuf(3 * d);
    qkv_ = fbuf(3 * d);
    g1_ = fbuf(d);
    g2_ = fbuf(d);
    beta_ = fbuf(h);
    gated_ = fbuf(d);
    q_ = fbuf(h * mla_k_);
    if (q_lora_ > 0) qa_ = fbuf(n * q_lora_), qan_ = fbuf(n * q_lora_);
    qn_ = fbuf(h * nope_);
    kva_ = fbuf(n * row_);
    lat_ = fbuf(n * kv_rank_);
    qabs_ = fbuf(h * kv_rank_);
    qf_ = fbuf(h * row_);
    att_ = fbuf(h * row_);
    kqv_ = fbuf(h * kv_rank_);
    vout_ = fbuf(h * mla_v_);
    agate_ = fbuf(h);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    if (rows_ >= kStageRows) awork_ = fbuf(h * (row_ + row_ / 2 + 3) + 1);
}

int Ling3::widest() const { return std::max({d_inner_, n_head_ * mla_v_, n_head_ * mla_k_, row_, kv_rank_, q_lora_}); }

size_t Ling3::recurrent_floats() const {
    return size_t(n_kda_) * (size_t(3) * d_inner_ * (kConv - 1) + size_t(n_head_) * kda_dim_ * kda_dim_);
}

void Ling3::alloc_caches() {
    const size_t conv_n = size_t(3) * d_inner_ * (kConv - 1), s_n = size_t(n_head_) * kda_dim_ * kda_dim_;
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.mla) {
            B.kv = (uint16_t*) kv_alloc((size_t) row_ * sizeof(uint16_t), 1);
        } else {
            B.conv = st;
            st += conv_n;
            B.s = st;
            st += s_n;
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

void Ling3::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (B.mla) return;
    const size_t n = size_t(3) * d_inner_ * (kConv - 1) + size_t(n_head_) * kda_dim_ * kda_dim_;
    ck(cudaMemcpy(dst + (size_t) (B.conv - state_) * sizeof(float), B.conv, n * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
}

size_t Ling3::card_row_bytes() const { return (size_t) n_mla_ * row_ * sizeof(uint16_t); }

void Ling3::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) row_ * sizeof(uint16_t);
    for (const Block& B : blocks_) {
        if (!B.mla || !B.kv || n == 0) continue;
        ck(cudaMemcpy(dst, (const uint8_t*) B.kv + (size_t) first * row, (size_t) n * row, cudaMemcpyDeviceToHost),
           "save rows");
        dst += (size_t) n * row;
    }
}

void Ling3::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) row_ * sizeof(uint16_t);
    for (const Block& B : blocks_) {
        if (!B.mla || !B.kv || n == 0) continue;
        ck(cudaMemcpy((uint8_t*) B.kv + (size_t) first * row, src + (size_t) (first - from) * row, (size_t) n * row,
                      cudaMemcpyHostToDevice),
           "load rows");
        src += (size_t) count * row;
    }
}

void Ling3::kda(Block& B, int n) {
    const size_t d = (size_t) n * d_inner_;
    lin(B.wq, xn_, proj_, n, xq_);
    lin(B.wk, xn_, proj_ + d, n, xq_);
    lin(B.wv, xn_, proj_ + 2 * d, n, xq_);
    kda_conv(proj_, B.conv, fp(B.conv_q), fp(B.conv_k), fp(B.conv_v), qkv_, d_inner_, n, st_,
             snap_bits_ ? snap_at_ + (B.conv - state_) : nullptr, rec_n_, std::popcount(snap_bits_));
    lin(B.f_a, xn_, g1_, n, xq_);
    add_bias(fp(B.dt_b), g1_, d_inner_, n, st_);
    lin(B.beta, xn_, beta_, n, xq_);
    lin(B.g_a, xn_, g2_, n, xq_);
    kda_step(qkv_, g1_, fp(B.a), beta_, B.s, g2_, fp(B.o_norm), gated_, kda_dim_, n_head_, n, kda_lb_, rms_eps_,
             st_, snap_bits_ ? snap_at_ + (B.s - state_) : nullptr, rec_n_, std::popcount(snap_bits_));
    lin(B.wo, gated_, ao_, n);
}

// The rows n tokens from `pos` leave in an MLA layer's cache.  xn_ and xq_ hold the layer's input.
void Ling3::mla_store(Block& B, int64_t pos, int n) {
    lin(B.kv_a, xn_, kva_, n, xq_);
    copy_rows(lat_, kv_rank_, kva_, row_, kv_rank_, n, st_);
    rms_norm(lat_, fp(B.kv_a_norm), kva_, kv_rank_, n, rms_eps_, st_, row_);
    dsv4_rope(kva_, kva_, row_, 1, kv_rank_, n_rot_, rope_, pos, 1, n, false, 0.f, false, st_);
    store_f16(kva_, B.kv, row_, pos, n, st_);
}

void Ling3::mla(Block& B, int64_t pos, int n) {
    const size_t h = (size_t) n * n_head_;
    if (q_lora_ > 0) {   // llama.cpp: q_b(rms_norm(q_a x) * q_a_norm)
        lin(B.q_a, xn_, qa_, n, xq_);
        rms_norm(qa_, fp(B.q_a_norm), qan_, q_lora_, n, rms_eps_, st_);
        lin(B.q_b, qan_, q_, n);
    } else {
        lin(B.q, xn_, q_, n, xq_);
    }
    dsv4_rope(q_, q_, mla_k_, n_head_, nope_, n_rot_, rope_, pos, 1, n, false, 0.f, false, st_);
    copy_rows(qn_, nope_, q_, mla_k_, nope_, h, st_);
    lin(B.k_b, qn_, qabs_, n, nullptr, kv_rank_);   // the key's projection, absorbed into the query: per head
    copy_rows(qf_, row_, qabs_, kv_rank_, kv_rank_, h, st_);
    copy_rows(qf_ + kv_rank_, row_, q_ + nope_, mla_k_, n_rot_, h, st_);
    mla_store(B, pos, n);
    const float scale = 1.f / std::sqrt((float) mla_k_);
    if (n >= kStageRows)
        attend_batch(blas_, qf_, B.kv, B.kv, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context, awork_,
                     att_, row_, n_head_, 1, scale, st_);
    else
        attend(qf_, B.kv, B.kv, pos, n, score_, chunk_, att_, row_, n_head_, 1, scale, st_);
    copy_rows(kqv_, kv_rank_, att_, row_, kv_rank_, h, st_);
    lin(B.v_b, kqv_, vout_, n, nullptr, mla_v_);
    lin(B.gate, xn_, agate_, n, xq_);
    mul_sigmoid(vout_, agate_, mla_v_, (int) h, 1, true, st_);
    lin(B.wo, vout_, ao_, n);
}

void Ling3::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

void Ling3::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    if (B.mla) mla(B, pos_, n);
    else kda(B, n);
    dbg("attn", il, ao_, n_embd_);
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

void Ling3::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Ling3::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: sigmoid gates; gate + bias summed over each group's best two picks the
// groups, the best group_used of them; the n_used largest of gate + bias in those; weights from the bare gates,
// made to sum to one, times the scale.
void Ling3::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    if (n_group_ > 1 && group_used_ < n_group_) {
        const int per = n_expert_ / n_group_;
        float gs[64];
        bool keep[64] = {};
        for (int g = 0; g < n_group_; ++g) {
            float a = -INFINITY, b = -INFINITY;
            for (int e = g * per; e < (g + 1) * per; ++e) {
                if (pick[e] > a) b = a, a = pick[e];
                else if (pick[e] > b) b = pick[e];
            }
            gs[g] = a + b;
        }
        for (int k = 0; k < group_used_; ++k) {
            int best = -1;
            for (int g = 0; g < n_group_; ++g)
                if (!keep[g] && (best < 0 || gs[g] > gs[best])) best = g;
            keep[best] = true;
        }
        for (int g = 0; g < n_group_; ++g)
            if (!keep[g]) std::fill(pick + g * per, pick + (g + 1) * per, -INFINITY);
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
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

void Ling3::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

void Ling3::block_attend(int64_t pos, int n) { mla(blocks_[(size_t) n_layer_], pos, n); }
void Ling3::block_rows(int64_t pos, int n) { mla_store(blocks_[(size_t) n_layer_], pos, n); }

}  // namespace strata::poly
