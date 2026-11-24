// src/poly/laguna.cpp - the layers of Laguna.  The reference for the math is llama.cpp's laguna graph
// (src/models/laguna.cpp) and its rope: llama_model_laguna::get_rope_params(), llama-context.cpp's YaRN factor
// and ggml's rope_yarn().  The caches, the window rings and the MoE are Flint's (flint.cpp), which this
// architecture shares but for the sinks.
#include "laguna.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
constexpr size_t kQ8WorkRows = 2048;               // the K or V blocks a q8 prompt batch dequantizes at a time
constexpr int64_t kGatingSigmoid = 2;              // llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
constexpr int kPeriod = 4;                         // without a pattern in the file: full at il % 4 == 0
constexpr float kPi = 3.14159265358979323846f;     // (float) M_PI, which MSVC does not define by default

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

// ggml_rope_yarn_corr_dim(): the pair whose wavelength fits n_rot turns into the original context.
float yarn_corr_dim(int n_dims, float n_ctx_orig, float n_rot, float base) {
    return (float) n_dims * std::log(n_ctx_orig / (n_rot * 2.0f * kPi)) / (2.0f * std::log(base));
}

}  // namespace

void Laguna::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

void Laguna::load_geometry() {
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
    auto num = [&](const char* key, double otherwise) {
        const MetaValue* v = find(key);
        return v && v->is_num() ? v->num() : otherwise;
    };

    // the query heads differ by layer (48 full, 72 window in S 2.1); the key and value heads do not
    const MetaValue& heads = must("attention.head_count");
    const MetaValue& kv = must("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(shape_i("attention.value_length") == head_dim_, "the keys and values have different widths");
    need(head_dim_ % 32 == 0 && head_dim_ <= 256, "bad head size");
    heads_.assign((size_t) n_layer_, 0);
    n_kv_ = (int) of_layer(kv, 0);
    max_head_ = 0;
    for (int il = 0; il < n_layer_; ++il) {
        const int h = (int) of_layer(heads, il);
        need((int) of_layer(kv, il) == n_kv_, "the key and value heads differ by layer");
        need(h > 0 && n_kv_ > 0 && h % n_kv_ == 0 && h <= 256, "bad attention.head_count(_kv)");
        heads_[(size_t) il] = h;
        max_head_ = std::max(max_head_, h);
    }

    // the window layers: from the file's pattern where it has one, else llama.cpp's load_swa_pattern(4, dense
    // first), full at il % 4 == 0.  No window in the file: every layer is full (Laguna M.1).
    const MetaValue* window = find("attention.sliding_window");
    window_ = window ? (int) window->u : 0;
    need(window_ >= 0, "bad attention.sliding_window");
    swa_.assign((size_t) n_layer_, 0);
    if (window_ > 0) {
        const MetaValue* pattern = find("attention.sliding_window_pattern");
        for (int il = 0; il < n_layer_; ++il)
            swa_[(size_t) il] = pattern ? of_layer(*pattern, il) != 0 : il % kPeriod != 0;
    }

    // the rotation: a full layer turns rope.dimension_count values with rope.freq_base and the file's scaling; a
    // window layer its own count and base, plain (llama.cpp zeroes YaRN's terms for them)
    rot_full_ = shape_i("rope.dimension_count");
    const MetaValue* rot_swa = find("rope.dimension_count_swa");
    rot_swa_ = rot_swa ? (int) rot_swa->u : rot_full_;
    need(rot_full_ > 0 && rot_full_ % 2 == 0 && rot_full_ <= head_dim_, "bad rope.dimension_count");
    need(rot_swa_ > 0 && rot_swa_ % 2 == 0 && rot_swa_ <= head_dim_, "bad rope.dimension_count_swa");
    const float base_full = shape_f("rope.freq_base");
    const float base_swa = (float) num("rope.freq_base_swa", base_full);
    theta_full_ = std::pow(base_full, -2.f / (float) rot_full_);
    theta_swa_ = std::pow(base_swa, -2.f / (float) rot_swa_);

    const MetaValue* scaling = find("rope.scaling.type");
    const std::string kind = scaling && scaling->type == MetaType::STRING ? scaling->s : std::string("none");
    need(kind == "none" || kind == "yarn" || kind == "linear", "rope.scaling.type is not none, linear or yarn");
    const float factor = (float) num("rope.scaling.factor", 1.0);
    need(kind == "none" || factor > 0.f, "bad rope.scaling.factor");
    yarn_ = RopeYarn{};
    yarn_on_ = false;
    if (kind != "none" && factor != 1.f) {
        yarn_on_ = true;
        yarn_.freq_scale = 1.0f / factor;
        // llama-context.cpp: the attention factor; for YaRN its own magnitude, cancelled there and put back by
        // ggml's rope_yarn (1 + 0.1 * ln(1 / freq_scale)), so the rotated values come out scaled by about that
        float attn = (float) num("rope.scaling.yarn_attn_factor", 1.0);
        if (kind == "yarn") {
            yarn_.ext_factor = 1.0f;
            auto mscale = [](float scale, float m) { return scale <= 1.0f ? 1.0f : 0.1f * m * std::log(scale) + 1.0f; };
            const float log_mul = (float) num("rope.scaling.yarn_log_multiplier", 0.0);
            attn = log_mul != 0.f ? mscale(factor, 1.0f) / mscale(factor, log_mul) : mscale(factor, 1.0f);
            attn *= 1.0f / (1.0f + 0.1f * std::log(factor));
            const MetaValue* orig = find("rope.scaling.original_context_length");
            const float n_orig = (float) (orig ? orig->u : (uint64_t) shape_i("context_length"));
            const float fast = (float) num("rope.scaling.yarn_beta_fast", 32.0);
            const float slow = (float) num("rope.scaling.yarn_beta_slow", 1.0);
            yarn_.corr_low = std::max(0.0f, std::floor(yarn_corr_dim(rot_full_, n_orig, fast, base_full)));
            yarn_.corr_high =
                std::min((float) (rot_full_ - 1), std::ceil(yarn_corr_dim(rot_full_, n_orig, slow, base_full)));
        }
        attn *= (float) num("rope.scaling.attn_factor", 1.0);
        yarn_.mscale = yarn_.ext_factor != 0.f ? attn * (1.0f + 0.1f * std::log(1.0f / yarn_.freq_scale)) : attn;
        std::fprintf(stderr,
                     "strata-poly: laguna: full layers turn %d of %d values by %s (factor %g, pairs %g to %g, "
                     "magnitude %.4f); window layers %d, plain\n",
                     rot_full_, head_dim_, kind.c_str(), (double) factor, (double) yarn_.corr_low,
                     (double) yarn_.corr_high, (double) yarn_.mscale, rot_swa_);
    }

    // the gate's width a layer, for its scratch: one a head or one a value (llama.cpp reads it from the tensor)
    max_gate_ = 0;
    for (int il = 0; il < n_layer_; ++il) {
        const TensorInfo* g = tensor(il, "attn_gate.weight");
        need(g != nullptr && g->shape.size() == 2, "a layer without attn_gate.weight");
        const int w = (int) g->shape[1], h = heads_[(size_t) il];
        need(w == h || w == h * head_dim_, "attn_gate.weight is neither a gate a head nor one a value");
        max_gate_ = std::max(max_gate_, w);
    }

    const MetaValue* gating = find("expert_gating_func");
    need(!gating || (int64_t) gating->u == kGatingSigmoid, "the expert gate is not a sigmoid");
    const MetaValue* shared = find("expert_shared_count");
    need(!shared || shared->u == 1, "expert_shared_count is not 1");
    const MetaValue* dense = find("leading_dense_block_count");
    n_dense_ = dense ? (int) dense->u : 0;
    need(n_dense_ >= 0 && n_dense_ < n_layer_, "bad leading_dense_block_count");
    n_ff_ = n_dense_ > 0 ? shape_i("feed_forward_length") : 0;
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : false;   // llama.cpp's default
    w_scale_ = (float) num("expert_weights_scale", 0.0);

    // this architecture puts no limit on a SwiGLU's inputs
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    // a prompt batch's products guard their half sums until a run shows Laguna's stay in range (Flint's did not)
    half_sum_guard_ = true;
}

void Laguna::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    B.swa = swa_[(size_t) il] != 0;
    B.n_head = heads_[(size_t) il];
    B.n_gate = (int) tensor(il, "attn_gate.weight")->shape[1];
    B.n_rot = B.swa ? rot_swa_ : rot_full_;
    B.theta = B.swa ? theta_swa_ : theta_full_;
    B.yarn = !B.swa && yarn_on_;
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) B.n_head * head_dim_);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wg = mat("attn_gate.weight", n_embd_, B.n_gate);
    B.q_norm = f32("attn_q_norm.weight", head_dim_);
    B.k_norm = f32("attn_k_norm.weight", head_dim_);
    B.wo = mat("attn_output.weight", (int64_t) B.n_head * head_dim_, n_embd_);
    if (il < n_dense_) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il, true);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

// The caches: a full layer keeps a row a position, a window layer a ring of the rows a call can read (Flint's).
void Laguna::alloc_caches() {
    ring_ = window_ > 0 ? std::min(opt_.max_context, window_ + rows_) : 0;
    keep_ = std::min(window_, ring_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t row = (size_t) n_kv_ * head_dim_;
    const size_t scale_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);   // a Q8_0 block's scale
    if (q8) std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", row + scale_row);
    size_t rings = 0, codes = 0, scales = 0;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        if (!runs(il) || !blocks_[(size_t) il].swa) continue;
        if (q8) {
            codes += 2 * (size_t) ring_ * row;
            scales += 2 * (size_t) ring_ * scale_row;
        } else {
            rings += 2 * (size_t) ring_ * row;
        }
    }
    uint8_t* rc = codes ? (uint8_t*) seq_alloc(codes, "the attention cache") : nullptr;
    uint8_t* rs = scales ? (uint8_t*) seq_alloc(scales, "the attention cache") : nullptr;
    uint16_t* kv = rings ? (uint16_t*) seq_alloc(rings * sizeof(uint16_t), "the attention cache") : nullptr;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.swa) {
            if (q8) {
                B.k_codes = (int8_t*) rc;
                rc += (size_t) ring_ * row;
                B.k_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
                B.v_codes = (int8_t*) rc;
                rc += (size_t) ring_ * row;
                B.v_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
            } else {
                B.keys = kv;
                kv += ring_ * row;
                B.vals = kv;
                kv += ring_ * row;
            }
        } else if (q8) {
            B.k_codes = (int8_t*) kv_alloc(row, 1);
            B.k_scales = (uint16_t*) kv_alloc(scale_row, 1);
            B.v_codes = (int8_t*) kv_alloc(row, 1);
            B.v_scales = (uint16_t*) kv_alloc(scale_row, 1);
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) max_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row, sized for the layer with the most query heads.
void Laguna::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * max_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    gate_ = fbuf(n * max_gate_);
    score_n_ = (size_t) chunk_ * max_head_ * opt_.max_context;
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * max_head_);   // a prompt batch's attention
    if (rows_ >= kStageRows && opt_.kv_format == KvFmt::kQ8_0) {
        work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
        q8w_ = fbuf((2 * work_halves_ + 1) / 2);
    }
}

// A sequence's rows are the full layers': a window layer's are part of its state (save_layer()).  Under --kv q8_0
// a row is its codes and, in a second array, a scale every 32 of them, saved one after the other.
size_t Laguna::card_row_bytes() const {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    size_t layers = 0;
    for (const Block& B : blocks_)
        if ((q8 ? B.k_codes != nullptr : B.keys != nullptr) && !B.swa) ++layers;
    return layers * 2 * (code_row + scale_row);
}

void Laguna::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    for (const Block& B : blocks_) {
        if (B.swa || n == 0 || !(q8 ? B.k_codes != nullptr : B.keys != nullptr)) continue;
        if (q8) {
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            continue;
        }
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
        }
    }
}

void Laguna::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    const size_t sk_c = (size_t) (first - from) * code_row, sk_s = (size_t) (first - from) * scale_row;
    const size_t all_c = (size_t) count * code_row, all_s = (size_t) count * scale_row;
    for (const Block& B : blocks_) {
        if (B.swa || n == 0 || !(q8 ? B.k_codes != nullptr : B.keys != nullptr)) continue;
        if (q8) {
            ck(cudaMemcpy((uint8_t*) B.k_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.k_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            ck(cudaMemcpy((uint8_t*) B.v_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.v_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            continue;
        }
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
        }
    }
}

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).  A position before the sequence's start has no run.
template <class F>
void Laguna::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

// What a window layer has of the sequence: the rows the next token attends (Flint's).
size_t Laguna::extra_bytes() const {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.swa) ++layers;
    return layers * 2 * (size_t) keep_ * (code_row + scale_row);
}

// A window layer's rows, behind the window layers before it (there is no recurrent state).
void Laguna::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (!blocks_[(size_t) il].swa) return;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    const size_t whole = 2 * (size_t) keep_ * (code_row + scale_row);
    dst += state_n_ * sizeof(float);
    for (int j = 0; j < il; ++j)
        if (blocks_[(size_t) j].swa) dst += whole;
    const Block& B = blocks_[(size_t) il];
    auto save_ring = [&](const uint8_t* cache, size_t one_row) {
        std::memset(dst, 0, (size_t) keep_ * one_row);   // the places of positions before the sequence's start
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpyAsync(dst + slot * one_row, cache + at * one_row, m * one_row, cudaMemcpyDeviceToHost,
                               cs(st_)),
               "save state");
        });
        dst += (size_t) keep_ * one_row;
    };
    if (q8) {
        save_ring((const uint8_t*) B.k_codes, code_row);
        save_ring((const uint8_t*) B.k_scales, scale_row);
        save_ring((const uint8_t*) B.v_codes, code_row);
        save_ring((const uint8_t*) B.v_scales, scale_row);
        return;
    }
    save_ring((const uint8_t*) B.keys, code_row);
    save_ring((const uint8_t*) B.vals, code_row);
}

void Laguna::load_extra(const uint8_t* src, int64_t pos) {
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t code_row = (size_t) n_kv_ * head_dim_ * (q8 ? 1 : sizeof(uint16_t));
    const size_t scale_row = q8 ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        auto load_ring = [&](uint8_t* cache, size_t one_row) {
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy(cache + at * one_row, src + slot * one_row, m * one_row, cudaMemcpyHostToDevice),
                   "load state");
            });
            src += (size_t) keep_ * one_row;
        };
        if (q8) {
            load_ring((uint8_t*) B.k_codes, code_row);
            load_ring((uint8_t*) B.k_scales, scale_row);
            load_ring((uint8_t*) B.v_codes, code_row);
            load_ring((uint8_t*) B.v_scales, scale_row);
            continue;
        }
        load_ring((uint8_t*) B.keys, code_row);
        load_ring((uint8_t*) B.vals, code_row);
    }
}

// The recordings of a full layer's head are numbered by the rows they are sized for; a window layer's never
// reads more than its window.
int Laguna::head_key(int il, int n) const { return blocks_[(size_t) il].swa ? 0 : room_key(n); }

// A side of a layer's cache: a full layer's at the positions, a window layer's at their places in the ring.
void Laguna::store(const float* x, const Block& B, int64_t pos, int n, bool value) {
    const int width = n_kv_ * head_dim_;
    if (opt_.kv_format == KvFmt::kQ8_0) {
        int8_t* codes = value ? B.v_codes : B.k_codes;
        uint16_t* scales = value ? B.v_scales : B.k_scales;
        if (recording_) return store_q8_at(x, codes, scales, width, at_, n, B.swa ? ring_ : 0, st_);
        if (!B.swa) return store_q8(x, codes, scales, width, pos, n, st_);
        const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
        store_q8(x, codes, scales, width, at, head, st_);
        if (head < n) store_q8(x + (size_t) head * width, codes, scales, width, 0, n - head, st_);
        return;
    }
    uint16_t* cache = value ? B.vals : B.keys;
    if (recording_) return store_f16_at(x, cache, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

// A head's norm and its turn, the layer's kind: a full layer of a scaled file by YaRN, else plain RoPE.
void Laguna::rope_heads(const Block& B, const float* x, const Dev& norm, float* y, int heads, int64_t pos, int rows,
                  const long long* at) {
    if (B.yarn)
        return head_norm_rope_yarn(x, head_dim_, fp(norm), y, head_dim_, heads, B.n_rot, B.theta, pos, rows, rms_eps_,
                                   st_, yarn_, at);
    head_norm_rope(x, head_dim_, fp(norm), y, head_dim_, heads, B.n_rot, B.theta, pos, rows, rms_eps_, st_, nullptr,
                   nullptr, 1, at);
}

// The attention's output times the softplus of the gate (gate_, computed from the layer's normed input next to the
// queries): a gate a head broadcast over its values, or one a value.
void Laguna::gate(const Block& B, int n) {
    if (B.n_gate == B.n_head)
        mul_softplus(att_, gate_, head_dim_, n * B.n_head, 1, true, st_);
    else
        mul_softplus(att_, gate_, B.n_head * head_dim_, n, B.n_gate, false, st_);
}

// The attention: queries and keys normed a head and turned by the position; the keys and values into the caches;
// each head's softmax over the rows (no sinks); the gate; the output matrix.
void Laguna::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    const long long* rot = at ? at + 1 : nullptr;       // and the turn each of them is given, read there too
    const int heads = B.n_head;
    lin(B.wq, xn_, qraw_, n, xq_);
    rope_heads(B, qraw_, B.q_norm, q_, heads, pos, n, rot);
    lin(B.wk, xn_, kraw_, n, xq_);
    rope_heads(B, kraw_, B.k_norm, k_, n_kv_, pos, n, rot);
    lin(B.wv, xn_, v_, n, xq_);
    lin(B.wg, xn_, gate_, n, xq_);
    store(k_, B, pos, n, false);
    store(v_, B, pos, n, true);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const Kvq8 kq = q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{};
    const Kvq8 vq = q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{};
    if (n >= kStageRows)
        attend_batch(blas_, q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, score_n_, awork_,
                     att_, head_dim_, heads, n_kv_, scale, st_, window, ring, false, kq, vq,
                     q8 ? (uint16_t*) q8w_ : nullptr, q8 ? (uint16_t*) q8w_ + work_halves_ : nullptr,
                     q8 ? work_halves_ : 0);
    else
        attend(q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, chunk_, att_, head_dim_, heads, n_kv_,
               scale, st_, window, ring, false, at,
               !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n), score_n_, kq, vq);
    gate(B, n);
    lin(B.wo, att_, ao_, n);
}

// step()'s rows, each of its own sequence: the products for all of them at once, and a row at a time the turn (it
// takes one position and counts on from it), the stored keys and values and the attention, in the row's sequence.
void Laguna::attn_rows(Block& B, int n) {
    const int heads = B.n_head;
    lin(B.wq, xn_, qraw_, n, xq_);
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    lin(B.wg, xn_, gate_, n, xq_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const size_t kv = (size_t) n_kv_ * head_dim_, hq = (size_t) heads * head_dim_;
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at_ + 2 * i;
        rope_heads(B, qraw_ + i * hq, B.q_norm, q_ + i * hq, heads, 0, 1, ai + 1);
        rope_heads(B, kraw_ + i * kv, B.k_norm, k_ + i * kv, n_kv_, 0, 1, ai + 1);
        uint16_t* kch = q8 ? nullptr : (uint16_t*) seq_buf(r.seq, B.keys);
        uint16_t* vch = q8 ? nullptr : (uint16_t*) seq_buf(r.seq, B.vals);
        const Kvq8 kq = q8 ? Kvq8{(int8_t*) seq_buf(r.seq, B.k_codes), (uint16_t*) seq_buf(r.seq, B.k_scales)}
                           : Kvq8{};
        const Kvq8 vq = q8 ? Kvq8{(int8_t*) seq_buf(r.seq, B.v_codes), (uint16_t*) seq_buf(r.seq, B.v_scales)}
                           : Kvq8{};
        if (q8) {
            store_q8_at(k_ + i * kv, (int8_t*) kq.codes, (uint16_t*) kq.scales, (int) kv, ai, 1, ring, st_);
            store_q8_at(v_ + i * kv, (int8_t*) vq.codes, (uint16_t*) vq.scales, (int) kv, ai, 1, ring, st_);
        } else {
            store_f16_at(k_ + i * kv, kch, (int) kv, ai, 1, ring, st_);
            store_f16_at(v_ + i * kv, vch, (int) kv, ai, 1, ring, st_);
        }
        const int64_t keep = pos_;
        pos_ = r.pos;
        const int sized = B.swa ? std::min(window_, opt_.max_context) : room(1);
        pos_ = keep;
        attend(q_ + i * hq, kch, vch, r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, heads, n_kv_, scale, st_,
               window, ring, false, ai, sized, score_n_, kq, vq);
    }
    gate(B, n);
    lin(B.wo, att_, ao_, n);
}

void Laguna::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// A layer from its input to its router's copy (a dense layer: to its end).  Pre-norms only, as llama.cpp's graph.
void Laguna::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);

    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

void Laguna::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Laguna::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights from the
// bare gates, made to sum to one (expert_weights_norm) and scaled (expert_weights_scale).
void Laguna::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
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
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool Laguna::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert, ungated: the routed experts' sum starts from its output.
void Laguna::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

}  // namespace strata::poly
