// tools/vision/own/vit_geometry.cpp - see vit_geometry.h
#include "vit_geometry.h"

#include <algorithm>
#include <cmath>

namespace strata_vit {

namespace {

std::string blk(int il, const char* what, const char* kind) {
    return "v.blk." + std::to_string(il) + "." + what + "." + kind;
}

bool need_u32(const Gguf& gg, const char* key, int& out, std::string& err) {
    uint32_t v = 0;
    if (!gg.get_u32(key, v) || v == 0 || v > (1u << 24)) { err = std::string("the vision projector has no usable ") + key; return false; }
    out = (int) v;
    return true;
}

// shape check: the tensor exists and has exactly these dimensions (as stored, fastest first)
bool shape_is(const Gguf& gg, const std::string& name, std::initializer_list<int64_t> ne, std::string& err) {
    const GgufTensorInfo* t = gg.tensor(name);
    if (!t) { err = "the vision projector lacks the tensor " + name; return false; }
    if (t->ne.size() != ne.size() || !std::equal(ne.begin(), ne.end(), t->ne.begin())) {
        std::string have;
        for (int64_t d : t->ne) have += (have.empty() ? "" : " x ") + std::to_string(d);
        err = "tensor " + name + " has shape [" + have + "], not what this tower's geometry needs";
        return false;
    }
    return true;
}

// Gemma 4's tower (mtmd's clip_graph_gemma4v and its hparams): RMS norms, separate q/k/v with no biases and a norm per head on q and
// k, a 2D NeoX rotary of theta 100, norms after the attention and after the FFN, a gated FFN, no patch bias, two learned position
// tables (x, then y), then an average pool of pool x pool tokens, the standardisation and one projection.  Pictures of 70 to 1,120
// tokens, sides a multiple of patch * pool.
bool gemma4_from_gguf(const Gguf& gg, VitGeometry& g, std::string& err) {
    g.norm = NormKind::RMSNorm;
    g.fused_qkv = false;
    g.gated_ffn = true;
    g.patch_taps = 1;
    g.rope_theta = 100.0f;
    g.merge = 3;
    uint32_t u = 0;
    if (gg.get_u32("clip.vision.projector.scale_factor", u) && u > 0 && u <= 16) g.merge = (int) u;
    g.pool = g.merge;
    g.min_tokens = 70;
    g.max_tokens = 1120;
    const int64_t W = g.width, F = g.ffn, HD = g.head_dim;
    if (!shape_is(gg, "v.patch_embd.weight", {g.patch, g.patch, 3, W}, err)) return false;
    if (gg.tensor("v.patch_embd.bias")) { err = "a gemma4v patch embedding with a bias, which this tower does not read"; return false; }
    const GgufTensorInfo* pos = gg.tensor("v.position_embd.weight");
    if (!pos || pos->ne.size() != 3 || pos->ne[0] != W || pos->ne[2] != 2) {
        err = "the gemma4v position tables are missing or not [width, positions, 2]";
        return false;
    }
    g.pos_grid = (int) pos->ne[1];   // positions a side the tables have
    for (int il = 0; il < g.layers; ++il) {
        for (const char* n : {"ln1", "ln2", "attn_post_norm", "ffn_post_norm"})
            if (!shape_is(gg, blk(il, n, "weight"), {W}, err)) return false;
        for (const char* n : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
            if (!shape_is(gg, blk(il, n, "weight"), {W, W}, err)) return false;
            if (gg.tensor(blk(il, n, "bias"))) { err = "a gemma4v attention with biases, which this tower does not read"; return false; }
        }
        if (!shape_is(gg, blk(il, "attn_q_norm", "weight"), {HD}, err) || !shape_is(gg, blk(il, "attn_k_norm", "weight"), {HD}, err)) return false;
        if (!shape_is(gg, blk(il, "ffn_gate", "weight"), {W, F}, err) || !shape_is(gg, blk(il, "ffn_up", "weight"), {W, F}, err) ||
            !shape_is(gg, blk(il, "ffn_down", "weight"), {F, W}, err))
            return false;
    }
    // Gemma4ClippableLinear's bounds, scalars beside a matrix (mtmd's clamp_info_map): not read here, so a file with them is refused
    for (const char* n : {"mm.input_projection.input_max", "v.blk.0.attn_q.input_max", "v.blk.0.ffn_up.input_max"})
        if (gg.tensor(n)) { err = "a gemma4v tower with clipped linear layers, which this tower does not read"; return false; }
    g.std_norm = gg.tensor("v.std_bias") != nullptr && gg.tensor("v.std_scale") != nullptr;
    if (g.std_norm && (!shape_is(gg, "v.std_bias", {W}, err) || !shape_is(gg, "v.std_scale", {W}, err))) return false;
    return shape_is(gg, "mm.input_projection.weight", {W, g.out_dim}, err);
}

// MiniCPM-V 4.7's tower (mtmd's clip_graph_minicpmv4_6 with n_merge 4, llama.cpp 69f201a2): a SigLIP ViT (LayerNorm, q / k / v with
// biases, a tanh-GELU MLP, a patch bias) with a learned position table of side pos_grid read through buckets; after layer insert_layer a
// window merger (attention inside each 2x2 window, then the 2x2 windows' four rows concatenated through a LayerNorm and an MLP, plus
// their mean); the remaining layers on the quarter grid; a post norm; a final 2x2 merger (LayerNorm, MLP with the erf GELU).  The text
// separators between a picture's tiles are rows of v.tok_embd_sep.
bool minicpmv47_from_gguf(const Gguf& gg, VitGeometry& g, std::string& err) {
    g.fused_qkv = false;
    g.patch_taps = 1;
    g.merge = 4;
    uint32_t u = 0;
    if (gg.get_u32("clip.vision.projector.scale_factor", u)) g.merge = (int) u;
    if (g.merge != 4) { err = "a minicpmv4_7 tower without the window merger (scale factor " + std::to_string(g.merge) + "), which this tower does not read"; return false; }
    std::vector<int64_t> wa;
    if (!gg.get_int_array("clip.vision.wa_layer_indexes", wa) || wa.empty() || wa[0] < 0 || wa[0] >= g.layers - 1) {
        err = "the minicpmv4_7 tower names no usable layer for its window merger (clip.vision.wa_layer_indexes)";
        return false;
    }
    g.insert_layer = (int) wa[0];
    g.slice_size = g.image_size;
    if (g.slice_size <= 0 || g.slice_size % g.align() != 0) { err = "the minicpmv4_7 tile size (clip.vision.image_size) is not a multiple of patch * 4"; return false; }
    if (gg.get_u32("clip.vision.max_slice_nums", u) && u > 0 && u <= 64) g.max_slices = (int) u;
    if (gg.tensor("v.pre_ln.weight")) { err = "a minicpmv4_7 tower with a norm before its first layer, which this tower does not read"; return false; }
    const int64_t W = g.width, F = g.ffn, W4 = 4 * W;
    if (!shape_is(gg, "v.patch_embd.weight", {g.patch, g.patch, 3, W}, err) || !shape_is(gg, "v.patch_embd.bias", {W}, err)) return false;
    const GgufTensorInfo* pos = gg.tensor("v.position_embd.weight");
    if (!pos || pos->ne.size() != 2 || pos->ne[0] != W) { err = "the vision projector's position table is missing or not [width, rows]"; return false; }
    g.pos_grid = (int) std::lround(std::sqrt((double) pos->ne[1]));
    if ((int64_t) g.pos_grid * g.pos_grid != pos->ne[1]) { err = "the position table's rows are not a square grid"; return false; }
    auto ln = [&](const std::string& p, int64_t n) { return shape_is(gg, p + ".weight", {n}, err) && shape_is(gg, p + ".bias", {n}, err); };
    auto lin = [&](const std::string& p, int64_t in, int64_t out) { return shape_is(gg, p + ".weight", {in, out}, err) && shape_is(gg, p + ".bias", {out}, err); };
    for (int il = 0; il < g.layers; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        if (!ln(p + "ln1", W) || !ln(p + "ln2", W) || !lin(p + "attn_q", W, W) || !lin(p + "attn_k", W, W) || !lin(p + "attn_v", W, W) ||
            !lin(p + "attn_out", W, W) || !lin(p + "ffn_up", W, F) || !lin(p + "ffn_down", F, W))
            return false;
        if (gg.tensor(p + "ffn_gate.weight") || gg.tensor(p + "ls1.weight") || gg.tensor(p + "ls2.weight")) {
            err = "a minicpmv4_7 layer with a gate or layer scales, which this tower does not read";
            return false;
        }
    }
    const GgufTensorInfo* ds = gg.tensor("v.vit_merger.ds_ffn_up.weight");
    if (!ds || ds->ne.size() != 2) { err = "the minicpmv4_7 tower lacks v.vit_merger.ds_ffn_up.weight"; return false; }
    g.ds_ffn = (int) ds->ne[1];
    if (!ln("v.vit_merger.ln1", W) || !lin("v.vit_merger.attn_q", W, W) || !lin("v.vit_merger.attn_k", W, W) || !lin("v.vit_merger.attn_v", W, W) ||
        !lin("v.vit_merger.attn_out", W, W) || !ln("v.vit_merger.ds_ln", W4) || !lin("v.vit_merger.ds_ffn_up", W4, g.ds_ffn) ||
        !lin("v.vit_merger.ds_ffn_down", g.ds_ffn, W) || !ln("v.post_ln", W) || !ln("mm.input_norm", W4))
        return false;
    const GgufTensorInfo* up = gg.tensor("mm.up.weight");
    if (!up || up->ne.size() != 2 || up->ne[0] != W4) { err = "the minicpmv4_7 tower's mm.up.weight is missing or not [width * 4, hidden]"; return false; }
    g.merger_in = (int) W4;
    g.merger_hidden = (int) up->ne[1];
    if (!lin("mm.up", W4, g.merger_hidden) || !lin("mm.down", g.merger_hidden, g.out_dim)) return false;
    const GgufTensorInfo* sep = gg.tensor("v.tok_embd_sep");
    if (!sep || sep->ne.size() != 2 || sep->ne[0] != g.out_dim || sep->ne[1] < 4) { err = "the minicpmv4_7 tower's separators (v.tok_embd_sep) are missing or not [width, 4]"; return false; }
    g.seps = (int) sep->ne[1];
    return true;
}

// Inkling 2 Mini's tower (llama.cpp's mtmd_graph_inkling2v and its hparams): patches without a bias, a learned position table of side pos_grid
// resized bilinearly to the picture's patch grid, a CLS token in front (its own embedding plus its own position), a LayerNorm before the first
// layer, then LayerNorm layers with q / k / v / o biases, a 2D rotary (the first half of a head by the column, the second by the row, adjacent
// pairs, positions counted from 1 with the CLS token at 0), layer scales on the attention's and the MLP's outputs and an erf-GELU MLP; the
// CLS row dropped, every 2x2 patches merged into one row, then mm.1, the erf GELU and mm.2.  Pictures of 1 to 4,096 tokens, sides a multiple
// of patch * merge, padded and not resized (image_prep.h).
bool inkling2v_from_gguf(const Gguf& gg, VitGeometry& g, std::string& err) {
    g.norm = NormKind::LayerNorm;
    g.act = ActKind::GeluErf;
    g.fused_qkv = false;
    g.gated_ffn = false;
    g.patch_taps = 1;
    g.pre_ln = true;
    g.min_tokens = 1;
    g.max_tokens = 4096;
    const int64_t W = g.width, F = g.ffn;
    if (!shape_is(gg, "v.patch_embd.weight", {g.patch, g.patch, 3, W}, err)) return false;
    if (gg.tensor("v.patch_embd.bias")) { err = "an inkling2v patch embedding with a bias, which this tower does not read"; return false; }
    const GgufTensorInfo* pos = gg.tensor("v.position_embd.weight");
    if (!pos || pos->ne.size() != 2 || pos->ne[0] != W) { err = "the vision projector's position table is missing or not [width, rows]"; return false; }
    g.pos_grid = (int) std::lround(std::sqrt((double) pos->ne[1]));
    if ((int64_t) g.pos_grid * g.pos_grid != pos->ne[1]) { err = "the position table's rows are not a square grid"; return false; }
    if (!shape_is(gg, "v.class_embd", {W}, err) || !shape_is(gg, "v.class_position_embd.weight", {W}, err) ||
        !shape_is(gg, "v.pre_ln.weight", {W}, err) || !shape_is(gg, "v.pre_ln.bias", {W}, err))
        return false;
    g.post_ln = gg.tensor("v.post_ln.weight") != nullptr;
    if (g.post_ln && (!shape_is(gg, "v.post_ln.weight", {W}, err) || !shape_is(gg, "v.post_ln.bias", {W}, err))) return false;
    auto vec = [&](const std::string& n, int64_t len) { return shape_is(gg, n, {len}, err); };
    for (int il = 0; il < g.layers; ++il) {
        for (const char* n : {"ln1", "ln2"})
            if (!vec(blk(il, n, "weight"), W) || !vec(blk(il, n, "bias"), W)) return false;
        for (const char* n : {"attn_q", "attn_k", "attn_v", "attn_out"})
            if (!shape_is(gg, blk(il, n, "weight"), {W, W}, err) || !vec(blk(il, n, "bias"), W)) return false;
        if (!vec(blk(il, "ls1", "weight"), W) || !vec(blk(il, "ls2", "weight"), W)) return false;
        if (!shape_is(gg, blk(il, "ffn_up", "weight"), {W, F}, err) || !vec(blk(il, "ffn_up", "bias"), F) ||
            !shape_is(gg, blk(il, "ffn_down", "weight"), {F, W}, err) || !vec(blk(il, "ffn_down", "bias"), W))
            return false;
        if (gg.tensor(blk(il, "ffn_gate", "weight"))) { err = "an inkling2v layer with a gated MLP, which this tower does not read"; return false; }
    }
    g.merger_in = (int) (W * g.merge * g.merge);
    const GgufTensorInfo* m1 = gg.tensor("mm.1.weight");
    if (!m1 || m1->ne.size() != 2 || m1->ne[0] != g.merger_in) { err = "the inkling2v tower's mm.1.weight is missing or not [width * merge^2, hidden]"; return false; }
    g.merger_hidden = (int) m1->ne[1];
    if (!shape_is(gg, "mm.2.weight", {g.merger_hidden, g.out_dim}, err)) return false;
    g.mm_bias = gg.tensor("mm.1.bias") != nullptr;
    if (g.mm_bias != (gg.tensor("mm.2.bias") != nullptr)) { err = "the inkling2v tower has a bias on one of mm.1 and mm.2 only"; return false; }
    if (g.mm_bias && (!vec("mm.1.bias", g.merger_hidden) || !vec("mm.2.bias", g.out_dim))) return false;
    return true;
}

}  // namespace

bool vit_geometry_from_gguf(const Gguf& gg, VitGeometry& g, std::string& err) {
    g = VitGeometry{};
    std::string arch;
    if (!gg.get_str("general.architecture", arch) || arch != "clip") { err = "not an mmproj file (general.architecture is not clip)"; return false; }
    bool has_v = false;
    if (!gg.get_bool("clip.has_vision_encoder", has_v) || !has_v) { err = "the mmproj file has no vision encoder"; return false; }
    // (newer files name it per modality, as mtmd reads it too)
    if ((!gg.get_str("clip.projector_type", g.projector) || g.projector.empty()) &&
        (!gg.get_str("clip.vision.projector_type", g.projector) || g.projector.empty())) {
        err = "the mmproj file names no projector type";
        return false;
    }
    if (!need_u32(gg, "clip.vision.block_count", g.layers, err) || !need_u32(gg, "clip.vision.embedding_length", g.width, err) ||
        !need_u32(gg, "clip.vision.attention.head_count", g.heads, err) || !need_u32(gg, "clip.vision.feed_forward_length", g.ffn, err) ||
        !need_u32(gg, "clip.vision.patch_size", g.patch, err) || !need_u32(gg, "clip.vision.projection_dim", g.out_dim, err))
        return false;
    if (g.width % g.heads != 0) { err = "the vision projector's width is not a multiple of its head count"; return false; }
    g.head_dim = g.width / g.heads;
    uint32_t u = 0;
    if (gg.get_u32("clip.vision.image_size", u)) g.image_size = (int) u;
    if (gg.get_u32("clip.vision.spatial_merge_size", u) && u > 0 && u <= 16) g.merge = (int) u;
    gg.get_f32("clip.vision.attention.layer_norm_epsilon", g.eps);
    std::vector<float> v3;
    if (gg.get_f32_array("clip.vision.image_mean", v3) && v3.size() == 3) for (int i = 0; i < 3; ++i) g.mean[i] = v3[i];
    if (gg.get_f32_array("clip.vision.image_std", v3) && v3.size() == 3) for (int i = 0; i < 3; ++i) g.stdv[i] = v3[i];
    if (gg.get_u32("clip.vision.n_wa_pattern", u)) g.window_pattern = (int) u;

    bool use_gelu = false, use_silu = false;
    gg.get_bool("clip.use_gelu", use_gelu);
    gg.get_bool("clip.use_silu", use_silu);
    g.act = use_gelu ? ActKind::GeluTanh : use_silu ? ActKind::Silu : ActKind::GeluQuick;   // llama.cpp's own default when neither is set
    g.norm = g.projector == "qwen2.5vl_merger" ? NormKind::RMSNorm : NormKind::LayerNorm;
    if (g.projector == "gemma4v") return gemma4_from_gguf(gg, g, err);
    if (g.minicpmv()) return minicpmv47_from_gguf(gg, g, err);
    if (g.inkling()) return inkling2v_from_gguf(gg, g, err);

    // the layer layout, from which tensors exist
    g.fused_qkv = gg.tensor(blk(0, "attn_qkv", "weight")) != nullptr;
    g.gated_ffn = gg.tensor(blk(0, "ffn_gate", "weight")) != nullptr;
    g.patch_taps = gg.tensor("v.patch_embd.weight.1") != nullptr ? 2 : 1;
    g.pre_ln = gg.tensor("v.pre_ln.weight") != nullptr;
    std::vector<int64_t> ds;
    if (gg.get_int_array("clip.vision.is_deepstack_layers", ds))
        for (size_t i = 0; i < ds.size(); ++i) if (ds[i]) g.deepstack.push_back((int) i);
    for (int il = 0; il < g.layers; ++il) {
        if (gg.tensor("v.deepstack." + std::to_string(il) + ".fc1.weight") && std::find(g.deepstack.begin(), g.deepstack.end(), il) == g.deepstack.end())
            g.deepstack.push_back(il);
    }

    const int64_t W = g.width, F = g.ffn;
    if (!shape_is(gg, "v.patch_embd.weight", {g.patch, g.patch, 3, W}, err) || !shape_is(gg, "v.patch_embd.bias", {W}, err)) return false;
    if (g.patch_taps == 2 && !shape_is(gg, "v.patch_embd.weight.1", {g.patch, g.patch, 3, W}, err)) return false;
    const GgufTensorInfo* pos = gg.tensor("v.position_embd.weight");
    if (!pos || pos->ne.size() != 2 || pos->ne[0] != W) { err = "the vision projector's position table is missing or not [width, rows]"; return false; }
    g.pos_grid = (int) std::lround(std::sqrt((double) pos->ne[1]));
    if ((int64_t) g.pos_grid * g.pos_grid != pos->ne[1]) { err = "the position table's rows are not a square grid"; return false; }
    for (int il = 0; il < g.layers; ++il) {
        for (const char* n : {"ln1", "ln2"})
            if (!shape_is(gg, blk(il, n, "weight"), {W}, err) || !shape_is(gg, blk(il, n, "bias"), {W}, err)) return false;
        if (g.fused_qkv) {
            if (!shape_is(gg, blk(il, "attn_qkv", "weight"), {W, 3 * W}, err) || !shape_is(gg, blk(il, "attn_qkv", "bias"), {3 * W}, err)) return false;
        } else {
            for (const char* n : {"attn_q", "attn_k", "attn_v"})
                if (!shape_is(gg, blk(il, n, "weight"), {W, W}, err)) return false;
        }
        if (!shape_is(gg, blk(il, "attn_out", "weight"), {W, W}, err) || !shape_is(gg, blk(il, "attn_out", "bias"), {W}, err)) return false;
        if (!shape_is(gg, blk(il, "ffn_up", "weight"), {W, F}, err) || !shape_is(gg, blk(il, "ffn_up", "bias"), {F}, err)) return false;
        if (!shape_is(gg, blk(il, "ffn_down", "weight"), {F, W}, err) || !shape_is(gg, blk(il, "ffn_down", "bias"), {W}, err)) return false;
        if (g.gated_ffn && !shape_is(gg, blk(il, "ffn_gate", "weight"), {W, F}, err)) return false;
    }
    if (!shape_is(gg, "v.post_ln.weight", {W}, err) || !shape_is(gg, "v.post_ln.bias", {W}, err)) return false;
    g.merger_in = (int) (W * g.merge * g.merge);
    const GgufTensorInfo* m0 = gg.tensor("mm.0.weight");
    if (!m0 || m0->ne.size() != 2 || m0->ne[0] != g.merger_in) { err = "the vision projector's mm.0.weight is missing or not [width * merge^2, hidden]"; return false; }
    g.merger_hidden = (int) m0->ne[1];
    if (!shape_is(gg, "mm.0.bias", {g.merger_hidden}, err) || !shape_is(gg, "mm.2.weight", {g.merger_hidden, g.out_dim}, err) ||
        !shape_is(gg, "mm.2.bias", {g.out_dim}, err))
        return false;
    return true;
}

bool vit_geometry_qwen(const VitGeometry& g, std::string& why) {
    if (g.projector == "qwen3vl_merger") return true;
    why = "the " + g.projector + " tower is read by the CPU picture reader only";
    return false;
}

bool vit_geometry_supported(const VitGeometry& g, std::string& why) {
    if (g.projector == "gemma4v") {
        if (g.head_dim % 4 != 0) { why = "a head size that is not a multiple of 4 (the 2D rotary turns half a head by each coordinate)"; return false; }
        if (g.pool < 1) { why = "no pool size"; return false; }
        return true;
    }
    if (g.minicpmv()) {
        if (g.act != ActKind::GeluTanh) { why = "an activation other than the tanh GELU"; return false; }
        return true;
    }
    if (g.inkling()) {
        if (g.head_dim % 4 != 0) { why = "a head size that is not a multiple of 4 (the 2D rotary turns pairs in each half of a head)"; return false; }
        if (g.merge < 1 || g.pos_grid < 1) { why = "no merge size or position table"; return false; }
        return true;
    }
    if (g.projector != "qwen3vl_merger") { why = "projector type " + g.projector + " is not one of this tower's families"; return false; }
    if (g.norm != NormKind::LayerNorm) { why = "RMS norm"; return false; }
    if (g.act != ActKind::GeluTanh) { why = "an activation other than the tanh GELU"; return false; }
    if (g.gated_ffn) { why = "a gated FFN"; return false; }
    if (!g.fused_qkv) { why = "separate q/k/v projections"; return false; }
    if (g.window_pattern != 0) { why = "window attention"; return false; }
    if (g.pre_ln) { why = "a norm before the first layer"; return false; }
    if (!g.deepstack.empty()) { why = "deepstack outputs"; return false; }
    if (g.head_dim % 4 != 0) { why = "a head size that is not a multiple of 4 (the 2D rotary splits it in 4)"; return false; }
    if (g.merge != 2) { why = "a merge size other than 2"; return false; }
    return true;
}

}  // namespace strata_vit
