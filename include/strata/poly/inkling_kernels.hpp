// include/strata/poly/inkling_kernels.hpp - the kernels Inkling 2 Mini (llama.cpp architecture "inkling2_loop",
// ggml-org/llama.cpp 29b52d398b, src/models/inkling2.cpp and inkling2-loop.cpp) needs beyond kernels.hpp's.
//
// Every call enqueues on the given CUDA stream and returns.  Pointers are device pointers.  A call works on `rows`
// tokens at consecutive positions, a buffer of per-token values holding them row after row (kernels.hpp's rule).
// ggml shapes are written [ne0, ne1, ...]: ne0 is the contiguous one, so ggml [a, b] is b rows of a floats.
//
// ============================================================================================================
// THE MATH (derived from upstream at 29b52d398b; file:line are that commit's)
// ============================================================================================================
//
// clip(x, s, c)   (inkling2.cpp graph::clip)
//     v = x * s (s per channel, optional);  n = sqrt(sum v^2) + 1e-6;  y = v * clamp(c / n, 0, 1).
//     Per head: Q with s = attn_q_clip [128, 24] and c = qk_clip (22.627417), K with no s and c = qk_clip,
//     R with s = attn_rel_clip [16, 24] and c = relative_clip (8).  V is rms_norm(v, eps = f_norm_rms_eps) per head
//     without a weight.
//
// tau (full-attention layers only; inkling2.cpp build_inputs)
//     tau(p) = 1 + alpha * log(max(1, (p + 1) * (1 / floor))),  floor = log_scale_floor (8192), alpha = 0.1,
//     p the token's position.  Q and R (after their clips) are multiplied by tau.  SWA layers: no tau.
//
// attention   (inkling2.cpp build_layer; llama-graph.cpp:4436-4515; llama-kv-cache.cpp:2010-2022)
//     For query token i (position p) and head h, against the cached key at position j (group g = h / 3):
//         d = p - j                                              (llama-kv-cache.cpp:2017)
//         idx = (0 <= d < extent) ? d : extent                   (llama-kv-cache.cpp:2018; an empty cell is d = -1,
//                                                                 so idx = extent too, and the mask hides it)
//         table = mul_mat(pad(attn_rel_bias, +1 row of zeros), R): bias(i,h,e) = sum_r rel_bias[e][r] * R[i][h][r]
//                 for e < extent, and 0 for e = extent         (the padded row;
//                                                                 inkling2.cpp "ggml_pad(..., 0, 1, ...)")
//         logit(i,h,j) = scale * (Q[i][h] . K[j][g]) + bias(i,h,idx) - sink[i][h]   + mask(i,j)
//         the softmax has one more logit, 0, with no value (the "zero sink" passed to build_attn)
//     scale = 1/sqrt(128).  The mask (llama-hparams.h:563-575, LLAMA_SWA_TYPE_STANDARD) hides j > p and, on SWA
//     layers, p - j >= n_swa (1024): a SWA token attends the 1024 positions p-1023..p, itself among them.  Full
//     layers attend 0..p; distances >= extent (1024) get bias 0.  extent = relative_extent on full layers and n_swa
//     on SWA layers (attn_rel_bias is [16, extent] either way; both are 1024 in the released GGUF).
//     Since exp(s - t) / (1 + sum exp(s - t)) = exp(s) / (exp(t) + sum exp(s)), this is the same softmax as
//         logit(i,h,j) = scale * Q.K + bias(i,h,idx),  with one extra logit t = sink[i][h] that weighs no value,
//     which is what inkling_attend() computes (as Flint's sinks, but per query token: sink = attn_sink_proj(x_q),
//     [24] a token, no scale).  The bias is added after the 1/sqrt(128) scale (ggml_soft_max_ext / flash attn add
//     the mask to the scaled product).  Upstream with flash attention casts the summed mask (bias - sink) to F16; we
//     keep it in float.
//     Then out *= sigmoid(attn_gate(x_q)) per value (mul_sigmoid), then wo, then the post-conv.
//
// convolve(x, w, taps)   (inkling2.cpp graph::convolve; ggml_ssm_conv = ggml-cpu/ops.cpp:9942-9993)
//     n_conv = 4.  xx is the channel's inputs, the 3 stored past ones (oldest first) and then the call's.
//         y_t = sum_{k=0..3} w[c][k] * xx[t - 3 + k]            (ssm_conv: w[c][3] is the current token's tap)
//         with taps (full layers' K and V; taps = attn_{k,v}_conv_proj(x_{k,v}) [4 * 8] a token, head-major):
//         y_t += sum_{lag=0..3} taps[t][h][lag] * xx[t - lag]   (h = c / 128;  NOTE the other orientation: lag 0
//                                                                 is the current token)
//         out_t = (x_t + y_t) * 1/sqrt((taps ? 2 : 1) * 4 + 1)    (1/3 with taps, 1/sqrt(5) without)
//     The state keeps the last 3 inputs x (before the convolution), per channel, oldest first.  Upstream keeps the
//     four convolutions of a layer in one state at channel offsets K 0, V 1024, attn-out 2048, ffn-out 2048 + 2304
//     (n_sconv_channels 6656); here each call takes the pointer to its own part (state + offset * 3).
//     K conv input: wk(x_k) (1024), V: wv(x_v) (1024), attn-out: wo(gated attention) (2304), ffn-out: the MoE
//     output (2304).  The clip / rms of K and V come after their convolutions.
//
// mix(blocks, cur, query)   (inkling2.cpp graph::mix)
//     S = {blocks..., cur}, each rms_norm'ed without weight, eps 1e-6: s_hat = s / sqrt(mean(s^2) + 1e-6).
//     score[q][k] = (s_hat_k . query[q]) / sqrt(n_embd);  w = softmax over k;  out[q] = sum_k w[q][k] * s_hat_k.
//     The output mixes the NORMED states.  The order of S does not matter (upstream concatenates in reverse).
//     Queries: attn_res_query [2304, 1] on SWA layers (one input for Q, K and V), [2304, 3] on full layers (Q, K, V
//     inputs), ffn_res_query [2304], output_res_query [2304], loop_msg_query [2304, 4] (Q, K, V, FFN messages).
//
// control(cur, msg)   (inkling2.cpp graph::control)
//     z = [rms_norm(cur), rms_norm(msg)] (eps 1e-6, no weights, 2 * 2304);  r = ctrl_down z (64);
//     gain = tanh(ctrl_gain r), forget = tanh(ctrl_forget r) (2304 each);  out = (cur - forget * cur) + gain * msg.
//
// SiTU   (llama-graph.cpp:2666-2680; the dense build_ffn path aborts, only the MoE path has it)
//     gate_up row = [gate (n_ff), up (n_ff)] (the fused ffn_gate_up_exps output, gate first, llama-graph.cpp:2580ff)
//     act = beta * tanh(gate * (1 / beta)) * sigmoid(gate);  up' = lb > 0 ? lb * tanh(up * (1 / lb)) : up;
//     out = act * up'.   beta = situ_beta (4), lb = situ_linear_beta (25).
//
// router   (llama-graph.cpp:2420-2560): probs = sigmoid(logits) (logits in F32), pick top-8 of probs + exp_probs_b,
//     weights = probs of the picked, divided by (sum + 1e-20) (expert_weights_norm_eps), times expert_weights_scale
//     (8).  kernels.hpp route_rows(gate 1, bias exp_probs_b, norm true, w_scale 8) does this, with a floor of
//     6.1e-5 under the sum instead of + 1e-20: the same for any real sum of 8 sigmoids.
//
// ============================================================================================================
// WHICH KERNEL DOES WHAT   (src/poly/inkling.cpp; "few tokens" is up to Inkling::kFusedRows, 16)
// ============================================================================================================
//   tok_norm, output_norm                                      rms_norm() (kernels.hpp)
//   attn_norm, q/k/v_inp_norm, ffn_norm                        native_add_rms_norm_q8_1 (native_mmvq.hpp), which
//                                                              also gives the projections their Q8_1 input
//   wq, wk, wv, wo, attn_gate, attn_rel, sink_proj, conv_proj  the engine's lin()
//   ffn_gate_inp (F32)                                         inkling_gemv() for a few tokens, else the engine's
//                                                              router_scores()
//   loop ctrl_down / gain / forget (F32)                       inkling_control_fused() for a few tokens, else
//                                                              inkling_control()
//   K and V: conv, K clip, V rms norm, into the cache          inkling_kv_store()
//                                                              (rings as Flint's: row of position p at p % ring)
//   Q and R clips, tau                                         inkling_clip_many()
//   attention with the relative bias and per-token sinks       inkling_attend()
//   output gate                                                mul_sigmoid(att, gate, 3072, rows, 3072, false)
//   attn-out conv: a block's first layer (the stream afresh)   inkling_conv()
//   attn-out conv and residual add: the other layers           inkling_conv_acc()
//   ffn-out conv and residual add                              inkling_conv_acc()
//   depth mix (attn/ffn/output residual queries, messages)     inkling_mix()
//   router                                                     route_rows(gate 1, exp_probs_b, norm, 8.0)
//   experts' gate_up -> activation                             situ_rows() (the engine's act_rows(); the CPU
//                                                              experts have their own)
//   expert sum                                                 the engine's moe_sum(), from zero (no shared expert)
//   final softcap (30)                                         softcap()
//   step()'s rows (--batch), each of its own sequence: K/V,    inkling_kv_store_seqs(), inkling_clip_many_seqs(),
//   tau, attention and the post-convolutions with each row's   inkling_attend_seqs(), inkling_conv_seqs(),
//   caches, state and position; the rest as above, once       inkling_conv_acc_seqs()
//   for all the rows
#pragma once

#include "strata/poly/kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::poly {

// ---- clips ---------------------------------------------------------------------------------------------------

/// The log scale of the full-attention layers: tau(p) = 1 + alpha * log(max(1, (p + 1) * (1 / floor))).  A
/// default-constructed one (floor 0) is off: tau = 1.
struct InklingTau {
    float alpha = 0.f;
    int floor = 0;
};

/// clip() for `rows` tokens of `heads` heads of `dim` values: y = v * min(1, threshold / (|v| + 1e-6)), v = x * scale
/// (scale: heads * dim values, head-major as ggml's [dim, heads]; null: none), then times tau(the token's position)
/// when tau.floor > 0.  A token's input starts x_stride floats after the one before it (0: heads * dim); y has
/// heads * dim values a token.  y may be x when x_stride is 0 or heads * dim.  Token t is at position pos + t.
/// dim <= 1024.
void inkling_clip(const float* x, int x_stride, const float* scale, float* y, int dim, int heads, int rows,
                  float threshold, InklingTau tau, int64_t pos, void* stream);

// ---- attention -----------------------------------------------------------------------------------------------

/// The relative-position bias of a layer: r is the clipped (and on full layers tau-scaled) R, n_head * n_rel
/// floats a token (head-major); table is attn_rel_bias as stored, extent rows of n_rel floats (ggml [n_rel, extent]).
/// n_rel <= 32.
struct InklingRel {
    const float* r = nullptr;
    const float* table = nullptr;
    int n_rel = 16;
    int extent = 1024;
};

/// The floats inkling_attend()'s `work` needs for its split path: calls of up to kInklingSplitRows tokens of
/// n_head heads.  0 work (null) is allowed: every call then takes the unsplit path.
inline constexpr int kInklingSplitRows = 16;   ///< calls of more tokens never split (a prompt batch)
/// a call's keys go in at most this many splits (+1 for a window's offset)
inline constexpr int kInklingMaxSplits = 64;
size_t inkling_attend_work_floats(int n_head, int dim);
/// calls of at least this many tokens take the tensor-core path on cards that have it (Turing or newer, CUDA): every
/// call of more than kInklingSplitRows tokens (on an RTX 2070 Super, 17 tokens at position 8192 took 0.88 ms on the
/// tensor cores against 5.2 ms unsplit in a full layer, 0.15 against 0.74 ms in a window layer)
inline constexpr int kInklingMmaRows = kInklingSplitRows + 1;

/// Inkling's attention for the `rows` tokens from position `pos`: token t (position p = pos + t) of head h attends
/// the cached rows of positions max(0, p - window + 1) .. p (window 0: 0 .. p) with
///     logit(j) = scale * q[t][h] . k[j][h / (n_head / kv_heads)] + bias(t, h, p - j),
///     bias(t, h, d) = d < rel.extent ? sum_r rel.table[d * n_rel + r] * rel.r[(t * n_head + h) * n_rel + r] : 0,
/// and one more logit, sinks[t * n_head + h], that weighs no value (sinks may be null: none).  out: n_head * dim a
/// token.  The keys and values must already be in the caches (store them first, as Flint does).
/// Caches: fp16, kv_heads * dim halves a row (kcache / vcache), or Q8_0 where kq8 / vq8 codes are not null (the
/// fp16 pointer is then ignored), in kernels.hpp's Kvq8 layout.  With `ring`, position j is row j % ring; ring must
/// hold window + rows - 1 rows (Flint's ring_ = window + rows_ does).
/// dim must be 128 and n_head / kv_heads 3 (Inkling's 24 / 8); else std::invalid_argument.
///
/// Three paths, the same math; the first two float throughout (the caches' halves are the only rounding):
///  - split (rows <= kInklingSplitRows and work != null): the keys go in splits of S keys counted from position 0,
///    S = max(32, keys / kInklingMaxSplits rounded up to 32), keys = room (a window layer: min(room, window)),
///    one block a split, group and token (each of its warps four keys at a time, no shared-memory tiles), merged by
///    a second launch when there is more than one split (with one, the block writes the output itself).  A token's
///    result then does not depend on the tokens beside it in the call (for the same room).  `room` is the most
///    positions + 1 a token of the call can have: the launch is sized for it (as attend()'s room); 0 means
///    pos + rows.  Calls given the same room split the same way.  q and the caches must be 16- and 8-byte aligned
///    (cudaMalloc'd rows of 128 values are).
///  - unsplit: one block walks all the keys of a tile of up to 8 tokens.
///  - tensor cores (rows >= kInklingMmaRows, rel.n_rel 16, a CUDA card of Turing or newer; else unsplit): a prompt
///    batch.  One block walks all the keys of 21 tokens of a group; the queries, R, the softmax's weights and the bias
///    table's rows are rounded to halves, the caches' values too (a Q8_0 code times its scale), and the products sum
///    floats (llama.cpp's flash attention rounds Q, P and V the same way); the bias stays a float.  A token's result
///    does not depend on the tokens beside it in the call.
/// Throws std::runtime_error on a launch error.
void inkling_attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, Kvq8 kq8, Kvq8 vq8,
                    InklingRel rel, const float* sinks, int64_t pos, int rows, float* out, int dim, int n_head,
                    int kv_heads, float scale, int window, int ring, float* work, size_t work_floats, void* stream,
                    int64_t room = 0);

// ---- short convolutions --------------------------------------------------------------------------------------

/// convolve() over `channels` channels with 4 taps: out_t = (x_t + y_t) / sqrt(taps ? 9 : 5), see the top.  x and
/// out: channels values a token (out must not be x).  w: attn_*_conv / ffn_post_conv, (4, channels) as ggml
/// [4, channels] (w[c * 4 + k], k = 3 the current token).  taps: null, or `heads` * 4 values a token (the
/// conv_proj output, taps[t * heads * 4 + h * 4 + lag], lag 0 the current token), channel c in head
/// c / (channels / heads).  state: 3 past inputs a channel, oldest first, advanced here by `rows` tokens.
void inkling_conv(const float* x, float* state, const float* w, const float* taps, int heads, float* out,
                  int channels, int rows, void* stream);

/// A window's snapshots of a convolution's state (Engine::window()): for the k-th token t of the call whose bit is
/// set in `bits` (t < rows), the state as inkling_conv() leaves it after tokens 0 to t, into snap + k * stride
/// (floats).  x and state as inkling_conv()'s, before the call advances the state.
void inkling_conv_snaps(const float* x, const float* state, int channels, int rows, unsigned bits, float* snap,
                        size_t stride, void* stream);

// ---- depth mix -----------------------------------------------------------------------------------------------

inline constexpr int kInklingMaxStates = 8;
/// mix()'s inputs: up to kInklingMaxStates states (the earlier blocks' outputs, then cur), each rows * n values,
/// and n_query queries (query: n_query rows of n floats, ggml [n, n_query]) each with its own output
/// (rows * n floats).  n_query <= 4.
struct InklingMix {
    const float* state[kInklingMaxStates] = {};
    int n_state = 0;
    const float* query = nullptr;
    int n_query = 1;
    float* out[4] = {};
};
/// out[q][t] = sum_k softmax_k(s_hat_k[t] . query[q] / sqrt(n)) * s_hat_k[t], s_hat = rms_norm(s, eps) without a
/// weight (upstream's eps is 1e-6).  An output may be one of the states only when n_query is 1.
void inkling_mix(const InklingMix& m, int n, int rows, float eps, void* stream);

// ---- loop controller -----------------------------------------------------------------------------------------

/// control() whole, for rows tokens: z = [rms_norm(cur), rms_norm(msg)] (eps), r = down z, then
/// out = (cur - tanh(forget r) * cur) + tanh(gain r) * msg.  down: rank rows of 2 * n floats (ggml [2n, rank]);
/// gain and forget: n rows of rank floats (ggml [rank, n]); all F32.  work: rows * (2 * n + rank + 2 * n) floats.
/// Uses gemv_f32 for fewer than 8 rows, else gemm_f32 on `blas` (blas_create on this stream; may be null below 8
/// rows).  out may be cur.
void inkling_control(void* blas, const float* cur, const float* msg, const float* down, const float* gain,
                     const float* forget, float* work, float* out, int n, int rank, int rows, float eps,
                     void* stream);

// ---- fused decode-path steps ---------------------------------------------------------------------------------
//
// Each of these does what a sequence of the calls above does, in fewer launches, for the few tokens of a decode
// step.  The results agree with the sequence's to float rounding (the sums go in another order), except where a
// comment says "the same bits".

/// inkling_control_fused()'s largest rank (loop.gate_rank): a lane of a warp holds two of its values.
inline constexpr int kInklingMaxRank = 64;

/// One loop controller of inkling_control()'s: cur and msg (rows * n each), its three F32 matrices, and where out
/// goes (out may be cur).
struct InklingCtrl {
    const float* cur = nullptr;
    const float* msg = nullptr;
    const float* down = nullptr;     ///< rank rows of 2n (ggml [2n, rank])
    const float* gain = nullptr;     ///< n rows of rank (ggml [rank, n])
    const float* forget = nullptr;   ///< n rows of rank
    float* out = nullptr;
};
/// inkling_control() for `count` (1 to 4) controllers at once (a looped visit's Q, K and V, whose inputs are ready
/// together; or one), in two launches whatever the count: the norms and the down projection, then gain, forget
/// and the combine.  Sums in float.  rank <= kInklingMaxRank, n a multiple of 4; cur, msg and down 16-byte aligned
/// (they are read a float4 at a time; cudaMalloc'd rows of a multiple of 4 floats are).  work: count * rows * rank
/// floats.  Rows go four at a time (each four re-read the matrices).  Throws std::invalid_argument.
void inkling_control_fused(const InklingCtrl* ctrl, int count, int n, int rank, int rows, float eps, float* work,
                           void* stream);

/// y[t][o] = w[o] . x[t] for rows (<= 4 at a time; more go four by four) tokens: gemv_f32 / gemm_f32's product
/// (rows_per_x = n_out) for a decode step, a warp an output, the matrix read once for the tokens.  w: F32.  n_in a
/// multiple of 4, w and x 16-byte aligned.  E.g. the router's ffn_gate_inp [2304, 128].
void inkling_gemv(const float* w, const float* x, float* y, int n_in, int n_out, int rows, void* stream);

/// Several inkling_clip() calls of the same tokens in one launch (Q and R of a layer; K too when it is not done by
/// inkling_kv_store()).  The same bits as the separate calls.
struct InklingClipJob {
    const float* x = nullptr;
    int x_stride = 0;
    const float* scale = nullptr;
    float* y = nullptr;
    int dim = 0, heads = 0;
    float threshold = 0.f;
    InklingTau tau;
};
/// count <= 4.
void inkling_clip_many(const InklingClipJob* jobs, int count, int rows, int64_t pos, void* stream);

/// Where inkling_kv_store() puts a layer's keys and values: fp16 (k, v) or Q8_0 (kc/ks, vc/vs, kernels.hpp's Kvq8
/// layout), each row kv_heads * dim; with ring > 0 position p is row p % ring.
struct InklingKvDst {
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
    int8_t* kc = nullptr;
    uint16_t* ks = nullptr;
    int8_t* vc = nullptr;
    uint16_t* vs = nullptr;
    int ring = 0;
};
/// A layer's K and V from their projections to the cache, in one launch for up to 16 tokens:
///   k = inkling_conv(kraw, k_state, k_w, k_taps) then inkling_clip(k, no scale, k_clip),
///   v = inkling_conv(vraw, v_state, v_w, v_taps) then rms_norm(v, no weight, eps) a head,
///   both into dst at the tokens' positions (from pos) as store_f16 / store_q8 would write them (the same bits),
///   and into k_out / v_out (rows * kv_heads * dim floats each) when they are not null.
/// Taps (null: none) and the states as inkling_conv()'s.  dim 128.  More than 16 tokens take the separate calls
/// (the same results), which work in k_out and v_out: both must then be given (else std::invalid_argument).
void inkling_kv_store(const float* kraw, const float* vraw, float* k_state, float* v_state, const float* k_w,
                      const float* v_w, const float* k_taps, const float* v_taps, int kv_heads, int dim, int rows,
                      float k_clip, float eps, const InklingKvDst& dst, int64_t pos, float* k_out, float* v_out,
                      void* stream);

/// inkling_conv() whose output is added to acc (acc += out) instead of written: the attention's and the FFN's
/// post-convolutions and the residual adds after them (axpy) in one launch.  The same bits as inkling_conv() then
/// axpy(out, 1, acc, add).
void inkling_conv_acc(const float* x, float* state, const float* w, const float* taps, int heads, float* acc,
                      int channels, int rows, void* stream);

// ---- a token of each of several sequences --------------------------------------------------------------------
//
// Engine::step()'s rows (--batch): row t of a call is the next token of a sequence of its own, with that sequence's
// caches, convolution state and position.  Each call below gives a row the same bits as the call above it gives
// that row alone (rows 1, the row's own state, caches and position), in one launch for all the rows.

/// The most rows of these calls.
inline constexpr int kInklingSeqRows = 16;

/// inkling_clip_many() for rows each at its own position: row t's tau is at pos[t] (host array of rows values).
void inkling_clip_many_seqs(const InklingClipJob* jobs, int count, int rows, const int64_t* pos, void* stream);

/// A row's destination and state in inkling_kv_store_seqs(): its sequence's cache (ring as InklingKvDst's), the K
/// and V convolutions' states, and its position.
struct InklingKvSeq {
    InklingKvDst dst;
    float* k_state = nullptr;
    float* v_state = nullptr;
    int64_t pos = 0;
};
/// inkling_kv_store() of one token for each of `rows` rows (seqs: a host array), into each row's own cache and
/// state.  The rest as inkling_kv_store()'s; k_out / v_out may be null.
void inkling_kv_store_seqs(const float* kraw, const float* vraw, const InklingKvSeq* seqs, const float* k_w,
                           const float* v_w, const float* k_taps, const float* v_taps, int kv_heads, int dim,
                           int rows, float k_clip, float eps, float* k_out, float* v_out, void* stream);

/// A row's caches in inkling_attend_seqs(): its sequence's (fp16, or Q8_0 where the codes are not null), its
/// position and the room it is sized for (inkling_attend()'s room; 0: pos + 1).
struct InklingAttnSeq {
    const uint16_t* kcache = nullptr;
    const uint16_t* vcache = nullptr;
    Kvq8 kq8, vq8;
    int64_t pos = 0;
    int64_t room = 0;
};
/// inkling_attend()'s split path for one token of each of `rows` rows (seqs: a host array), each against its own
/// caches at its own position, split as inkling_attend() splits that row alone.  Every row's caches are fp16, or
/// every row's Q8_0 (the same --kv for all).  q, rel.r, sinks and out hold the rows one after the other, as
/// inkling_attend()'s tokens.  work: inkling_attend_work_floats() floats (not null).  Throws
/// std::invalid_argument for more than kInklingSeqRows rows or caches of mixed kinds.
void inkling_attend_seqs(const float* q, const InklingAttnSeq* seqs, int rows, InklingRel rel, const float* sinks,
                         float* out, int dim, int n_head, int kv_heads, float scale, int window, int ring,
                         float* work, size_t work_floats, void* stream);

/// inkling_conv() / inkling_conv_acc() of one token for each of `rows` rows, row t with its own state states[t]
/// (a host array of rows pointers), without taps.
void inkling_conv_seqs(const float* x, float* const* states, const float* w, float* out, int channels, int rows,
                       void* stream);
void inkling_conv_acc_seqs(const float* x, float* const* states, const float* w, float* acc, int channels, int rows,
                           void* stream);

// ---- experts' activation -------------------------------------------------------------------------------------

/// SiTU for rows that each hold n gate values then n up values (the fused gate_up product); out: n a row,
///     out = beta * tanh(gate / beta) * sigmoid(gate) * (lb > 0 ? lb * tanh(up / lb) : up).
/// `scales` as swiglu_clamp_rows()'s (null: none).
void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream,
               const ExpertScales* scales = nullptr);

}  // namespace strata::poly
