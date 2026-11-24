// src/poly/expert_parity.cpp - the engine's products on the model's own weights, against ggml's dequantized ones.
//
//   poly_expert_parity MODEL.gguf
//
// For a few experts of every tensor type the file uses, the GPU path (native_mmvq over the GGUF blocks) and the
// CPU path (cpu_experts) are compared with a double-precision product over ggml's dequantized weights, and the
// GPU path with two experts a launch (native_mmvq_slots) must give the bits it gives with one.  Then
// every kind of matrix in the file (a type and a shape) takes a batch of tokens through the two GPU products,
// native_mmvq and the batch product (mmq.hpp): batches of several sizes, and a routed expert's kind three experts
// a launch with the rows going elsewhere, as the engine launches them.  Most of them quantize their input to
// 8 bits, so none is exact; the test fails when one is far from the reference.  A model in several files: give
// the file that holds the tensors to check.
#include "cpu_experts.hpp"
#include "mmq.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "gguf_types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using namespace strata;

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// rows x n_in weights in the GGUF's blocks, as floats
std::vector<float> dequant(uint32_t type, const uint8_t* w, int n_in, int rows) {
    std::vector<float> out((size_t) rows * n_in);
    poly::type_to_floats(type, w, out.data(), (int64_t) rows * n_in);
    return out;
}

std::vector<double> matvec(const std::vector<float>& w, const std::vector<double>& x, int n_in, int rows) {
    std::vector<double> y((size_t) rows);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int i = 0; i < n_in; ++i) acc += (double) w[(size_t) r * n_in + i] * x[(size_t) i];
        y[(size_t) r] = acc;
    }
    return y;
}

double rel_error(const float* got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        num += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
        den += want[i] * want[i];
    }
    return std::sqrt(num / den);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: poly_expert_parity MODEL.gguf\n");
        return 2;
    }
    try {
        GgufFile g(argv[1]);
        const float limit = 10.f;

        // the first layer of every (gate/up type, down type) pair; the experts' shape says the rest
        std::map<std::pair<uint32_t, uint32_t>, int> layers;
        int n_embd = 0, n_ff = 0, n_expert = 0;
        for (const TensorInfo& tg : g.tensors()) {
            const size_t at = tg.name.find(".ffn_gate_exps.weight");
            if (at == std::string::npos || tg.shape.size() != 3) continue;
            const std::string b = tg.name.substr(0, at);
            const TensorInfo* tu = g.find(b + ".ffn_up_exps.weight");
            const TensorInfo* td = g.find(b + ".ffn_down_exps.weight");
            if (!tu || !td || tu->type != tg.type) continue;
            n_embd = (int) tg.shape[0];
            n_ff = (int) tg.shape[1];
            n_expert = (int) tg.shape[2];
            layers.emplace(std::make_pair(tg.type, td->type), std::stoi(b.substr(4)));
        }
        if (layers.empty()) throw std::runtime_error("the GGUF has no routed experts");

        // a residual-like input: mostly small, a few large values
        std::mt19937 rng(1234);
        std::normal_distribution<float> small(0.f, 0.05f);
        std::vector<float> x((size_t) n_embd);
        for (float& v : x) v = small(rng);
        for (int i = 0; i < n_embd; i += 97) x[(size_t) i] *= 20.f;
        const std::vector<double> xd(x.begin(), x.end());

        cudaStream_t st = nullptr;
        ck(cudaStreamCreate(&st));
        float *d_x, *d_gu, *d_h, *d_out;
        void *d_xq, *d_hq;
        // room for two experts: the second check below takes two in a launch
        const size_t hq_bytes = kernels::native_q8_1_bytes(n_ff);
        ck(cudaMalloc(&d_x, (size_t) n_embd * 4));
        ck(cudaMalloc(&d_gu, (size_t) 4 * n_ff * 4));
        ck(cudaMalloc(&d_h, (size_t) 2 * n_ff * 4));
        ck(cudaMalloc(&d_out, (size_t) 2 * n_embd * 4));
        ck(cudaMalloc(&d_xq, kernels::native_q8_1_bytes(n_embd)));
        ck(cudaMalloc(&d_hq, 2 * hq_bytes));
        ck(cudaMemcpy(d_x, x.data(), (size_t) n_embd * 4, cudaMemcpyHostToDevice));
        kernels::native_quantize_q8_1(d_x, d_xq, n_embd, 1, st);

        poly::Workers workers(8);
        bool ok = true;
        for (const auto& [types, il] : layers) {
            const std::string b = "blk." + std::to_string(il) + ".";
            const TensorInfo& tg = *g.find(b + "ffn_gate_exps.weight");
            const TensorInfo& tu = *g.find(b + "ffn_up_exps.weight");
            const TensorInfo& td = *g.find(b + "ffn_down_exps.weight");
            const size_t gu_bytes = tensor_payload_bytes(tg) / (size_t) n_expert,
                         d_bytes = tensor_payload_bytes(td) / (size_t) n_expert;
            std::string err;
            if (!poly::cpu_expert_types_ok((int) tg.type, (int) td.type, n_embd, n_ff, err)) throw std::runtime_error(err);
            const bool gpu = kernels::native_mmvq_supported((int) tg.type) && kernels::native_mmvq_supported((int) td.type);
            uint8_t* d_w = nullptr;
            ck(cudaMalloc(&d_w, 2 * gu_bytes + d_bytes + 256));
            // NVFP4 converted from NVIDIA's checkpoints: each expert's global scales (Engine::load_moe())
            const TensorInfo* sc[3] = {g.find(b + "ffn_gate_exps.scale"), g.find(b + "ffn_up_exps.scale"),
                                       g.find(b + "ffn_down_exps.scale")};
            const uint32_t sc_type[3] = {tg.type, tu.type, td.type};
            for (int k = 0; k < 3; ++k)
                if (sc_type[k] != poly::kTypeNVFP4) sc[k] = nullptr;
            const bool scaled = sc[0] || sc[1] || sc[2];
            float* d_s = nullptr;   // this expert's gate and up scales, ExpertScales' table for it
            if (scaled) ck(cudaMalloc(&d_s, 2 * sizeof(float)));

            for (int e : {0, n_expert / 2, n_expert - 1}) {
                const uint8_t* gate = g.tensor_data(tg) + (size_t) e * gu_bytes;
                const uint8_t* up = g.tensor_data(tu) + (size_t) e * gu_bytes;
                const uint8_t* down = g.tensor_data(td) + (size_t) e * d_bytes;

                float s3[3] = {1.f, 1.f, 1.f};
                for (int k = 0; k < 3; ++k)
                    if (sc[k]) s3[k] = ((const float*) g.tensor_data(*sc[k]))[e];
                poly::ExpertScales es;
                if (scaled) {
                    ck(cudaMemcpy(d_s, s3, 2 * sizeof(float), cudaMemcpyHostToDevice));
                    es.gate = sc[0] ? d_s : nullptr;
                    es.up = sc[1] ? d_s + 1 : nullptr;
                }
                const std::vector<double> gv = matvec(dequant(tg.type, gate, n_embd, n_ff), xd, n_embd, n_ff);
                const std::vector<double> uv = matvec(dequant(tu.type, up, n_embd, n_ff), xd, n_embd, n_ff);
                std::vector<double> h((size_t) n_ff);
                for (int i = 0; i < n_ff; ++i) {
                    const double gg = std::min(gv[(size_t) i] * s3[0], (double) limit);
                    h[(size_t) i] = gg / (1.0 + std::exp(-gg)) *
                                    std::clamp(uv[(size_t) i] * s3[1], (double) -limit, (double) limit);
                }
                std::vector<double> want = matvec(dequant(td.type, down, n_ff, n_embd), h, n_ff, n_embd);
                for (double& v : want) v *= s3[2];

                std::vector<float> cpu((size_t) n_embd), dev((size_t) n_embd);
                const poly::CpuExpert ce{gate, up, down, s3[2], 0, s3[0], s3[1]};
                poly::cpu_experts(workers, &ce, 1, 1, (int) tg.type, (int) tu.type, (int) td.type, n_embd, n_ff, x.data(),
                                  limit, false, cpu.data());
                const double e_cpu = rel_error(cpu.data(), want);
                double e_gpu = 0.0;
                if (gpu) {
                    ck(cudaMemcpy(d_w, gate, gu_bytes, cudaMemcpyHostToDevice));
                    ck(cudaMemcpy(d_w + gu_bytes, up, gu_bytes, cudaMemcpyHostToDevice));
                    ck(cudaMemcpy(d_w + 2 * gu_bytes, down, d_bytes, cudaMemcpyHostToDevice));
                    kernels::native_mmvq((int) tg.type, d_w, d_xq, d_gu, n_embd, 2 * n_ff, 1, st);
                    if (scaled) poly::swiglu_clamp_rows(d_gu, d_h, n_ff, 1, limit, st, &es);
                    else poly::swiglu_clamp(d_gu, d_gu + n_ff, d_h, n_ff, limit, st);
                    kernels::native_quantize_q8_1(d_h, d_hq, n_ff, 1, st);
                    kernels::native_mmvq((int) td.type, d_w + 2 * gu_bytes, d_hq, d_out, n_ff, n_embd, 1, st);
                    ck(cudaMemcpyAsync(dev.data(), d_out, (size_t) n_embd * 4, cudaMemcpyDeviceToHost, st));
                    ck(cudaStreamSynchronize(st));
                    std::vector<float> dev_s(dev);   // down's scale goes on the weight, in FP32
                    for (float& v : dev_s) v *= s3[2];
                    e_gpu = rel_error(dev_s.data(), want);
                }
                // The expert twice in one launch a stage, as the engine takes a token's cached experts: each copy
                // must come out with the bits the launches above gave.
                const char* together = "";
                bool same = true;
                if (gpu) {
                    kernels::MmvqSlots gu{}, dn{};
                    for (int j = 0; j < 2; ++j) {
                        gu.w[j] = d_w;
                        gu.x[j] = d_xq;
                        gu.y[j] = d_gu + (size_t) j * 2 * n_ff;
                        dn.w[j] = d_w + 2 * gu_bytes;
                        dn.x[j] = (const uint8_t*) d_hq + (size_t) j * hq_bytes;
                        dn.y[j] = d_out + (size_t) j * n_embd;
                    }
                    together = ", two in a launch: no kernel";
                    if (kernels::native_mmvq_slots((int) tg.type, gu, 2, n_embd, 2 * n_ff, st)) {
                        poly::swiglu_clamp_rows(d_gu, d_h, n_ff, 2, limit, st, &es);
                        kernels::native_quantize_q8_1(d_h, d_hq, n_ff, 2, st);
                        if (kernels::native_mmvq_slots((int) td.type, dn, 2, n_ff, n_embd, st)) {
                            std::vector<float> two((size_t) 2 * n_embd);
                            ck(cudaMemcpyAsync(two.data(), d_out, two.size() * 4, cudaMemcpyDeviceToHost, st));
                            ck(cudaStreamSynchronize(st));
                            same = std::memcmp(two.data(), dev.data(), (size_t) n_embd * 4) == 0 &&
                                   std::memcmp(two.data() + n_embd, dev.data(), (size_t) n_embd * 4) == 0;
                            together = same ? ", two in a launch: the same bits" : ", two in a launch: OTHER BITS";
                        }
                    }
                    ck(cudaStreamSynchronize(st));
                }
                const bool pass = e_cpu < 0.05 && e_gpu < 0.05 && same;
                ok = ok && pass;
                std::printf("blk.%d expert %3d  %s/%s  relative error: CPU %.4f, GPU %s%s%s\n", il, e, tg.type_name(),
                            td.type_name(), e_cpu, gpu ? std::to_string(e_gpu).c_str() : "(no kernel)", together,
                            pass ? "" : "  FAIL");
            }
            ck(cudaFree(d_w));
            if (d_s) ck(cudaFree(d_s));
        }

        // Every kind of matrix: a type and a shape, a routed expert's being a matrix each.  The weights sit before
        // a megabyte of 0xFF bytes, which a product that reads past its matrix turns into NaN.
        // The batch product takes each kind at several sizes: a launch's tiles are as wide as its rows need, and
        // the last size is several tiles.  A routed expert's kind then goes as the engine launches it: three
        // experts a launch, the middle one without rows, the products to other rows of a wider destination.
        constexpr int kBatch = 40, kMost = 300, kRows = 60, kCut = 23;
        const int sizes[] = {7, 16, 31, kBatch, 100, kMost};
        constexpr float kUntouched = 12345.f;
        constexpr size_t kTail = size_t(1) << 20;
        std::map<std::tuple<uint32_t, uint64_t, uint64_t>, const TensorInfo*> kinds;
        for (const TensorInfo& t : g.tensors())
            if ((t.shape.size() == 2 || t.shape.size() == 3) && kernels::native_mmvq_supported((int) t.type))
                kinds.emplace(std::make_tuple(t.type, t.shape[0], t.shape[1]), &t);
        poly::Mmq mmq;
        for (const auto& [key, t] : kinds) {
            const int n_in = (int) t->shape[0], n_out = (int) t->shape[1];
            const size_t bytes = tensor_payload_bytes(*t) / (t->shape.size() == 3 ? (size_t) t->shape[2] : 1);
            const size_t row_bytes = bytes / (size_t) n_out;
            const uint8_t* w = g.tensor_data(*t);
            const int n_mat = t->shape.size() == 3 && t->shape[2] >= 3 ? 3 : 1;

            std::vector<float> xb((size_t) kMost * n_in);
            for (float& v : xb) v = small(rng);
            for (size_t i = 0; i < xb.size(); i += 97) xb[i] *= 20.f;
            // the first, the middle and the last rows: the last ones are where a tile runs out
            std::set<int> rows;
            for (int r = 0; r < std::min(n_out, 48); ++r) {
                rows.insert(r);
                rows.insert(n_out / 2 + r < n_out ? n_out / 2 + r : r);
                rows.insert(n_out - 1 - r);
            }
            // want[m][k * kMost + b]: row k of `rows` of matrix m times row b of the batch (matrix 1 takes no rows)
            std::vector<double> want[3];
            std::vector<float> wr((size_t) n_in);
            for (int m = 0; m < n_mat; m += 2)
                for (int r : rows) {
                    poly::type_to_floats(t->type, w + (size_t) m * bytes + (size_t) r * row_bytes, wr.data(), n_in);
                    for (int b = 0; b < (m == 0 ? kMost : kRows); ++b) {
                        double acc = 0.0;
                        for (int i = 0; i < n_in; ++i) acc += (double) wr[(size_t) i] * xb[(size_t) b * n_in + i];
                        want[m].push_back(acc);
                    }
                    if (m > 0) want[m].resize(want[m].size() + kMost - kRows);
                }

            uint8_t* d_w = nullptr;
            float *d_xb, *d_y;
            void *d_q, *d_mq;
            const size_t token = kernels::native_q8_1_bytes(n_in);
            ck(cudaMalloc(&d_w, n_mat * bytes + kTail));
            ck(cudaMemset(d_w + n_mat * bytes, 0xFF, kTail));
            ck(cudaMemcpy(d_w, w, n_mat * bytes, cudaMemcpyHostToDevice));
            ck(cudaMalloc(&d_xb, xb.size() * 4));
            ck(cudaMalloc(&d_y, (size_t) kMost * n_out * 4));
            ck(cudaMalloc(&d_q, (size_t) kBatch * token));
            ck(cudaMalloc(&d_mq, poly::Mmq::q8_bytes(kMost, n_in)));
            ck(cudaMemcpy(d_xb, xb.data(), xb.size() * 4, cudaMemcpyHostToDevice));

            // the products of matrix m with batch rows [b0, b1), which lie at y[at(b) * ld + off + row]
            std::vector<float> y((size_t) kMost * n_out);
            auto error = [&](int m, int b0, int b1, int64_t ld, int64_t off, auto&& at) {
                std::vector<float> got;
                std::vector<double> exp;
                size_t k = 0;
                for (int r : rows) {
                    for (int b = b0; b < b1; ++b) {
                        got.push_back(y[(size_t) at(b) * (size_t) ld + (size_t) off + (size_t) r]);
                        exp.push_back(want[m][k * kMost + (size_t) b]);
                    }
                    ++k;
                }
                const double e = rel_error(got.data(), exp);
                return std::isfinite(e) ? e : 1e9;
            };
            auto fetch = [&](size_t floats) {
                ck(cudaMemcpyAsync(y.data(), d_y, floats * 4, cudaMemcpyDeviceToHost, st));
                ck(cudaStreamSynchronize(st));
            };
            const auto same_row = [](int b) { return b; };
            kernels::native_quantize_q8_1(d_xb, d_q, n_in * kBatch, 1, st);
            for (int at = 0; at < kBatch; at += 8)
                kernels::native_mmvq((int) t->type, d_w, (const uint8_t*) d_q + (size_t) at * token,
                                     d_y + (size_t) at * n_out, n_in, n_out, std::min(8, kBatch - at), st);
            fetch((size_t) kBatch * n_out);
            const double e_vec = error(0, 0, kBatch, n_out, 0, same_row);
            const bool tile = poly::Mmq::fits((int) t->type, n_out, n_in);
            double e_mat = 0.0, e_group = 0.0;
            bool kept = true;
            if (tile) {
                for (int nb : sizes) {
                    ck(cudaMemsetAsync(d_y, 0, (size_t) nb * n_out * 4, st));
                    poly::Mmq::quantize(d_xb, nullptr, d_mq, (int) t->type, n_in, n_in, nb, st);
                    mmq.mul(d_w, (int) t->type, n_out, n_in, 0, 1, d_mq, nullptr, nullptr, nb, nb, d_y, n_out, st);
                    fetch((size_t) nb * n_out);
                    e_mat = std::max(e_mat, error(0, 0, nb, n_out, 0, same_row));
                }
            }
            if (tile && n_mat == 3 && 2 * kRows <= kMost) {
                // rows [0, kCut) to matrix 0 and [kCut, kRows) to matrix 2; row i's products to the upper half of
                // row kRows - 1 - i of a destination twice as wide, whose lower half stays as it is
                const int32_t bounds[4] = {0, kCut, kCut, kRows};
                std::vector<int32_t> ids((size_t) kRows);
                for (int i = 0; i < kRows; ++i) ids[(size_t) i] = kRows - 1 - i;
                int32_t *d_bounds, *d_ids;
                ck(cudaMalloc(&d_bounds, sizeof bounds));
                ck(cudaMalloc(&d_ids, ids.size() * 4));
                ck(cudaMemcpy(d_bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice));
                ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                const size_t floats = (size_t) kRows * 2 * n_out;
                std::fill(y.begin(), y.begin() + (std::ptrdiff_t) floats, kUntouched);
                ck(cudaMemcpy(d_y, y.data(), floats * 4, cudaMemcpyHostToDevice));
                poly::Mmq::quantize(d_xb, nullptr, d_mq, (int) t->type, n_in, n_in, kRows, st);
                mmq.mul(d_w, (int) t->type, n_out, n_in, bytes, 3, d_mq, d_bounds, d_ids, kRows, kRows - kCut, d_y + n_out,
                        2 * (int64_t) n_out, st);
                fetch(floats);
                const auto back = [](int b) { return kRows - 1 - b; };
                e_group = std::max(error(0, 0, kCut, 2 * (int64_t) n_out, n_out, back),
                                   error(2, kCut, kRows, 2 * (int64_t) n_out, n_out, back));
                for (int b = 0; b < kRows && kept; ++b)
                    for (int c = 0; c < n_out && kept; ++c) kept = y[(size_t) b * 2 * n_out + (size_t) c] == kUntouched;
                ck(cudaFree(d_bounds));
                ck(cudaFree(d_ids));
            }
            const bool pass = e_vec < 0.05 && e_mat < 0.05 && e_group < 0.05 && kept;
            ok = ok && pass;
            std::printf("%-8s [%5d -> %6d]  relative error: vector %.4f, batch %s%s%s%s  (%s)\n", t->type_name(), n_in,
                        n_out, e_vec, tile ? std::to_string(e_mat).c_str() : "(no tile)",
                        tile && n_mat == 3 ? (", three a launch " + std::to_string(e_group)).c_str() : "",
                        kept ? "" : ", WROTE OTHER ROWS", pass ? "" : "  FAIL", t->name.c_str());
            for (void* p : {(void*) d_w, (void*) d_xb, (void*) d_y, d_q, d_mq}) ck(cudaFree(p));
        }
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "poly_expert_parity: %s\n", e.what());
        return 1;
    }
}
