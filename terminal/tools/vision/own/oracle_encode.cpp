// tools/vision/own/oracle_encode.cpp - mtmd's side of the tower check (built in tools/vision's mtmd build only, EXCLUDE_FROM_ALL).
//
//   strata-vision-enc-oracle [--gpu] <mmproj.gguf> <out dir> <image> [<image> ...]
// For each image: decodes it with mtmd's own helper, runs the preprocessor mtmd_tokenize uses for this projector (MiniCPM-V 4.6's slicer, else the
// dynamic-size one of Qwen and Gemma 4), encodes what it gives with clip_image_encode on the CPU (the overview first, then each slice, as
// MiniCPM-V's prompt has them), and writes <out dir>/<image file name>.enc: per encoded picture one strata-vision record, int32
// {0x31455653 'SVE1', n_tokens, nx, ny, n_embd} then float32 [n_tokens][n_embd].  The own towers' output is compared with these files.
// --gpu (a build with STRATA_VISION_ORACLES_CUDA=ON): mtmd's GPU backend with flash attention on auto, as the mtmd reader ran it, to time it: each
// picture is encoded twice and the second time printed.
#include "clip.h"
#include "clip-impl.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-image.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static std::string base_name(const std::string& p) {
    const size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

int main(int argc, char** argv) {
    const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
    const int a0 = gpu ? 2 : 1;
    if (argc < a0 + 3) { std::fprintf(stderr, "usage: strata-vision-enc-oracle [--gpu] <mmproj> <out dir> <image>...\n"); return 2; }
    const std::string mmproj = argv[a0], out_dir = argv[a0 + 1];
    clip_context_params cp{};
    cp.use_gpu = gpu;
    cp.flash_attn_type = gpu ? CLIP_FLASH_ATTN_TYPE_AUTO : CLIP_FLASH_ATTN_TYPE_DISABLED;
    cp.image_min_tokens = -1;
    cp.image_max_tokens = -1;
    cp.warmup = false;
    clip_init_result r = clip_init(mmproj.c_str(), cp);
    if (!r.ctx_v) { std::fprintf(stderr, "cannot load the vision encoder %s\n", mmproj.c_str()); return 1; }
    clip_ctx* ctx = r.ctx_v;
    std::unique_ptr<mtmd_image_preprocessor> pre;
    if (clip_get_projector_type(ctx) == PROJECTOR_TYPE_MINICPMV4_6) pre = std::make_unique<mtmd_image_preprocessor_minicpmv>(ctx);
    else pre = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx);
    const int threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    const int n_embd = clip_n_mmproj_embd(ctx);
    int rc = 0;
    for (int a = a0 + 2; a < argc; ++a) {
        const std::string img = argv[a];
        FILE* f = std::fopen(img.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", img.c_str()); rc = 1; continue; }
        std::vector<unsigned char> buf;
        unsigned char tmp[1 << 16];
        size_t k;
        while ((k = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + k);
        std::fclose(f);
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(nullptr, buf.data(), buf.size(), false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) { std::fprintf(stderr, "cannot decode %s\n", img.c_str()); rc = 1; continue; }
        clip_image_u8 u8;
        u8.set_size({(int) mtmd_bitmap_get_nx(bw.bitmap), (int) mtmd_bitmap_get_ny(bw.bitmap)}, false);
        u8.cpy_buf(std::vector<uint8_t>(mtmd_bitmap_get_data(bw.bitmap), mtmd_bitmap_get_data(bw.bitmap) + mtmd_bitmap_get_n_bytes(bw.bitmap)));
        mtmd_bitmap_free(bw.bitmap);
        mtmd_image_preproc_out out = pre->preprocess(u8);
        std::vector<const clip_image_f32*> pics;
        if (out.has_overview()) pics.push_back(&out.overview);
        for (const auto& e : out.entries) pics.push_back(&e);
        const std::string path = out_dir + "/" + base_name(img) + ".enc";
        FILE* o = std::fopen(path.c_str(), "wb");
        if (!o) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); rc = 1; continue; }
        std::printf("%s: %dx%d ->", img.c_str(), u8.get_size().width, u8.get_size().height);
        for (const clip_image_f32* p : pics) {
            const int n = clip_n_output_tokens(ctx, p);
            std::vector<float> v((size_t) n * n_embd);   // (clip_image_encode copies into the buffer it is given, sized by the caller)
            if (!clip_image_encode(ctx, threads, p, v)) { std::fprintf(stderr, "%s: the encoder failed\n", img.c_str()); rc = 1; break; }
            const auto t0 = std::chrono::steady_clock::now();
            if (gpu && !clip_image_encode(ctx, threads, p, v)) { std::fprintf(stderr, "%s: the encoder failed\n", img.c_str()); rc = 1; break; }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if ((size_t) n * n_embd != v.size()) { std::fprintf(stderr, "%s: %zu values for %d tokens\n", img.c_str(), v.size(), n); rc = 1; break; }
            const int32_t hdr[5] = {0x31455653, n, clip_n_output_tokens_x(ctx, p), clip_n_output_tokens_y(ctx, p), n_embd};
            std::fwrite(hdr, sizeof hdr, 1, o);
            std::fwrite(v.data(), sizeof(float), v.size(), o);
            std::printf(" [%dx%d: %d tokens %dx%d", p->nx(), p->ny(), n, hdr[2], hdr[3]);
            if (gpu) std::printf(", %.0f ms", ms);
            std::printf("]");
        }
        std::printf("\n");
        std::fclose(o);
    }
    clip_free(ctx);
    return rc;
}
