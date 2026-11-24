// tools/vision/own/own_reader.cpp - see own_reader.h
#include "own_reader.h"

#include "cpu_caps.h"
#include "vit_cpu.h"
#include "vit_ref.h"
#ifdef STRATA_VIT_GPU
#include "vit_gpu.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>

namespace strata_vit {

namespace {

// "ENC <image> <out>": the image path ends at the last space before the output path (a path of the image may hold spaces)
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

// MiniCPM-V 4.7: the overview and the slices through the tower one at a time, written as poly.10's reader wrote mtmd's canvas chunk: one record
// per tile, and after each tile but the last one 1x1 record per text separator (</image><slice> after the overview, </slice><slice> between two
// slices of a row, </slice>\n<slice> at the end of a row), each a row of v.tok_embd_sep.  No token limit applies: the tiling bounds a picture.
template <class Enc>
bool encode_tiles(const ImageU8& src, const VitGeometry& g, const std::vector<float>& seps, const std::string& path, Enc& encode, int& n_all, int& nx0, int& ny0,
                  std::string& err) {
    TiledImage ti;
    if (!prepare_tiles(src, g, ti, err)) { err = "the image could not be preprocessed (" + err + ")"; return false; }
    struct Rec {
        int nx, ny;
        std::vector<float> rows;
    };
    std::vector<Rec> recs;
    auto sep = [&](const std::vector<int>& idx) {
        for (int i : idx) recs.push_back({1, 1, std::vector<float>(seps.begin() + (ptrdiff_t) i * g.out_dim, seps.begin() + (ptrdiff_t) (i + 1) * g.out_dim)});
    };
    const int tiles = 1 + (int) ti.slices.size();
    for (int k = 0; k < tiles; ++k) {
        if (k == 1) sep({0, 1});
        else if (k > 1) sep((k - 2) % ti.grid_x == ti.grid_x - 1 ? std::vector<int>{2, 3, 1} : std::vector<int>{2, 1});
        Rec r;
        if (!encode(k == 0 ? ti.overview : ti.slices[(size_t) k - 1], r.rows, r.nx, r.ny, err)) { err = "the vision encoder failed (" + err + ")"; return false; }
        recs.push_back(std::move(r));
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot write " + path; return false; }
    bool ok = true;
    n_all = 0;
    for (const Rec& r : recs) {
        const int n = r.nx * r.ny;
        const int32_t hdr[5] = {0x31455653, n, r.nx, r.ny, g.out_dim};
        ok = ok && std::fwrite(hdr, sizeof hdr, 1, f) == 1 && std::fwrite(r.rows.data(), sizeof(float) * (size_t) g.out_dim, (size_t) n, f) == (size_t) n;
        n_all += n;
    }
    std::fclose(f);
    if (!ok) { err = "short write to " + path; return false; }
    nx0 = recs[0].nx;
    ny0 = recs[0].ny;
    return true;
}

}  // namespace

bool own_reader_inject_failure(const char* point) {
    const char* v = std::getenv("STRATA_VIT_TEST_FAIL");
    return v && point && std::string(v) == point;
}

int own_reader_run(const OwnReaderOptions& o) {
    using clock = std::chrono::steady_clock;
    const auto load_t0 = clock::now();
    // every return before READY goes through here: an "ERR ..." line (the server then starts the CPU reader where this was the GPU one)
    auto fail = [](const std::string& msg, int code) {
        std::printf("ERR %s\n", msg.c_str());
        std::fflush(stdout);
        return code;
    };
    const std::string injected = " (injected for a test: STRATA_VIT_TEST_FAIL)";
#ifndef STRATA_VIT_GPU
    if (o.gpu) return fail("this build has no GPU backend for the own tower", 3);
#endif
    Gguf gg;
    VitGeometry geo;
    std::string err;
    if (!gg.open(o.mmproj, err) || !vit_geometry_from_gguf(gg, geo, err) || !vit_geometry_supported(geo, err))
        return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
    std::vector<float> seps;   // minicpmv4_7: the separator rows written between a picture's tiles
    if (geo.minicpmv()) {
        const GgufTensorInfo* t = gg.tensor("v.tok_embd_sep");
        seps.resize((size_t) t->nelem());
        if (!gg.read_f32(*t, seps.data(), err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
    }
    int threads = o.threads;
    std::unique_ptr<VitRef> cpu;
    std::unique_ptr<VitCpu> fast;
#ifdef STRATA_VIT_GPU
    std::unique_ptr<VitGpu> gpu;
    if (o.gpu) {
        std::string why;
        if (geo.head_dim > kVitGpuMaxHeadDim)
            return fail("this picture reader file has attention heads of size " + std::to_string(geo.head_dim) + "; the GPU picture reader handles up to " +
                            std::to_string(kVitGpuMaxHeadDim) + " (the CPU picture reader reads it)", 1);
        if (own_reader_inject_failure("gpu")) return fail("this picture reader found no GPU" + injected, 3);
        if (!VitGpu::available(why)) return fail("this picture reader found no GPU (" + why + ")", 3);
        gpu.reset(new VitGpu());
        // buffers for the warm-up picture below (2048 x 2048: 4,096 tokens at most, fewer under a lower limit): claimed now, before READY.  A larger limit
        // (the app allows up to 16,384 image tokens) grows them when such a picture comes: slower for that picture, never an error.
        // (MiniCPM-V reads a picture tile by tile, each about one slice: twice a slice's patches leaves room for the stretched shapes)
        const int tokens = std::min(o.max_tokens > 0 ? o.max_tokens : geo.max_tokens, 4096);
        const int side = geo.minicpmv() ? geo.slice_size / geo.patch : 0;
        if (!gpu->load(gg, geo, 0, geo.minicpmv() ? 2 * side * side : tokens * 4, err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
        if (own_reader_inject_failure("load")) return fail("cannot load the vision encoder " + o.mmproj + injected, 1);   // after the load: its memory is freed on return
    }
#endif
    if (!o.gpu) {
        // the CPU path of the own tower is the fast tower (VitCpu); STRATA_VIT_CPU=ref selects the float32 reference (VitRef: the oracle, for tests)
        const char* sel = std::getenv("STRATA_VIT_CPU");
        if (!(sel && std::string(sel) == "ref")) {
            fast.reset(new VitCpu());
            if (!fast->load(gg, geo, err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
        } else {
            cpu.reset(new VitRef());
            if (!cpu->load(gg, geo, err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
        }
        if (own_reader_inject_failure("load")) return fail("cannot load the vision encoder " + o.mmproj + injected, 1);
        if (threads <= 0) threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
    }
    std::fprintf(stderr, "strata-vision: own tower (%s, %d layers, width %d) loaded in %.1f s, %s\n", geo.projector.c_str(), geo.layers, geo.width,
                 std::chrono::duration<double>(clock::now() - load_t0).count(), o.gpu ? "on the GPU" : ("on the CPU, " + std::to_string(threads) + " threads").c_str());
    auto encode = [&](const PreparedImage& prep, std::vector<float>& rows, int& nx, int& ny, std::string& e) {
#ifdef STRATA_VIT_GPU
        if (gpu) return gpu->encode(prep, rows, nx, ny, e);
#endif
        if (fast) return fast->encode(prep, threads, rows, nx, ny, e);
        return cpu->encode(prep, threads, rows, nx, ny, e);
    };
#ifdef STRATA_VIT_GPU
    if (gpu) {   // warm up at the largest picture before READY: the work buffers are taken now, not at the first real picture
        const auto w0 = clock::now();
        ImageU8 gray;
        gray.w = gray.h = 2048;
        gray.rgb.assign((size_t) 2048 * 2048 * 3, 128);
        PreparedImage wp;
        TiledImage wt;
        std::vector<float> rows;
        int nx = 0, ny = 0;
        err.clear();
        if (own_reader_inject_failure("warmup")) err = "out of GPU memory" + injected;
        const bool prepared = err.empty() && (geo.minicpmv() ? prepare_tiles(gray, geo, wt, err) : prepare_image(gray, geo, o.min_tokens, o.max_tokens, wp, err));
        if (prepared && geo.minicpmv()) wp = std::move(wt.overview);
        if (prepared && encode(wp, rows, nx, ny, err))
            std::fprintf(stderr, "strata-vision: warmed up at %d image tokens in %.1f s, %.0f MiB of GPU memory\n", nx * ny,
                         std::chrono::duration<double>(clock::now() - w0).count(), (double) gpu->device_bytes() / 1048576.0);
        else
            return fail("the vision encoder failed its warm-up (" + err + ")", 1);
    }
#endif
    std::printf("READY %d own\n", geo.out_dim);   // the third word names the reader (a server reads only the start: "READY")
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img, out;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = clock::now();
        ImageU8 src;
        PreparedImage prep;
        if (!image_load_file(img, src, err)) { std::printf("ERR cannot read the image %s\n", img.c_str()); std::fflush(stdout); continue; }
        if (geo.minicpmv()) {
            int n = 0, nx0 = 0, ny0 = 0;
            if (encode_tiles(src, geo, seps, out, encode, n, nx0, ny0, err))
                std::printf("OK %d %d %d %.0f\n", n, nx0, ny0, std::chrono::duration<double, std::milli>(clock::now() - t0).count());
            else
                std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (!prepare_image(src, geo, o.min_tokens, o.max_tokens, prep, err)) { std::printf("ERR the image could not be preprocessed (%s)\n", err.c_str()); std::fflush(stdout); continue; }
        const int n_expect = (prep.w / geo.align()) * (prep.h / geo.align());
        // never silently produce many times the budget (I7, #3690): over the limit (a rounding margin of an eighth) the picture is refused
        if (o.max_tokens > 0 && n_expect > o.max_tokens + o.max_tokens / 8) {
            std::printf("ERR the picture became %d image tokens, over the limit of %d (vision.max_tokens); send a smaller picture or raise the limit\n", n_expect,
                        o.max_tokens);
            std::fflush(stdout);
            continue;
        }
        std::vector<float> rows;
        int nx = 0, ny = 0;
        if (std::getenv("STRATA_VIT_PROF"))
            std::fprintf(stderr, "vit-prof: read + decode + prepare %.2f ms\n", std::chrono::duration<double, std::milli>(clock::now() - t0).count());
        if (!encode(prep, rows, nx, ny, err)) { std::printf("ERR the vision encoder failed (%s)\n", err.c_str()); std::fflush(stdout); continue; }
        const int n = nx * ny;
        FILE* f = std::fopen(out.c_str(), "wb");
        if (!f) { std::printf("ERR cannot write %s\n", out.c_str()); std::fflush(stdout); continue; }
        const int32_t hdr[5] = {0x31455653, n, nx, ny, geo.out_dim};
        const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 && std::fwrite(rows.data(), sizeof(float) * (size_t) geo.out_dim, (size_t) n, f) == (size_t) n;
        std::fclose(f);
        if (!ok) { std::printf("ERR short write to %s\n", out.c_str()); std::fflush(stdout); continue; }
        const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace strata_vit
