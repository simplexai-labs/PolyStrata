// tools/vision/own/image_prep.h - a picture to the tower's input: decode, size, resize, normalise (own-stack phase 3).
//
// Follows what llama.cpp's mtmd does for a Qwen-VL tower, step for step, so the pixels that reach the tower are the same ones (the resize
// is a port of mtmd's Pillow-compatible fixed-point bicubic; its maths and its rounding are MIT, from llama.cpp tools/mtmd/mtmd-image.cpp):
//   1. stb_image decodes to 8-bit RGB (JPEG, PNG, BMP, GIF first frame, ...);
//   2. the target size is "smart_resize": both sides rounded to a multiple of patch * merge, then scaled down to the largest picture of at
//      most max_pixels, or up to at least min_pixels (the limits are token counts times patch^2 * merge^2);
//   3. the picture is resized with the aspect ratio kept (bicubic, a = -0.5; ceil rounding of the resized side) into the target and centred on
//      a black canvas (mtmd's PAD_CEIL; nothing is padded when the picture already has the target's shape);
//   4. (v / 255 - mean) / std per channel, planar CHW float32.
#pragma once

#include "vit_geometry.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata_vit {

struct ImageU8 {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;   // h rows of w * 3 bytes
};

struct PreparedImage {
    int w = 0, h = 0;               // the size the tower sees (multiples of patch * merge)
    std::vector<float> chw;         // 3 * h * w, normalised, channel planes
};

struct Size2 {
    int w = 0, h = 0;
    bool operator==(const Size2& o) const { return w == o.w && h == o.h; }
};

// stb_image to RGB8.  False and `err` when the bytes are not a picture stb_image reads.
bool image_decode(const uint8_t* data, size_t n, ImageU8& out, std::string& err);
bool image_load_file(const std::string& path, ImageU8& out, std::string& err);

// mtmd's calc_size_preserved_ratio (HF smart_resize): sides rounded to `align`, then bounded by max_pixels / min_pixels (0 = no bound)
Size2 smart_resize(int w, int h, int align, int min_pixels, int max_pixels);

// Pillow-compatible separable bicubic (a = -0.5), 22-bit fixed point on uint8, as mtmd's resize_pillow
void resize_bicubic(const ImageU8& src, ImageU8& dst, int target_w, int target_h);

// mtmd's img_tool::resize with PAD_CEIL and a black canvas: aspect kept, centred, padded
void resize_pad_ceil(const ImageU8& src, ImageU8& dst, Size2 target);

// Inkling 2 Mini (g.inkling()) is prepared as llama.cpp's mtmd_image_preprocessor_inkling2v does it: the picture as it is, padded at the right
// and the bottom to multiples of patch * merge with the raw value -1 (normalised as a pixel would be), not resized; only a picture whose padded
// area is over the token limit is first resized (bicubic, aspect kept, sides multiples of patch * merge) to the largest that fits.
// The whole step 2-4 for a geometry.  min_tokens / max_tokens > 0 replace the geometry's limits (the helper's --min-tokens / --max-tokens).
// The resized picture (before the normalisation) is returned in `resized` when it is not null (for tests).
bool prepare_image(const ImageU8& src, const VitGeometry& g, int min_tokens, int max_tokens, PreparedImage& out, std::string& err,
                   ImageU8* resized = nullptr);

// the size prepare_image would give (no pixels touched)
Size2 prepared_size(int w, int h, const VitGeometry& g, int min_tokens, int max_tokens);

// MiniCPM-V 4.7 (g.minicpmv()): mtmd's llava-uhd slicing for MiniCPM-V (mtmd_image_preprocessor_minicpmv).  A picture of at most one tile's
// area becomes the overview alone, stretched (no padding) to the tile-sized shape of its aspect; a larger one an overview of that shape plus
// grid_x * grid_y slices of one refined picture, row by row.  Sides are multiples of patch * 4, rounded half to even as Python's round().
struct TiledImage {
    PreparedImage overview;
    std::vector<PreparedImage> slices;   // grid_y rows of grid_x, empty for a picture of one tile
    int grid_x = 0, grid_y = 0;
};
bool prepare_tiles(const ImageU8& src, const VitGeometry& g, TiledImage& out, std::string& err);

// the sizes prepare_tiles would give: the overview's, then one slice's (0 x 0 without slices) and the grid
void tile_sizes(int w, int h, const VitGeometry& g, Size2& overview, Size2& slice, int& grid_x, int& grid_y);

}  // namespace strata_vit
