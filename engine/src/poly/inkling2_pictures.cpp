// src/poly/inkling2_pictures.cpp - Inkling 2 Mini's pictures: the rows strata-vision's inkling2v tower makes (tools/vision/own) stand where
// the chat template writes <|unused_200054|>, one a row, after <|content_image|>, as llama.cpp's mtmd places them.
//
// llama.cpp reads a picture's rows in a batch of their own and leaves out the token norm there ("do not normalize raw embeddings input",
// llama_model_inkling2::graph::build_inputs()); the attention over them stays causal, as over text.  So a call is a picture or has no row
// of one, and Inkling2::layer_head() norms the first layer's input only when the call is text.
#include "inkling2.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strata::poly {

int64_t Inkling2::whole(int64_t at, int64_t m) const {
    for (const Image& im : images_) {
        if (at >= im.at + im.n) continue;
        if (at < im.at) return std::min(m, im.at - at);
        if (at != im.at) throw std::logic_error("a sequence cannot go on from the middle of a picture");
        if (im.n > rows_)
            throw std::runtime_error("a picture of " + std::to_string(im.n) + " rows is more than a prompt batch (" +
                                     std::to_string(rows_) + "): raise --prompt-batch or lower the picture's tokens");
        return im.n;
    }
    return m;
}

}  // namespace strata::poly
