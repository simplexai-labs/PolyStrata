// Per-engine GDN tensor-core prefill switch, independent of the CUDA kernel implementation: on where the card
// and the shape allow it, unless STRATA_GDN_MMA=0.
#pragma once
#include <cstdlib>
namespace strata::poly {
class GdnMmaControl {
public:
    void capture_request() {
        if (captured_) return;
        const char* value = std::getenv("STRATA_GDN_MMA");
        requested_ = !(value && *value && std::atoi(value) == 0);
        captured_ = true;
    }
    // True once when the caller should report its capability decision. The
    // counting and allocation passes both call this, including on fallback.
    template<class Available> bool initialize(Available available) {
        if (!requested_ || checked_) return false;
        checked_ = true;
        enabled_ = available();
        return true;
    }
    bool enabled() const { return enabled_; }
private:
    bool captured_ = false, requested_ = false, checked_ = false, enabled_ = false;
};
} // namespace strata::poly
