#pragma once
#include "strata/gpu/metal.hpp"
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <climits>
#include <cstdint>
#include <cstdlib>
#endif
namespace strata::poly::metal_detail {
// `name` in the folder metal/ beside the running program (its real path), or empty where that is not known
inline std::string beside_program(const std::string& name) {
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string raw(size + 1, '\0');
    if (_NSGetExecutablePath(raw.data(), &size) != 0) return {};
    char real[PATH_MAX];
    if (!realpath(raw.c_str(), real)) return {};
    const std::string exe(real);
    return exe.substr(0, exe.find_last_of('/') + 1) + "metal/" + name;
#else
    (void) name;
    return {};
#endif
}
// A kernel's source at the path compiled in (the source tree an engine is built in, as setup builds it), else the file
// of the same name in metal/ beside the program: a packaged engine keeps its sources there.
inline std::string read_source(const char* path) {
    std::ifstream f(path);
    std::string other;
    if (!f) {
        const std::string p(path);
        other = beside_program(p.substr(p.find_last_of('/') + 1));
        if (!other.empty()) f.open(other);
    }
    if (!f)
        throw std::runtime_error(std::string("cannot read Metal kernel: ") + path + (other.empty() ? "" : " or " + other));
    return std::string(std::istreambuf_iterator<char>(f), {});
}
template<class Args>
void dispatch(const char* library, const std::string& source, const char* name, const Args& args,
              void* stream, gpu::metal::Grid groups, uint32_t threads, gpu::metal::Resources resources = {},
              size_t threadgroup_bytes = 0) {
    std::string error;
    auto* p = gpu::metal::pipeline(library, source.c_str(), name, error);
    if (!p) throw std::runtime_error(error);
    if (!resources.complete && !resources.indirect_table) {
        if constexpr (requires { args.p; }) resources = {args.p, std::size(args.p), true};
    }
    auto rc = gpu::metal::dispatch(static_cast<gpu::Stream>(stream), p, &args, sizeof(args), groups, {threads,1,1},
                                   threadgroup_bytes, resources);
    if (rc != gpu::kSuccess) throw std::runtime_error(gpu::error_string(rc));
}
}
