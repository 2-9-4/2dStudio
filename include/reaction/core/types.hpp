#pragma once

#include <cstdint>
#include <string>
#include <variant>

namespace reaction {

using NodeId = std::uint64_t;
using LinkId = std::uint64_t;
using SocketId = std::uint64_t;

enum class ValueType { Float, Image2D, AnyNumeric };
enum class SocketDirection { Input, Output };

struct ImageHandle {
    std::uint32_t texture = 0;
    int width = 0;
    int height = 0;
    explicit operator bool() const noexcept { return texture != 0; }
};

using Value = std::variant<std::monostate, float, ImageHandle>;

struct Vec2 {
    float x = 0.0F;
    float y = 0.0F;
};

inline std::string toString(ValueType type) {
    switch (type) {
    case ValueType::Float: return "float";
    case ValueType::Image2D: return "image2d";
    case ValueType::AnyNumeric: return "numeric";
    }
    return "unknown";
}

} // namespace reaction

