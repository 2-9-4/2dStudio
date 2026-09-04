#pragma once

#include <cstdint>
#include <string>
#include <variant>

namespace reaction {

using NodeId = std::uint64_t;
using LinkId = std::uint64_t;
using SocketId = std::uint64_t;

enum class ValueType { Float, Vec2, Image2D, AnyNumeric, AnyVector };
enum class SocketDirection { Input, Output };

struct ImageHandle {
    std::uint32_t texture = 0;
    int width = 0;
    int height = 0;
    explicit operator bool() const noexcept { return texture != 0; }
};

struct Vec2 {
    float x = 0.0F;
    float y = 0.0F;
};

using Value = std::variant<std::monostate, float, Vec2, ImageHandle>;

inline std::string toString(ValueType type) {
    switch (type) {
    case ValueType::Float: return "float";
    case ValueType::Vec2: return "vec2";
    case ValueType::Image2D: return "image2d";
    case ValueType::AnyNumeric: return "numeric";
    case ValueType::AnyVector: return "vector";
    }
    return "unknown";
}

} // namespace reaction
