#pragma once

// Shared runtime conventions for procedural nodes.  Per-pixel scalar data is
// read from R, vector data from RG, and all generated textures remain RGBA16F.
// This deliberately mirrors the graph's Float/Vec2 broadcasting rules.

#include "node_support.hpp"

#include <cmath>
#include <span>

namespace reaction::procedural {

inline constexpr float kEpsilon = 1.0e-6F;
inline constexpr float kTwoPi = 6.28318530717958647692F;

struct NumericInput {
    ImageHandle field{};
    float constant = 0.0F;
    [[nodiscard]] bool isField() const noexcept { return static_cast<bool>(field); }
};

struct VectorInput {
    ImageHandle field{};
    Vec2 constant{};
    bool canvasCoordinates = false;
    [[nodiscard]] bool isField() const noexcept {
        return static_cast<bool>(field) || canvasCoordinates;
    }
};

inline NumericInput numericInput(std::span<const Value> values, std::size_t index,
                                 float fallback = 0.0F) {
    NumericInput result;
    result.constant = fallback;
    if (index >= values.size()) return result;
    if (const auto* image = std::get_if<ImageHandle>(&values[index])) result.field = *image;
    else if (const auto* value = std::get_if<float>(&values[index])) result.constant = *value;
    return result;
}

// Parameter-backed Numeric sockets must use this helper. It makes the slider
// value the unconnected input value instead of requiring every node author to
// remember to read parameters_ separately.
inline NumericInput numericParameterInput(std::span<const Value> values, std::size_t index,
                                          const nlohmann::json& parameters,
                                          const char* key, float fallback) {
    return numericInput(values, index, node_support::parameter(parameters, key, fallback));
}

inline VectorInput vectorInput(std::span<const Value> values, std::size_t index,
                               Vec2 fallback = {}) {
    VectorInput result;
    result.constant = fallback;
    if (index >= values.size()) return result;
    if (const auto* image = std::get_if<ImageHandle>(&values[index])) result.field = *image;
    else if (const auto* value = std::get_if<Vec2>(&values[index])) result.constant = *value;
    // A scalar is valid Vector Numeric input and broadcasts to both components.
    else if (const auto* value = std::get_if<float>(&values[index])) result.constant = {*value, *value};
    return result;
}

// Spatial coordinate sockets default to a generated canvas UV field. This is
// represented explicitly, so it cannot accidentally sample an unbound/reused texture.
inline VectorInput coordinateInput(std::span<const Value> values, std::size_t index) {
    auto result = vectorInput(values, index);
    if (index >= values.size() || std::holds_alternative<std::monostate>(values[index]))
        result.canvasCoordinates = true;
    return result;
}

[[nodiscard]] inline float safeSignedDenominator(float value) noexcept {
    return std::abs(value) < kEpsilon ? std::copysign(kEpsilon, value == 0.0F ? 1.0F : value) : value;
}

[[nodiscard]] inline float fract(float value) noexcept { return value - std::floor(value); }
[[nodiscard]] inline float positivePeriod(float period) noexcept {
    return std::abs(period) < kEpsilon ? std::copysign(kEpsilon, period == 0.0F ? 1.0F : period) : period;
}

// Bind a broadcastable numeric/vector input for compute shaders. `hasField`
// tells GLSL whether to sample `samplerName`; otherwise `constantName` is used.
inline void bindNumericInput(GLuint program, int textureUnit, const char* samplerName,
                             const char* hasFieldName, const char* constantName,
                             const NumericInput& input) {
    if (input.isField()) node_support::bindTexture(textureUnit, input.field.texture);
    node_support::uniform(program, samplerName, textureUnit);
    node_support::uniform(program, hasFieldName, input.isField() ? 1 : 0);
    node_support::uniform(program, constantName, input.constant);
}

inline void bindVectorInput(GLuint program, int textureUnit, const char* samplerName,
                            const char* hasFieldName, const char* constantName,
                            const VectorInput& input) {
    if (input.field) node_support::bindTexture(textureUnit, input.field.texture);
    node_support::uniform(program, samplerName, textureUnit);
    node_support::uniform(program, hasFieldName,
                          input.field ? 1 : (input.canvasCoordinates ? -1 : 0));
    glUniform2f(glGetUniformLocation(program, constantName), input.constant.x, input.constant.y);
}

inline void bindColorInput(GLuint program, int textureUnit, const char* samplerName,
                           const char* hasImageName, ImageHandle image) {
    if (image) node_support::bindTexture(textureUnit, image.texture);
    node_support::uniform(program, samplerName, textureUnit);
    node_support::uniform(program, hasImageName, image ? 1 : 0);
}

// Include this in compute shaders that use the binding helpers above.
inline constexpr std::string_view kGlslBroadcastHelpers = R"GLSL(
float proceduralScalar(sampler2D source, int hasField, float constantValue, vec2 uv) {
    return hasField != 0 ? texture(source, uv).r : constantValue;
}
vec2 proceduralVector(sampler2D source, int hasField, vec2 constantValue, vec2 uv) {
    return hasField > 0 ? texture(source, uv).rg :
           (hasField < 0 ? uv : constantValue);
}
)GLSL";

} // namespace reaction::procedural
