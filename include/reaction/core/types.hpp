#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>

namespace reaction {

using NodeId = std::uint64_t;
using LinkId = std::uint64_t;
using SocketId = std::uint64_t;

// Concrete semantic graph values. Field type is deliberately independent of
// the texture format used to store a materialized value.
enum class ValueType { Float, Vec2, ScalarField, VectorField, ColorImage };

// A socket contract says what is semantically admissible. It is not a request
// to perform every conversion that happens to exist in the widening table.
enum class SocketContract {
    FloatOnly,
    Vec2Only,
    ScalarFieldOnly,
    VectorFieldOnly,
    ColorImageOnly,
    Numeric,
    VectorNumeric,
    AnyField,
    AnyImageValue,
};

enum class Coercion {
    Identity,
    FloatToVec2,
    FloatToScalarField,
    FloatToVectorField,
    FloatToColorImage,
    Vec2ToVectorField,
    Vec2ToColorImage,
    ScalarFieldToVectorField,
    ScalarFieldToColorImage,
    VectorFieldToColorImage,
};

enum class SocketDirection { Input, Output };

struct ImageHandle {
    std::uint32_t texture = 0;
    int width = 0;
    int height = 0;
    ValueType semanticType = ValueType::ColorImage;
    explicit operator bool() const noexcept { return texture != 0; }
};

struct Vec2 {
    float x = 0.0F;
    float y = 0.0F;
};

using Value = std::variant<std::monostate, float, Vec2, ImageHandle>;

[[nodiscard]] constexpr bool isFieldType(ValueType type) noexcept {
    return type == ValueType::ScalarField || type == ValueType::VectorField ||
           type == ValueType::ColorImage;
}

[[nodiscard]] constexpr int componentCount(ValueType type) noexcept {
    switch (type) {
    case ValueType::Float:
    case ValueType::ScalarField: return 1;
    case ValueType::Vec2:
    case ValueType::VectorField: return 2;
    case ValueType::ColorImage: return 4;
    }
    return 1;
}

[[nodiscard]] constexpr ValueType fieldTypeForWidth(int width) noexcept {
    return width <= 1 ? ValueType::ScalarField
         : width <= 2 ? ValueType::VectorField : ValueType::ColorImage;
}

[[nodiscard]] constexpr SocketContract exactContract(ValueType type) noexcept {
    switch (type) {
    case ValueType::Float: return SocketContract::FloatOnly;
    case ValueType::Vec2: return SocketContract::Vec2Only;
    case ValueType::ScalarField: return SocketContract::ScalarFieldOnly;
    case ValueType::VectorField: return SocketContract::VectorFieldOnly;
    case ValueType::ColorImage: return SocketContract::ColorImageOnly;
    }
    return SocketContract::FloatOnly;
}

[[nodiscard]] constexpr std::optional<ValueType> fixedType(
    SocketContract contract) noexcept {
    switch (contract) {
    case SocketContract::FloatOnly: return ValueType::Float;
    case SocketContract::Vec2Only: return ValueType::Vec2;
    case SocketContract::ScalarFieldOnly: return ValueType::ScalarField;
    case SocketContract::VectorFieldOnly: return ValueType::VectorField;
    case SocketContract::ColorImageOnly: return ValueType::ColorImage;
    default: return std::nullopt;
    }
}

[[nodiscard]] constexpr bool contractAccepts(SocketContract contract,
                                              ValueType type) noexcept {
    if (const auto exact = fixedType(contract)) return *exact == type;
    switch (contract) {
    case SocketContract::Numeric:
        return type == ValueType::Float || type == ValueType::ScalarField;
    case SocketContract::VectorNumeric:
        return type == ValueType::Vec2 || type == ValueType::VectorField;
    case SocketContract::AnyField: return isFieldType(type);
    case SocketContract::AnyImageValue: return true;
    default: return false;
    }
}

[[nodiscard]] constexpr std::optional<Coercion> coercionBetween(
    ValueType from, ValueType to) noexcept {
    if (from == to) return Coercion::Identity;
    if (from == ValueType::Float && to == ValueType::Vec2) return Coercion::FloatToVec2;
    if (from == ValueType::Float && to == ValueType::ScalarField)
        return Coercion::FloatToScalarField;
    if (from == ValueType::Float && to == ValueType::VectorField)
        return Coercion::FloatToVectorField;
    if (from == ValueType::Float && to == ValueType::ColorImage)
        return Coercion::FloatToColorImage;
    if (from == ValueType::Vec2 && to == ValueType::VectorField)
        return Coercion::Vec2ToVectorField;
    if (from == ValueType::Vec2 && to == ValueType::ColorImage)
        return Coercion::Vec2ToColorImage;
    if (from == ValueType::ScalarField && to == ValueType::VectorField)
        return Coercion::ScalarFieldToVectorField;
    if (from == ValueType::ScalarField && to == ValueType::ColorImage)
        return Coercion::ScalarFieldToColorImage;
    if (from == ValueType::VectorField && to == ValueType::ColorImage)
        return Coercion::VectorFieldToColorImage;
    return std::nullopt;
}

[[nodiscard]] constexpr ValueType widestValue(std::span<const ValueType> values) noexcept {
    bool field = false;
    int width = 1;
    for (const auto value : values) {
        field |= isFieldType(value);
        width = componentCount(value) > width ? componentCount(value) : width;
    }
    if (field) return fieldTypeForWidth(width);
    return width > 1 ? ValueType::Vec2 : ValueType::Float;
}

[[nodiscard]] inline std::string toString(ValueType type) {
    switch (type) {
    case ValueType::Float: return "float";
    case ValueType::Vec2: return "vec2";
    case ValueType::ScalarField: return "scalar_field";
    case ValueType::VectorField: return "vector_field";
    case ValueType::ColorImage: return "color_image";
    }
    return "unknown";
}

[[nodiscard]] inline std::string toString(SocketContract contract) {
    switch (contract) {
    case SocketContract::FloatOnly: return "float";
    case SocketContract::Vec2Only: return "vec2";
    case SocketContract::ScalarFieldOnly: return "scalar_field";
    case SocketContract::VectorFieldOnly: return "vector_field";
    case SocketContract::ColorImageOnly: return "color_image";
    case SocketContract::Numeric: return "numeric";
    case SocketContract::VectorNumeric: return "vector_numeric";
    case SocketContract::AnyField: return "any_field";
    case SocketContract::AnyImageValue: return "any_image_value";
    }
    return "unknown";
}

[[nodiscard]] inline std::string coercionDescription(Coercion coercion) {
    switch (coercion) {
    case Coercion::Identity: return "Identity";
    case Coercion::FloatToVec2: return "s -> (s, s)";
    case Coercion::FloatToScalarField: return "s -> scalar field (uniform broadcast)";
    case Coercion::FloatToVectorField: return "s -> (s, s) (uniform broadcast)";
    case Coercion::FloatToColorImage: return "s -> (s, s, s, 1) (uniform broadcast)";
    case Coercion::Vec2ToVectorField: return "(x, y) -> vector field (uniform broadcast)";
    case Coercion::Vec2ToColorImage: return "(x, y) -> (x, y, 0, 1) (uniform broadcast)";
    case Coercion::ScalarFieldToVectorField: return "s -> (s, s)";
    case Coercion::ScalarFieldToColorImage: return "s -> (s, s, s, 1)";
    case Coercion::VectorFieldToColorImage: return "(x, y) -> (x, y, 0, 1)";
    }
    return "Unknown conversion";
}

[[nodiscard]] inline std::span<const ValueType> acceptedTypes(SocketContract contract) {
    static constexpr std::array floatOnly{ValueType::Float};
    static constexpr std::array vec2Only{ValueType::Vec2};
    static constexpr std::array scalarOnly{ValueType::ScalarField};
    static constexpr std::array vectorOnly{ValueType::VectorField};
    static constexpr std::array colorOnly{ValueType::ColorImage};
    static constexpr std::array numeric{ValueType::Float, ValueType::ScalarField};
    static constexpr std::array vectorNumeric{ValueType::Vec2, ValueType::VectorField};
    static constexpr std::array fields{ValueType::ScalarField, ValueType::VectorField,
                                       ValueType::ColorImage};
    static constexpr std::array any{ValueType::Float, ValueType::Vec2,
                                    ValueType::ScalarField, ValueType::VectorField,
                                    ValueType::ColorImage};
    switch (contract) {
    case SocketContract::FloatOnly: return floatOnly;
    case SocketContract::Vec2Only: return vec2Only;
    case SocketContract::ScalarFieldOnly: return scalarOnly;
    case SocketContract::VectorFieldOnly: return vectorOnly;
    case SocketContract::ColorImageOnly: return colorOnly;
    case SocketContract::Numeric: return numeric;
    case SocketContract::VectorNumeric: return vectorNumeric;
    case SocketContract::AnyField: return fields;
    case SocketContract::AnyImageValue: return any;
    }
    return {};
}

} // namespace reaction
