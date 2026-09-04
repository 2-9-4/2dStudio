#pragma once

#include <array>
#include <string_view>

namespace reaction {

// These numeric values are persisted in project files. Append new operations;
// never reorder the existing entries.
enum class MathOperation : int {
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Divide = 3,
    Power = 4,
    Minimum = 5,
    Maximum = 6,
    Absolute = 7,
    Sine = 8,
    Cosine = 9,
    Clamp = 10,
    Remap = 11,
};

inline constexpr std::array<std::string_view, 12> kMathOperationNames = {
    "Add", "Subtract", "Multiply", "Divide", "Power", "Minimum",
    "Maximum", "Absolute", "Sine", "Cosine", "Clamp", "Remap"};

[[nodiscard]] MathOperation mathOperation(float persistedValue);
[[nodiscard]] std::string_view mathOperationName(MathOperation operation);
[[nodiscard]] int mathOperationOperandCount(MathOperation operation);
[[nodiscard]] float applyMathOperation(MathOperation operation, float a, float b, float c,
                                       float inMin, float inMax,
                                       float outMin, float outMax);

} // namespace reaction
