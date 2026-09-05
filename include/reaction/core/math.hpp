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
    Floor = 12,
    Ceil = 13,
    Round = 14,
    Fraction = 15,
    SquareRoot = 16,
    Exp = 17,
    NaturalLog = 18,
    Log2 = 19,
    Sign = 20,
    Tangent = 21,
    ArcSine = 22,
    ArcCosine = 23,
    ArcTangent = 24,
    Modulo = 25,
    ArcTangent2 = 26,
    Step = 27,
    Hypotenuse = 28,
    Smoothstep = 29,
    MultiplyAccumulate = 30,
};

inline constexpr std::array<std::string_view, 31> kMathOperationNames = {
    "Add", "Subtract", "Multiply", "Divide", "Power", "Minimum",
    "Maximum", "Absolute", "Sine", "Cosine", "Clamp", "Remap",
    "Floor", "Ceil", "Round", "Fraction", "Square Root", "Exp",
    "Natural Log", "Log2", "Sign", "Tangent", "Arc Sine", "Arc Cosine",
    "Arc Tangent", "Modulo", "Arc Tangent 2", "Step", "Hypotenuse",
    "Smoothstep", "Multiply Accumulate"};

[[nodiscard]] MathOperation mathOperation(float persistedValue);
[[nodiscard]] std::string_view mathOperationName(MathOperation operation);
[[nodiscard]] int mathOperationOperandCount(MathOperation operation);
[[nodiscard]] float applyMathOperation(MathOperation operation, float a, float b, float c,
                                       float inMin, float inMax,
                                       float outMin, float outMax);

} // namespace reaction
