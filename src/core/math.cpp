#include "reaction/core/math.hpp"

#include <algorithm>
#include <cmath>

namespace reaction {

MathOperation mathOperation(float persistedValue) {
    return static_cast<MathOperation>(std::clamp(static_cast<int>(persistedValue), 0, 11));
}

std::string_view mathOperationName(MathOperation operation) {
    return kMathOperationNames[static_cast<std::size_t>(operation)];
}

int mathOperationOperandCount(MathOperation operation) {
    switch (operation) {
    case MathOperation::Absolute:
    case MathOperation::Sine:
    case MathOperation::Cosine:
    case MathOperation::Remap: return 1;
    case MathOperation::Clamp: return 3;
    default: return 2;
    }
}

float applyMathOperation(MathOperation operation, float a, float b, float c,
                         float inMin, float inMax, float outMin, float outMax) {
    switch (operation) {
    case MathOperation::Add: return a + b;
    case MathOperation::Subtract: return a - b;
    case MathOperation::Multiply: return a * b;
    case MathOperation::Divide:
        return a / (std::abs(b) < 1.0e-6F
            ? std::copysign(1.0e-6F, b == 0 ? 1.0F : b) : b);
    case MathOperation::Power:
        return std::copysign(std::pow(std::max(std::abs(a), 1.0e-6F), b), a);
    case MathOperation::Minimum: return std::min(a, b);
    case MathOperation::Maximum: return std::max(a, b);
    case MathOperation::Absolute: return std::abs(a);
    case MathOperation::Sine: return std::sin(a);
    case MathOperation::Cosine: return std::cos(a);
    case MathOperation::Clamp: return std::clamp(a, b, c);
    case MathOperation::Remap: {
        const auto t = std::clamp((a - inMin) / std::max(inMax - inMin, 1.0e-6F),
                                  0.0F, 1.0F);
        return std::lerp(outMin, outMax, t);
    }
    }
    return a;
}

} // namespace reaction
