#include "reaction/core/math.hpp"

#include <algorithm>
#include <cmath>

namespace reaction {

MathOperation mathOperation(float persistedValue) {
    return static_cast<MathOperation>(std::clamp(
        static_cast<int>(persistedValue), 0, static_cast<int>(kMathOperationNames.size()) - 1));
}

std::string_view mathOperationName(MathOperation operation) {
    return kMathOperationNames[static_cast<std::size_t>(operation)];
}

int mathOperationOperandCount(MathOperation operation) {
    switch (operation) {
    case MathOperation::Absolute:
    case MathOperation::Sine:
    case MathOperation::Cosine:
    case MathOperation::Remap:
    case MathOperation::Floor:
    case MathOperation::Ceil:
    case MathOperation::Round:
    case MathOperation::Fraction:
    case MathOperation::SquareRoot:
    case MathOperation::Exp:
    case MathOperation::NaturalLog:
    case MathOperation::Log2:
    case MathOperation::Sign:
    case MathOperation::Tangent:
    case MathOperation::ArcSine:
    case MathOperation::ArcCosine:
    case MathOperation::ArcTangent: return 1;
    case MathOperation::Clamp:
    case MathOperation::Smoothstep:
    case MathOperation::MultiplyAccumulate: return 3;
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
    case MathOperation::Floor: return std::floor(a);
    case MathOperation::Ceil: return std::ceil(a);
    case MathOperation::Round: return std::round(a);
    case MathOperation::Fraction: return a - std::floor(a);
    case MathOperation::SquareRoot: return std::sqrt(std::max(a, 0.0F));
    case MathOperation::Exp: return std::exp(a);
    case MathOperation::NaturalLog: return std::log(std::max(a, 1.0e-6F));
    case MathOperation::Log2: return std::log2(std::max(a, 1.0e-6F));
    case MathOperation::Sign: return a > 0.0F ? 1.0F : (a < 0.0F ? -1.0F : 0.0F);
    case MathOperation::Tangent: return std::tan(a);
    case MathOperation::ArcSine: return std::asin(std::clamp(a, -1.0F, 1.0F));
    case MathOperation::ArcCosine: return std::acos(std::clamp(a, -1.0F, 1.0F));
    case MathOperation::ArcTangent: return std::atan(a);
    case MathOperation::Modulo: {
        const auto divisor = std::abs(b) < 1.0e-6F
            ? std::copysign(1.0e-6F, b == 0.0F ? 1.0F : b) : b;
        return a - divisor * std::floor(a / divisor);
    }
    case MathOperation::ArcTangent2: return std::atan2(a, b);
    case MathOperation::Step: return b >= a ? 1.0F : 0.0F;
    case MathOperation::Hypotenuse: return std::hypot(a, b);
    case MathOperation::Smoothstep: {
        const auto t = std::clamp((a - b) / std::max(c - b, 1.0e-6F), 0.0F, 1.0F);
        return t * t * (3.0F - 2.0F * t);
    }
    case MathOperation::MultiplyAccumulate: return a * b + c;
    }
    return a;
}

} // namespace reaction
