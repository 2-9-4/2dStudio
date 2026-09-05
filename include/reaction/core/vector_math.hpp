#pragma once

#include "reaction/core/node.hpp"

#include <string_view>

namespace reaction {

// Persisted numeric values: append new modes rather than reordering them.
enum class VectorMathOperation : int {
    Add, Subtract, MultiplyComponents, DivideComponents, Scale, Negate, Absolute,
    Minimum, Maximum, Clamp, Normalize, SetLength, ClampLength, LimitLength,
    Rotate, PerpendicularClockwise, PerpendicularCounterClockwise, Reflect, Project,
    Reject, Mix, Length, LengthSquared, Distance, DistanceSquared, Dot, Cross,
    Angle, AngleBetween, MinimumComponent, MaximumComponent, SumComponents,
    ProductComponents
};

[[nodiscard]] VectorMathOperation vectorMathOperation(float persistedValue);
[[nodiscard]] std::string_view vectorMathOperationName(VectorMathOperation operation);
[[nodiscard]] NodeDescriptor vectorMathDescriptor(VectorMathOperation operation);

} // namespace reaction
