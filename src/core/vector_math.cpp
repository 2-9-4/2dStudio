#include "reaction/core/vector_math.hpp"

#include <algorithm>
#include <array>

namespace reaction {
namespace {

constexpr std::array<std::string_view, 33> kNames{
    "Add", "Subtract", "Multiply Components", "Divide Components", "Scale", "Negate",
    "Absolute", "Minimum", "Maximum", "Clamp", "Normalize", "Set Length",
    "Clamp Length", "Limit Length", "Rotate", "Perpendicular Clockwise",
    "Perpendicular Counter-Clockwise", "Reflect", "Project", "Reject", "Mix",
    "Length", "Length Squared", "Distance", "Distance Squared", "Dot", "Cross",
    "Angle", "Angle Between", "Minimum Component", "Maximum Component",
    "Sum Components", "Product Components"};

bool needsB(VectorMathOperation operation) {
    switch (operation) {
    case VectorMathOperation::Add: case VectorMathOperation::Subtract:
    case VectorMathOperation::MultiplyComponents: case VectorMathOperation::DivideComponents:
    case VectorMathOperation::Minimum: case VectorMathOperation::Maximum:
    case VectorMathOperation::Reflect: case VectorMathOperation::Project:
    case VectorMathOperation::Reject: case VectorMathOperation::Mix:
    case VectorMathOperation::Distance: case VectorMathOperation::DistanceSquared:
    case VectorMathOperation::Dot: case VectorMathOperation::Cross:
    case VectorMathOperation::AngleBetween: return true;
    default: return false;
    }
}

bool scalarResult(VectorMathOperation operation) {
    switch (operation) {
    case VectorMathOperation::Length: case VectorMathOperation::LengthSquared:
    case VectorMathOperation::Distance: case VectorMathOperation::DistanceSquared:
    case VectorMathOperation::Dot: case VectorMathOperation::Cross: case VectorMathOperation::Angle:
    case VectorMathOperation::AngleBetween: case VectorMathOperation::MinimumComponent:
    case VectorMathOperation::MaximumComponent: case VectorMathOperation::SumComponents:
    case VectorMathOperation::ProductComponents: return true;
    default: return false;
    }
}

void addNumeric(NodeDescriptor& descriptor, const char* key, const char* label,
                float value, float minimum = -100.0F, float maximum = 100.0F) {
    descriptor.sockets.push_back({key, label, ValueType::AnyNumeric, SocketDirection::Input, true});
    descriptor.parameters.push_back({key, label, value, minimum, maximum});
}

} // namespace

VectorMathOperation vectorMathOperation(float persistedValue) {
    return static_cast<VectorMathOperation>(std::clamp(static_cast<int>(persistedValue), 0,
        static_cast<int>(kNames.size()) - 1));
}

std::string_view vectorMathOperationName(VectorMathOperation operation) {
    return kNames[static_cast<std::size_t>(operation)];
}

NodeDescriptor vectorMathDescriptor(VectorMathOperation operation) {
    std::vector<std::string> names;
    names.reserve(kNames.size());
    for (const auto name : kNames) names.emplace_back(name);
    NodeDescriptor result{"vector_math", 1, "Vector Math", "Math", {},
        {{"operation", "Operation", static_cast<float>(operation), 0.0F,
          static_cast<float>(kNames.size() - 1), ParameterDescriptor::Control::Enum,
          std::move(names)}}};
    result.sockets.push_back({"a", "A", ValueType::AnyVector, SocketDirection::Input, true, true});
    if (needsB(operation))
        result.sockets.push_back({"b", "B", ValueType::AnyVector, SocketDirection::Input, true, true});

    switch (operation) {
    case VectorMathOperation::Scale: case VectorMathOperation::SetLength:
    case VectorMathOperation::LimitLength: case VectorMathOperation::Rotate:
        addNumeric(result, "scalar", operation == VectorMathOperation::Rotate ? "Angle" : "Scalar",
                   operation == VectorMathOperation::Rotate ? 0.0F : 1.0F);
        break;
    case VectorMathOperation::ClampLength:
        addNumeric(result, "min", "Min", 0.0F);
        addNumeric(result, "max", "Max", 1.0F);
        break;
    case VectorMathOperation::Clamp:
        addNumeric(result, "min", "Min", 0.0F);
        addNumeric(result, "max", "Max", 1.0F);
        break;
    case VectorMathOperation::Mix:
        addNumeric(result, "scalar", "Factor", 0.5F);
        break;
    default: break;
    }
    result.sockets.push_back({"result", "Result",
        scalarResult(operation) ? ValueType::AnyNumeric : ValueType::AnyVector,
        SocketDirection::Output, false, true});
    result.lowerable = true;
    return result;
}

} // namespace reaction
