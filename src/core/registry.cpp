#include "reaction/core/node.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace reaction {

ValueType disconnectedType(SocketContract contract) {
    if (const auto exact = fixedType(contract)) return *exact;
    switch (contract) {
    case SocketContract::Numeric: return ValueType::Float;
    case SocketContract::VectorNumeric: return ValueType::Vec2;
    case SocketContract::AnyField: return ValueType::ColorImage;
    case SocketContract::AnyImageValue: return ValueType::Float;
    default: return ValueType::Float;
    }
}

std::vector<std::string> typePolicyInputs(const NodeDescriptor& descriptor,
                                          const SocketDescriptor& output) {
    if (!output.typeInputs.empty()) return output.typeInputs;
    std::vector<std::string> result;
    for (const auto& input : descriptor.sockets) {
        if (input.direction != SocketDirection::Input) continue;
        const bool relevant = output.contract == SocketContract::Numeric
            ? input.contract == SocketContract::Numeric
            : output.contract == SocketContract::VectorNumeric
            ? input.contract == SocketContract::VectorNumeric
            : output.contract == SocketContract::AnyImageValue
            ? input.contract == SocketContract::AnyImageValue
            : output.contract == SocketContract::AnyField
            ? input.contract == SocketContract::AnyField : false;
        if (relevant) result.push_back(input.key);
    }
    return result;
}

SocketDescriptor::TypePolicy effectiveTypePolicy(const SocketDescriptor& output) {
    if (output.typePolicy != SocketDescriptor::TypePolicy::Fixed) return output.typePolicy;
    if (fixedType(output.contract)) return SocketDescriptor::TypePolicy::Fixed;
    if (output.contract == SocketContract::Numeric)
        return SocketDescriptor::TypePolicy::NumericPromotion;
    if (output.contract == SocketContract::VectorNumeric)
        return SocketDescriptor::TypePolicy::VectorPromotion;
    if (output.contract == SocketContract::AnyField)
        return SocketDescriptor::TypePolicy::PreserveInput;
    return SocketDescriptor::TypePolicy::WidestValue;
}

ValueType resolveOutputType(
    const NodeDescriptor& descriptor, const SocketDescriptor& output,
    const std::function<std::optional<ValueType>(std::string_view)>& inputType) {
    ValueType resolved = disconnectedType(output.contract);
    std::vector<ValueType> operands;
    for (const auto& key : typePolicyInputs(descriptor, output)) {
        if (const auto type = inputType(key)) operands.push_back(*type);
    }
    switch (effectiveTypePolicy(output)) {
    case SocketDescriptor::TypePolicy::Fixed:
        resolved = fixedType(output.contract).value_or(resolved);
        break;
    case SocketDescriptor::TypePolicy::NumericPromotion:
        resolved = std::ranges::any_of(operands, isFieldType)
            ? ValueType::ScalarField : ValueType::Float;
        break;
    case SocketDescriptor::TypePolicy::VectorPromotion:
        resolved = std::ranges::any_of(operands, isFieldType)
            ? ValueType::VectorField : ValueType::Vec2;
        break;
    case SocketDescriptor::TypePolicy::WidestValue:
        if (!operands.empty()) resolved = widestValue(operands);
        break;
    case SocketDescriptor::TypePolicy::PreserveInput:
        if (!operands.empty()) resolved = operands.front();
        break;
    }
    if (!isFieldType(resolved) && !output.fieldInputs.empty()) {
        const bool spatial = std::ranges::any_of(output.fieldInputs,
            [&](const std::string& key) {
                const auto type = inputType(key);
                return type && isFieldType(*type);
            });
        if (spatial) resolved = fieldTypeForWidth(componentCount(resolved));
    }
    if (descriptor.producedField && !isFieldType(resolved))
        resolved = fieldTypeForWidth(componentCount(resolved));
    return resolved;
}

ValueType resolveInputType(
    const NodeDescriptor& descriptor, std::string_view inputKey, ValueType sourceType,
    const std::function<std::optional<ValueType>(std::string_view)>& outputType) {
    std::vector<ValueType> targets{sourceType};
    for (const auto& output : descriptor.sockets) {
        if (output.direction != SocketDirection::Output ||
            effectiveTypePolicy(output) != SocketDescriptor::TypePolicy::WidestValue)
            continue;
        const auto keys = typePolicyInputs(descriptor, output);
        if (std::ranges::find(keys, inputKey) == keys.end()) continue;
        if (const auto type = outputType(output.key)) targets.push_back(*type);
    }
    return widestValue(targets);
}

void addParameterInputSockets(NodeDescriptor& descriptor) {
    for (const auto& parameter : descriptor.parameters) {
        if (parameter.control != ParameterDescriptor::Control::Float &&
            parameter.control != ParameterDescriptor::Control::Integer) continue;
        const bool exists = std::ranges::any_of(descriptor.sockets, [&](const SocketDescriptor& socket) {
            return socket.direction == SocketDirection::Input && socket.key == parameter.key;
        });
        if (exists) continue;
        const auto position = std::ranges::find_if(descriptor.sockets, [](const SocketDescriptor& socket) {
            return socket.direction == SocketDirection::Output;
        });
        descriptor.sockets.insert(position, {parameter.key, parameter.label, ValueType::Float,
                                             SocketDirection::Input, true});
    }
}

void NodeRegistry::add(NodeDescriptor descriptor, Factory factory) {
    if (descriptor.type.empty() || !factory) {
        throw std::invalid_argument("A node registration needs a type and factory");
    }
    addParameterInputSockets(descriptor);
    if (descriptor.producedField && !descriptor.lowerable)
        throw std::invalid_argument("Node '" + descriptor.type +
                                    "' marks producedField but is not lowerable");
    for (const auto& socket : descriptor.sockets) {
        if (socket.requiresImage && socket.direction != SocketDirection::Input)
            throw std::invalid_argument("Node '" + descriptor.type +
                                        "' marks a non-input socket as requiresImage");
    }
    if (!descriptor.neighborhoodSocket.empty()) {
        const auto socket = std::ranges::find_if(descriptor.sockets, [&](const auto& item) {
            return item.direction == SocketDirection::Input &&
                   item.key == descriptor.neighborhoodSocket;
        });
        if (socket == descriptor.sockets.end() || !socket->requiresImage)
            throw std::invalid_argument("Neighborhood socket '" +
                descriptor.neighborhoodSocket + "' on node '" + descriptor.type +
                "' must be a requiresImage input");
    }
    std::unordered_set<std::string> parameterKeys;
    for (const auto& parameter : descriptor.parameters) {
        if (parameter.key.empty() || parameter.label.empty())
            throw std::invalid_argument("Node '" + descriptor.type + "' has an unnamed parameter");
        if (!parameterKeys.insert(parameter.key).second)
            throw std::invalid_argument("Node '" + descriptor.type +
                                        "' has duplicate parameter key '" + parameter.key + "'");
        if (parameter.control == ParameterDescriptor::Control::Enum) {
            const int expected = static_cast<int>(parameter.maximum - parameter.minimum) + 1;
            if (parameter.minimum != 0.0F || parameter.enumOptions.empty() ||
                static_cast<int>(parameter.enumOptions.size()) != expected) {
                throw std::invalid_argument("Enum parameter '" + parameter.key + "' on node '" +
                                            descriptor.type +
                                            "' needs labels for every zero-based value");
            }
        }
        if (parameter.control == ParameterDescriptor::Control::Float ||
            parameter.control == ParameterDescriptor::Control::Integer) {
            const auto socket = std::ranges::find_if(descriptor.sockets, [&](const auto& item) {
                return item.direction == SocketDirection::Input && item.key == parameter.key;
            });
            if (socket == descriptor.sockets.end() ||
                (socket->contract != SocketContract::FloatOnly &&
                 socket->contract != SocketContract::Numeric &&
                 socket->contract != SocketContract::AnyImageValue)) {
                throw std::invalid_argument("Numeric parameter '" + parameter.key + "' on node '" +
                                            descriptor.type +
                                            "' needs a Float, Numeric, or AnyImageValue input with the same key");
            }
        }
    }
    const auto key = descriptor.type;
    if (!entries_.emplace(key, Entry{std::move(descriptor), std::move(factory)}).second) {
        throw std::invalid_argument("Duplicate node type: " + key);
    }
}

bool NodeRegistry::contains(std::string_view type) const {
    return entries_.contains(std::string(type));
}

const NodeDescriptor* NodeRegistry::descriptor(std::string_view type) const {
    const auto it = entries_.find(std::string(type));
    return it == entries_.end() ? nullptr : &it->second.descriptor;
}

std::unique_ptr<NodeInstance> NodeRegistry::create(std::string_view type) const {
    const auto it = entries_.find(std::string(type));
    return it == entries_.end() ? nullptr : it->second.factory();
}

std::vector<const NodeDescriptor*> NodeRegistry::descriptors() const {
    std::vector<const NodeDescriptor*> result;
    result.reserve(entries_.size());
    for (const auto& [_, entry] : entries_) result.push_back(&entry.descriptor);
    std::ranges::sort(result, {}, [](const NodeDescriptor* item) {
        return item->category + item->displayName;
    });
    return result;
}

} // namespace reaction
