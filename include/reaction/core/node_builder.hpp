#pragma once

#include "reaction/core/node.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace reaction {

// Small declarative builder for NodeDescriptor metadata. It deliberately
// produces the existing descriptor structs rather than introducing a second
// metadata model.
class NodeDescriptorBuilder {
public:
    NodeDescriptorBuilder(std::string type, int version,
                          std::string displayName, std::string category)
        : descriptor_{std::move(type), version, std::move(displayName),
                      std::move(category), {}, {}} {}

    NodeDescriptorBuilder& input(std::string key, std::string label,
                                 SocketContract contract) {
        descriptor_.sockets.emplace_back(std::move(key), std::move(label), contract,
                                         SocketDirection::Input);
        return *this;
    }
    NodeDescriptorBuilder& input(std::string key, std::string label, ValueType type) {
        return input(std::move(key), std::move(label), exactContract(type));
    }
    NodeDescriptorBuilder& optionalInput(std::string key, std::string label,
                                         SocketContract contract) {
        descriptor_.sockets.emplace_back(std::move(key), std::move(label), contract,
                                         SocketDirection::Input, true);
        return *this;
    }
    NodeDescriptorBuilder& optionalInput(std::string key, std::string label,
                                         ValueType type) {
        return optionalInput(std::move(key), std::move(label), exactContract(type));
    }
    NodeDescriptorBuilder& output(std::string key, std::string label,
                                  SocketContract contract) {
        descriptor_.sockets.emplace_back(std::move(key), std::move(label), contract,
                                         SocketDirection::Output);
        return *this;
    }
    NodeDescriptorBuilder& output(std::string key, std::string label, ValueType type) {
        return output(std::move(key), std::move(label), exactContract(type));
    }

    NodeDescriptorBuilder& requireImage(std::string_view inputKey) {
        inputSocket(inputKey).requiresImage = true;
        return *this;
    }
    NodeDescriptorBuilder& fieldDefault(std::string_view inputKey) {
        inputSocket(inputKey).fieldDefault = true;
        return *this;
    }
    NodeDescriptorBuilder& typePolicy(
        std::string_view outputKey, SocketDescriptor::TypePolicy policy,
        std::initializer_list<std::string_view> inputs = {}) {
        auto& socket = outputSocket(outputKey);
        socket.typePolicy = policy;
        socket.typeInputs = strings(inputs);
        return *this;
    }
    NodeDescriptorBuilder& fieldIf(
        std::string_view outputKey,
        std::initializer_list<std::string_view> inputs) {
        outputSocket(outputKey).fieldInputs = strings(inputs);
        return *this;
    }

    NodeDescriptorBuilder& floatParameter(std::string key, std::string label,
                                          float defaultValue, float minimum,
                                          float maximum) {
        descriptor_.parameters.emplace_back(
            std::move(key), std::move(label), defaultValue, minimum, maximum,
            ParameterDescriptor::Control::Float);
        return *this;
    }
    NodeDescriptorBuilder& integerParameter(std::string key, std::string label,
                                            int defaultValue, int minimum,
                                            int maximum) {
        descriptor_.parameters.emplace_back(
            std::move(key), std::move(label), static_cast<float>(defaultValue),
            static_cast<float>(minimum), static_cast<float>(maximum),
            ParameterDescriptor::Control::Integer);
        return *this;
    }
    NodeDescriptorBuilder& booleanParameter(std::string key, std::string label,
                                            bool defaultValue = false) {
        descriptor_.parameters.emplace_back(
            std::move(key), std::move(label), defaultValue ? 1.0F : 0.0F,
            0.0F, 1.0F, ParameterDescriptor::Control::Boolean);
        return *this;
    }
    NodeDescriptorBuilder& enumParameter(
        std::string key, std::string label, int defaultIndex,
        std::initializer_list<std::string_view> options) {
        if (options.size() == 0)
            throw std::invalid_argument("Enum parameter needs at least one option");
        if (defaultIndex < 0 || defaultIndex >= static_cast<int>(options.size()))
            throw std::invalid_argument("Enum parameter default index is out of range");
        descriptor_.parameters.emplace_back(
            std::move(key), std::move(label), static_cast<float>(defaultIndex),
            0.0F, static_cast<float>(options.size() - 1),
            ParameterDescriptor::Control::Enum, strings(options));
        return *this;
    }

    NodeDescriptorBuilder& lowerable(bool value = true) {
        descriptor_.lowerable = value;
        return *this;
    }
    NodeDescriptorBuilder& producedField(bool value = true) {
        descriptor_.producedField = value;
        return *this;
    }
    NodeDescriptorBuilder& stateful(bool value = true) {
        descriptor_.stateful = value;
        return *this;
    }
    NodeDescriptorBuilder& timeDependent(bool value = true) {
        descriptor_.timeDependent = value;
        return *this;
    }
    NodeDescriptorBuilder& neighborhood(std::string_view inputKey) {
        auto& socket = inputSocket(inputKey);
        socket.requiresImage = true;
        descriptor_.neighborhoodSocket = std::string(inputKey);
        return *this;
    }

    [[nodiscard]] NodeDescriptor build() {
        return std::move(descriptor_);
    }

private:
    static std::vector<std::string> strings(
        std::initializer_list<std::string_view> values) {
        std::vector<std::string> result;
        result.reserve(values.size());
        for (const auto value : values) result.emplace_back(value);
        return result;
    }

    SocketDescriptor& inputSocket(std::string_view key) {
        return socket(key, SocketDirection::Input);
    }
    SocketDescriptor& outputSocket(std::string_view key) {
        return socket(key, SocketDirection::Output);
    }
    SocketDescriptor& socket(std::string_view key, SocketDirection direction) {
        const auto found = std::ranges::find_if(
            descriptor_.sockets, [&](const SocketDescriptor& candidate) {
                return candidate.key == key && candidate.direction == direction;
            });
        if (found == descriptor_.sockets.end()) {
            throw std::invalid_argument(
                std::string(direction == SocketDirection::Input ? "Input" : "Output") +
                " socket '" + std::string(key) + "' does not exist on node '" +
                descriptor_.type + "'");
        }
        return *found;
    }

    NodeDescriptor descriptor_;
};

} // namespace reaction
