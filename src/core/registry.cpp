#include "reaction/core/node.hpp"

#include <algorithm>
#include <stdexcept>

namespace reaction {

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

