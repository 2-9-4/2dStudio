#pragma once

#include "reaction/core/types.hpp"

#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace reaction {

struct SocketDescriptor {
    std::string key;
    std::string label;
    ValueType type = ValueType::Float;
    SocketDirection direction = SocketDirection::Input;
    bool optional = false;
};

struct ParameterDescriptor {
    enum class Control { Float, Integer, Boolean };
    std::string key;
    std::string label;
    float defaultValue = 0.0F;
    float minimum = 0.0F;
    float maximum = 1.0F;
    Control control = Control::Float;
};

struct NodeDescriptor {
    std::string type;
    int version = 1;
    std::string displayName;
    std::string category;
    std::vector<SocketDescriptor> sockets;
    std::vector<ParameterDescriptor> parameters;
    bool timeDependent = false;
    bool stateful = false;
};

struct EvaluationContext {
    int width = 1024;
    int height = 1024;
    double time = 0.0;
    double deltaTime = 0.0;
    bool playing = true;
    void* gpu = nullptr;
};

class NodeInstance {
public:
    virtual ~NodeInstance() = default;
    [[nodiscard]] virtual const NodeDescriptor& descriptor() const = 0;
    virtual void evaluate(EvaluationContext& context,
                          std::span<const Value> inputs,
                          std::span<Value> outputs) = 0;
    virtual void reset(EvaluationContext&) {}
    [[nodiscard]] virtual nlohmann::json parameters() const = 0;
    virtual void setParameters(const nlohmann::json& values) = 0;
};

class NodeRegistry {
public:
    using Factory = std::function<std::unique_ptr<NodeInstance>()>;

    void add(NodeDescriptor descriptor, Factory factory);
    [[nodiscard]] bool contains(std::string_view type) const;
    [[nodiscard]] const NodeDescriptor* descriptor(std::string_view type) const;
    [[nodiscard]] std::unique_ptr<NodeInstance> create(std::string_view type) const;
    [[nodiscard]] std::vector<const NodeDescriptor*> descriptors() const;

private:
    struct Entry { NodeDescriptor descriptor; Factory factory; };
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace reaction
