#pragma once

#include "reaction/core/types.hpp"

#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

namespace reaction {

class ShaderLoweringContext;
struct NodeDescriptor;

// Exposes every slider-style parameter (Float or Integer control) as an optional
// input socket, so a connection can override the slider value. Enum selectors,
// checkboxes, and node-specific editors are left socket-free.
void addParameterInputSockets(NodeDescriptor& descriptor);

struct SocketDescriptor {
    std::string key;
    std::string label;
    ValueType type = ValueType::Float;
    SocketDirection direction = SocketDirection::Input;
    bool optional = false;
    // Sampling accessors require a real field on this input. Constants are a
    // typed lowering error instead of being silently broadcast.
    bool requiresImage = false;
};

struct ParameterDescriptor {
    enum class Control { Float, Integer, Boolean, Enum };
    std::string key;
    std::string label;
    float defaultValue = 0.0F;
    float minimum = 0.0F;
    float maximum = 1.0F;
    Control control = Control::Float;
    std::vector<std::string> enumOptions;

    ParameterDescriptor() = default;
    ParameterDescriptor(std::string keyValue, std::string labelValue, float defaultValue,
                        float minimum, float maximum, Control control = Control::Float)
        : key(std::move(keyValue)), label(std::move(labelValue)), defaultValue(defaultValue),
          minimum(minimum), maximum(maximum), control(control) {}
    ParameterDescriptor(std::string keyValue, std::string labelValue, float defaultValue,
                        float minimum, float maximum, Control control,
                        std::vector<std::string> options)
        : key(std::move(keyValue)), label(std::move(labelValue)), defaultValue(defaultValue),
          minimum(minimum), maximum(maximum), control(control), enumOptions(std::move(options)) {}
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
    // Lowerable nodes execute through generated shader regions. A particular
    // instance may still opt out for parameter-dependent native escape hatches.
    bool lowerable = false;
    // Pure generators that read canvas position have field outputs even when
    // all of their explicit inputs are constants.
    bool producedField = false;
    // When non-empty, links feeding this socket sample a pixel neighborhood and
    // therefore form a materialization boundary in generated shader regions.
    std::string neighborhoodSocket{};
};

struct EvaluationContext {
    int width = 1024;
    int height = 1024;
    double time = 0.0;
    double deltaTime = 0.0;
    std::uint64_t frame = 0;
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
    // Lets multi-output nodes avoid materializing outputs that have no consumers.
    // Implementations must leave required outputs in their descriptor order.
    virtual void setOutputRequirements(const std::vector<bool>&) {}
    // GPU-capable nodes may lower themselves into a generated shader region.
    // The default keeps existing and third-party node implementations unfused.
    virtual bool lowerShader(ShaderLoweringContext&) const { return false; }
    // Contributes a stable specialization key so parameter edits that change the
    // emitted GLSL (baked branches, unrolled loops, const kernels) rebuild regions.
    virtual std::string shaderVariantKey(const nlohmann::json&) const { return {}; }
    // A lowerable node may still refuse fusion for a given parameter set (for
    // example, multi-pass iterations). Such nodes stay materialized boundaries.
    virtual bool supportsRegionFusion(const nlohmann::json&) const { return true; }
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
