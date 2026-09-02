#include "nodes_internal.hpp"
#include "node_support.hpp"

#include <memory>

namespace reaction {
namespace {

using node_support::ParameterNode;
using node_support::parameter;

class FloatNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        return {"float", 1, "Float", "Input",
                {{"value", "Value", ValueType::Float, SocketDirection::Output}},
                {{"value", "Value", 0.5F, -10.0F, 10.0F}}};
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext&, std::span<const Value>, std::span<Value> outputs) override {
        outputs[0] = parameter(parameters_, "value", 0.5F);
    }
};

class TimeNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"time", 1, "Time", "Input",
            {{"time", "Time", ValueType::Float, SocketDirection::Output},
             {"delta", "Delta", ValueType::Float, SocketDirection::Output}},
            {{"speed", "Speed", 1.0F, -4.0F, 4.0F}}};
        result.timeDependent = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext& context, std::span<const Value>, std::span<Value> outputs) override {
        const float speed = parameter(parameters_, "speed", 1.0F);
        outputs[0] = static_cast<float>(context.time) * speed;
        outputs[1] = static_cast<float>(context.deltaTime) * speed;
    }
};

template <typename T> void addNode(NodeRegistry& registry) {
    registry.add(T::describe(), [] { return std::make_unique<T>(); });
}

} // namespace

void registerInputNodes(NodeRegistry& registry) {
    addNode<FloatNode>(registry);
    addNode<TimeNode>(registry);
}

} // namespace reaction
