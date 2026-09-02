# Adding a built-in node

Nodes are registered through `NodeRegistry` during startup. Each implementation derives from `NodeInstance` and provides a stable `NodeDescriptor` plus evaluation, parameter serialization, and optional reset behavior.

```cpp
class InvertNode final : public NodeInstance {
public:
    static NodeDescriptor describe() {
        return {"example.invert", 1, "Invert", "Color",
            {{"image", "Image", ValueType::Image2D, SocketDirection::Input},
             {"result", "Result", ValueType::Image2D, SocketDirection::Output}}, {}};
    }

    const NodeDescriptor& descriptor() const override {
        static const auto descriptor = describe();
        return descriptor;
    }

    void evaluate(EvaluationContext& context,
                  std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        // Allocate or reuse an output through the GpuRuntime in context.gpu,
        // dispatch compute work, then assign an ImageHandle to outputs[0].
    }

    nlohmann::json parameters() const override { return parameters_; }
    void setParameters(const nlohmann::json& value) override { parameters_ = value; }
private:
    nlohmann::json parameters_;
};
```

Register the descriptor and factory in `registerBuiltInNodes`. Socket keys and the node type are serialized API: never rename them without a type-version migration. A stateful node must set `NodeDescriptor::stateful`, own its persistent textures, and implement `reset`. It must not expose a graph-level feedback edge.

Runtime-loaded shared libraries are intentionally outside the first milestone; this interface is the source-compatible SDK boundary from which a versioned plugin ABI can later be designed.

