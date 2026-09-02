# Adding a built-in node

Nodes are registered through `NodeRegistry` during startup. Implement nodes in a focused
`src/gpu/*_nodes.cpp` file (or a dedicated `*_node.cpp` for a larger node), expose its
family registration function in `src/gpu/nodes_internal.hpp`, and call that function from
`registerBuiltInNodes`. Shared parameter, texture, and uniform helpers live in
`src/gpu/node_support.hpp`.

Each implementation derives from `NodeInstance` and provides a stable `NodeDescriptor`
plus evaluation, parameter serialization, and optional reset behavior.

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

Register the descriptor and factory in the node family's registration function. Socket keys and the node type are serialized API: never rename them without a type-version migration. A stateful node must set `NodeDescriptor::stateful`, own its persistent textures, and implement `reset`. It must not expose a graph-level feedback edge.

## Editable simulation subgraphs

Subgraph instances use the node type `subgraph` and a stable `subgraphId`. Their node descriptor is derived from `SubgraphInterfaceItem` records rather than the static registry. Interface keys are serialized identities; labels can be edited safely. Float, integer, and boolean control metadata drives the corresponding node widget without type-specific UI code.

The bundled discrete reaction-diffusion definition demonstrates simulation kernels. Its serialized `SubgraphKernelNode` DAG has explicit initial-state and next-state endpoints. `KernelCompiler` fuses the reachable primitive operations into an initialization shader and a single update shader, while the node owns its private RG16F ping-pong state. Do not introduce a graph-level feedback edge for simulation state.

Built-in definitions live in `builtInSubgraphs()` and are immutable. Editable copies belong to `Graph::subgraphs()` and are serialized once at project level; any number of instances may reference the same definition.

Node controls that open popups must request them through `node_widgets::PopupState` and
render them through `node_widgets::renderPopup`. That function is called only while the
node editor is suspended, so popup layout and hit testing remain in ImGui screen space.

Runtime-loaded shared libraries are intentionally outside the first milestone; this interface is the source-compatible SDK boundary from which a versioned plugin ABI can later be designed.
