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

`SubgraphDefinition::body` is a `GraphBody`, the same node-and-link storage used by the root `Graph`. It contains normal `NodeRecord` and `LinkRecord` values and uses the same `addNode`, `removeNode`, `addLink`, and `removeLink` behavior. IDs are local to the body and remain stable when nodes are inserted, removed, or reordered. Links identify descriptor socket keys rather than array positions, and adding a link to an occupied input replaces the previous link.

Editable simulation bodies support the registered Float, Math, Threshold, Select, Canvas Coordinates, and Laplacian node types. `resolveSubgraphBodyDescriptor` returns their canonical registry descriptors, so their names, pins, parameters, widgets, and validation rules do not diverge from the root editor. The subgraph canvas renders these records with the normal node-editor interactions: background-menu creation, pin-to-pin linking, replacement links, selection, deletion, movement, panning, and zooming.

Only nodes that cross or define the simulation boundary need specialized descriptors: subgraph inputs and outputs and Initial/Previous/Next Simulation State. They are offered only in the simulation-subgraph add menu because they depend on the subgraph interface or its private RG16F feedback state. The body itself remains acyclic; Previous Simulation State is the controlled feedback boundary rather than a graph-level feedback edge.

The simulation backend follows the body links and fuses the reachable nodes into an initialization shader and one update shader. It prunes interface declarations separately for those two endpoints, shares direct previous-state Laplacian neighborhoods with identical scale expressions as vector values, and emits output conversion stores only for the first or externally connected outputs. Runtime signatures are based on generated source and output mappings, so layout, labels, and unreachable nodes do not reset live state. To make another ordinary registered node available in simulation subgraphs, reuse its existing `NodeDescriptor`, add it to the permitted subgraph node types and add matching shader lowering. Do not create a second node record, socket vocabulary, or editor widget for the simulation form.

The simulation compiler is a planner/executor around the ordinary lowering semantics, not a replacement for them. Root Math fusion consumes acyclic image values once; a simulation additionally owns an initialization phase, an iterative RG16F ping-pong state, a controlled Previous-to-Next feedback boundary, neighborhood sampling, and exported state channels. Those responsibilities remain simulation-specific even while both paths share typed Math-expression lowering.

Definitions returned by `builtInSubgraphs()` are source templates. Adding one to a project creates an editable definition in `Graph::subgraphs()`, serialized once at project level; any number of instances may reference that shared definition. Project format 3 stores each body as ordinary `nodes` and `links`, while the loader migrates older project definitions that used the format-2 expression representation.

Node controls that open popups must request them through `node_widgets::PopupState` and
render them through `node_widgets::renderPopup`. That function is called only while the
node editor is suspended, so popup layout and hit testing remain in ImGui screen space.

Runtime-loaded shared libraries are intentionally outside the first milestone; this interface is the source-compatible SDK boundary from which a versioned plugin ABI can later be designed.

## Generated shader lowering

`NodeInstance::lowerShader` is an optional GPU extension point. Nodes that do not override it
continue through `evaluate` unchanged. A lowerable node asks `ShaderLoweringContext` for only
the linked inputs and numeric parameters required by its selected operation, then emits a
typed expression. The region builder records those requests as deduplicated image samplers or
scalar uniforms and records the emitted value as an SSA-style instruction with its contributor
node ID.

The root planner currently invokes this capability for image-valued Math nodes. It specializes
single nodes and fuses safe linear chains, but stops at branches, joins, unsupported nodes, and
GPU resource limits. The IR itself supports several candidate outputs so future planners can
form DAG regions without changing node lowerers. Numeric controls should remain uniforms when
interactive editing must not trigger recompilation; parameters that change shader structure,
such as Math's operation, belong in the specialization key.

Generated sources must preserve the node's legacy `evaluate` semantics because that evaluator
is also the runtime fallback for compilation failures and temporarily unavailable image inputs.
Use the shared Math operation helpers as the model: persisted operation IDs, CPU behavior, root
GLSL, simulation-subgraph GLSL, and UI names all come from one definition.
