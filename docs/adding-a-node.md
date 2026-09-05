# Adding a built-in node

Nodes are registered through `NodeRegistry` during startup. Implement nodes in a focused
`src/gpu/*_nodes.cpp` file (or a dedicated `*_node.cpp` for a larger node), expose its
family registration function in `src/gpu/nodes_internal.hpp`, and call that function from
`registerBuiltInNodes`. Shared parameter, texture, and uniform helpers live in
`src/gpu/node_support.hpp`.

Each implementation derives from `NodeInstance` directly or through
`node_support::ParameterNode`, and provides a stable `NodeDescriptor`, parameter serialization,
and either generated lowering or native evaluation. Prefer lowering for per-pixel work;
`ParameterNode` supplies the empty native fallback needed by a fully lowerable node, so that node
does not implement `evaluate` or own a standalone compute program.

```cpp
class InvertNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"example.invert", 1, "Invert", "Color",
            {{"image", "Image", ValueType::Image2D, SocketDirection::Input},
             {"result", "Result", ValueType::Image2D, SocketDirection::Output}}, {}};
        result.lowerable = true;
        return result;
    }

    const NodeDescriptor& descriptor() const override {
        static const auto descriptor = describe();
        return descriptor;
    }

    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto image = context.color("image", "image", 0.0F);
        (void)context.emitTyped(
            "vec4(vec3(1.0)-(" + image.name + ").rgb,(" + image.name + ").a)",
            ShaderValueType::Vec4, "result");
        return true;
    }
};
```

Register the descriptor and factory in the node family's registration function. Socket keys and the node type are serialized API: never rename them without a type-version migration. A stateful node must set `NodeDescriptor::stateful`, own its persistent textures, and implement `reset`. It must not expose a graph-level feedback edge.

## Use the descriptor as the source of truth

Describe every editable constant as a `ParameterDescriptor`. Float and Integer parameters
automatically acquire an optional input socket when the registry sees no socket with the same
key. If you declare the socket yourself (for example because it is `AnyNumeric`), its key must
exactly match the parameter key:

```cpp
{{"rotation", "Rotation", ValueType::AnyNumeric, SocketDirection::Input, true},
 {"result", "Result", ValueType::AnyVector, SocketDirection::Output}},
{{"rotation", "Rotation", 0.0F, -6.2831853F, 6.2831853F}}
```

For native nodes, the runtime supplies an unconnected Float/Integer parameter socket with the
current slider value before calling `evaluate`; a connected value replaces it. Lowerable nodes
request the socket and parameter together through `input`, `scalar`, `vector`, or `color`.
Do not invent a second unrelated default or read a different key. Native procedural nodes may
use `procedural::numericParameterInput` to make the relationship explicit.

Native input indices count input sockets only, in descriptor order. Output sockets do not consume
an input index. Keep the descriptor and implementation beside each other and add a
descriptor/runtime test when changing socket order. A safer native procedural-node convention is:

```cpp
const auto rotation = procedural::numericParameterInput(
    inputs, 2, parameters_, "rotation", 0.0F);
```

## Enum controls and popups

Enums must provide their visible labels in the descriptor. The generic editor renders these
through the same deferred popup mechanism as Math, after the node editor is suspended; never
call `ImGui::BeginCombo`, `OpenPopup`, or `BeginPopup` directly while drawing a node.

```cpp
{"mode", "Mode", 0.0F, 0.0F, 2.0F, ParameterDescriptor::Control::Enum,
 {"Euclidean", "Manhattan", "Chebyshev"}}
```

The stored value is the zero-based label index. The minimum should be `0`, the maximum should
be `labels.size() - 1`, and evaluate should clamp before converting to its C++ enum. The generic
path needs no node-specific widget code. Truly custom popup editors must request a popup through
`node_widgets::PopupState` and render it in `node_widgets::renderPopup`.

Array-valued node data is serialized directly in the node parameter JSON rather than as a
`ParameterDescriptor`; the descriptor currently models scalar controls only. Render a focused
node-specific editor in `node_widgets` and sanitize its stored JSON in the node implementation.
Generated lowering has scalar and image graph inputs but no shader-owned parameter-texture
resource, so bounded arrays are currently specialized as GLSL constants.

## Numeric, vector, and coordinate inputs

Use `ValueType::AnyNumeric` for `Float | Scalar Field` sockets and `ValueType::AnyVector` for
`Float2 | Vector Field` sockets. Per-pixel scalars use R; per-pixel vectors use RG; colors use
RGBA. A coordinate transform or sampler takes one `AnyVector` Coordinates socket—not separate
U and V sockets. `Combine Vector` constructs a Float2 or RG vector field, and `Separate Vector`
splits one back into its numeric X and Y components.

For a spatial node whose omitted Coordinates input means canvas coordinates, use the typed
helper rather than an unbound sampler or a magic uniform value:

```cpp
const auto coordinates = procedural::coordinateInput(inputs, 0);
procedural::bindVectorInput(program, 0, "coordinatesImage", "hasCoordinates",
                            "coordinatesConstant", coordinates);
```

`coordinateInput` explicitly represents the canvas-UV default as a generated field.
`bindVectorInput` sends `1` for a real texture, `0` for a constant Float2, and `-1` for canvas
coordinates; `proceduralVector` handles those three states safely. Do not test “nonzero means
texture,” because the canvas state is negative and must not sample an unbound or reused texture.

## Before registering a node

- Confirm every Float/Integer slider key matches the socket key consumed by lowering or evaluate.
- Give every Enum a complete ordered label list and use the generic popup path.
- Use one RG `AnyVector` socket for coordinates and `coordinateInput` for canvas defaults.
- Native nodes must initialize every output on every evaluation path, including missing required inputs.
- Preserve Float/field promotion: constants return Float or Float2; any field input returns an image.
- Compile the full application and exercise the node once with inputs disconnected and connected.

## Editable simulation subgraphs

Subgraph instances use the node type `subgraph` and a stable `subgraphId`. Their node descriptor is derived from `SubgraphInterfaceItem` records rather than the static registry. Interface keys are serialized identities; labels can be edited safely. Float, integer, and boolean control metadata drives the corresponding node widget without type-specific UI code.

`SubgraphDefinition::body` is a `GraphBody`, the same node-and-link storage used by the root `Graph`. It contains normal `NodeRecord` and `LinkRecord` values and uses the same `addNode`, `removeNode`, `addLink`, and `removeLink` behavior. IDs are local to the body and remain stable when nodes are inserted, removed, or reordered. Links identify descriptor socket keys rather than array positions, and adding a link to an occupied input replaces the previous link.

Editable simulation bodies support the registered Float, Math, Bit Test / Integer Mask, Threshold, Select, Canvas Coordinates, and Laplacian node types. `resolveSubgraphBodyDescriptor` returns their canonical registry descriptors, so their names, pins, parameters, widgets, and validation rules do not diverge from the root editor. The subgraph canvas renders these records with the normal node-editor interactions: background-menu creation, pin-to-pin linking, replacement links, selection, deletion, movement, panning, and zooming.

Only nodes that cross or define the simulation boundary need specialized descriptors: subgraph inputs and outputs and Initial/Previous/Next Simulation State. They are offered only in the simulation-subgraph add menu because they depend on the subgraph interface or its private RG16F feedback state. The body itself remains acyclic; Previous Simulation State is the controlled feedback boundary rather than a graph-level feedback edge.

The simulation backend follows the body links and fuses the reachable nodes into an initialization shader and one update shader. It prunes interface declarations separately for those two endpoints, shares direct previous-state Laplacian neighborhoods with identical scale expressions as vector values, and emits output conversion stores only for the first or externally connected outputs. Runtime signatures are based on generated source and output mappings, so layout, labels, and unreachable nodes do not reset live state. To make another ordinary registered node available in simulation subgraphs, reuse its existing `NodeDescriptor`, add it to the permitted subgraph node types and add matching shader lowering. Do not create a second node record, socket vocabulary, or editor widget for the simulation form.

If a node has a root-only native escape hatch or a custom editor, account for that explicitly in
the simulation form. The subgraph canvas must render the same custom editor, while parameters
that cannot lower in a simulation must be removed from its contextual descriptor and rejected by
validation for existing project data. Convolution is the reference case: its kernel editor is
available in a simulation body, but its native multi-pass `iterations` control is not.

The simulation compiler is a planner/executor around the ordinary lowering semantics, not a replacement for them. Root shader fusion consumes acyclic image values once; a simulation additionally owns an initialization phase, an iterative RG16F ping-pong state, a controlled Previous-to-Next feedback boundary, neighborhood sampling, and exported state channels. Those responsibilities remain simulation-specific even while both paths share typed node lowering.

Definitions returned by `builtInSubgraphs()` are source templates. Adding one to a project creates an editable definition in `Graph::subgraphs()`, serialized once at project level; any number of instances may reference that shared definition. Project format 3 stores each body as ordinary `nodes` and `links`, while the loader migrates older project definitions that used the format-2 expression representation.

Node controls that open custom popups must request them through `node_widgets::PopupState` and
render them through `node_widgets::renderPopup`. Generic Enum parameters already do this.
That function is called only while the node editor is suspended, so popup layout and hit
testing remain in ImGui screen space.

Runtime-loaded shared libraries are intentionally outside the first milestone; this interface is the source-compatible SDK boundary from which a versioned plugin ABI can later be designed.

## Generated shader lowering

Set `NodeDescriptor::lowerable` and implement `NodeInstance::lowerShader` for per-pixel nodes.
The runtime always uses generated execution for supported parameters: fusion on combines safe
linear chains, while fusion off creates one generated region per node. Planning stops at branches,
joins, neighborhood edges, unsupported nodes, and GPU resource limits.

Ask only for operands used by the selected specialization. `scalar`, `vector`, and `color`
automatically sample field legs and broadcast or project constant legs. For a component-wise node
whose width follows its operands, use `input`, compute `promotedShaderType(operands)`, normalize
with `convertShaderValue`, and emit with `emitTyped`. The IR derives whether the result is constant
or a field; node code does not dispatch on that category.

Mark UV-dependent generators with `NodeDescriptor::producedField`. Mark neighborhood/source
sockets with `SocketDescriptor::requiresImage` (and `neighborhoodSocket` when the edge must split
regions). `inputAt` and `inputTexel` then reject constant legs with an attributed typed error.

Numeric controls remain uniforms so interactive edits do not recompile. Parameters that change
shader structure belong in `shaderVariantKey`. If only some settings lower correctly, implement
`supportsRegionFusion` and keep `evaluate` solely for those native escape-hatch settings; the
single-pass/multipass convolution split is the model.

Concrete boundary kinds (`Float`, `Vector`, `Field`, `Empty`) are baked into generated source.
The runtime re-lowers when a live boundary kind changes and caches generated programs by source.
Scalar/vector constant outputs use a one-pixel fold and remain `Float`/`Vec2`; field results and
values handed to native image consumers are materialized as textures.

Tests for a new lowerable node must cover constant and compatible field legs, verify constant-only
outputs remain semantic values, verify required-image errors, and compare fused pixels with solo
lowered pixels. Do not add a "generated versus native" duplicate-implementation test.
