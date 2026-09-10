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
            {{"image", "Image", SocketContract::AnyField, SocketDirection::Input},
             {"result", "Result", SocketContract::AnyField, SocketDirection::Output}}, {}};
        result.sockets.back().typePolicy = SocketDescriptor::TypePolicy::PreserveInput;
        result.sockets.back().typeInputs = {"image"};
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
key. If you declare the socket yourself (for example because it is `Numeric`), its key must
exactly match the parameter key:

```cpp
{{"rotation", "Rotation", SocketContract::Numeric, SocketDirection::Input, true},
 {"result", "Result", SocketContract::VectorNumeric, SocketDirection::Output}},
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

## Semantic socket typing

Use a concrete `ValueType` for fixed sockets and a `SocketContract` for unions. Common patterns:

```cpp
// Scalar-only field operation.
{"height", "Height", ValueType::ScalarField, SocketDirection::Input}

// Vector-only field operation.
{"flow", "Flow", ValueType::VectorField, SocketDirection::Input}

// Component-wise filter: input/output AnyField; output PreserveInput from {"image"}.
// Mixed-width Mix: a/b/output AnyImageValue; output WidestValue from {"a", "b"},
// with fieldInputs={"a", "b", "factor"} so a field factor makes the result spatial.
// Multi-output Separate Vector: each output resolves independently from {"vector"}.
```

`Numeric` means `Float | ScalarField`; `VectorNumeric` means `Vec2 | VectorField`;
`AnyField` means the three per-pixel semantic types; `AnyImageValue` admits all five values.
Never use `AnyImageValue` merely because a conversion exists: the contract must describe what the
operation means. Widening is compiler-owned. Color-to-scalar/vector narrowing requires an explicit
`Color to R`, `Color to Luminance`, or `Color to RG` node. For component-wise color operations,
state whether alpha is transformed or preserved and test it.

A coordinate transform or sampler takes one `VectorNumeric` Coordinates socket—not separate U
and V sockets. Per-pixel scalars use R, vectors use RG, and colors use RGBA. `Combine Vector`
constructs a Vec2 or Vector Field, and `Separate Vector` produces independently resolved numeric
X and Y outputs.

For a spatial node whose omitted Coordinates input means canvas coordinates, use the typed
helper rather than an unbound sampler or a magic uniform value:

```cpp
const auto coordinates = procedural::coordinateInput(inputs, 0);
procedural::bindVectorInput(program, 0, "coordinatesImage", "hasCoordinates",
                            "coordinatesConstant", coordinates);
```

`coordinateInput` explicitly represents the canvas-UV default as a generated field. Its socket
descriptor must also set `fieldDefault = true` so compile-time inference agrees with lowering.
`bindVectorInput` sends `1` for a real texture, `0` for a constant Float2, and `-1` for canvas
coordinates; `proceduralVector` handles those three states safely. Do not test “nonzero means
texture,” because the canvas state is negative and must not sample an unbound or reused texture.

## Before registering a node

- Confirm every Float/Integer slider key matches the socket key consumed by lowering or evaluate.
- Give every Enum a complete ordered label list and use the generic popup path.
- Use one `VectorNumeric` socket for coordinates, mark its canvas default with `fieldDefault`, and
  use `coordinateInput` while evaluating it.
- Native nodes must initialize every output on every evaluation path, including missing required inputs.
- Set every polymorphic output's `typePolicy` and `typeInputs`; never infer one node-wide type.
- Native `ImageHandle` outputs must retain the compiler-resolved semantic type.
- Test contracts, rejected narrowing, exact widening expressions, multi-output independence, and
  fused versus materialized behavior.
- Compile the full application and exercise the node once with inputs disconnected and connected.

## Editable simulation subgraphs

### Generalized simulation state

Simulation definitions may declare named `stateSlots`. A slot has a stable serialized key,
an editable label, and one field type: Scalar Field, Vector Field, or Color Image. The editor's
**Subgraph Settings** dialog creates the matching Previous State, Initial State, and Next State
nodes for a new slot; their State Slot enum selects the declaration and therefore fixes their
socket type. Each slot needs exactly one of each endpoint. Initial State receives the reset
expression, Previous State reads the value at the beginning of an iteration, and Next State
receives the value to write after it. The generated update dispatch samples every previous slot
and writes every next slot to separate ping-pong textures, so reads cannot observe another slot's
write in the same iteration.

State slot keys are identities and must not be renamed implicitly when labels change. Existing
projects with the legacy implicit RG state migrate on load to one `Chemicals` Vector Field slot;
their A/B presentation paths are preserved through an explicit Channel Split. State slots are
fields, not uniform Float/Vec2 values, because persistent simulation state has one value per
canvas pixel.

Subgraph instances use the node type `subgraph` and a stable `subgraphId`. Their node descriptor is derived from `SubgraphInterfaceItem` records rather than the static registry. Interface keys are serialized identities; labels can be edited safely. Float, integer, and boolean control metadata drives the corresponding node widget without type-specific UI code.

There is no separate public “control” interface kind. Every public value is an Input or Output.
A Float input carries optional default/range/widget metadata and renders as a local slider (or
integer/boolean/dropdown control) while unwired; wiring its socket overrides that local value. A
dropdown stores and lowers its zero-based selected value as a Float; its labels are serialized on
the interface record and are editable in **Subgraph Settings**. Older project
files containing `kind: "slider"` are migrated to these optional Float inputs on load.

`SubgraphDefinition::body` is a `GraphBody`, the same node-and-link storage used by the root `Graph`. It contains normal `NodeRecord` and `LinkRecord` values and uses the same `addNode`, `removeNode`, `addLink`, and `removeLink` behavior. IDs are local to the body and remain stable when nodes are inserted, removed, or reordered. Links identify descriptor socket keys rather than array positions, and adding a link to an occupied input replaces the previous link.

Editable simulation bodies support the registered Float, Math, Bit Test / Integer Mask, Threshold, Select, Canvas Coordinates, Laplacian, Table, and Convolution node types. `resolveSubgraphBodyDescriptor` returns their canonical registry descriptors, so their names, pins, parameters, widgets, and validation rules do not diverge from the root editor.

Do not create an ImGui child window inside a node body: node-editor coordinates and child-window
coordinates diverge, causing the child control to appear away from its node. A large custom editor
such as Convolution's kernel grid belongs in a deferred `node_widgets::PopupState` popup; the node
body keeps only its compact launcher.

Both canvases render the node box through the same `renderNodeBody` helper in `src/app/application.cpp`, so a node looks identical in the root graph and inside a subgraph: the descriptor title, pin columns, parameter rows, and per-node widget editors (Convolution kernel, Table values, Image picker, Color Ramp swatches) are one code path. Each canvas supplies the same shared editor utilities and only the root graph adds runtime inspection in the middle column — fusion status, live value/image previews, GPU timing, Reset Simulation, and the subgraph/output instance actions — because those are keyed to root-graph state that the simulation body does not own. `NodeRecord::label` is not a canvas title anywhere: it only labels generated-shader instructions in the Shader Inspector (and the Shader IR), matching root rendering. The subgraph canvas renders its records with the normal node-editor interactions: background-menu creation, pin-to-pin linking, replacement links, selection, deletion, movement, panning, and zooming.

Only nodes that cross or define the simulation boundary need specialized descriptors: subgraph inputs and outputs and Initial/Previous/Next Simulation State. They are offered only in the simulation-subgraph add menu because they depend on the subgraph interface or its private RG16F feedback state. The body itself remains acyclic; Previous Simulation State is the controlled feedback boundary rather than a graph-level feedback edge.

The simulation backend follows the body links and fuses the reachable nodes into an initialization shader and one update shader. It prunes interface declarations separately for those two endpoints, shares direct previous-state Laplacian neighborhoods with identical scale expressions as vector values, and emits output conversion stores only for the first or externally connected outputs. Runtime signatures are based on generated source and output mappings, so layout, labels, and unreachable nodes do not reset live state. To make another ordinary registered node available in simulation subgraphs, reuse its existing `NodeDescriptor`, add it to the permitted subgraph node types and add matching shader lowering. Do not create a second node record, socket vocabulary, or editor widget for the simulation form.

If a node has a root-only native escape hatch or a custom editor, account for that explicitly in
the simulation form. The subgraph canvas must render the same custom editor, while parameters
that cannot lower in a simulation must be removed from its contextual descriptor and rejected by
validation for existing project data. Convolution is the reference case: its kernel editor is
available in a simulation body, but its native multi-pass `iterations` control is not.

Root-only runtime inspection that lives inside the node box (fusion state, live previews, GPU
timing, Reset Simulation) has no subgraph equivalent and is not reproduced there; node IDs in a
subgraph body are local to the body and would collide with root-graph runtime maps. The legacy
`needsAttention` migration cue for directly wired Previous Simulation State `a`/`b` links is a
subgraph breadcrumb notice, not a node-body line, so migrated nodes keep their guidance without
differing from the root node box.

The simulation compiler is a planner/executor around the ordinary lowering semantics, not a replacement for them. Root shader fusion consumes acyclic image values once; a simulation additionally owns an initialization phase, an iterative RG16F ping-pong state, a controlled Previous-to-Next feedback boundary, neighborhood sampling, and exported state channels. Those responsibilities remain simulation-specific even while both paths share typed node lowering.

Definitions returned by `builtInSubgraphs()` are source templates. Adding one to a project creates an editable definition in `Graph::subgraphs()`, serialized once at project level; any number of instances may reference that shared definition. Project format 4 stores each body as ordinary `nodes` and `links` plus explicit interface contracts; the loader migrates older project definitions and inserts explicit channel-extraction nodes for legacy generic-image narrowing.

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
Use `inputSample` for a field sampled at a caller-controlled normalized coordinate: it is the
typed arbitrary-coordinate counterpart to `inputTexel`, preserving selected nearest/linear
filtering while the node applies clamp/repeat/mirror/border semantics. Mark that source socket
`requiresImage` and as the node's `neighborhoodSocket`, so a sampled producer is materialized
before the sampling region rather than incorrectly read from an in-region SSA value.

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
