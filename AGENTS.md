@docs/adding-a-node.md
@docs/conventions.md
@docs/nodes.md

The built-in-node guide describes GPU lowering, parameter serialization, and editor controls.

When a root node is also available in a simulation subgraph, keep its custom editor available in
the subgraph canvas. Exclude any native-only controls from the contextual subgraph descriptor and
validate persisted values that would prevent simulation shader lowering.

Node boxes must render identically in the root graph and inside a subgraph body. All canvases
share one `renderNodeBody` path in `src/app/application.cpp`; do not add per-canvas title,
pin, parameter, or widget divergence there. Root-only runtime inspection (fusion state, live
values, GPU timing, Reset Simulation, subgraph/output instance actions) stays in the root
middle column because subgraph body IDs are body-local and would collide with root runtime maps.
`NodeRecord::label` is shader-source naming, not a canvas title.

Auto-layout lives in `src/core/layout.cpp` (forward pass + nearest-row single-consumer snap
pass that packs same-consumer siblings into one row band, plus a source relocation rule for
competing merge inputs); see the Auto-layout section of `docs/conventions.md` for the contract
and its hard limits. A `layoutGraph` change is not done until
`./reaction_core_tests "auto-layout*"` passes and `build/layout_probe text_test --iterations 4`
shows no regression in exact/packed vs far offenders. Layout is order-seeded from current
positions, so evaluate after several passes.

Semantic graph typing uses concrete `ValueType` values (`Float`, `Vec2`, `ScalarField`,
`VectorField`, `ColorImage`) and separate `SocketContract` admissibility. Resolve polymorphic
types per socket through the shared `resolveOutputType`; root compilation, UI previews,
subgraph validation, and simulation lowering must not grow independent promotion logic. Plan
only the lossless coercions in `types.hpp`; Color Image narrowing is always an explicit node.
Preserve each materialized `ImageHandle::semanticType`, keep widened constants as uniforms until
a real materialization boundary, and show coercing versus invalid links consistently on every
canvas.

`SocketDescriptor` historically accumulated positional boolean constructor arguments that could
silently reinterpret descriptors. Its constructor now keeps only positional `optional`; construct
the socket and then assign named fields such as `requiresImage`, `fieldDefault`, `typePolicy`, and
`typeInputs`/`fieldInputs`. Do not add more positional flags. Project formats before 4 used a generic image type; migration inserts explicit
Color-to-R/RG nodes where those projects previously relied on implicit channel projection.

Simulation lowering is recursive and its frame stack is a `std::vector`. Never retain a reference
to `frames_.back()` across a recursive `lower`/`input` call: pushing a child frame may reallocate the
vector and invalidate that reference. Copy the current `NodeRecord*` and any needed values first.

Simulation iteration and step-information nodes are intrinsic simulation-body nodes. Their values
are dispatch uniforms declared only when reachable; bind them for initialization as well as every
update dispatch, and advance simulation time/step once per outer step rather than per iteration.

update [AGENTS.md](AGENTS.md) and associated files with whatever required more reading to accomplish your task. REPORT ANY FOOTGUNS/EERGONOMIC ISSUES THAT CAUSED BUGS TO THE USER FOR LATER FIXING. Update @docs/nodes.md when adding any nodes


Arbitrary-coordinate field samplers use `ShaderLoweringContext::inputSample`; apply the node's
address behavior before calling it and set the sampled source as both `requiresImage` and the
`neighborhoodSocket`. This keeps the source materialized rather than attempting to sample an
in-region SSA value. Simulation state samplers take raw normalized coordinates; each consumer,
not the state accessor, owns wrapping/clamping semantics.

Particle rendering: Sands must have exactly ONE registered Species Flow built-in
(`builtin.particles.species_flow`) instantiated twice on the root graph,
not separate C++ implementations or two cloned subgraph definitions. Both
trigonometric motion branches are regular editable nodes, chosen by the
`motionMode` enum (0 Tangent, 1 Sine/Cosine). Deposition is independently
chosen by `depositMode` enum (0 Inside, 1 Outside), checked against common
`regionStart`/`regionEnd` radians at final particle position. Both sample
the externally supplied scalar `field`; no baked-in Perlin. The example
wires two shared Float controls to both instances for complementary borders.
One instance has private position RG32F, lifetime, and output-only visible
RG32F state slots; ONLY the first two feed back into physics. Positions
outside the chosen territory are exported as (-1,-1), which splat ignores.
Two ordinary root splats provide ink and feed existing Mix(Add, factor 1).
Keep swatches compact. Tests must verify the actual enum parameter
interface and both paths, shared definition identity, independent motion,
complementary visibility, persistence and Mix compositing.
