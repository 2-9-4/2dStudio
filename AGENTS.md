@docs/adding-a-node.md
@docs/conventions.md
@docs/nodes.md

The built-in-node guide describes GPU lowering, parameter serialization, and editor controls.

## Architecture philosophy — start generic, not feature-specific

The aim is a *small composable visual programming language* for procedural art, not
a catalog of hardcoded artworks. Before implementing an example, identify its
**minimal reusable building blocks**, define their typed inputs/outputs and
orthogonal controls, and assemble the example entirely from those blocks.
Treat project examples as graph compositions, NOT as reasons for special-purpose
C++ pipelines. It should be possible to change the aesthetic by reconnecting
existing nodes rather than asking for another kernel rewrite.

- **Separate physics, data sources, rendering, and compositing.** A particle
  simulation outputs positions; a separate Splat owns ink and trail history;
  Mix composites images. Noise is an external input, never embedded in the
  particle simulation. Geometry and colors don't belong in motion semantics.
- **Build one parameterized definition, instantiate many times.** When two
  subgraphs differ only by one trigonometric function, numeric range, mask
  polarity, seed, or color, those are **independent exposed parameters** (enum,
  float, field, or vector socket). Don't fork subgraphs, add color-coded IDs,
  or generate near-duplicate C++ definitions. Avoid one-argument built-in
  factories that merely produce redundant copies of the same topology.
- **Orthogonal selection is a core principle.** Choosing a velocity law must
  not dictate where deposition is allowed. Expose separate controls for
  algorithm, region, polarity, seed, gain, etc.; model their implementation
  as ordinary math/Select nodes visible and editable in the graph.
- **The absence of force coupling does not imply the absence of shared visual
  rules.** Two particle species can have independent lifetimes and positions
  while being spatially partitioned by a shared read-only field and
  complementary output masks. Audit *both* physical behavior and visual
  constraints before splitting an old simulation. Mask output positions,
  never feedback, unless that is the user's explicit requirement.
- **Prefer general math and field operators to custom artwork nodes.** Add
  nonlinear functions to the canonical Math operation vocabulary; make noise
  a scalar field that any flow, color ramp, or domain warp can consume. Reuse
  existing Perlin, Worley, Domain Warp, Wave, Gradient, Combine/Separate,
  Select, Splat, and Mix nodes before proposing a new family.
- **Check the catalog before adding nodes.** Assess whether the request is
  expressible with existing operations or a simple selector. When a new node
  is justified, it should be useful outside the originating example and
  obey the existing typing, serialization, CPU/GPU, and UI conventions.
- **Plan architecture before wiring the example.** Write down the graph at
  the root and subgraph levels, state ownership, socket semantics, and
  compositional controls first. Avoid implementing a monolith that then
  requires multiple rounds of de-monolithization into a generic graph.
- **Test both abstraction and aesthetic invariants.** Tests should enforce
  only one registered Species Flow definition, independent per-instance
  RNG/state, correct enum behavior, correct shared-field territory masks,
  positive/nontrivial output, Mix(Add) semantics, and saved-project migration.
  Add checks of actual rendered/sampled field values, not only node IDs.
- **Preserve persisted enum IDs.** Append rather than reorder Math operation
  IDs and motion modes; when replacing identifiers, migrate old projects.
  Keep shader lowering and CPU numeric implementations consistent and add a
  GPU test for genuinely new operations.

Potential footguns and ergonomics to improve:
- SubgraphInterfaceItem's final positional constructor argument is
  **std::string role**, not enumOptions. Assign enumOptions explicitly or
  replace that positional constructor with a safe named builder.
- The simulation engine currently requires a Next State node as its output
  endpoint, so a presentation-only masked output uses a third state slot;
  don't confuse that bookkeeping requirement with physics coupling.
- Headless OpenGL regression tests must have sequential hidden contexts and
  scoped GPU runtime lifetimes to avoid teardown crashes. Keep new tests
  bounded in resolution and frame count.

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

SubgraphInterfaceItem C++ FOOTGUN: constructor argument #10 is `role` (std::string),
not `enumOptions`. NEVER pass a brace-list of enum strings there: it can
mistakenly call the string iterator-pair overload and throw
`basic_string::_M_create` when constructing the entire built-in catalog.
Instead construct the Enum interface item with Control::Enum and then set
`item.enumOptions = {...}` explicitly; ideally redesign this API to use
named builder arguments.

Particle rendering: Sands must have exactly ONE registered Species Flow built-in
(`builtin.particles.species_flow`) instantiated twice on the root graph,
not separate C++ implementations or two cloned subgraph definitions. All
motion laws are ordinary editable Math/Vector/Select nodes, chosen by the
stable `motionMode` enum (0 Tangent, 1 Sine/Cosine, 2 Double Angle,
3 Harmonic Rosette, 4 Soft Tangent). Append future values; do not renumber
existing saved modes. Deposition is independently
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

Noise toolkit: Fractal Noise is ONE generic Scalar Field node with gradient fBm,
billow, ridged, sharp turbulence, and value fBm modes; it accepts optional
Vector Numeric coordinates and Numeric frequency so Domain Warp and any other
coordinate producer can modulate it. Curl Field consumes ANY Scalar Field and
produces a Vector Field, independent of where the source came from. Treat
these as reusable building blocks, not Sands-exclusive nodes. Every new
noise family should have GPU tests across its modes, source/seed variation,
and composition into typed field consumers.
