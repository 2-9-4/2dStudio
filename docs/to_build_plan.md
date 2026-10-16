# Dense node implementation plan

This plan reconciles the numbered proposals in [`to_build_dense.md`](to_build_dense.md)
with the node catalog in [`nodes.md`](nodes.md) and the currently registered GPU nodes.
It is organized by shared implementation work so the groups can later be assigned to
subagents with limited overlap.

## Planning result

The dense backlog contains 100 numbered proposals. The repository already has useful
coverage for many of them, but `nodes.md` is not a complete implementation inventory:
several registered nodes exist only in `src/gpu/`. The work below therefore distinguishes
three states:

- **Covered:** the requested node family is already usable and should not be rebuilt.
- **Reconcile/extend:** an implementation exists, but its current sockets or modes do not
  yet satisfy the dense specification.
- **Build:** no adequate implementation is currently registered.

### Existing coverage to preserve

These proposals have a usable implementation or a close built-in/subgraph equivalent:

| Dense IDs | Existing coverage | Follow-up |
|---|---|---|
| 2 | `transform_2d` | Add the catalog entry and verify inverse-transform semantics. |
| 3 | `polar_coordinates` | Add the catalog entry and test both directions. |
| 13 | `domain_warp` | Verify vector-field displacement and simulation lowering. |
| 18 | `worley_noise` | Verify all advertised outputs and distance metrics. |
| 28 | `gradient` | Add the catalog entry and reconcile mode names. |
| 29 | `wave` | Add the catalog entry and reconcile ring/linear behavior. |
| 54 | `combine_vector` / `separate_vector` | Document in `nodes.md`; no duplicate node. |
| 55 | `vector_math` | Extend only for missing dense operations. |
| 71 | `distance_transform` | Verify approximation and unit semantics. |
| 72 | `sdf_generator` | Keep separate from analytic SDF nodes; verify sign convention. |
| 89 | Lenia built-in subgraph | Reconcile public controls and kernel semantics. |
| 98 | Flood Fill built-in subgraph | Add optional distance output if required by the dense contract. |

### Existing implementations that need reconciliation

These should be assigned as compatibility work, not treated as greenfield nodes:

| Dense ID | Current implementation | Gap to close |
|---|---|---|
| 1 | `texture_sample` | Current code has nearest/linear and a vector coordinate input. The dense spec additionally calls for U/V inputs, cubic filtering, and a border value matching the source semantic type. Decide whether to extend the node or explicitly narrow the spec. |
| 10–11 | `repeat_fold` | Current node is scalar numeric repeat/fold. The dense proposals are vector coordinate operators with independent per-axis modes. Add a coordinate form or split the scalar helper from a new coordinate node. |
| 15 | `hash` | Current deterministic hash is a good primitive, but the dense white-noise contract adds frequency/offset-driven lattice sampling and optional field outputs. Add a wrapper mode or document the existing node as the canonical implementation. |
| 81 | `dither` | Current node covers threshold, Bayer through 8×8, halftone, and deterministic noise. Blue-noise and Floyd–Steinberg remain separate implementation decisions. |
| 88 | Generic Cellular Automata subgraph | Current public form is binary Life-like CA. The dense contract also describes Von Neumann boundaries and multi-state totalistic rules. Add modes without breaking existing B/S projects. |
| 96 | Skeletonization subgraph | Zhang–Suen exists. Medial-axis and radius output do not. |

Do not schedule a second implementation for the covered rows until an integration pass has
confirmed their registration, docs, persistence, and tests.

## Shared foundation before parallel node work

These are cross-cutting prerequisites. One owner should land them before the independent
node batches begin.

1. **Shared lowering and sampling helpers**
   - Reuse `resolveOutputType`, `promotedShaderType`, `convertShaderValue`, and the
     `inputSample`/`inputTexel` contract rather than adding per-node promotion logic.
   - Standardize normalized-coordinate addressing, texel-center clamping, border values,
     neighborhood radius, and resolution-aware pixel size.
   - Add common derivative, reduction, color-conversion, hash, noise, and SDF helpers in
     focused support files rather than copying GLSL into every node.

2. **Data-type decisions**
   - Keep `Float`, `Vec2`, `ScalarField`, `VectorField`, and `ColorImage` as the existing
     concrete graph types.
   - Design `PointSet`, `ParticleBuffer`, and `Histogram` before implementing IDs 50–52,
     91–92, or 95. These are explicitly marked “NOT USED YET” in the dense document.
   - Define serialization, editor pin contracts, GPU ownership, reset behavior, and
     materialization boundaries for each new type before adding nodes that consume it.

3. **Resolution-changing and global execution hooks**
   - Decide how a node such as Resize changes downstream canvas dimensions and how semantic
     `ImageHandle` metadata survives it.
   - Define a reduction/readback path for histograms, percentiles, component counts, and
     global labels. These are not ordinary fused per-pixel nodes.
   - Define a safe custom-shader error/fallback path before implementing ID 100.

4. **Registration and test scaffolding**
   - Keep each family in focused translation units and add one registration function per
     family. The integration owner alone should edit central registration/CMake lists when
     possible.
   - Add descriptor/default-lowering coverage for every registered node and every new
     simulation template.
   - Every family must include unit tests for formulas and edge cases, GPU/source tests for
     lowering, persistence tests for editable parameters, and subgraph tests where allowed.

## Functional implementation batches

### Batch A — Coordinate spaces and domain transforms

**Dense IDs:** 4–12, 14, plus the reconciliation work for 1, 10, and 11.

**Nodes:**

- Log-Polar Coordinates
- Kaleidoscope / Angular Fold
- Twirl / Swirl
- Radial / Lens Distortion
- Perspective / Projective Transform
- Quad / Bilinear Warp
- Mirror / Fold Coordinates
- Repeat / Wrap Coordinates
- Coordinate Quantize
- Complex Plane Transform

**Shared overlap:** vector-coordinate inputs, polar decomposition, inverse sampling
semantics, matrix/homography solving, singularity safeguards, and repeat/address modes.
Most of these should be lowerable stateless coordinate nodes. They become useful through
composition with Texture Sample rather than by owning image data themselves.

**Suggested subagents:**

- A1: Log-polar, twirl, radial/lens, and complex-plane math.
- A2: perspective and bilinear quadrilateral solvers, including degenerate fallbacks.
- A3: kaleidoscope/angular fold, mirror/fold, vector repeat/wrap, and coordinate quantization; reconcile
  `repeat_fold` without changing existing scalar projects.
- A4: Texture Sample contract completion and shared coordinate/addressing tests.

**Dependencies:** shared vector promotion and arbitrary-coordinate sampling. Perspective
and bilinear work should not block the simpler analytic transforms.

### Batch B — Noise and stochastic fields

**Dense IDs:** 16–17, 19–27; reconcile 15 and verify 18.

**Nodes:**

- Value Noise
- Simplex / OpenSimplex Noise
- Fractal Noise Composer
- Ridged Multifractal
- Turbulence Noise
- Billow Noise
- Curl Noise
- Flow Noise
- Gabor Noise
- Blue Noise
- Noise Derivatives

**Shared overlap:** frozen deterministic hashing, lattice coordinate conventions, octave
composition, output normalization, time/seed compatibility, and numerical derivatives.
Value/simplex noise should expose reusable helpers so the four fractal variants do not each
ship a private noise implementation.

**Suggested subagents:**

- B1: canonical hash/value/simplex core and compatibility tests.
- B2: fractal composer plus ridged, turbulence, and billow modes.
- B3: curl, flow, and numerical/analytic noise derivatives.
- B4: Gabor noise and the bundled canonical blue-noise asset.

**Dependencies:** B2–B4 depend on B1's frozen helper API. Blue Noise must also coordinate
with Dither's optional blue-noise mode.

### Batch C — Analytic patterns and SDF algebra

**Dense IDs:** 30–42; verify 28–29.

**Nodes:**

- Checker / Grid
- Stripe Generator
- Ring Generator
- Radial Spokes
- Spiral Generator
- Moiré / Interference Generator
- SDF Primitive
- SDF Boolean
- SDF Transform
- SDF Repeat
- SDF Stroke / Contour
- Superformula / Supershape
- Harmonic Pattern

**Shared overlap:** coordinate-to-cell math, periodic waveforms, signed-distance sign
conventions, analytic shape helpers, smooth Boolean behavior, and explicit field typing.
Keep analytic SDFs distinct from the approximate SDF Generator/distance-transform nodes.

**Suggested subagents:**

- C1: checker/stripe/ring/spokes/spiral and harmonic patterns.
- C2: SDF primitive library and superformula.
- C3: SDF Boolean, transform, repeat, stroke, and shared SDF tests.
- C4: Moiré composition and catalog/documentation reconciliation for Gradient/Wave.

**Dependencies:** shared coordinate helpers and stable SDF conventions. C1 can proceed
independently of C2, while C3 depends on C2's signed-distance representation.

### Batch D — Tiling and spatial arrangement

**Dense IDs:** 43–49.

**Nodes:**

- Tile Generator
- Tile Sampler
- Brick / Bond Generator
- Lattice Generator
- Truchet Tiles
- Wang Tile Generator
- Quasiperiodic Tiling

**Shared overlap:** deterministic integer cell IDs, local UV, stable per-cell randomization,
atlas sampling, and cell/boundary geometry. Tile Generator and Tile Sampler must use the same
cell mapping so a generated tile can be replaced by an image tile without changing layout.

**Suggested subagents:**

- D1: common lattice/cell/random helpers plus Tile Generator and Brick/Bond.
- D2: Tile Sampler and Truchet Tiles.
- D3: Wang Tile edge-label assignment and atlas validation.
- D4: Quasiperiodic/Penrose implementation, likely with a CPU or buffer-backed path.

**Dependencies:** D1's cell helper; Texture Sample for D2/D3. D4 is intentionally isolated
because it may not fit the ordinary fragment/compute lowering model.

### Batch E — Point sets, scatter, and packing

**Dense IDs:** 50–52.

**Nodes:**

- Density Scatter
- Poisson-Disc Scatter
- Packing / Relaxation

**Shared overlap:** the new `PointSet` type, deterministic candidate generation, spatial
acceleration, stable IDs, masks, and rasterization back into fields. These are globally
coupled operations and should not be forced into the ordinary one-pixel shader contract.

**Suggested subagents:**

- E1: PointSet graph/GPU representation, serialization, pin rendering, and rasterization.
- E2: density scatter and deterministic ID preservation.
- E3: Poisson-disc generation with spatial grid acceleration.
- E4: packing/relaxation and boundary projection.

**Dependencies:** E1 is a hard prerequisite. E3/E4 should share the spatial index but may
otherwise proceed independently after E1.

### Batch F — Vector fields and differential operators

**Dense IDs:** 53, 56–62; extend 54–55 only where needed.

**Nodes:**

- Image Gradient
- Divergence
- Curl / Vorticity
- Vector Field Transform
- Advect Image
- Flow Map Distort
- Streamline Integration
- Vector Field Visualizer

**Shared overlap:** central/Sobel/Scharr derivative kernels, normalized coordinate spacing,
vector-field semantic width, arbitrary-coordinate sampling, and line integration.

**Suggested subagents:**

- F1: derivative kernel library, Image Gradient, Divergence, and Curl.
- F2: Vector Field Transform and Flow Map Distort.
- F3: Advect Image, including multi-step semi-Lagrangian sampling.
- F4: Streamline Integration and Vector Field Visualizer.

**Dependencies:** F1 and Texture Sample. F4 may need PointSet seeds or a density-seed
fallback; keep that interface narrow so it can land before the full PointSet system.

### Batch G — Filters, morphology, and structure extraction

**Dense IDs:** 63–70, 73–74; verify 71–72.

**Nodes:**

- Gaussian Blur HQ
- Directional Blur
- Radial / Zoom Blur
- Vector Motion Blur
- Slope Blur
- Median Filter
- Bilateral Filter
- Kuwahara Filter
- Bevel from Distance
- Morphology Suite

**Shared overlap:** typed neighborhood sampling, radius/address behavior, separable or
multi-pass execution, semantic type preservation, and reusable erosion/dilation primitives.
Morphology should reuse the existing Convolution erosion/dilation semantics rather than
forking kernels. Gaussian Blur HQ should have a native separable path; arbitrary iteration
controls must not leak into simulation descriptors where they cannot lower.

**Suggested subagents:**

- G1: Gaussian, directional, radial/zoom, and vector motion blur.
- G2: slope, median, bilateral, and Kuwahara filters.
- G3: Bevel from Distance and derivative-to-normal output.
- G4: Morphology Suite and composition over Convolution.

**Dependencies:** shared neighborhood/derivative helpers and SDF/distance semantics.

### Batch H — Color and tonal mapping

**Dense IDs:** 75–82; extend 81.

**Nodes:**

- Color Space Convert
- HSV/HSL Adjust
- Levels
- Curves
- Posterize / Quantize
- Multi-stop Color Ramp
- Dither extensions
- Channel Split / Combine / Swizzle

**Shared overlap:** Color Image-only contracts, alpha preservation, channel semantics,
editable curve/ramp serialization, LUT generation, and color-space conversion helpers.
Explicit narrowing remains mandatory: these nodes must not silently turn Color Images into
Scalar/Vector fields.

**Suggested subagents:**

- H1: color-space conversion core and HSV/HSL Adjust.
- H2: Levels, Posterize, and Channel Split/Combine/Swizzle.
- H3: editable Curves and Multi-stop Color Ramp, including popup/editor persistence.
- H4: Dither extensions and blue-noise integration.

**Dependencies:** H1's conversion helpers for optional HSV/OKLab ramp interpolation.
H3/H4 must share parameter serialization conventions with the existing Table and Color Ramp
editors.

### Batch I — Time, feedback, and dynamic systems

**Dense IDs:** 83–87; verify/extend 88–89.

**Nodes:**

- Feedback
- Frame Delay
- Temporal Accumulate
- Trails / Decay
- Optical Flow
- Generic Cellular Automata extensions
- Continuous Cellular Automata / Lenia reconciliation

**Recommended execution model:**

- Implement Feedback, Temporal Accumulate, and Trails/Decay as built-in simulation-subgraph
  templates where possible. They naturally use Previous/Next State, reset, and outer-step
  timing already supported by the project.
- Implement Frame Delay as a root runtime node with an explicit ring buffer; it is a
  multi-frame history feature, not an ordinary body-local state slot.
- Implement Optical Flow as a native multi-pass node after the reduction/pyramid strategy
  is decided. It should report confidence and define the previous/current frame convention.
- Extend Generic Cellular Automata without changing existing binary Life-like project
  migration. Add boundary/neighborhood and optional multi-state lookup as separate modes.

**Suggested subagents:**

- I1: simulation templates for Feedback, Accumulate, and Trails, plus reset/timing tests.
- I2: Frame Delay history storage and persistence behavior.
- I3: Optical Flow algorithm and confidence output.
- I4: Generic CA extension and Lenia contract reconciliation.

**Dependencies:** simulation uniform binding must work for initialization and every update;
do not retain references to `frames_.back()` across recursive lowering.

### Batch J — Larger simulations and new persistent data

**Dense IDs:** 90–92.

**Nodes:**

- 2D Fluid Simulation
- Particle Field / GPU Particles
- Physarum / Slime Mold Simulation

**Shared overlap:** persistent state ownership, reset/seed determinism, multi-pass dispatch,
boundary conditions, and synchronization between image textures and buffer data.

**Suggested subagents:**

- J1: fluid state layout, advection/projection pipeline, pressure solver, and outputs.
- J2: ParticleBuffer storage, emitter/update kernels, and rasterization.
- J3: Physarum agent/trail coupling, using J2's buffer and image-state interfaces.

**Dependencies:** J1 depends on the simulation execution contract and F3's advection
semantics. J2 is blocked on the ParticleBuffer design. J3 is blocked on J2 and trail
diffusion/decay helpers.

### Batch K — Analysis, global operations, and graph escape hatches

**Dense IDs:** 93–95, 97, 99–100; extend 96 and 98.

**Nodes:**

- Edge / Derivative Suite
- Local Statistics
- Histogram / Normalize
- Connected Components
- Resize / Resample / Crop
- Custom GLSL / Expression
- Skeletonize medial-axis mode
- Flood Fill distance output

**Shared overlap:** neighborhood reductions, GPU global reductions, iterative propagation,
resolution changes, editable data, and robust diagnostics.

**Suggested subagents:**

- K1: Edge/Derivative Suite and Local Statistics using the shared derivative/neighborhood
  library.
- K2: Histogram/Normalize and the `Histogram` type, including percentile/equalization
  reductions.
- K3: Connected Components and Skeletonize medial-axis/radius outputs.
- K4: Resize/Resample/Crop and resolution-aware downstream execution.
- K5: Custom GLSL/Expression compiler, typed dynamic sockets, source locations, and safe
  fallback behavior.

**Dependencies:** K2/K3 need a global reduction/propagation strategy. K4 should land before
nodes that assume every image has the project canvas dimensions. K5 should be last because it
depends on the stabilized socket/type/lowering contracts.

## Suggested delivery waves

The batches above are functional ownership groups. A practical parallel schedule is:

| Wave | Work that can run in parallel | Main gate |
|---|---|---|
| 0 | Shared helpers, data-type design, registration/test scaffolding | Stable contracts and no duplicate promotion logic. |
| 1 | A, B1, C1/C2, F1, H1/H2 | Coordinate, noise, derivative, and color primitives. |
| 2 | B2–B4, C3–C4, D1–D3, F2–F3, G1–G4, H3–H4 | Shared helper APIs from Wave 1 are frozen. |
| 3 | D4, E, F4, I1–I4, K1/K4 | Global/iterative work starts after typed fields and sampling are stable. |
| 4 | J, K2/K3, K5 | New buffers/reductions and custom shader escape hatch. |
| 5 | Integration, migrations, docs, performance, and end-to-end examples | All registered nodes have lowering/default/error coverage. |

Agents should avoid editing the same central registration, CMake, descriptor, or application
rendering files concurrently. Prefer family-local files and hand the integration owner a
small registration patch plus tests.

## Definition of done for every node or family

1. A concrete `NodeDescriptor` defines contracts, optionality, defaults, type policy, field
   inputs, and any `requiresImage`/`neighborhoodSocket` boundary.
2. The node is registered, searchable, serializable, and documented in `nodes.md`.
3. Root and subgraph rendering use the shared `renderNodeBody` path. Any root-only native
   control is omitted from the contextual simulation descriptor and invalid persisted values
   are rejected or migrated.
4. Lowering uses shared semantic type resolution and lossless widening. No node invents
   Color-to-Scalar/Vector narrowing or treats a constant as a fake sampled image.
5. GPU work has formula/edge-case tests, generated-source or shader tests, and a smoke test
   for materialized semantic types. Stateful/global work has reset, resize, determinism, and
   failure-path tests.
6. Persistence migration is versioned when sockets, editable arrays, state slots, or types
   change. Existing projects retain their semantic values.
7. The implementation records performance limits and approximation choices in `nodes.md`.

## Footguns and ergonomics to fix before delegating broadly

- `nodes.md` currently omits several registered nodes, including 2D Transform, Domain Warp,
  Polar Coordinates, Repeat/Fold, Worley Noise, Gradient, and Wave. Treat source registration
  plus tests as the implementation inventory until the catalog is synchronized.
- `Texture Sample` currently exposes only nearest/linear and a vector coordinate socket,
  while the dense spec asks for U/V, cubic, and source-typed border values. This mismatch can
  cause agents to build a duplicate UV node.
- `repeat_fold` is a scalar numeric node, not a complete per-axis coordinate remapper. Its
  name is likely to mislead agents implementing IDs 10–11.
- New global types (`PointSet`, `ParticleBuffer`, `Histogram`) cannot be safely added as
  cosmetic socket labels; they need persistence, runtime ownership, and UI semantics first.
- Resolution-changing nodes invalidate assumptions in existing generated shaders that every
  image uses the global canvas size.
- Root runtime inspection, live previews, GPU timing, reset controls, and subgraph/output
  instance actions must remain root-only. They must not be reintroduced as divergent
  subgraph node bodies.
- Simulation lowering is recursive and its frame vector may reallocate. Copy node pointers
  and needed values before recursive `lower`/`input` calls.
- Arbitrary-coordinate field samplers must apply address behavior before `inputSample` and
  mark the source as both `requiresImage` and `neighborhoodSocket`.
