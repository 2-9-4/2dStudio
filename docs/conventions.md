# Shared type and evaluation rules

## Core data types

* **Float** — one scalar value for the entire graph/frame.
* **Float2 or Vector** — one XY pair for the entire graph/frame.
* **Scalar Field** — one meaningful scalar per pixel.
* **Vector Field** — XY per pixel, stored in RG.
* **Color Image** — RGBA per pixel.

NOT USED YET
* **Point Set** — unordered collection of 2D points with optional scalar/vector attributes. This is a proposed new graph type required by scatter/packing nodes.
* **Particle Buffer** — persistent collection of particles containing at minimum position, velocity, age, lifetime, and ID. Proposed new graph/state type.
* **Histogram** — fixed-size array of scalar bin counts. Proposed auxiliary type for analysis nodes.

## Socket contracts, colors, and type indicators

`ValueType` is always one concrete semantic type. `SocketContract` separately declares which
concrete types a pin admits. Do not encode a union in `ValueType`, and do not mistake the fact that
a widening conversion exists for permission to connect it to a semantically narrower contract.

| Concrete `ValueType` | Meaning | Square color |
|---|---|---|
| `Float` | one frame-wide scalar | blue |
| `Vec2` | one frame-wide XY pair | teal |
| `ScalarField` | one scalar per pixel (R) | purple |
| `VectorField` | one XY pair per pixel (RG) | green |
| `ColorImage` | one RGBA color per pixel | amber |

| `SocketContract` | Accepted concrete types |
|---|---|
| `FloatOnly` | Float |
| `Vec2Only` | Vec2 |
| `ScalarFieldOnly` | Scalar Field |
| `VectorFieldOnly` | Vector Field |
| `ColorImageOnly` | Color Image |
| `Numeric` | Float, Scalar Field |
| `VectorNumeric` | Vec2, Vector Field |
| `AnyField` | Scalar Field, Vector Field, Color Image |
| `AnyImageValue` | all five concrete types |

Every pin renders one square per accepted concrete type. A resolved output renders just its actual
type. Hovering a pin lists its contract members and direction; hovering a wire names its concrete
source and destination types. Widening wires are orange and thicker, with the exact conversion in
their tooltip. Invalid drag targets are red and explain the rejected source type and target
contract.

### Materialized outputs and wires

An **output pin or wire whose concrete type is known** collapses to that single type: one square in
the resolved color, a label in the resolved color, and a tooltip naming exactly that type. The root
canvas first looks up the runtime's evaluated value for the output (Float / Float2 / Image), so
anything that has actually run states its result with certainty; when a node has not run yet it
falls back to the promotion rules below. Subgraph body nodes resolve by promotion only (their
body-local ids cannot be correlated with root runtime values).

Union outputs show the full accept set only while their concrete type is undetermined.

## Type aliases

**Numeric**

* `Float | Scalar Field`

**Vector Numeric**

* `Float2 | Vector Field`

**Image-Like**

* `Scalar Field | Vector Field | Color Image`

## Promotion

Numeric inputs broadcast naturally. A Float combined with a Scalar Field is treated as a constant at every pixel. If all Numeric inputs are Floats, the result is a Float; if any is a Scalar Field, the result is a Scalar Field.

The equivalent rule applies to Float2 and Vector Field.

Image-Like filters preserve the source type unless otherwise specified. Scalar operations applied to Vector Fields or Color Images operate component-wise unless explicitly defined otherwise.

Resolution is per output socket. `Fixed`, `NumericPromotion`, `VectorPromotion`, `WidestValue`,
and `PreserveInput` policies identify both the rule and the width-driving input keys. A separate
`fieldInputs` list lets a condition or factor make the result spatial without changing its
component width; unrelated inputs and outputs must never leak into the result type. Root compilation, root UI,
subgraph UI, validation, and simulation shader lowering all use `resolveOutputType`.

Admissibility is checked first against the destination `SocketContract`. Only then may compilation
plan one of these lossless widenings:

| Source | Destination | Shader value |
|---|---|---|
| Float | Vec2 | `(s, s)` |
| Float | Scalar Field | `s`, kept as a uniform expression until materialization |
| Float | Vector Field | `(s, s)` |
| Float | Color Image | `(s, s, s, 1)` |
| Vec2 | Vector Field | `(x, y)` |
| Vec2 | Color Image | `(x, y, 0, 1)` |
| Scalar Field | Vector Field | `(s, s)` |
| Scalar Field | Color Image | `(s, s, s, 1)` |
| Vector Field | Color Image | `(x, y, 0, 1)` |

No implicit narrowing is legal. In particular, Color Image to Scalar Field or Vector Field must
use an explicit `Color to R`, `Color to Luminance`, or `Color to RG` node. Constants widened into
field expressions remain uniforms; coercion alone does not allocate an intermediate texture.
`CompileResult` records the concrete type of every input/output socket and the exact coercion of
every edge. Materialized `ImageHandle` values retain that semantic type independently of their GPU
texture format.

These widenings are structural channel expansion, not semantic interpretation: Vector Field to
Color Image places X/Y in R/G with B=0 and A=1; it is not a color-space conversion, normal-map
encoding, magnitude, or claim that the vector is meaningful RGB.

## Coordinate convention

All spatial nodes use normalized coordinates unless otherwise stated:

* `(0,0)` = top-left.
* `(1,1)` = bottom-right.
* +X points right.
* +Y points down.
* Pixel coordinates refer to pixel centers.
* Angle `0` points right.
* Positive angles rotate clockwise.
* Angles are radians.
* `fract(x) = x - floor(x)`.
* `mod(a,b) = a - b*floor(a/b)`.

Neighborhood filters default to `Clamp` boundary handling unless an Address Mode is exposed.

## Generated-node execution contract

Built-in per-pixel nodes have one GPU implementation: `NodeInstance::lowerShader`. The runtime
executes a lowerable node as a solo generated region when chain fusion is disabled, and may fuse
it with adjacent lowerable nodes when fusion is enabled. `evaluate` is reserved for nodes that
cannot be lowered and for explicit parameter-dependent escape hatches such as multipass
convolution.

Lowering works with concrete runtime boundary values:

* `Float` and `Vector` become scalar or `vec2` uniforms.
* Scalar Field, Vector Field, and Color Image become one typed sampler and are sampled at their
  semantic width.
* `Empty` prevents generated dispatch until the source becomes available.
* Values produced inside a region remain expressions and are substituted directly.

The `ShaderValue` width (`Scalar`, `Vec2`, or `Vec4`) is independent of its constant/field
category. Use `scalar`, `vector`, or `color` when a socket has a fixed semantic width, and use
the raw `input` value plus `promotedShaderType` for component-wise operations whose width follows
their operands. `convertShaderValue` implements the central widening table. It must not perform
`.x`/`.xy` projection; semantic narrowing belongs in an explicit graph node.

Field-ness is derived by the lowering infrastructure. An emitted value is a field when any value
it uses is a field. Position-dependent generators set `NodeDescriptor::producedField`; ordinary
nodes must not mark constant expressions as images. Constant scalar/vector results are folded by
a one-pixel generated dispatch and returned as `Float`/`Vec2`. They are materialized at canvas
size only when a native image consumer requires a texture.

A socket sampled at arbitrary coordinates or neighboring texels must set
`SocketDescriptor::requiresImage`. Supplying a constant to it is a typed lowering error attributed
to that node; it is never silently broadcast as a fake texture. Concrete boundary kinds are part
of a region specialization. The runtime re-lowers when a live value changes between constant,
vector, field, and empty, while the generated-source cache reuses prior specializations.

## Float-backed integer data

Graph values are floats. Discrete lookup inputs that are not already integer controls use
`round` before their integer conversion. The **Bit Test / Integer Mask** node likewise rounds
its Mask and Bit inputs, rejects NaN and infinities, and supports mask bits `0..23`: IEEE-754
single-precision values represent every integer through `2^24` exactly. It clamps a negative
mask to zero and returns zero for an invalid bit index before shifting.

---
## Auto-layout

`layoutGraph` (`src/core/layout.cpp`) is a deterministic layered left-to-right pass:

- A node's column is `globalDepth - sinkDepth` (how many edge-hops separate it from the
  nearest downstream sink). A true source that drives a merge from the same column as a
  non-source single-consumer chain node is relocated to column 0 so that one of the two can
  sit exactly on the consumer's row.
- Forward placement: a node with exactly one consumer follows its longest dedicated input
  line; a junction averages its inputs' rows; sources spread symmetrically; then each column
  is compacted by a cursor over target-sorted nodes.
- Snap pass: every node feeding a single consumer is moved to the row *closest* to its
  consumer where it fits in its own column. Columns are processed **right-to-left** so a
  chain's rows settle before their feeders align to them (left-to-left feeds chase a row that
  later snaps away). Rows are not a fixed 200px grid — siblings that
  target the same consumer share the row band, so small nodes tuck directly against one
  another (half `kRowGap` breath) and a second source lands at its consumer's row plus one
  box, not a whole row away. Node sizes come from the caller (the editor passes measured
  sizes), so small source icons pack tightly and shrink the total layout height.
- The layout is order-seeded from the current positions, so `autoLayoutBody` (and anyone
  calling `layoutGraph`) should run it a few passes to reach the stable fixed point.
- Hard limit: a column has only so much vertical room per consumer band. Once the consumer's
  row plus the tight-pack neighbor slots are taken (typically two small sources per consumer),
  additional same-column sources land a whole row away and stay there unless relocated.
  Relocation trades that for an edge routed through other boxes, so it is only done for the
  non-source/source competition, never for source/source pairs. Do not try to force reachable
  extra density with an iterative solver — relaxation versions diverge or regress
  already-aligned junctions.

The auto-layout test contract lives in `tests/core_tests.cpp` (cases around lines 688-760):
chain collinearity, merge-at-average, disjoint components, single-consumer-through-merge snap.

To measure layout quality run `build/layout_probe text_test --iterations 4 [--height 40]`:
it reports exact / packed (within one row band) / far offenders per iteration and dumps the
node grid with `--dump`. The single-consumer pack is judged with measured node heights; pass
`--height` to simulate small sources the way the editor measures them.
