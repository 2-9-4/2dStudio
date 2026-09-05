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

## Socket colors and type indicators

Every input and output pin renders one small filled square per concrete type it accepts, using the
per-type color below. A union socket (Numeric / Vector Numeric) renders one square per member so
the acceptable inputs are visible at a glance; a single-type socket renders exactly one square.

| ValueType | Accepts | Square color | Tooltip |
|---|---|---|---|
| `Float` | Float | blue | `Float — Input` / `Output` |
| `Vec2` | Float2 / Vector | teal | `Float2 / Vector — Input` / `Output` |
| `Image2D` | per-pixel field | amber | `Image (per-pixel) — Input` / `Output` |
| `AnyNumeric` | Float, per-pixel field | blue + amber | `Float \| Image (per-pixel) — Input` |
| `AnyVector` | Float, Float2 / Vector, per-pixel field | blue + teal + amber | `Float \| Float2 / Vector \| Image (per-pixel) — Input` |

Hovering a pin shows its accepted types plus direction; hovering a wire shows the type it carries.
`Image2D` is the GPU storage type for every per-pixel value: a socket's contract gives it meaning
(scalar field = R, vector field = RG, color = RGBA), so all three share the amber color and the
"per-pixel" tooltip wording.

### Materialized outputs and wires

An **output pin or wire whose concrete type is known** collapses to that single type: one square in
the resolved color, a label in the resolved color, and a tooltip naming exactly that type. The root
canvas first looks up the runtime's evaluated value for the output (Float / Float2 / Image), so
anything that has actually run states its result with certainty; when a node has not run yet it
falls back to the promotion rules below. Subgraph body nodes resolve by promotion only (their
body-local ids cannot be correlated with root runtime values).

Union outputs still show the full accept set while their type is undetermined. When the strong
`Image2D` subtyping the roadmap calls for arrives, these resolved lookups gain a fourth concrete
type and the promotion rules extend to it unchanged.

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
* `Field` becomes one sampler and is sampled automatically.
* `Empty` prevents generated dispatch until the source becomes available.
* Values produced inside a region remain expressions and are substituted directly.

The `ShaderValue` width (`Scalar`, `Vec2`, or `Vec4`) is independent of its constant/field
category. Use `scalar`, `vector`, or `color` when a socket has a fixed semantic width, and use
the raw `input` value plus `promotedShaderType` for component-wise operations whose width follows
their operands. `convertShaderValue` implements scalar broadcast, vector extension, and `.x`/`.xy`
projection consistently.

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
- Snap pass: any node feeding a single consumer moves onto that consumer's row when that row
  is free in the node's own column (overlap-checked against every node in the column, same
  component only, `kRowGap` effective spacing). Chains thus hug their downstream node.
- Hard limit: two single-consumer nodes in the same column can never both sit on their shared
  consumer's row; only relocating one to its own column (and only if that row is free there)
  fixes it. Dense graphs necessarily leave such merges one row off. Do not try to force this
  with an iterative solver — relaxation versions diverge or regress already-aligned junctions.

The auto-layout test contract lives in `tests/core_tests.cpp` (cases around lines 688-760):
chain collinearity, merge-at-average, disjoint components, single-consumer-through-merge snap.
