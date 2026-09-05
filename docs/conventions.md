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

---
