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

---
