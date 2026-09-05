# Reaction Studio nodes

## Purpose

Reaction Studio is a Linux-first, real-time 2D generative-art node editor. It lets you
build forward-only image and scalar graphs, generate animated textures, process images on
the GPU, and create reaction-diffusion patterns. Graphs can combine ordinary image/scalar
operations with editable simulation subgraphs, and the result can be previewed, exported as
PNG, or recorded as H.264 video.

This document covers the built-in nodes available in the root graph, their selectable
submodes, and the special nodes available inside simulation subgraphs.

## Root-graph nodes

### Input

- **Image** (`image`) — Loads a PNG file and outputs it as an image. An empty or invalid
  path produces no image.
- **Float** (`float`) — Outputs a constant scalar value.
- **Resolution** (`resolution`) — Outputs render `Resolution` (width, height) in pixels,
  normalized `Pixel Size` (1/width, 1/height), and scalar `Aspect Ratio` (width/height).
- **Time** (`time`) — Outputs elapsed time and frame delta, both multiplied by `Speed`.
  This is the usual source for time-based animation.
- **Canvas Coordinates** (`coordinates`) — Outputs one Vector Field containing normalized XY
  canvas coordinates. Enable **Pixels** to output pixel-center coordinates instead.

### Generators

- **Perlin Noise** (`perlin`) — Generates animated fractal noise. Its controls are `Seed`,
  `Scale`, `Octaves`, `Persistence`, `Lacunarity`, `Speed`, `Offset X`, and `Offset Y`.
  More octaves add detail; persistence controls how quickly successive octaves fade.

### Math and logic

- **Table** (`table`) — Maps a Numeric **Index** through an editable scalar table. **Sampling**
  selects nearest or linear interpolation; **Address** selects clamp, repeat, or mirror;
  **Index Units** selects direct table positions or normalized 0–1 positions. Its output follows
  the Index promotion rule: a Float Index produces a Float and a Scalar Field produces a Scalar
  Field. Direct indices use `round(Index)` for nearest sampling. Normalized indices first use
  `Index * (N - 1)`. Tables currently contain 1–64 values and are emitted as specialized shader
  constants.

- **Math** (`math`) — Applies one numeric operation to up to three scalar or image-valued
  inputs. Its operation submodes are:

  - **Add** — Adds A and B.
  - **Subtract** — Computes A minus B.
  - **Multiply** — Multiplies A and B.
  - **Divide** — Divides A by B with a small-denominator safeguard.
  - **Power** — Raises A to the power B, with sign-preserving handling for negative A.
  - **Minimum** — Selects the lower of A and B.
  - **Maximum** — Selects the higher of A and B.
  - **Absolute** — Returns the absolute value of A.
  - **Sine** — Returns the sine of A.
  - **Cosine** — Returns the cosine of A.
  - **Clamp** — Restricts A to the range B–C.
  - **Remap** — Remaps A from `Input Min`–`Input Max` to `Output Min`–`Output Max`.

- **Vector Math** (`vector_math`) — Performs explicit 2D vector operations on a Vector or
  Vector Field. The active operation determines which sockets are visible: for example,
  **Normalize** exposes `A → Result` as Vector Numeric, **Dot** exposes `A`, `B → Result`
  with a Numeric result, and **Scale** exposes `A`, `Scalar → Result`. It includes component
  arithmetic, normalize/length controls, rotation and perpendiculars, reflect/project/reject,
  interpolation, distances, dot/cross products, and angle/component reductions. Vector Fields
  are semantically XY vectors stored in RG, never RGBA color math.

- **Deterministic Hash** (`hash`) — Produces stable scalar and XY vector hashes in `[0,1)`
  from a Vector Numeric position plus scalar Seed and optional Salt. Its fixed 32-bit integer
  mixing algorithm is versioned with the node. Input Handling provides **Canvas Space** for
  per-pixel vector-field variation and **Pixel Space** for integer cell hashing; the canonical
  cell use is `Hash(floor(position), seed)`.

- **Threshold** (`threshold`) — Converts a value to 0 or 1 depending on whether it is at
  least the configured threshold.
- **Select** (`select`) — Outputs `If True` when `Condition` is nonzero; otherwise outputs
  `If False`.

### Color

- **Mix** (`mix`) — Blends A and B using `Factor`. The blend submodes are:

  - **Mix** — Ordinary interpolation between A and B.
  - **Add** — A plus B.
  - **Multiply** — A multiplied by B.
  - **Screen** — A brightening blend that preserves highlights.
  - **Overlay** — Combines multiply and screen behavior according to A.
  - **Difference** — The absolute difference between A and B.
  - **Darken** — The darker value of A and B.
  - **Lighten** — The lighter value of A and B.
  - **Color Dodge** — Brightens A using B.
  - **Color Burn** — Darkens A using B.

- **Color Ramp** (`color_ramp`) — Maps a scalar value to a gradient between configurable
  start and end colors. `Low threshold` and `High threshold` define the mapped range.

### Filters and utilities

- **Convolution** (`convolution`) — Applies a configurable 3x3 through 15x15 neighborhood
  kernel, with optional weight normalization, bias, and repeated `Iterations`. Its
  operation submodes are **Convolution** (weighted sum), **Erosion** (minimum over enabled
  kernel taps), and **Dilation** (maximum over enabled kernel taps).

  The preset menu provides:

  - **Custom** — Keeps the manually edited kernel.
  - **Identity** — Leaves the current pixel unchanged.
  - **Box Blur** — A uniform averaging blur when normalization is enabled.
  - **Gaussian Blur** — A center-weighted blur.
  - **Sharpen** — Emphasizes the center relative to its neighbors.
  - **Edge Detect** — Highlights local changes.
  - **Emboss** — Produces a directional relief-style effect.
  - **Erosion** — Uses a disk-shaped morphology mask with the erosion operation.
  - **Dilation** — Uses a disk-shaped morphology mask with the dilation operation.

- **Laplacian** (`laplacian`) — Computes a weighted neighborhood difference, useful for
  detecting local structure or implementing reaction-diffusion terms. `Scale` controls the
  sampling radius; values other than 1 average several radii.
- **Image Invert** (`invert`) — Inverts RGB while preserving alpha.
- **Float Preview** (`float_preview`) — Materializes/displays a scalar value in the editor
  while allowing an optional input connection to override its fallback value.

### Simulation and output

- **Reaction Diffusion (Monolithic)** (`reaction_diffusion`) — A stateful Gray–Scott-style
  simulation implemented as one built-in node. It exposes feed/kill controls, diffusion
  rates, structure scale, timestep, iteration count, optional feed/kill multiplier images,
  and an optional seed image. `Auto Reset` reinitializes the simulation when activity has
  collapsed.
- **Output** (`output`) — Marks the image that becomes the graph’s rendered result.

## Built-in subgraph

- **Reaction Diffusion (Discrete)** — An editable simulation subgraph with the same public
  feed, kill, diffusion, timestep, iteration, auto-reset, multiplier, and seed controls as
  the monolithic node. It exposes three outputs: **Image**, **Chemical A**, and
  **Chemical B**. Unlike the monolithic node, its internal computation is visible and can be
  edited using the supported subgraph nodes below.

## Simulation-subgraph-only nodes

These nodes are scoped to a simulation body and are not ordinary root-graph nodes:

- **Subgraph Input** (`subgraph_input`) — Provides an interface input or slider to the
  internal graph. Input interfaces also provide a `Connected` flag so the graph can choose
  between a supplied value and a default.
- **Subgraph Output** (`subgraph_output`) — Exports an internal value through one of the
  subgraph’s named output interfaces.
- **Previous Simulation State** (`simulation_previous_state`) — Reads the previous
  iteration’s two-channel state. This is the controlled feedback boundary of the simulation.
- **Simulation Channel Split** (`simulation_channel`) — Splits a two-channel state into
  scalar **Chemical A** and **Chemical B** outputs.
- **Initial Simulation State** (`simulation_initial_state`) — Combines initial Chemical A
  and B expressions into the state used when the simulation resets.
- **Next Simulation State** (`simulation_next_state`) — Combines the updated Chemical A and
  B expressions and publishes them as the next simulation state and output channels.

The editable simulation canvas also permits these ordinary registered nodes: **Float**,
**Math**, **Threshold**, **Select**, **Canvas Coordinates**, and **Laplacian**. Nested
subgraphs and arbitrary pipeline subgraphs are not currently supported.
