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

- **Image** (`image`) — Loads a PNG file and outputs a Color Image. An empty or invalid
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
  More octaves add detail; persistence controls how quickly successive octaves fade. Its output
  is a Scalar Field, not an untyped image.

### Math and logic

- **Bit Test / Integer Mask** (`bit_test`) — Tests whether a nonnegative, Float-backed integer
  **Mask** has the selected integer **Bit** enabled. **Bit** is Numeric, so a scalar field is
  tested per pixel and produces a scalar field of exact `0` or `1`. Inputs are rounded; invalid
  values are safe and masks support independently addressable bits `0..23`.

- **Integer Mask** (`integer_mask`) — Produces a Float-backed integer mask for Bit Test and
  similar nodes. Its editor accepts a decimal value, shows all 24 supported bits, and lets you
  toggle each bit directly.

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
  - **Floor**, **Ceil**, **Round**, and **Fraction** — Discrete and fractional-value operations.
  - **Square Root**, **Exp**, **Natural Log**, and **Log2** — Exponential and logarithmic operations.
  - **Sign**, **Tangent**, **Arc Sine**, **Arc Cosine**, **Arc Tangent**, and **Arc Tangent 2** —
    sign and trigonometric operations.
  - **Modulo**, **Step**, **Hypotenuse**, **Smoothstep**, and **Multiply Accumulate** —
    remaining utility operations.

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

- **Threshold** (`threshold`) — Converts a Numeric value to 0 or 1. Float input produces Float;
  Scalar Field input produces Scalar Field.
- **Select** (`select`) — Outputs `If True` when its Numeric `Condition` is nonzero; otherwise
  outputs `If False`. The two `AnyImageValue` branches alone determine the widest output type.

### Color

- **Mix** (`mix`) — Blends `AnyImageValue` A and B using a Numeric `Factor`. A and B resolve to
  their common widest semantic type; Factor does not participate in result-width inference. The
  blend submodes are:

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
  start and end colors. `Low threshold` and `High threshold` define the mapped range. It accepts
  Numeric input and always outputs a Color Image.
- **Color to R** (`color_r`) — Explicitly narrows a Color Image to its red Scalar Field.
- **Color to Luminance** (`color_luminance`) — Explicitly narrows a Color Image to a weighted
  RGB luminance Scalar Field.
- **Color to RG** (`color_rg`) — Explicitly narrows a Color Image to an RG Vector Field.

### Filters and utilities

- **Convolution** (`convolution`) — Applies a configurable 3x3 through 101x101 neighborhood
  kernel, with optional weight normalization, bias, sampling `Scale`, and repeated `Iterations`. Its
  operation submodes are **Convolution** (weighted sum), **Erosion** (minimum over enabled
  kernel taps), and **Dilation** (maximum over enabled kernel taps). It accepts Any Field and
  preserves Scalar Field, Vector Field, or Color Image identity. The size-aware **Lenia Ring**
  preset regenerates its normalized annular weights when Kernel Size changes. **Minimum kernel
  value** is a local slider (not a graph input): taps whose absolute weight is smaller are rounded
  to zero before normalization and omitted from generated shaders.

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
  sampling radius; values other than 1 average several radii. It accepts Any Field and preserves
  the input's semantic type.
- **Image Invert** (`invert`) — Accepts Any Field and preserves its semantic type. Scalar and
  vector inputs are inverted component-wise; Color Image RGB is inverted while alpha is preserved.
- **Texture Sample** (`texture_sample`) — Samples an Any Field at Vector Numeric coordinates and
  preserves the sampled field's scalar, vector, or color identity. Sampling selects nearest or
  linear filtering; Address selects clamp, repeat, mirror, or a supplied border value.
- **State/Input Sample at Offset** (`state_input_sample_offset`) — Samples an Any Field at the
  canvas coordinate plus **Offset Pixels** times Pixel Size. Offset Pixels accepts a Float2 or
  Vector Field and defaults to `(0,0)`; Sampling selects nearest or linear filtering and Address
  selects clamp, repeat, mirror, or a zero border. This is useful in simulation bodies for
  readable neighborhoods such as North `(0,-1)` and East `(1,0)`.
- **Float Preview** (`float_preview`) — Materializes/displays a scalar value in the editor
  while allowing an optional input connection to override its fallback value.

### Simulation and output

- **Reaction Diffusion (Monolithic)** (`reaction_diffusion`) — A stateful Gray–Scott-style
  simulation implemented as one built-in node. It exposes feed/kill controls, diffusion
  rates, structure scale, timestep, iteration count, optional feed/kill multiplier images,
  and an optional Scalar Field seed. Its result is a Scalar Field. `Auto Reset` reinitializes
  the simulation when activity has collapsed.
- **Output** (`output`) — Marks an Any Field value as the graph’s rendered result and preserves
  its Scalar Field, Vector Field, or Color Image identity.

## Built-in subgraph

- **Reaction Diffusion (Discrete)** — An editable simulation subgraph with the same public
  feed, kill, diffusion, timestep, iteration, auto-reset, multiplier, and seed controls as
  the monolithic node. Its Seed input and all three outputs—**Image**, **Chemical A**, and
  **Chemical B**—are Scalar Fields. Unlike the monolithic node, its internal computation is
  visible and can be edited using the supported subgraph nodes below.

- **Generic Cellular Automata** — An editable scalar-state simulation for Life-like rules.
  Connect a binary **Initial State**, then set Float-backed integer **Birth Mask** and
  **Survival Mask** values; bit *n* enables a birth or survival at *n* Moore neighbors.
  The defaults are Conway's Life (`B3/S23`, masks `8` and `12`). **Reset** reinitializes
  from Initial State, and **Iterations Per Step** controls update dispatches per frame.
  **Rule Preset** is a dropdown whose zero-based Float is consumed inside by Table nodes; choose
  Custom to respect the editable mask inputs. It includes Conway's Life, HighLife, Seeds, Day &
  Night, Maze, Replicator, and Life without Death.

- **Flood Fill** — An editable scalar region-grow simulation. **Passable Mask** limits the
  region, and **Seed Mask** starts it; an optional **Initial State** overrides the seed mask
  on reset. **Connectivity** selects 4- or 8-connected propagation, and **Reset** restarts the
  fill. Its **Region** output expands through passable pixels over successive simulation steps.

- **Skeletonization (Zhang-Suen)** — An editable binary-image thinning simulation. It thresholds
  **Initial Binary Image** at 0.5, then applies alternating Zhang-Suen sub-passes to preserve
  connected strokes while removing their interior. **Iterations Per Step** controls how many
  thinning sub-passes run per frame; use an even value to complete matching pairs in one frame,
  although the internal phase is preserved correctly for odd values. Pixels outside the canvas
  are treated as background. Its **Skeleton** output is a Scalar Field of 0 or 1.

- **Lenia** — An editable continuous scalar-state simulation. It convolves **State** through a
  normalized editable **Lenia Ring** kernel preset, then applies an exponential growth curve controlled by
  **Growth Center μ** and **Growth Width σ**. **Kernel Radius / Scale** controls the kernel's
  sampling radius, **Time Step** controls integration, and the clamped **State** output stays in
  the 0–1 range. The default kernel is a 15×15 annular shape; its editable **Kernel size** and
  live **Kernel Radius / Scale** control are intentionally separate, allowing both footprint and
  sampling radius to be tuned.

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
- **Simulation Iteration Info** (`simulation_iteration_info`) — Provides the current internal
  **Iteration Index**, **Iteration Count**, normalized index, and 0/1 first/last flags. The
  index runs from 0 through Count - 1 during each outer simulation step.
- **Simulation Step / Frame Info** (`simulation_step_info`) — Provides **Delta Time**,
  monotonically increasing **Simulation Step**, **Simulation Time**, 0/1 **Was Reset**, and
  render **Frame Index**. Simulation Time advances once per outer step, after all internal
  iterations complete.
- **Initial Simulation State** (`simulation_initial_state`) — Combines initial Chemical A
  and B expressions into the state used when the simulation resets.
- **Next Simulation State** (`simulation_next_state`) — Combines the updated Chemical A and
  B expressions and publishes them as the next simulation state and output channels.

The editable simulation canvas also permits lowerable ordinary registered nodes, including
**Float**, **Math**, **Bit Test / Integer Mask**, **Threshold**, **Select**, **Canvas Coordinates**,
**Laplacian**, **Texture Sample**, and **State/Input Sample at Offset**. Convolution exposes its preset/kernel editor there but is
single-pass only; its root-graph multi-pass **Iterations** control is unavailable. Nested
subgraphs and arbitrary pipeline subgraphs are not currently supported.
