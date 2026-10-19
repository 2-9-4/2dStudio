# Sands — reusable Species Flow simulation

Open `examples/noise-particles.reaction.json` in Reaction Studio.

```text
                           ┌─► Species Flow instance A ─► Splat A ─┐
External Perlin or image ──┤    Motion: Tangent                     Mix(Add) ─► Output
                           │    Deposit: Inside [3,6)              │
                           └─► Species Flow instance B ─► Splat B ─┘
                                Motion: Sine / Cosine
                                Deposit: Outside [3,6)
```

**There is exactly ONE built-in Species Flow subgraph**
(`builtin.particles.species_flow`). The two root nodes are **instances of the
same definition**, not copies or separately maintained templates. They have
independent position/lifetime state and their own seed, speed, jitter,
and rendering behavior. The shared scalar `field` image is an external,
read-only input; substitute another scalar image generator for Perlin.

## Per-instance controls

- **Motion** dropdown: **Tangent**, **Sine / Cosine**, **Double Angle**,
  **Harmonic Rosette**, or **Soft Tangent**, independent of ink
  deposit mode. Tangent computes
  `(clamp(tan(theta), ±maxJump) * random(horizontalMin,horizontalMax),
  clamp(tan(theta), ±maxJump))`; Sine/Cosine computes
  `(-sin(theta) * random(horizontalMin,horizontalMax), -cos(theta))`.
  Double Angle uses `(cos(2theta)*jitter, sin(2theta))`, Rosette
  `(sin(3theta)*jitter, cos(5theta))`, and Soft Tangent
  `(tanh(2*clampedTan)*jitter, tanh(2*clampedTan))`.
  The last three laws are built from ordinary editable Math nodes.
- **Deposit In** dropdown: **Inside Range** or **Outside Range**, evaluated at
  the particle's *final* position against the external field. The inside
  range is `[regionStart,regionEnd)` in radians, using the quantized field angle.
- **Region Angle Start** and **Region Angle End** are float inputs, adjustable
  per instance. The example wires **the same two Float nodes** to both,
  keeping the two territories complementary.
- **Boundary Fade (radians)** (`transitionWidth`): 0 restores the old hard
  cutoff. Larger values smoothly crossfade deposit opacity around BOTH angular
  thresholds, centered on each threshold. The example shares a third Float
  control between both species (initial value **0.45 radians**). Each
  instance computes a complementary per-particle weight, so a particle at
  the center of a boundary receives approximately 50% ink opacity.
- **Seed, Particle Speed, Horizontal Jitter Min/Max, Direction Quantization,
  Life Decay, Tangent Jump Limit, Reset, Iterations per Frame** remain
  adjustable. Tangent Jump Limit has no effect in Sine/Cosine mode.

Changing either dropdown affects just that instance. Both trig algorithms
and the inside/outside decision are constructed from ordinary **editable**
Math, Vector Math, Sample and Select nodes—not hardcoded C++ branches or
bespoke native simulation kernels.

## More reusable noise and nonlinear tools

The **Fractal Noise** generator outputs a scalar field in five styles:
gradient fBm, billow, ridged, sharp turbulence and value fBm, with
independent octave/seed/frequency/gain controls and an optional warped
coordinate-field input. Replace the example Perlin node with it, or
compose multiple noises using Math and Mix.

The **Curl Field** node converts any scalar field into a perpendicular
gradient Vector Field for downstream vector processing. The generic Math
node also supports Tanh, Sinh, Cosh, Sinc and Softsign for shaping movement
or field values. None of these nodes knows about species or ink.

## Independent physics, complementary appearance

Each instance has four GPU state slots: high precision **position**,
**lifetime**, high precision **visible positions**, and **per-particle ink
weights**. Only position and life feed back into motion. The two presentation
slots are output-only because simulation outputs must come from Next State
endpoints.

At deposition time, an agent's *exported* position is `(-1,-1)` where its
opacity weight is zero, and Particle Splat skips it. Along a soft boundary,
both species retain visible positions and export fractional weights between 0
and 1. The two weights add to 1 at the same field location. The underlying
simulated position/life always advance, including through invisible territory.
No inter-species physics or collision coupling is introduced.

Each `Particle Splat` now has a **generic optional `weights` Scalar Field
input** (one scalar per particle slot). Its vertex shader multiplies the
normal point opacity by that field. If unconnected, the splat behaves exactly
as before. Thus the fading behavior is represented as a reusable dataflow
contract, not color-specific special casing or a new custom compositor.

Two ordinary root-level **Particle Splat / Trails** nodes retain independent
trail histories and ink colors; the existing **Mix (Add, factor 1)** combines
them. The splat UI has a compact color swatch and independently connectable
R/G/B pins.

## Saved-project compatibility

Projects saved using the former built-ins
`builtin.particles.species_1_flow` and
`builtin.particles.species_2_flow` are automatically migrated on load
to the single `builtin.particles.species_flow` definition. Their former
motion mode and inside/outside deposition rule are reconstructed, with
existing per-instance controls preserved.

## Validation

CI validates that only one species-flow definition is registered, verifies
both enum dropdown options, compiles the shared shader with tangent and
sine/cosine math, runs two parameterized instances against an external image,
checks complementary hard and soft visibility, actual fractional ink weights,
weighted-versus-unweighted splat deposition, independently evolving state,
serialization, and pixelwise Mix(Add) composition, then generates
a 120-frame `sands-preview` artifact.
