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

- **Motion** dropdown: **Tangent** or **Sine / Cosine**, independent of ink
  deposit mode. Tangent computes
  `(clamp(tan(theta), ±maxJump) * random(horizontalMin,horizontalMax),
  clamp(tan(theta), ±maxJump))`; Sine/Cosine computes
  `(-sin(theta) * random(horizontalMin,horizontalMax), -cos(theta))`.
- **Deposit In** dropdown: **Inside Range** or **Outside Range**, evaluated at
  the particle's *final* position against the external field. The inside
  range is `[regionStart,regionEnd)` in radians, using the quantized field angle.
- **Region Angle Start** and **Region Angle End** are float inputs, adjustable
  per instance. The example wires **the same two Float nodes** to both,
  keeping the two territories complementary.
- **Seed, Particle Speed, Horizontal Jitter Min/Max, Direction Quantization,
  Life Decay, Tangent Jump Limit, Reset, Iterations per Frame** remain
  adjustable. Tangent Jump Limit has no effect in Sine/Cosine mode.

Changing either dropdown affects just that instance. Both trig algorithms
and the inside/outside decision are constructed from ordinary **editable**
Math, Vector Math, Sample and Select nodes—not hardcoded C++ branches or
bespoke native simulation kernels.

## Independent physics, complementary appearance

Each instance has three GPU state slots: high precision **position**,
**lifetime**, and high precision **visible positions**. Only position and life
feed back into the motion system; the third is output-only because the current
simulation engine requires outputs to come from a Next State endpoint.

At deposition time, an ineligible agent's *exported* position is `(-1,-1)`,
which Particle Splat ignores. Its underlying simulated position/life still
advance. Separate flows therefore produce complementary visual territories
without any collision, force, or cross-species simulation feedback.

Two ordinary root-level **Particle Splat / Trails** nodes retain independent
trail histories and ink colors; the existing **Mix (Add, factor 1)** combines
them. The splat UI has a compact color swatch and independently connectable
R/G/B pins.

## Validation

CI validates that only one species-flow definition is registered, verifies
both enum dropdown options, compiles the shared shader with tangent and
sine/cosine math, runs two parameterized instances against an external image,
checks visible particles' territorial assignment, independently evolving
state, serialization, and pixelwise Mix(Add) composition, then generates
a 120-frame `sands-preview` artifact.
