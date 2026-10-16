# Noise-driven particle trails (Sands)

Inspired by SamuelYAN's p5.js sketch ([Instagram](https://www.instagram.com/samuel_yan_1990/),
[Twitter](https://twitter.com/SamuelAnn0924/)). This is a procedural adaptation,
not a bit-for-bit port of p5.js Perlin noise or the original RNG.

Run the preset:

\`\`\`sh
./build/reaction_studio examples/noise-particles.reaction.json
\`\`\`

Graph: **Noise-Driven Particles (Sands)** → **Particle Splat / Trails** → **Output**.
Select the Sands node and press **Tab** to open the editable simulation body.

## Subgraph anatomy (the algorithm is NOT a primitive)

Everything below is a visible, connected, editable node:

1. **Initialize slots** — Canvas Coordinates → Deterministic Hash → initial
   position and lifetime state. Separate Hash seeds make initial UV coordinates
   and initial life independent.
2. **Sample flow** — Previous Position → Texture Sample, driven by a
   *stationary*, one-octave Perlin field. Noise Scale controls its spatial frequency.
3. **Quantize angle** — Multiply by Direction Quantization → Round → Divide by
   Direction Quantization → Multiply by 2π.
4. **Choose branch** — Step at 3 and 6 radians, subtract and multiply to
   produce a binary orange-region mask.
5. **Orange motion** — Tangent → Clamp (using Tangent Jump Limit), multiplied by
   random(1,3) in X, same tangent in Y → Combine Vector.
6. **Teal motion** — Sine, Cosine, independent random(0.1,1) in X, Y fixed to
   negative cosine → Combine Vector.
7. **Integrate** — Select orange/teal vector → Vector Scale by Particle Speed →
   Pixel Delta to UV → Vector Add to the previous position.
8. **Life** — Two 2-channel Deterministic Hash nodes generate four independent
   uniform values → three multiplies → multiply by 0.1 → subtract from life →
   Step to detect expiration.
9. **Respawn** — Random Position and Random Life hashes drive two Select nodes:
   dead particles get new random states, survivors keep their integrated states.
10. **Persist** — Next Position, Next Life and Next Branch state endpoints.
    Two exported texture fields feed the separate renderer.

No custom node owns the actual motion, branching, lifetime, or respawn rules.
The only new lowerable mathematical primitive is **Pixel Delta to UV**, which
exposes the shader's pixel size as a vector conversion. Existing Math, Vector
Math, Hash, Select, Texture Sample, and state nodes do the rest.

All Perlin parameters are visible on the noise node; noise remains stationary
(speed zero) while agents independently move through it.

## Precision and appearance

Position integration is **RGBA/RG32F** through the private simulation
state and exported position texture. The default RG16F used by other simulations
lost movement smaller than about a fraction of a pixel, so simply verifying
nonblack output could never validate this algorithm.

Splat is deliberately separate: the current subgraph compiler cannot execute
OpenGL framebuffer point drawing. It reads the position and branch fields and
draws independently colored, partially transparent points into a persistent
RGBA canvas, then fades the canvas every frame. The resulting strokes emerge
from tracked trajectories, rather than sampled white noise.

Important parameters:

- **Noise Scale**: approximately 5 noise cells across the image; increasing
  makes more tightly curved shapes.
- **Direction Quantization**: 1200 steps by default, matching the original.
- **Particle Speed**: displacement measured in pixels per simulation step.
- **Tangent Jump Limit**: avoids overflow near π/2 and 3π/2 poles.
- **Visible Particles**: 1,000 deposited points per frame.
- **Frame Retention**: 0.985 by default for the intended time window, rather
  than the previous 0.996 that could accumulate many stale dots.
- **Point Opacity**: 0.375 by default.

The life update is the product of four independent uniforms divided by 10,
which has the same marginal distribution as the original four nested
random() calls; exact stochastic sequences still differ.

The renderer clamps drawn positions to a 20-pixel inset (not the simulation
positions). Out-of-bounds noise samples repeat rather than extending unbounded
p5.js noise. Source-derived tangent displacement, alpha, and two-color branch
semantics are otherwise retained.

## Engineering limitations and footguns

- The existing backend encodes **one particle per simulation texel**, not
  packed SSBO agents. This is wasteful: simulation runtime scales with canvas
  resolution, not with the visible point count. A dedicated particle-buffer
  backend would be the correct longer-term optimization.
- The renderer is a separate stateful native node; its internal fading
  framebuffer isn't editable *inside* the simulation subgraph.
- Simulations require exactly one Previous/Initial/Next accessor for *every*
  private slot, even when the previous value is unused.
- Headless core tests use a miniature mock registry and must list every new
  built-in node contract separately from the actual GPU registry.
- Root and subgraph canvas node labels had been hidden (used only for shader
  inspection), which made large graphs very hard to understand. This PR
  shows the instance labels in both canvases.
- Cached project-local copies of the *previous* built-in definition are not
  replaced automatically. Open the updated example or re-add the built-in
  Sands subgraph to see the new wiring.
- Simulation texture contents are not persisted to project JSON; reload
  restarts the drawing.

## Verification

The CI run compiles generated initialization/update shaders with headless
Mesa, verifies the subgraph contains ordinary math/select/hash nodes rather
than a monolithic step, checks persistence of high-precision state, and
compares exported particle positions over multiple frames to detect frozen
motion. It also verifies the trails image contains nontrivial finite color.
