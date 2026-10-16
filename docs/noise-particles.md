# Noise-driven particle trails (Sands)

Inspired by SamuelYAN's p5.js sketch (credit: SamuelYAN; see the original artist's
[Instagram](https://www.instagram.com/samuel_yan_1990/) and
[Twitter](https://twitter.com/SamuelAnn0924)). This implementation adapts the
motion rules but is **not** numerically identical to p5.js noise or its RNG.

Open with:

```sh
./build/reaction_studio examples/noise-particles.reaction.json
```

Graph: **Noise-Driven Particles (Sands)** → **Particle Splat / Trails** → **Output**.

Select the simulation and press Tab to edit its body. It uses existing
Canvas Coordinates, Hash, Perlin Noise, Texture Sample, Simulation Step Info,
and state nodes, plus one lowerable **Particle Step** primitive.

The particle simulation is a *texel-per-agent* approximation: all pixels in the
simulation texture are independent particle slots. The Splat node deterministically
selects an evenly distributed subset of these slots and renders that number of
points with alpha blending onto a retained RGBA16F canvas. For performance,
use a 256–512 pixel project while experimenting; increasing resolution increases
simulation work quadratically even if Visible Particles stays at 1,000.

## Tunable settings

- **Noise Scale:** spatial frequency of the stationary noise used by the motion
  graph. This is the number of noise cells across the canvas, not the p5.js
  pixel divisor of 100.
- **Particle Speed:** scales pixel displacement per simulation step.
- **Tangent Jump Limit:** clamps the tangent singularity to avoid numerical
  overflow and unbounded particle coordinates.
- **Iterations per Frame:** one step per frame matches the original's schedule.
- **Visible Particles:** number of independently moving points deposited per frame.
- **Frame Retention:** retention multiplier applied once per rendered frame
  (0.996 by default).
- **Point Opacity:** overall opacity of each deposited point.

The two branches are retained: at quantized angle in (3,6), orange particles
move by `(tan(angle)*random(1,3),tan(angle))`. Otherwise teal particles move
by `(-sin(angle)*random(0.1,1),-cos(angle))`. Angles are quantized to 1,200 steps.
Lifetime starts uniformly random and decreases by the product of four
independent uniforms divided by ten; dead particles respawn at random UVs.
The renderer clamps **drawn points only** to a 20-pixel inset, without affecting
particle coordinates. Out-of-bounds noise sampling repeats the field rather than
continuing p5.js noise outside the canvas.

## Caveats / follow-up improvements

- The host's current simulation implementation runs on full-resolution image
  state. A dedicated particle-buffer execution backend would be more efficient.
- Particle Step combines the per-agent stochastic operations into one lowerable
  node because the existing graph language has no general per-agent random-stream
  or arbitrary multi-output state-update facility.
- Color and point-weight sampling approximate the original; no subpixel point
  rasterization or exact p5.js random seed compatibility is claimed.
- The trails buffer is deliberately not serialized; re-opening the project
  starts a fresh drawing. For a clean reset of both particle state and trails,
  use the graph's Reset Simulation action.
