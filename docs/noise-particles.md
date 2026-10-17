# Sands: editable particle flow with ordinary rendering nodes

Open `examples/noise-particles.reaction.json` in Reaction Studio.

```text
Noise-Driven Particles (editable simulation subgraph)
    ├── Orange Positions → Particle Splat / Trails ─┐
    └── Teal Positions   → Particle Splat / Trails ─┤
                                                 Mix (Add, factor = 1)
                                                        │
                                                     Output
```

The **only** subgraph in the example is the particle simulation. Its motion,
angular branching, lifetime and respawn are editable Math/Hash/Select/Vector
nodes. Particle Splat is a normal, reusable **root-graph** native GPU node;
there are two entirely independent instances. The existing **Mix** node
composes them, in Add mode with factor 1. No bespoke two-splat subgraph
or image-addition node is required.

## Controls

Physics: Noise Scale, Noise Pan X/Y, Orange Angle Start/End, Direction
Quantization, common Particle Speed, separate Orange/Teal Speed, Life Decay,
tangent jump cap, and simulation iterations. The two output Vector Fields
use an exact (-1,-1) sentinel for inactive branch members; live positions
may travel out of bounds.

Each Particle Splat: count, individual RGB ink, point opacity, frame
retention, point size, margin and roundness. The color UI has three small
optional R/G/B input pins and one compact swatch, which opens a full color
picker; no permanently expanded three-channel controls are needed.

The renderer skips the inactive (-1,-1) sentinel and clamps real live
particles to the canvas inset. Each splat retains its own trail history.
Color mixing and additional postprocessing use existing root nodes.

Position state and exported branch fields use 32-bit GPU textures to
preserve subpixel motion. The simulation is still one particle per image
texel rather than a packed SSBO; cost scales with canvas resolution.

## Validation

CI checks separately masked branches, continuous movement, shader validity,
both root-level splats, built-in Mix Add composition, serialization and a
120-frame accumulated PNG preview artifact.
