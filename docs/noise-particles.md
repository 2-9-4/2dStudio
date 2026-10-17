# Sands: composable particle flow and trail layers

Based on SamuelYAN's p5.js sketch. The motion is constructed from editable
nodes, rather than reproducing p5.js RNG sequences exactly.

Open `examples/noise-particles.reaction.json` in Reaction Studio.

## Composition

```text
Noise-Driven Particles (shader-lowered simulation)
    ├── orangePositions → Particle Splat: orange ink ──┐
    └── tealPositions   → Particle Splat: teal ink   ──┤
                         [Sands Trails native pipeline] │
                                    Add Image Layers ←──┘
                                          │
                                        Output
```

**Both subgraphs are editable** (select and press Tab). A separate reusable
`Particle Splat / Trails` node draws one RGB ink from an active-position field;
positions outside [0,1] are skipped. An `Add Image Layers` node sums two images
with independent gains and one background color.

## Controls

The physics subgraph exposes Noise Scale, Noise Pan X/Y, Orange Angle Start/End,
Direction Quantization, common Particle Speed, separate Orange/Teal Speed,
Life Decay, tangent jump limit and simulation iterations. The body contains
visible Math, Hash, Select, Texture Sample, and Vector Math operations.

The Trails native pipeline exposes independent ink RGB, opacity, trail retention,
point size, and layer gains, plus particle sample count and background RGB.
The app provides RGB color pickers on both pipeline and native splat nodes.
Open the pipeline body to adjust each splat's canvas inset or point roundness.

Positions use RG32F feedback and 32-bit exported textures, preserving subpixel
movement. Both color branches share the underlying particle simulation.

## Execution design

Simulation subgraphs run shader-lowered pixel expressions. Native pipeline
subgraphs execute ordinary registered **native** GPU nodes in topological order,
including framebuffer rasterization and additive compositing. This is a
reusable feature beyond Sands; native pipelines currently do not accept
shader-only operators or nested subgraphs.

Simulation still stores one agent per state-image texel (not a compact particle
buffer), so work scales with image resolution. Rendering state is transient;
reopening a project starts fresh. Reopen the updated example to pick up the
new built-ins; older project-local duplicated subgraphs remain unchanged.

## Tests

CI checks branch-exclusivity, motion, RG32F precision, project serialization,
native pipeline rendering, and a longer screenshot artifact for visual review.
