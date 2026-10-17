# Independent species flow / external field

Open `examples/noise-particles.reaction.json` in Reaction Studio.

```text
Perlin Noise (or ANY other scalar image source)
  ├── field → Species 1 Flow (Tangent)      → Particle Splat → Mix(Add) → Output
  └── field → Species 2 Flow (Sine/Cosine) → Particle Splat ─┘
```

**Both simulations are distinct editable subgraphs** with entirely separate
position and lifetime feedback. The shared field is read-only: species do
not collide, interact or alter each other's simulation states. The root
example shares one Perlin node, but you can connect any scalar image from
another generator, imported image (after converting to a scalar field), or
processing graph. Perlin is not generated inside either species subgraph.

Both emit just one `positions` Vector Field. Color is exclusively a property
of each ordinary root-level Particle Splat node, never of the simulation.
The splat nodes feed the existing Mix node with mode Add and factor 1.

## Motion

Species 1 uses `(clamp(tan(theta))*random(min,max), clamp(tan(theta)))`.
Species 2 uses `(-sin(theta)*random(min,max), -cos(theta))`.
Both interpret their input field value as a quantized fraction of 2π and
sample it at their moving particle position. Their parameters expose
independent random seeds, speed, horizontal jitter min/max, direction
quantization, life-decay rate, reset and iterations per frame. Species 1
also exposes its tangent jump cap.

The body of each subgraph comprises ordinary editable Hash, Texture Sample,
Math, Select and Vector operations and two feedback slots (position and
lifetime). Positions are stored/exported in 32-bit textures for smooth
subpixel motion. Each species consumes separate GPU state but reads the
same external source image.

## Rendering and UI

Each root-level Particle Splat has independent RGB ink, point count,
opacity, retention, point size, brush roundness and canvas margin.
The compact Ink swatch opens a picker while the R/G/B pins remain
individually connectable, on one narrow row.

## Performance

Each species still simulates one agent per canvas texel rather than using
a compact SSBO, so two species have roughly twice the per-pixel simulation
work of a single species at the same resolution. The independent species
populations are not pixel-by-pixel reproductions of the old shared
population's mutually exclusive branch masks.

## Tests

CI checks that both species subgraphs compile without internal Perlin nodes,
both sample an external field and move independently, root splat images
are nontrivial, and built-in Mix(Add) produces the sum of their colors.
Also exports a 120-frame `sands-preview` image artifact.
