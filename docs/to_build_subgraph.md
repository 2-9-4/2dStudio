# 1. Which proposed nodes should be editable subgraphs?

## Excellent stateless editable-subgraph candidates

These require no persistence. They are just reusable graphs composed from field math and a relatively small primitive set.

**2. 2D Transform**
**3. Polar Coordinates**
**4. Log-Polar Coordinates**
**5. Kaleidoscope / Angular Fold**
**6. Twirl / Swirl**
**7. Radial / Lens Distortion**
**10. Mirror / Fold Coordinates**
**11. Repeat / Wrap Coordinates**
**12. Coordinate Quantize**
**13. Domain Warp**
**14. Complex Plane Transform**

These are almost pure mathematical functions of coordinates. They are ideal subgraphs.

**16. Value Noise**
**19. Fractal Noise Composer**
**20. Ridged Multifractal**
**21. Turbulence Noise**
**22. Billow Noise**
**23. Curl Noise**

Value Noise becomes an editable subgraph once a deterministic Hash primitive exists. The fractal variants can then compose existing noise generators.

**27. Noise Derivatives**
**28. Gradient Generator**
**29. Wave Generator**
**30. Checker / Grid**
**31. Stripe Generator**
**32. Ring Generator**
**33. Radial Spokes**
**34. Spiral Generator**
**35. Moiré / Interference Generator**

All are straightforward field mathematics.

**36. SDF Primitive**
**37. SDF Boolean**
**38. SDF Transform**
**39. SDF Repeat**
**40. SDF Stroke / Contour**
**41. Superformula / Supershape**
**42. Harmonic Pattern**

SDFs are especially good candidates. I would seriously consider shipping most SDF shapes as editable built-in subgraphs rather than native nodes.

**43. Tile Generator**
**44. Tile Sampler**
**45. Brick / Bond Generator**
**46. Lattice Generator**
**47. Truchet Tiles**
**48. Wang Tile Generator**

These become practical with Hash, Texture Sample, floor/fract/modulo, and Vector Compose/Split.

**53. Image Gradient**
**56. Divergence**
**57. Curl / Vorticity**
**58. Vector Field Transform**
**59. Advect Image**
**60. Flow Map Distort**

These are excellent subgraphs once arbitrary texture sampling and vector operations are available.

**73. Bevel from Distance**
**74. Morphology Suite**

Morphology can directly reuse your existing Convolution node's `Erosion` and `Dilation` operations if Convolution is permitted inside editable subgraphs.

**75. Color Space Convert**
**76. HSV/HSL Adjust**
**77. Levels**
**79. Posterize / Quantize**

These are pointwise formulas and very good subgraphs.

**81. Dither**, for:

* Bayer
* Random
* Blue-noise thresholding

Floyd-Steinberg is a separate problem because it has raster-order dependencies.

**93. Edge / Derivative Suite**
**94. Local Statistics**

Both can largely be built from Convolution and ordinary Math.

---

# 2. Excellent simulation-subgraph candidates

These are where your existing persistent-state architecture becomes particularly valuable.

## 83. Feedback

This is almost the simplest possible simulation subgraph:

`Previous State → transformed/processed → Next State`

with Current Input mixed into the update.

The existing reset and initial-condition machinery already solves the awkward parts.

---

## 85. Temporal Accumulate

State is simply the previous accumulated image.

Examples:

`next = current + previous`

`next = mix(current, previous, decay)`

`next = max(current, previous)`

`next = min(current, previous)`

All of these should be editable subgraphs rather than native implementations.

---

## 86. Trails / Decay

Likewise:

`next = max(current, previous * decay)`

or:

`next = current + previous * decay`

This is almost a textbook use of the simulation-subgraph system.

---

## 88. Generic Cellular Automata

This is an **excellent** simulation-subgraph candidate and I think it can be expressed sanely.

For binary outer-totalistic automata, the state is one Scalar Field containing exactly `0` or `1`.

Use this convolution kernel:

```text
1 1 1
1 0 1
1 1 1
```

with normalization disabled.

That produces:

`NeighborCount = 0..8`

Then define two integer bit masks:

`Birth Mask`

`Survival Mask`

Bit N means "neighbor count N is enabled."

For Conway's Life:

`B3 = 1 << 3 = 8`

`S23 = (1 << 2) | (1 << 3) = 12`

The graph is conceptually:

```text
Previous State
      |
      +------------------------------ Alive
      |
Convolution
3x3 neighbor sum
      |
 Neighbor Count
      |
      +---- Bit Test(Birth Mask) ---- Birth
      |
      +---- Bit Test(Survival Mask) - Survive

Select(
    condition = Alive,
    true      = Survive,
    false     = Birth
)
      |
Next State
```

So Conway, HighLife, Seeds, Day & Night, Maze, etc. are all the **same editable graph** with different B/S parameters.

The public UI can display a human-readable rule such as `B36/S23`, while internally exposing two integer masks.

That is much cleaner than implementing each cellular automaton separately.

---

## Multi-state cellular automata

A more general totalistic CA can use:

`Current State`

plus:

`Neighbor Sum`

to index a lookup table.

For N possible cell states and maximum neighbor sum M:

`Index = CurrentState * (M+1) + NeighborSum`

Then:

`NextState = LookupTable[Index]`

That makes arbitrary finite-state outer-totalistic automata expressible without hardcoding their rules into shaders.

A general **Lookup Table primitive** is therefore worthwhile.

---

## 89. Continuous Cellular Automata / Lenia

Also an excellent candidate.

The graph is essentially:

```text
Previous State
      |
Convolution with radial kernel
      |
Neighborhood Potential U
      |
Growth Function
      |
* dt
      |
+ Previous State
      |
Clamp 0..1
      |
Next State
```

The canonical growth function can itself be ordinary Math:

`G(U) = 2*exp(-0.5*((U-mu)/sigma)^2)-1`

The only significant issue is convolution radius. Your existing 15×15 maximum is fairly restrictive for Lenia. Either increase the supported kernel size or eventually support a larger convolution primitive.

The dynamics themselves do not need native code.

---

## 71. Distance Transform

Surprisingly, this can be an editable simulation subgraph if you are comfortable implementing an approximate **Jump Flood Algorithm**.

State contains:

* nearest known seed position: Vector Field
* valid flag: Scalar Field

Initialization:

```text
if Mask:
    CandidatePosition = CanvasCoordinates
    Valid = 1
else:
    CandidatePosition = arbitrary
    Valid = 0
```

Each iteration samples candidates at:

```text
(-step,-step) (0,-step) (+step,-step)
(-step, 0)    (0, 0)    (+step, 0)
(-step,+step) (0,+step) (+step,+step)
```

and keeps the candidate whose stored seed position is closest to the current pixel.

The stride changes by iteration:

`stepPixels = 2^(TotalIterations - IterationIndex - 1)`

After the final iteration:

`Distance = length(CanvasCoordinates - CandidatePosition)`

This requires **Iteration Info** and arbitrary offset Texture Sample.

This would be a particularly compelling demonstration of what editable simulation graphs can do.

---

## 72. Signed Distance Transform

This can use the same Jump Flood algorithm twice in parallel:

* nearest foreground
* nearest background

Then:

```text
if foreground:
    -distanceToBackground
else:
    +distanceToForeground
```

This becomes easy if simulation state can contain multiple independent typed state fields.

---

## 84. Frame Delay

A fixed delay is expressible if multiple state slots exist.

For example, three-frame delay:

```text
State2 <- State1
State1 <- State0
State0 <- Current
Output = Previous State2
```

A dynamically selectable arbitrary N-frame ring buffer is better implemented natively, but small fixed delays work naturally as editable subgraphs.

---

## 87. Optical Flow

A full pyramidal production-quality optical-flow implementation is not a great subgraph candidate.

However, a **single-scale Horn-Schunck optical flow** absolutely is.

Persistent state:

* Flow X
* Flow Y
* Previous Frame

During iterative solving, calculate spatial and temporal derivatives and repeatedly update velocity from neighborhood averages.

This would make an excellent editable "educational/procedural" optical-flow node even if you later add a higher-quality native pyramidal implementation.

---

## 90. 2D Fluid Simulation

This is more complicated, but I think your simulation system is already close to being able to express it.

Persistent state needs at least:

* Velocity: Vector Field
* Dye: Color Image
* Pressure: Scalar Field
* possibly Divergence: Scalar Field

The complication is that one simulation timestep contains different phases:

1. apply forces
2. advect velocity
3. compute divergence
4. perform N pressure Jacobi iterations
5. subtract pressure gradient
6. advect dye

Your existing "repeat the whole graph N times" model becomes sufficient if the graph knows **which iteration it is currently evaluating**.

You can schedule phases:

```text
iteration 0:
    forces + velocity advection + divergence + pressure initialization

iterations 1 .. N-2:
    pressure Jacobi iterations

iteration N-1:
    pressure projection + dye advection
```

`Select` nodes choose which state update applies based on Iteration Index.

That is somewhat ugly internally, but it means even a fluid solver can remain an editable graph.

---

## 96. Skeletonize

The **Zhang-Suen thinning algorithm** is a very good simulation-subgraph candidate.

Each iteration samples the eight neighbors:

```text
P2 P3 P4
P9 P1 P5
P8 P7 P6
```

and calculates:

* number of occupied neighbors
* number of 0→1 transitions around the ring
* several products of specific neighbors

The algorithm alternates between two deletion conditions.

Use:

`IterationIndex mod 2`

to select which thinning pass applies.

The state converges naturally; additional iterations leave the skeleton unchanged.

This requires no global data structure.

---

## 98. Flood Fill / Region Grow

Another excellent simulation subgraph.

Persistent state is the reached-region mask.

Initialization:

`State = SeedMask AND PassableMask`

Update:

```text
Neighbors = max(N,S,E,W previous state)

Next =
    PassableMask
    AND
    (PreviousState OR Neighbors)
```

Use eight neighbors instead for 8-connectivity.

Run for enough iterations and the region propagates until convergence.

---

# 3. Nodes that are technically possible as subgraphs, but I would not make that the canonical implementation

## 8. Perspective Transform

It can be written entirely with scalar math by solving the closed-form homography coefficients.

But the resulting graph is large and inscrutable.

I would implement the projective solve natively and potentially expose the rest of the transform as a subgraph.

---

## 9. Quad / Bilinear Warp

Also mathematically expressible, but inversion of the bilinear mapping requires a quadratic solve or iterative Newton solve.

Possible, but not a good user-editable graph.

---

## 17. Simplex / OpenSimplex

Can be constructed from Hash + scalar/vector math.

But the graph would be enormous and difficult to inspect.

A native primitive noise implementation is justified.

---

## 18. Worley / Voronoi

Technically you can manually generate and test the 25 nearby feature points in a graph.

I would not.

The nearest/second-nearest neighborhood search should be native.

You can then build many specialized Voronoi effects as editable subgraphs on top of the primitive Worley outputs.

---

## 24. Flow Noise

Possible if gradient-noise internals are exposed, but sufficiently bulky that I would retain a native noise primitive.

---

## 25. Gabor Noise

The variable collection of nearby impulses makes this a poor graph representation.

Native.

---

## 49. Quasiperiodic Tiling

Substitution tilings involve dynamically generated geometry/collections.

Native or CPU preprocessing.

---

## 63–70. Larger filtering family

* Gaussian Blur
* Directional Blur
* Radial Blur
* Vector Motion Blur
* Slope Blur
* Median
* Bilateral
* Kuwahara

Some could technically be built if you add a generic loop-over-neighborhood primitive, but at that point that primitive is doing most of the actual algorithm.

I would keep these native/local-filter nodes.

---

## 78. Curves

The surrounding channel logic could be a subgraph, but the editable curve/LUT evaluation itself should be primitive.

---

## 80. Multi-stop Color Ramp

Same situation: color-space conversions around it can be subgraph logic, but the variable-size editable stop list is better represented by a native LUT primitive.

---

## 97. Connected Components

The label-propagation portion can be simulated:

`NextLabel = minimum(valid neighboring labels)`

repeated until stable.

But unique pixel/component IDs ideally require integer state. Component area and component count require global reductions.

So a partial editable implementation is possible, but a complete Connected Components node should probably remain native.

---

# 4. Nodes that do not fit the image-field subgraph architecture well

These need fundamentally different execution/data models.

**50. Density Scatter** — produces Point Set.
**51. Poisson-Disc Scatter** — globally coupled Point Set algorithm.
**52. Packing / Relaxation** — Point Set simulation/spatial acceleration.
**61. Streamline Integration** — dynamic trajectory/rasterization workload.
**91. GPU Particles** — persistent Particle Buffer rather than image state.
**92. Physarum** — requires particle state in addition to image state.
**95. Histogram / Normalize** — requires global reduction.
**99. Resize / Resample / Crop** — changes graph resolution/storage.
**100. Custom GLSL** — should itself be a compiler primitive/escape hatch.

These can become subgraphs later if your execution model expands beyond dense images.

---

# 5. Primitive nodes I would add

The following small primitive set dramatically expands what editable subgraphs can represent.

# Primitive A — Texture Sample / Sample Field - DONE

This is the previously specified Texture Sample node and should be considered a **compiler primitive**, not implemented as a subgraph.

## Inputs

**Source**

* `Scalar Field | Vector Field | Color Image`

**Coordinates**

* `Float2 | Vector Field`
* Default: Canvas Coordinates.

Alternatively expose U and V separately.

## Parameters

Sampling:

* Nearest
* Linear
* Cubic

Address:

* Clamp
* Repeat
* Mirror
* Border

## Output

Same type as Source.

For every output pixel:

`Output(p) = Source(Coordinates(p))`

The compiler must understand that Source is being sampled at a coordinate other than the current pixel.

That property is what makes this fundamentally different from Math.

---

# Primitive B — Pixel / Canvas Metrics

## Purpose

Expose render resolution so graph math can correctly express pixel-neighborhood operations.

## Inputs

None.

## Outputs

**Resolution**

* `Float2`
* `(width,height)` in pixels.

**Pixel Size**

* `Float2`
* `(1/width,1/height)` in normalized coordinates.

**Aspect Ratio**

* `Float`
* `width/height`.


We already have a "canvas coordinates" node that outputs seperate x and y channels. Merge those into 1 vector output, and give it a normalization on/off checkbox(so it goes from 0 to 1 instead of pixels). Fix the default reaction diffusion subgraph by replacing the work it currently does with our new vector math node.Do not worry  about fixing existing files, we can just break those
**Pixel Coordinates**

* `Vector Field`
* coordinates in pixel units.

**Normalized Coordinates**

* `Vector Field`
* ordinary Canvas Coordinates.

This makes an exact neighbor sample expressible as:

`CanvasCoordinates + PixelSize * (1,0)`

followed by Texture Sample.

---

# Primitive C — Extended Math operations - DONE

Your existing Math node should gain these modes because many proposed subgraphs otherwise need dedicated nodes solely for elementary mathematics.

## Unary

* `Floor`
* `Ceil`
* `Round`
* `Fraction`
* `Square Root`
* `Exp`
* `Natural Log`
* `Log2`
* `Sign`
* `Tangent`
* `Arc Sine`
* `Arc Cosine`
* `Arc Tangent`

## Binary

* `Modulo`
* `Arc Tangent 2`
* `Step`
* `Hypotenuse`

## Ternary

* `Smoothstep`
* `Multiply accumulate`

Normal Float/Scalar-Field promotion applies.

For Modulo:

`mod(a,b)=a-b*floor(a/b)`

rather than relying on implementation-dependent integer remainder semantics.

Unsafe operations use documented epsilon behavior.

---

# Primitive D — Compare / Logic - DONE

Threshold alone becomes awkward once graphs start implementing algorithms.

## Inputs

A and B:

* `Numeric`

Optional C:

* `Numeric`

## Modes

Comparison:

* `<`
* `<=`
* `>`
* `>=`
* `==`
* `!=`
* `Between Inclusive`
* `Between Exclusive`

Logic:

* `AND`
* `OR`
* `XOR`
* `NOT`

## Output

Numeric of promoted type, containing exactly:

`0.0` or `1.0`

For logic operations, an input is true when:

`abs(value) > epsilon`

Equality should either be exact or expose an explicit Epsilon parameter. I would favor an `Approximately Equal` mode rather than silently applying tolerance to `==`.

---

# Primitive E — Vector Compose / Split - DONE

## Compose

Inputs:

`X: Numeric`

`Y: Numeric`

Output:

`Float2 | Vector Field`

according to normal broadcasting.

## Split

Input:

`Float2 | Vector Field`

Outputs:

`X: Float | Scalar Field`

`Y: Float | Scalar Field`

No implicit coordinate meaning is attached.

This primitive becomes ubiquitous and should be extremely lightweight.

---

# Primitive F — Vector Math

## Inputs

A:

* `Vector Numeric`

B:

* optional `Vector Numeric`

Scalar:

* optional `Numeric`

## Operations

Vector outputs:

* Add
* Subtract
* Multiply Components
* Divide Components
* Scale
* Normalize
* Rotate
* Perpendicular CW
* Perpendicular CCW
* Reflect
* Min Components
* Max Components

Scalar outputs:

* Length
* Length Squared
* Dot
* Distance
* Angle

Normalize returns `(0,0)` when magnitude is below epsilon.

Rotate uses radians and the application's +Y-down rotation convention.

---

# Primitive G — Deterministic Hash

This is essential for procedural subgraphs.

## Inputs

**Position**

* `Float2 | Vector Field`

**Seed**

* `Float`

**Salt**

* optional Float

## Parameters

Input handling:

* `Integer Coordinates`
* `Floor Position`
* `Raw Float Bits`, potentially later.

## Outputs

**Scalar**

* Float/Scalar Field in `[0,1)`.

**Vector**

* Float2/Vector Field with independently hashed X/Y values in `[0,1)`.

The canonical use is:

`Hash(floor(position),seed)`

The implementation should use a fixed integer hash algorithm whose output will never silently change between versions.

Hash must be deterministic across GPU vendors.

---

# Primitive H — Bit Test / Integer Mask

This makes configurable cellular automata much cleaner.

## Inputs

**Mask**

* `Float`
* interpreted as a nonnegative integer.

**Bit**

* `Numeric`
* rounded to nearest integer.

## Output

Numeric containing `0` or `1`.

Semantically:

`Output = ((uint(Mask) >> uint(Bit)) & 1) != 0`

Out-of-range or negative bits return 0.

At minimum, masks up through 24 bits are safe even if represented through IEEE-754 float values.

For Life-like cellular automata only bits 0–8 are required.

This primitive is useful outside CA as well for compact rule/state masks.

---

# Primitive I — Lookup Table

## Purpose

Map discrete or continuous scalar indices through an editable table.

## Input

**Index**

* `Numeric`

## Parameter

An editable array:

`Values = [v0, v1, ... vN-1]`

## Parameters

Sampling:

* Nearest
* Linear

Address:

* Clamp
* Repeat
* Mirror

Index units:

* Direct Index
* Normalized 0–1

## Output

Numeric matching Index promotion.

For Direct/Nearest:

`Output = Values[clamp(round(Index),0,N-1)]`

For normalized indexing:

`tablePosition = Index*(N-1)`

then apply selected interpolation.

This primitive enables:

* multi-state CA rules
* arbitrary transfer functions
* palette indices
* waveform tables
* small procedural state machines

The table should compile either into constants or a small 1D texture depending on size.

---

# Primitive J — Convolution inside subgraphs

You already have this node. I would simply make it legal inside editable subgraphs.

Its existing capabilities already provide several critical primitives:

* arbitrary weighted neighborhood sum
* erosion / neighborhood minimum
* dilation / neighborhood maximum

For simulation work, add one requirement:

**Normalization, kernel coefficients, and radius must be resolved before each dispatch and remain fixed during that simulation iteration.**

Kernel values can still be driven by ordinary Float parameters if you later choose to make them connectable.

The existing 15×15 limit is sufficient for:

* cellular automata
* derivatives
* many morphology operations
* local statistics
* simple PDEs

but is restrictive for Lenia and larger physical kernels.

I would eventually raise it significantly.

---

# Primitive K — Generalized Simulation State Slots

This is the most important architecture extension.

Reaction Diffusion currently conceptually has:

`Previous State = RG`

containing two chemicals.

Generalize that into **named typed state slots**.

## State declaration

Each editable simulation subgraph may declare zero or more state variables:

```text
State "A"        Scalar Field
State "B"        Scalar Field
State "Velocity" Vector Field
State "Dye"      Color Image
State "Pressure" Scalar Field
```

Each slot has:

* Name
* Type
* Initial value expression
* Previous value
* Next value

## Previous State node

Parameter:

`State Slot`

Output:

same type as selected slot.

It returns the value stored at the beginning of the current simulation iteration.

## Next State node

Parameter:

`State Slot`

Input:

same type as slot.

At the completion of the current iteration, this becomes the stored value for the next iteration.

All state slots update **simultaneously**.

That last rule is important.

If:

```text
Next A = Previous B
Next B = Previous A
```

the values swap; evaluation order must not affect behavior.

## Initial State

Each state slot has an initial-value graph expression.

Reset atomically replaces all slots with their initial values before the next simulation iteration.

This generalization makes Fluid, Signed Distance JFA, optical flow, multi-species reaction systems, and many other simulations possible without special architecture.

---

# Primitive L — Simulation Iteration Info

This is the other major missing piece.

## Inputs

None.

## Outputs

**Iteration Index**

* Float integer value.
* First iteration = `0`.

**Iteration Count**

* Float integer value.

**Normalized Iteration**

* Float.
* `Index / max(Count-1,1)`.

**First Iteration**

* Float boolean 0/1.

**Last Iteration**

* Float boolean 0/1.

## Semantics

If a simulation body runs N iterations during one simulation step, its evaluations observe:

```text
Index = 0
Index = 1
...
Index = N-1
```

State written by iteration N becomes Previous State for iteration N+1.

External Subgraph Inputs remain fixed for the entire simulation step unless explicitly documented otherwise.

This primitive enables **different phases within one simulation step**, which is important for:

* Jump Flood
* Zhang-Suen thinning
* fluid pressure solving
* multi-pass iterative algorithms
* alternating red/black or odd/even solvers

---

# Primitive M — Simulation Step / Frame Info

This is distinct from iteration index.

## Outputs

**Delta Time**

* Float

**Simulation Step**

* monotonically increasing integer-valued Float

**Simulation Time**

* Float

**Was Reset**

* Float boolean

Optional:

**Frame Index**

* render frame index

Simulation Time advances once per outer simulation step, not once per internal iteration.

This prevents an eight-iteration simulation from accidentally advancing animation time eight times faster.

---

# Primitive N — State/Input Sample at Offset

This is optional because Texture Sample + Pixel Size can already express it, but I think it is worthwhile as a very small ergonomic primitive.

## Inputs

**Source**

* Image-Like

**Offset Pixels**

* `Float2 | Vector Field`
* Default `(0,0)`.

## Parameters

Sampling:

* Nearest
* Linear

Address:

* Clamp
* Repeat
* Mirror
* Border

## Output

Same type as Source.

Semantics:

`sampleCoordinate = CanvasCoordinates + OffsetPixels * PixelSize`

`Output = sample(Source,sampleCoordinate)`

This makes neighborhood algorithms visually understandable:

```text
North = Sample Offset(0,-1)
East  = Sample Offset(+1,0)
```

rather than requiring several coordinate arithmetic nodes.

I would still compile it to exactly the same machinery as Texture Sample.

---

# 6. What this primitive set buys you

With roughly these primitives:

* Texture Sample
* Pixel Metrics
* expanded Math
* Compare / Logic
* Vector Compose/Split
* Vector Math
* Hash
* Bit Test
* Lookup Table
* existing Convolution
* generalized state slots
* Iteration Info
* simulation time info

you can build an unexpectedly large fraction of the remaining system as editable graphs.

The important architectural point is that **Convolution + arbitrary sampling + persistent typed state + iteration index** is almost a small GPU cellular/PDE programming language.

Reaction Diffusion is only one example.

The same substrate can express:

```text
Reaction diffusion
Cellular automata
Lenia
Flood fill
Morphological evolution
Jump-flood distance fields
Skeletonization
Feedback systems
Temporal accumulation
Semi-Lagrangian advection
Iterative optical flow
Jacobi pressure solvers
Fluid dynamics
Wave equations
Diffusion equations
Gray-Scott variants
Multi-species reaction systems
Excitable-media models
```

without adding special execution concepts for each one.

# 7. Recommended immediate architecture changes

I would prioritize these four changes before implementing many more high-level nodes:

1. **Allow Texture Sample and Convolution inside simulation subgraphs.**

2. **Replace the special two-channel reaction-diffusion state with arbitrary named typed state slots.**

3. **Expose Iteration Index / Count inside the subgraph.**

4. **Add the missing mathematical primitives: floor, fract, modulo, sqrt, exp, log, atan2, comparisons, vector compose/split, and deterministic hash.**

At that point, **Generic Cellular Automata** would be the next simulation I would implement as an editable built-in. It is simple enough to validate the generalized architecture, but different enough from reaction diffusion to expose bad assumptions.

After that I would implement, in order:

```text
Flood Fill
Lenia
Jump-Flood Distance Transform
Skeletonization
Feedback / Trails
2D Fluid
```

That progression increasingly stress-tests neighborhood access, multiple state fields, iteration-dependent logic, and arbitrary coordinate sampling without requiring particle buffers or global reductions.
