# Shared type and evaluation rules

## Core data types

* **Float** — one scalar value for the entire graph/frame.
* **Float2** — one XY pair for the entire graph/frame.
* **Scalar Field** — one meaningful scalar per pixel.
* **Vector Field** — XY per pixel, stored in RG.
* **Color Image** — RGBA per pixel.
* **Point Set** — unordered collection of 2D points with optional scalar/vector attributes. This is a proposed new graph type required by scatter/packing nodes.
* **Particle Buffer** — persistent collection of particles containing at minimum position, velocity, age, lifetime, and ID. Proposed new graph/state type.
* **Histogram** — fixed-size array of scalar bin counts. Proposed auxiliary type for analysis nodes.

## Type aliases

**Numeric**

* `Float | Scalar Field`

**Vector Numeric**

* `Float2 | Vector Field`

**Image-Like**

* `Scalar Field | Vector Field | Color Image`

## Promotion

Numeric inputs broadcast naturally. A Float combined with a Scalar Field is treated as a constant at every pixel. If all Numeric inputs are Floats, the result is a Float; if any is a Scalar Field, the result is a Scalar Field.

The equivalent rule applies to Float2 and Vector Field.

Image-Like filters preserve the source type unless otherwise specified. Scalar operations applied to Vector Fields or Color Images operate component-wise unless explicitly defined otherwise.

## Coordinate convention

All spatial nodes use normalized coordinates unless otherwise stated:

* `(0,0)` = top-left.
* `(1,1)` = bottom-right.
* +X points right.
* +Y points down.
* Pixel coordinates refer to pixel centers.
* Angle `0` points right.
* Positive angles rotate clockwise.
* Angles are radians.
* `fract(x) = x - floor(x)`.
* `mod(a,b) = a - b*floor(a/b)`.

Neighborhood filters default to `Clamp` boundary handling unless an Address Mode is exposed.

---

# Coordinate spaces, transforms, and domain manipulation

## 1. Texture Sample / UV Remap

**Purpose:** Sample an arbitrary field or image at graph-provided coordinates.

**Inputs:** `Source: Image-Like`; `U: Numeric = Canvas U`; `V: Numeric = Canvas V`; `Border Value` matching Source type.

**Parameters:** Sampling = `Nearest | Linear | Cubic`; Address = `Clamp | Repeat | Mirror | Border`.

**Output:** same type as Source, always materialized at current canvas resolution.

For each output pixel:

`Output(x,y) = Source(U(x,y), V(x,y))`

Addressing is applied before sampling. `Repeat` uses `fract`; `Mirror` alternates direction every integer interval; `Border` returns Border Value outside `[0,1]²`. Floats connected to U/V produce constant coordinates across the image.

---

## 2. 2D Transform

**Purpose:** Produce transformed sampling coordinates.

**Inputs:** `Coordinates: Vector Numeric = Canvas Coordinates`; `Translation: Vector Numeric = (0,0)`; `Rotation: Numeric = 0`; `Scale: Vector Numeric = (1,1)`; `Shear: Vector Numeric = (0,0)`; `Pivot: Vector Numeric = (0.5,0.5)`.

**Output:** Float2 if all inputs are constant, otherwise Vector Field.

Controls use visual transform semantics. Positive X Translation makes subsequently sampled imagery move right; positive Rotation visually rotates clockwise. The node therefore outputs the inverse transform from destination coordinates to source coordinates.

The conceptual forward transform is:

`p = Pivot + Translation + R * H * S * (q-Pivot)`

and the node computes `q` from input `p`. Singular scale/shear values use an epsilon safeguard.

---

## 3. Polar Coordinates

**Purpose:** Convert between XY and radius/angle representations.

**Cartesian → Polar Inputs:** `Coordinates: Vector Numeric = Canvas`; `Center: Vector Numeric = (0.5,0.5)`; `Radius Scale: Numeric = 1`; `Angle Offset: Numeric = 0`.

**Outputs:** `Radius: Numeric`; `Angle: Numeric`; `Normalized Angle: Numeric`.

With `d = Coordinates-Center`:

`Radius = length(d)*RadiusScale`

`Angle = atan2(d.y,d.x)+AngleOffset`

`NormalizedAngle = fract(Angle/(2π))`

**Polar → Cartesian Inputs:** Radius, Angle, Center.

**Output:** Coordinates as Float2/Vector Field.

`X = Center.x + Radius*cos(Angle)`

`Y = Center.y + Radius*sin(Angle)`

Radius is not automatically normalized to canvas bounds.

---

## 4. Log-Polar Coordinates

**Purpose:** Convert Cartesian position into angle plus logarithmic radius, allowing multiplicative scaling to become translation.

**Inputs:** `Coordinates: Vector Numeric = Canvas`; `Center: Vector Numeric = (0.5,0.5)`; `Radial Scale: Numeric = 1`; `Log Base: Float = e`; `Radius Epsilon: Float = 1e-6`.

**Outputs:** `Log Radius: Numeric`; `Angle: Numeric`.

For radius `r = max(length(Coordinates-Center)*RadialScale, epsilon)`:

`LogRadius = log(r)/log(LogBase)`

`Angle = atan2(d.y,d.x)`

Inverse mode accepts Log Radius and Angle and computes:

`r = LogBase^LogRadius / RadialScale`

then reconstructs XY using sine/cosine.

Log Base must be positive and not approximately 1.

---

## 5. Kaleidoscope / Angular Fold

**Purpose:** Fold coordinates into repeated angular sectors around a center.

**Inputs:** `Coordinates: Vector Numeric = Canvas`; `Center: Vector Numeric = (0.5,0.5)`; `Segments: Numeric = 6`; `Rotation: Numeric = 0`; `Radial Scale: Numeric = 1`.

**Parameters:** `Mirror Alternate Sectors: bool = true`.

**Output:** transformed Coordinates.

Compute polar radius `r` and angle `a`. Sector width is:

`w = 2π / max(abs(Segments),epsilon)`

Local angle is:

`t = mod(a-Rotation,w)`

If mirrored:

`t = min(t,w-t)`

The returned angle is `t+Rotation`; radius is preserved except for Radial Scale. Convert back to XY.

Segments should normally be rounded to an integer ≥1 if supplied as a field.

---

## 6. Twirl / Swirl

**Purpose:** Rotate coordinates by an amount that varies with distance from a center.

**Inputs:** `Coordinates: Vector Numeric = Canvas`; `Center: Vector Numeric = (0.5,0.5)`; `Amount: Numeric = 1`; `Radius: Numeric = 0.5`; `Falloff Exponent: Numeric = 1`.

**Output:** transformed Coordinates.

For distance `r`, calculate normalized influence:

`t = clamp(1-r/Radius,0,1)^FalloffExponent`

For sampling semantics, source angle is:

`sourceAngle = destinationAngle - Amount*t`

so positive Amount makes subsequently sampled imagery visually rotate clockwise.

Outside Radius the coordinates remain unchanged. Radius ≤ epsilon disables the effect.

---

## 7. Radial / Lens Distortion

**Purpose:** Radially distort sampling coordinates around a center.

**Inputs:** `Coordinates`; `Center = (0.5,0.5)`; `Strength: Numeric`; optional `K2: Numeric = 0`; `Scale: Numeric = 1`.

**Parameters:** modes `Polynomial | Fisheye | Spherical`.

**Output:** Coordinates.

For Polynomial mode, normalize centered coordinate `d`, radius `r`, and compute source radius:

`rs = r * (1 + Strength*r² + K2*r⁴)`

Return `Center + normalize(d)*rs`.

Positive Strength therefore samples farther from center and visually produces barrel-like distortion; negative produces pincushion-like distortion.

Fisheye and Spherical modes use explicit monotonic radial mappings selected by mode. The center point maps to itself and zero Strength must be identity.

---

## 8. Perspective / Projective Transform

**Purpose:** Map one quadrilateral to another using an exact planar homography.

**Inputs:** `Coordinates: Vector Numeric = Canvas`; four `Source Corner: Float2`; four `Destination Corner: Float2`.

**Output:** transformed Coordinates.

Compute the unique 3×3 projective matrix `H` satisfying:

`DestinationCorner_i ~ H * SourceCorner_i`

For every destination coordinate `p`, calculate:

`q ~ inverse(H)*p`

and divide homogeneous X/Y by homogeneous W.

The output is q, suitable for Texture Sample.

Degenerate quads whose homography is singular should produce a documented fallback, preferably unchanged coordinates plus a node error flag rather than NaNs.

Corner positions are Float2 parameters, not fields.

---

## 9. Quad / Bilinear Warp

**Purpose:** Warp between arbitrary quadrilaterals without projective straight-line constraints.

**Inputs:** Coordinates; four Source corners; four Destination corners.

**Output:** transformed Coordinates.

Define bilinear interpolation inside a quad:

`B(s,t) = mix(mix(P00,P10,s), mix(P01,P11,s), t)`

For each destination position, solve:

`DestinationBilinear(s,t) = Coordinates`

for `(s,t)`, using analytic inversion where stable or a fixed deterministic Newton solve.

Then return:

`SourceBilinear(s,t)`

Unlike Perspective Transform, interior lines may curve and parallelism/projective geometry is not preserved.

Pixels outside the destination quad may either extrapolate `(s,t)` or return an explicit Outside Mask output; expose this as a parameter.

---

## 10. Mirror / Fold Coordinates

**Purpose:** Reflect 2D coordinates across simple or arbitrary axes.

**Inputs:** `Coordinates`; `Center: Float2 = (0.5,0.5)`; `Axis Angle: Float = 0`.

**Parameters:** `Vertical | Horizontal | Arbitrary Line | Diagonal X=Y`.

**Output:** Coordinates.

For an arbitrary line, construct unit axis vector `a=(cos θ,sin θ)` and perpendicular `n`. For centered point d:

`parallel = dot(d,a)*a`

`normal = dot(d,n)`

Fold one side onto the other by replacing normal with `abs(normal)` or `-abs(normal)` depending on selected side.

Recombine with Center.

Horizontal/Vertical modes are optimized equivalent cases.

---

## 11. Repeat / Wrap Coordinates

**Purpose:** Periodically remap coordinates.

**Inputs:** `Coordinates: Vector Numeric`; `Period: Vector Numeric = (1,1)`; `Offset: Vector Numeric = (0,0)`.

**Parameters:** each axis independently selects `Repeat | Mirror | Clamp`.

**Output:** Coordinates.

For Repeat:

`t = (value-offset)/period`

`out = offset + fract(t)*period`

Mirror maps each pair of periods through a triangle wave:

`m = mod(t,2)`

`out = offset + (1-abs(m-1))*period`

Clamp restricts to `[offset,offset+period]`.

Periods use epsilon protection against zero.

---

## 12. Coordinate Quantize

**Purpose:** Snap coordinates to a discrete spatial grid.

**Inputs:** `Coordinates`; `Step: Vector Numeric = (0.01,0.01)`; `Offset: Vector Numeric = (0,0)`.

**Parameters:** `Floor | Round | Ceil`.

**Output:** Coordinates.

For each axis:

`t = (Coordinate-Offset)/Step`

Apply selected integer quantizer Q:

`Output = Offset + Q(t)*Step`

A step of `(1/100,1/100)` therefore creates a 100×100 normalized sampling grid.

Step values near zero pass the input through unchanged.

---

## 13. Domain Warp

**Purpose:** Displace arbitrary coordinates using scalar offset fields.

**Inputs:** `Coordinates`; `X Offset: Numeric = 0`; `Y Offset: Numeric = 0`; `Strength: Numeric = 1`.

**Output:** Coordinates.

`Output.x = Coordinates.x + XOffset*Strength`

`Output.y = Coordinates.y + YOffset*Strength`

Offsets are normalized coordinate units.

The node performs exactly one displacement evaluation. It does not recursively reevaluate its displacement fields at the resulting position. Iterated domain warping should therefore be represented explicitly by graph composition unless upstream function reevaluation is later introduced.

---

## 14. Complex Plane Transform

**Purpose:** Treat centered XY coordinates as complex numbers and apply complex mappings.

**Inputs:** `Coordinates`; `Center = (0.5,0.5)`; mode-dependent `Exponent: Numeric`; complex coefficients `A,B,C,D: Float2`.

**Parameters:** `Conjugate | Reciprocal | Power | Exp | Log | Möbius`.

**Output:** Coordinates.

Represent:

`z = (x-Center.x) + i(y-Center.y)`

Apply selected operation.

Möbius:

`w = (A*z+B)/(C*z+D)`

Power:

`w = z^Exponent`

Log and reciprocal use epsilon handling around zero.

Return:

`Center + (Re(w),Im(w))`

Values are not automatically clamped, allowing Texture Sample addressing modes to determine behavior.

---

# Noise, randomness, and stochastic fields

## 15. Hash / White Noise

**Purpose:** Produce deterministic uncorrelated pseudo-random values from spatial lattice coordinates.

**Inputs:** `Coordinates = Canvas`; `Frequency: Numeric = 256`; `Offset: Vector Numeric = 0`; `Seed: Float = 0`.

**Outputs:** `Value: Scalar Field`; optionally `Vector: Vector Field`.

Compute:

`Cell = floor(Coordinates*Frequency + Offset)`

Hash integer `(Cell.x,Cell.y,Seed)` with a fixed documented 32-bit integer hash. Convert the result to `[0,1)` by multiplying by `1/2^32`.

Vector output uses two separately salted hashes.

The chosen hash function becomes part of project compatibility and should not change between versions.

---

## 16. Value Noise

**Purpose:** Smoothly interpolate random values assigned to a regular lattice.

**Inputs:** Coordinates, Frequency, Offset, Seed.

**Output:** Scalar Field.

Compute lattice position:

`P = Coordinates*Frequency+Offset`

For four surrounding integer corners, obtain Hash values.

Interpolate first in X then Y using quintic fade:

`f(t)=t³(t(6t-15)+10)`

The result is continuous with continuous first derivatives across lattice boundaries.

Output range is approximately `[0,1]`.

---

## 17. Simplex / OpenSimplex Noise

**Purpose:** Generate isotropic gradient noise without square-grid directional artifacts.

**Inputs:** Coordinates, Frequency, Offset, Seed.

**Parameters:** `Classic Simplex | OpenSimplex2S`.

**Output:** Scalar Field nominally normalized to `[-1,1]`.

Classic mode must use the standard 2D simplex skew/unskew algorithm and deterministic gradient hashing. OpenSimplex mode must use one explicitly frozen algorithm/version, preferably OpenSimplex2S 2D.

Seed changes only gradient/hash selection.

Output normalization constants must remain fixed so identical projects reproduce exactly across versions.

---

## 18. Worley / Voronoi Noise

**Purpose:** Generate cellular distance fields from randomly jittered lattice feature points.

**Inputs:** Coordinates, Frequency, Offset, `Jitter: Numeric = 1`, Seed.

**Parameters:** distance `Euclidean | Manhattan | Chebyshev`.

**Outputs:** `F1`, `F2`, `F2-F1`, `Cell ID: Scalar Field`; `Feature Position: Vector Field`.

Each integer lattice cell contains one deterministic feature point:

`Feature = Cell + mix((0.5,0.5),Hash2(Cell,Seed),Jitter)`

Measure the current coordinate to neighboring feature points, retaining nearest and second nearest.

Cell ID is a stable random value based on the winning lattice cell.

Feature Position is returned in the original coordinate space.

---

## 19. Fractal Noise Composer

**Purpose:** Combine multiple frequencies of a base noise algorithm into fBm.

**Inputs:** Coordinates; `Frequency=1`; `Octaves: Float=5`; `Persistence=.5`; `Lacunarity=2`; `Amplitude=1`; Seed.

**Parameters:** base noise `Value | Perlin | Simplex | OpenSimplex`.

**Output:** Scalar Field.

For integer octave count N:

`sum += amplitude_i * noise(Coordinates*frequency_i, Seed+i)`

with:

`frequency_{i+1}=frequency_i*Lacunarity`

`amplitude_{i+1}=amplitude_i*Persistence`

Normalize by the sum of absolute octave amplitudes when `Normalize` is enabled.

Fractional octave values may blend the final octave proportionally.

---

## 20. Ridged Multifractal

**Purpose:** Create sharp branching/ridge structures from octave noise.

**Inputs:** same general inputs as Fractal Noise plus `Ridge Offset: Numeric = 1`; `Gain: Numeric = 2`.

**Output:** Scalar Field.

For each octave, begin with base noise n in `[-1,1]`:

`signal = RidgeOffset - abs(n)`

`signal = signal²`

Optionally modulate the next octave by previous signal:

`weight = clamp(signal*Gain,0,1)`

Accumulate amplitude-weighted signal over octaves.

Frequency and amplitude advance by Lacunarity/Persistence.

Normalize by total amplitude when requested.

---

## 21. Turbulence Noise

**Purpose:** Produce folded fractal noise with sharp valleys.

**Inputs:** same as Fractal Noise.

**Output:** Scalar Field.

For every octave:

`signal = abs(baseNoise(...))`

Accumulate weighted signal.

If base noise is `[-1,1]`, turbulence naturally produces `[0,1]` octave values.

Frequency and amplitude progress through Lacunarity/Persistence exactly as in fBm.

---

## 22. Billow Noise

**Purpose:** Produce rounded cloud-like structures.

**Inputs:** same as Fractal Noise.

**Output:** Scalar Field.

For each octave:

`signal = 2*abs(baseNoise(...))-1`

This transforms noise into rounded positive lobes separated by narrow negative valleys.

Accumulate using standard octave frequency/amplitude progression.

Normalization behavior matches Fractal Noise Composer.

---

## 23. Curl Noise

**Purpose:** Generate a divergence-free 2D vector field from a scalar noise potential.

**Inputs:** Coordinates; Frequency; Seed; `Derivative Step: Float = 1e-3`; optional fractal controls.

**Output:** Vector Field.

Let scalar potential be `ψ(x,y)`.

Estimate or analytically evaluate:

`dx = ∂ψ/∂x`

`dy = ∂ψ/∂y`

Then return:

`Velocity = (dy,-dx)`

This field has zero divergence in the continuous case.

Optional Normalize mode returns unit vectors; otherwise magnitude reflects local gradient strength.

---

## 24. Flow Noise

**Purpose:** Generate smoothly evolving noise without simply translating through a static field.

**Inputs:** Coordinates; `Time: Numeric`; Frequency; `Speed`; Seed.

**Output:** Scalar Field.

Implement using gradient noise where each lattice gradient has a deterministic initial angle from Seed and rotates continuously:

`gradientAngle = hashAngle(cell,Seed)+Time*Speed`

The rotating gradients are then evaluated with the same interpolation rules as the chosen gradient-noise implementation.

The field should evolve continuously and loop exactly if an optional Loop Period is provided by mapping Time to a complete angular revolution.

---

## 25. Gabor Noise

**Purpose:** Generate directional, band-limited stochastic texture using sparse Gabor kernels.

**Inputs:** Coordinates; `Frequency`; `Orientation`; `Bandwidth`; `Impulse Density`; `Anisotropy`; Seed.

**Output:** Scalar Field.

Each nearby lattice cell deterministically contains zero or more impulses. Each impulse contributes:

`exp(-π*a²*r²) * cos(2π*f*dot(direction,d)+phase)`

where phase and small position offsets come from Seed hashes.

Sum contributions from cells whose Gaussian envelope can materially affect the current pixel.

Normalize statistically or by a fixed conservative gain.

This node is computationally expensive at high impulse densities.

---

## 26. Blue Noise

**Purpose:** Provide a spatially decorrelated threshold/sampling field with energy concentrated at high frequencies.

**Inputs:** `Coordinates = Canvas`; `Scale: Numeric = 1`; `Offset`; `Seed`.

**Output:** Scalar Field `[0,1)`.

The implementation should use a fixed canonical precomputed blue-noise tile, preferably at least 128×128. Seed chooses deterministic integer translation, rotation/reflection, and optional channel/permutation of the tile.

Coordinates repeat over the tile according to Scale.

Do not regenerate blue noise differently per GPU; the bundled tile is part of project compatibility.

---

## 27. Noise Derivatives

**Purpose:** Derive gradient information from any scalar field.

**Inputs:** `Source: Scalar Field`; `Step X: Float = one pixel`; `Step Y: Float = one pixel`.

**Outputs:** `dX: Scalar Field`; `dY`; `Magnitude`; `Direction`; `Gradient: Vector Field`.

Using central differences:

`dX = (S(x+h,y)-S(x-h,y))/(2h)`

`dY = (S(x,y+h)-S(x,y-h))/(2h)`

`Magnitude = sqrt(dX²+dY²)`

`Direction = atan2(dY,dX)`

`Gradient=(dX,dY)`

Sampling at boundaries follows configurable Address Mode.

---

# Analytic patterns, implicit geometry, and distance fields

## 28. Gradient Generator

**Purpose:** Generate continuous scalar coordinate fields.

**Inputs:** Coordinates, Center, Angle, Scale, Offset.

**Parameters:** `Linear | Reflected Linear | Radial | Angular | Diamond`; optional Clamp.

**Output:** Numeric.

Linear:

`dot(Coordinates-Center,(cos Angle,sin Angle))*Scale+Offset`

Radial:

`length(Coordinates-Center)*Scale+Offset`

Diamond:

`(|dx|+|dy|)*Scale+Offset`

Angular:

`fract(atan2(dy,dx)/(2π)*Scale+Offset)`

Reflected Linear takes the absolute value before scale/offset.

No coloring is performed.

---

## 29. Wave Generator

**Purpose:** Convert arbitrary Numeric phase into a periodic waveform.

**Inputs:** `Phase`; `Frequency=1`; `Phase Offset=0`; `Amplitude=1`; `Bias=0`; `Duty Cycle=.5`.

**Parameters:** `Sine | Cosine | Triangle | Saw | Reverse Saw | Square`; optional Normalize 0–1.

**Output:** Numeric.

Define:

`z=Phase*Frequency+PhaseOffset`

`q=fract(z)`

Sine is `sin(2πz)`, Triangle is `1-4|q-.5|`, Saw is `2q-1`, Square is ±1 according to Duty Cycle.

Apply optional `0.5*w+0.5`, then:

`Output=w*Amplitude+Bias`.

---

## 30. Checker / Grid

**Purpose:** Generate checker cells, grid lines, and local cell information.

**Inputs:** Coordinates; `Cells: Vector Numeric=(8,8)`; `Offset`; `Line Width: Vector Numeric=.05`.

**Outputs:** `Checker`, `Grid Lines`, `Cell ID: Scalar Field`; `Local UV: Vector Field`.

Compute:

`P=Coordinates*Cells+Offset`

`Cell=floor(P)`

`Local=fract(P)`

Checker is `(Cell.x+Cell.y) mod 2`.

Grid Lines is 1 where Local lies within configured normalized cell-edge width.

Cell ID is a deterministic hash of integer cell coordinates.

---

## 31. Stripe Generator

**Purpose:** Generate oriented periodic stripes directly from coordinates.

**Inputs:** Coordinates; Angle; Frequency; Phase; `Duty Cycle=.5`; `Softness=0`.

**Output:** Scalar Field.

Project coordinates onto direction:

`p=dot(Coordinates,(cos Angle,sin Angle))*Frequency+Phase`

`q=fract(p)`

Hard stripe value is `1` when `q<DutyCycle`, otherwise `0`.

Softness replaces the step boundaries with smoothstep transitions of the specified fraction of one period.

This node is convenience composition of Gradient + Wave.

---

## 32. Ring Generator

**Purpose:** Generate concentric periodic rings.

**Inputs:** Coordinates; Center; Frequency; Phase; Duty Cycle; Softness.

**Output:** Scalar Field.

Compute radius:

`r=length(Coordinates-Center)`

Then:

`q=fract(r*Frequency+Phase)`

Produce square, sine, triangle, or saw profile according to a Waveform parameter.

Square uses Duty Cycle and Softness.

---

## 33. Radial Spokes

**Purpose:** Generate angular repetition around a center.

**Inputs:** Coordinates; Center; `Spokes: Numeric=12`; Rotation; Duty Cycle; Softness.

**Output:** Scalar Field.

Compute turns:

`t=fract(atan2(dy,dx)/(2π)-Rotation/(2π))`

Then:

`q=fract(t*Spokes)`

Generate the selected waveform from q.

For Square mode, q below Duty Cycle is 1.

Spokes may be field-driven but should be clamped or rounded to a positive practical value for discontinuous modes.

---

## 34. Spiral Generator

**Purpose:** Generate spiral distance/phase fields.

**Inputs:** Coordinates; Center; `Turns/Frequency`; `Pitch`; Rotation; Width.

**Parameters:** `Archimedean | Logarithmic | Fermat`.

**Outputs:** `Phase`; `Line Mask: Scalar Field`.

Convert to `(r,θ)`.

Archimedean spiral compares:

`r` against `a+b*θ`.

Logarithmic compares `log(r)` against an affine function of θ.

Fermat compares `sqrt(r)` or equivalent canonical `r=a*sqrt(θ)` relation.

Phase is signed normalized distance from the nearest spiral turn; Line Mask thresholds its absolute value by Width.

---

## 35. Moiré / Interference Generator

**Purpose:** Combine multiple analytic periodic gratings into interference structures.

**Inputs:** Coordinates plus a configurable list of 2–8 layers, each containing Frequency, Angle, Phase, Amplitude, and waveform.

**Parameters:** combine `Add | Multiply | Difference | Product of Sines`.

**Output:** Scalar Field.

Each layer generates:

`Li = waveform(dot(Coordinates,Direction_i)*Frequency_i+Phase_i)`

Combine all active layers according to selected operation.

Optional normalization divides additive output by total absolute amplitudes.

This node is primarily a convenience macro around Gradient/Wave composition.

---

## 36. SDF Primitive

**Purpose:** Generate signed distance to a geometric primitive.

**Inputs:** Coordinates; Center; mode-dependent Size, Radius, Rotation, Corner Radius, Points/Sides.

**Parameters:** `Circle | Box | Rounded Box | Ellipse | Line Segment | Capsule | Triangle | Regular Polygon | Star`.

**Output:** `Distance: Scalar Field`, with **negative inside, zero on boundary, positive outside**.

Distances are expressed in normalized canvas units.

Circle uses `length(p)-radius`. Box and capsule use exact Euclidean signed-distance formulations. Polygon/star modes use exact or numerically equivalent signed-distance formulas.

Rotation is applied around Center before evaluation.

Any mode advertised as SDF should preserve the sign convention even if distance is approximate.

---

## 37. SDF Boolean

**Purpose:** Combine signed-distance fields.

**Inputs:** `A: Numeric`; `B: Numeric`; `Smooth Radius: Numeric=0`.

**Parameters:** `Union | Intersection | Subtract B from A | XOR | Smooth Union | Smooth Intersection | Smooth Subtract`.

**Output:** Numeric.

Exact operations:

* Union: `min(A,B)`
* Intersection: `max(A,B)`
* Subtract: `max(A,-B)`
* XOR: combine regions whose signs differ.

Smooth variants use one fixed documented polynomial smooth-min/max implementation controlled by Smooth Radius.

Zero Smooth Radius must exactly equal the hard operation.

---

## 38. SDF Transform

**Purpose:** Transform an SDF spatially.

**Inputs:** `Source SDF: Scalar Field`; Translation; Rotation; `Uniform Scale: Numeric=1`; Pivot.

**Output:** Scalar Field.

Compute inverse-transformed sampling coordinates, sample Source SDF there, then multiply sampled distance by `abs(UniformScale)`.

Uniform scaling therefore preserves true distance units.

Optional Nonuniform Scale may be provided only if clearly labeled approximate; multiply sampled distance by `min(abs(scaleX),abs(scaleY))`.

Translation and rotation do not change distance magnitude.

---

## 39. SDF Repeat

**Purpose:** Periodically repeat an SDF source.

**Inputs:** `Source SDF`; `Period: Float2`; Offset; optional Mirror.

**Output:** Scalar Field.

For every destination coordinate, map it into the canonical repeated cell using Repeat or Mirror coordinate behavior, then sample Source SDF at that coordinate.

If the repeated source itself occupies one canonical normalized cell, Period defines the spacing between instances.

No additional scale correction is needed unless the coordinate mapping scales the source.

Optional Cell ID output exposes the integer repeat index hash.

---

## 40. SDF Stroke / Contour

**Purpose:** Convert signed distance into fills, outlines, bands, and soft contours.

**Inputs:** `Distance: Numeric`; `Width: Numeric`; `Softness: Numeric=0`; `Offset: Numeric=0`.

**Parameters:** `Fill Inside | Fill Outside | Centered Stroke | Inner Stroke | Outer Stroke | Repeated Contours`.

**Output:** Numeric mask, normally `[0,1]`.

Centered stroke uses distance from the requested contour:

`a=abs(Distance-Offset)`

and maps values below Width/2 to 1, optionally with smoothstep Softness.

Fill Inside maps negative distances to 1.

Repeated Contours applies periodic folding to Distance before stroking.

---

## 41. Superformula / Supershape

**Purpose:** Generate the broad Gielis superformula family of radial shapes.

**Inputs:** Coordinates; Center; `m,n1,n2,n3,a,b`; Scale; Rotation.

**Outputs:** `Distance Approximation`; `Mask`; `Boundary Radius`.

For angle θ:

`r(θ) = ( |cos(mθ/4)/a|^n2 + |sin(mθ/4)/b|^n3 )^(-1/n1)`

Compute current radius rc from Center.

Boundary Radius is scaled r.

Distance Approximation is:

`rc - BoundaryRadius`

Negative values are inside.

This is a radial distance proxy, not guaranteed Euclidean SDF, and should be labeled accordingly.

---

## 42. Harmonic Pattern

**Purpose:** Sum several oriented sinusoidal components.

**Inputs:** Coordinates and a configurable list of harmonics containing Direction/Angle, Frequency, Phase, and Amplitude.

**Output:** Scalar Field.

For each harmonic:

`Hi = Ai*sin(2π*(dot(Coordinates,Direction_i)*Frequency_i+Phase_i))`

Output:

`sum(Hi)`

Optional Normalize divides by `sum(abs(Ai))`.

A convenience Radial mode may replace the projection with radius or angle, but the canonical implementation is oriented planar harmonics.

---

# Tiling, distribution, and spatial arrangement

## 43. Tile Generator

**Purpose:** Produce repeated procedural tile masks with per-tile variation.

**Inputs:** Coordinates; `Tile Count: Float2`; `Tile Size`; `Gap`; global Offset/Rotation; Seed; optional modulation fields for Scale, Rotation, Position, Value.

**Outputs:** `Mask`; `Tile ID`; `Random`; `Local UV`.

Determine integer cell and Local UV from the base lattice. Hash cell coordinates for stable random values. Apply per-cell scale/rotation/offset to Local UV, then evaluate a built-in shape such as Rectangle, Circle, Capsule, Polygon, or supplied SDF-style primitive.

All random variation must be deterministic from Cell+Seed and independent of output resolution.

---

## 44. Tile Sampler

**Purpose:** Repeat an input image inside procedural cells with per-tile variation.

**Inputs:** `Source: Image-Like`; Coordinates; Tile Count; Gap; per-tile Scale/Rotation/Offset; Seed; optional modulation fields.

**Outputs:** sampled image matching Source type; optional Tile ID and Mask.

Determine cell and Local UV exactly as Tile Generator. Apply stable per-cell transformations to Local UV, then sample Source using those local coordinates.

Addressing inside the source tile is separately configurable.

Pixels falling in Gap output Border Value.

Per-cell randomness is derived only from integer Cell and Seed.

---

## 45. Brick / Bond Generator

**Purpose:** Generate masonry-style rectangular tile arrangements.

**Inputs:** Coordinates; Brick Size; Gap; Offset; Seed.

**Parameters:** `Stack | Running Bond 1/2 | Running Bond 1/3 | Flemish-like | Herringbone | Basket Weave`.

**Outputs:** `Brick Mask`; `Mortar Mask`; `Brick ID`; `Local UV`.

Each mode maps Coordinates into a deterministic brick cell and local brick coordinates. Running bonds offset alternating rows by a configured fraction. Herringbone rotates alternating rectangular cells by 90° according to its canonical interlocking lattice.

Gap is measured in normalized brick-cell units or explicit canvas units; choose one convention and expose it consistently.

---

## 46. Lattice Generator

**Purpose:** Generate regular 2D lattice coordinate systems.

**Inputs:** Coordinates; Scale; Rotation; Offset.

**Parameters:** `Square | Rectangular | Triangular | Hexagonal | Rhombic | Oblique`.

**Outputs:** `Cell ID`; `Local Coordinates`; `Distance to Cell Center`; `Distance to Boundary`.

Map world coordinates into the selected lattice basis. Determine the containing or nearest canonical cell. Local Coordinates are expressed relative to that cell.

Cell ID is a deterministic hash of integer lattice indices.

Boundary distance should represent geometric distance to the nearest cell edge where practical.

---

## 47. Truchet Tiles

**Purpose:** Generate Truchet-style connectivity patterns with deterministic tile orientations.

**Inputs:** Coordinates; Tile Count; Seed; Line Width; Randomness.

**Parameters:** motif family `Diagonal | Quarter Circle | Double Arc | Triangle`.

**Outputs:** `Pattern Mask`; `Tile ID`; `Orientation`.

Determine cell and Local UV. Hash each cell to choose one of the motif's allowed rotations/reflections.

Evaluate the chosen motif analytically within Local UV.

Randomness 0 may select a fixed orientation; 1 uses fully hashed orientation.

Neighbor continuity is not guaranteed unless a constrained motif mode explicitly provides it.

---

## 48. Wang Tile Generator

**Purpose:** Create non-obviously repeating tilings whose neighboring edge labels always match.

**Inputs:** Coordinates; `Tile Atlas: Image-Like`; `Atlas Columns/Rows`; Tile Count; Seed; `Edge Colors: integer parameter`.

**Outputs:** sampled tiled image; Tile ID.

Assign deterministic edge labels to lattice boundaries using hashes of each shared boundary coordinate. Because each physical boundary is hashed once, adjacent cells receive identical shared labels.

A tile's required `(north,east,south,west)` labels index into the atlas according to a documented enumeration.

Sample the selected atlas tile with cell Local UV.

Atlas must contain every possible edge-label combination required by the selected Edge Colors.

---

## 49. Quasiperiodic Tiling

**Purpose:** Generate nonperiodic mathematical tilings such as Penrose patterns.

**Inputs:** Coordinates; Scale; Rotation; Offset; optional Seed/Phase; `Subdivision Depth`.

**Parameters:** `Penrose P2 | Penrose P3` initially.

**Outputs:** `Tile Mask`; `Edge Mask`; `Tile Type`; `Tile ID`.

Use one frozen substitution construction for each mode. Generate a finite patch covering the visible canvas plus margin, recursively subdivide to requested depth, transform into canvas space, then rasterize.

Tile ID must remain deterministic for identical transform/depth settings.

This is substantially more complex than ordinary shader-local generators and may require CPU preprocessing or GPU buffers.

---

## 50. Density Scatter

**Purpose:** Produce random points according to a spatial density field.

**Inputs:** `Density: Numeric`; `Count: Float`; Seed; optional Mask.

**Output:** `Point Set`.

Interpret Density as nonnegative relative sampling weight over the canvas. Generate Count deterministic candidate points and accept/distribute them according to normalized density, using rejection sampling, alias sampling, or an equivalent method whose statistical distribution matches Density.

Each point receives stable ID and Random attributes.

Changes to Count should ideally preserve earlier point IDs/positions by using an index-based deterministic random sequence.

Requires Point Set support.

---

## 51. Poisson-Disc Scatter

**Purpose:** Generate points with a minimum separation radius.

**Inputs:** `Radius: Numeric`; Density/Mask optional; `Maximum Points`; Seed.

**Output:** Point Set.

Implement variable-radius Bridson-style Poisson-disc sampling or an equivalent deterministic algorithm.

For two accepted points i,j, enforce separation according to a documented radius rule, preferably:

`distance(i,j) >= max(radius_i,radius_j)`

Radius is sampled from the field at each candidate position.

Seed fixes candidate generation order.

This is globally coupled and substantially harder to implement as a pure fragment/compute shader.

---

## 52. Packing / Relaxation

**Purpose:** Iteratively arrange circles or points to reduce overlap while obeying masks/size fields.

**Inputs:** initial `Point Set` or Count+Seed; `Radius: Numeric`; attraction/repulsion fields; boundary Mask.

**Outputs:** relaxed Point Set; optional rasterized occupancy Scalar Field.

Each iteration computes overlaps between nearby particles and applies symmetric separation impulses. Optional attraction pulls points toward target positions or high-density regions.

Boundary handling projects points back into allowed Mask regions.

Expose Iterations, Relaxation Strength, Damping, and Radius Scale.

Results must be deterministic for a fixed seed and iteration count.

This requires spatial acceleration and is one of the more computationally involved proposed nodes.

---

# Vector fields and differential operators

## 53. Image Gradient

**Purpose:** Compute the spatial gradient of a scalar field.

**Input:** `Source: Scalar Field`; `Step/Scale: Numeric`.

**Outputs:** `Gradient: Vector Field`; `dX`; `dY`; `Magnitude`; `Angle`.

Use central differences by default:

`dx=(right-left)/(2h)`

`dy=(down-up)/(2h)`

Gradient is `(dx,dy)`.

Magnitude is Euclidean length.

Angle is `atan2(dy,dx)`.

Optional operators `Sobel | Scharr | Central Difference` may change derivative kernels while preserving output semantics.

---

## 54. Vector Compose / Split

**Purpose:** Convert between scalar components and vector values.

**Compose Inputs:** `X: Numeric`; `Y: Numeric`.

**Compose Output:** Float2 if both inputs Float, otherwise Vector Field.

`Output=(X,Y)`.

**Split Input:** `Vector: Float2 | Vector Field`.

**Split Outputs:** X and Y as Float/Scalar Field according to input type.

No normalization, clamping, or coordinate semantics are applied.

---

## 55. Vector Math

**Purpose:** Perform common vector operations.

**Inputs:** `A: Vector Numeric`; optional `B: Vector Numeric`; optional scalar inputs.

**Parameters:** `Add | Subtract | Multiply Components | Divide Components | Scale | Length | Normalize | Dot | Distance | Reflect | Perpendicular CW | Perpendicular CCW | Angle | Min | Max`.

**Outputs:** Vector Numeric or Numeric depending on operation.

Normalize returns zero for vectors below epsilon length.

Reflect computes:

`A - 2*dot(A,N)*N`

after normalizing B as N.

Angle returns `atan2(A.y,A.x)`.

Promotion follows standard Float2/Vector Field rules.

---

## 56. Divergence

**Purpose:** Measure local expansion/compression of a vector field.

**Input:** `Vector: Vector Field`; Step.

**Output:** Scalar Field.

For vector `V=(u,v)`:

`div(V)=∂u/∂x + ∂v/∂y`

Derivatives use central differences or selected derivative kernel.

Positive values indicate local outward flow; negative values indicate convergence.

Coordinate spacing should use normalized canvas units so output is resolution-independent when possible.

---

## 57. Curl / Vorticity

**Purpose:** Measure local rotational tendency of a 2D vector field.

**Input:** Vector Field; Step.

**Output:** Scalar Field.

For `V=(u,v)`:

`curl_z = ∂v/∂x - ∂u/∂y`

Positive/negative sign follows the shared +Y-down coordinate convention and must remain consistent across the application.

Optional Absolute output may expose `abs(curl)` but should not replace signed curl.

---

## 58. Vector Field Transform

**Purpose:** Modify vector values without moving their pixel locations.

**Inputs:** `Vector: Vector Numeric`; `Rotation: Numeric`; `Magnitude Scale: Numeric`; `Max Magnitude: Numeric`.

**Parameters:** `Rotate | Normalize | Clamp Magnitude | Invert | Set Magnitude`.

**Output:** same promoted Vector Numeric type.

Rotation applies the standard screen-space rotation matrix.

Clamp Magnitude preserves direction but limits vector length.

Set Magnitude normalizes then scales to supplied magnitude, returning zero for zero input.

This node transforms vectors themselves, unlike 2D Transform, which transforms coordinates.

---

## 59. Advect Image

**Purpose:** Transport an image/field through a velocity field.

**Inputs:** `Source: Image-Like`; `Velocity: Vector Field`; `Time Step: Numeric`; `Iterations: Float=1`.

**Parameters:** sampling method; Address Mode.

**Output:** same type as Source.

For one semi-Lagrangian step:

`sourcePos = currentPos - Velocity(currentPos)*TimeStep`

`Output(currentPos)=Source(sourcePos)`

For multiple iterations, repeatedly advect the previous iteration result with timestep divided by iteration count unless explicitly configured otherwise.

Velocity units are normalized canvas units per unit time.

---

## 60. Flow Map Distort

**Purpose:** Perform a single displacement sample from a vector field.

**Inputs:** `Source: Image-Like`; `Flow: Vector Field`; `Strength: Numeric`.

**Output:** same type as Source.

At each output coordinate:

`samplePos = Coordinates + Flow*Strength`

`Output = sample(Source,samplePos)`

Optional Direction parameter may switch between `+Flow` and `-Flow`.

Unlike Advect, no physical interpretation or iterative integration is implied.

---

## 61. Streamline Integration

**Purpose:** Trace trajectories through a vector field and rasterize them.

**Inputs:** `Vector: Vector Field`; Seeds as Point Set or Density/Count; `Step Size`; `Steps`; optional Mask.

**Outputs:** `Lines/Accumulation: Scalar Field`; optionally Endpoint Point Set.

For each seed, numerically integrate:

`p_{n+1}=p_n+h*V(p_n)`

using selectable `Euler | RK2 | RK4`.

Forward, backward, or bidirectional integration is supported.

Rasterize each segment into the accumulation field with configurable Width and additive/max blending.

This can be expensive for many seeds and long trajectories.

---

## 62. Vector Field Visualizer

**Purpose:** Convert vector data into human-readable or artistic raster representations.

**Input:** Vector Field.

**Parameters:** `Hue/Angle | Arrows | Streamlets | LIC`; Scale, Grid Density, Line Length.

**Output:** Color Image or Scalar Field depending on mode.

Hue mode maps angle around a color wheel and vector magnitude to brightness/saturation.

Arrow mode samples a regular grid and rasterizes oriented arrows.

LIC performs line-integral convolution of white noise along local streamlines, producing a Scalar Field.

LIC is significantly more expensive than the other modes.

---

# Filtering, morphology, and structure extraction

## 63. Gaussian Blur HQ

**Purpose:** Perform scalable Gaussian convolution.

**Inputs:** `Source: Image-Like`; `Sigma X: Numeric`; `Sigma Y: Numeric`.

**Parameters:** Address Mode; optional Rotation.

**Output:** same type as Source.

Kernel weight for offset `(x,y)` is proportional to:

`exp(-0.5*((x/σx)²+(y/σy)²))`

Normalize weights to sum to 1.

Kernel radius should cover at least `ceil(3σ)` in each direction.

For axis-aligned blur, a separable implementation is mathematically equivalent and strongly preferred, though not required by node semantics.

---

## 64. Directional Blur

**Purpose:** Blur along a straight direction.

**Inputs:** Source; `Angle: Numeric`; `Distance: Numeric`; `Samples: Float`.

**Output:** same type as Source.

Sample evenly along a centered line:

`p_i = p + direction * t_i * Distance`

where `t_i` spans `[-0.5,0.5]`.

Average all valid samples using uniform weights, or optional Gaussian weighting.

Distance is normalized canvas units.

Angle may be a field, allowing direction to vary per pixel.

---

## 65. Radial / Zoom Blur

**Purpose:** Blur along radial or rotational paths around a center.

**Inputs:** Source; Center; Amount; Samples.

**Parameters:** `Zoom | Spin`.

**Output:** same type as Source.

Zoom mode samples points along the line between the pixel and Center.

Spin mode samples coordinates rotated around Center over an angular interval of Amount radians.

Average the samples with normalized weights.

Amount 0 returns Source exactly.

---

## 66. Vector Motion Blur

**Purpose:** Blur each pixel along an arbitrary vector field.

**Inputs:** Source; `Motion: Vector Field`; Strength; Samples.

**Output:** same type as Source.

For destination position p and vector v:

sample positions cover:

`p + v*Strength*t`

for t uniformly spanning `[-0.5,0.5]`.

Average samples.

Optional One-Sided mode instead spans `[0,1]`.

Motion vectors use normalized coordinate units.

---

## 67. Slope Blur

**Purpose:** Distort/aggregate Source according to the local gradient of another field.

**Inputs:** `Source: Image-Like`; `Slope: Scalar Field`; `Strength: Numeric`; `Samples: Float`.

**Parameters:** `Blur | Minimum | Maximum`.

**Output:** same type as Source.

Compute gradient g of Slope. Starting at the destination coordinate, repeatedly step:

`p = p - normalize_or_raw(g(p))*Strength/Samples`

Sample Source along the path.

Blur averages samples. Minimum/Maximum reduce them component-wise.

Expose whether gradient magnitude affects displacement or only direction.

---

## 68. Median Filter

**Purpose:** Remove isolated extrema while preserving hard boundaries.

**Inputs:** Source; `Radius: Float`.

**Output:** same type as Source.

Collect all samples in the selected square/disk neighborhood and return the median.

For Scalar Field use scalar median.

For Vector Field and Color Image, compute the median independently per channel unless a separate Vector Median mode is explicitly added.

Radius is integer pixels or normalized sampling radius according to a clearly selected unit mode.

---

## 69. Bilateral Filter

**Purpose:** Smooth while preserving edges based on value similarity.

**Inputs:** `Source: Image-Like`; optional `Guide: Image-Like = Source`; `Spatial Sigma`; `Range Sigma`; Radius.

**Output:** same type as Source.

For neighbor q:

`weight = spatialGaussian(distance(p,q)) * rangeGaussian(distance(Guide(p),Guide(q)))`

Output is weighted average of Source neighbors divided by total weight.

For multi-channel Guide, range distance is Euclidean channel distance over meaningful channels.

---

## 70. Kuwahara Filter

**Purpose:** Edge-preserving region smoothing with a painterly character.

**Inputs:** Source; Radius.

**Output:** same type as Source.

Partition the neighborhood around each pixel into four overlapping quadrants. Compute mean and variance for each quadrant.

Choose the quadrant having the smallest scalar variance.

For Color Image, define variance as the sum of channel variances or luminance variance according to a parameter.

Output the chosen quadrant's mean.

Optional Generalized Kuwahara may add more angular sectors but should remain a separate mode.

---

## 71. Distance Transform

**Purpose:** Measure distance from each pixel to the nearest selected pixel.

**Inputs:** `Mask: Scalar Field`; `Threshold: Numeric=.5`.

**Parameters:** `Distance To Foreground | Distance To Background`; `Euclidean | Manhattan | Chebyshev`.

**Output:** Scalar Field.

Binary foreground is `Mask >= Threshold`.

For every pixel return geometric distance to the nearest target-class pixel.

Distances are expressed in normalized canvas units by default, with optional Pixels unit mode.

Implementation may use exact EDT or an approximation such as jump flooding, but if approximate it should be documented.

---

## 72. Signed Distance Transform

**Purpose:** Convert an arbitrary binary mask into a signed-distance-like field.

**Inputs:** Mask; Threshold.

**Output:** Scalar Field with negative inside.

Compute:

* `din` = distance from foreground pixels to nearest background.
* `dout` = distance from background pixels to nearest foreground.

Return:

* inside: `-din`
* outside: `+dout`

Zero lies at the discrete class boundary.

Optional half-pixel correction may center zero between foreground/background pixels, but the selected convention must be stable.

---

## 73. Bevel from Distance

**Purpose:** Turn signed distance into height, shading, or normal information near a boundary.

**Inputs:** `Distance: Scalar Field`; `Width: Numeric`; `Height: Numeric`; `Profile: Float`.

**Outputs:** `Height Field`; `Normal: Vector Field`; optional `Shade: Scalar Field`.

Convert distance within Width into normalized bevel parameter t and evaluate a selected profile such as Linear, Smoothstep, Round.

Height is the resulting profile times Height.

Compute spatial gradient of Height to derive XY normal/slope.

Outside the bevel region, height is constant.

---

## 74. Morphology Suite

**Purpose:** Provide common binary/grayscale morphological operations.

**Input:** `Source: Scalar Field`; Radius/Kernel.

**Parameters:** `Erode | Dilate | Open | Close | Morphological Gradient | Top Hat | Black Hat | Outline`.

**Output:** Scalar Field.

Erode = neighborhood minimum.

Dilate = neighborhood maximum.

Open = Dilate(Erode(Source)).

Close = Erode(Dilate(Source)).

Gradient = Dilate−Erode.

Top Hat = Source−Open.

Black Hat = Close−Source.

Outline may use absolute morphological gradient or an inside/outside option.

Kernel shape may be Disk, Square, Diamond, or custom.

---

# Color and tonal mapping

## 75. Color Space Convert

**Purpose:** Explicitly reinterpret/convert color channels between color spaces.

**Input:** `Color: Color Image`.

**Parameters:** source/destination `RGB | Linear RGB | sRGB | HSV | HSL | XYZ | Lab | LCh | OKLab | OKLCh`.

**Output:** Color Image.

RGB channels of the output hold the selected destination-space components; alpha passes through unchanged.

Conversions involving XYZ/Lab require one fixed reference white, preferably D65, documented globally.

sRGB transfer functions must use the standard piecewise transform.

Hue-like components should be normalized to `[0,1)` for graph friendliness unless an explicit radians/degrees mode is selected.

---

## 76. HSV/HSL Adjust

**Purpose:** Adjust perceptual-ish hue/saturation/value or lightness controls.

**Input:** Color Image.

**Inputs:** `Hue Shift: Numeric`; `Saturation Scale: Numeric=1`; `Value/Lightness Scale: Numeric=1`; optional offsets.

**Parameters:** `HSV | HSL`.

**Output:** Color Image.

Convert RGB to selected space, apply:

`Hue = fract(Hue + HueShift)`

`Saturation *= SaturationScale`

`ValueOrLightness *= Scale`

then convert back.

Optional Clamp controls whether S/V/L are clamped to `[0,1]`.

Alpha is unchanged.

---

## 77. Levels

**Purpose:** Remap input range and gamma.

**Input:** `Source: Numeric | Image-Like`.

**Inputs:** `Input Black`; `Input White`; `Gamma`; `Output Black`; `Output White`.

**Output:** same semantic type as Source.

Per component:

`t = (x-InputBlack)/(InputWhite-InputBlack)`

optionally clamp t.

`g = t^(1/Gamma)`

`Output = mix(OutputBlack,OutputWhite,g)`

Gamma and denominator use epsilon safeguards.

Color Image may expose RGB-together, Per-Channel, or Luminance modes.

---

## 78. Curves

**Purpose:** Apply an editable 1D transfer function.

**Input:** Numeric or Image-Like.

**Parameter:** editable control-point curve from X to Y; interpolation `Linear | Smooth | Monotone Cubic`.

**Output:** same type as Source.

For Numeric/Scalar Field, evaluate the curve directly.

For Vector Field, evaluate each component.

For Color Image, support `RGB Combined`, separate `R/G/B`, and `Luminance` modes.

Values outside curve-domain bounds use configurable Clamp or linear extrapolation.

Curve lookup should be deterministic, typically via a generated 1D LUT.

---

## 79. Posterize / Quantize

**Purpose:** Reduce continuous values to discrete levels.

**Inputs:** `Source: Numeric | Image-Like`; `Levels: Numeric`.

**Parameters:** `Nearest | Floor | Ceil`; optional Dither Amount.

**Output:** same type as Source.

For normalized x and integer levels N≥2:

Floor:

`floor(x*N)/(N-1)` with endpoint handling.

Nearest:

`round(x*(N-1))/(N-1)`.

Values may optionally be clamped to `[0,1]` before quantization.

Color Images quantize channels independently unless Luminance mode is selected.

---

## 80. Multi-stop Color Ramp

**Purpose:** Map a scalar into an arbitrary editable color gradient.

**Input:** `Value: Numeric`.

**Parameter:** ordered list of `(position,color,interpolation)` stops.

**Parameters:** Address `Clamp | Repeat | Mirror`; interpolation `Constant | Linear | Smooth | Cubic`.

**Output:** Color Image if Value is a Scalar Field; a constant Color value if scalar-color values are supported internally.

Find the two surrounding stops and interpolate according to the lower segment's interpolation mode.

Stop positions need not be evenly spaced but must be sorted deterministically.

Color interpolation may optionally occur in RGB, HSV, OKLab, or OKLCh.

---

## 81. Dither

**Purpose:** Convert smooth values/colors into spatially distributed quantization error.

**Input:** Numeric, Scalar Field, or Color Image.

**Inputs:** Levels; Amount; Seed.

**Parameters:** `Bayer 2/4/8/16 | Blue Noise | Random | Floyd-Steinberg`.

**Output:** same image/scalar-field type.

Ordered/blue/random modes add a threshold perturbation before quantization.

Bayer matrices are fixed canonical matrices.

Blue Noise uses the Blue Noise node's canonical tile.

Floyd-Steinberg propagates quantization error in raster order with canonical weights `7/16,3/16,5/16,1/16`; this introduces sequential dependencies and is substantially less GPU-friendly.

---

## 82. Channel Split / Combine / Swizzle

**Purpose:** Route arbitrary image channels.

**Split Input:** Color Image or Vector Field.

**Split Outputs:** Scalar Fields for each component.

Color gives R,G,B,A. Vector gives X,Y.

**Combine Inputs:** R/G/B/A as Numeric.

**Combine Output:** Color Image.

Numeric constants broadcast over fields.

**Swizzle Input:** Image-Like.

**Parameters:** mapping for each output channel from existing channels, 0, or 1.

**Output:** chosen target type, generally Color Image or Vector Field.

No color-space conversion is implied.

---

# Time, feedback, and dynamic systems

## 83. Feedback

**Purpose:** Create an explicit one-frame delayed cycle in an otherwise forward-only graph.

**Inputs:** `Update: Image-Like`; `Initial: Image-Like`; `Reset: Numeric=0`.

**Output:** same type as Update.

During frame N evaluation, Output returns stored state from frame N−1. After the graph finishes evaluating frame N, Update is copied into internal state for frame N+1.

If Reset is nonzero, Output returns Initial and stored state is reset to Initial before accepting the next update.

This node must be treated as a special legal feedback boundary rather than an ordinary graph cycle.

---

## 84. Frame Delay

**Purpose:** Access values from several frames in the past.

**Input:** Source: Image-Like; `Frames: Float=1`; Initial value.

**Output:** same type as Source.

Maintain a ring buffer of the previous N materialized Source frames.

At frame k return Source from `k-N`.

Before sufficient history exists, return Initial or oldest available frame according to parameter.

Changing N may clear history or preserve overlapping slots; behavior must be explicit.

N is an integer parameter rather than a per-pixel field.

---

## 85. Temporal Accumulate

**Purpose:** Combine the current input with persistent history.

**Input:** Source; `Decay/Alpha: Numeric`; Reset.

**Parameters:** `Add | Average | Exponential Average | Maximum | Minimum`.

**Output:** same type as Source.

Examples:

Add:

`state = state + Source`

EMA:

`state = mix(Source,state,Decay)`

Max:

`state=max(state,Source)`

Running Average tracks sample count and calculates true arithmetic mean.

Reset initializes state to zero or supplied Initial image.

---

## 86. Trails / Decay

**Purpose:** Produce simple persistent motion trails.

**Input:** Source; `Decay: Numeric=.95`.

**Parameters:** `Maximum | Additive | Alpha`.

**Output:** same type as Source.

Maximum:

`state = max(Source,state*Decay)`

Additive:

`state = Source + state*Decay`

Alpha:

`state = mix(Source,state,Decay)` or explicitly selected foreground/background convention.

State updates once per rendered frame and resets on graph reset or explicit Reset input.

---

## 87. Optical Flow

**Purpose:** Estimate apparent 2D pixel motion between consecutive frames.

**Input:** `Current: Scalar Field | Color Image`; optional Previous; parameters Window Radius, Pyramid Levels, Iterations, Regularization.

**Outputs:** `Flow: Vector Field`; `Confidence: Scalar Field`.

If Previous is disconnected, retain the prior Current frame internally.

Implement a fixed dense pyramidal Lucas-Kanade or equivalent documented algorithm. Flow represents normalized UV displacement **from previous frame to current frame**, measured per frame.

Confidence represents local solvability/reliability normalized to `[0,1]`.

This is computationally substantial.

---

## 88. Generic Cellular Automata

**Purpose:** Simulate discrete grid-based cellular rules.

**Inputs:** Seed/Initial State Scalar Field; Reset; optional external field.

**Parameters:** Neighborhood `Moore | Von Neumann`; Boundary; Steps Per Frame; rule.

**Outputs:** current `State: Scalar Field`.

Binary Life-like mode expresses rule as Birth set B and Survival set S. Count live neighbors, then update each cell synchronously.

Multi-state totalistic mode supports integer states `0..N-1` and a rule table indexed by current state and neighborhood total.

Cell values represent exact discrete states, not interpolated continuous values.

---

## 89. Continuous Cellular Automata / Lenia

**Purpose:** Simulate continuous convolution-and-growth automata.

**Inputs:** Initial State; Reset; optional growth/kernel modulation fields.

**Parameters:** Kernel Radius, ring profile parameters, Growth Center μ, Growth Width σ, Time Step, Iterations.

**Output:** State Scalar Field, normally clamped `[0,1]`.

For state A:

`U = convolution(A,Kernel)`

Growth:

`G(U)=2*exp(-((U-μ)/σ)²/2)-1`

Update:

`A_next = clamp(A + dt*G(U),0,1)`

Kernel is normalized to unit sum.

Alternative growth functions may be modes but must expose their formulas.

---

## 90. 2D Fluid Simulation

**Purpose:** Simulate incompressible 2D velocity plus transported dye.

**Inputs:** Initial Dye; Initial Velocity; injected Dye; Force Vector Field; Reset.

**Parameters:** Time Step, Viscosity, Diffusion, Pressure Iterations, Vorticity Confinement, Simulation Iterations.

**Outputs:** `Dye: Color Image`; `Velocity: Vector Field`; `Pressure`; `Divergence`.

Use a stable-fluids style pipeline each iteration:

1. add forces,
2. advect velocity,
3. diffuse velocity if enabled,
4. compute divergence,
5. solve pressure Poisson equation,
6. subtract pressure gradient,
7. advect dye,
8. add injected dye.

Boundary conditions must be explicit.

This is architecturally and computationally substantial.

---

## 91. Particle Field / GPU Particles

**Purpose:** Create and simulate persistent particles driven by image/vector fields.

**Inputs:** Emitter Mask/Density; Initial Velocity Vector Field; Force Vector Field; Color Image optional; Radius field; Reset.

**Parameters:** Spawn Rate, Lifetime, Drag, Max Particles, Seed, Time Step.

**Outputs:** `Particles: Particle Buffer`; `Rasterized Image`; optional Density Scalar Field.

Each particle contains stable ID, position, velocity, age, lifetime and attributes.

Per step:

`v += force(position)*dt`

`v *= drag`

`p += v*dt`

Particles expire when age≥lifetime or according to boundary policy.

Emission is deterministic from Seed and frame/spawn index.

Requires a new non-image data type.

---

## 92. Physarum / Slime Mold Simulation

**Purpose:** Simulate trail-following agents that create branching transport networks.

**Inputs:** Initial Agent Density/Mask; optional attractant/repellent fields; Reset.

**Parameters:** Agent Count, Move Speed, Sensor Distance, Sensor Angle, Turn Speed, Deposit Amount, Trail Diffusion, Trail Decay, Seed.

**Outputs:** `Trail: Scalar Field`; optional `Agents: Particle Buffer`.

Each agent samples trail intensity at forward, left, and right sensor positions, turns toward the strongest sample according to canonical rules, moves forward, and deposits trail.

After all agents move, blur/diffuse Trail and multiply by Decay.

This is a hybrid particle/image simulation and is comparatively complex.

---

# Analysis, masks, utility, and graph-enabling nodes

## 93. Edge / Derivative Suite

**Purpose:** Detect spatial transitions using standard derivative operators.

**Input:** `Source: Scalar Field`; Scale.

**Parameters:** `Sobel | Scharr | Prewitt | Laplacian | Laplacian of Gaussian`.

**Outputs:** `X`; `Y`; `Magnitude`; `Angle`; `Response`.

For directional operators, convolve with canonical X/Y kernels.

Magnitude:

`sqrt(X²+Y²)`

Angle:

`atan2(Y,X)`

Response is operator-specific; for Laplacian it is the signed scalar Laplacian.

LoG applies Gaussian smoothing followed by Laplacian or an equivalent combined kernel.

---

## 94. Local Statistics

**Purpose:** Measure neighborhood statistics around every pixel.

**Input:** `Source: Scalar Field`; Radius.

**Parameters:** neighborhood `Square | Disk`; statistic selection.

**Outputs:** `Mean`; `Minimum`; `Maximum`; `Variance`; `Standard Deviation`; `Range`.

For neighborhood samples xi:

`Mean = sum(xi)/N`

`Variance = sum((xi-Mean)²)/N`

`StdDev=sqrt(Variance)`

`Range=Max-Min`

All outputs are Scalar Fields.

A mode selector may expose one output at a time for performance, but semantics should remain identical.

---

## 95. Histogram / Normalize

**Purpose:** Compute global value distribution and use it for normalization/equalization.

**Input:** `Source: Scalar Field`; optional Mask.

**Parameters:** Bin Count; mode `Histogram | Normalize MinMax | Percentile Normalize | Equalize`.

**Outputs:** `Histogram` auxiliary output and/or normalized Scalar Field; global Min/Max/Mean Floats where useful.

MinMax:

`(x-globalMin)/(globalMax-globalMin)`

Percentile mode substitutes selected low/high percentiles.

Equalize maps values through the cumulative normalized histogram.

This node requires global reductions and therefore differs substantially from ordinary local shaders.

---

## 96. Skeletonize / Medial Axis

**Purpose:** Reduce binary shapes to centerline structures.

**Input:** Mask; Threshold.

**Parameters:** `Zhang-Suen Skeleton | Medial Axis`.

**Outputs:** Skeleton Scalar Field; optional Radius Scalar Field.

Zhang-Suen mode repeatedly performs the canonical two-pass topology-preserving thinning rules until stable or Max Iterations.

Medial Axis mode computes Signed/Unsigned Distance Transform and selects ridge points equidistant from multiple boundary locations.

Radius output gives distance to the nearest boundary along the skeleton.

This is iterative/global and relatively difficult to GPU-accelerate robustly.

---

## 97. Connected Components

**Purpose:** Identify distinct connected foreground regions.

**Input:** Mask; Threshold.

**Parameters:** connectivity `4 | 8`.

**Outputs:** `Label Field: Scalar Field`; `Random Per Component`; `Area Per Component`; `Component Count: Float`.

Every foreground connected region receives one integer label. Background is label 0.

Random Per Component hashes that integer label into `[0,1)`.

Area Per Component stores the number of pixels, or optionally normalized canvas area, at every pixel belonging to that component.

Component Count is a graph-global scalar.

Implementation may use label propagation or union-find but final labels must be deterministic.

---

## 98. Flood Fill / Region Grow

**Purpose:** Select the connected region reachable from one or more seeds.

**Inputs:** `Passable Mask: Scalar Field`; `Seed Mask: Scalar Field` or `Seed Position: Float2`; Threshold.

**Parameters:** connectivity `4 | 8`; Max Iterations.

**Output:** Region Scalar Field.

Initialize Region from valid seed pixels. Iteratively add passable neighboring pixels connected to the current region until stable or Max Iterations.

Output 1 for reached pixels and 0 elsewhere.

Optional Distance output records iteration/geodesic step count from the nearest seed.

This is an iterative global propagation operation.

---

## 99. Resize / Resample / Crop

**Purpose:** Explicitly change image resolution or select a rectangular source region.

**Input:** `Source: Image-Like`.

**Inputs/Parameters:** Target Width/Height integers; Crop Min/Max Float2; Fit mode `Stretch | Fit | Fill | Crop`; Sampling `Nearest | Linear | Cubic | Lanczos`.

**Output:** same semantic image type at the requested resolution.

Crop coordinates are normalized source coordinates.

Fit preserves aspect ratio and letterboxes; Fill preserves aspect ratio and crops excess; Stretch independently scales axes.

Because this changes resolution, downstream nodes must evaluate using the output texture's dimensions rather than assuming global canvas resolution.

---

## 100. Custom GLSL / Expression

**Purpose:** Provide an explicit escape hatch for operations not represented by built-in nodes.

**Parameters:** mode `Scalar Expression | Field Expression | Pixel Shader`; editable source; dynamically declared typed inputs.

**Inputs:** user-defined `Float`, `Float2`, `Scalar Field`, `Vector Field`, and `Color Image` sockets.

**Outputs:** one or more explicitly declared supported graph types.

Field/Pixel mode executes once per output pixel and receives normalized canvas coordinates, pixel coordinates, resolution, and declared inputs. Image inputs may be either current-pixel values or explicitly sampled through provided sampling helpers.

Scalar mode executes once per frame and cannot sample images unless an explicit reduction/sampling API is provided.

Compilation failures must preserve graph validity, report source location/error text, and produce a documented fallback output rather than corrupting GPU state.

Generated shaders should execute in a restricted environment: no arbitrary filesystem, host memory, or unsupported GPU side effects.

A number of these naturally collapse into larger nodes: **Gradient/Wave**, the fractal-noise variants, **SDF operations**, morphology, color adjustments, and temporal accumulation are particularly obvious candidates. The specifications above intentionally keep their behaviors separable so combining them later does not create ambiguity.
