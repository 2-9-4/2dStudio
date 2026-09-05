# Reaction Studio

Reaction Studio is a Linux-first, real-time 2D generative-art node editor. It combines GPU Perlin noise, Gray–Scott reaction diffusion, editable fused simulation subgraphs, polymorphic scalar/image math, color mapping, custom image convolution, and a separate live preview in a forward-only workflow.

Image-valued Math nodes are compiled into operation-specific compute shaders. Consecutive
linear Math nodes are fused into one dispatch, while scalar-only Math continues to run on
the CPU. Branches and joins remain materialization boundaries. The normal node canvas marks
generated regions and reports their shared GPU time.

## Build

Requirements are CMake 3.24+, a C++20 compiler, GLFW 3.3, OpenGL 4.3, Python (used by GLAD generation), and FFmpeg with libx264 for video recording. Remaining dependencies are pinned and downloaded by CMake.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/reaction_studio
```

Pass a project path to open it immediately at launch:

```sh
./build/reaction_studio examples/benchmark.reaction.json
```

The app opens an editor and a shared-context preview window. Right-click the canvas to add nodes. Drag between pins to link them; select links or nodes and press Delete to remove them. Add **Input / Image** and choose a PNG to use an external image in the graph. The default project is immediately animated.

Projects use `*.reaction.json`. They store project settings, graph topology, parameters, and editor positions. Simulation buffers and elapsed time deliberately restart when a project loads.

The add-node menu contains both **Reaction Diffusion (Monolithic)** and **Reaction Diffusion (Discrete)**. Adding the discrete version creates an editable project-local subgraph with Image, Chemical A, and Chemical B outputs; older projects that reference the original built-in template are converted when you enter them. Select the node and press **Tab** to open its full-size graph, or press **Tab** again to return. Inside, use the normal editor interactions: right-click to add Float, Math, Threshold, Select, Canvas Coordinates, Laplacian, or Simulation Channel Split nodes, drag between named pins to add or replace links, select nodes or links and press Delete, and pan or zoom the canvas normally.

A subgraph is stored exactly like the root topology: a `GraphBody` containing normal `NodeRecord` and `LinkRecord` values with stable IDs, parameters, positions, and socket keys. Float, Math, Threshold, Select, Canvas Coordinates, and Laplacian are the same canonical registered node types available in the root graph, including the same controls and pin names. Only subgraph interface, state endpoints, and Simulation Channel Split nodes are scoped to a simulation body because they expose its private inputs, outputs, and Vec2 feedback state. The simulation backend lowers endpoint-reachable work through the same node contract as root shader regions and materializes only the first and connected exported outputs. Presentation-only or unreachable edits preserve live state; edits that change either executable shader rebuild and reset every instance sharing the project subgraph.

Use **File → Export Current Frame as PNG…** to write the graph output at its configured resolution. The export is an 8-bit RGB PNG and does not include the editor UI.

Use **Record** in the toolbar or **File → Start Video Recording…** to capture the graph output as an H.264 MP4 at the project resolution and target FPS. Recording uses a fixed frame timestep, and Perlin animation advances by evaluated frame rather than wall time, so slower encoding does not introduce animation jumps. Pausing also pauses capture; use **Stop Recording** to finalize the file. Video dimensions must be even. The preview window remains constrained to the project's aspect ratio while it is resized.

Use **Window → Shader Inspector** to inspect generated shader regions. Each node contribution
is commented and color-coded, compiler diagnostics remain attached to the failed source,
and **Fuse lowered shader chains** can switch between fused and solo generated regions. Interior Math values
do not normally allocate textures; use **Preview Intermediate** to temporarily materialize one.
Inspector and execution-debug settings are session-only. Lowering/type errors and GLSL compiler
errors are attached to the affected region; native evaluation remains available only for nodes
or parameter modes that explicitly opt out of lowering.

## Current boundaries

- Linux and OpenGL 4.3 compute are the supported first target.
- Graphs are acyclic; simulation subgraphs contain a controlled private feedback boundary.
- Custom subgraphs are currently created by duplicating a built-in definition. Blank creation, wrapping selections, nesting, and recursion are not supported yet.
- There is no timeline, undo/redo, runtime plugin ABI, or saved simulation state yet.
- Images are linear RGBA16F GPU textures. Reaction state uses RG16F textures.

See [docs/adding-a-node.md](docs/adding-a-node.md) for the node extension interface and [docs/smoke-test.md](docs/smoke-test.md) for the interactive acceptance pass.
