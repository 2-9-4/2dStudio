# Reaction Studio

Reaction Studio is a Linux-first, real-time 2D generative-art node editor. It combines GPU Perlin noise, Gray–Scott reaction diffusion, editable fused simulation subgraphs, polymorphic scalar/image math, color mapping, custom image convolution, and a separate live preview in a forward-only workflow.

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

The app opens an editor and a shared-context preview window. Right-click the canvas to add nodes. Drag between pins to link them; select links or nodes and press Delete to remove them. The default project is immediately animated.

Projects use `*.reaction.json`. They store project settings, graph topology, parameters, and editor positions. Simulation buffers and elapsed time deliberately restart when a project loads.

The add-node menu contains both **Reaction Diffusion (Monolithic)** and **Reaction Diffusion (Discrete)**. The discrete version is an immutable built-in subgraph with Image, A, and B outputs. Select it and press **Tab** to replace the root canvas with its primitive expression graph; press **Tab** again to return. Use **Duplicate as Editable** on an instance to create a shared project-local copy. Changes to that copy rebuild and reset all of its instances.

Use **File → Export Current Frame as PNG…** to write the graph output at its configured resolution. The export is an 8-bit RGB PNG and does not include the editor UI.

Use **Record** in the toolbar or **File → Start Video Recording…** to capture the graph output as an H.264 MP4 at the project resolution and target FPS. Pausing also pauses capture; use **Stop Recording** to finalize the file. Video dimensions must be even. The preview window remains constrained to the project's aspect ratio while it is resized.

## Current boundaries

- Linux and OpenGL 4.3 compute are the supported first target.
- Graphs are acyclic; simulation subgraphs contain a controlled private feedback boundary.
- Custom subgraphs are currently created by duplicating a built-in definition. Blank creation, wrapping selections, nesting, and recursion are not supported yet.
- There is no timeline, undo/redo, external image input, runtime plugin ABI, or saved simulation state yet.
- Images are linear RGBA16F GPU textures. Reaction state uses RG16F textures.

See [docs/adding-a-node.md](docs/adding-a-node.md) for the node extension interface and [docs/smoke-test.md](docs/smoke-test.md) for the interactive acceptance pass.
