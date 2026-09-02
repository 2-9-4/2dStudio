# Reaction Studio

Reaction Studio is a Linux-first, real-time 2D generative-art node editor. It combines GPU Perlin noise, Gray–Scott reaction diffusion, polymorphic scalar/image math, color mapping, custom image convolution, and a separate live preview in a forward-only workflow.

## Build

Requirements are CMake 3.24+, a C++20 compiler, GLFW 3.3, OpenGL 4.3, and Python (used by GLAD generation). Remaining dependencies are pinned and downloaded by CMake.

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

Use **File → Export Current Frame as PNG…** to write the graph output at its configured resolution. The export is an 8-bit RGB PNG and does not include the editor UI.

## Current boundaries

- Linux and OpenGL 4.3 compute are the supported first target.
- Graphs are acyclic; Reaction Diffusion contains the only private feedback loop.
- There is no timeline, undo/redo, image export, external image input, runtime plugin ABI, or saved simulation state yet.
- Images are linear RGBA16F GPU textures. Reaction state uses RG16F textures.

See [docs/adding-a-node.md](docs/adding-a-node.md) for the node extension interface and [docs/smoke-test.md](docs/smoke-test.md) for the interactive acceptance pass.
