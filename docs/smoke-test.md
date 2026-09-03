# Interactive smoke test

1. Launch `reaction_studio` on an OpenGL 4.3-capable desktop and confirm both windows open.
2. Confirm the default noise-driven reaction-diffusion graph animates and the preview matches the Output node thumbnail.
3. Right-click the canvas and confirm Simulation lists both monolithic and discrete reaction diffusion nodes.
4. Add and select the discrete node, confirm its Image/Chemical A/Chemical B outputs and integer/boolean controls, then press Tab and confirm its read-only graph uses the same node-editor canvas, with full descriptive node and input names.
5. Press Tab to return, choose **Duplicate as Editable**, press Tab to enter the copy, rename it, rewire an eligible primitive input, and confirm all instances of that copy reset and change together.
6. Save and reload the project; confirm the custom definition, instance controls, and external links survive.
3. Change feed, kill, noise scale, and color ramp values and confirm the preview responds live.
4. Pause and confirm Time and simulation state stop while node editing remains responsive; resume and confirm forward evolution continues.
5. Reset and confirm the simulation reinitializes without moving or deleting nodes.
6. Add Math and Mix nodes through the searchable background menu, connect them, then delete a link.
7. Attempt to create a cycle and confirm the editor reports the graph error while preserving the last valid preview.
8. Save, modify the graph, load the saved file, and confirm topology, values, positions, resolution, and target FPS restore while the simulation restarts.
9. Close the preview, keep editing, and reopen it from Window → Open Preview.
10. Change the project to a non-square resolution, resize the preview, and confirm it remains constrained to the project aspect ratio without stretching.
11. Start recording with animated Perlin noise, let the project run, pause and resume once, then stop. Confirm the MP4 has the project resolution, excludes the editor UI, advances smoothly by one animation frame at a time, and does not advance while paused.
12. Set 1024×1024 at 60 FPS and observe per-node GPU timings and overall FPS. Outside an active recording, after warm-up there should be no recurring texture allocations or CPU image readback.
