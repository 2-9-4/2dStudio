# Interactive smoke test

1. Launch `reaction_studio` on an OpenGL 4.3-capable desktop and confirm both windows open.
2. Confirm the default noise-driven reaction-diffusion graph animates and the preview matches the Output node thumbnail.
3. Right-click the canvas and confirm Simulation lists both monolithic and discrete reaction diffusion nodes.
4. Add and select the discrete node, confirm its Image/Chemical A/Chemical B outputs and integer/boolean controls, then press Tab. Confirm it opens an editable, full-size node canvas with descriptive node and pin names rather than an inspector-only expression view.
5. Right-click the subgraph canvas and add Float, Math, Threshold, Select, Canvas Coordinates, and Laplacian nodes. Confirm they have the same names, pins, controls, and operation selector as their root-graph versions. Delete the temporary Threshold node and confirm its incident links, if any, disappear with it.
6. Connect the new Float output to a compatible Math input, then drag a different output onto that occupied input. Confirm the old link is replaced. Select and delete the replacement link, reconnect the intended source, and confirm links can run in either screen direction without an insertion-order restriction.
7. Pan the subgraph with a right-button drag and zoom with the wheel at the center and near every canvas edge. Confirm navigation remains smooth, the toolbar does not scroll or move, the viewport does not pan a second time, and nodes are clipped only by the actual canvas boundary rather than an invisible inner panel.
8. Press Tab to return to the root graph and confirm its camera and node positions were not changed by subgraph navigation. Re-enter the discrete subgraph and confirm it is still a full-size canvas.
9. Open **Subgraph Settings…**, rename the definition, and edit an interface label or control range. Rewire an existing Math input and confirm every root instance sharing that project subgraph resets and uses the change.
10. Save and reload the project; confirm added and deleted subgraph nodes, replacement links, node positions, interface edits, instance controls, and external links survive.
11. Change feed, kill, noise scale, and color ramp values and confirm the preview responds live.
12. Pause and confirm Time and simulation state stop while node editing remains responsive; resume and confirm forward evolution continues.
13. Reset and confirm the simulation reinitializes without moving or deleting nodes.
14. Add Math and Mix nodes to the root graph through the searchable background menu, connect them, then delete a link.
15. Attempt to create a cycle and confirm the editor reports the graph error while preserving the last valid preview.
16. Save, modify the root graph, load the saved file, and confirm topology, values, positions, resolution, and target FPS restore while the simulation restarts.
17. Close the preview, keep editing, and reopen it from Window → Open Preview.
18. Change the project to a non-square resolution, resize the preview, and confirm it remains constrained to the project aspect ratio without stretching.
19. Start recording with animated Perlin noise, let the project run, pause and resume once, then stop. Confirm the MP4 has the project resolution, excludes the editor UI, advances smoothly by one animation frame at a time, and does not advance while paused.
20. Set 1024×1024 at 60 FPS and observe per-node GPU timings and overall FPS. Outside an active recording, after warm-up there should be no recurring texture allocations or CPU image readback.
