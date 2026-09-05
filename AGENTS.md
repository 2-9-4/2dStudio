@docs/adding-a-node.md
@docs/conventions.md
@docs/nodes.md

The built-in-node guide describes GPU lowering, parameter serialization, and editor controls.

When a root node is also available in a simulation subgraph, keep its custom editor available in
the subgraph canvas. Exclude any native-only controls from the contextual subgraph descriptor and
validate persisted values that would prevent simulation shader lowering.

Node boxes must render identically in the root graph and inside a subgraph body. All canvases
share one `renderNodeBody` path in `src/app/application.cpp`; do not add per-canvas title,
pin, parameter, or widget divergence there. Root-only runtime inspection (fusion state, live
values, GPU timing, Reset Simulation, subgraph/output instance actions) stays in the root
middle column because subgraph body IDs are body-local and would collide with root runtime maps.
`NodeRecord::label` is shader-source naming, not a canvas title.

Auto-layout lives in `src/core/layout.cpp` (forward pass + single-consumer snap pass, plus a
source relocation rule for competing merge inputs); see the Auto-layout section of
`docs/conventions.md` for the contract and its hard limits. A `layoutGraph` change is not done
until `./reaction_core_tests "auto-layout*"` passes and the offenders list of
`single-consumer total=` in the text_test probe showed no regressions.

update [AGENTS.md](AGENTS.md) and associated files with whatever required more reading to accomplish your task. REPORT ANY FOOTGUNS/EERGONOMIC ISSUES THAT CAUSED BUGS TO THE USER FOR LATER FIXING
