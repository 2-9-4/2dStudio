#pragma once

#include "reaction/core/graph.hpp"

#include <unordered_map>

namespace reaction {

// Recomputes node positions for a graph body using a layered left-to-right
// layout: sources sit on the left, dependencies flow to the right, every merge
// is vertically placed at the average of the lines feeding into it, and any
// node with exactly one consumer is snapped onto that consumer's line so
// chains hug their consumer instead of drifting onto a stale input lane.
//
// The layout is deterministic and only depends on the graph topology (current
// positions are used as a tie-breaker), so it can be re-run to organize any
// graph. Pass measured node sizes to avoid overlapping tall nodes; missing
// entries fall back to a default size.
void layoutGraph(GraphBody& body,
                 const std::unordered_map<NodeId, Vec2>* nodeSizes = nullptr);

} // namespace reaction
