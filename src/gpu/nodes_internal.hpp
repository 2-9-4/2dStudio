#pragma once

#include "reaction/core/node.hpp"

namespace reaction {

#define REACTION_NODE_FAMILY(registrationFunction) \
    void registrationFunction(NodeRegistry& registry);
#include "builtin_node_families.inc"
#undef REACTION_NODE_FAMILY

} // namespace reaction
