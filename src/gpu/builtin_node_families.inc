// Built-in node-family registration manifest.
//
// Add one REACTION_NODE_FAMILY(...) entry for each focused family registration
// function. nodes_internal.hpp derives declarations from this file and
// registerBuiltInNodes() derives calls from the same list.
//
// Focused node implementation files should use the *_node.cpp naming convention;
// CMake discovers those sources automatically.
REACTION_NODE_FAMILY(registerInputNodes)
REACTION_NODE_FAMILY(registerTextureSampleNode)
REACTION_NODE_FAMILY(registerTransform2DNode)
REACTION_NODE_FAMILY(registerDomainWarpNodes)
REACTION_NODE_FAMILY(registerGradientNodes)
REACTION_NODE_FAMILY(registerPolarCoordinatesNodes)
REACTION_NODE_FAMILY(registerRepeatFoldNode)
REACTION_NODE_FAMILY(registerWaveNode)
REACTION_NODE_FAMILY(registerWorleyNoiseNode)
REACTION_NODE_FAMILY(registerCompareNode)
REACTION_NODE_FAMILY(registerHashNode)
REACTION_NODE_FAMILY(registerTableNode)
REACTION_NODE_FAMILY(registerBitTestNode)
REACTION_NODE_FAMILY(registerIntegerMaskNode)
REACTION_NODE_FAMILY(registerDistanceTransformNodes)
REACTION_NODE_FAMILY(registerDitherNode)
REACTION_NODE_FAMILY(registerConvolutionNode)
