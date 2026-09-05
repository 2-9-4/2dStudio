#pragma once

#include "reaction/core/node.hpp"

namespace reaction {

void registerInputNodes(NodeRegistry& registry);
void registerConvolutionNode(NodeRegistry& registry);
void registerTextureSampleNode(NodeRegistry& registry);
void registerTransform2DNode(NodeRegistry& registry);
void registerDomainWarpNodes(NodeRegistry& registry);
void registerGradientNodes(NodeRegistry& registry);
void registerPolarCoordinatesNodes(NodeRegistry& registry);
void registerRepeatFoldNode(NodeRegistry& registry);
void registerWaveNode(NodeRegistry& registry);
void registerWorleyNoiseNode(NodeRegistry& registry);
void registerCompareNode(NodeRegistry& registry);
void registerHashNode(NodeRegistry& registry);
// Procedural node families live in focused translation units.  Keep their
// registration functions here and call them from registerBuiltInNodes so each
// family can evolve without turning nodes.cpp into another shared hotspot.
void registerCoordinateNodes(NodeRegistry& registry);
void registerProceduralNoiseNodes(NodeRegistry& registry);
void registerProceduralGeneratorNodes(NodeRegistry& registry);

} // namespace reaction
