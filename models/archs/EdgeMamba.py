"""Re-export of the EdgeMamba building blocks.

The paper's ``EdgeMamba`` module (gradient-prior guided selective
state space model) is implemented in ``models/edgemamba.py``.
This module re-exports the building blocks under the unified name
``models.archs.EdgeMamba`` for use by ``models/piecesmamba.py``.
"""
from models.edgemamba import (
    GradientExtractor,
    TextureExtractor,
    FrequencyExtractor,
    NoPriorExtractor,
    build_prior_extractor,
    PRIOR_TYPES,
    SimplifiedGradientToPriority,
    GradStateSpaceBlock,
    GradientGuidedMamba,
    GradientGuidedSelectiveScan,
    semantic_neighbor,
    index_reverse,
)

__all__ = [
    "GradientExtractor",
    "TextureExtractor",
    "FrequencyExtractor",
    "NoPriorExtractor",
    "build_prior_extractor",
    "PRIOR_TYPES",
    "SimplifiedGradientToPriority",
    "GradStateSpaceBlock",
    "GradientGuidedMamba",
    "GradientGuidedSelectiveScan",
    "semantic_neighbor",
    "index_reverse",
]
