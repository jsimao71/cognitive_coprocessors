"""Semantic-alignment experiments for Paper 1."""

from .bottleneck import (
    asl_to_bottleneck,
    lower_bottleneck_to_asl,
    parse_bottleneck,
    render_bottleneck,
)
from .compositional_generator import (
    TIER_CONTROLS,
    GenerationControls,
    freeze_compositional_pilots,
    generate_compositional_record,
)

__all__ = [
    "TIER_CONTROLS",
    "GenerationControls",
    "asl_to_bottleneck",
    "freeze_compositional_pilots",
    "generate_compositional_record",
    "lower_bottleneck_to_asl",
    "parse_bottleneck",
    "render_bottleneck",
]
