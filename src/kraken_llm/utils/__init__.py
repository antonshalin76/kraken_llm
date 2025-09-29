"""
Kraken utilities

Common utilities for logging, media processing and visualization helpers.
"""

from .logging import *  # noqa: F401,F403
from .media import *    # noqa: F401,F403
from .color import (
    confidence_to_hue,
    confidence_to_rgb,
    rgb_to_ansi_color,
    confidence_to_ansi_color,
    rgb_to_hex,
    confidence_to_html_color,
    colorize_text_ansi,
    colorize_text_html,
    colorize_tokens_ansi,
    colorize_tokens_html,
    get_confidence_legend_ansi,
    get_confidence_legend_html,
    get_confidence_description,
)

__all__ = [
    # Color utilities
    "confidence_to_hue",
    "confidence_to_rgb",
    "rgb_to_ansi_color",
    "confidence_to_ansi_color",
    "rgb_to_hex",
    "confidence_to_html_color",
    "colorize_text_ansi",
    "colorize_text_html",
    "colorize_tokens_ansi",
    "colorize_tokens_html",
    "get_confidence_legend_ansi",
    "get_confidence_legend_html",
    "get_confidence_description",
]
