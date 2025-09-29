#!/usr/bin/env python3
"""
Color utilities for visualizing token and response confidence as color gradients.

- Terminal ANSI 24-bit colors (TrueColor)
- HTML color helpers for web UIs
- Blue (low confidence) -> Red (high confidence) gradient
"""
from __future__ import annotations

import colorsys
from typing import Tuple, List, Dict, Any


def confidence_to_hue(confidence: float, min_hue: float = 240, max_hue: float = 0) -> float:
    """Map confidence in [0,1] to HSV hue in degrees.

    By default: 240° (blue) -> 0° (red).
    """
    normalized_confidence = max(0.0, min(1.0, confidence))
    return min_hue + (max_hue - min_hue) * normalized_confidence


def confidence_to_rgb(confidence: float, saturation: float = 0.8, value: float = 0.9) -> Tuple[int, int, int]:
    """Convert confidence to an RGB color.

    Returns an (r,g,b) tuple with 0..255 ints.
    """
    hue = confidence_to_hue(confidence)
    hue_normalized = hue / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue_normalized, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb_to_ansi_color(r: int, g: int, b: int, background: bool = False) -> str:
    """Convert RGB to an ANSI 24-bit escape code."""
    if background:
        return f"\033[48;2;{r};{g};{b}m"
    return f"\033[38;2;{r};{g};{b}m"


def confidence_to_ansi_color(confidence: float, background: bool = False) -> str:
    """Convenience: confidence -> ANSI color code."""
    r, g, b = confidence_to_rgb(confidence)
    return rgb_to_ansi_color(r, g, b, background)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to a hex color string (e.g., #ff0000)."""
    return f"#{r:02x}{g:02x}{b:02x}"


def confidence_to_html_color(confidence: float) -> str:
    """Convenience: confidence -> hex color string."""
    r, g, b = confidence_to_rgb(confidence)
    return rgb_to_hex(r, g, b)


def colorize_text_ansi(text: str, confidence: float, background: bool = False) -> str:
    """Wrap text with ANSI color based on confidence."""
    color_code = confidence_to_ansi_color(confidence, background)
    reset_code = "\033[0m"
    return f"{color_code}{text}{reset_code}"


def colorize_text_html(text: str, confidence: float, background: bool = False) -> str:
    """Wrap text with HTML span and inline style based on confidence."""
    color = confidence_to_html_color(confidence)
    style_prop = "background-color" if background else "color"
    return f'<span style="{style_prop}: {color}">{text}</span>'


def _clean_token(token: str) -> str:
    """Normalize token strings by removing common tokenizer markers."""
    return token.replace('Ġ', ' ').replace('▁', ' ')


def colorize_tokens_ansi(token_data: List[Dict[str, Any]], background: bool = False) -> str:
    """Colorize a list of tokens with individual confidences (terminal)."""
    colored_parts = []
    for t in token_data:
        token = _clean_token(t.get('token', ''))
        conf = float(t.get('confidence', 0.5) or 0.0)
        colored_parts.append(colorize_text_ansi(token, conf, background))
    return ''.join(colored_parts)


def colorize_tokens_html(token_data: List[Dict[str, Any]], background: bool = False) -> str:
    """Colorize a list of tokens with individual confidences (HTML)."""
    colored_parts = []
    for t in token_data:
        token = _clean_token(t.get('token', ''))
        conf = float(t.get('confidence', 0.5) or 0.0)
        safe = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        colored_parts.append(colorize_text_html(safe, conf, background))
    return ''.join(colored_parts)


def get_confidence_legend_ansi(steps: int = 10) -> str:
    """Build a small color legend for terminals."""
    parts = ["Уверенность: "]
    for i in range(steps):
        conf = i / (steps - 1) if steps > 1 else 1.0
        parts.append(colorize_text_ansi("██", conf))
    parts.append(" (синий=низкая, красный=высокая)")
    return ''.join(parts)


def get_confidence_legend_html(steps: int = 10) -> str:
    """Build a small color legend for HTML."""
    parts = ["<div>Уверенность: "]
    for i in range(steps):
        conf = i / (steps - 1) if steps > 1 else 1.0
        color = confidence_to_html_color(conf)
        parts.append(f'<span style="background-color: {color}; padding: 2px 4px;">  </span>')
    parts.append(" (синий=низкая, красный=высокая)</div>")
    return ''.join(parts)


def get_confidence_description(confidence: float) -> str:
    """Return a human-readable label for confidence."""
    if confidence >= 0.9:
        return "очень высокая"
    if confidence >= 0.7:
        return "высокая"
    if confidence >= 0.5:
        return "средняя"
    if confidence >= 0.3:
        return "низкая"
    return "очень низкая"