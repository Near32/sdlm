#!/usr/bin/env python3
"""Render text strings to PNG images with Unicode-aware font selection.

Uses Noto Sans font family to cover Latin, CJK, Arabic, Devanagari, Bengali,
Thai, Hebrew, and symbol scripts. Each character is routed to the appropriate
font based on its Unicode codepoint.
"""

import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# --- Font paths (system Noto Sans) ---

FONT_PATHS = {
    "default": "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "cjk": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "arabic": "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "devanagari": "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "bengali": "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "thai": "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "hebrew": "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
    "symbols": "/usr/share/fonts/truetype/noto/NotoSansSymbols-Regular.ttf",
    "symbols2": "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
    "dejavu": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}


def get_font_stack(size: int) -> dict[str, ImageFont.FreeTypeFont]:
    """Load all available fonts at the given size.

    Args:
        size: Font size in points.

    Returns:
        Dictionary mapping script name to loaded font object.
        Only includes fonts whose files exist on the system.
    """
    stack = {}
    for name, path in FONT_PATHS.items():
        if Path(path).exists():
            try:
                stack[name] = ImageFont.truetype(path, size)
            except Exception as e:
                print(f"  Warning: could not load font {path}: {e}", file=sys.stderr)
        else:
            print(f"  Warning: font not found: {path}", file=sys.stderr)

    # Ensure we always have a default
    if "default" not in stack:
        stack["default"] = ImageFont.load_default()

    return stack


def select_font_for_char(char: str, font_stack: dict[str, ImageFont.FreeTypeFont]) -> ImageFont.FreeTypeFont:
    """Return the appropriate font for a character based on its Unicode codepoint.

    Args:
        char: A single character.
        font_stack: Dictionary of loaded fonts keyed by script name.

    Returns:
        The font to use for rendering this character.
    """
    cp = ord(char)

    # Arabic: U+0600-U+06FF, U+0750-U+077F, U+FB50-U+FDFF, U+FE70-U+FEFF
    if (0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or
            0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF):
        return font_stack.get("arabic", font_stack["default"])

    # Devanagari: U+0900-U+097F
    if 0x0900 <= cp <= 0x097F:
        return font_stack.get("devanagari", font_stack["default"])

    # Bengali: U+0980-U+09FF
    if 0x0980 <= cp <= 0x09FF:
        return font_stack.get("bengali", font_stack["default"])

    # Thai: U+0E00-U+0E7F
    if 0x0E00 <= cp <= 0x0E7F:
        return font_stack.get("thai", font_stack["default"])

    # Hebrew: U+0590-U+05FF
    if 0x0590 <= cp <= 0x05FF:
        return font_stack.get("hebrew", font_stack["default"])

    # CJK: U+3000-U+9FFF, U+F900-U+FAFF, U+AC00-U+D7AF (Korean Hangul)
    if (0x3000 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or
            0xAC00 <= cp <= 0xD7AF):
        return font_stack.get("cjk", font_stack["default"])

    # Symbols: U+2000-U+2BFF
    if 0x2000 <= cp <= 0x2BFF:
        font = font_stack.get("symbols", font_stack.get("symbols2", font_stack["default"]))
        return font

    # Emoji / Misc Symbols: U+1F000+
    if cp >= 0x1F000:
        return font_stack.get("symbols2", font_stack.get("symbols", font_stack["default"]))

    # Default (Latin, Greek, Cyrillic, etc.)
    return font_stack["default"]


def _char_width(char: str, font: ImageFont.FreeTypeFont) -> int:
    """Get the advance width of a single character."""
    bbox = font.getbbox(char)
    return bbox[2] - bbox[0]


def render_text_to_png(
    text: str,
    filepath: str,
    font_size: int = 56,
    max_width: Optional[int] = None,
    bg_color: tuple = (255, 255, 255, 0),
    text_color: tuple = (0, 0, 0, 255),
    padding: int = 4,
    highlight_whitespace: bool = True,
    char_colors: Optional[list[tuple]] = None,
) -> Path:
    """Render a text string to a PNG image with per-character font selection.

    Args:
        text: The text to render.
        filepath: Output PNG file path.
        font_size: Font size in points.
        max_width: Maximum image width in pixels (enables word wrap). None = no wrap.
        bg_color: Background RGBA color tuple.
        text_color: Text RGBA color tuple.
        padding: Padding around the text in pixels.
        highlight_whitespace: If True, render whitespace characters with visible
            markers (\\n, \\t, ␣) in a muted highlight colour.

    Returns:
        Path to the saved PNG file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    font_stack = get_font_stack(font_size)

    # Clean text (\t handled in the rendering loop with optional highlighting)
    text = text.replace("\r", "")

    # Estimate canvas size (will be cropped)
    line_height = int(font_size * 1.4)
    estimated_lines = max(1, len(text) // 40) + text.count("\n") + 5
    canvas_width = max_width if max_width else max(2000, len(text) * font_size)
    canvas_height = max(line_height * estimated_lines + 2 * padding, line_height * 3)

    img = Image.new("RGBA", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(img)

    x = padding
    y = padding

    # Highlight colour for whitespace markers
    ws_color = (140, 140, 140, 180)

    for i, char in enumerate(text):
        if char == "\n":
            if highlight_whitespace:
                ws_fill = char_colors[i] if (char_colors and i < len(char_colors) and char_colors[i] != text_color) else ws_color
                # Render literal "\n" marker instead of breaking to a new line
                for marker_ch in "\\n":
                    mfont = font_stack["default"]
                    mcw = _char_width(marker_ch, mfont)
                    if max_width and (x + mcw) > (max_width - padding):
                        x = padding
                        y += line_height
                    draw.text((x, y), marker_ch, font=mfont, fill=ws_fill)
                    x += mcw
            else:
                x = padding
                y += line_height
            continue

        if char == "\t":
            if highlight_whitespace:
                ws_fill = char_colors[i] if (char_colors and i < len(char_colors) and char_colors[i] != text_color) else ws_color
                for marker_ch in "\\t":
                    mfont = font_stack["default"]
                    mcw = _char_width(marker_ch, mfont)
                    if max_width and (x + mcw) > (max_width - padding):
                        x = padding
                        y += line_height
                    draw.text((x, y), marker_ch, font=mfont, fill=ws_fill)
                    x += mcw
            else:
                x += _char_width(" ", font_stack["default"]) * 4
            continue

        if char == " " and highlight_whitespace:
            ws_fill = char_colors[i] if (char_colors and i < len(char_colors) and char_colors[i] != text_color) else ws_color
            ws_font = font_stack.get("dejavu", font_stack["default"])
            cw = _char_width("\u2423", ws_font)
            if max_width and (x + cw) > (max_width - padding):
                x = padding
                y += line_height
            # Draw open-box (bounded underscore) as space marker
            draw.text((x, y), "\u2423", font=ws_font, fill=ws_fill)
            x += cw
            continue

        # Normal character (unchanged)
        font = select_font_for_char(char, font_stack)
        cw = _char_width(char, font)

        # Word wrap
        if max_width and (x + cw) > (max_width - padding):
            x = padding
            y += line_height

        color = char_colors[i] if char_colors and i < len(char_colors) else text_color
        draw.text((x, y), char, font=font, fill=color)
        x += cw

    # Crop vertically to content, keep full width for consistent sizing
    bbox = img.getbbox()
    if bbox:
        # Fixed width (full canvas), variable height (content only)
        crop_box = (
            0,
            max(0, bbox[1] - padding),
            canvas_width,
            min(img.height, bbox[3] + padding),
        )
        img = img.crop(crop_box)
    else:
        # Empty text — create image with correct width
        img = Image.new("RGBA", (canvas_width, line_height), bg_color)

    img.save(str(filepath), "PNG")
    return filepath


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="Render text to PNG")
    parser.add_argument("text", help="Text to render")
    parser.add_argument("-o", "--output", default="test_output.png", help="Output PNG path")
    parser.add_argument("--font-size", type=float, default=8, help="Font size in points (default: 8)")
    parser.add_argument("--image-width", type=float, default=2, help="Image width in cm (default: 2)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution in DPI (default: 300)")
    args = parser.parse_args()

    font_size_px = round(args.font_size * args.dpi / 72)
    max_width_px = round(args.image_width * args.dpi / 2.54)

    out = render_text_to_png(args.text, args.output, font_size=font_size_px, max_width=max_width_px)
    print(f"Saved to {out} (font={font_size_px}px, width={max_width_px}px @ {args.dpi}dpi)")
