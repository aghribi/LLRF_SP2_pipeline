#!/usr/bin/env python3
"""Render a figure PDF to PNG at high DPI for visual verification (not shipped)."""
import sys
import fitz  # PyMuPDF


def render(pdf_path: str, png_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    pix.save(png_path)
    print(f"{pdf_path} -> {png_path} ({pix.width}x{pix.height})")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300)
