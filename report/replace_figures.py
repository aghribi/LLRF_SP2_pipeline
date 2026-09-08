#!/usr/bin/env python3
"""Replace all \includegraphics commands with empty boxes"""

import re
import sys
from pathlib import Path

def replace_figures_with_boxes(content):
    """Replace \includegraphics with placeholder boxes"""
    # Pattern: \includegraphics[width=...]{...}
    pattern = r'\\includegraphics\[width=([^\]]+)\]\{[^}]+\}'

    def make_box(match):
        width = match.group(1)
        # Create a placeholder box with the same width
        return f'\\fbox{{\\parbox{{{width}}}{{\\centering [Figure Placeholder]\\\\Height: 0.3\\textwidth}}}}'

    return re.sub(pattern, make_box, content)

def replace_table_widths(content):
    """Replace tabular with tabular* using \linewidth"""
    # Replace \begin{tabular} with \begin{tabular*}{\linewidth}
    content = re.sub(
        r'\\begin\{tabular\}\{([^}]+)\}',
        r'\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}\1}',
        content
    )
    # Replace \end{tabular} with \end{tabular*}
    content = re.sub(r'\\end\{tabular\}', r'\\end{tabular*}', content)
    return content

def process_file(filepath):
    """Process a single .tex file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace figures
    content = replace_figures_with_boxes(content)

    # Replace table widths
    content = replace_table_widths(content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Processed: {filepath}")

def main():
    sections_dir = Path("sections")
    for tex_file in sections_dir.glob("*.tex"):
        process_file(tex_file)

    print("\nDone! All figures replaced with boxes and tables adjusted to linewidth.")

if __name__ == "__main__":
    main()
