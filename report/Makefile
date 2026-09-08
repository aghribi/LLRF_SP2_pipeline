# Makefile for compiling SPIRAL2 LLRF Anomaly Detection Article
# Author: Adnan Ghribi

# Main document
MAIN = main
BIBTEX = bibtex
PDFLATEX = pdflatex
PYTHON = python3
# Figure scripts unpickle trained models (sklearn 1.6.1) and need PyPostMortem
# on the path -- both only available in this env, not the ambient python3.
PYTHON_ENV = /pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python

# Directories
BUILD_DIR = build
SECTIONS_DIR = sections
FIGURES_DIR = figures
CODE_DIR = code

# Flags
PDFLATEX_FLAGS = -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR)

# Default target
all: $(MAIN).pdf

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Generate all data-driven figures from the current pipeline outputs (real V6
# pickles/npz under /sps/.../cooked_data_v6 -- see each script's own docstring
# for its exact data source). Each script applies the shared style module
# (code/plot_style.py: Okabe-Ito colorblind-safe palette, consistent fonts/
# widths matching main.tex's revtex4-2 column widths) and writes directly to
# figures/*.pdf. Re-run this after any pipeline rerun that could change the
# figures' underlying data (a new V6 extraction, a re-run step 05/08/09c/10).
figures:
	@echo "Generating publication-quality figures from current pipeline outputs..."
	$(PYTHON_ENV) $(CODE_DIR)/fig_signal_examples.py
	$(PYTHON_ENV) $(CODE_DIR)/fig_precursor.py
	$(PYTHON_ENV) $(CODE_DIR)/fig_subtype_profiles.py
	$(PYTHON_ENV) $(CODE_DIR)/fig_shap_lime.py
	$(PYTHON_ENV) $(CODE_DIR)/fig_confusion_matrix.py
	@echo "Figures generated successfully!"

# Compile PDF (full build with bibliography)
$(MAIN).pdf: $(BUILD_DIR) $(MAIN).tex $(wildcard $(SECTIONS_DIR)/*.tex)
	@echo "First LaTeX pass..."
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN).tex
	@echo "Running BibTeX..."
	cd $(BUILD_DIR) && $(BIBTEX) $(MAIN)
	@echo "Second LaTeX pass..."
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN).tex
	@echo "Third LaTeX pass (for references)..."
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN).tex
	@echo "Copying PDF to main directory..."
	cp $(BUILD_DIR)/$(MAIN).pdf .
	@echo "PDF compiled successfully: $(MAIN).pdf"

# Quick compile (single pass, no bibliography)
quick: $(BUILD_DIR)
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf .

# Clean auxiliary files
clean:
	rm -rf $(BUILD_DIR)
	rm -f *.aux *.log *.bbl *.blg *.toc *.out *.lof *.lot

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# View PDF
view: $(MAIN).pdf
	@if command -v evince > /dev/null; then \
		evince $(MAIN).pdf & \
	elif command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf & \
	else \
		echo "No PDF viewer found. Please open $(MAIN).pdf manually."; \
	fi

# Count words (approximate, excluding LaTeX commands)
wordcount:
	@echo "Approximate word count:"
	@detex $(SECTIONS_DIR)/*.tex | wc -w

# Check for missing references
check-refs:
	@echo "Checking for undefined references..."
	@grep -i "undefined" $(BUILD_DIR)/$(MAIN).log || echo "No undefined references found."

# Spell check (requires aspell)
spell:
	@for file in $(SECTIONS_DIR)/*.tex; do \
		echo "Checking $$file..."; \
		aspell --lang=en --mode=tex check $$file; \
	done

# Full workflow: figures + compile + view
full: figures all view

# Sync report numbers with the pipeline's current results_manifest state.
# Run this after: re-running any canonical pipeline script (each ends in
# save_manifest()), or hand-editing report/sections/*.tex prose. Regenerates
# _generated_macros.tex from analysis/results_manifest/*.yaml, then checks
# that no bare (un-macro'd, non-% MANUAL:) number crept into any section --
# this is the actual "did I keep the report honest" gate, not a compile
# check. Fails loudly (non-zero exit) if the checker finds anything.
sync:
	@echo "Regenerating macros from analysis/results_manifest/*.yaml..."
	@$(PYTHON) $(CODE_DIR)/generate_macros.py
	@echo ""
	@echo "Checking for un-allowlisted bare numbers..."
	@$(PYTHON) $(CODE_DIR)/check_no_hand_typed_numbers.py

# Help
help:
	@echo "SPIRAL2 LLRF Article - Makefile Targets:"
	@echo "  make all        - Compile PDF with bibliography (default)"
	@echo "  make quick      - Quick compile (single pass, no bibliography)"
	@echo "  make figures    - Generate all data-driven figures from current pipeline outputs"
	@echo "  make view       - Open compiled PDF"
	@echo "  make clean      - Remove auxiliary files"
	@echo "  make distclean  - Remove all generated files including PDF"
	@echo "  make wordcount  - Count words in document"
	@echo "  make check-refs - Check for undefined references"
	@echo "  make spell      - Run spell checker (requires aspell)"
	@echo "  make full       - Generate figures, compile, and view"
	@echo "  make sync       - Regenerate result macros + check for hand-typed numbers"
	@echo "                    (run after any pipeline rerun or prose edit)"
	@echo "  make help       - Show this help message"

.PHONY: all quick clean distclean view wordcount check-refs spell figures full help
