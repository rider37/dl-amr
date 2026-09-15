.PHONY: help install solver download-models download-reference download-fields download-artifacts \
        figs smoke-test clean clean-cases

# Python interpreter used by smoke-test and figs targets.
# Override with `make figs PYTHON=python3.11`.
PYTHON ?= $(shell command -v python || command -v python3)

help:
	@echo "DL-AMR reproduction targets"
	@echo ""
	@echo "Setup:"
	@echo "  make install            - install Python environment (env.yml)"
	@echo "  make solver             - build hexRef4 library and OpenFOAM solvers"
	@echo "  make download-models    - download pretrained models from Zenodo/Release"
	@echo "  make download-reference - download minimal reference data (Figs 8, 9, E.1)"
	@echo "  make download-fields    - download cached wake-field data (Figs 4, 5, 7, D.1, F.1)"
	@echo "  make download-artifacts - all three of the above (one shot)"
	@echo ""
	@echo "Quick check:"
	@echo "  make smoke-test         - quick reproducibility check (no full simulation)"
	@echo ""
	@echo "Reproduce paper figures:"
	@echo "  make figs               - regenerate all paper figures from cached data"
	@echo ""
	@echo "Full simulations (long-running):"
	@echo "  make run-<geometry>-<variant>, e.g. run-circular-dl_amr, run-square-static"
	@echo "  variants: fine coarse grad_amr vort_amr q_amr dl_amr static dl_amr_meanhead (all)"
	@echo "            vort_wrapper kelly_wake (circular only)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              - remove all build artifacts and case results"
	@echo "  make clean-cases        - remove only case time directories and outputs"

install:
	@if command -v conda >/dev/null 2>&1; then \
		conda env create -f env.yml || conda env update -f env.yml; \
	else \
		pip install -r requirements.txt; \
	fi

solver:
	@if [ -z "$$LIBTORCH_DIR" ]; then \
	    echo "ERROR: LIBTORCH_DIR is not set."; \
	    echo "Export the path to your LibTorch install before building, e.g.:"; \
	    echo "  export LIBTORCH_DIR=/path/to/libtorch"; \
	    exit 1; \
	fi
	export FOAM_USER_SRC=$$(pwd)/solver && cd solver/hexRef4 && wmake libso
	cd solver/amrPimpleFoam && wmake

download-models:
	bash scripts/download_models.sh

download-reference:
	bash scripts/download_reference_data.sh

download-fields:
	bash scripts/download_reference_fields.sh

download-artifacts: download-models download-reference download-fields

smoke-test:
	PYTHON=$(PYTHON) bash scripts/run_smoke_test.sh

figs:
	PYTHON=$(PYTHON) bash scripts/generate_figures.sh

# Generic run target: make run-<geometry>-<variant>  (geometry: circular, square, diamond)
run-circular-%: ; cd cases/circular_Re200/$* && ./Allrun
run-square-%:   ; cd cases/square_Re150/$*   && ./Allrun
run-diamond-%:  ; cd cases/diamond_Re150/$*  && ./Allrun

clean-cases:
	@echo "Cleaning case time directories and outputs..."
	@find cases -type d -regex '.*/\(processor[0-9]+\|postProcessing\|VTK\)' -prune -exec rm -rf {} + 2>/dev/null || true
	@find cases -type d -regex '.*/[1-9][0-9]*' -prune -exec rm -rf {} + 2>/dev/null || true
	@find cases -name "log.*" -delete 2>/dev/null || true
	@find cases -name "*.foam" -delete 2>/dev/null || true

clean: clean-cases
	@echo "Cleaning Python and LaTeX artifacts..."
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.aux" -o -name "*.log" -o -name "*.out" -delete 2>/dev/null || true
