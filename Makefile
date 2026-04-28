.PHONY: help install solver download-models download-reference \
        run-circular-fine run-circular-coarse run-circular-dl-amr run-circular-grad-amr \
        figs smoke-test clean clean-cases

help:
	@echo "DL-AMR reproduction targets"
	@echo ""
	@echo "Setup:"
	@echo "  make install            - install Python environment (env.yml)"
	@echo "  make solver             - build OpenFOAM solver (amrPimpleFoam)"
	@echo "  make download-models    - download pretrained models from Zenodo/Release"
	@echo "  make download-reference - download minimal reference data"
	@echo ""
	@echo "Quick check:"
	@echo "  make smoke-test         - quick reproducibility check (no full simulation)"
	@echo ""
	@echo "Reproduce paper figures:"
	@echo "  make figs               - regenerate all paper figures from cached data"
	@echo ""
	@echo "Full simulations (long-running):"
	@echo "  make run-circular-fine"
	@echo "  make run-circular-coarse"
	@echo "  make run-circular-dl-amr"
	@echo "  make run-circular-grad-amr"
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
	cd solver/amrPimpleFoam && wmake

download-models:
	bash scripts/download_models.sh

download-reference:
	bash scripts/download_reference_data.sh

smoke-test:
	bash scripts/run_smoke_test.sh

figs:
	bash scripts/generate_figures.sh

run-circular-fine:
	cd cases/circular_Re200/fine && ./Allrun

run-circular-coarse:
	cd cases/circular_Re200/coarse && ./Allrun

run-circular-dl-amr:
	cd cases/circular_Re200/dl_amr && ./Allrun

run-circular-grad-amr:
	cd cases/circular_Re200/grad_amr && ./Allrun

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
