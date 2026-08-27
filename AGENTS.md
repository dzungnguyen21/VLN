# AGENTS.md

## Repository Overview

InternNav is a Python toolbox for embodied navigation built around PyTorch, Habitat, and Isaac Sim.

- `internnav/`: main package and navigation implementations.
- `tests/`: project tests; pytest is configured to discover tests here.
- `scripts/`: dataset conversion, evaluation, training, and deployment entry points.
- `requirements/`: dependency sets for core, Habitat, Isaac, models, and InternVLA-N1.
- `habitat-lab/`: vendored Habitat Lab source and its own project files; treat it as a separate dependency tree unless a task explicitly targets it.
- `vln_subgoal_pipeline/`: pipeline application with its own tests and requirements.
- `assets/`, `data/`, and `results/`: datasets, generated artifacts, and experiment outputs; avoid modifying large artifacts during code changes.

## Python and Dependencies

- Supported Python versions are 3.8 through 3.12, as enforced by `setup.py`.
- Install only the dependency set required for the task. Optional extras include `habitat`, `isaac`, `model`, `baseline`, and `internvla_n1`.
- Keep changes compatible with the existing package layout and public APIs.

## Validation

Run focused tests first, then the broader suite when practical:

```bash
python -m pytest tests/path/to/test_file.py -q
python -m pytest -q
```

Useful static checks configured by the repository:

```bash
black --check .
isort --check-only .
flake8 .
pre-commit run --all-files
```

Some tests and integrations require optional dependencies, a GPU, Habitat assets, Isaac Sim, or downloaded datasets. Report such environment limitations rather than weakening tests or changing unrelated configuration.

## Style

- Follow the existing Python style and preserve public APIs unless the task requires a change.
- Black uses a 120-character line length and preserves string quoting.
- isort uses the Black profile.
- Keep edits focused; do not reformat unrelated files or commit generated outputs.
- Add tests for behavior changes, placing them under `tests/` and using the existing pytest markers (`slow` and `gpu`) when appropriate.
