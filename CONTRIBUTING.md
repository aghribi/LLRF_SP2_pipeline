# Contributing

Contributions are welcome, especially toward making this pipeline usable at facilities beyond
SPIRAL2/GANIL.

## Ways to contribute

- **Generalization**: this codebase currently has CC-IN2P3-cluster-specific paths and
  SPIRAL2-specific data conventions (see `docs/`). Abstracting these into configuration so
  another accelerator's LLRF post-mortem data can be plugged in is the highest-value
  contribution right now.
- **Bug reports**: open an issue with the pipeline step, the input that triggered it, and the
  observed vs. expected behavior.
- **Documentation**: the `docs/` guides are kept in sync with the pipeline as it changes;
  corrections and clarifications are welcome.

## Before submitting a change

- If you're changing a modeling step or feature-engineering logic, note which
  `analysis/results_manifest/*.yaml` entries it affects — every number in the paper traces to
  one of these, and `report/code/check_no_hand_typed_numbers.py` checks that the report never
  hand-types a number that should come from a manifest instead.
- Keep changes to `report/sections/*.tex` scoped to what your code change actually affects.
- Describe what you tested, and how (this project has a documented history of unverified
  claims causing real problems — see the discussion in `report/` — so a described, reproducible
  test matters more than a confident description of the fix).

## Getting help

Open an issue for questions about the pipeline architecture or how to adapt it to a different
facility's data.
