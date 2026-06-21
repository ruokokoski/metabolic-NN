#!/usr/bin/env python
"""Fit MINN split fluxomics with a MINN-like policy in iML1515.

This is a separate trial from `fit_minn_fluxomics_soft_inputs.py`.
It uses the same LP fitting implementation, but changes the defaults to be
closer to the observed original MINN fitted-data policy while keeping the
iML1515 GEM context used by this project:

    - model: models/iML1515.xml
    - source: MINN_data/fluxomics_iAF1260_reduced_split.csv
    - biomass fixed by default
    - no hard glucose/O2 soft-input band by default

Glucose and oxygen still have weighted absolute-deviation terms in the fitting
objective, but they are not constrained to remain inside a tolerance band.
This is closer to the observed original MINN fitted file, where exchange
values were allowed to move, while still fitting into iML1515 rather than the
paper's reduced split GEM.

All fitting, reporting, and comparison logic is implemented in
`fit_minn_fluxomics_soft_inputs.py`. This wrapper only changes defaults. The
default post-fit comparison is anchored to the non-fitted source file; the
original MINN fitted file is treated only as an optional descriptive reference
via `--compare-output-to-reference`.
"""

from __future__ import annotations

from pathlib import Path

from fit_minn_fluxomics_soft_inputs import build_arg_parser, run_with_args


DEFAULT_MINN_LIKE_MODEL_SBML = Path("models/iML1515.xml")
DEFAULT_MINN_LIKE_OUTPUT_CSV = Path("MINN_data/fluxomics_iML1515_minn_like_fit.csv")
DEFAULT_MINN_LIKE_OBJECTIVE_REACTION = "BIOMASS_Ec_iML1515_core_75p37M"


def main() -> int:
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(
        model_sbml=DEFAULT_MINN_LIKE_MODEL_SBML,
        output_csv=DEFAULT_MINN_LIKE_OUTPUT_CSV,
        objective_reaction=DEFAULT_MINN_LIKE_OBJECTIVE_REACTION,
        soft_input_relative_tolerance=None,
        run_label="MINN-like iML1515 fluxomics fitter",
    )
    return run_with_args(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
