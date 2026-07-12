# Yeast9 Experiment Notes

This file tracks the active Yeast9 FluxTransformer evaluation and training
efficiency experiments.

## Active Notebook

- Main evaluation notebook: `yeast9_model_testing.ipynb`.
- Active checkpoint:
  `./models/yeast9_d256_h8_l3_ff1024/yeast9_d256_h8_l3_ff1024_checkpoint.pth`.
- Training dataset recorded in checkpoint metadata:
  `./data/2025-11-07_yeast9_data_246923_samples.csv`.
- Test file used by the notebook:
  `./data/yeast9_test_data.csv`.

The Yeast9 checkpoint was trained with the older FluxTransformer implementation.
That model evaluates all tokens directly and does not use injected input-token
indices. The notebook therefore defines the older non-injected
`FluxTransformer` class locally and should not import the current
`flux_transformer.py` implementation for this checkpoint.

Do not "fix" the Yeast9 notebook by switching it to the current injected-token
FluxTransformer API unless the Yeast9 checkpoint is retrained with that API.

## Model Scale

- Yeast9 uses 64 input/exchange features in the notebook data.
- The checkpoint has `vocab_size = 4195` and `n_inputs = 64`.
- The output reaction set contains 4,131 flux targets.
- The saved model has about 3.45 million trainable parameters.

This experiment is mainly a scaling test: it asks whether FluxTransformer can
still approximate simulated FBA-like flux behavior for a larger eukaryotic
genome-scale reconstruction.

## Flux Prediction Diagnostics

The main diagnostic fluxes are:

- `r_2111`: Yeast9 biomass.
- `r_1110`: ADP/ATP transporter.
- `r_2100`: water exchange.
- `r_1672`: carbon dioxide exchange.

Current pooled metrics across all output flux values:

- pooled `R^2`: 0.9116
- pooled MAE: 0.064661
- pooled RMSE: 1.002208
- active-output count: 2,327 / 4,131

## Difficult Flux Diagnostics

The notebook includes reaction-level diagnostics for the full output set:

- per-flux `R^2`
- per-flux MAE and RMSE
- true-value standard deviation
- zero fraction
- 95th percentile absolute error
- RMSE/MAE ratio
- share of total squared error

The notebook displays two compact tables:

- largest contributors to total squared error
- lowest active-flux `R^2` values

It also automatically selects and plots four difficult fluxes, excluding the
already discussed biomass and high-flux diagnostic examples. These plots are
intended to show that Yeast9 performance is uneven across the full reaction set,
even when biomass and several major fluxes are predicted accurately.

Current selected difficult examples:

- `r_0491`: broad true range, compressed predictions.
- `r_0770`: broad true range, compressed predictions.
- `r_0649`: sparse low-variance flux with small absolute errors but unstable
  `R^2`.
- `r_2274`: sparse low-variance flux with small absolute errors but unstable
  `R^2`.

## Training Efficiency Experiments

Yeast9 is also used for output-subset training experiments because full-output
training is expensive for a large reaction vocabulary.

Relevant files include:

- `yeast9_rs_transformer.py`
- `yeast9_rs_transformer_corr.py`
- `yeast9_rs_transformer_sample.py`
- `yeast9_rs_transformer_sample2.py`
- `yeast9_explore_samples.ipynb`

Current interpretation:

- Full-output training gives the strongest direct objective but is slow.
- Very low output-subset ratios lose too much reaction coverage.
- Uniform output sampling is a simple baseline but is not strong enough for this
  model.
- Flux-magnitude weighting improves only modestly.
- Correlation-based output sampling is the most promising tested subset strategy.
- Output-subset training should be described as a speed-accuracy tradeoff, not
  as a free improvement.

## Reporting Guidance

For Yeast9 flux prediction results, keep the section concise:

- report biomass performance;
- mention a few high-flux diagnostic examples;
- report pooled full-output metrics;
- add the caveat that reaction-level accuracy is uneven;
- use difficult-flux diagnostics to support that caveat.

Do not repeat the full E. coli core embedding/pathway analysis for Yeast9 unless
new pathway-level Yeast9 embedding diagnostics are added.
