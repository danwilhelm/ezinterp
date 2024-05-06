# ezinterp

Minimalistic small/medium transformer interpretability library for interactive exploration (NumPy-based).
- `EZModel` stores the model weights, and
- `EZRun` stores the intermediate values of a model run.


Easily:
- Easily import any GPT2-style model via `TransformerLens` or PTH files.
- Selectively disable and re-route nearly every aspect of the model.
- Source contains all computations on one screen (ensures same results as `TransformerLens`)
- Load, observe, and modify weights per-run as `ndarrays`.
- Access all intermediate model values as `ndarrays`.
- Pre-computes special values useful for interpretability.

To view available options, introspect one of the objects.


**NOTE:** This is an early, preliminary version. Naming and capabilities are not yet finalized. Currently only works on GPT2-style models (e.g. gpt2, gelu-, ChessGPT)