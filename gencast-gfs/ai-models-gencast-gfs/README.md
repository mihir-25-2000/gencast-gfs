# ai-models-gencast-gfs

This repository wires the **`ai-models-gencast`** plugin into the `ai-models-gfs` interface so you can run GenCast on GFS input data. It reuses the upstream GenCast implementation (GraphCast repo) and handles the GFS plumbing.

## Quick start

```bash
git clone <this-repo> ai-models-gencast-gfs
cd ai-models-gencast-gfs
pip install ai-models-gencast  # pulls ai-models and helpers
# Install JAX/GraphCast/GenCast as per the upstream instructions (GPU wheels)
pip install -e .
```

## Assets layout

Keep code and assets separate. Point `AI_MODELS_ASSETS` (or your config) at the shared path:

```
/Datastorage/mihir.more/gencast-gfs/ai_models_assets/gencast/
├── checkpoints/         # GenCast weights go here
├── norm_stats/          # Normalisation stats per variable/level
├── data/                # Preprocessed GFS data (zarr/nc/parquet)
└── logs/                # Runs, evals, metadata
```

Create it now:

```bash
mkdir -p /Datastorage/mihir.more/gencast-gfs/ai_models_assets/gencast/{checkpoints,norm_stats,data,logs}
```

## What’s stubbed vs TODO

- `src/ai_models_gencast_gfs/model.py` exposes `GencastModel` and the entry-point hook expected by `ai-models-gfs`, but the `load_model` and `run` methods are intentionally unimplemented. Replace them with your GenCast loading/inference pipeline.
- `requirements*.txt` are empty placeholders; add whatever GenCast/JAX/PyTorch/TF stack you use.
- `input.py` and `output.py` are minimal helpers ready for you to flesh out with your preprocessing/postprocessing steps.

## Suggested next work items

1. Implement `load_model` to load GenCast weights and initialise the model with your framework (JAX/Haiku/Flax/PyTorch/etc.).
2. Implement `run` to:
   - ingest GFS inputs (surface + pressure level fields) and optional forcings,
   - normalise using stats under `norm_stats/`,
   - call the GenCast predictor,
   - denormalise and write outputs to the expected `ai-models-gfs` format.
3. Add download/setup scripts for weights and stats into `scripts/` (create it if missing).
4. Add smoke tests in `tests/` that mock a forward pass on synthetic data to validate shapes and entry-point wiring.
5. Wire your configs to the asset root above for consistent training/eval runs.
