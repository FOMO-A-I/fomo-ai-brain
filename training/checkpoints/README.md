# Checkpoint storage

No trained weights are included in this repository. Put model artifacts on a
secured GPU machine or private model store. Real training runs write their
own `.fomo-checkpoint.json` integrity manifest after completion. Register a
completed run with `python -m training.model_registry add --checkpoint PATH`.

The registry, model weights and datasets are ignored by Git. A manifest records
training metadata supplied by the operator and hashes files; it is **not**
independent proof that data quality, model licensing, or safety was reviewed.