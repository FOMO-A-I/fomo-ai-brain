# Deployment building blocks

These files target a machine or cluster **you operate**. They do not provision
GPUs, provision a production database, deploy a model, or train a checkpoint.

- `docker/Dockerfile` builds a CPU development image. Replace its PyTorch wheel
  with a CUDA-compatible one when preparing a GPU image.
- `gpu/launch.py` validates the actual checkpoint and CUDA availability before
  starting the API process. Its checks fail rather than loading an unrelated model.
- `workers/router.py` is a bounded HTTP worker pool for configured inference
  backends; use only trusted endpoints on a private network.
- `postgres/schema.sql` documents the server-side storage and ownership schema
  for a future multi-user site. The reference API does **not** automatically
  switch to PostgreSQL or provide user accounts.
- `monitoring/metrics.py` exports local counters in Prometheus text format.

**Not yet provided:** distributed GPU scheduler, secure remote code execution,
production authentication, GPU autoscaling, a deployed moderation model, and a
trained FOMO checkpoint. These need datasets, infrastructure, access controls,
and load/safety testing; they must not be simulated with placeholders.