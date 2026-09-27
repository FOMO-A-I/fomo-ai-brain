# Infrastructure

Components for operator-managed deployments:

- `docker/Dockerfile` builds a CPU development image. A GPU image needs a
  CUDA-compatible PyTorch build.
- `gpu/launch.py` checks the checkpoint and CUDA availability before starting
  the API.
- `workers/router.py` routes requests across configured inference workers.
  Configure only trusted endpoints on a private network.
- `postgres/schema.sql` defines storage and ownership tables for a site
  backend. The local API does not use this schema or manage user accounts.
- `monitoring/metrics.py` exports local counters in Prometheus format.

GPU provisioning, distributed scheduling, remote code isolation, production
authentication, autoscaling, and moderation require separate services.