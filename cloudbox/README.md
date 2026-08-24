# Cloud-box container configuration

`container_config.json` is passed to `run_agent.py --container-config`, which
forwards it verbatim to `docker.containers.create`. It replaces mle-bench's
`environment/config/container_configs/default.json` — a bare Docker default
rather than the benchmark's stated resource baseline.

MLE-bench's README asks runs to report any deviation from its reference
resources: **24-hour runtime, 36 vCPUs, 440GB RAM, one 24GB A10 GPU.** Against
that, the upstream default gives an agent 4 vCPUs and no GPU at all — a gap far
larger than any difference between cloud instances, and not a deviation anyone
chose.

| | Reference | Upstream default | Here |
|---|---|---|---|
| GPU | one 24GB A10 | none | all host GPUs (`Count: -1`) |
| vCPUs | 36 | 4 | 30 |
| RAM | 440GB | unlimited | unlimited (host-supplied) |
| Runtime | 24h | — | set per run, below 24h |

Lambda's `gpu_1x_a10` is the reference card with 30 vCPUs, so `nano_cpus` is
pinned to the tier rather than left uncapped — runs stay comparable if a
different tier is ever used, and Docker rejects a value above host capacity, so
a tier change fails loudly instead of silently. **Re-check this value if the
instance type changes.**

Reported alongside results as deviations from the reference: fewer vCPUs and
less RAM than the reference host, and a shorter runtime (budget-constrained).

## GPU under Sysbox is unverified

mle-bench runs agents under the `sysbox-runc` runtime, and Sysbox has not
historically coexisted with NVIDIA's container runtime. The `device_requests`
above are therefore a hypothesis until a run confirms a container actually sees
a GPU.

If it cannot, the choice is between Sysbox's isolation and GPU training. That is
a real trade — Sysbox is what lets the agent run Docker-in-Docker safely — and
the decision belongs in DECISIONS.md rather than in a silent config edit.
