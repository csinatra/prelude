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

## Why `runc` rather than mle-bench's `sysbox-runc`

Upstream runs agents under Sysbox. Sysbox cannot reach a GPU on this box, and
the reason is structural rather than a misconfiguration (diagnosed 2026-08-24,
NVIDIA Container Toolkit 1.18.1, Docker 29.2.1, driver 580.105.08):

- Sysbox remaps user namespaces, so NVIDIA's OCI hooks cannot read containerd's
  state directory — every GPU container dies at `Running hook #0 … failed to
  open OCI spec file … permission denied`.
- CDI does not route around it: the `nvidia-cdi-hook` entries fail the same way.
  Specs live in **both** `/etc/cdi` and `/var/run/cdi`, and the latter is the
  live one.
- With the hooks stripped the container starts, but `update-ldcache` and
  `create-symlinks` are exactly what made the injected libraries resolvable, so
  `nvidia-smi` cannot find `libnvidia-ml.so.1`.
- Under `runc`, CDI and `--gpus` both work unmodified.

Keeping Sysbox would mean replicating NVIDIA's hooks ourselves — an entrypoint
wrapper running `ldconfig` and synthesizing versioned symlinks, plus keeping two
CDI specs stripped across reboots (`/var/run/cdi` is tmpfs) and driver updates.
That is bespoke infrastructure standing in for the vendor's own tooling, with
several silent-failure modes during an eval run.

What Sysbox provides is unprivileged Docker-in-Docker and userns isolation
against container escape. AIDE writes training scripts; it does not spawn
containers. The isolation guards a threat model — untrusted submissions on a
shared harness — that does not apply to our own agent on a disposable instance.

`agents/run.py` branches artifact extraction on the runtime and handles both, so
this is a supported configuration upstream, not a workaround. It is still a
deviation from the reference harness and is reported as one.
