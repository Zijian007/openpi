# OpenPI G2 inference-time RTC

This uses the existing Pi05 checkpoint and its saved normalization statistics.
No training configuration or checkpoint weights change. Both server and client
must be updated; the opt-in protocol extension is `openpi.rtc.v1`.

## GPU server

Use the same config and step directory as ordinary inference:

```bash
bash scripts/g2/deploy/serve_policy.sh \
  --config-name pi05_g2_vr_low_mem \
  --policy-dir /path/to/existing/checkpoint/step \
  --port 8001 --rtc --rtc-horizon 10 --rtc-guidance 10
```

Warmup runs both ordinary and guided inference before opening the port. Metadata
advertises `rtc_supported`, `rtc_protocol`, `rtc_execution_horizon`, and
`rtc_max_guidance_weight`. Starting without `--rtc` keeps ordinary inference;
clients that request RTC will fail explicitly rather than silently ignoring it.

## wholebody client

```bash
cd /home/agi/wholebody
MODE=smoke RTC=true bash launcher/real/start_openpi_pi05_infer.sh
RTC=true INFERENCE_DELAY=0 bash launcher/real/start_openpi_pi05_infer.sh \
  --profile fake --max-ticks 100
RTC=true INFERENCE_DELAY=0 bash launcher/real/start_openpi_pi05_infer.sh \
  --dry-run true --force-control no
```

Smoke checks connection/ordinary first-chunk inference. Fake uses synthetic
images; neither establishes task success. Real dry-run still opens hardware
sessions and runs the configured startup mode gate; it skips HAL publishing.

## Prefix and timing contract

The client snapshots the planned trajectory at the start of observation
acquisition, including sampler-held knots, fractional interpolation, and any
active bridge. Prefix knots are absolute EE20 at `start + n / policy_hz`.

The server inserts the prefix as `actions` into the same input transforms used
for training: G2 input mapping, subtraction of the **current measured pose** for
delta-enabled dimensions, checkpoint normalization, and model dimension padding.
Grippers remain absolute. Padded time steps and action dimensions have zero
guidance weight. Nothing is subtracted or normalized on the domain controller.

Inside each flow step the endpoint estimate is `x_t - t * velocity(x_t)`. RTC
uses its full vector-Jacobian product to guide that endpoint toward the prefix;
it does not detach the velocity Jacobian. Committed steps use weight one, then
a linear soft overlap decays to zero. Guidance is capped by `--rtc-guidance`.
`--rtc-horizon` is the soft overlap end in policy steps; a longer committed delay
extends the fully weighted prefix beyond this setting.

`INFERENCE_DELAY=0` estimates the delay from the largest of the last five measured
observation-to-inference latencies, rounded up to policy steps plus one step of
margin. A positive value is a minimum bound. Guidance adds computation: measure
latency with RTC enabled. At arrival the actual elapsed delay is computed once,
and only that many new-chunk steps are discarded. The initial chunk retains its
prefix until execution begins. If committed actions or the prediction horizon
expire, inference fails and the runtime applies its existing halt behavior.

At 30 Hz, horizon 50 and refill ratio 0.5 leave about 0.83 s to produce the next
chunk. Increase `chunk_refill_ratio` if measured guided inference approaches this
buffer; do not change `policy_hz` independently of the checkpoint's training FPS.
Logs include `[pi05 RTC] prefix_steps=... estimated_delay=... real_delay=...`.

## Tests

In the GPU's OpenPI environment:

```bash
uv run pytest -q src/openpi/models/rtc_test.py \
  src/openpi/models/rtc_sampler_test.py src/openpi/policies/rtc_policy_test.py
```

These verify the guidance Jacobian and sign, JAX JIT integration through the
Flax bridge, zero-mask parity, current-pose reanchoring, quantile/z-score
normalization, and padding masks without loading a full checkpoint. Then run
the server warmup and client smoke/fake checks using the actual checkpoint.

Algorithm reference: [Physical Intelligence RTC implementation](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py).
