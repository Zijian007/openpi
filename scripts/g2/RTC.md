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
  --port 8001 --rtc --rtc-soft-overlap-steps 5 --rtc-guidance 10
```

Warmup runs both ordinary and guided inference before opening the port. Metadata
advertises `rtc_supported`, `rtc_protocol`, `rtc_soft_overlap_steps`,
`rtc_prefix_weighting=delay_plus_soft_overlap`, and `rtc_max_guidance_weight`.
Starting without `--rtc` keeps ordinary inference;
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
for training: G2 input mapping, the configured ActionCodec reference/representation,
checkpoint normalization, and model dimension padding. The default is component
delta against the current measured pose, independently of the model state space.
Grippers remain absolute. Padded time steps and action dimensions have zero
guidance weight. Nothing is subtracted or normalized on the domain controller.

Inside each flow step the endpoint estimate is `x_t - t * velocity(x_t)`. RTC
uses its full vector-Jacobian product to guide that endpoint toward the prefix;
it does not detach the velocity Jacobian. Estimated committed steps use weight
one, followed by `--rtc-soft-overlap-steps` linearly decaying weights. For delay
13 and soft overlap 5, weights are 1 for steps 0–12, then 5/6, 4/6, 3/6, 2/6,
1/6 for steps 13–17, then zero. The overlap width remains 5 when delay changes.
If the prefix ends early, unavailable weights are masked without changing the
decay slope. The response's `rtc_info` reports requested/applied overlap length.
Guidance is capped by `--rtc-guidance`; weight one is not a hard action clamp.

The old absolute `--rtc-horizon` has been removed. Migrate to an explicit soft
width, e.g. 5; copying an old horizon of 18 into the new parameter would change
its meaning. The wire fields and `openpi.rtc.v1` protocol are unchanged.

`INFERENCE_DELAY=0` estimates the delay from the largest of the last five measured
observation-to-inference latencies, rounded up to policy steps plus one step of
margin. A positive value is a minimum bound. Guidance adds computation: measure
latency with RTC enabled. At arrival the actual elapsed delay is computed once,
and only that many new-chunk steps are discarded. The initial chunk retains its
prefix until execution begins. If committed actions or the prediction horizon
expire, inference fails and the runtime applies its existing halt behavior.

Client `replan_interval_steps` independently sets the nominal interval between
observation starts. With chunk 50 and interval 13, refill triggers at 37 remaining
steps; elapsed steps trimmed on arrival count toward the interval. At 30 Hz this
is about 0.433 s. If inference takes longer, request the next chunk immediately
after arrival; there is only one request in flight. The scheduler validates the
interval against the server-resolved chunk size. Null uses `chunk_refill_ratio`;
an explicit interval overrides the ratio. Neither setting changes model horizon.

The wholebody baseline uses interval 13, delay floor 13 and bridge 0.08 s; the
server defaults to soft overlap 5 and guidance 10. These are separate controls,
not a single execution horizon. Do not change `policy_hz` independently of the
checkpoint's action sampling frequency (sample_hz / stride_steps). Logs include prefix/estimated/actual delay plus
`[pi05 RTC guidance]` with applied soft overlap, and warn if it is truncated.

## Tests

In the GPU's OpenPI environment:

```bash
uv run --extra g2 pytest -q src/openpi/models/rtc_test.py \
  src/openpi/models/rtc_sampler_test.py src/openpi/adapters/g2/tests/rtc_policy_test.py
```

These verify the guidance Jacobian and sign, JAX JIT integration through the
Flax bridge, zero-mask parity, current-pose reanchoring, quantile/z-score
normalization, and padding masks without loading a full checkpoint. Then run
the server warmup and client smoke/fake checks using the actual checkpoint.

Algorithm reference: [Physical Intelligence RTC implementation](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py).

Additional control quantities, histories, references, and target offsets use the explicit
`g2.openpi.v1` adapter protocol; see [Policy configuration](POLICY_CONFIG.md).
RTC prefixes always use that control quantity's full wire width and canonical coordinates,
and are encoded without applying training-label time offsets or subsampling again.
