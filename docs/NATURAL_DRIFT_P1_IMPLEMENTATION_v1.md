# XA-202609 P1 implementation v1

## Scope

This implementation is an analysis and decision layer above the frozen h1/h2
campaign. It does not alter campaign v2, submit hardware work, perform pulse
calibration, reconstruct a complete noise channel, or claim sub-queue control.

## T6: observable environment proxy

Source: append-only collected raw counts from campaign v2. Each record contains
interleaved-reference and readout-derived proxies, two measurement-defined
effective fields with full `2x2` delta covariance, available platform-calibration
metadata, shot-noise floors, provenance, and a previous-snapshot assessment.
The anchor-derived task label uses a disjoint set of task labels. First snapshots
cannot assert drift because they have no predecessor. A drift item is
claim-permitted only when its absolute change divided by combined shot-noise floor
is at least 2.

Artifacts:

- `E:/TianYan/XA-202609/artifacts/analysis/T6_observable_environment_proxy_v1/features_*.jsonl`
- `E:/TianYan/XA-202609/artifacts/analysis/T6_observable_environment_proxy_v1/feature_extraction_report.json`
- `E:/TianYan/XA-202609/artifacts/analysis/T6_observable_environment_proxy_v1/normalization_artifact.json`

Acceptance: proxy and label task IDs have empty intersection; one frozen feature
specification per corpus; output contains none of the prohibited direct-sensor
terms; every claimed change clears the 2x floor rule.

## T7: forecast head

Ridge-Gaussian forecast head evaluates future `delta(h1,h2,readout_zero,readout_one)`
at horizon `H` snapshots with forward-only rolling-origin folds. It compares
persistence, calendar-fixed schedule, local linear extrapolation, and climatology.
It writes Brier comparisons, bootstrap BSS confidence intervals, joint coverage,
NLL, reliability bins, and three leakage checks. It refuses a forecasting claim
unless all three main baselines are beaten, BSS 95% lower CI is positive, ECE is
below `0.1`, distribution metrics are finite, and every leakage check passes.

`H` is a snapshot horizon, not a claimed execution-time horizon. Scheduled slots
support the calendar baseline; client wall-clock fields remain provenance only.
No analysis may infer or fabricate platform execution timestamps.

Artifacts:

- `E:/TianYan/XA-202609/artifacts/analysis/T7_forecast_head_v1/tianyan-287/forecast_report.json`
- `E:/TianYan/XA-202609/artifacts/analysis/T7_forecast_head_v1/tianyan176/forecast_report.json`

Acceptance: `shuffle_used` is false; all training snapshot indices precede all
test indices; all three baselines and leakage files exist. The required corpus is
at least 74 snapshots per backend for three forward-chain folds. Insufficient
corpus is neither a positive nor a negative forecasting result.

## T8: linear contextual bandit and shield

`LinTS` and `LinUCB` use an analytic linear-Gaussian posterior and a 45-action
space. Policy execution requires five true prerequisites: calibrated sigma, an
artifact-bound conservative T3 prior when sim-to-real coverage requires it,
passed twin gate, signed T6 corpus, and passed T7 forecast reports for every
collected backend. The external,
non-learned shield checks five gates in order: confidence/support, h1/h2 range,
Jz rejection, action amplitude, and shot budget. Shielded actions do not update
the posterior. No hardware executor is included.

For multi-backend T6 corpus, preflight requires exactly one T7 report for every
backend in `records_by_backend`; all must pass. Missing, duplicate, or
out-of-scope reports block policy execution.

T6 signs its feature-corpus hash. Each T7 report signs its own contents and
records that same corpus hash. Preflight rejects unsigned T6/T7 reports or a
T7 report built from a different corpus.

Artifact:

- `E:/TianYan/XA-202609/artifacts/analysis/T8_bandit_v1/prerequisite_gate.json`

Acceptance: exactly 45 candidate actions; state feature map at most 20 values;
all unmet prerequisites prevent execution; all five shield gates have tests.

## Outcome branches

- T7 passes for every collected backend: report forecasted calibration-cycle
  drift. T8 remains separately conditional on all signed prerequisites and its
  external shield; this is not a general RL claim.
- T7 reaches the registered corpus but fails any forecast gate: report dual-
  backend drift characterization and safety-response closure only. Use a fixed
  rule scheduler if needed; do not label it forecast-driven, bandit, or RL.
- T6 has no registered observable drift: report that this protocol did not
  resolve drift at registered power. Do not infer a physical null or dispatch
  an adaptive policy.

These branches are fixed before T7 reaches its corpus threshold. They do not
alter frozen campaign v2 collection, source hashes, raw counts, or its audit
chain.

## Analysis refresh and schedule disposition

Each refresh includes only scheduled slots for which every registered backend
has a collected raw-count artifact. The refresh script writes a new T6/T7/T8
directory keyed by complete-pair count and refuses to overwrite prior evidence.
It has no platform client.

The `:00/:25/:50` slot calculation creates an observed ten-minute scheduled
gap at each hour boundary despite the frozen `25` minute nominal configuration.
The analysis-side cadence ledger records this deviation without editing v2.
T7's horizon remains a snapshot-index horizon; client journal times are
provenance only and never platform execution timestamps or evidence of equal
time spacing.

The preregistered no-skill branch is `RuleScheduleConfig`: observed
`p_exceed >= 0.50` selects fixed gain `0.25` and 72 probes, otherwise abstains
with 18 probes. Every selected action still traverses the existing confidence,
range, Jz, amplitude, and budget shield. It has no posterior update, reward,
bandit claim, RL claim, or hardware executor.
