# XA-202609 natural-drift campaign preregistration v1

## Scope and frozen route

Campaign ID: `xa202609_tianyan_h1h2_dual_backend_natural_drift_v1`.

Hardware route is only TianYan `tianyan-287` on `[62,55,61,68,76,69]` and
`tianyan176` on `[42,36,31,37,44,49]`. Both run frozen V8 two-CZ digital
effective-field task at `h1=0.25`, `h2=-0.35`, times `[0.16,0.31,0.47]`, and
nine Pauli bases. This is a 2-logical-qubit `h1/h2` route carried by six
physical qubits. It is not V2.0's unimplemented `tianyan-294`/six-qubit Neel
route.

Every snapshot/backend has 33 settings x 1024 shots:

- 27 anchor settings: 3 times x 9 Pauli bases.
- Four bit-identical nominal `t=0.31, ZZ` references at fixed positions
  `[0,11,22,32]`.
- `all_zero` and `all_one` readout probes at fixed positions `10` and `21`.

Cadence is 25 minutes. The campaign ends at `2026-09-30T15:59:59+00:00` or
at hard cap `250,000,000` submitted shots, whichever comes first. Raw results
are append-only; a returned `query_id` is durable before any retrieval call.
No unavailable server execution timestamp is fabricated. `wallclock_submit_utc`
and `wallclock_retrieve_utc` are client-observed provenance only.

## Evidence boundary

The campaign measures observable environment proxies, not temperature or EMI:
raw platform calibration metadata when available, recalibration/version fields
when returned, interleaved task-reference change, and all-zero/all-one readout
change. Missing platform fields are stored as `null` with request error, never
imputed.

The immediate empirical question is whether these observable quantities vary
over calibration cycles beyond their shot-noise floor. Existing historical
readout movement motivates observation but is not a substitute for this
prospective campaign. Failure to detect drift means only that this protocol did
not resolve drift at its registered power; it does not imply no physical drift.

Allowed eventual wording: calibration-cycle-scale, digital effective-field and
readout drift sensing/prediction/safe response over a weeks-long evidence
window. Prohibited wording: pulse calibration, full noise-channel
reconstruction, direct temperature/EMI measurement, microsecond real time,
months-long stability, deep RL, or universal AI superiority over classical
estimators.

## Primary drift endpoint

For anchor snapshot `s`, pre-specified constrained NLS produces
`theta_s=(h1_s,h2_s)`. The primary scalar is CRB-whitened displacement from the
registered reference estimate:

`D_s^2 = (theta_s-theta_ref)^T Sigma_ref^{-1} (theta_s-theta_ref)`.

`Sigma_ref` is complete `inverse_fim` from frozen E4 Step 3 matching the
selected Pauli/time/shot design. It must be retained as its full matrix, not
only diagonal CRB entries. E4 regeneration output is
`E:/TianYan/XA-202609/artifacts/analysis/E4_identifiability_ceiling/fim_scan.json`.
If that design is rank deficient or no matching complete matrix exists, this
primary endpoint is reported unavailable; no diagonal substitute is allowed.

Readout all-zero fidelity, all-one fidelity, all four reference estimates, and
each raw Pauli observable are registered secondary endpoints. Their role is to
separate task-level drift, readout-layer movement, and within-batch queue/order
variation; they do not identify a physical noise channel.

A natural-drift signal can be called only after multiplicity adjustment when
estimated magnitude divided by its registered shot-noise floor is greater than
`2.0`. All effects and CIs are reported even when this gate fails.

## Design power

Before first hardware submission, `power_report.json` is produced by
`scripts/drift_campaign.py power`. It evaluates target change magnitudes
`0.01`, `0.02`, and `0.05`, using binomial shot variance, Bonferroni family
control (`15` Pauli; `2` readout), and registered AR(1) correlation `rho=0.3`.
It reports required independent-pair equivalents and MDD, rather than testing a
fixed 0.05 threshold under unresolved 0.17-scale noise.

At analysis lock, empirical variance and serial correlation replace conservative
planning bounds but neither cadence, endpoints, family definitions, nor the
two-times-floor claim gate can change. The report is a design calculation, not
a hardware effect estimate.

## Future-time forecast test

Forecast work starts only after enough prospective snapshots exist. Prediction
target is future `D_s` risk or registered secondary drift risk at horizon
`H in {1,3}` snapshots. Splits are rolling-origin only; test times follow train
times with a gap of at least `H`. No shuffled folds or future calibration data
may enter features.

Required baselines are persistence, calendar-fixed recalibration schedule, and
recent linear extrapolation. Forecast claim needs all of: Brier skill over
persistence greater than zero, bootstrap 95% lower CI greater than zero, and
ECE below `0.1`. Coverage, NLL, and reliability curves are reported. Otherwise
label result `no forecasting skill at current corpus length`.

## Future safe-control comparison

No bandit or automatic compensation runs under this v1 collector. After its
separate P1 gate, primary strategy comparison is LinTS/LinUCB under external
five-gate shield versus fixed recalibration strategy at equal shot budget.
Shield always enforces high-confidence admission, physical range, permanent
`Jz` rejection, action-amplitude limit, and budget limit. Any shield violation
is an implementation failure and stops policy deployment.

Primary task metric is paired mirror-circuit success rate from raw counts.
Readout-mitigated result is secondary only. The protocol does not turn injected
offset recovery into evidence of natural drift; injections remain a separate
controlled-response validation.

## Stops, failures, and audit

- Stop new submissions at campaign deadline or before the hard shot cap.
- Stop automatic new submission after a client-side unresolved submission;
  reconcile returned query IDs manually before retrying to avoid duplicate work.
- Retrieval retries five times with 30, 60, 120, 240-second backoff. Partial
  response remains `partial` with missing query IDs, not `failed`.
- Three consecutive failed snapshots create a durable alert but do not erase
  prior raw evidence.
- No efficacy or futility early stopping. Collection continues until a registered
  operational stop, then every missing setting and retry history is reported.

Frozen source/config/document hashes, probe-manifest hashes, JSONL hash chain,
query IDs, raw results, and materialized count arrays form audit evidence. This
repository currently has no pre-existing Git history; a local P0 commit and the
SHA-256 campaign manifest are the freeze evidence. A later semantic change
creates v2 and cannot mix its corpus with v1.
