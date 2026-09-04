# B4/B5 ring-four handoff — state as of 2026-08-23, and the T176 migration

**Audience:** Codex, picking up the XA-202609 ring-four (B-5 cadence-pair injected live closed loop)
collection line.
**Scope:** everything that happened between 2026-08-15 and 2026-08-23, plus the decision that is now
blocking collection.
**Companion document:** `PROJECT_NOTES.md` (commit `a59d24b`) covers the account migration and the
AEMTN model/paper line. This document covers the competition collection line and does not repeat it.
Read both.

**HEAD at time of writing:** `a59d24b`. Working tree carries 175 dirty/untracked paths — see
§7 *Repository hygiene* before committing anything.

---

## 1. Where the collection stands

The registered endpoint is the **cadence-ratio permutation gate** applied to paired per-cycle
endpoint residual squared, frozen in `config/b4_cadence_pair_loop_cycle_paired_v4.json` under
`collection_correction`. The design point:

| quantity | value |
|---|---|
| sessions | 2 (one per day, `operational_inter_session_seconds = 86400`) |
| cycles per cadence per session | 20 |
| registered cycle pairs total | 40 (minimum adjudicated: 30) |
| settings per session | 88 = 80 sensing + 4 baseline + 4 mirror QC |
| jobs per session | 44 = 40 sensing + 2 baseline + 2 mirror QC |
| shots per session | 621 920 (total 1 243 840) |
| sensing shots/setting | 6 186 |
| baseline shots/setting | 27 664 |
| `endpoint_shot_floor_total` | 4.693874996564006e-04 |
| `expected_ratio` | 0.5066932197965022 |
| `preregistered_ratio_prediction` | 0.5070213621785998, interval [0.3260557361133296, 0.7884314361673062] |
| `expected_power` | 0.90155 (0.8046 at 30 pairs) |
| `measured_boundary_size` | 0.048075 |
| `modelled_busy_seconds_per_session` | 1184.967318298686 |
| `machine_time_ceiling_seconds` | 2400.0 (operator constraint: 40 minutes total) |
| `daily_window_seconds` | 1200.0 (platform, **enforced in code**) |

**No hardware record exists under v2, v3 or v4.** The amendment scope is pre-collection only. The
gate module, the registered endpoint, the cadence pair (90 s / 360 s), the OU injection law and the
exclusion rules are unchanged from v2.

**The blocker:** T287 has been in maintenance since before 2026-08-17 and is still unavailable on
2026-08-23, 23 days from the 2026-09-15 deadline. The platform has allocated T176 time instead. The
collection must move to T176 or not happen.

---

## 2. What changed, 2026-08-15 → 2026-08-23

Six commits, 29 files.

| commit | date | subject |
|---|---|---|
| `86e7d20` | 08-15 | feat: pre-register the v4 forty-minute B-5 cadence-pair collection |
| `385c72f` | 08-15 | feat: probe T287 acceptance and delivery of the v4 baseline shot level |
| `a695a6d` | 08-17 | feat: measure the mirror task metric's residual-field sensitivity on hardware |
| `8a056cb` | 08-17 | feat(b4): pre-register and build the residual-amplified closed-loop probe |
| `60e47f0` | 08-17 | docs(b4): report the amplified closed-loop result and fold it into the v4 amendment |
| `a59d24b` | 08-23 | docs: add PROJECT_NOTES.md for Codex account migration |

### 2.1 The v3 → v4 amendment (`86e7d20`)

`docs/B4_B5_COLLECTION_AMENDMENT_v3_TO_v4_20260815.md`.

v3 needed twelve sessions and 3.8 hours of busy time to reach eighty registered pairs. The operator
constraint is forty minutes total. v4 bought the pair count back by **recalibrating the critical
value instead of adding pairs**, and by amortising the zero-field baseline over the session.

Two structural changes, both in `src/adaptive/`:

- **`cadence_permutation.py`** — the primary adjudication became an exact within-pair label-swap
  permutation test. The point statistic is byte-identical to the frozen `cadence_ratio_gate` ratio
  (asserted in tests); only the critical value is recalibrated. The frozen delta-method interval is
  retained as a reported secondary and is **not** used to decide the endpoint, because it is
  anti-conservative on the exponential endpoint law at these pair counts (measured boundary size
  0.0878 at 16 pairs, 0.0723 at 24, 0.0524 at 56 — it cannot hold its own size criterion anywhere
  inside the forty-minute budget).
- **`shared_baseline_sensing.py`** — the zero-field baseline moved from four settings per cycle to
  two measurements per session shared by every cycle. Sensing settings per cycle halve (4 → 2), and
  the per-cycle shot floor halves from `4.746/S` to `2.373/S` plus a shared `2.373/S_baseline` term.
  The shared baseline must span **both** cadence blocks (a per-block baseline would put two
  independent offsets into the fast and slow arms and destroy within-pair exchangeability). The
  surviving offset is common-additive, which pushes the ratio *up* toward the null of 1.0 — it costs
  power and cannot manufacture a pass. Drift is measured (`baseline_drift_qc`) and converted into a
  pre-registered three-shape sensitivity re-run (`drift_sensitivity_offsets`), not assumed away.

`freeze_b4_cadence_v4_start_only_baseline.py` is the script that computed and wrote the
`collection_correction` block. Reachability evidence:
`E:\TianYan\XA-202609\artifacts\analysis\B4_CADENCE_V4_PREREGISTRATION_20260815\forty_minute_prediction_start_only_baseline.json`,
sha256 `98d4c9b923eedb14b35882086401c21ad6ecb40e4b470fe57805e5dc39f18488`.

### 2.2 The three hardware probes (`385c72f`, `a695a6d`, `8a056cb`)

All three are capability probes. None contributes to the registered endpoint. All carry
`--confirm-hardware` as `required=True` and refuse to overwrite existing artifacts.

- **`probe_t287_baseline_shot_ceiling.py`** — does the platform accept and deliver 27 664
  shots/setting without truncation.
- **`probe_mirror_field_sensitivity.py`** — measures the mirror task metric's response to a known
  residual field. **This refuted the configured mirror sensitivity model** (see §6, open item).
- **`probe_amplified_closed_loop.py`** — the residual-amplified closed loop. This is the one that
  matters (§3).

### 2.3 The T176 amplified closed-loop result (`60e47f0`)

Run 2026-08-17 on T176 because T287 was already down. Artifacts under the quarantine tree:
`E:\TianYan\XA-202609\quarantine\tianyan176\B4_TB5_amplified_closed_loop_20260817_run2\`, marked
`pooling_permitted: false`, `not_main_endpoint: true`, `registered_endpoint_contribution: "none"`.

Headline numbers, all from `amplified_closed_loop_report.json`:

- `measured_suppression_factor` = **7.084** (injected squared norm 1.953e-03 → sensed residual
  squared 2.766e-04)
- `adaptive_fit_over_registered_floor` = **0.601** — inside the pre-registered [0.25, 4.0] band
- `fixed_fit_over_declared_injection` = **1.024** — the independent second modality agrees with the
  declared injection to 2.4%
- `all_shots_delivered_in_full` = true at 27 664 shots/setting on T176

This is the result Murphy remembers as optimistic. It is. It says the shield's digital inverse
compensation really does shrink a real injected field on real hardware, read out through the mirror
task metric, and that the registered shot levels deliver on T176.

**Amplification disclosure (do not lose this):** every reported gap was produced with the residual
scaled by an amplification factor N; measurement gain is N². The unamplified gap at the registered
amplitude is below shot noise at any affordable shot count. The shield's frozen
`max_action_amplitude` of 0.1 caps the residual at ‖r‖ = 0.1414; the mirror's one-sigma detection
threshold at 16 384 shots is ‖r‖ = 0.179; closing that gap with shots alone needs 9.4e8 of them.
Amplification is the only lever, and the artifact says so in `why_amplification_is_the_only_lever`.

---

## 3. The T176 migration audit — statistics

**Verdict: the registered statistics transfer. Nothing needs re-registration on the statistical side.**

The one quantity that could plausibly have been backend-specific is
`PER_CONDITION_FLOOR_CONSTANT = 2.373` in `src/adaptive/shared_baseline_sensing.py:79`. It is
*empirically calibrated*, not derived: `:74-79` records F(S) = 4.746/S from two r2 hardware arms
(S = 22 050 gave F·S = 4.7552; S = 88 200 gave F·S = 4.7364, agreeing to 0.4% across a 4× shot
change). Theory gives F·S = 1/T² = 4.5269 at T = 0.47, so the calibrated value exceeds the
ideal-contrast value by 4.84%, which looks like a readout-contrast deficit of the qubit set it was
measured on.

**It was measured directly on T176 on 2026-08-17, at the registered shot counts.** The amplified
closed-loop probe ran the registered estimator with `injected_shots_per_setting = 6186` and
`baseline_shots_per_setting = 27664` — identical to the registered design — and recorded
`sensing.shot_sigmas = [0.015421466183332873, 0.015158230953373642]`.

These sigmas are **not circular**: `shared_baseline_sensing.differential_estimate:220-222` computes
them purely from the delta-method `phase_variance` of the observed Y/Z expectations
(`phase_from_expectations:100-109`). `PER_CONDITION_FLOOR_CONSTANT` does not enter.

| quantity | T287 (registered) | T176 (measured 2026-08-17) |
|---|---|---|
| C implied by the injected term | 2.373 | 2.3655 |
| C implied by the baseline term | 2.373 | 2.3569 |
| combined | 2.373 | **2.3639** |
| Σσ² over the two controlled fields | 4.693875e-04 | **4.675936e-04** |
| ratio measured / registered | — | **0.99618** |
| excess over ideal 1/T² | +4.84% | **+4.44%** |

Two conclusions:

1. **The constant transfers within 0.4%**, which is the same agreement the original two-arm
   calibration achieved across a 4× shot change. The T176 value sits *below* the registered floor,
   so the registered floor is very slightly pessimistic — the safe direction. `expected_ratio`,
   `preregistered_ratio_prediction`, the interval centre, `expected_power` and
   `measured_boundary_size` all stand unchanged.
2. **The 4.8% excess over ideal contrast is not a T287 fingerprint.** It appears on both backends at
   nearly the same size. It is a property of the estimator / observed Bloch radius, not of the T287
   qubit set. Any argument that says "2.373 carries T287's readout contrast, therefore re-measure"
   is refuted by this table.

**Caveat, stated honestly:** this is one session, one qubit pair, one measurement. It is a direct
measurement at the right shot counts, cross-checked by an independent second modality, and it lands
on the safe side — but it is n = 1. If T176 time is available before collection, re-running the
two-setting baseline job alone (2 settings, ~52 s) would confirm it at negligible cost.

Everything else on the statistical side is software: the injection law, the cadence pair, the
permutation calibration, the drift sensitivity shapes. Adjudication purity is confirmed in code —
`src/adaptive/cadence_permutation.py:242-243` consumes only `fast_endpoint_squared_residual` and
`slow_endpoint_squared_residual`; there is no mirror model and no `p0` anywhere in the module's 249
lines.

---

## 4. The T176 migration audit — timing. **This is the blocker.**

### 4.1 The economics inverted

| | T287 | T176 | ratio |
|---|---|---|---|
| effective shots/second | 1 486.1 | 3 792.6 | **2.55× faster** |
| fixed overhead per job (2 settings) | 13.9802 s | 25.9038 s | **1.85× more expensive** |
| fixed overhead per setting | 6.9901 s | 12.9519 s | 1.85× |

T176 is much faster per shot and much more expensive per job. **The v4 design is
overhead-dominated** — 1 139.8 s of a 1 303.7 s session is overhead — so the swap is strictly bad.
There is no shot-count reduction that fixes it, because shots are only 164 s of the session.

### 4.2 The daily window is enforced, not decorative

`scripts/run_b4_cadence_pair_hardware.py:698-707` raises:

```
ValueError(f"corrected collection exceeds the {daily_window:.0f}-second daily busy-time budget")
```

`daily_budget_passed` is computed at `:311-318` as
`all(estimated_busy_seconds <= daily_window_seconds for each session row)`. The window is 1 200 s.
A naive backend swap does not degrade gracefully; it raises before submitting anything.

### 4.3 A trap in the runner: rate is overridable, overhead is not

`run_b4_cadence_pair_hardware.py:262-264`:

```python
overhead = float(timing["fixed_overhead_seconds_per_setting"])   # from the BACKEND config
rate = float(correction["shot_rate_per_second_used"]) if ... else measured_rate   # from the CORRECTION
```

The correction pins `shot_rate_per_second_used = 1091.4`, which is the **T287-derated worst
pairing**. So pointing the runner at `config/b4_drift_campaign_v4_tianyan176.json` without also
re-freezing the correction gives:

```
88 × 12.9519 + 621920 / 1091.4 = 1139.77 + 569.84 = 1709.60 s/session   (3419.2 s total)
```

against a 2 400 s ceiling — far worse than the honest T176 estimate. **Anyone who swaps the backend
config and nothing else will see a wildly pessimistic number and may draw the wrong conclusion.**
The correction must be re-frozen for T176 before any timing figure means anything.

### 4.4 New finding: the per-setting overhead model is structurally wrong

Every T-B6 throughput measurement was taken at `settings = 2`
(`E:\TianYan\XA-202609\artifacts\hardware\B4_TB6\throughput.json`, all eight entries). So
`fixed_overhead_seconds_per_setting` is simply `fixed_overhead_seconds_per_job / 2`, and **the split
between per-job and per-setting cost was never identified by that experiment.** The v4 config admits
this in `seconds_per_job_note` and sidesteps it by giving every job exactly two settings, so
`settings × 6.9901 == jobs × 13.9802` for the registered design.

The 2026-08-17 run submitted a **20-setting job** on T176 — the first job in the project with
settings ≠ 2. It breaks the degeneracy, and it kills the per-setting model:

```
20 settings × 12.9519 s/setting = 259.0 s of modelled busy time
observed roundtrip (which INCLUDES queue wait) = 133.4 s
```

Modelled busy time exceeds wall-clock roundtrip. **The per-setting marginal cost cannot be 12.95 s
on T176.** Fitting `t = A + B·settings + shots/R` to the three run-2 jobs gives

```
R = 4157.4 shots/s,   A = 36.86 s/job,   B = 0.887 s/setting
```

(A is roundtrip-derived and therefore an upper bound — it contains queue wait.)

**What this does and does not change.** It does *not* change the registered design's number, because
every job there carries exactly two settings and both readings agree at 1 139.8 s/session. It *does*
mean the only real time lever would be packing more settings per job — and the cadence forbids it.
The 40 sensing jobs are one per cycle, locked to the 90 s / 360 s cadence ticks, and the shield needs
each cycle's estimate online before the next. Jobs are irreducible at `2·cycles + 2 + mirror_qc`.

Record this in the config so nobody re-derives it: **`fixed_overhead_seconds_per_setting` is only
valid at `settings = 2`; the T-B6 experiment cannot identify the per-job / per-setting split.**

### 4.5 What actually fits

`busy = jobs × 25.9038 + shots / R`, jobs = `2·cycles + 2 + mirror_qc_jobs`, window 1 200 s/session.

| cycles/cadence | pairs | mirror QC | jobs | shots | R = 3792.6 | R = 2340.4 (T-B6 worst) |
|---|---|---|---|---|---|---|
| 20 | **40 (registered)** | 2 | 44 | 621 920 | 1303.7 ✗ | 1405.5 ✗ |
| 19 | 38 | 2 | 42 | 597 176 | 1245.4 ✗ | 1343.1 ✗ |
| **18** | **36** | **2** | **40** | **572 432** | **1187.1 ✓** | 1280.7 ✗ |
| 17 | 34 | 2 | 38 | 547 688 | 1128.8 ✓ | 1218.4 ✗ |
| 16 | 32 | 2 | 36 | 522 944 | 1070.4 ✓ | 1156.0 ✓ |
| 19 | 38 | 0 | 40 | 580 792 | 1189.3 ✓ | 1284.3 ✗ |
| 17 | 34 | 0 | 36 | 531 304 | 1072.6 ✓ | 1159.5 ✓ |
| 16 | 32 | 0 | 34 | 506 560 | 1014.3 ✓ | 1097.2 ✓ |

Dropping the mirror QC jobs is legitimate on endpoint grounds — `mirror_subset_note` already records
that the matched mirror hardware job "contributes no term to that endpoint", so restricting or
removing it "costs zero endpoint information". It costs the hardware task-metric readout, which is
the descriptive link the ring-four story leans on. That is a presentation trade, not a validity one.

**The answer depends entirely on which rate you believe, and the spread is 32 to 38 pairs.** Both
rates are roundtrip-derived and therefore contaminated by queue wait. Resolving the rate is worth
6 registered pairs.

### 4.6 The unexplained order-of-magnitude gap

The platform ledger
(`E:\TianYan\XA-202609\artifacts\analysis\B4_CADENCE_PLATFORM_TASK_TIME_LEDGER_20260815_r1\platform_task_time_ledger.json`,
420 T287 cadence tasks, `runtime_source: "finishTime-runStartTime"`, which **excludes** queue) shows
median true runtime of **37.6 s/task**, with modes at 37–38 s (304 entries) and 54 s (68 entries).
Each "setting" is a separate `experimentTaskId`, so that is 37.6 s **per setting**.

The frozen model charges 6.99 s/setting on T287. If the platform bills on runtime rather than on the
frozen model, **the T287 design never fit 1 200 s/session either** — 88 × 37.6 = 3 309 s. That is a
more serious problem than the backend swap, and it is unresolved.

This is the single highest-value open question and it costs zero machine time to answer (§5.1).

---

## 5. Ordered instructions for Codex

### 5.1 FIRST — settle the billing basis. Zero machine time. Blocks everything else.

Murphy must export the platform task-detail responses for the 24 T176 query_ids from the 2026-08-17
run, using the same manual browser-DevTools method already used twice for the T287 ledgers
(`retrieval_method: "manual browser DevTools platform-response export"` — there is no API path;
do **not** attempt to bypass the Akamai bot challenge, that is a standing decision).

The ids are already collected for him at **`C:\Users\Mercu\t176_20260817_query_ids.json`**, grouped
by job: BASELINE 2 × 27 664 shots, SENSE 2 × 6 186, MIRROR 20 × 16 384.

Extend `scripts/build_b4_platform_time_ledger.py` to ingest a T176 export and answer three things:

1. **Per-setting true runtime on T176** (`finishTime - runStartTime`), regressed against the known
   shot counts. This gives the queue-free rate and the queue-free overhead, replacing both
   contaminated T-B6 readings.
2. **Do the 20 mirror settings serialise or batch?** Sum their runtimes and compare against the
   job's 133.4 s roundtrip. If the sum greatly exceeds it, settings run in parallel and the
   per-setting cost model is wrong in a second, more interesting way.
3. **Does quota decrement track runtime or the frozen model?** Reconcile against the decrement
   Murphy observed after the 2026-08-17 run. This is what decides whether §4.6 is a real problem or
   an artifact of confusing runtime with billing.

Then re-run the §4.5 table with the queue-free numbers and pick the design point.

### 5.2 SECOND — re-freeze the correction for T176, without re-blessing the statistics

`scripts/freeze_b4_cadence_v4_start_only_baseline.py` **cannot currently be pointed at T176 at all.**
It takes no backend-config argument and hardcodes:

```
:26   CONFIG_PATH   = config/b4_cadence_pair_loop_cycle_paired_v4.json
:48   SHOT_RATE     = 1091.4
:54   SECONDS_PER_SETTING = 6.990118313333854
:59   FLOOR_CONSTANT = 2.373
:68   POWER          = 0.90155
:69   BOUNDARY_SIZE  = 0.048075
:89   PREDICTION     = {...}
:132  busy = shots / SHOT_RATE + jobs * SECONDS_PER_JOB
:558  raise SystemExit(f"design exceeds the machine-time ceiling: ...")
:572  CONFIG_PATH.write_text(...)      # rewrites the ENTIRE collection_correction block
```

**The hazard:** editing only `:48` and `:54` for T176 rewrites the timing fields correctly *and
silently rewrites the statistical fields too* — `endpoint_shot_floor_total`, `expected_ratio`,
`preregistered_ratio_prediction`, `expected_power`, `measured_boundary_size` — stamping them with a
fresh freeze timestamp as if they had been re-derived for T176. They would not have been. No guard
exists.

Required work, in this order:

1. Add a `--backend-config` argument and read `shot_rate_per_second_used` and the overhead from it
   rather than from module constants.
2. **Split the write-back at `:572` into a timing block and a statistical block**, and add an
   explicit `--rewrite-statistics` flag defaulting to `False`. A backend migration rewrites timing
   only. Fail loudly if a statistical field would change while the flag is off.
3. Carry the §3 measurement into the config as provenance: record that `2.373` was verified on T176
   at 2.3639 (ratio 0.99618) against
   `quarantine/tianyan176/B4_TB5_amplified_closed_loop_20260817_run2/amplified_closed_loop_report.json`,
   with that file's sha256. The constant does not change; its *justification* now spans two backends
   and must say so.
4. Regenerate the reachability evidence at the new design point and update
   `reachability_evidence_sha256`. The current hash
   (`98d4c9b9…f18488`) is bound to the 40-pair point and will be stale the moment cycles change.
5. Re-run the power and boundary-size simulation at the chosen pair count. **Do not interpolate
   between 0.90155 at 40 pairs and 0.8046 at 30** — the registered numbers are Monte Carlo at 40 000
   replicates and 2 000 permutations, and the design's own rule is "largest pair count within two
   Monte Carlo standard errors of the maximum". Re-run it.

### 5.3 THIRD — fix the provenance gate. This is a live integrity bug, independent of the migration.

`run_b4_cadence_pair_hardware.py:1737` hardcodes the string `"backend_id": "tianyan-287"` into the
report payload. `analyze_b4_t287_cadence_residual_curve.py:123` then checks:

```python
if not hardware_report.get("completed") or hardware_report.get("backend_id") != "tianyan-287":
```

— which reads back the hardcoded literal, not the backend the job actually ran on. **A T176 run
passes the T287 provenance gate.** The same hardcoded literal appears at `:727` in the plan payload
and at `analyze_…:162` where it propagates false provenance into the analysis output.

Fix: write the *actual* `backend_config["backend"]["backend_id"]` into both payloads, and make the
analyzer compare against an expected value it is told, not one it reads from the artifact it is
supposed to be validating.

While there: `backend_id` string formats are inconsistent — `"tianyan-287"` (hyphen) versus
`"tianyan176"` (no hyphen) — and are compared with raw `==` throughout, including at `:2014` against
the platform machine list. Config *filenames* use `tianyan287` (no hyphen) while config *values* use
`tianyan-287` (hyphen). Normalise, or at minimum add a test that pins each spelling.

### 5.4 FOURTH — the other three defects found in the same audit

1. **`scripts/preflight_b4_cadence_collection_correction.py:30`** — `DEFAULT_CONFIG` points at
   `b4_cadence_pair_loop_cycle_paired_v2.json`, **not v4**. It does not preflight the config that is
   actually frozen. `:252` repeats the v2 path in a help string. Point both at v4 and re-run.
   Also `:147` hardcodes `"status": "READY_FOR_CORRECTED_T287_COLLECTION"` and `:149`/`:319`
   hardcode `"t176_quarantine_read": False` — all three become false statements after a migration.
2. **`run_b4_cadence_pair_hardware.py:483-484`** — `raise ValueError("B4 cadence supplement is
   frozen to tianyan-287")`. This freeze is *correct as a safety property* and must not simply be
   deleted. Replace it with a freeze against a backend id declared in the loop config, so the
   collection is still pinned to exactly one backend but the pin is a registered value rather than a
   source literal. Record the change of pin in an amendment document.
3. **`run_b4_cadence_pair_hardware.py:811-813`** — the plan-reuse guard compares only
   `loop_config_sha256`. A plan built against the T287 backend config will be silently reused when
   the runner is pointed at T176. Add `backend_config_sha256` to `source_hashes` (`:744`, `:819`)
   and to the guard. Note that `:743-746` digests module constants rather than the caller's actual
   paths — check that too.

### 5.5 FIFTH — resolve the B9 isolation question before submitting anything to T176

`analyze_b4_t287_cadence_residual_curve.py:138-139`:

```python
if ledger.get("hardware_submission_performed") or ledger.get("t176_quarantine_read"):
    raise ValueError("platform ledger violates B9 isolation boundary")
```

**Open governance question, ungrounded in the repo:** does collecting the registered cadence
supplement *on* T176 trip this boundary, or does the quarantine cover only the T176 **natural-drift
corpus** that B9 was written to isolate? The two readings differ on whether the whole migration is
permissible at all.

This must be settled — by reading the original B9 statement, not by inferring from the code — and
written down **before** any T176 hardware submission. If the boundary does cover the supplement, the
migration is blocked on governance and not on timing, and Murphy needs to know that immediately.
Note that both existing ledgers carry `t176_quarantine_read: False`.

### 5.6 Standing open item, unrelated to the migration

Task #19: the mirror sensitivity model in `scripts/run_cadence_pair_loop.py:301` is **refuted by
hardware** (`a695a6d`). It uses

```python
probability = float(mirror["baseline_success_probability"]) * math.exp(-float(mirror["residual_sensitivity"]) * residual_squared)
```

The exact circuit law is `∏ cos²(h_i·T)`. Replace it, and mark `residual_sensitivity = 600.0` as
refuted with the evidence hash in all four configs where it appears:
`b4_cadence_pair_loop_v1.json:69`, `_v2.json:91`, `_v3.json:131`, `_v4.json:223`.

**Verified safe:** no registered adjudication number moves. The mirror is
`raw_mirror_role = task-metric QC and descriptive readout only`, and the permutation gate never
touches it. This is a truth-in-reporting fix, not an endpoint change.

---

## 6. What must NOT be touched

1. **The registered statistical fields.** `endpoint_shot_floor_total`, `expected_ratio`,
   `preregistered_ratio_prediction`, `preregistered_ratio_interval`, `measured_boundary_size` and
   the gate module are frozen. §3 shows they survive the migration; they must not be silently
   rewritten by the freeze script (§5.2.2). `gate_module_change_permitted = False` and
   `outlier_exclusion_permitted = False` are registered values.
2. **`--confirm-hardware`** stays `required=True` on the runner, the smoke, the shot-ceiling probe,
   the sensitivity probe and the amplified probe. `write_new` / `write_new_json` and the probes'
   `output.mkdir(parents=True)` must keep refusing to overwrite evidence artifacts.
3. **The quarantine tree.** `E:\TianYan\XA-202609\quarantine\tianyan176\` artifacts stay
   `pooling_permitted: false`, `not_main_endpoint: true`,
   `registered_endpoint_contribution: "none"`. The 2026-08-17 result is read *as evidence about the
   instrument* (§3), never pooled into an adjudication.
4. **Credentials.** Keys live only in the process environment as `B4_TIANYAN287_LOGIN_KEY` /
   `B4_TIANYAN176_LOGIN_KEY`, injected inline as an env-var prefix. Never written to a file, a
   config, or a commit. Scan for keys before staging.

   **Outstanding security action for Murphy:** the TianYan platform key
   `kFVl70A+…` has appeared in plaintext in session transcripts under
   `C:\Users\Mercu\.claude\projects\C--Users-Mercu\*.jsonl`. It should be rotated on the platform.
5. **The cadence pair.** 90 s / 360 s defines the contrast and therefore the endpoint. The wall
   clock cannot be compressed — 20 pairs at 90 s + 360 s is an irreducible 9 000 s nominal grid per
   session. Reducing *cycles* is permitted (§4.5); reducing the *periods* is not.
6. **`optional_stopping_permitted = False`.** The pair count is not inspected mid-collection.
   `minimum_adjudicated_cycle_pairs = 30` is a tolerance for lost cycles, not a sequential test.

---

## 7. Repository hygiene

175 dirty/untracked paths at `a59d24b`. Before any of this is committed:

1. Scan for credentials across the whole diff.
2. Curate a Stage-1 commit rather than a bulk `git add -A`.
3. Two freeze-manifest hashes remain uncovered; B0 24 h acceptance is still open; there is wording
   residue (the B7 "T-floor" mislabel, and two older configs still carrying
   `freeze_status = "pending T-B6"`); and the §153 protocol interpretation fork is still awaiting
   Murphy's decision.

---

## 8. One-paragraph summary

The registered v4 collection is statistically ready and its numbers survive the move to T176 — the
one constant that could have been backend-specific, `PER_CONDITION_FLOOR_CONSTANT = 2.373`, was
measured directly on T176 on 2026-08-17 at 2.3639, a 0.38% agreement, on the safe side, and the 4.8%
excess over ideal contrast that looked like a T287 fingerprint turns out to appear on both backends.
What does not survive is the schedule: T176 charges 1.85× per job against an overhead-dominated
design, so the 40-pair point needs 1 303.7 s against an enforced 1 200 s daily window, and the honest
answer sits somewhere between 32 and 38 registered pairs depending on a shot rate that nobody has
yet measured free of queue wait. The single highest-value next action costs no machine time at all:
export the platform task-detail records for the 24 T176 query_ids already listed in
`C:\Users\Mercu\t176_20260817_query_ids.json`, settle whether billing tracks true runtime or the
frozen model, and re-run the design table. Everything else — re-freezing the correction without
re-blessing the statistics, the inert provenance gate, the v2/v4 preflight mismatch, and the B9
isolation question — is downstream of that one number.

---

## 9. Codex completion update — 2026-08-23

The 24 query IDs were recovered 24/24 from the supplied task-list exports, all on `tianyan176`. Queue-free role runtimes are BASELINE `18.735 s`, SENSE `3.556 s`, and MIRROR `72.057 s`; all 20 MIRROR task records share one exact execution interval, so the platform executed that job as a parallel/batched unit. The formal timing ledger is frozen at SHA256 `D0089FE19A96DC8B0D95A07B53F4DE34A64AEE196BBBC21979005D9A7D673F10`.

The migration now retains all 40 registered cadence pairs. The frozen role-envelope budget is `323.824 s/session` execution wall and a conservative `647.648 s/session` task-runtime-sum quota; total quota is `1295.296 s < 2400 s`. No registered endpoint, threshold, prediction interval, power, boundary-size, shot-allocation, exclusion, or stopping field changed.

Backend provenance, backend pinning, plan reuse, analyzer, v4 preflight, timing-only freeze, safe ledger, CRLF/LF evidence verification, and B9 isolation gates are implemented and covered by the focused regression suite. The operative governance document is `docs/B4_B5_T176_BACKEND_MIGRATION_AMENDMENT_20260823.md`; the runner defaults to the registered T176 v4 config and the sealed T176 quarantine path.

No hardware job was submitted. Software/evidence status is `READY_FOR_OPERATOR_HARDWARE_CONFIRMATION`; actual execution still requires explicit `--confirm-hardware`.
