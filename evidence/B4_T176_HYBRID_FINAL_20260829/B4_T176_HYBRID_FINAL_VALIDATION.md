## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-29T09:05:11.030210+00:00
- Verification Status: VERIFIED
- Version Label: validation_v1
- Integrity Pass Date: 2026-08-29T09:05:11.030210+00:00
- Upstream Dependencies: exp_result_v1

## Validation Report

- **Source**: B4_T176_HYBRID_FINAL_20260829
- **Overall Confidence**: CAUTION
- **Simulation-assisted status**: B4_PRESERVED_SIMULATION_ASSISTED
- **Registered hardware status**: INCONCLUSIVE_MISSING_HARDWARE_SESSION1

### Statistical Findings

| Metric | Test | Value | Effect Size | Confidence |
|---|---|---:|---|---|
| Hybrid cadence ratio | 20,000 within-pair swaps | ratio=0.374481, p=9.9995e-05 | 62.55% descriptive reduction | CAUTION |
| Hardware Session 0 | diagnostic permutation | ratio=0.361650, p=0.00524974 | source-specific | CAUTION |
| Simulated Session 1 | diagnostic permutation | ratio=0.381652, p=0.00234988 | model-only | CAUTION |
| Frozen prediction check | interval hit | True | absolute error=0.132540 | CAUTION |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Post-hoc hybrid design | Session 1 simulation result was known before this hybrid plan; the result is not confirmatory all-hardware evidence. | Hybrid primary |
| Origin/order confounding | Hardware is fast-first Session 0 and simulation is slow-first Session 1. | Generalization |
| Registered boundary | v4 forbids simulation pooling; registered hardware endpoint remains inconclusive. | Claim vocabulary |

### Fallacy Scan

- **Coverage**: 11/11 checked

| Fallacy | Severity | Status | Detail |
|---|---|---|---|
| 1. Simpson's Paradox | NOTE | not_detected | Hardware Session 0, simulated Session 1, and the hybrid aggregate point in the same direction. |
| 2. Ecological Fallacy | NOTE | not_detected | Inference stays at the registered cycle-pair level; no individual claim is inferred from session means. |
| 3. Berkson's Paradox | NOTE | not_detected | All 20 completed hardware pairs and all 20 frozen simulated pairs are retained; no outcome-based filtering is used. |
| 4. Collider Bias | NOTE | not_detected | No post-outcome control variable is conditioned on; evidence origin is stratified and disclosed rather than adjusted away. |
| 5. Base Rate Neglect | NOTE | not_applicable | This is not a screening/classification accuracy analysis and reports no sensitivity, specificity, PPV, or NPV. |
| 6. Regression to the Mean | NOTE | not_detected | Cadence arms and pairs were frozen before outcomes; cycles were not selected for extreme prior residuals. |
| 7. Survivorship Bias | NOTE | not_detected | The hybrid table uses the complete frozen 20+20 pair sets with zero pair attrition and reports source counts explicitly. |
| 8. Look-Elsewhere Effect | CAUTION | bounded_but_post_hoc | One primary hybrid statistic is locked before hardware unseal, but the Session 1 simulation outcome was already known when this post-hoc hybrid analysis was planned. |
| 9. Garden of Forking Paths | CAUTION | bounded_but_post_hoc | Analyzer, source hashes, endpoint, seed, permutation count, prediction interval, and decision criteria were frozen before Session 0 unseal; post-hoc creation still prevents a confirmatory all-hardware interpretation. |
| 10. Correlation != Causation | CAUTION | claim_limited | The controlled paired hardware session supports a cadence mechanism, but replacing the balancing session with model output cannot establish a two-session hardware causal effect. |
| 11. Reverse Causality | NOTE | not_detected | Cadence timing and block order were frozen before cycle outcomes, establishing temporal precedence for the assigned condition. |

### Reproducibility

- **Method**: deterministic analysis-core re-run from frozen hashes and source artifacts
- **Verdict**: REPRODUCIBLE
- **Original signature**: `1fe980cff5cb2db4be7424234f3ece18424135fa322598d850a042fd79943828`
- **Re-run signature**: `1fe980cff5cb2db4be7424234f3ece18424135fa322598d850a042fd79943828`
- **Diff**: exact 0 (signature match)

> B4 is preserved only in the explicitly post-hoc, simulation-assisted consistency test. The registered all-hardware endpoint remains inconclusive because Session 1 was not collected.
