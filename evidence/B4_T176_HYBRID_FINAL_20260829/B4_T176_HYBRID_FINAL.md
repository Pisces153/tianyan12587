## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-29T09:05:00.888389+00:00
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: B4_T176_HYBRID_FINAL_20260829
- **Type**: analysis
- **Status**: completed
- **Frozen plan**: `E:\TianYan\XA-202609\artifacts\analysis\B4_T176_HYBRID_FINAL_20260829_v2.plan.json`
- **Analysis label**: `POST_HOC_HYBRID_SIMULATION_ASSISTED`
- **Hardware raw counts / NPZ read**: no / no

### Outcome

- Simulation-assisted status: **B4_PRESERVED_SIMULATION_ASSISTED**
- Registered all-hardware status: **INCONCLUSIVE_MISSING_HARDWARE_SESSION1**
- Hardware Session 0: ratio `0.361650`, p `0.00524974`, n `20` (diagnostic)
- Simulated Session 1: ratio `0.381652`, p `0.00234988`, n `20` (diagnostic)
- Hybrid 20+20: ratio `0.374481`, p `9.9995e-05`, critical ratio `0.635760`
- Frozen prediction interval: `[0.3260557361133296, 0.7884314361673062]`; hit: `True`

### Required disclosure

> B4 is preserved only in the explicitly post-hoc, simulation-assisted consistency test. The registered all-hardware endpoint remains inconclusive because Session 1 was not collected.

The hybrid p-value is a post-hoc model-assisted consistency result. It is not an independent all-hardware confirmation, and it never changes the registered endpoint from INCONCLUSIVE.
