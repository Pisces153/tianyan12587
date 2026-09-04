# AEMTN Deployment Lineage

`AEMTN` names a shared architecture, not one interchangeable checkpoint.

| Lineage | Input contract | Training family | Deployment status |
| --- | --- | --- | --- |
| Original AEMTN | Original manuscript observable contract | Continuous-time QuTiP datasets | Manuscript/model study only |
| IBM local6 V4-V7 | Pauli15-derived local6 plus time | IBM-aligned digital simulation | IBM initial phase completed; after phase unavailable because of account usage limit |
| TianYan V7 native-shallow | Pauli15-derived local6 plus time | Independent zero-CZ separable Rx synthetic corpus | TianYan r4 completed; transparent pilot only |
| TianYan V8 | Same local6 plus time contract | To be generated from one fixed noncommuting low-CZ V8 circuit family | Draft only; no checkpoint or hardware task exists yet |

Results must not pool numbers across rows.  Each checkpoint manifest records
the architecture configuration, observable order, training-data manifest, and
the permitted deployment task family.
