# Methodological scope

## Minimum control authority

The analysis reports two related quantities. The first is the minimum number of manipulated variables required to satisfy the purity and recovery constraints. The second is the minimum normalized intervention effort among feasible policies with that minimum cardinality.

This separates the question "how many control handles are needed?" from "how far must they move?".

## Operability terminology

- **Passive robustness:** the frozen nominal policy remains on-spec.
- **Adaptive operability:** specification can be recovered by allowed operating changes.
- **Equipment-realizable adaptive operability:** the required changes also satisfy the declared actuator-screening bounds.

## Equipment limits

The default regeneration-pressure and timing bounds are engineering screening assumptions, not measured limits of a specific vacuum system. They are recorded in `data/engineering/equipment_realizability_assumptions.csv`, are user-configurable, and are reported separately from unconstrained mathematical authority.

## Uncertainty

Calibration-conditioned parameter uncertainty is kept separate from independent thermodynamic-source sensitivity, discrepancy stress tests and kinetic/front-utilization sensitivity. The front-factor range is a sensitivity analysis rather than a probability distribution.

## Kinetic evidence

Published 5%, 15% and 30% CO2 breakthrough conditions and independent kinetic scalar information are included with provenance. Exact breakthrough-trace coordinates are not fabricated. An optional audited trace CSV can be supplied through `--breakthrough-trace-csv`.

## High-fidelity CSS

The non-isothermal CSS model is an optional validation layer. It is not used to support the main manuscript claims unless independent same-rig validation passes. Numerical timeout is reported as indeterminate rather than as physical-model falsification.

## Energy

The reported energy quantity is an idealized vacuum-work proxy and is not presented as a validated vacuum-machine energy model.
## Conventional 95/90 benchmark

No frozen operating interval satisfies both 95% purity and 90% recovery over the evaluated feed-composition domain. At the nominal 15% CO2 feed, purity is approximately 95.5% and recovery approximately 88.1%; adaptive operation restores the conventional benchmark. This case is reported as absence of passive robustness rather than as missing data.

## Intervention economy

The quasi-steady disturbance comparison reports on-spec state recovery together with normalized actuator movement. The intervention-economy statistic is the number of additional on-spec states recovered relative to the frozen policy per unit of total log-actuator movement. It is a descriptive comparison, not a general optimal-control objective.

