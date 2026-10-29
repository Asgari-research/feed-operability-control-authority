# P10 data contract

The study is intentionally narrow: CALF-20, dry CO2/N2, nominal 15% CO2, composition perturbation only.

## Required parser-ready roles

1. **Unary equilibrium**: enough CO2 and N2 rows to fit the retained isotherm models.
2. **Nominal process anchor**: a 15% CO2 purity/recovery calibration point with provenance.
3. **Composition-condition anchors**: published 5/15/30% CO2 conditions with provenance/status metadata. Exact time-series coordinates are optional and must never be synthesized.

## Evidence hierarchy

- raw or machine-readable trace -> direct breakthrough trace anchor;
- published condition without trace coordinates -> published composition-condition anchor, no trace;
- between published composition conditions -> between-condition support label; this does not imply an interpolated experimental breakthrough trace;
- other states inside 5-30% -> primary-envelope but untraced;
- 2-5% and 30-40% -> exploratory numerical envelope.

## Exclusions

No humidity, SOx/NOx/O2 impurity matrix, temperature-robustness atlas, other MOFs, membrane data, or database-scale ranking is part of this package.

## Boundary claim-control rule

A lower/upper intervention-window edge coincident with 0.02 or 0.40 feed CO2 fraction is an explored-domain censoring point. It may support a one-sided statement (for example, no transition resolved up to 40% CO2), but must not be reported as a finite physical boundary.
