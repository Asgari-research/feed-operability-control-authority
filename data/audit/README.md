# Audit records and scope

The `literature_data_integrity_audit.csv` includes a historical **data-assembly pipeline** version `2.3.0`; it is **not** the current research-software version `6.6.0`. The earlier audit is retained unchanged as provenance. New current-package assertions are in `tests/test_scientific_data_consistency.py` and `docs/INDEPENDENT_PACKAGE_AUDIT_2026-10-29.md`. Verify the original primary literature independently before claiming exhaustive transcription fidelity.

The source_id `CALDERON2025_INORGCHEM_CALF20` is an older, inaccurate surname-based internal identifier for Pereira et al. (2025), DOI `10.1021/acs.inorgchem.4c04727`. Its published scalar values are correct; the stable ID is retained only for existing CSV cross-references.
