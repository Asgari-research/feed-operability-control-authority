# Provenance method

A. `experimental_anchor`: directly reported experimental scalar/table row.
B. `published_parameter`: a parameter printed in a source.
C. `derived_reconstruction`: computed from published equations/parameters.
D. `engineering_assumption`: a scenario/control setting defined by the code.

Manuscript wording must preserve the provenance class.
Missing information is left blank or registered in `catalog/data_gap_register.csv`; it is never silently imputed as experimental.
