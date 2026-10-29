# Software verification

The supported user interfaces are the two root-level scripts. Internal numerical modules are stored under `src/p10/` with neutral names and use the single canonical `data/` tree.

Before release, the package is syntax-compiled, the unit tests are run, a fresh packaged-input quickcheck is executed, and the final ZIP is reopened for integrity checking.
