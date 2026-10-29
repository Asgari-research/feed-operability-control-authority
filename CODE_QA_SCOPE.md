# Code and numerical-QA scope

The released Python modules were reviewed for compact public headers and unnecessary legacy package trees. `python -m compileall` checks source syntax; `python -m pytest -q tests` is the recommended package/regression check; a small `--level quickcheck` run is a pipeline smoke test; the full 2000-draw comprehensive run is the actual independent reproducibility experiment.

**No comprehensive scientific rerun was performed while preparing these five editorial/handover packages.** An attempted command-line check in this environment was not completed because Matplotlib font-cache initialisation exceeded the local check window. This is not evidence of an algorithmic error, and it does not establish successful execution either. Run the listed commands and retain logs on the intended computer before asserting numerical reproducibility.

**Important:** the optional CSS validation does not count as an independent high-fidelity result unless the same-rig validation gate passes. Original experimental data may have separate licence restrictions. Check these before publishing the repository.
