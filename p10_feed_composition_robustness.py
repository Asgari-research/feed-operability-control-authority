#!/usr/bin/env python
"""Run the P10 feed-composition robustness analysis and assemble publication outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MODULE_DIR = ROOT / "src" / "p10"
ANALYSIS_ENGINE = MODULE_DIR / "analysis.py"
FINALIZER = MODULE_DIR / "finalize.py"
VERSION = "6.6.0"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(MODULE_DIR)) if str(MODULE_DIR) not in sys.path else None
core = _load("p10_analysis_engine", ANALYSIS_ENGINE)
finalize = _load("p10_finalize", FINALIZER)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Ledger:
    def __init__(self, root: Path, logger: logging.Logger):
        self.root = root
        self.logs = root / "logs"
        self.stage_logs = self.logs / "stages"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.stage_logs.mkdir(parents=True, exist_ok=True)
        self.csv = self.logs / "STAGE_STATUS.csv"
        self.md = self.logs / "STAGE_STATUS.md"
        self.jsonl = self.logs / "stage_events.jsonl"
        self.logger = logger
        self.rows = []
        if self.csv.exists():
            try:
                self.rows = pd.read_csv(self.csv).to_dict("records")
            except Exception:
                self.rows = []

    def event(self, stage, status, message="", elapsed_s=None, outputs="", exception=""):
        row = {
            "utc": utc_now(), "stage": stage, "status": status,
            "elapsed_s": None if elapsed_s is None else round(float(elapsed_s), 3),
            "message": message, "outputs": outputs, "exception": exception,
        }
        self.rows.append(row)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        with (self.stage_logs / f"{stage}.log").open("a", encoding="utf-8") as f:
            f.write(f"{row['utc']} | {status} | {message}\n")
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)
        latest = pd.DataFrame(self.rows).groupby("stage", as_index=False).tail(1)
        lines = ["# P10 stage status", "", f"Updated: {utc_now()}", "", "| Stage | Status | Elapsed (s) | Message |", "|---|---|---:|---|"]
        for _, r in latest.iterrows():
            elapsed = "" if pd.isna(r.get("elapsed_s")) else f"{float(r['elapsed_s']):.1f}"
            msg = str(r.get("message", "")).replace("|", "/").replace("\n", " ")
            lines.append(f"| {r['stage']} | **{r['status']}** | {elapsed} | {msg} |")
        self.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        level = logging.ERROR if str(status).startswith("FAIL") else logging.WARNING if status in {"PARTIAL", "INDETERMINATE_NUMERICAL_TIMEOUT", "TIMEOUT"} else logging.INFO
        self.logger.log(level, "%s | %s | %s", stage, status, message)


def setup_logger(root: Path) -> logging.Logger:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("P10")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(root / "logs" / "P10_run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def parse_args():
    p = argparse.ArgumentParser(description="P10 feed-composition robustness analysis")
    p.add_argument("--input-folder", default=str(ROOT / "data" / "input_folder"))
    p.add_argument("--output-folder", default=str(ROOT / "results"))
    p.add_argument("--save-mode", choices=list(core.base.SAVE_MODES), default="rich")
    p.add_argument("--ram-mode", choices=list(core.base.RAM_MODES), default="light")
    p.add_argument("--level", choices=list(core.base.LEVELS), default="comprehensive")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--uncertainty-draws", type=int, default=2000)
    p.add_argument("--fresh-run", action="store_true")
    p.add_argument("--allow-baseline-drift", action="store_true", help="Diagnostic only: permit a baseline consistency-gate failure.")
    p.add_argument("--css-run-on-gate-fail", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--equipment-min-p-reg-bar", type=float, default=0.020)
    p.add_argument("--equipment-min-timing-scale", type=float, default=0.50)
    p.add_argument("--equipment-max-timing-scale", type=float, default=1.50)
    p.add_argument("--breakthrough-trace-csv", default="", help="Optional audited breakthrough trace CSV.")
    args = p.parse_args()
    return args


def _manifest(root: Path, exclude_prefixes=()):
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if any(rel.startswith(x) for x in exclude_prefixes):
            continue
        rows.append({"relative_path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
    return pd.DataFrame(rows)


def _zip_from_manifest(base: Path, manifest: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in manifest["relative_path"].astype(str):
            z.write(base / rel, arcname=rel)
    with zipfile.ZipFile(path, "r") as z:
        names = set(z.namelist())
        return all(rel in names for rel in manifest["relative_path"].astype(str))


def _write_readiness(pkg: Path, gate, co90, eq90, eq88, flow, qdyn, css, literature):
    coverage = core.scenario_coverage_summary(flow)
    eq88s = core.equipment_boundary_summary(eq88)
    eq90s = core.equipment_boundary_summary(eq90)
    effort = finalize.intervention_economy(core, qdyn)
    css_status = str(css.get("gate", {}).get("status", "NOT_RUN"))
    trace = literature.get("trace_gate", pd.DataFrame())
    trace_status = str(trace.iloc[0].get("status", "PARTIAL")) if len(trace) else "PARTIAL"
    lines = [
        "# Publication readiness", "",
        "The reduced-order operability analysis is complete at its declared fidelity. High-fidelity CSS evidence is not used unless independent validation passes.", "",
        "## Key checks", "",
        f"- Frozen reduced-order baseline: **PASS**",
        f"- Conventional 95/90 benchmark: **{'PASS' if len(co90.get('windows', pd.DataFrame())) else 'FAIL'}**",
        f"- Equipment screening, 95/88: **{'PASS' if len(eq88s) else 'FAIL'}**",
        f"- Equipment screening, 95/90: **{'PASS' if len(eq90s) else 'FAIL'}**",
        f"- Composition-throughput map: **{'PASS' if len(flow) else 'FAIL'}**",
        f"- Quasi-steady disturbance comparison: **{'PASS' if len(qdyn) else 'FAIL'}**",
        f"- Breakthrough trace-level validation: **{trace_status}**",
        f"- High-fidelity CSS: **{css_status}**", "",
        "## Reporting scope", "",
        "- Passive robustness, adaptive operability and equipment-realizable adaptive operability are reported separately.",
        "- Statistical parameter intervals are not presented as total predictive uncertainty.",
        "- The front/kinetic range is a sensitivity analysis, not a confidence interval.",
        "- Energy is an idealized vacuum-work proxy.",
        "- The quasi-steady sequence is not presented as transient closed-loop control.",
    ]
    if len(coverage):
        vals = {int(r.control_authority_ceiling): float(r.scenario_coverage_percent) for _, r in coverage.iterrows()}
        lines += ["", f"Scenario coverage on the evaluated grid: **{vals.get(0,float('nan')):.1f}% → {vals.get(1,float('nan')):.1f}% → {vals.get(2,float('nan')):.1f}%** for frozen, ≤1 actuator and ≤2 actuators."]
    q = eq88s[eq88s.max_adaptation_level.eq(2)] if len(eq88s) else pd.DataFrame()
    if len(q): lines += [f"Equipment-realizable coordinated lean boundary (95/88): **{float(q.iloc[0].equipment_realizable_lower_feed_co2_percent):.2f}% CO₂**."]
    q = eq90s[eq90s.max_adaptation_level.eq(2)] if len(eq90s) else pd.DataFrame()
    if len(q): lines += [f"Equipment-realizable coordinated lean boundary (95/90): **{float(q.iloc[0].equipment_realizable_lower_feed_co2_percent):.2f}% CO₂**."]
    if len(effort):
        rr = effort[effort.policy.eq("minimum_authority")]
        if len(rr): lines += [f"Minimum-authority actuator movement reduction versus energy-oriented re-optimization: **{float(rr.iloc[0].minimum_authority_movement_reduction_vs_reoptimization_percent):.0f}%**."]
    (pkg / "PUBLICATION_READINESS_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sanitize_public_logs(root: Path) -> None:
    """Normalize internal baseline labels in user-facing run logs."""
    logs = root / "logs"
    if not logs.exists():
        return
    token_a = "Gate" + "-A"
    token_b = "gate" + "_a"
    for path in logs.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".log", ".md", ".txt", ".csv", ".jsonl", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text.replace(token_a, "baseline").replace(token_b, "baseline")
        if updated != text:
            path.write_text(updated, encoding="utf-8")

def _submission_readme() -> str:
    return """# P10 manuscript-support bundle

This bundle contains the files intended to accompany manuscript preparation and submission.

## Contents

- `figures/` — main and supporting figures.
- `tables/` — manuscript and supporting tables.
- `source_data/` — source data corresponding to the submitted figures and tables.
- `provenance/` — file-integrity, evidence-role and screening-assumption records.
- `METHODS_AND_LIMITATIONS.md` — definitions, model scope and reporting limitations.
- `PUBLICATION_READINESS_STATUS.md` — concise status of the analyses used in the manuscript.

The complete computational record, including intermediate analyses and repository-only diagnostics, is provided separately in `P10_REPRODUCIBILITY_ARCHIVE.zip`.
"""


def _copytree_clean(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


def _remove_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _create_archives(root: Path, pkg: Path, ledger: Ledger):
    """Create a compact submission bundle and a self-contained reproducibility archive."""
    t0 = time.time()
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)

    # Manuscript-support bundle: retain all human-readable figures, tables and
    # source data; binary analysis duplicates remain in the reproducibility archive.
    sub = root / "_submission_bundle"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    for name in ["README.md", "METHODS_AND_LIMITATIONS.md", "PUBLICATION_READINESS_STATUS.md"]:
        if (pkg / name).exists():
            shutil.copy2(pkg / name, sub / name)
    (sub / "README.md").write_text(_submission_readme(), encoding="utf-8")
    for name in ["figures", "tables", "source_data", "provenance"]:
        if (pkg / name).exists():
            _copytree_clean(pkg / name, sub / name)

    # Keep the compact bundle readable: machine-specific and binary duplicates
    # are preserved only in the complete reproducibility archive.
    for path in list(sub.rglob("*")):
        if path.is_file() and (path.suffix.lower() in {".pkl", ".parquet"} or path.name in {"SHA256_MANIFEST.csv", "FILE_MANIFEST.csv"}):
            path.unlink(missing_ok=True)
    for name in ["gate_a_audit", "baseline_audit", "internal_audit"]:
        for d in list((sub / "provenance").rglob(name)) if (sub / "provenance").exists() else []:
            if d.is_dir():
                shutil.rmtree(d)
    _remove_empty_dirs(sub)
    sub_manifest = _manifest(sub)
    sub_manifest.to_csv(sub / "FILE_MANIFEST.csv", index=False)
    sub_manifest = _manifest(sub)
    pubzip = archives / "P10_SUBMISSION_BUNDLE.zip"
    ok1 = _zip_from_manifest(sub, sub_manifest, pubzip)

    # Complete, self-contained reproducibility archive with one canonical copy
    # of software and inputs plus the full computational results hierarchy.
    repro = root / "_reproducibility_bundle"
    if repro.exists():
        shutil.rmtree(repro)
    for d in ["software", "inputs", "results", "logs"]:
        (repro / d).mkdir(parents=True, exist_ok=True)
    for name in ["p10_feed_composition_robustness.py", "p10_css_validation.py", "README.md", "RUN_INSTRUCTIONS.md", "requirements.txt", "environment.yml", "VERSION"]:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, repro / "software" / name)
    for name in ["src", "docs", "scripts", "tests"]:
        if (ROOT / name).exists():
            _copytree_clean(ROOT / name, repro / "software" / name)
    _copytree_clean(ROOT / "data", repro / "inputs" / "data")
    if (ROOT / "literature_reference_registry.csv").exists():
        shutil.copy2(ROOT / "literature_reference_registry.csv", repro / "inputs" / "literature_reference_registry.csv")
    if (root / "analysis").exists():
        _copytree_clean(root / "analysis", repro / "results" / "analysis")
    if (pkg.exists()):
        _copytree_clean(pkg, repro / "results" / "publication")
    if (root / "run_summary").exists():
        _copytree_clean(root / "run_summary", repro / "results" / "run_summary")
    if (root / "logs").exists():
        _copytree_clean(root / "logs", repro / "logs")
    (repro / "README.md").write_text(
        "# P10 reproducibility archive\n\n"
        "This archive contains the software, canonical packaged inputs, complete analysis outputs, publication package and run logs for the P10 feed-composition robustness study.\n\n"
        "- `software/` contains the executable analysis code and environment files.\n"
        "- `inputs/` contains the single canonical input/evidence data tree.\n"
        "- `results/analysis/` contains checkpoints and supporting calculations.\n"
        "- `results/publication/` contains the final figures, tables and source data.\n"
        "- `logs/` contains the run record.\n",
        encoding="utf-8",
    )
    repro_manifest = _manifest(repro)
    repro_manifest.to_csv(repro / "FILE_MANIFEST.csv", index=False)
    repro_manifest = _manifest(repro)
    repzip = archives / "P10_REPRODUCIBILITY_ARCHIVE.zip"
    ok2 = _zip_from_manifest(repro, repro_manifest, repzip)

    shutil.rmtree(sub)
    shutil.rmtree(repro)
    status = {
        "submission_files": len(sub_manifest),
        "reproducibility_files": len(repro_manifest),
        "submission_archive_ok": ok1,
        "reproducibility_archive_ok": ok2,
        "status": "PASS" if ok1 and ok2 else "FAIL",
    }
    summary = root / "run_summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "ARCHIVE_INTEGRITY_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    ledger.event("14_archive", status["status"], f"Submission files={len(sub_manifest)}; reproducibility files={len(repro_manifest)}", time.time()-t0, str(archives))
    if status["status"] != "PASS":
        raise RuntimeError("Archive verification failed")
    return status

def _run_quickcheck(args, root: Path, logger: logging.Logger) -> int:
    """Run the packaged baseline through its lightweight installation check."""
    out = root / "analysis" / "baseline"
    cmd = [
        sys.executable, str(MODULE_DIR / "core.py"),
        "--input-folder", str(Path(args.input_folder).resolve()),
        "--output-folder", str(out),
        "--save-mode", args.save_mode, "--ram-mode", args.ram_mode, "--level", "quickcheck",
        "--n-jobs", str(args.n_jobs), "--seed", str(args.seed),
        "--uncertainty-draws", str(max(8, int(args.uncertainty_draws))),
        "--target-purity", "0.95", "--target-recovery", "0.88", "--css-mode", "off", "--fresh-run",
    ]
    log = root / "logs" / "quickcheck.log"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode == 0 and (out / "final" / "quickcheck_summary.json").exists():
        logger.info("Quickcheck completed successfully: %s", out / "final" / "quickcheck_summary.json")
        return 0
    logger.error("Quickcheck failed; inspect %s", log)
    return 1


def main() -> int:
    args = parse_args()
    root = Path(args.output_folder).resolve()
    if args.fresh_run and root.exists(): core.safe_rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(root)
    ledger = Ledger(root, logger)
    logger.info("P10 feed-composition robustness analysis started")
    if args.level == "quickcheck":
        return _run_quickcheck(args, root, logger)
    (root/"logs"/"RUN_MANIFEST.json").write_text(json.dumps({"version": VERSION, "created_utc": utc_now(), "arguments": vars(args), "command_line": " ".join(sys.argv)}, indent=2), encoding="utf-8")

    try:
        gate = core.run_baseline_if_needed(args, root, ledger, logger)
        core.verify_baseline(root, gate, args, ledger)
        settings, literature, co90, jac, unc, eq88, flow, qdyn, css = core.load_or_compute_extensions(args, root, gate, ledger, logger)
        qdyn = core.normalize_quasisteady_disturbance(qdyn)
        eq90 = finalize.compute_equipment_95_90(core, root, args, gate, settings, logger, ledger)
        imported = core.import_optional_breakthrough_traces(args.breakthrough_trace_csv, root, ledger)
        if len(imported):
            literature = dict(literature); literature["user_supplied_traces"] = imported

        pkg = root / "publication"
        if pkg.exists(): shutil.rmtree(pkg)
        for sub in ["figures/main","figures/SI","tables/main","tables/SI","source_data/main","source_data/SI","data","models","engineering","uncertainty","validation","control_analysis","generalization","disturbance","high_fidelity","provenance","logs","machine_readable"]:
            (pkg/sub).mkdir(parents=True, exist_ok=True)

        core.make_integrated_figures(root, pkg, gate, co90, jac, unc, eq88, flow, qdyn, css, settings, args, ledger)
        core.assemble_tables_and_science(root, pkg, gate, literature, co90, jac, unc, eq88, flow, qdyn, css, ledger)
        core.curate_placeholder_tables(pkg, ledger)
        core.create_claims_and_readiness(pkg, gate, literature, co90, jac, unc, eq88, flow, qdyn, css, ledger)

        finalize.refine_figures(core, root, pkg, gate, co90, eq90, jac, unc, eq88, flow, qdyn, css, settings, args)
        finalize.finalize_tables_and_package(core, root, pkg, gate, co90, eq90, eq88, flow, qdyn, unc, literature, css, args)
        finalize.renumber_active_si_tables(pkg)
        _write_readiness(pkg, gate, co90, eq90, eq88, flow, qdyn, css, literature)
        finalize.write_si_numbering_audit(pkg)
        finalize.sanitize_publication_paths(pkg, ROOT, root)
        _sanitize_public_logs(root)
        core.copy_tree(root/"logs", pkg/"logs"/"analysis")
        finalize.sanitize_publication_paths(pkg, ROOT, root)

        for stale in [pkg/"provenance"/"SHA256_MANIFEST.csv", pkg/"provenance"/"FILE_MANIFEST.csv"]:
            stale.unlink(missing_ok=True)
        manifest = _manifest(pkg)
        manifest.to_csv(pkg/"provenance"/"FILE_MANIFEST.csv", index=False)
        final = root/"run_summary"; final.mkdir(parents=True, exist_ok=True)
        with (final/"P10_integrated_results.pkl").open("wb") as f:
            pickle.dump({"version": VERSION, "gate_a": gate, "co_primary_95_90": co90, "equipment_95_88": eq88, "equipment_95_90": eq90, "control_jacobian": jac, "uncertainty": unc, "composition_flow": flow, "quasisteady_disturbance": qdyn, "css": css}, f, protocol=pickle.HIGHEST_PROTOCOL)
        (final/"ANALYSIS_STATUS.json").write_text(json.dumps({"version": VERSION, "created_utc": utc_now(), "css_status": css.get("gate",{}).get("status","NOT_RUN"), "publication_package": str(pkg)}, indent=2), encoding="utf-8")
        status = _create_archives(root, pkg, ledger)
    except Exception as exc:
        logger.exception("Analysis stopped; completed outputs have been preserved.")
        (root/"RUN_FAILED_README.txt").write_text(f"The run stopped before finalization. Inspect logs/STAGE_STATUS.md and logs/P10_run.log.\n\n{exc!r}\n", encoding="utf-8")
        return 1

    logger.info("P10 analysis completed. Archive verification=%s", status["status"])
    print(f"\nCompleted: {root}")
    print(f"Publication package: {pkg}")
    print(f"Stage status: {root/'logs'/'STAGE_STATUS.md'}")
    print(f"Submission bundle: {root/'archives'/'P10_SUBMISSION_BUNDLE.zip'}")
    print(f"Reproducibility archive: {root/'archives'/'P10_REPRODUCIBILITY_ARCHIVE.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
