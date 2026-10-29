"""Final presentation and reporting utilities for the P10 analysis."""
from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def authority_definition_text() -> str:
    return (
        "Minimum control authority is reported in two parts: d*(y), the minimum "
        "number of manipulated variables needed to satisfy purity and recovery, "
        "and E*(y), the minimum normalized movement among solutions with that cardinality."
    )


def compute_equipment_95_90(core, root: Path, args, gate: Mapping[str, Any], settings, logger, ledger):
    """Calculate the equipment-constrained 95/90 benchmark if it is not already present."""
    out = root / "analysis" / "06b_equipment_realizability_95_90" / "variant_result.pkl"
    if out.exists():
        with out.open("rb") as f:
            res = pickle.load(f)
        ledger.event("07b_equipment_95_90", "PASS", "Reused equipment-constrained 95/90 benchmark")
        return res
    s90 = core.dataclasses.replace(settings, target_purity=0.95, target_recovery=0.90)
    seq = core.dataclasses.replace(
        s90,
        regeneration_pressure_floor_bar=args.equipment_min_p_reg_bar,
        timing_min_scale=args.equipment_min_timing_scale,
        timing_max_scale=args.equipment_max_timing_scale,
    )
    return core.ext.run_stage05_variant(
        "06b_equipment_realizability_95_90", root, args, gate, seq, logger, ledger
    )


def intervention_economy(core, qdyn: pd.DataFrame) -> pd.DataFrame:
    """Summarize specification recovery per unit of actuator movement."""
    out = core.control_effort_summary(qdyn).copy()
    if out.empty:
        return out
    frozen = out.loc[out["policy"].eq("frozen")]
    base = int(frozen.iloc[0]["on_spec_states"]) if len(frozen) else 0
    recovered = []
    economy = []
    for _, row in out.iterrows():
        n = max(0, int(row["on_spec_states"]) - base)
        mv = float(row["total_log_actuator_movement"])
        recovered.append(n)
        economy.append(n / mv if n > 0 and mv > 0 else np.nan)
    out["additional_on_spec_states_vs_frozen"] = recovered
    out["recovered_states_per_unit_log_actuator_movement"] = economy
    return out


def equipment_basis_table(args) -> pd.DataFrame:
    rows = [
        ("minimum regeneration pressure", args.equipment_min_p_reg_bar, "bar", "declared engineering screening assumption", "User-configurable lower vacuum-pressure bound; not presented as a measured hardware limit."),
        ("minimum timing scale", args.equipment_min_timing_scale, "dimensionless", "declared engineering screening assumption", "Lower operating envelope used for realizability screening."),
        ("maximum timing scale", args.equipment_max_timing_scale, "dimensionless", "declared engineering screening assumption", "Upper operating envelope used for realizability screening."),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "evidence_role", "basis"])


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _window_rows(core, wdf: pd.DataFrame):
    return core._window_rows(wdf)


def _ladder(core, ax, wdf: pd.DataFrame, title: str, xlim=(9, 40.5), equipment: pd.DataFrame | None = None):
    core._plot_ladder(ax, wdf, title, xlim=xlim)
    if isinstance(equipment, pd.DataFrame) and len(equipment):
        for _, r in equipment.iterrows():
            lev = int(r["max_adaptation_level"])
            x = float(r["equipment_realizable_lower_feed_co2_percent"])
            if np.isfinite(x):
                ax.plot(x, lev, marker="D", ms=4.5, mfc="white", mec="#333333", mew=.8, zorder=8)


def _nearest_composition_row(df: pd.DataFrame, feed_fraction: float = 0.15) -> pd.Series | None:
    if not isinstance(df, pd.DataFrame) or df.empty or "feed_co2_fraction" not in df:
        return None
    y = pd.to_numeric(df["feed_co2_fraction"], errors="coerce")
    if not y.notna().any():
        return None
    return df.loc[(y - feed_fraction).abs().idxmin()]


def conventional_95_90_source_table(core, co90: Mapping[str, Any], equipment: pd.DataFrame, nominal_feed: float = 0.15) -> pd.DataFrame:
    windows = co90.get("windows", pd.DataFrame()) if isinstance(co90, Mapping) else pd.DataFrame()
    fixed = co90.get("fixed", pd.DataFrame()) if isinstance(co90, Mapping) else pd.DataFrame()
    selected = co90.get("selected", pd.DataFrame()) if isinstance(co90, Mapping) else pd.DataFrame()
    rows_by_level = {r[0]: r for r in core._window_rows(windows)}
    fixed_nominal = _nearest_composition_row(fixed, nominal_feed)
    selected_nominal = _nearest_composition_row(selected, nominal_feed)
    out = []
    for lev, label in [(0, "Frozen"), (1, "≤1 actuator"), (2, "≤2 actuators")]:
        row = rows_by_level.get(lev)
        lo = hi = np.nan
        lstat = ustat = "no feasible interval"
        if row is not None:
            _, _, lo, hi, lstat, ustat = row
        eq = equipment[equipment["max_adaptation_level"].eq(lev)] if isinstance(equipment, pd.DataFrame) and len(equipment) else pd.DataFrame()
        eq_lo = float(eq.iloc[0]["equipment_realizable_lower_feed_co2_percent"]) if len(eq) else np.nan
        record = {
            "control_authority_level": lev,
            "control_authority": label,
            "lower_boundary_feed_co2_percent": 100 * lo if np.isfinite(lo) else np.nan,
            "upper_boundary_feed_co2_percent": 100 * hi if np.isfinite(hi) else np.nan,
            "lower_boundary_status": lstat,
            "upper_boundary_status": ustat,
            "equipment_realizable_lower_boundary_percent": eq_lo,
            "nominal_feed_co2_percent": 100 * nominal_feed,
        }
        if fixed_nominal is not None:
            pur = float(pd.to_numeric(pd.Series([fixed_nominal.get("purity_estimate", np.nan)]), errors="coerce").iloc[0])
            rec = float(pd.to_numeric(pd.Series([fixed_nominal.get("recovery_estimate", np.nan)]), errors="coerce").iloc[0])
            record.update({
                "nominal_frozen_purity_percent": 100 * pur if np.isfinite(pur) else np.nan,
                "nominal_frozen_recovery_percent": 100 * rec if np.isfinite(rec) else np.nan,
                "nominal_recovery_shortfall_to_90_percent_pp": max(0.0, 90.0 - 100 * rec) if np.isfinite(rec) else np.nan,
                "nominal_frozen_on_spec_95_90": bool(np.isfinite(pur) and np.isfinite(rec) and pur >= 0.95 and rec >= 0.90),
            })
        if selected_nominal is not None:
            record.update({
                "nominal_selected_adaptation_level": int(pd.to_numeric(pd.Series([selected_nominal.get("adaptation_level", np.nan)]), errors="coerce").fillna(-1).iloc[0]),
                "nominal_selected_p_reg_bar": float(pd.to_numeric(pd.Series([selected_nominal.get("p_reg_bar", np.nan)]), errors="coerce").iloc[0]),
                "nominal_selected_timing_scale": float(pd.to_numeric(pd.Series([selected_nominal.get("timing_scale", np.nan)]), errors="coerce").iloc[0]),
                "nominal_selected_purity_percent": 100 * float(pd.to_numeric(pd.Series([selected_nominal.get("purity_estimate", np.nan)]), errors="coerce").iloc[0]),
                "nominal_selected_recovery_percent": 100 * float(pd.to_numeric(pd.Series([selected_nominal.get("recovery_estimate", np.nan)]), errors="coerce").iloc[0]),
            })
        out.append(record)
    return pd.DataFrame(out)


def _plot_conventional_95_90(core, ax, co90: Mapping[str, Any], equipment: pd.DataFrame, xlim=(9, 40.5)) -> pd.DataFrame:
    source = conventional_95_90_source_table(core, co90, equipment)
    colors = {0: "#B8B8B8", 1: core._class_color(1), 2: core._class_color(2)}
    for lev, label in [(0, "Frozen"), (1, "≤1 actuator"), (2, "≤2 actuators")]:
        r = source[source["control_authority_level"].eq(lev)].iloc[0]
        lo = r["lower_boundary_feed_co2_percent"]
        hi = r["upper_boundary_feed_co2_percent"]
        if lev == 0 and not (np.isfinite(lo) and np.isfinite(hi)):
            ax.axhspan(lev - .18, lev + .18, color="#E8E8E8", alpha=.9, zorder=0)
            ax.text(24.5, lev, "No frozen 95/90-feasible interval", ha="center", va="center", fontsize=7, color="#555555")
            pur = r.get("nominal_frozen_purity_percent", np.nan)
            rec = r.get("nominal_frozen_recovery_percent", np.nan)
            deficit = r.get("nominal_recovery_shortfall_to_90_percent_pp", np.nan)
            ax.plot(15, lev, marker="x", ms=7, mew=1.2, color="#7A3E3E", zorder=6)
            if np.isfinite(pur) and np.isfinite(rec):
                ax.text(16.1, lev - .22, f"15% nominal: {pur:.1f}% purity / {rec:.1f}% recovery  (−{deficit:.1f} pp recovery)", fontsize=6.2, color="#555555", va="center")
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        ax.plot([lo, hi], [lev, lev], lw=7, solid_capstyle="butt", color=colors[lev], alpha=.9)
        ax.plot(lo, lev, marker="|", ms=16, mew=1.2, color="#222222")
        ax.text(lo + .25, lev + .17, f"{lo:.2f}%", fontsize=7, ha="left", va="bottom")
        if "censor" in str(r["upper_boundary_status"]).lower() or hi >= xlim[1] - .7:
            ax.plot(min(hi, xlim[1] - .1), lev, marker=">", ms=8, color=colors[lev])
            ax.text(min(hi, xlim[1] - .8), lev + .17, ">40% (domain limit)", fontsize=6.5, ha="right", va="bottom")
        else:
            ax.plot(hi, lev, marker="|", ms=16, mew=1.2, color="#222222")
        eq_lo = r.get("equipment_realizable_lower_boundary_percent", np.nan)
        if np.isfinite(eq_lo):
            ax.plot(eq_lo, lev, marker="D", ms=4.5, mfc="white", mec="#333333", mew=.8, zorder=8)
    sel = source[source["control_authority_level"].eq(1)]
    if len(sel):
        r = sel.iloc[0]
        al = r.get("nominal_selected_adaptation_level", np.nan)
        if np.isfinite(al) and al <= 1:
            ts = r.get("nominal_selected_timing_scale", np.nan)
            pr = r.get("nominal_selected_p_reg_bar", np.nan)
            if np.isfinite(ts) and np.isfinite(pr):
                nominal_pr = 0.05
                pressure_changed = not np.isclose(pr, nominal_pr, rtol=0, atol=5e-4)
                timing_changed = not np.isclose(ts, 1.0, rtol=0, atol=5e-3)
                if pressure_changed and not timing_changed:
                    note = "15% restored by pressure-only intervention"
                elif timing_changed and not pressure_changed:
                    note = "15% restored by timing-only intervention"
                else:
                    note = "15% restored by one-variable intervention"
                ax.text(15.35, .72, note, fontsize=6.3, color="#555555", va="center")
    ax.set_yticks([0,1,2], ["Frozen", "≤1 actuator", "≤2 actuators"])
    ax.set_ylim(-.55, 2.55)
    ax.set_xlim(*xlim)
    ax.axvline(15, ls=":", lw=.9, color="#555555")
    ax.set_xlabel("Feed CO$_2$ (%)")
    ax.set_title("Conventional benchmark — 95% purity / 90% recovery")
    ax.grid(axis="x", alpha=.12)
    return source


def relocate_empty_si_tables(pkg: Path) -> pd.DataFrame:
    active = pkg / "tables" / "SI"
    repository = pkg / "repository_only" / "not_run_tables"
    repository.mkdir(parents=True, exist_ok=True)
    rows = []
    if active.exists():
        for csvp in sorted(active.rglob("*.csv")):
            if "not_run_placeholders" in csvp.parts:
                continue
            try:
                df = pd.read_csv(csvp)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            except Exception:
                continue
            if len(df):
                continue
            moved = []
            stem = csvp.stem
            for candidate in list(csvp.parent.glob(stem + ".*")):
                dest = repository / candidate.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(candidate), str(dest))
                moved.append(str(dest.relative_to(pkg)))
            rows.append({
                "table": stem,
                "status": "NOT_RUN_OR_NO_DATA",
                "publication_action": "excluded from active SI; retained in repository-only archive",
                "reason": "zero data rows",
                "repository_location": ";".join(moved),
            })
    inherited = active / "not_run_placeholders"
    if inherited.exists():
        grouped = {}
        for item in sorted(inherited.iterdir()):
            if item.is_file():
                grouped.setdefault(item.stem, []).append(item)
        for stem, files in grouped.items():
            moved = []
            for item in files:
                dest = repository / item.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(item), str(dest))
                moved.append(str(dest.relative_to(pkg)))
            if not any(r["table"] == stem for r in rows):
                rows.append({
                    "table": stem,
                    "status": "NOT_RUN_OR_NO_DATA",
                    "publication_action": "excluded from active SI; retained in repository-only archive",
                    "reason": "zero data rows",
                    "repository_location": ";".join(moved),
                })
        shutil.rmtree(inherited, ignore_errors=True)
    out = pd.DataFrame(rows, columns=["table","status","publication_action","reason","repository_location"])
    prov = pkg / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    out.to_csv(prov / "EMPTY_PLACEHOLDER_AUDIT.csv", index=False)
    return out



def _legacy_table_key(stem: str):
    """Return a stable sort key for inherited SI table identifiers."""
    import re
    m = re.match(r"Table_S(\d+)([a-z]?)(\d*)_(.*)", stem)
    if not m:
        return (10**6, 99, 99, stem)
    number = int(m.group(1))
    letter = m.group(2)
    suffix_number = int(m.group(3) or 0)
    letter_rank = 0 if not letter else ord(letter) - 96
    return (number, letter_rank, suffix_number, m.group(4))


def renumber_active_si_tables(pkg: Path) -> pd.DataFrame:
    """Renumber active supporting tables consecutively and write a traceable mapping."""
    import re

    active = pkg / "tables" / "SI"
    provenance = pkg / "provenance"
    repository = pkg / "repository_only" / "legacy_tables"
    provenance.mkdir(parents=True, exist_ok=True)
    repository.mkdir(parents=True, exist_ok=True)

    legacy = active / "legacy_main_tables"
    if legacy.exists():
        for item in sorted(legacy.iterdir()):
            if item.is_file():
                dest = repository / item.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(item), str(dest))
        shutil.rmtree(legacy, ignore_errors=True)

    csvs = sorted(active.glob("Table_S*.csv"), key=lambda p: _legacy_table_key(p.stem))
    stems = []
    for path in csvs:
        if path.stem not in stems:
            stems.append(path.stem)

    mappings = []
    rename_ops = []
    for index, stem in enumerate(stems, start=1):
        m = re.match(r"Table_S(\d+)([a-z]?)(\d*)_(.*)", stem)
        if not m:
            continue
        legacy_id = f"S{m.group(1)}{m.group(2)}{m.group(3)}"
        descriptor = m.group(4)
        final_stem = f"Table_S{index}_{descriptor}"
        formats = []
        for source in sorted(active.glob(stem + ".*")):
            temporary = source.with_name("__REN__" + final_stem + source.suffix)
            source.rename(temporary)
            rename_ops.append((temporary, active / (final_stem + source.suffix)))
            formats.append(source.suffix.lstrip("."))
        mappings.append({
            "legacy_table_id": legacy_id,
            "legacy_stem": stem,
            "final_table_id": f"S{index}",
            "final_stem": final_stem,
            "descriptor": descriptor,
            "formats": ";".join(sorted(set(formats))),
        })

    for temporary, destination in rename_ops:
        destination.unlink(missing_ok=True)
        temporary.rename(destination)

    mapping = pd.DataFrame(mappings)
    mapping.to_csv(provenance / "SI_TABLE_RENUMBERING_MAP.csv", index=False)

    numbering_rows = []
    for row in mappings:
        csv_path = active / (row["final_stem"] + ".csv")
        try:
            n_rows = len(pd.read_csv(csv_path))
        except Exception:
            n_rows = np.nan
        numbering_rows.append({
            "table": row["final_table_id"],
            "stem": row["final_stem"],
            "rows": n_rows,
            "status": "PRESENT",
        })
    pd.DataFrame(numbering_rows).to_csv(provenance / "SI_TABLE_NUMBERING_AUDIT.csv", index=False)

    # The curated claim matrix is the final claim record. Keep the SI copy synchronized.
    curated = provenance / "CURATED_CLAIM_EVIDENCE_MATRIX.csv"
    claim_rows = mapping.loc[mapping.get("descriptor", pd.Series(dtype=str)).eq("claim_evidence_matrix")] if len(mapping) else pd.DataFrame()
    if curated.exists() and len(claim_rows):
        claims = pd.read_csv(curated)
        stem = str(claim_rows.iloc[0]["final_stem"])
        base = active / stem
        claims.to_csv(base.with_suffix(".csv"), index=False)
        try:
            claims.to_excel(base.with_suffix(".xlsx"), index=False)
        except Exception:
            pass
        try:
            base.with_suffix(".tex").write_text(claims.to_latex(index=False, escape=True), encoding="utf-8")
        except Exception:
            pass

    note = provenance / "TABLE_NUMBERING_README.md"
    note.write_text(
        "# Supporting-table numbering\n\n"
        "Active supporting tables are numbered consecutively after zero-row placeholders are removed. "
        "`SI_TABLE_RENUMBERING_MAP.csv` records the inherited-to-final identifier mapping. Source-literature "
        "references such as ‘SI Table S1’ are citations to the original publication and are not renumbered.\n",
        encoding="utf-8",
    )
    return mapping

def refine_figures(core, root: Path, pkg: Path, gate, co90, eq90, jac, unc, eq88, flow, qdyn, css, settings, args):
    """Write the final five-figure set and selected supporting figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import FancyBboxPatch

    core.base.configure_matplotlib()
    main = pkg / "figures" / "main"
    si = pkg / "figures" / "SI"
    srcm = pkg / "source_data" / "main"
    srcs = pkg / "source_data" / "SI"
    for p in (main, si, srcm, srcs):
        p.mkdir(parents=True, exist_ok=True)

    atlas = gate["feed_composition_robustness"]
    intrinsic = atlas.get("intrinsic", pd.DataFrame())
    fixed = atlas.get("fixed", pd.DataFrame())
    selected = atlas.get("selected", pd.DataFrame())
    effort = atlas.get("continuous_one_actuator_effort", pd.DataFrame())
    qdyn = core.normalize_quasisteady_disturbance(qdyn)
    coverage = core.scenario_coverage_summary(flow)
    eqsum = core.equipment_boundary_summary(eq88)
    eqsum90 = core.equipment_boundary_summary(eq90)
    uh = core.uncertainty_hierarchy_summary(unc, gate)
    ces = intervention_economy(core, qdyn)

    # Figure 1
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    try:
        core.base._evidence_landscape(axs[0, 0], gate.get("composition_evidence", {}), gate.get("validation", {}), settings)
    except Exception:
        axs[0, 0].text(.5, .5, "Composition-resolved evidence", ha="center", va="center", transform=axs[0, 0].transAxes)
    core.base.panel_label(axs[0, 0], "a")

    data = gate.get("harmonized", {}).get("data", pd.DataFrame())
    ax = axs[0, 1]
    if len(data):
        for gas, ls in [("CO2", "-"), ("N2", "--")]:
            g = data[(data["material"] == "CALF-20") & (data["adsorbate"] == gas)]
            fs = gate.get("fits", {}).get("fits", {}).get(("CALF-20", gas), {})
            model = fs.get("best_model")
            if g.empty or not model:
                continue
            theta = fs["models"][model]["theta"]
            temps = sorted(g["temperature_K"].unique())
            temps = [temps[0], temps[len(temps)//2], temps[-1]] if len(temps) >= 3 else temps
            for temp in temps:
                gg = g[np.isclose(g["temperature_K"], temp)]
                ax.scatter(gg["pressure_bar"], gg["loading_mol_kg"], s=10, alpha=.55)
                pp = np.geomspace(max(float(gg["pressure_bar"].min()), 1e-4), float(gg["pressure_bar"].max()), 150)
                ax.plot(pp, core.base.model_loading(model, pp, np.full_like(pp, temp), theta), ls=ls, lw=1.1, label=f"{gas}, {temp:g} K")
        ax.set_xscale("log")
        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel("Loading (mol kg$^{-1}$)")
        ax.set_title("Thermodynamic foundation")
        ax.grid(alpha=.12)
        ax.legend(frameon=False, fontsize=6, ncol=2)
    core.base.panel_label(ax, "b")

    ax = axs[1, 0]
    pilot = gate.get("validation", {}).get("external_pilot", {}).get("raw", pd.DataFrame())
    medrows = []
    if isinstance(pilot, pd.DataFrame) and len(pilot):
        for y, g in pilot.groupby("feed_co2_fraction"):
            rec = pd.to_numeric(g["recovery_fraction"], errors="coerce").dropna()
            if len(rec):
                medrows.append((100*float(y), 100*float(rec.median()), 100*float(rec.quantile(.25)), 100*float(rec.quantile(.75)), len(rec)))
        for x0, med, q1, q3, n in sorted(medrows):
            ax.errorbar([x0], [med], yerr=np.array([[med-q1], [q3-med]]), fmt="o", capsize=4, lw=1.2)
            ax.text(x0, med+1.5, f"n={n}", ha="center", fontsize=7)
        ax.text(.02, .04, "Heterogeneous pilot conditions; groups are shown independently", transform=ax.transAxes, fontsize=6.8, color="#555555")
        ax.set_xlabel("Feed CO$_2$ (%)")
        ax.set_ylabel("Pilot recovery median (%, IQR)")
        ax.set_title("Independent pilot-scale composition sensitivity")
        ax.grid(alpha=.12)
    core.base.panel_label(ax, "c")

    ax = axs[1, 1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [(.03,.61,.20,.18,"Frozen\nprocess",core._class_color(0)),(.29,.61,.20,.18,"One\nactuator",core._class_color(1)),(.55,.61,.20,.18,"Two coordinated\nactuators",core._class_color(2)),(.81,.61,.16,.18,"Outside declared\ncontrol envelope",core._class_color(3))]
    for x0,y0,w,h,txt,c in boxes:
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc=c,ec="none",alpha=.16))
        ax.text(x0+w/2, y0+h/2, txt, ha="center", va="center", fontsize=9)
    for a,b in [(.23,.29),(.49,.55),(.75,.81)]:
        ax.annotate("", xy=(b,.70), xytext=(a,.70), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
    ax.text(.5, .40, "Minimum intervention needed to preserve\npurity and recovery as the feed changes", ha="center", va="center", fontsize=9)
    ax.text(.5, .22, r"$d^*(y)=\min\|u-u_0\|_0$ subject to $P\geq P^*$ and $R\geq R^*$", ha="center", va="center", fontsize=9.2)
    ax.text(.5, .10, r"$E^*(y)=\min\sum_i w_i\,|\ln(u_i/u_{i,0})|$ at $d=d^*(y)$", ha="center", va="center", fontsize=9.2)
    ax.set_title("Minimum control authority")
    core.base.panel_label(ax, "d")
    _save(fig, main / "Figure_1_evidence_and_control_authority_framework")
    if medrows:
        pd.DataFrame(medrows, columns=["feed_co2_percent","recovery_median_percent","recovery_q25_percent","recovery_q75_percent","n_runs"]).to_csv(srcm/"Figure_1c_pilot_summary.csv", index=False)

    # Figure 2
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    if len(intrinsic):
        x = 100*intrinsic["feed_co2_fraction"]
        axs[0,0].plot(x, intrinsic["q_co2_mol_kg"], label="CO$_2$")
        axs[0,0].plot(x, intrinsic["q_n2_mol_kg"], label="N$_2$")
        axs[0,0].set_ylabel("Mixture loading (mol kg$^{-1}$)")
        axs[0,0].set_title("Competitive mixture loading")
        axs[0,0].legend(frameon=False)
    if len(fixed):
        x = 100*fixed["feed_co2_fraction"]
        axs[0,1].plot(x, 100*(fixed["purity_estimate"]-settings.target_purity), label="purity margin")
        axs[0,1].plot(x, 100*(fixed["recovery_estimate"]-settings.target_recovery), label="recovery margin")
        axs[0,1].axhline(0, ls="--", lw=.9, color="#444444")
        axs[0,1].set_xlim(10, 30)
        axs[0,1].set_ylabel("Constraint margin (percentage points)")
        axs[0,1].set_title("Active-constraint switching")
        axs[0,1].legend(frameon=False, fontsize=7)
    if len(jac):
        j = jac.copy()
        x = 100*j["feed_co2_fraction"]
        bad = ~j.get("jacobian_quality_flag", pd.Series("OK", index=j.index)).astype(str).eq("OK")
        if bad.any():
            transition_lo = float(np.nanmin(x[bad]))
            transition_hi = float(np.nanmax(x[bad]))
            for a in (axs[1,0], axs[1,1]):
                a.axvspan(transition_lo, transition_hi, color="#D9D9D9", alpha=.28, zorder=-4)
            axs[1,0].text(.03,.05,"shaded: policy-transition / non-smooth region",transform=axs[1,0].transAxes,fontsize=6.3,color="#666666")
        axs[1,0].plot(x[~bad], j.loc[~bad,"pressure_response_norm"], marker="o", ms=3, lw=1.1, label="pressure sensitivity")
        axs[1,0].plot(x[~bad], j.loc[~bad,"timing_response_norm"], marker="s", ms=3, lw=1.1, label="timing sensitivity")
        if bad.any():
            axs[1,0].scatter(x[bad], j.loc[bad,"pressure_response_norm"], marker="x", s=22, color="#777777", label="boundary/non-smooth")
        axs[1,0].set_ylabel("Local response norm")
        axs[1,0].set_title("Control-sensitivity magnitude")
        axs[1,0].legend(frameon=False, fontsize=6.5)

        cond = np.maximum(pd.to_numeric(j["condition_number"], errors="coerce"), 1e-6)
        ang = pd.to_numeric(j["actuator_response_angle_deg"], errors="coerce")
        good = ~bad & np.isfinite(cond) & np.isfinite(ang)
        axs[1,1].semilogy(x[good], cond[good], lw=1.5, label="$\\kappa(J)$")
        axs[1,1].axhline(100, ls="--", lw=.8, color="#777777")
        axs[1,1].set_ylabel("Jacobian condition number")
        ax2 = axs[1,1].twinx()
        ax2.plot(x[good], ang[good], ls=":", lw=1.2, color="#666666")
        ax2.set_ylabel("Response-vector angle (deg)")
        axs[1,1].set_title("Local actuator coupling")
        if bad.any():
            axs[1,1].text(.03,.05,"shaded: policy-transition / non-smooth region",transform=axs[1,1].transAxes,fontsize=6.3,color="#666666")
        med_cond = float(np.nanmedian(cond[good])) if good.any() else float("nan")
        med_ang = float(np.nanmedian(ang[good])) if good.any() else float("nan")
        axs[1,1].text(.03,.93,f"median $\\kappa$ ≈ {med_cond:.0f}\nmedian angle ≈ {med_ang:.1f}°\nnear-collinear local response",transform=axs[1,1].transAxes,va="top",fontsize=7,bbox=dict(fc="white",ec="none",alpha=.8,pad=2))
    for i, a in enumerate(axs.flat):
        if a.has_data():
            a.axvline(15, ls=":", lw=.8, color="#555555")
            a.set_xlabel("Feed CO$_2$ (%)")
            a.grid(alpha=.12)
        core.base.panel_label(a, "abcd"[i])
    _save(fig, main / "Figure_2_mechanism_and_controllability")
    intrinsic.to_csv(srcm/"Figure_2a_competitive_loading.csv", index=False)
    fixed.to_csv(srcm/"Figure_2b_constraint_margins.csv", index=False)
    jac.to_csv(srcm/"Figure_2cd_control_jacobian.csv", index=False)

    # Figure 3
    fig, axs = plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    if len(fixed):
        x = 100*fixed["feed_co2_fraction"]
        valid = (pd.to_numeric(fixed["working_capacity_co2_mol_kg"], errors="coerce") > 1e-8) & np.isfinite(pd.to_numeric(fixed["specific_vacuum_work_kJ_mol_CO2"], errors="coerce"))
        frozen_lo = frozen_hi = np.nan
        for lev,_,lo,hi,_,_ in _window_rows(core, atlas.get("windows", pd.DataFrame())):
            if lev == 0: frozen_lo, frozen_hi = lo, hi
        for a in axs.flat:
            if (~valid).any(): a.axvspan(float(x[~valid].min()), float(x[~valid].max()), color="#D9D9D9", alpha=.30, hatch="///", zorder=-5)
            if np.isfinite(frozen_lo) and np.isfinite(frozen_hi): a.axvspan(100*frozen_lo, 100*frozen_hi, color=core._class_color(0), alpha=.10, zorder=-4)
        axs[0,0].plot(x, np.where(valid,100*fixed["purity_estimate"],np.nan)); axs[0,0].axhline(95,ls="--",lw=1,color="#333333"); axs[0,0].set_ylabel("CO$_2$ purity (%)"); axs[0,0].set_title("Frozen process: purity")
        axs[0,1].plot(x, np.where(valid,100*fixed["recovery_estimate"],np.nan)); axs[0,1].axhline(88,ls="--",lw=1,color="#333333",label="88% experiment-anchored target"); axs[0,1].axhline(90,ls=":",lw=1,color="#666666",label="90% conventional benchmark"); axs[0,1].set_ylabel("CO$_2$ recovery (%)"); axs[0,1].set_title("Frozen process: recovery"); axs[0,1].legend(frameon=False,fontsize=6.5)
        axs[1,0].plot(x, np.where(valid,fixed["productivity_mol_kg_h"],np.nan)); axs[1,0].set_ylabel("Productivity (mol kg$^{-1}$ h$^{-1}$)"); axs[1,0].set_title("Frozen process: productivity")
        axs[1,1].plot(x, np.where(valid,fixed["specific_vacuum_work_kJ_mol_CO2"],np.nan)); axs[1,1].set_ylabel("Idealized vacuum-work proxy\n(kJ mol$^{-1}$ CO$_2$)"); axs[1,1].set_title("Frozen process: resource burden")
        if (~valid).any(): axs[0,0].text(.03,.07,"hatched: outside reduced-order applicability / no positive frozen CO$_2$ swing",transform=axs[0,0].transAxes,fontsize=6.4)
        if np.isfinite(frozen_lo) and np.isfinite(frozen_hi):
            width = 100*(frozen_hi-frozen_lo)
            axs[0,0].text(.55,.91,f"frozen operability width ≈ {width:.2f} percentage points",transform=axs[0,0].transAxes,fontsize=7,bbox=dict(fc="white",ec="none",alpha=.8,pad=2))
    for i,a in enumerate(axs.flat):
        a.axvline(15,ls=":",lw=.8,color="#555555")
        a.set_xlabel("Feed CO$_2$ (%)")
        a.grid(alpha=.12)
        core.base.panel_label(a,"abcd"[i])
    _save(fig, main/"Figure_3_frozen_process_offdesign_failure")
    fixed.to_csv(srcm/"Figure_3_source_data.csv", index=False)

    # Figure 4
    fig, axs = plt.subplots(2,2,figsize=(11.4,7.7),constrained_layout=True)
    d = selected.copy()
    if len(d): d = d[d.get("success",False).astype(bool)]
    eqlean = np.nan
    q = eqsum[eqsum.max_adaptation_level.eq(2)] if len(eqsum) else pd.DataFrame()
    if len(q): eqlean = float(q.iloc[0].equipment_realizable_lower_feed_co2_percent)
    if len(d):
        x = 100*d["feed_co2_fraction"]
        axs[0,0].plot(x,d["p_reg_bar"],lw=1.6,marker="o",markevery=max(1,len(d)//18),ms=3)
        axs[0,0].axhline(settings.nominal_regeneration_pressure_bar,ls="--",lw=.8,color="#333333",label="nominal pressure")
        axs[0,0].axhline(args.equipment_min_p_reg_bar,ls=":",lw=.9,color="#777777",label="screening floor")
        axs[0,0].set_yscale("log"); axs[0,0].set_ylabel("Selected regeneration pressure (bar)"); axs[0,0].legend(frameon=False,fontsize=6.5)
        axs[0,1].plot(x,d["timing_scale"],lw=1.6,marker="o",markevery=max(1,len(d)//18),ms=3)
        axs[0,1].axhline(1,ls="--",lw=.8,color="#333333")
        axs[0,1].axhline(args.equipment_min_timing_scale,ls=":",lw=.8,color="#777777")
        axs[0,1].axhline(args.equipment_max_timing_scale,ls=":",lw=.8,color="#777777")
        axs[0,1].set_ylabel("Selected timing scale")
        if len(effort) and "actuator_effort_delta_pressure_minus_timing_continuous" in effort:
            ex=100*effort["feed_co2_fraction"]; ey=effort["actuator_effort_delta_pressure_minus_timing_continuous"]
            axs[1,0].plot(ex,ey,lw=1.6); axs[1,0].axhline(0,ls="--",lw=.8,color="#444444"); axs[1,0].fill_between(ex,-.05,.05,alpha=.08)
            axs[1,0].set_ylabel("Pressure effort − timing effort")
            axs[1,0].text(.03,.92,"<0 pressure preferred\n≈0 comparable effort\n>0 timing preferred",transform=axs[1,0].transAxes,va="top",fontsize=7)
            exv=np.asarray(ex,float); eyv=np.asarray(pd.to_numeric(ey,errors="coerce"),float); finite=np.isfinite(exv)&np.isfinite(eyv)
            exv=exv[finite]; eyv=eyv[finite]
            switches=[]
            for k in range(max(0,len(eyv)-1)):
                if eyv[k] == 0 or eyv[k]*eyv[k+1] < 0:
                    x0,x1=exv[k],exv[k+1]; y0,y1=eyv[k],eyv[k+1]
                    xs=x0 if y0==0 else x0+(0-y0)*(x1-x0)/(y1-y0)
                    switches.append(float(xs))
            if switches:
                preferred=[v for v in switches if v>15]
                switch_x=preferred[0] if preferred else switches[0]
                axs[1,0].axvline(switch_x,ls="-.",lw=.9,color="#777777")
                axs[1,0].annotate("preferred-actuator switch",xy=(switch_x,0),xytext=(switch_x+1.0,.22*np.nanmax(np.abs(eyv)) if len(eyv) else .1),fontsize=6.5,color="#555555",arrowprops=dict(arrowstyle="-",lw=.7,color="#777777"))
                axs[1,1].axvline(switch_x,ls="-.",lw=.9,color="#777777")
        if len(fixed):
            basew=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["specific_vacuum_work_kJ_mol_CO2"])
            basep=np.interp(d["feed_co2_fraction"],fixed["feed_co2_fraction"],fixed["productivity_mol_kg_h"])
            axs[1,1].plot(x,100*(d["specific_vacuum_work_kJ_mol_CO2"]/np.maximum(basew,1e-12)-1),label="vacuum-work proxy")
            axs[1,1].plot(x,100*(d["productivity_mol_kg_h"]/np.maximum(basep,1e-12)-1),label="productivity")
            axs[1,1].axhline(0,ls="--",lw=.8,color="#444444"); axs[1,1].set_ylabel("Change from frozen policy (%)"); axs[1,1].legend(frameon=False,fontsize=7)
    titles=["Minimum pressure intervention","Minimum timing intervention","Relative actuation effort / actuator competition","Resource and productivity consequence"]
    for i,a in enumerate(axs.flat):
        a.axvline(15,ls=":",lw=.8,color="#555555")
        if np.isfinite(eqlean): a.axvline(eqlean,ls="--",lw=.9,color="#777777")
        a.set_xlabel("Feed CO$_2$ (%)"); a.grid(alpha=.12); a.set_title(titles[i]); core.base.panel_label(a,"abcd"[i])
    if np.isfinite(eqlean):
        axs[0,0].text(.03,.06,f"equipment-realizable lean limit ≈ {eqlean:.2f}%",transform=axs[0,0].transAxes,fontsize=6.7)
        axs[0,1].text(.03,.06,f"equipment-realizable lean limit ≈ {eqlean:.2f}%",transform=axs[0,1].transAxes,fontsize=6.7)
    axs[0,0].text(.72,.88,"nominal feed",transform=axs[0,0].transAxes,fontsize=6.5,color="#555555")
    _save(fig, main/"Figure_4_minimum_intervention_policy")
    d.to_csv(srcm/"Figure_4_selected_policy.csv", index=False)
    effort.to_csv(srcm/"Figure_4_actuator_competition.csv", index=False)
    eqsum.to_csv(srcm/"Figure_4_equipment_boundaries.csv", index=False)

    # Figure 5
    fig, axs = plt.subplots(2,2,figsize=(11.8,8.1),constrained_layout=True)
    _ladder(core, axs[0,0], atlas.get("windows",pd.DataFrame()), "Experiment-anchored specification — 95% purity / 88% recovery", equipment=eqsum)
    if len(uh):
        for lev in (0,1,2):
            kin=uh[(uh.authority_level==lev)&uh.layer.astype(str).eq("kinetic_front_utilization_sensitivity")]
            mc=uh[(uh.authority_level==lev)&uh.layer.astype(str).eq("calibration_conditioned_monte_carlo")]
            if len(kin):
                r=kin.iloc[0]; axs[0,0].plot([r.lower_boundary_min_percent,r.lower_boundary_max_percent],[lev-.15,lev-.15],lw=5,alpha=.22,color=core._class_color(lev),solid_capstyle="butt")
            if len(mc):
                r=mc.iloc[0]; axs[0,0].plot([r.lower_boundary_min_percent,r.lower_boundary_max_percent],[lev-.05,lev-.05],lw=2.5,color="#111111",solid_capstyle="butt")
        axs[0,0].text(.02,.03,"dark: calibration-conditioned parameter interval\npale: kinetic/model sensitivity range",transform=axs[0,0].transAxes,fontsize=6.3)
    figure5b_source = _plot_conventional_95_90(core, axs[0,1], co90, eqsum90)

    if len(flow):
        piv=flow.pivot(index="feed_flow_scale",columns="feed_co2_fraction",values="minimum_control_authority_level").sort_index()
        xx=100*piv.columns.to_numpy(float); yy=piv.index.to_numpy(float); X,Y=np.meshgrid(xx,yy)
        axs[1,0].pcolormesh(X,Y,piv.values,shading="nearest",vmin=-.5,vmax=3.5,cmap=ListedColormap([core._class_color(i) for i in range(4)]))
        axs[1,0].scatter(np.repeat(xx,len(yy)),np.tile(yy,len(xx)),s=3,c="k",alpha=.12)
        axs[1,0].set_xlabel("Feed CO$_2$ (%)"); axs[1,0].set_ylabel("Feed-flow scale")
        vals={int(r.control_authority_ceiling):float(r.scenario_coverage_percent) for _,r in coverage.iterrows()}
        axs[1,0].set_title(f"Control authority across composition and throughput\non-spec coverage: {vals.get(0,np.nan):.0f}% → {vals.get(1,np.nan):.0f}% → {vals.get(2,np.nan):.0f}%")
        for lev,lab in {0:"Frozen",1:"≤1 actuator",2:"≤2 actuators",3:"Outside"}.items():
            pts=flow[pd.to_numeric(flow.minimum_control_authority_level,errors="coerce").eq(lev)]
            if len(pts): axs[1,0].text(100*pts.feed_co2_fraction.median(),pts.feed_flow_scale.median(),lab,ha="center",va="center",fontsize=7,bbox=dict(fc="white",ec="none",alpha=.58,pad=1.5))
    if len(qdyn):
        qd=qdyn.copy(); qd["spec_margin_pp"]=100*np.minimum(pd.to_numeric(qd.get("purity",np.nan),errors="coerce")-settings.target_purity,pd.to_numeric(qd.get("recovery",np.nan),errors="coerce")-settings.target_recovery)
        for pol,g in qd.groupby("policy"):
            gg=g.sort_values("segment"); axs[1,1].plot(gg["segment"],gg["spec_margin_pp"],marker="o",label=str(gg["policy_display"].iloc[0]))
        axs[1,1].axhline(0,ls="--",lw=.9,color="#333333"); axs[1,1].set_xticks([1,2,3,4],["15%","12%","18%","15%"]); axs[1,1].set_xlabel("Feed-composition sequence"); axs[1,1].set_ylabel("Minimum purity/recovery margin (percentage points)"); axs[1,1].set_title("Quasi-steady disturbance: specification recovery"); axs[1,1].legend(frameon=False,fontsize=6.2,loc="best"); axs[1,1].grid(alpha=.12)
        box=[]
        for pol in ["frozen","minimum_authority","full_reoptimization_grid"]:
            rr=ces[ces.policy.eq(pol)]
            if len(rr): box.append(f"{rr.iloc[0].policy_display}: {int(rr.iloc[0].on_spec_states)}/{int(rr.iloc[0].total_states)} on-spec")
        rr=ces[ces.policy.eq("minimum_authority")]
        if len(rr) and np.isfinite(rr.iloc[0].minimum_authority_movement_reduction_vs_reoptimization_percent): box.append(f"minimum authority: ~{rr.iloc[0].minimum_authority_movement_reduction_vs_reoptimization_percent:.0f}% less actuator movement")
        axs[1,1].text(.02,.03,"\n".join(box),transform=axs[1,1].transAxes,fontsize=6.4,bbox=dict(boxstyle="round,pad=.3",fc="white",ec="0.75",alpha=.86))
    for i,a in enumerate(axs.flat): core.base.panel_label(a,"abcd"[i])
    _save(fig, main/"Figure_5_nested_operability_and_generalization")
    flow.to_csv(srcm/"Figure_5c_composition_flow.csv", index=False)
    qdyn.to_csv(srcm/"Figure_5d_quasisteady_disturbance.csv", index=False)
    coverage.to_csv(srcm/"Figure_5c_scenario_coverage_summary.csv", index=False)
    ces.to_csv(srcm/"Figure_5d_control_effort_summary.csv", index=False)
    uh.to_csv(srcm/"Figure_5a_uncertainty_hierarchy_summary.csv", index=False)
    for obsolete in [srcm/"Figure_5b.csv", srcm/"Figure_5b_operability.csv"]:
        obsolete.unlink(missing_ok=True)
    figure5b_source.to_csv(srcm/"Figure_5b_conventional_95_90_operability.csv", index=False)

    # Supporting figures
    ext = core.fixed_external_boundaries(gate)
    if len(ext):
        fig, ax = plt.subplots(figsize=(8.8,4.8),constrained_layout=True)
        for _, g in ext.groupby(ext.get("source_file",pd.Series("Independent literature parameterization",index=ext.index))):
            gg=g.sort_values("max_adaptation_level"); ax.plot(gg.max_adaptation_level,100*gg.lower_feed_co2_fraction,marker="o",label="Independent literature parameterization")
        w=atlas.get("windows",pd.DataFrame())
        if len(w): ax.plot(w.max_adaptation_level,100*w.lower_feed_co2_fraction,marker="s",ls="--",label="Primary parameterization")
        ax.set_xticks([0,1,2],["Frozen","≤1 actuator","≤2 actuators"]); ax.set_ylabel("Lean-side boundary (feed CO$_2$ %)"); ax.set_title("Independent thermodynamic-source sensitivity"); ax.grid(alpha=.12); ax.legend(frameon=False)
        _save(fig, si/"Figure_S16_external_thermodynamic_source_sensitivity")
        ext.to_csv(srcs/"Figure_S16_external_thermodynamic_source_sensitivity.csv",index=False)

    # Rename the uncertainty hierarchy to S24; CSS status remains repository-only.
    old = si/"Figure_S25_uncertainty_hierarchy.png"
    if len(uh):
        layers=["calibration_conditioned_monte_carlo","independent_thermodynamic_parameterization","thermodynamic_model_discrepancy_stress","kinetic_front_utilization_sensitivity"]
        labels={"calibration_conditioned_monte_carlo":"Calibration-conditioned parameter interval","independent_thermodynamic_parameterization":"Independent thermodynamic source","thermodynamic_model_discrepancy_stress":"Thermodynamic discrepancy stress","kinetic_front_utilization_sensitivity":"Kinetic/front sensitivity"}
        fig,ax=plt.subplots(figsize=(9.0,5.3),constrained_layout=True); ytick=[]; ylab=[]; y=0
        for lev in (0,1,2):
            for layer in layers:
                g=uh[(uh.authority_level==lev)&uh.layer.astype(str).eq(layer)]
                if g.empty: continue
                r=g.iloc[0]; lo=r.lower_boundary_min_percent; hi=r.lower_boundary_max_percent; ce=r.lower_boundary_central_percent
                ax.plot([lo,hi],[y,y],lw=3 if layer!="calibration_conditioned_monte_carlo" else 1.8,alpha=.7); ax.plot(ce,y,"o",ms=4)
                ytick.append(y); ylab.append(f"{r.control_authority} — {labels[layer]}"); y+=1
            y+=.45
        ax.set_yticks(ytick,ylab,fontsize=6.5); ax.set_xlabel("Lean-side boundary (feed CO$_2$ %)"); ax.set_title("Boundary uncertainty and sensitivity hierarchy"); ax.grid(axis="x",alpha=.12)
        _save(fig,si/"Figure_S24_uncertainty_hierarchy")
        uh.to_csv(srcs/"Figure_S24_uncertainty_hierarchy.csv",index=False)
    for p in list(si.glob("Figure_S25_uncertainty_hierarchy.*")) + list(srcs.glob("Figure_S25_uncertainty_hierarchy*")):
        p.unlink(missing_ok=True)


def finalize_tables_and_package(core, root: Path, pkg: Path, gate, co90, eq90, eq88, flow, qdyn, unc, literature, css, args):
    """Apply final manuscript-facing table, provenance and archive cleanup."""
    tables = pkg/"tables"; main = tables/"main"; si = tables/"SI"; src = pkg/"source_data"/"main"; prov = pkg/"provenance"
    for p in (main,si,src,prov): p.mkdir(parents=True,exist_ok=True)

    eq88s=core.equipment_boundary_summary(eq88); eq90s=core.equipment_boundary_summary(eq90)
    numeric=[]; display=[]
    specs=[("95% purity / 88% recovery",gate["feed_composition_robustness"].get("windows",pd.DataFrame()),eq88s,"experiment-anchored"),("95% purity / 90% recovery",co90.get("windows",pd.DataFrame()),eq90s,"conventional benchmark")]
    for spec,w,eqs,role in specs:
        rows={r[0]:r for r in core._window_rows(w)}
        for lev,label in [(0,"Frozen"),(1,"≤1 actuator"),(2,"≤2 actuators")]:
            r=rows.get(lev)
            if r is None:
                lo=hi=np.nan; lstat=ustat="not feasible"
            else:
                _,_,lo,hi,lstat,ustat=r
            er=eqs[eqs.max_adaptation_level.eq(lev)].iloc[0] if len(eqs) and (eqs.max_adaptation_level==lev).any() else None
            elo=float(er.equipment_realizable_lower_feed_co2_percent) if er is not None else np.nan
            numeric.append({"specification":spec,"specification_role":role,"control_authority":label,"lean_boundary_feed_co2_percent":100*lo if np.isfinite(lo) else np.nan,"rich_boundary_feed_co2_percent":100*hi if np.isfinite(hi) else np.nan,"equipment_realizable_lean_boundary_percent":elo,"lower_boundary_status":lstat,"upper_boundary_status":ustat})
            if np.isfinite(lo): lean=f"{100*lo:.2f}%"
            else: lean="No feasible interval" if lev==0 and "90%" in spec else "—"
            if np.isfinite(hi) and "censor" not in str(ustat).lower(): rich=f"{100*hi:.2f}%"
            elif "censor" in str(ustat).lower(): rich=">40% (search-domain limit)"
            else: rich="—"
            equip=f"{elo:.2f}%" if np.isfinite(elo) else "—"
            display.append({"Specification":spec,"Control authority":label,"Lean boundary (feed CO2)":lean,"Rich boundary (feed CO2)":rich,"Equipment-realizable lean boundary":equip,"Evidence basis":"Reduced-order adaptive boundary refinement"})
    num=pd.DataFrame(numeric); human=pd.DataFrame(display)
    num.to_csv(src/"Table_1_minimum_control_authority_summary_numeric.csv",index=False)
    human.to_csv(main/"Table_1_minimum_control_authority_summary.csv",index=False)
    for p in main.glob("*.pkl"): p.unlink(missing_ok=True)

    effort=intervention_economy(core,qdyn)
    effort.to_csv(si/"Table_S36_disturbance_control_effort_summary.csv",index=False)
    eq90s.to_csv(si/"Table_S37_equipment_realizability_95_90.csv",index=False)
    equipment_basis_table(args).to_csv(prov/"equipment_realizability_basis.csv",index=False)

    # CSS status is retained for reproducibility but removed from journal SI.
    repo = pkg/"high_fidelity"/"repository_only"; repo.mkdir(parents=True,exist_ok=True)
    for stem in ["Figure_S24_high_fidelity_css_status"]:
        for p in list((pkg/"figures"/"SI").glob(stem+".*")) + list((pkg/"source_data"/"SI").glob(stem+"*")):
            shutil.move(str(p), str(repo/p.name))
    for name in ["Table_S32_css_gate_status.csv"]:
        p=si/name
        if p.exists(): shutil.move(str(p),str(repo/p.name))

    # Remove inherited table previews from the manuscript-facing package.
    if (pkg/"main_tables").exists():
        shutil.rmtree(pkg/"main_tables")
    for d in list(pkg.rglob("visual_previews")):
        if d.is_dir():
            shutil.rmtree(d)

    relocate_empty_si_tables(pkg)

    claim=prov/"CURATED_CLAIM_EVIDENCE_MATRIX.csv"
    if claim.exists():
        c=pd.read_csv(claim)
        if "claim_id" in c:
            c.loc[c.claim_id.eq("C1"),"claim"]="The feed-composition locations and incremental gains of the zero-, one- and two-actuator operability transitions are quantified under explicit purity/recovery specifications."
            c.loc[c.claim_id.eq("C7"),"figure"]="Figure 5A; Figure S21"
            c.loc[c.claim_id.eq("C9"),"figure"]="Repository-only CSS status; not a journal figure"
        c.to_csv(claim,index=False)

    methods = f"""# Methods and interpretation notes

## Minimum control authority

{authority_definition_text()}

The cardinality term distinguishes passive robustness from adaptive operability. The effort term compares feasible policies with the same minimum cardinality; it is not a general optimal-control cost function. Passive robustness refers to the frozen cycle, adaptive operability to specification recovery using allowed controls, and equipment-realizable adaptive operability to solutions that also satisfy the declared screening bounds.

For the conventional 95% purity / 90% recovery benchmark, no frozen feasible interval is found in the evaluated feed-composition domain. The nominal 15% feed is off-spec because recovery is below 90%, while adaptive policies recover feasibility.

## Equipment-realizable authority

The screening limits used here are declared engineering assumptions rather than measured hardware limits: regeneration pressure >= {args.equipment_min_p_reg_bar:.3f} bar and timing scale between {args.equipment_min_timing_scale:.2f} and {args.equipment_max_timing_scale:.2f}. They are user-configurable and are reported separately from unconstrained mathematical authority.

## Uncertainty

Calibration-conditioned parameter intervals, independent thermodynamic-source sensitivity, discrepancy stress tests and kinetic/front sensitivity are kept separate. The kinetic/front range is a declared non-probabilistic sensitivity, not a confidence interval.

## High-fidelity cycle model

The CSS extension is not used to support manuscript claims unless independent same-rig validation passes. Numerical timeout is treated as indeterminate rather than as physical falsification.

## Energy

Energy is reported as an idealized vacuum-work proxy. It is not a validated vacuum-machine energy model.
"""
    (pkg/"METHODS_AND_LIMITATIONS.md").write_text(methods,encoding="utf-8")

    readme=f"""# P10 feed-composition robustness

This directory contains the final figures, tables, source data and supporting evidence for the CALF-20 feed-composition operability analysis.

The study asks how much operating intervention is required to maintain explicit purity and recovery specifications as feed composition and throughput vary. The frozen reduced-order baseline is preserved; later analyses add conventional 95/90 targets, local actuator sensitivity, equipment screening, composition-throughput coverage and a quasi-steady disturbance comparison.

## Contents

- `figures/main` — Figures 1–5.
- `figures/SI` — supporting figures used in the journal SI.
- `tables/main` — one manuscript table.
- `tables/SI` — supporting tables.
- `source_data` — figure and table source data.
- `validation`, `uncertainty`, `engineering`, `control_analysis`, `generalization`, `disturbance` — supporting analyses.
- `provenance` — literature, equipment-assumption and file-integrity records.
- `high_fidelity/repository_only` — CSS status retained for reproducibility but not used as journal evidence unless the validation gate passes.

See `METHODS_AND_LIMITATIONS.md` for the definitions and scope of the reported quantities.
"""
    (pkg/"README.md").write_text(readme,encoding="utf-8")



def sanitize_publication_paths(pkg: Path, repository_root: Path, results_root: Path) -> None:
    """Remove workstation-specific absolute paths from publication files."""
    import re
    drive_path = re.compile(r"[A-Za-z]:\\[^\r\n,\"']+")
    for path in pkg.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".tex"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text.replace(str(repository_root), "<repository>").replace(str(results_root), "<results>")
        updated = updated.replace(str(repository_root).replace("/", "\\"), "<repository>")
        updated = updated.replace(str(results_root).replace("/", "\\"), "<results>")
        updated = drive_path.sub(lambda m: Path(m.group(0).replace("\\", "/")).name or "<local path>", updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")

def write_si_numbering_audit(pkg: Path) -> pd.DataFrame:
    """Document intentional gaps retained to match the manuscript/SI numbering."""
    fig_dir = pkg / "figures" / "SI"
    stems = sorted({p.stem for p in fig_dir.glob("Figure_S*.png")})
    present = set()
    for stem in stems:
        m = __import__('re').match(r"Figure_S(\d+)([A-Za-z]?)_", stem)
        if m:
            present.add((int(m.group(1)), m.group(2).lower()))
    max_n = max((n for n,_ in present), default=0)
    rows=[]
    for n in range(1, max_n+1):
        variants=sorted(s for nn,s in present if nn==n)
        if variants:
            rows.append({"si_number":f"S{n}","status":"PRESENT","note":"Figure number retained to match manuscript/SI cross-references."})
            for s in variants:
                if s:
                    rows.append({"si_number":f"S{n}{s}","status":"PRESENT","note":"Lettered supporting figure retained intentionally."})
        else:
            rows.append({"si_number":f"S{n}","status":"RESERVED_NOT_SUBMITTED","note":"Number intentionally unused in the final figure set; no file is missing from the submission bundle."})
    out=pd.DataFrame(rows)
    (pkg/"provenance").mkdir(parents=True,exist_ok=True)
    out.to_csv(pkg/"provenance"/"SI_FIGURE_NUMBERING_AUDIT.csv",index=False)
    return out


def submission_manifest(pkg: Path) -> pd.DataFrame:
    rows=[]
    keep=("README.md","METHODS_AND_LIMITATIONS.md","figures/","tables/","source_data/","provenance/")
    for p in sorted(pkg.rglob("*")):
        if not p.is_file(): continue
        rel=str(p.relative_to(pkg)).replace("\\","/")
        if rel.startswith("high_fidelity/repository_only"):
            continue
        if rel in keep or any(rel.startswith(k) for k in keep if k.endswith("/")):
            rows.append({"relative_path":rel,"bytes":p.stat().st_size})
    return pd.DataFrame(rows)
