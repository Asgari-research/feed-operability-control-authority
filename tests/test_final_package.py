from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("p10_main",ROOT/"p10_feed_composition_robustness.py")
main=importlib.util.module_from_spec(spec); spec.loader.exec_module(main)
fin=main.finalize; core=main.core


def test_clean_public_surface():
    assert main.VERSION=="6.6.0"
    assert not (ROOT/"src/p10/_frozen").exists()
    names=[p.name for p in (ROOT/"src/p10").glob("*.py")]
    assert not any("v4" in n.lower() or "v5" in n.lower() or "v6_1" in n.lower() for n in names)


def test_single_data_tree():
    assert (ROOT/"data/input_folder/equilibrium.csv").exists()
    assert not (ROOT/"src/p10/data").exists()
    assert not (ROOT/"src/p10/_frozen/data").exists()


def test_no_duplicate_stress_policy():
    files=list((ROOT/"data/external_equilibrium_sensitivity").glob("*stress_policy*.csv"))
    assert [p.name for p in files]==["CALF20_thermodynamic_discrepancy_stress_policy.csv"]


def test_public_header_is_concise():
    text=(ROOT/"p10_feed_composition_robustness.py").read_text(encoding="utf-8")
    header=text.split("from __future__",1)[0]
    assert len(header.splitlines())<=5
    assert "high-ceiling" not in text.lower()


def test_authority_definition():
    text=fin.authority_definition_text()
    assert "number of manipulated variables" in text
    assert "normalized movement" in text


def test_95_90_source_marks_no_frozen_window():
    co90={"windows":pd.DataFrame([
        {"max_adaptation_level":0,"lower_feed_co2_fraction":np.nan,"upper_feed_co2_fraction":np.nan,"lower_boundary_status":"no feasible interval","upper_boundary_status":"no feasible interval"},
        {"max_adaptation_level":1,"lower_feed_co2_fraction":.134,"upper_feed_co2_fraction":.40,"lower_boundary_status":"refined","upper_boundary_status":"right_censored"},
        {"max_adaptation_level":2,"lower_feed_co2_fraction":.106,"upper_feed_co2_fraction":.40,"lower_boundary_status":"refined","upper_boundary_status":"right_censored"},
    ]),"fixed":pd.DataFrame([{"feed_co2_fraction":.15,"purity_estimate":.955,"recovery_estimate":.881}]),"selected":pd.DataFrame([{"feed_co2_fraction":.15,"adaptation_level":1,"p_reg_bar":.049,"timing_scale":1.0,"purity_estimate":.955,"recovery_estimate":.9274}])}
    eq=pd.DataFrame([{"max_adaptation_level":1,"equipment_realizable_lower_feed_co2_percent":13.4},{"max_adaptation_level":2,"equipment_realizable_lower_feed_co2_percent":10.95}])
    out=fin.conventional_95_90_source_table(core,co90,eq)
    r=out[out.control_authority_level.eq(0)].iloc[0]
    assert np.isnan(r.lower_boundary_feed_co2_percent)
    assert np.isclose(r.nominal_recovery_shortfall_to_90_percent_pp,1.9)


def test_empty_si_relocation_and_audit(tmp_path):
    pkg=tmp_path/"publication"; si=pkg/"tables/SI"; si.mkdir(parents=True); (pkg/"provenance").mkdir(parents=True)
    pd.DataFrame(columns=["x"]).to_csv(si/"Table_S19_breakthrough_validation.csv",index=False)
    audit=fin.relocate_empty_si_tables(pkg)
    assert len(audit)==1
    assert not (si/"Table_S19_breakthrough_validation.csv").exists()
    assert (pkg/"repository_only/not_run_tables/Table_S19_breakthrough_validation.csv").exists()


def test_claim_reference_patch_present():
    text=(ROOT/"src/p10/finalize.py").read_text(encoding="utf-8")
    assert 'claim_id.eq("C7")' in text and 'Figure 5A; Figure S21' in text


def test_clean_data_filenames():
    names=[p.name for p in (ROOT/"data/literature_evidence").iterdir() if p.is_file()]
    assert not any("_v2_3_0" in n or "_v5" in n.lower() for n in names)


def test_public_archive_names():
    text=(ROOT/"p10_feed_composition_robustness.py").read_text(encoding="utf-8")
    assert "P10_SUBMISSION_BUNDLE.zip" in text
    assert "P10_REPRODUCIBILITY_ARCHIVE.zip" in text


def test_public_material_has_no_release_history_labels():
    blocked=("V5_HIGH_CEILING","V6.1","V6.2","V6.3","V6.4","v2.3.0","high-ceiling")
    paths=[ROOT/"README.md",ROOT/"RUN_INSTRUCTIONS.md",ROOT/"p10_feed_composition_robustness.py",ROOT/"p10_css_validation.py"]
    paths += [x for base in (ROOT/"docs",ROOT/"data") for x in base.rglob("*") if x.is_file() and x.suffix.lower() in {".md",".txt",".csv",".json",".tex",".bib"}]
    for path in paths:
        text=path.read_text(encoding="utf-8",errors="ignore")
        assert not any(term in text for term in blocked), f"stale release label in {path}"


def test_archive_builder_creates_distinct_clean_archives(tmp_path):
    import logging, zipfile
    root=tmp_path/"run"; pkg=root/"publication"
    for d in ["figures/main","tables/main","source_data/main","provenance"]:
        (pkg/d).mkdir(parents=True,exist_ok=True)
    for name in ["README.md","METHODS_AND_LIMITATIONS.md","PUBLICATION_READINESS_STATUS.md"]:
        (pkg/name).write_text(name+"\n",encoding="utf-8")
    (pkg/"figures/main/Figure_1.txt").write_text("figure",encoding="utf-8")
    (pkg/"tables/main/Table_1.csv").write_text("a\n1\n",encoding="utf-8")
    (pkg/"source_data/main/Figure_1.csv").write_text("x\n1\n",encoding="utf-8")
    (pkg/"provenance/test.csv").write_text("x\n1\n",encoding="utf-8")
    (root/"analysis").mkdir(parents=True); (root/"analysis/dummy.txt").write_text("analysis",encoding="utf-8")
    (root/"run_summary").mkdir(parents=True); (root/"run_summary/dummy.json").write_text("{}",encoding="utf-8")
    logger=logging.getLogger("test_archive_builder"); logger.handlers.clear(); logger.addHandler(logging.NullHandler())
    status=main._create_archives(root,pkg,main.Ledger(root,logger))
    assert status["status"]=="PASS"
    for name in ["P10_SUBMISSION_BUNDLE.zip","P10_REPRODUCIBILITY_ARCHIVE.zip"]:
        z=root/"archives"/name
        with zipfile.ZipFile(z) as fh:
            assert fh.testzip() is None


def test_si_table_renumbering_is_contiguous(tmp_path):
    pkg=tmp_path/"publication"
    si=pkg/"tables/SI"; si.mkdir(parents=True)
    prov=pkg/"provenance"; prov.mkdir(parents=True)
    pd.DataFrame({"x":[1]}).to_csv(si/"Table_S1_input.csv",index=False)
    pd.DataFrame({"x":[2]}).to_csv(si/"Table_S1a_parser.csv",index=False)
    pd.DataFrame({"x":[3]}).to_csv(si/"Table_S6b_anchor.csv",index=False)
    mapping=fin.renumber_active_si_tables(pkg)
    assert list(mapping["final_table_id"])==["S1","S2","S3"]
    assert (si/"Table_S1_input.csv").exists()
    assert (si/"Table_S2_parser.csv").exists()
    assert (si/"Table_S3_anchor.csv").exists()
    assert (prov/"SI_TABLE_RENUMBERING_MAP.csv").exists()


def test_frozen_final_results_snapshot_has_no_empty_active_si():
    result=ROOT/"results/final_submission"
    si=result/"tables/SI"
    csvs=sorted(si.glob("Table_S*.csv"),key=lambda p:int(p.stem.split("_",2)[1][1:]))
    assert len(csvs)==59
    assert [int(p.stem.split("_",2)[1][1:]) for p in csvs]==list(range(1,60))
    for p in csvs:
        assert len(pd.read_csv(p))>0, p.name
    audit=pd.read_csv(result/"provenance/EMPTY_PLACEHOLDER_AUDIT.csv")
    assert len(audit)==8


def test_emptydata_si_relocation(tmp_path):
    pkg=tmp_path/"publication"; si=pkg/"tables/SI"; si.mkdir(parents=True)
    (si/"Table_S6_breakthrough_anchor_summary.csv").write_text("\n",encoding="utf-8")
    (si/"Table_S6_breakthrough_anchor_summary.tex").write_text("",encoding="utf-8")
    audit=fin.relocate_empty_si_tables(pkg)
    assert len(audit)==1
    assert not (si/"Table_S6_breakthrough_anchor_summary.csv").exists()
    assert (pkg/"repository_only/not_run_tables/Table_S6_breakthrough_anchor_summary.csv").exists()
