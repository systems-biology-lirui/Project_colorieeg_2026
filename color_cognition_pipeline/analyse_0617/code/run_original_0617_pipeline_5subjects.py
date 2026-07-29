"""Run the existing analyse_0617 scripts for five subjects without changing logic.

Only the subject list and workspace root are injected at runtime.  Each stage is
executed from the original source file, including its normal ``__main__`` path.
test004 is intentionally excluded because the legacy Step1_1 requires an
anatomical localisation workbook.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path("/home/lirui/liulab_project/ieeg/Project_colorieeg_2026")
ANALYSE = ROOT / "color_cognition_pipeline" / "analyse_0617"
CODE = ANALYSE / "code"
RUN = ANALYSE / "run_5subjects_original"
FEATURE = RUN / "feature"
SUBJECTS = ["test001", "test002", "test003", "test005", "test006"]

STAGES = {
    "step1_1": "step1_1_select_channel_extended.py",
    "step1_2": "step1_2_color_selectivity.py",
    "step1_2_corr": "step1_2_color_selectivity_correlation.py",
    "step2_1": "step2_1_memory_color_significance.py",
    "step2_2": "step2_2_memory_color_decoding_glmm.py",
    "step2_3": "step2_3_single_electrode_decoding_correlation.py",
    "step3_1": "step3_1_color_block_decoding.py",
    "step3_2": "step3_2_cross_decoding_generalization.py",
    "step3_2_union": "step3_2_cross_decoding_generalization_union.py",
    "step3_3_single": "step3_3_single_electrode_generalization.py",
    "step4": "step4_real_fake_color_decoding.py",
    "step5": "step5_memory_color_clusters_decoding.py",
    "step6": "step6_temporal_pole_true_fake_erp_difference.py",
    "step7": "step7_color_with_sti_electrode_analyses.py",
    "step8_2": "step8_2_whole_brain_erp_strategy_table_and_glassbrain.py",
    "step8_cws": "step8_cws_brain_and_memory_erp_latency.py",
}


def _link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(source)


def prepare_feature_workspace(subjects):
    """Recreate Step0's data contract, using the original ERP/HG MAT sources."""
    for subject in subjects:
        for task in (1, 2, 3):
            if subject in {"test001", "test002", "test003"}:
                erp = ANALYSE / "feature" / subject / f"task{task}_ERP_epoched.mat"
                hg = ANALYSE / "feature" / subject / f"task{task}_hg_subband.mat"
            else:
                erp = ROOT / "processed_data" / subject / f"task{task}_ERP_epoched.mat"
                hg = ROOT / "color_cognition_pipeline" / "feature" / "subband_60_150" / subject / f"task{task}_hg_subband.mat"
            if not erp.exists():
                raise FileNotFoundError(f"Missing legacy ERP input: {erp}")
            if not hg.exists():
                raise FileNotFoundError(f"Missing legacy 60–150 Hz HG input: {hg}")
            _link(erp, FEATURE / subject / erp.name)
            _link(hg, FEATURE / subject / hg.name)


def run_original(stage: str):
    source_path = CODE / STAGES[stage]
    source = source_path.read_text(encoding="utf-8")
    # These are the only two runtime substitutions: an isolated results root
    # and the five-subject list. The analysis code below remains byte-for-byte
    # the original script.
    replaced = False
    for orig in [
        "analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')",
        "analyse_dir = os.path.join(pipeline, 'analyse_0617')",
        "analyse    = os.path.join(pipeline, 'analyse_0617')",
        "analyse = os.path.join(pipeline, 'analyse_0617')",
        "analyse_dir = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617')"
    ]:
        if orig in source:
            if "analyse_dir" in orig:
                source = source.replace(orig, "analyse_dir = os.environ['ANALYSE_0617_RUNTIME_ROOT']")
            else:
                source = source.replace(orig, "analyse = os.environ['ANALYSE_0617_RUNTIME_ROOT']")
            replaced = True
    if not replaced:
        source = source.replace("os.path.join(pipeline, 'analyse_0617')", "os.environ['ANALYSE_0617_RUNTIME_ROOT']")
        source = source.replace("os.path.join(pipeline_dir, 'analyse_0617')", "os.environ['ANALYSE_0617_RUNTIME_ROOT']")
    source = source.replace("subjects = ['test001', 'test002', 'test003']", "subjects = os.environ['ANALYSE_0617_RUNTIME_SUBJECTS'].split(',')")
    # Preserve the original calculations while respecting the workstation cap.
    source = source.replace("Parallel(n_jobs=-1)", "Parallel(n_jobs=4)")
    namespace = {"__name__": "__main__", "__file__": str(source_path), "__package__": None}
    exec(compile(source, str(source_path), "exec"), namespace, namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=SUBJECTS,
                        help="Only use this for missing subject-level stages; existing subjects are not rerun.")
    args = parser.parse_args()
    (RUN / "doc").mkdir(parents=True, exist_ok=True)
    (RUN / "result").mkdir(parents=True, exist_ok=True)
    prepare_feature_workspace(args.subjects)
    os.environ["ANALYSE_0617_RUNTIME_ROOT"] = str(RUN)
    os.environ["ANALYSE_0617_RUNTIME_SUBJECTS"] = ",".join(args.subjects)
    sys.path.insert(0, str(CODE))
    for stage in args.stages:
        print(f"\n{'=' * 22} {stage} {'=' * 22}")
        run_original(stage)


if __name__ == "__main__":
    main()
