"""Execution of multi-cohort configs -- analysis_plan.pairwise ONLY (Step 15).

A config with a top-level ``analysis_plan`` key is a multi-cohort config
(lib/cohort_analysis/analysis_plan.py detects this). ``run_cohort_analysis.py``
routes such a config here when it is run WITHOUT ``--dry-run-plan``.

Scope of this step, deliberately narrow:

    * analysis_plan.pairwise      -> executed, one folder per selected cohort
    * analysis_plan.combined      -> parsed and echoed, NOT executed
    * analysis_plan.comparisons   -> parsed and echoed, NOT executed
    * random percentile validation -> NOT run

Nothing about the single-analysis pipeline (run_cohort_analysis.run) changes.
The per-cohort pairwise computation reuses the exact same functions the single
pipeline uses -- data_loading.load_cohorts, pairwise.compute_distance_matrix,
pairwise.compute_rank_matrix, plotting.plot_pairwise_heatmap -- so the numbers
match the single pipeline for an equivalent single-cohort config.

Output layout::

    <output_root>/
      run_<timestamp>_<short_config_hash>/
        run_config.json              (resolved config, via config.dump_run_config)
        run_metadata.json            (provenance, via run_metadata.write_run_metadata)
        analysis_plan_resolved.json  (expanded plan + warnings + what ran)
        pairwise/
          <COHORT>/
            target_cohort_metadata.tsv
            pairwise_distance_matrix.tsv
            pairwise_rank_matrix.tsv
            <COHORT>_validation_pairwise_rank_single.svg   (unless --skip-plots)
            analysis_metadata.json
"""

import json
from datetime import datetime
from pathlib import Path

from lib.cohort_analysis import analysis_plan
from lib.cohort_analysis import config as config_module
from lib.cohort_analysis import data_loading
from lib.cohort_analysis import pairwise
from lib.cohort_analysis import plotting
from lib.cohort_analysis import reports
from lib.cohort_analysis import run_metadata as run_metadata_module

# Files a pairwise cohort folder is expected to contain (heatmap excluded: it is
# conditional on --skip-plots; analysis_metadata.json is written last).
PAIRWISE_TABLE_OUTPUTS = (
    "target_cohort_metadata.tsv",
    "pairwise_distance_matrix.tsv",
    "pairwise_rank_matrix.tsv",
)


def make_run_dir(config, config_path=None):
    """Create and return <output_root>/run_<timestamp>_<short_config_hash>/."""
    root = Path(config["output_root"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    digest = run_metadata_module.sha256_file(config_path) if config_path else None
    short_hash = digest[:12] if digest else "nohash"

    run_dir = root / "run_{}_{}".format(stamp, short_hash)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _warnings_for_cohort(entries, name):
    """Console/metadata warnings relevant to one cohort (own + shared embedding)."""
    by_name = {e["name"]: e for e in entries}
    entry = by_name.get(name, {})
    out = []

    if entry.get("warning"):
        out.append(str(entry["warning"]))

    emb = entry.get("embedding_file")
    if emb:
        shared = [
            other["name"]
            for other in entries
            if other["name"] != name and str(other.get("embedding_file")) == str(emb)
        ]
        if shared:
            out.append(
                "shares embedding_file with {} ({})".format(shared, emb)
            )
    return out


def _print_tree(root):
    """Print a simple sorted file tree rooted at ``root``."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts) - 1
        marker = "/" if path.is_dir() else ""
        print("  " + "  " * depth + rel.parts[-1] + marker)


def _dump_resolved_plan(plan, run_dir, pairwise_cohorts, skip_plots):
    """Write the expanded plan (with what executed) to analysis_plan_resolved.json."""
    payload = {
        "step": "15 - multi-cohort pairwise only",
        "format": plan["format"],
        "config_path": plan.get("config_path"),
        "available_cohorts": [
            {
                "name": e["name"],
                "label": e.get("label"),
                "embedding_file": e.get("embedding_file"),
                "img_name_parser": e.get("img_name_parser"),
                "warning": e.get("warning"),
            }
            for e in plan["available_cohorts"]
        ],
        "executed": {
            "pairwise": {
                "mode": plan["pairwise"]["mode"],
                "cohorts": list(pairwise_cohorts),
                "skip_plots": bool(skip_plots),
            }
        },
        "not_executed_yet": {
            "combined": plan["combined"],
            "comparisons": {
                "mode": plan["comparisons"]["mode"],
                "directed_pairs": [list(p) for p in plan["comparisons"]["directed_pairs"]],
                "n_directed": len(plan["comparisons"]["directed_pairs"]),
            },
        },
        "warnings": list(plan.get("warnings", [])),
    }
    out_path = Path(run_dir) / "analysis_plan_resolved.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out_path


def run_pairwise_plan(config, plan, skip_plots=False, config_path=None, repo_root=None):
    """Execute analysis_plan.pairwise for a multi-cohort config.

    ``config`` is the path-resolved config dict (build_config output). ``plan``
    is analysis_plan.build_plan(raw_config). Returns the top-level run directory.
    """
    if plan["format"] != analysis_plan.MULTI_COHORT:
        raise ValueError(
            "run_pairwise_plan expects a multi-cohort config (top-level 'analysis_plan')."
        )

    entries = plan["available_cohorts"]
    pairwise_cohorts = list(plan["pairwise"]["cohorts"])

    run_dir = make_run_dir(config, config_path=config_path)
    run_config_path = config_module.dump_run_config(config, run_dir)
    run_metadata_path = run_metadata_module.write_run_metadata(
        config,
        run_dir,
        run_config_path=run_config_path,
        config_path=config_path,
        repo_root=repo_root,
    )
    resolved_plan_path = _dump_resolved_plan(
        plan, run_dir, pairwise_cohorts, skip_plots
    )

    print("=" * 72)
    print("Multi-cohort config detected (top-level 'analysis_plan').")
    print("Step 15 scope: executing analysis_plan.pairwise ONLY.")
    print("  combined analyses ....... parsed, NOT run in this step")
    print("  cohort-vs-cohort ....... parsed, NOT run in this step")
    print("  random percentile val .. NOT run in this step")
    print("=" * 72)
    print("run dir: {}".format(run_dir))
    print("wrote:   {}".format(run_config_path))
    print("wrote:   {}".format(run_metadata_path))
    print("wrote:   {}".format(resolved_plan_path))
    print("")
    print(
        "Pairwise plan (mode: {}) -> {} cohort(s): {}".format(
            plan["pairwise"]["mode"],
            len(pairwise_cohorts),
            ", ".join(pairwise_cohorts) or "(none)",
        )
    )
    print(
        "Deferred: {} combined group(s), {} directed comparison(s).".format(
            len(plan["combined"]), len(plan["comparisons"]["directed_pairs"])
        )
    )
    if plan.get("warnings"):
        print("")
        print("Plan warnings:")
        for w in plan["warnings"]:
            print("  ! {}".format(w))
    print("")

    if not pairwise_cohorts:
        print("analysis_plan.pairwise selected no cohorts; nothing to execute.")
        return run_dir

    # ------------------------------------------------------------------
    # Shared inputs -- loaded once, reused for every cohort.
    # ------------------------------------------------------------------
    photo_df, image_to_subject, _, _, _ = data_loading.load_photo_metadata(
        config["photo_metadata_file"],
        sep=config["photo_metadata_sep"],
        frontal_face_only=config["frontal_face_only"],
    )
    gallery_image_ids, _ = data_loading.load_gallery_image_ids(
        config["gallery_metadata_files"]
    )
    print("Photo metadata rows:", len(photo_df))
    print("Gallery image IDs:", len(gallery_image_ids))
    print("")
    print("Loading gallery embeddings (large pickle, this takes a while) ...")
    gallery_representation_df = data_loading.load_embedding_df(
        config["gallery_embedding_file"],
        img_name_parser="first_token",
    )
    gallery_representation_df = gallery_representation_df[
        gallery_representation_df["img_name"].astype(str).isin(set(gallery_image_ids))
    ].copy()
    print("Gallery images:", len(gallery_representation_df))
    print("")

    pairwise_root = run_dir / "pairwise"

    for cohort_name in pairwise_cohorts:
        print("-" * 72)
        print("PAIRWISE: {}".format(cohort_name))

        cohort_dir = pairwise_root / cohort_name
        cohort_dir.mkdir(parents=True, exist_ok=True)

        cohort_warnings = _warnings_for_cohort(entries, cohort_name)
        for w in cohort_warnings:
            print("  ! {}".format(w))

        # Per-cohort view of the config: reuse load_cohorts / pairwise / heatmap
        # unchanged by pointing active_cohorts + target_name at this one cohort.
        cohort_cfg = dict(config)
        cohort_cfg["active_cohorts"] = [cohort_name]
        cohort_cfg["target_name"] = cohort_name

        cohort_reps = data_loading.load_cohorts(cohort_cfg)
        target_representation_df = cohort_reps[cohort_name]
        target_image_ids = target_representation_df["img_name"].astype(str).values

        target_df = data_loading.build_target_df(cohort_reps, image_to_subject)
        target_meta_path = reports.write_table(
            target_df, cohort_dir, "target_cohort_metadata.tsv"
        )

        n_images = int(target_df["image_id"].nunique())
        n_patients = int(target_df["subject_id"].nunique())
        print("  images: {}   patients: {}".format(n_images, n_patients))
        print("  wrote:", target_meta_path)

        distance_df, target_tta = pairwise.compute_distance_matrix(
            target_representation_df, target_image_ids, cohort_cfg
        )
        distance_path = reports.write_table(
            distance_df, cohort_dir, "pairwise_distance_matrix.tsv", index=True
        )
        print("  distance matrix {} -> {}".format(distance_df.shape, distance_path))

        rank_df = pairwise.compute_rank_matrix(
            target_tta, gallery_representation_df, target_image_ids, cohort_cfg
        )
        rank_path = reports.write_table(
            rank_df, cohort_dir, "pairwise_rank_matrix.tsv", index=True
        )
        print("  rank matrix     {} -> {}".format(rank_df.shape, rank_path))

        heatmap_path = None
        if not skip_plots:
            heatmap_path = plotting.plot_pairwise_heatmap(
                distance_df, rank_df, cohort_dir, cohort_cfg
            )

        cohort_entry = next((e for e in entries if e["name"] == cohort_name), {})
        resolved_embedding = (
            config.get("cohorts", {}).get(cohort_name, {}).get("embedding_file")
            or cohort_entry.get("embedding_file")
        )
        analysis_meta = {
            "analysis_type": "pairwise",
            "cohort": cohort_name,
            "label": cohort_entry.get("label", cohort_name),
            "embedding_file": resolved_embedding,
            "embedding_file_config": cohort_entry.get("embedding_file"),
            "img_name_parser": cohort_entry.get("img_name_parser"),
            "n_images": n_images,
            "n_patients": n_patients,
            "distance_metric": config.get("distance_metric"),
            "n_tta": config.get("n_tta"),
            "skip_plots": bool(skip_plots),
            "heatmap": Path(heatmap_path).name if heatmap_path else None,
            "warnings": cohort_warnings,
            "outputs": sorted(p.name for p in cohort_dir.glob("*")),
        }
        meta_path = cohort_dir / "analysis_metadata.json"
        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(analysis_meta, f, indent=2)
            f.write("\n")
        print("  wrote:", meta_path)
        print("")

    print("=" * 72)
    print("Output tree ({}):".format(run_dir))
    _print_tree(run_dir)
    print("")
    print(
        "Done -- Step 15 pairwise only. {} cohort folder(s) under {}".format(
            len(pairwise_cohorts), pairwise_root
        )
    )
    if skip_plots:
        print("(--skip-plots was set: no heatmaps generated.)")
    print(
        "Not generated (by design this step): combined analyses, "
        "cohort-vs-cohort comparisons, random percentile validation."
    )
    return run_dir
