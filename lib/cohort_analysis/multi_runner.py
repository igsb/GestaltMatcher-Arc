"""Execution of multi-cohort configs -- pairwise + combined (Step 15B).

A config with a top-level ``analysis_plan`` key is a multi-cohort config
(lib/cohort_analysis/analysis_plan.py detects this). ``run_cohort_analysis.py``
routes such a config here when it is run WITHOUT ``--dry-run-plan``.

Scope, growing one section per step:

    * analysis_plan.pairwise      -> executed (Step 15A), one folder per cohort
    * analysis_plan.combined      -> executed (Step 15B), one folder per group
    * analysis_plan.comparisons   -> parsed and recorded only, NOT executed yet
    * random percentile validation -> NOT run

Nothing about the single-analysis pipeline (run_cohort_analysis.run) changes.
Both the per-cohort and the combined pairwise computation reuse the exact same
functions the single pipeline uses -- data_loading.load_cohorts,
data_loading.build_target_df, pairwise.compute_distance_matrix,
pairwise.compute_rank_matrix, plotting.plot_pairwise_heatmap, reports.write_table
-- so an equivalent single-cohort/single-group config reproduces the single
pipeline's numbers.

Known data caveat (not fixed here): the example config's LINS1 cohort points at
the PRMT7 embedding file. A combined group containing both PRMT7 and LINS1 then
concatenates the same 8 images twice, so its target metadata and its distance /
rank matrices contain duplicate image IDs. That is surfaced as a loud warning in
the console and in analysis_metadata.json, the matrices are written as-is (not
de-duplicated), and the heatmap for that group is skipped (``df.loc`` with
duplicate labels would cartesian-explode the figure). Providing a real LINS1
embedding removes the duplication.

Output layout::

    <output_root>/
      run_<timestamp>_<short_config_hash>/
        run_config.json              (resolved config, via config.dump_run_config)
        run_metadata.json            (provenance, via run_metadata.write_run_metadata)
        analysis_plan_resolved.json  (expanded plan + what executed + warnings)
        pairwise/
          <COHORT>/
            target_cohort_metadata.tsv
            pairwise_distance_matrix.tsv
            pairwise_rank_matrix.tsv
            <COHORT>_validation_pairwise_rank_single.svg   (unless --skip-plots)
            analysis_metadata.json
        combined/
          <GROUP>/
            target_cohort_metadata.tsv
            pairwise_distance_matrix.tsv
            pairwise_rank_matrix.tsv
            <GROUP>_validation_pairwise_rank_single.svg    (unless --skip-plots
                                                            or duplicate image IDs)
            analysis_metadata.json
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from lib.cohort_analysis import analysis_plan
from lib.cohort_analysis import config as config_module
from lib.cohort_analysis import data_loading
from lib.cohort_analysis import pairwise
from lib.cohort_analysis import plotting
from lib.cohort_analysis import reports
from lib.cohort_analysis import run_metadata as run_metadata_module


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


def _write_resolved_plan(
    plan, run_dir, pairwise_cohorts, combined_groups, skip_plots, final
):
    """(Re)write analysis_plan_resolved.json.

    Called once early (final=False) for crash-resilience and once at the end
    (final=True). Records what pairwise/combined actually executed and keeps
    analysis_plan.comparisons under ``not_executed_yet``.
    """
    payload = {
        "step": "15B - multi-cohort pairwise + combined",
        "status": "complete" if final else "in-progress",
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
            },
            "combined": {
                "groups": [
                    {"name": g["name"], "cohorts": list(g["cohorts"])}
                    for g in combined_groups
                ],
                "skip_plots": bool(skip_plots),
            },
        },
        "not_executed_yet": {
            "comparisons": {
                "mode": plan["comparisons"]["mode"],
                "directed_pairs": [
                    list(p) for p in plan["comparisons"]["directed_pairs"]
                ],
                "n_directed": len(plan["comparisons"]["directed_pairs"]),
            }
        },
        "warnings": list(plan.get("warnings", [])),
    }
    out_path = Path(run_dir) / "analysis_plan_resolved.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out_path


def _run_one_pairwise(
    cohort_name, config, entries, image_to_subject,
    gallery_representation_df, pairwise_root, skip_plots,
):
    """Step 15A per-cohort pairwise analysis. Behaviour unchanged."""
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
    return analysis_meta


def _run_one_combined(
    group, config, entries, image_to_subject,
    gallery_representation_df, combined_root, skip_plots,
):
    """Step 15B: pooled pairwise analysis over the member cohorts of one group."""
    name = group["name"]
    members = list(group["cohorts"])

    print("-" * 72)
    print("COMBINED: {}  <- [{}]".format(name, ", ".join(members)))

    group_dir = combined_root / name
    group_dir.mkdir(parents=True, exist_ok=True)

    # Member-level warnings (own + shared-embedding), prefixed with the member.
    group_warnings = []
    for m in members:
        for w in _warnings_for_cohort(entries, m):
            group_warnings.append("{}: {}".format(m, w))
    for w in group_warnings:
        print("  ! {}".format(w))

    # Reuse load_cohorts by pointing active_cohorts at the member list, in order.
    combined_cfg = dict(config)
    combined_cfg["active_cohorts"] = members
    combined_cfg["target_name"] = name

    cohort_reps = data_loading.load_cohorts(combined_cfg)

    target_representation_df = pd.concat(cohort_reps.values(), ignore_index=True)
    target_image_ids = target_representation_df["img_name"].astype(str).values

    target_df = data_loading.build_target_df(cohort_reps, image_to_subject)
    target_meta_path = reports.write_table(
        target_df, group_dir, "target_cohort_metadata.tsv"
    )

    n_rows = int(len(target_image_ids))
    n_unique_ids = len(set(map(str, target_image_ids)))
    n_images = int(target_df["image_id"].nunique())
    n_patients = int(target_df["subject_id"].nunique())
    has_duplicate_ids = n_unique_ids != n_rows

    per_member_counts = {
        m: {
            "n_images": int(cohort_reps[m]["img_name"].astype(str).nunique()),
            "n_rows": int(len(cohort_reps[m])),
        }
        for m in members
    }

    print(
        "  member rows: {}   unique image IDs: {}   patients: {}".format(
            n_rows, n_unique_ids, n_patients
        )
    )
    for m in members:
        print(
            "    {}: {} image(s)".format(m, per_member_counts[m]["n_images"])
        )
    print("  wrote:", target_meta_path)

    if has_duplicate_ids:
        dup_msg = (
            "DUPLICATE IMAGE IDS: combined group '{grp}' concatenates {rows} "
            "member rows but only {uniq} unique image IDs ({dup} duplicate(s)). "
            "Member cohorts share embedding files (see the LINS1 / shared-embedding "
            "warnings above). The distance and rank matrices are written {rows}x{rows} "
            "with repeated index labels and are NOT de-duplicated; the heatmap is "
            "skipped for this group. Provide a correct LINS1 embedding to resolve this."
        ).format(grp=name, rows=n_rows, uniq=n_unique_ids, dup=n_rows - n_unique_ids)
        group_warnings.append(dup_msg)
        print("  ! {}".format(dup_msg))

    distance_df, target_tta = pairwise.compute_distance_matrix(
        target_representation_df, target_image_ids, combined_cfg
    )
    distance_path = reports.write_table(
        distance_df, group_dir, "pairwise_distance_matrix.tsv", index=True
    )
    print("  distance matrix {} -> {}".format(distance_df.shape, distance_path))

    rank_df = pairwise.compute_rank_matrix(
        target_tta, gallery_representation_df, target_image_ids, combined_cfg
    )
    rank_path = reports.write_table(
        rank_df, group_dir, "pairwise_rank_matrix.tsv", index=True
    )
    print("  rank matrix     {} -> {}".format(rank_df.shape, rank_path))

    # The heatmap is skipped when --skip-plots is set OR when the group has
    # duplicate image IDs (df.loc with repeated labels would cartesian-explode
    # the figure). Both reasons are recorded when both apply.
    skip_reasons = []
    if skip_plots:
        skip_reasons.append("--skip-plots")
    if has_duplicate_ids:
        skip_reasons.append(
            "duplicate image IDs in combined group would cartesian-explode the figure"
        )
    heatmap_skipped_reason = "; ".join(skip_reasons) or None

    heatmap_path = None
    if heatmap_skipped_reason is None:
        heatmap_path = plotting.plot_pairwise_heatmap(
            distance_df, rank_df, group_dir, combined_cfg
        )
    elif not skip_plots:
        print("  heatmap skipped: {}".format(heatmap_skipped_reason))

    member_embeddings = {
        m: {
            "embedding_file": config.get("cohorts", {}).get(m, {}).get("embedding_file"),
            "embedding_file_config": next(
                (e.get("embedding_file") for e in entries if e["name"] == m), None
            ),
        }
        for m in members
    }

    analysis_meta = {
        "analysis_type": "combined_pairwise",
        "group": name,
        "member_cohorts": members,
        "embedding_files": member_embeddings,
        "n_images": n_images,
        "n_images_in_matrix": n_rows,
        "n_unique_image_ids": n_unique_ids,
        "n_duplicate_image_ids": n_rows - n_unique_ids,
        "has_duplicate_image_ids": has_duplicate_ids,
        "per_member_counts": per_member_counts,
        "n_patients": n_patients,
        "distance_metric": config.get("distance_metric"),
        "n_tta": config.get("n_tta"),
        "skip_plots": bool(skip_plots),
        "heatmap": Path(heatmap_path).name if heatmap_path else None,
        "heatmap_skipped_reason": heatmap_skipped_reason,
        "matrix_shape": [int(distance_df.shape[0]), int(distance_df.shape[1])],
        "warnings": group_warnings,
        "outputs": sorted(p.name for p in group_dir.glob("*")),
    }
    meta_path = group_dir / "analysis_metadata.json"
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(analysis_meta, f, indent=2)
        f.write("\n")
    print("  wrote:", meta_path)
    print("")
    return analysis_meta


def run_multi_plan(config, plan, skip_plots=False, config_path=None, repo_root=None):
    """Execute analysis_plan.pairwise and analysis_plan.combined for a multi-cohort config.

    ``config`` is the path-resolved config dict (build_config output). ``plan``
    is analysis_plan.build_plan(raw_config). Returns the top-level run directory.
    analysis_plan.comparisons is recorded but not executed.
    """
    if plan["format"] != analysis_plan.MULTI_COHORT:
        raise ValueError(
            "run_multi_plan expects a multi-cohort config (top-level 'analysis_plan')."
        )

    entries = plan["available_cohorts"]
    pairwise_cohorts = list(plan["pairwise"]["cohorts"])
    combined_groups = list(plan["combined"])
    n_directed = len(plan["comparisons"]["directed_pairs"])

    run_dir = make_run_dir(config, config_path=config_path)
    run_config_path = config_module.dump_run_config(config, run_dir)
    run_metadata_path = run_metadata_module.write_run_metadata(
        config,
        run_dir,
        run_config_path=run_config_path,
        config_path=config_path,
        repo_root=repo_root,
    )
    resolved_plan_path = _write_resolved_plan(
        plan, run_dir, pairwise_cohorts, combined_groups, skip_plots, final=False
    )

    print("=" * 72)
    print("Multi-cohort config detected (top-level 'analysis_plan').")
    print("Step 15B scope: executing analysis_plan.pairwise + analysis_plan.combined.")
    print("  pairwise analyses ...... EXECUTED")
    print("  combined analyses ...... EXECUTED")
    print("  cohort-vs-cohort ....... parsed + recorded, NOT run yet")
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
        "Combined plan -> {} group(s): {}".format(
            len(combined_groups),
            ", ".join(g["name"] for g in combined_groups) or "(none)",
        )
    )
    print(
        "Deferred (not run this step): {} directed cohort-vs-cohort comparison(s).".format(
            n_directed
        )
    )
    if plan.get("warnings"):
        print("")
        print("Plan warnings:")
        for w in plan["warnings"]:
            print("  ! {}".format(w))
    print("")

    # ------------------------------------------------------------------
    # Shared inputs -- loaded once, reused for every cohort and group.
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

    # ------------------------------------------------------------------
    # 1. analysis_plan.pairwise  (Step 15A, unchanged)
    # ------------------------------------------------------------------
    pairwise_root = run_dir / "pairwise"
    executed_pairwise = []
    for cohort_name in pairwise_cohorts:
        _run_one_pairwise(
            cohort_name, config, entries, image_to_subject,
            gallery_representation_df, pairwise_root, skip_plots,
        )
        executed_pairwise.append(cohort_name)
    if not pairwise_cohorts:
        print("analysis_plan.pairwise selected no cohorts.\n")

    # ------------------------------------------------------------------
    # 2. analysis_plan.combined  (Step 15B)
    # ------------------------------------------------------------------
    combined_root = run_dir / "combined"
    executed_combined = []
    for group in combined_groups:
        _run_one_combined(
            group, config, entries, image_to_subject,
            gallery_representation_df, combined_root, skip_plots,
        )
        executed_combined.append(group)
    if not combined_groups:
        print("analysis_plan.combined defined no groups.\n")

    resolved_plan_path = _write_resolved_plan(
        plan, run_dir, executed_pairwise, executed_combined, skip_plots, final=True
    )

    print("=" * 72)
    print("Output tree ({}):".format(run_dir))
    _print_tree(run_dir)
    print("")
    print("Done -- Step 15B.")
    print("  pairwise analyses executed: {} ({})".format(
        len(executed_pairwise), ", ".join(executed_pairwise) or "none"))
    print("  combined analyses executed: {} ({})".format(
        len(executed_combined),
        ", ".join(g["name"] for g in executed_combined) or "none"))
    print("  cohort-vs-cohort comparisons: deferred, NOT run yet "
          "({} directed pair(s) recorded in analysis_plan_resolved.json)".format(n_directed))
    if skip_plots:
        print("(--skip-plots was set: no heatmaps generated.)")
    return run_dir
