"""Execution of multi-cohort configs -- pairwise + combined + comparisons (Step 15C).

A config with a top-level ``analysis_plan`` key is a multi-cohort config
(lib/cohort_analysis/analysis_plan.py detects this). ``run_cohort_analysis.py``
routes such a config here when it is run WITHOUT ``--dry-run-plan``.

Scope, one section added per step:

    * analysis_plan.pairwise      -> executed (Step 15A), one folder per cohort
    * analysis_plan.combined      -> executed (Step 15B), one folder per group
    * analysis_plan.comparisons   -> executed (Step 15C), one folder per directed pair
    * random percentile validation -> NOT run

Nothing about the single-analysis pipeline (run_cohort_analysis.run) changes.
Every section reuses the exact functions the single pipeline uses --
data_loading.load_cohorts / build_target_df / load_same_different_distributions,
pairwise.compute_distance_matrix / compute_rank_matrix,
cohort_comparison.build_group_image_ids / run_cross_cohort_comparison,
plotting.plot_pairwise_heatmap / plot_cohort_comparison_boxplot,
reports.write_table -- so an equivalent single-cohort config reproduces the
single pipeline's numbers.

For comparisons, each directed pair (base, comparison) from
analysis_plan.comparisons.directed_pairs is run by injecting a one-pair
``cross_cohort_comparison`` block into a copy of the config and calling
cohort_comparison.run_cross_cohort_comparison unchanged.

Known data caveat (not fixed here): the example config's LINS1 cohort points at
the PRMT7 embedding file. A combined group with both PRMT7 and LINS1 then
concatenates the same 8 images twice (duplicate image IDs, matrices written
as-is, heatmap skipped). A comparison whose base and comparison cohorts share an
embedding file compares a cohort against an effective copy of itself; it still
runs (the comparison code handles it) but its distances / PPV are not
biologically meaningful. Both cases are surfaced loudly in the console and in
analysis_metadata.json. Providing a real LINS1 embedding removes the issue.

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
        comparisons/
          <BASE>_vs_<COMPARISON>/
            cohort_comparison_summary.tsv
            cohort_comparison_distributions.tsv
            <BASE>_vs_<COMPARISON>_cohort_comparison_boxplot.{png,svg,jpeg}  (unless --skip-plots)
            analysis_metadata.json          (status: ok | skipped)
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from lib.cohort_analysis import analysis_plan
from lib.cohort_analysis import cohort_comparison
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
    plan, run_dir, pairwise_cohorts, combined_groups, comparison_results,
    skip_plots, final,
):
    """(Re)write analysis_plan_resolved.json.

    Called once early (final=False) for crash-resilience and once at the end
    (final=True). Records what pairwise / combined / comparisons actually
    executed. As of Step 15C every section is executed, so ``not_executed_yet``
    is empty.

    ``comparison_results`` is a list of dicts:
        {"base": ..., "comparison": ..., "label": ..., "status": "planned"|"ok"|"skipped",
         "reason": <only when skipped>}
    """
    n_ok = sum(1 for r in comparison_results if r.get("status") == "ok")
    n_skipped = sum(1 for r in comparison_results if r.get("status") == "skipped")

    payload = {
        "step": "15C - multi-cohort pairwise + combined + comparisons",
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
            "comparisons": {
                "mode": plan["comparisons"]["mode"],
                "n_planned": len(plan["comparisons"]["directed_pairs"]),
                "n_executed": n_ok,
                "n_skipped": n_skipped,
                "pairs": comparison_results,
                "skip_plots": bool(skip_plots),
            },
        },
        "not_executed_yet": {},
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


def _pair_shared_embedding(config, base, comp):
    """Return the shared embedding_file path if base and comp resolve to it, else None."""
    be = config.get("cohorts", {}).get(base, {}).get("embedding_file")
    ce = config.get("cohorts", {}).get(comp, {}).get("embedding_file")
    if be is not None and ce is not None and str(be) == str(ce):
        return str(be)
    return None


def _num(x):
    """Coerce a numpy/pandas scalar to a JSON-serialisable float, else pass through."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def _run_one_comparison(
    base, comp, config, entries, image_to_subject,
    same_diff_df, same_vals, diff_vals, comparisons_root, skip_plots,
):
    """Step 15C: one directed cohort-vs-cohort comparison (base vs comp).

    Reuses cohort_comparison.run_cross_cohort_comparison unchanged by injecting a
    one-pair ``cross_cohort_comparison`` block. Returns a result dict for
    analysis_plan_resolved.json: {base, comparison, label, status, reason?}.
    """
    label = "{} vs {}".format(base, comp)
    dir_name = "{}_vs_{}".format(base, comp)

    print("-" * 72)
    print("COMPARISON: {}".format(label))

    pair_dir = comparisons_root / dir_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Warnings: each cohort's own + shared-embedding warnings, plus a pair-level
    # note when base and comparison resolve to the same embedding file.
    warnings_out = []
    for role, name in (("base", base), ("comparison", comp)):
        for w in _warnings_for_cohort(entries, name):
            warnings_out.append("{} ({}): {}".format(name, role, w))

    shared_emb = _pair_shared_embedding(config, base, comp)
    if shared_emb:
        warnings_out.append(
            "base '{}' and comparison '{}' resolve to the SAME embedding file ({}); "
            "this cross-cohort comparison is effectively a cohort against a copy of "
            "itself and its distances / PPV are NOT biologically meaningful. Provide "
            "a correct LINS1 embedding to fix this.".format(base, comp, shared_emb)
        )
    for w in warnings_out:
        print("  ! {}".format(w))

    embedding_files = {
        name: {
            "embedding_file": config.get("cohorts", {}).get(name, {}).get("embedding_file"),
            "embedding_file_config": next(
                (e.get("embedding_file") for e in entries if e["name"] == name), None
            ),
        }
        for name in (base, comp)
    }

    # Per-pair config view: run_cross_cohort_comparison reads
    # config["cross_cohort_comparison"] (base_cohort / comparison_cohorts /
    # tunables) and config["id_column"] / config["random_seed"].
    cc_settings = dict(config.get("cross_cohort_comparison") or {})
    cc_settings["base_cohort"] = base
    cc_settings["comparison_cohorts"] = [comp]

    pair_cfg = dict(config)
    pair_cfg["active_cohorts"] = [base, comp]
    pair_cfg["id_column"] = config.get("id_column", "subject_id")
    pair_cfg["cross_cohort_comparison"] = cc_settings
    pair_cfg["target_name"] = label

    meta = {
        "analysis_type": "cohort_comparison",
        "base_cohort": base,
        "comparison_cohort": comp,
        "comparison_label": label,
        "embedding_files": embedding_files,
        "shares_embedding_file": shared_emb,
        "skip_plots": bool(skip_plots),
        "warnings": warnings_out,
    }

    try:
        cohort_reps = data_loading.load_cohorts(pair_cfg)
        target_df = data_loading.build_target_df(cohort_reps, image_to_subject)
        group_image_ids = cohort_comparison.build_group_image_ids(
            target_df, [base, comp]
        )
        summary_df, distributions_df = cohort_comparison.run_cross_cohort_comparison(
            cohort_reps, target_df, group_image_ids, same_vals, diff_vals, pair_cfg
        )
    except Exception as exc:  # record ANY failure, skip this pair, keep going
        reason = "{}: {}".format(type(exc).__name__, exc)
        print("  SKIPPED: {}".format(reason))
        meta.update({
            "status": "skipped",
            "reason": reason,
            "outputs": sorted(p.name for p in pair_dir.glob("*")),
        })
        _write_json(pair_dir / "analysis_metadata.json", meta)
        print("")
        return {
            "base": base, "comparison": comp, "label": label,
            "status": "skipped", "reason": reason,
        }

    summary_path = reports.write_table(
        summary_df, pair_dir, "cohort_comparison_summary.tsv"
    )
    dist_path = reports.write_table(
        distributions_df, pair_dir, "cohort_comparison_distributions.tsv"
    )
    print("  summary -> {}".format(summary_path))
    print("  dist    -> {}  ({} rows)".format(dist_path, len(distributions_df)))

    settings = cohort_comparison.resolve_comparison_settings(pair_cfg)
    row = summary_df.iloc[0].to_dict() if len(summary_df) else {}

    plot_path = None
    if not skip_plots:
        plot_path = plotting.plot_cohort_comparison_boxplot(
            same_diff_df=same_diff_df,
            comparison_distributions=distributions_df,
            summary_df=summary_df,
            output_dir=pair_dir,
            base_key=dir_name,
            threshold=settings["thr"],
            top_k=settings["top_k"],
        )

    meta.update({
        "status": "ok",
        "base_n_images": int(row.get("base_n_images", 0)),
        "base_n_patients": int(row.get("base_n_patients", 0)),
        "comparison_n_images": int(row.get("comparison_n_images", 0)),
        "comparison_n_patients": int(row.get("comparison_n_patients", 0)),
        "settings": {
            "threshold_c": settings["thr"],
            "pretest_probability": settings["p_pretest"],
            "n_samples": int(settings["n_samples"]),
            "min_base_images": int(settings["min_a"]),
            "min_comparison_images": int(settings["min_b"]),
            "top_k": int(settings["top_k"]),
            "random_seed": int(config.get("random_seed", 0)),
            "id_column": pair_cfg["id_column"],
        },
        "ppv_summary": {
            "mean_pw_distance_all_pairs": _num(row.get("mean_pw_distance_all_pairs")),
            "median_sampled_mean_pw_dist": _num(row.get("median_sampled_mean_pw_dist")),
            "prop_sampled_above_c": _num(row.get("prop_sampled_above_c")),
            "evidence_different": (
                bool(row["evidence_different"]) if "evidence_different" in row else None
            ),
            "PPV_min": _num(row.get("PPV_min")),
            "PPV_25th": _num(row.get("PPV_25th")),
            "PPV_median": _num(row.get("PPV_median")),
            "PPV_75th": _num(row.get("PPV_75th")),
            "PPV_max": _num(row.get("PPV_max")),
        },
        "n_samples": int(row.get("n_samples", settings["n_samples"])),
        "plot": Path(plot_path).name if plot_path else None,
        "outputs": sorted(p.name for p in pair_dir.glob("*")),
    })
    _write_json(pair_dir / "analysis_metadata.json", meta)
    print("  wrote:", pair_dir / "analysis_metadata.json")
    print("")
    return {"base": base, "comparison": comp, "label": label, "status": "ok"}


def _write_json(path, obj):
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def run_multi_plan(config, plan, skip_plots=False, config_path=None, repo_root=None):
    """Execute analysis_plan.pairwise + combined + comparisons for a multi-cohort config.

    ``config`` is the path-resolved config dict (build_config output). ``plan``
    is analysis_plan.build_plan(raw_config). Returns the top-level run directory.
    """
    if plan["format"] != analysis_plan.MULTI_COHORT:
        raise ValueError(
            "run_multi_plan expects a multi-cohort config (top-level 'analysis_plan')."
        )

    entries = plan["available_cohorts"]
    pairwise_cohorts = list(plan["pairwise"]["cohorts"])
    combined_groups = list(plan["combined"])
    directed_pairs = [tuple(p) for p in plan["comparisons"]["directed_pairs"]]
    n_directed = len(directed_pairs)

    planned_comparison_results = [
        {"base": b, "comparison": c, "label": "{} vs {}".format(b, c),
         "status": "planned"}
        for b, c in directed_pairs
    ]

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
        plan, run_dir, pairwise_cohorts, combined_groups,
        planned_comparison_results, skip_plots, final=False,
    )

    print("=" * 72)
    print("Multi-cohort config detected (top-level 'analysis_plan').")
    print("Step 15C scope: executing pairwise + combined + comparisons.")
    print("  pairwise analyses ...... EXECUTED")
    print("  combined analyses ...... EXECUTED")
    print("  cohort-vs-cohort ....... EXECUTED")
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
        "Comparisons plan (mode: {}) -> {} directed pair(s): {}".format(
            plan["comparisons"]["mode"],
            n_directed,
            ", ".join("{}_vs_{}".format(b, c) for b, c in directed_pairs) or "(none)",
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

    # ------------------------------------------------------------------
    # 3. analysis_plan.comparisons  (Step 15C)
    # ------------------------------------------------------------------
    comparisons_root = run_dir / "comparisons"
    comparison_results = []
    if directed_pairs:
        same_diff_df, same_vals, diff_vals = (
            data_loading.load_same_different_distributions(
                config["same_different_distribution_file"]
            )
        )
        for base, comp in directed_pairs:
            result = _run_one_comparison(
                base, comp, config, entries, image_to_subject,
                same_diff_df, same_vals, diff_vals, comparisons_root, skip_plots,
            )
            comparison_results.append(result)
    else:
        print("analysis_plan.comparisons produced no directed pairs.\n")

    resolved_plan_path = _write_resolved_plan(
        plan, run_dir, executed_pairwise, executed_combined,
        comparison_results, skip_plots, final=True,
    )

    n_cmp_ok = sum(1 for r in comparison_results if r["status"] == "ok")
    skipped = [r for r in comparison_results if r["status"] == "skipped"]

    print("=" * 72)
    print("Output tree ({}):".format(run_dir))
    _print_tree(run_dir)
    print("")
    print("Done -- Step 15C.")
    print("  pairwise analyses executed: {} ({})".format(
        len(executed_pairwise), ", ".join(executed_pairwise) or "none"))
    print("  combined analyses executed: {} ({})".format(
        len(executed_combined),
        ", ".join(g["name"] for g in executed_combined) or "none"))
    print("  cohort-vs-cohort comparisons executed: {} of {} planned pair(s)".format(
        n_cmp_ok, n_directed))
    if skipped:
        print("  comparisons SKIPPED: {}".format(len(skipped)))
        for r in skipped:
            print("    - {}: {}".format(r["label"], r["reason"]))
    else:
        print("  comparisons skipped: 0")
    if skip_plots:
        print("(--skip-plots was set: no heatmaps or comparison plots generated.)")
    return run_dir
