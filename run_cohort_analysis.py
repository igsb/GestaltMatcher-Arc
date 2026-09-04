"""CLI entrypoint for the GestaltMatcher cohort analysis pipeline.

Replaces notebooks/GestaltMatcher_cohort_analysis_template.ipynb. The notebook
stays as the exploratory front-end; this script runs the same pipeline
end-to-end from a YAML config.

Usage:
    python run_cohort_analysis.py --config configs/cohort_analysis/prmt7_ctnnd2.yaml

The pipeline is complete: it runs every step of the notebook end to end, from
the config through the tables and figures to the final output listing. Pass
--skip-plots to write the tables only.

Environment: this must be run under the conda "gestalt" environment
(Python 3.8.19). Other interpreters may not have seaborn, and a different
numpy/BLAS produces tiny floating-point differences in the distance matrices.

Cost: the rank matrix needs the ~1.9 GB GMDB gallery pickle, which is read in
full before being filtered. Expect roughly 5 minutes and several GB of RAM.

Behaviour is ported from the notebook without changes: a run of
configs/cohort_analysis/lins1_ctnnd2.yaml reproduces every table in
analysis_output/LINS1_2026_07_01 byte for byte.

Multi-cohort configs
--------------------
A config with a top-level ``analysis_plan`` key is a multi-cohort config. It is
auto-detected and routed as follows, with no extra CLI flag:

    --dry-run-plan   -> print the expanded plan and exit (Step 14)
    (otherwise)      -> execute analysis_plan.pairwise + combined + comparisons,
                        writing
                        <output_root>/run_<timestamp>_<short_config_hash>/
                          pairwise/<COHORT>/
                          combined/<GROUP>/
                          comparisons/<BASE>_vs_<COMPARISON>/

Random percentile validation is not run for multi-cohort configs. Single-analysis
configs are unaffected.
"""

import argparse
from pathlib import Path

import pandas as pd

from lib.cohort_analysis import analysis_plan
from lib.cohort_analysis import cohort_comparison
from lib.cohort_analysis import config as config_module
from lib.cohort_analysis import data_loading
from lib.cohort_analysis import multi_runner
from lib.cohort_analysis import pairwise
from lib.cohort_analysis import plotting
from lib.cohort_analysis import random_validation
from lib.cohort_analysis import reports
from lib.cohort_analysis import run_metadata as run_metadata_module

REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a GestaltMatcher cohort analysis from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the cohort analysis YAML config.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Override output_root from the config. Relative paths are taken "
            "relative to the current working directory."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Compute and write tables only; skip all figures.",
    )
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
        help=(
            "Load the config, detect single-analysis vs multi-cohort format, "
            "print the expanded analysis plan, and exit. No paths are resolved, "
            "no embeddings are loaded, and no outputs are written."
        ),
    )
    return parser.parse_args(argv)


def build_config(config_path, output_root=None):
    """Load the YAML config, resolve its paths, and apply the output override.

    Returns the effective config dict. Nothing is written to disk here.
    """
    config = config_module.load_config(config_path)
    config = config_module.resolve_paths(config, REPO_ROOT)

    if output_root is not None:
        config["output_root"] = str(Path(output_root).expanduser().resolve())

    return config


def run(config, skip_plots=False, config_path=None):
    """Run the full pipeline for one config.

    config_path is the config location as passed on the command line. It is
    used only for run_metadata.json provenance (path + sha256 of the original
    file); the effective config still comes from the already-resolved `config`
    dict, so passing it does not change any result.

    Mirrors the notebook top to bottom:

        1. prepare output dir, dump run_config.json      (cell 5)
           + run_metadata.json (provenance; not in the notebook)
        2. load photo metadata, gallery IDs, cohorts     (cells 9, 10)
        3. pairwise distance matrix + rank matrix        (cells 12, 13)
        4. clustering heatmap                            (cell 16)
        5. random percentile analysis + box/KDE plots    (cells 19, 21)
        6. cross-cohort comparison + PPV + box plot      (cells 24, 27, 28)
        7. list outputs                                  (cell 32)

    Steps 4, 5's plots and 6's plot are skipped when skip_plots is set.
    """
    output_dir = config_module.prepare_output_dir(config)
    run_config_path = config_module.dump_run_config(config, output_dir)
    run_metadata_path = run_metadata_module.write_run_metadata(
        config,
        output_dir,
        run_config_path=run_config_path,
        config_path=config_path,
        repo_root=REPO_ROOT,
    )

    print("target_name: {}".format(config["target_name"]))
    print("output_dir:  {}".format(output_dir))
    print("wrote:       {}".format(run_config_path))
    print("wrote:       {}".format(run_metadata_path))
    print("")

    # ------------------------------------------------------------------
    # Data loading (notebook cells 9 and 10)
    # ------------------------------------------------------------------
    photo_df, image_to_subject, image_to_gene, image_to_patient_name, subject_to_image = (
        data_loading.load_photo_metadata(
            config["photo_metadata_file"],
            sep=config["photo_metadata_sep"],
            frontal_face_only=config["frontal_face_only"],
        )
    )

    gallery_image_ids, gallery_image_to_label = data_loading.load_gallery_image_ids(
        config["gallery_metadata_files"]
    )

    cohort_reps = data_loading.load_cohorts(config)

    target_representation_df = pd.concat(cohort_reps.values(), ignore_index=True)
    target_image_ids = target_representation_df["img_name"].astype(str).values

    target_df = data_loading.build_target_df(cohort_reps, image_to_subject)
    target_metadata_path = reports.write_table(
        target_df, output_dir, "target_cohort_metadata.tsv"
    )

    print("Photo metadata rows:", len(photo_df))
    print("Gallery image IDs:", len(gallery_image_ids))
    print("Loaded target images:", len(target_image_ids))
    print("Loaded target patients:", target_df["subject_id"].nunique())
    print("")
    print(
        target_df.groupby("cohort").agg(
            n_images=("image_id", "nunique"),
            n_patients=("subject_id", "nunique"),
        )
    )
    print("")
    print("wrote:", target_metadata_path)
    print("")

    # ------------------------------------------------------------------
    # Gallery embeddings (notebook cell 10)
    #
    # This is the ~1.9 GB GMDB pickle. It is read in full and then filtered down
    # to the gallery image IDs, exactly as the notebook does.
    # ------------------------------------------------------------------
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
    # Pairwise analysis (notebook cells 12 and 13)
    # ------------------------------------------------------------------
    distance_df, target_tta = pairwise.compute_distance_matrix(
        target_representation_df, target_image_ids, config
    )

    # index=True: these matrices carry their image IDs in the index.
    distance_path = reports.write_table(
        distance_df, output_dir, "pairwise_distance_matrix.tsv", index=True
    )

    print("distance matrix:", distance_df.shape)
    print("wrote:", distance_path)
    print("")

    rank_df = pairwise.compute_rank_matrix(
        target_tta, gallery_representation_df, target_image_ids, config
    )

    rank_path = reports.write_table(
        rank_df, output_dir, "pairwise_rank_matrix.tsv", index=True
    )

    print("rank matrix:", rank_df.shape)
    print("wrote:", rank_path)
    print("")

    # ------------------------------------------------------------------
    # Clustering heatmap (notebook cell 16)
    # ------------------------------------------------------------------
    if not skip_plots:
        plotting.plot_pairwise_heatmap(distance_df, rank_df, output_dir, config)
        print("")

    # ------------------------------------------------------------------
    # Random percentile validation (notebook cell 19)
    # ------------------------------------------------------------------
    random_summary_df, combined_random_distribution = (
        random_validation.run_random_validation(target_df, distance_df, config)
    )

    random_summary_path = reports.write_table(
        random_summary_df, output_dir, "target_random_percentile_summary.tsv"
    )
    random_dist_path = reports.write_table(
        combined_random_distribution, output_dir,
        "target_random_percentile_distributions.tsv",
    )

    print("")
    print(random_summary_df.to_string(index=False))
    print("")
    print("wrote:", random_summary_path)
    print("wrote:", random_dist_path)
    print("")

    # ------------------------------------------------------------------
    # Random vs target plots (notebook cell 21)
    # ------------------------------------------------------------------
    if not skip_plots:
        cohort_name = config.get("target_name", "cohort")

        plotting.plot_random_vs_target_boxplot(
            distribution_df=combined_random_distribution,
            output_dir=output_dir,
            cohort_name=cohort_name,
            value_col="mean_pw_dist",
            random_label="random",
        )

        plotting.plot_random_vs_target_kde(
            distribution_df=combined_random_distribution,
            output_dir=output_dir,
            cohort_name=cohort_name,
            value_col="mean_pw_dist",
            random_label="random",
            xlim=(0.4, 1.3),
        )
        print("")

    # ------------------------------------------------------------------
    # Cross-cohort comparison and PPV report (notebook cells 24, 25, 27)
    # ------------------------------------------------------------------
    # same_diff_df is not used yet; the cohort comparison box plot (cell 28,
    # step 8) plots it as the Same/Different syndrome control distributions.
    same_diff_df, same_vals, diff_vals = data_loading.load_same_different_distributions(
        config["same_different_distribution_file"]
    )

    group_image_ids = cohort_comparison.build_group_image_ids(
        target_df, config["active_cohorts"]
    )

    print("Prepared cohorts:")
    for key, ids in group_image_ids.items():
        print("  {}: {} images".format(key, len(ids)))
    print("")

    comparison_summary_df, comparison_distributions = (
        cohort_comparison.run_cross_cohort_comparison(
            cohort_reps, target_df, group_image_ids, same_vals, diff_vals, config
        )
    )

    base_key = config["cross_cohort_comparison"]["base_cohort"]

    comparison_summary_path = reports.write_table(
        comparison_summary_df, output_dir,
        "{}_cohort_comparison_summary.tsv".format(base_key),
    )
    comparison_dist_path = reports.write_table(
        comparison_distributions, output_dir,
        "{}_cohort_comparison_distributions.tsv".format(base_key),
    )

    print(comparison_summary_df[[
        "comparison", "mean_pw_distance_all_pairs", "prop_sampled_above_c",
        "evidence_different", "PPV_median",
    ]].to_string(index=False))
    print("")
    print("wrote:", comparison_summary_path)
    print("wrote:", comparison_dist_path)
    print("")

    # ------------------------------------------------------------------
    # Cohort comparison box plot (notebook cell 28)
    # ------------------------------------------------------------------
    if not skip_plots:
        plotting.plot_cohort_comparison_boxplot(
            same_diff_df=same_diff_df,
            comparison_distributions=comparison_distributions,
            summary_df=comparison_summary_df,
            output_dir=output_dir,
            base_key=base_key,
            threshold=cohort_comparison.resolve_comparison_settings(config)["thr"],
            top_k=cohort_comparison.resolve_comparison_settings(config)["top_k"],
        )
        print("")

    # ------------------------------------------------------------------
    # Final output listing (notebook cell 32)
    # ------------------------------------------------------------------
    print("Output files:")
    outputs = reports.print_outputs(output_dir)

    print("")
    print("Done. {} file(s) in {}".format(len(outputs), output_dir))

    if skip_plots:
        print("(--skip-plots was set: no figures were generated.)")

    return output_dir


def dry_run_plan(config_path):
    """Print the expanded analysis plan for a config and return it.

    Planning only: the raw config is read (no path resolution), the format is
    detected, and the plan is expanded and printed. No embeddings are loaded and
    nothing is written to disk.
    """
    raw_config = config_module.load_config(config_path)
    plan = analysis_plan.build_plan(raw_config, config_path=config_path)
    print(analysis_plan.format_plan(plan))
    return plan


def main(argv=None):
    args = parse_args(argv)

    if args.dry_run_plan:
        dry_run_plan(args.config)
        return

    raw_config = config_module.load_config(args.config)

    if analysis_plan.detect_format(raw_config) == analysis_plan.MULTI_COHORT:
        # Multi-cohort config: execute analysis_plan.pairwise + combined + comparisons.
        config = build_config(args.config, output_root=args.output_root)
        plan = analysis_plan.build_plan(raw_config, config_path=args.config)
        multi_runner.run_multi_plan(
            config,
            plan,
            skip_plots=args.skip_plots,
            config_path=args.config,
            repo_root=REPO_ROOT,
        )
        return

    config = build_config(args.config, output_root=args.output_root)
    run(config, skip_plots=args.skip_plots, config_path=args.config)


if __name__ == "__main__":
    main()
