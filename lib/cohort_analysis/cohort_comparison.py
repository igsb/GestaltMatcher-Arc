"""Cross-cohort comparison and PPV reporting.

Ported from notebook cells 23 through 27.

The question this answers: does the base cohort look DIFFERENT from each
comparison cohort? Sub-blocks of the cross-cohort distance matrix are sampled,
and each sampled mean distance is converted into a PPV for "different syndrome"
against the same/different syndrome reference distributions.

Higher mean pairwise distance = more evidence the two cohorts are different.
"""

import numpy as np
import pandas as pd

from lib.cohort_analysis import embeddings


def build_group_image_ids(target_df, active_cohorts):
    """Map each active cohort key to its list of image IDs (notebook cell 24).

    Cohorts with no images are warned about and omitted, as in the notebook.
    """
    group_image_ids = {}

    for cohort_name in active_cohorts:
        ids = (
            target_df[target_df["cohort"] == cohort_name]["image_id"]
            .astype(str)
            .tolist()
        )

        if len(ids) == 0:
            print("Warning: no images found for cohort {}".format(cohort_name))
        else:
            group_image_ids[cohort_name] = ids

    return group_image_ids


def build_rep_df_for_comparison(cohort_reps):
    """Concatenate the per-cohort representation tables into one table.

    Notebook cell 26 sniffed globals() for all_target_reps / target_rep_df /
    rep_all / cohort_reps and used whichever existed. Only the cohort_reps branch
    ever ran, because the notebook never defines the other three. That branch is
    reproduced here exactly; cohort_reps is now an explicit argument.
    """
    frames = []

    for cohort_name, rep in cohort_reps.items():
        tmp = rep.copy()
        tmp["cohort"] = cohort_name

        # If image ID is stored in the index, move it into img_name
        if "img_name" not in tmp.columns and "image_id" not in tmp.columns:
            tmp = tmp.reset_index()
            first_col = tmp.columns[0]
            tmp = tmp.rename(columns={first_col: "img_name"})

        frames.append(tmp)

    return pd.concat(frames, ignore_index=True)


def sample_mean_cross_distances(D, n_samples=100, min_a=2, min_b=2, seed=0):
    """Randomly sample sub-blocks of the cross-cohort matrix and take their means.

    D: rows are base cohort images, columns are comparison cohort images.
    """
    rng = np.random.default_rng(seed)

    n_a, n_b = D.shape

    if n_a < min_a:
        raise ValueError("Base cohort has only {} images, but min_a={}".format(n_a, min_a))

    if n_b < min_b:
        raise ValueError(
            "Comparison cohort has only {} images, but min_b={}".format(n_b, min_b)
        )

    out = []

    for _ in range(n_samples):
        k_a = int(rng.integers(min_a, n_a + 1))
        k_b = int(rng.integers(min_b, n_b + 1))

        idx_a = rng.choice(n_a, size=k_a, replace=False)
        idx_b = rng.choice(n_b, size=k_b, replace=False)

        out.append(float(D[np.ix_(idx_a, idx_b)].mean()))

    return np.array(out)


def count_unique_patients(image_ids, target_df, id_column):
    """Number of distinct patients among the given images.

    TODO: see the note in random_validation. For the current cohorts every image
    falls back to being its own patient, so this returns the image count.
    """
    tmp = target_df[["image_id", id_column]].copy()
    tmp["image_id"] = tmp["image_id"].astype(str)
    tmp[id_column] = tmp[id_column].astype(str)

    return tmp[tmp["image_id"].isin(set(map(str, image_ids)))][id_column].nunique()


def ppv_different_at_distance(d, same_vals, diff_vals, pretest_probability=0.5):
    """PPV for "different syndrome" at distance threshold d.

    Positive evidence for different = mean_pw_dist >= d, so:
        sensitivity = P(distance >= d | Different syndromes)
        specificity = P(distance <  d | Same syndromes)
    """
    p = float(pretest_probability)

    sensitivity = float((diff_vals >= d).mean())
    specificity = float((same_vals < d).mean())

    denom = sensitivity * p + (1.0 - specificity) * (1.0 - p)

    ppv = (sensitivity * p) / denom if denom > 0 else np.nan

    return ppv, sensitivity, specificity


def ppv_range_from_sampled_distances(sampled_means, same_vals, diff_vals,
                                     pretest_probability=0.5):
    """PPV for every sampled mean distance, summarized as a range."""
    sampled_means = np.asarray(sampled_means, dtype=float)

    ppv_values = []
    sensitivity_values = []
    specificity_values = []

    for d in sampled_means:
        ppv, sens, spec = ppv_different_at_distance(
            d,
            same_vals=same_vals,
            diff_vals=diff_vals,
            pretest_probability=pretest_probability,
        )

        ppv_values.append(ppv)
        sensitivity_values.append(sens)
        specificity_values.append(spec)

    ppv_values = np.asarray(ppv_values, dtype=float)
    sensitivity_values = np.asarray(sensitivity_values, dtype=float)
    specificity_values = np.asarray(specificity_values, dtype=float)

    return {
        "distance_min": float(np.min(sampled_means)),
        "distance_median": float(np.median(sampled_means)),
        "distance_mean": float(np.mean(sampled_means)),
        "distance_max": float(np.max(sampled_means)),

        "PPV_min": float(np.nanmin(ppv_values)),
        "PPV_25th": float(np.nanquantile(ppv_values, 0.25)),
        "PPV_median": float(np.nanmedian(ppv_values)),
        "PPV_mean": float(np.nanmean(ppv_values)),
        "PPV_75th": float(np.nanquantile(ppv_values, 0.75)),
        "PPV_max": float(np.nanmax(ppv_values)),

        "sensitivity_median": float(np.nanmedian(sensitivity_values)),
        "specificity_median": float(np.nanmedian(specificity_values)),
        "pretest_p": float(pretest_probability),
    }


def resolve_comparison_settings(config):
    """Pull the cross-cohort settings out of the config (notebook cell 23).

    The threshold falls back to config["ppv_threshold"], then to the notebook's
    hardcoded literal, exactly as the notebook does.
    """
    cross_cfg = config["cross_cohort_comparison"]

    return {
        "base_key": cross_cfg["base_cohort"],
        "comparison_cohorts": cross_cfg.get("comparison_cohorts", "all"),
        "n_samples": cross_cfg.get("n_samples", 100),
        "min_a": cross_cfg.get("min_base_images", 2),
        "min_b": cross_cfg.get("min_comparison_images", 2),
        "thr": float(
            cross_cfg.get("threshold", config.get("ppv_threshold", 0.8963266102947289))
        ),
        "p_pretest": float(cross_cfg.get("pretest_probability", 0.5)),
        "top_k": int(cross_cfg.get("top_k", 8)),
    }


def run_cross_cohort_comparison(cohort_reps, target_df, group_image_ids,
                                same_vals, diff_vals, config):
    """Compare the base cohort against each comparison cohort (notebook cell 27).

    Returns (summary_df, distributions_df). The summary is sorted by
    mean_pw_distance_all_pairs, as in the notebook.
    """
    settings = resolve_comparison_settings(config)

    base_key = settings["base_key"]
    comparison_cohorts = settings["comparison_cohorts"]

    rep_df_for_comparison = build_rep_df_for_comparison(cohort_reps)

    if base_key not in group_image_ids:
        raise KeyError(
            "Base cohort '{}' not found in group_image_ids. Available keys: {}".format(
                base_key, list(group_image_ids.keys())
            )
        )

    if comparison_cohorts == "all":
        comparison_keys = [k for k in group_image_ids.keys() if k != base_key]
    else:
        comparison_keys = list(comparison_cohorts)

    missing = [k for k in comparison_keys if k not in group_image_ids]
    if missing:
        raise KeyError(
            "Comparison cohort(s) not found: {}. Available keys: {}".format(
                missing, list(group_image_ids.keys())
            )
        )

    base_ids = group_image_ids[base_key]

    E_base, base_ids_used = embeddings.get_embeddings_tensor_for_ids(
        base_ids, rep_df_for_comparison
    )

    base_n_images = len(base_ids_used)
    base_n_patients = count_unique_patients(base_ids_used, target_df, config["id_column"])

    comparison_distribution_frames = []
    summary_rows = []

    for comparison_key in comparison_keys:
        comp_ids = group_image_ids[comparison_key]

        E_comp, comp_ids_used = embeddings.get_embeddings_tensor_for_ids(
            comp_ids, rep_df_for_comparison
        )

        D = embeddings.cosine_distance_matrix_mean_tta(E_base, E_comp)

        sampled_means = sample_mean_cross_distances(
            D,
            n_samples=settings["n_samples"],
            min_a=settings["min_a"],
            min_b=settings["min_b"],
            seed=config.get("random_seed", 0),
        )

        comparison_name = "{} vs {}".format(base_key, comparison_key)

        comparison_distribution_frames.append(pd.DataFrame({
            "mean_pw_dist": sampled_means,
            "distribution": comparison_name,
        }))

        prop_above_c = float((sampled_means > settings["thr"]).mean())
        evidence_different = bool(prop_above_c > 0.5)

        ppv_stats = ppv_range_from_sampled_distances(
            sampled_means=sampled_means,
            same_vals=same_vals,
            diff_vals=diff_vals,
            pretest_probability=settings["p_pretest"],
        )

        comp_n_images = len(comp_ids_used)
        comp_n_patients = count_unique_patients(
            comp_ids_used, target_df, config["id_column"]
        )

        summary_rows.append({
            "comparison": comparison_name,
            "base_cohort": base_key,
            "comparison_cohort": comparison_key,

            "mean_pw_distance_all_pairs": float(D.mean()),
            "median_sampled_mean_pw_dist": float(np.median(sampled_means)),
            "mean_sampled_mean_pw_dist": float(np.mean(sampled_means)),

            "threshold_c": settings["thr"],
            "prop_sampled_above_c": prop_above_c,
            "evidence_different": evidence_different,

            **ppv_stats,

            "n_samples": int(len(sampled_means)),

            "base_n_images": int(base_n_images),
            "base_n_patients": int(base_n_patients),
            "comparison_n_images": int(comp_n_images),
            "comparison_n_patients": int(comp_n_patients),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("mean_pw_distance_all_pairs")
        .reset_index(drop=True)
    )

    distributions_df = pd.concat(comparison_distribution_frames, ignore_index=True)

    return summary_df, distributions_df
