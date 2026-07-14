"""Cohort-level random percentile analysis.

Ported from the sampling helpers in notebook cell 7 and the analysis in cell 19.

The question this answers: are the faces in the target cohort more similar to
each other than an arbitrary group of unrelated patients would be? Distance is
used throughout, so a LOWER mean pairwise distance means MORE facially similar.

TODO (question for Tzung-Chien): the one-image-per-patient constraint is
currently inert. build_target_df falls back to subject_id = image_id whenever an
image is absent from the patient metadata table, and for the PRMT7 / CTNND2 /
LINS1 cohorts NOT ONE of the image IDs is present there (the cohorts use IDs like
F1-1, N_17, CTNND2_N_58; the metadata table is keyed on numeric GMDB IDs). So
every image is treated as its own patient, and sampling "at most one image per
patient" places no constraint at all. Images like F1-1 and F1-2 look like two
members of one family, which is exactly what this constraint exists to exclude.
Preserved as-is for now; this must be resolved before the numbers are trusted.
"""

import itertools
import random
from collections import defaultdict

import numpy as np
import pandas as pd


def one_image_per_patient_combinations(image_ids, patient_ids, n_samples=1000,
                                       min_patients=2, max_patients=None, seed=0):
    """Sample random subsets of images, taking at most one image per patient.

    Each sample picks a random number of patients between min_patients and
    max_patients, then one random image from each chosen patient.
    """
    rng = random.Random(seed)

    patient_images = defaultdict(list)
    for img_id, pat_id in zip(image_ids, patient_ids):
        patient_images[str(pat_id)].append(str(img_id))

    patients = list(patient_images.keys())
    if len(patients) < min_patients:
        return []

    if max_patients is None:
        max_patients = len(patients)
    max_patients = min(max_patients, len(patients))

    combos = []
    for _ in range(n_samples):
        r = rng.randint(min_patients, max_patients)
        chosen_patients = rng.sample(patients, r)
        chosen_images = [rng.choice(patient_images[p]) for p in chosen_patients]
        combos.append(chosen_images)

    return combos


def mean_pairwise_distance_for_combo(combo, distance_df):
    """Mean distance over every image pair within one sampled subset."""
    values = []
    for a, b in itertools.combinations(map(str, combo), 2):
        values.append(float(distance_df.loc[a, b]))

    return float(np.mean(values)) if values else np.nan


def random_percentile(value, random_values):
    """Locate a value within the random reference distribution.

    Returns (percentile_lower_or_equal, similarity_percentile). Distance is used,
    so a lower value means more facially similar; the similarity percentile is
    therefore inverted, and higher means more similar.
    """
    # For distance: lower values mean more facially similar.
    random_values = np.asarray(random_values, dtype=float)

    percentile_lower_or_equal = 100 * np.mean(random_values <= value)
    similarity_percentile = 100 - percentile_lower_or_equal

    return percentile_lower_or_equal, similarity_percentile


def ppv_at_threshold(distribution_df, threshold, value_col="mean_pw_dist",
                     label_col="distribution", positive_label="same",
                     negative_label="different"):
    """Confusion matrix and PPV at a distance threshold ("same" when <= threshold).

    TODO: dead code. Defined in notebook cell 7 and never called anywhere; the
    cross-cohort comparison uses its own ppv_different_at_distance instead.
    Ported for parity only. Decide whether to keep it once the pipeline is
    reproduced.
    """
    # For distance: classify "same/similar" when distance <= threshold.
    pred_positive = distribution_df[value_col] <= threshold

    tp = int(((distribution_df[label_col] == positive_label) & pred_positive).sum())
    fp = int(((distribution_df[label_col] == negative_label) & pred_positive).sum())
    fn = int(((distribution_df[label_col] == positive_label) & ~pred_positive).sum())
    tn = int(((distribution_df[label_col] == negative_label) & ~pred_positive).sum())

    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "PPV": ppv,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def run_random_validation(target_df, distance_df, config):
    """Compare the target cohort against the random reference distribution.

    Notebook cell 19. Returns (summary_df, combined_random_distribution), where
    the combined frame stacks the random reference rows and the target's sampled
    rows and is what the plots consume.
    """
    raw_random_distribution = pd.read_csv(config["random_distribution_file"])
    raw_random_distribution = raw_random_distribution[["mean_pw_dist", "distribution"]].copy()

    print("Original distribution labels:")
    print(raw_random_distribution["distribution"].value_counts(dropna=False))

    # Keep only rows that are truly random
    random_distribution = raw_random_distribution[
        raw_random_distribution["distribution"].astype(str).str.lower() == "random"
    ].copy()

    random_distribution["distribution"] = "random"

    target_name = config.get("target_name", "target")

    tmp = target_df.copy()
    image_ids = tmp["image_id"].astype(str).values
    patient_ids = tmp[config["id_column"]].astype(str).values

    combos = one_image_per_patient_combinations(
        image_ids,
        patient_ids,
        n_samples=config["n_random_combinations"],
        seed=config["random_seed"],
    )

    vals = [mean_pairwise_distance_for_combo(combo, distance_df) for combo in combos]
    vals = [v for v in vals if not np.isnan(v)]

    target_distribution = pd.DataFrame({
        "mean_pw_dist": vals,
        "distribution": target_name,
    })

    target_mean = float(np.mean(vals)) if vals else np.nan

    lower_pct, similarity_pct = random_percentile(
        target_mean,
        random_distribution["mean_pw_dist"].values,
    )

    summary_df = pd.DataFrame([{
        "target": target_name,
        "n_images": len(set(image_ids)),
        "n_patients": len(set(patient_ids)),
        "n_sampled_combinations": len(vals),
        "mean_of_sampled_mean_pw_dist": target_mean,
        "random_percentile_lower_or_equal": lower_pct,
        "similarity_percentile_higher_is_more_similar": similarity_pct,
    }])

    combined_random_distribution = pd.concat(
        [random_distribution, target_distribution],
        ignore_index=True,
    )

    print("Final plot distributions:")
    print(combined_random_distribution["distribution"].value_counts())

    return summary_df, combined_random_distribution
