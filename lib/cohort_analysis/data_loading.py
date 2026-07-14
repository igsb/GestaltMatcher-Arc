"""Loading of metadata, gallery image IDs, and cohort embedding pickles.

Ported from notebook cells 7 (loader helpers), 9 (cohort loading loop), 10
(gallery loading) and 25 (same/different distributions).

Behaviour is preserved exactly as in the notebook, including the quirks called
out in the TODOs below. Do not "fix" them until the refactored pipeline has been
shown to reproduce the notebook's outputs.
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def parse_img_name(x, mode="drop_last_token"):
    """Derive the image ID from an embedding table's img_name column.

    "none"            -> unchanged
    "first_token"     -> text before the first underscore
    "drop_last_token" -> everything except the final underscore-separated token
    """
    x = str(x)

    if mode == "none":
        return x

    if mode == "first_token":
        return str(x.split("_")[0])

    if mode == "drop_last_token":
        parts = x.split("_")
        return str("_".join(parts[:-1])) if len(parts) > 1 else x

    raise ValueError("Unknown img_name_parser: {}".format(mode))


def load_embedding_df(path, img_name_parser="drop_last_token",
                      img_col="img_name", rep_col="representations"):
    """Load an embedding pickle and collapse its per-TTA rows into one row per image.

    Each image appears once per TTA/model representation, so the table is grouped
    by img_col and every column is aggregated into a list. The image name is then
    reduced to an image ID by the configured parser.
    """
    df = pd.read_pickle(path)

    if img_col not in df.columns:
        raise ValueError("{} does not contain column {!r}".format(path, img_col))

    if rep_col not in df.columns:
        raise ValueError("{} does not contain column {!r}".format(path, rep_col))

    # group by img_name because each image has multiple TTA/model representations
    df = df.groupby(img_col).agg(lambda x: list(x)).reset_index()
    df[img_col] = df[img_col].apply(lambda x: parse_img_name(x, img_name_parser))
    df = df.rename(columns={img_col: "img_name", rep_col: "representations"})
    df["img_name"] = df["img_name"].astype(str)

    return df


def load_photo_metadata(path, sep="\t", frontal_face_only=True):
    """Load the patient metadata table and build the image_id lookup dicts.

    Returns (df, image_to_subject, image_to_gene, image_to_patient_name,
    subject_to_image).

    TODO: only image_to_subject is used downstream. image_to_gene,
    image_to_patient_name and subject_to_image are built and never read; the
    latter also silently collapses duplicate patient names. Kept for parity.

    TODO: the notebook's default for sep was the two-character string "\\t"
    rather than a tab. It never mattered, because the caller always passes
    config["photo_metadata_sep"] (a real tab). The default is a real tab here.
    """
    df = pd.read_csv(path, sep=sep)

    if frontal_face_only and "image_type" in df.columns:
        df = df[df["image_type"] == "Frontal face"].copy()

    df["image_id"] = df["image_id"].astype(str)

    image_to_subject = {}
    image_to_gene = {}
    image_to_patient_name = {}
    subject_to_image = {}

    if "patient_id" in df.columns:
        image_to_subject = {str(r["image_id"]): str(r["patient_id"]) for _, r in df.iterrows()}

    if "gene_names" in df.columns:
        image_to_gene = {str(r["image_id"]): str(r["gene_names"]) for _, r in df.iterrows()}

    if "patient_name" in df.columns:
        image_to_patient_name = {str(r["image_id"]): str(r["patient_name"]) for _, r in df.iterrows()}
        subject_to_image = {str(r["patient_name"]): str(r["image_id"]) for _, r in df.iterrows()}

    return df, image_to_subject, image_to_gene, image_to_patient_name, subject_to_image


def load_gallery_image_ids(metadata_files):
    """Collect the GMDB reference gallery image IDs from the metadata CSVs.

    Where a "split" column exists (the rare gallery file), only split == 0 is
    kept, matching the older notebooks.

    TODO: image_to_label is never read downstream. Kept for parity.
    """
    image_ids = []
    image_to_label = {}

    for path in metadata_files:
        df = pd.read_csv(path)

        if "split" in df.columns:
            # For rare gallery file, old notebooks often used split == 0
            df_use = df[df["split"] == 0].copy()
        else:
            df_use = df.copy()

        df_use["image_id"] = df_use["image_id"].astype(str)
        image_ids.extend(df_use["image_id"].tolist())

        if "label" in df_use.columns:
            image_to_label.update({str(r["image_id"]): r["label"] for _, r in df_use.iterrows()})

    return np.array(sorted(set(map(str, image_ids)))), image_to_label


def load_same_different_distributions(path):
    """Load the same/different syndrome ROC reference distributions.

    Normalizes the distribution labels to "Same syndromes" / "Different
    syndromes" and returns (df, same_vals, diff_vals). Used by the cross-cohort
    comparison; it lives here because it is a file loader.
    """
    df = pd.read_csv(path)

    required = {"distribution", "mean_pw_dist"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing columns in same/different distribution file: {}".format(missing)
        )

    df = df[["distribution", "mean_pw_dist"]].copy()

    # Normalize labels robustly
    label = df["distribution"].astype(str).str.strip().str.lower()

    df["distribution"] = np.where(
        label.isin(["same", "same syndrome", "same syndromes"]),
        "Same syndromes",
        np.where(
            label.isin(["different", "different syndrome", "different syndromes"]),
            "Different syndromes",
            df["distribution"].astype(str).str.strip(),
        ),
    )

    same_vals = df.loc[df["distribution"] == "Same syndromes", "mean_pw_dist"].dropna().to_numpy()
    diff_vals = df.loc[df["distribution"] == "Different syndromes", "mean_pw_dist"].dropna().to_numpy()

    print("Normalized distribution labels:")
    print(df["distribution"].value_counts())

    if same_vals.size == 0 or diff_vals.size == 0:
        raise ValueError(
            "Need both 'Same syndromes' and 'Different syndromes' in the ROC distribution file."
        )

    return df, same_vals, diff_vals


def load_cohorts(config):
    """Load the embedding table for every cohort in config["active_cohorts"].

    Notebook cell 9. Each cohort's excluded image IDs are dropped and a "cohort"
    column is attached. Returns cohort_reps: dict of cohort key -> DataFrame.
    """
    cohort_reps = {}

    for cohort_key in config["active_cohorts"]:
        cohort = config["cohorts"][cohort_key]

        rep = load_embedding_df(
            cohort["embedding_file"],
            img_name_parser=cohort.get("img_name_parser", "drop_last_token"),
        )

        exclude_ids = set(map(str, cohort.get("exclude_image_ids", [])))
        if exclude_ids:
            rep = rep[~rep["img_name"].astype(str).isin(exclude_ids)].copy()

        rep["cohort"] = cohort_key
        cohort_reps[cohort_key] = rep

    return cohort_reps


def build_target_df(cohort_reps, image_to_subject):
    """Build the target cohort metadata table: one row per image.

    Notebook cell 7. Images with no metadata entry fall back to using the image
    ID as the subject ID.

    TODO: family_id is set to the same value as subject_id, which makes the
    family-member logic a no-op (config["exclude_family_members"] is never read
    either, and id_column: family_id would therefore change nothing). Preserved
    as-is.
    """
    rows = []

    for cohort_key, rep_df in cohort_reps.items():
        for img in rep_df["img_name"].astype(str):
            rows.append({
                "image_id": img,
                "cohort": cohort_key,
                "subject_id": image_to_subject.get(str(img), str(img)),
                "family_id": image_to_subject.get(str(img), str(img)),
            })

    return pd.DataFrame(rows)
