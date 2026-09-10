#!/usr/bin/env python3
"""
Run GestaltMatcher prediction from a precomputed encoding pickle file.

This script does NOT evaluate against NTUH.xlsx. It only produces the GM result
for each patient/case in the encoding pickle. You can use a second script later
to compare these results with the Excel truth table.

Typical use inside the GestaltMatcher service/docker repo root:

python run_gm_predictions_from_encoding_pickle.py \
  --encoding_pkl encodings_v1.1.0/test_encodings_v1.1.0_keep_0_4_11.pkl \
  --output_dir ntuh_prediction_results

Expected repo files:
- evaluation.py in the same folder or on PYTHONPATH
- data/image_gene_and_syndrome_metadata_pp4_12062025_max.p
- data/gallery_encodings/GMDB_gallery_encodings_12062025_v1.1.0_service.pkl
- data/transformation_probabilities_07052025.csv optional, but recommended

Input pickle:
- A pandas DataFrame produced by the v1.1.0 encoder, with columns:
  img_name, model, flip, gray, class_conf, representations

Output:
- json/<case_id>.json: full GM result for each patient/case
- gm_prediction_summary.csv: compact top suggestions per patient/case
- gm_prediction_summary.xlsx: same as CSV, if openpyxl is installed
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from lib.evaluation import get_gallery_encodings_set, predict
except ImportError:
    from evaluation import get_gallery_encodings_set, predict


METADATA_DEFAULT = "data/image_gene_and_syndrome_metadata_pp4_12062025_max.p"
PROBABILITIES_DEFAULT = "data/transformation_probabilities_07052025.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GM v1.1.0 prediction from a precomputed encoding pickle. No Excel matching."
    )
    parser.add_argument(
        "--encoding_pkl",
        default="encodings_v1.1.0/test_encodings_v1.1.0_keep_0_4_11.pkl",
        help="Input encoding pickle file.",
    )
    parser.add_argument(
        "--output_dir",
        default="ntuh_prediction_results",
        help="Output folder for per-case JSON files and compact summary tables.",
    )
    parser.add_argument(
        "--metadata",
        default=METADATA_DEFAULT,
        help="GM image/gene/syndrome metadata pickle.",
    )
    parser.add_argument(
        "--probabilities",
        default=PROBABILITIES_DEFAULT,
        help="Syndrome probability CSV. Optional but recommended.",
    )
    parser.add_argument(
        "--top_n_summary",
        type=int,
        default=30,
        help="Number of top genes/syndromes to keep in the compact summary columns.",
    )
    parser.add_argument(
        "--json_top_n",
        type=int,
        default=0,
        help=(
            "If >0, trim suggested_genes_list/suggested_syndromes_list/suggested_patients_list "
            "inside each JSON to this top N. Default 0 saves the full result."
        ),
    )
    return parser.parse_args()


def normalize_case_id(value: object) -> str:
    """Normalize image names like N10_RussellSilver.jpg or P01_xxx.jpg to N10/P1."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    text = Path(text).stem
    text = re.split(r"[_\s]", text)[0]
    m = re.match(r"^([A-Za-z]+)0*(\d+)$", text)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return text.upper()


def load_metadata(metadata_path: str, probabilities_path: str):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "rb") as f:
        data = pickle.load(f)

    images_synds_dict = data["disorder_level_metadata"]
    images_genes_dict = data["gene_level_metadata"]
    genes_metadata_dict = data["gene_metadata"]
    synds_metadata_dict = data["disorder_metadata"]

    gallery_df = get_gallery_encodings_set(images_synds_dict)

    synds_probabilities_dict: Dict[str, Dict[str, float]] = {}
    if probabilities_path and os.path.exists(probabilities_path):
        prob_df = pd.read_csv(probabilities_path, sep=",")
        synds_probabilities_dict = {
            row["syndrome"]: {
                "(Intercept)": row["(Intercept)"],
                "syn_scores": row["syn_scores"],
                "v_00": row["v_00"],
                "v_10": row["v_10"],
                "v_11": row["v_11"],
            }
            for _, row in prob_df.iterrows()
        }
    else:
        print(
            f"Warning: probability file not found: {probabilities_path}. Probability fields will be None.",
            file=sys.stderr,
        )

    return (
        gallery_df,
        images_synds_dict,
        images_genes_dict,
        genes_metadata_dict,
        synds_metadata_dict,
        synds_probabilities_dict,
    )


def load_case_encodings(encoding_pkl: str) -> pd.DataFrame:
    if not os.path.exists(encoding_pkl):
        raise FileNotFoundError(f"Encoding pickle not found: {encoding_pkl}")

    raw_df = pd.read_pickle(encoding_pkl)
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got: {type(raw_df)}")

    required_cols = {"img_name", "representations"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"Encoding pickle is missing required columns: {sorted(missing)}")

    # Important:
    # Do NOT group with prep_csv here. evaluation.predict() expects one row per
    # model/TTA encoding for the case. Grouping first would create one row whose
    # representations cell contains a list of vectors, leading to a 3D array error
    # in sklearn pairwise_distances.
    raw_df = raw_df.copy()
    raw_df["case_id"] = raw_df["img_name"].apply(normalize_case_id)

    print(f"Loaded raw encoding rows: {len(raw_df)}")
    print(f"Unique img_name values: {raw_df['img_name'].nunique()}")
    print(f"Unique case_id values: {raw_df['case_id'].nunique()}")

    counts = raw_df.groupby("case_id").size()
    print("Rows per case_id:")
    print(counts.value_counts().sort_index().to_string())

    return raw_df

def trim_result(result: dict, top_n: int) -> dict:
    if top_n <= 0:
        return result
    trimmed = dict(result)
    for key in ["suggested_genes_list", "suggested_syndromes_list", "suggested_patients_list"]:
        if key in trimmed and isinstance(trimmed[key], list):
            trimmed[key] = trimmed[key][:top_n]
    return trimmed


def join_names(rows: Sequence[dict], key: str, top_n: int) -> str:
    return "; ".join(str(row.get(key, "")) for row in rows[:top_n])


def join_distances(rows: Sequence[dict], top_n: int) -> str:
    return "; ".join(str(row.get("distance", "")) for row in rows[:top_n])


def make_summary_row(case_id: str, original_img_name: str, result: dict, status: str, error_message: str, top_n: int) -> dict:
    genes = result.get("suggested_genes_list", []) if result else []
    syndromes = result.get("suggested_syndromes_list", []) if result else []
    patients = result.get("suggested_patients_list", []) if result else []

    top_gene = genes[0] if genes else {}
    top_syndrome = syndromes[0] if syndromes else {}
    top_patient = patients[0] if patients else {}

    return {
        "case_id": case_id,
        "img_name": original_img_name,
        "status": status,
        "error_message": error_message,
        "model_version": result.get("model_version", "") if result else "",
        "gallery_version": result.get("gallery_version", "") if result else "",
        "has_genetic_disorder": result.get("has_genetic_disorder", "") if result else "",
        "has_genetic_disorder_dist_threshold": result.get("has_genetic_disorder_dist_threshold", "") if result else "",
        "top_gene": top_gene.get("gene_name", ""),
        "top_gene_distance": top_gene.get("distance", ""),
        "top_gene_gestalt_score": top_gene.get("gestalt_score", ""),
        "top_gene_reference_subject_id": top_gene.get("subject_id", ""),
        "top_syndrome": top_syndrome.get("syndrome_name", ""),
        "top_syndrome_omim_id": top_syndrome.get("omim_id", ""),
        "top_syndrome_distance": top_syndrome.get("distance", ""),
        "top_syndrome_gestalt_score": top_syndrome.get("gestalt_score", ""),
        "top_syndrome_reference_subject_id": top_syndrome.get("subject_id", ""),
        "top_reference_patient_subject_id": top_patient.get("subject_id", ""),
        "top_reference_patient_gene": top_patient.get("gene_name", ""),
        "top_reference_patient_syndrome": top_patient.get("syndrome_name", ""),
        f"top{top_n}_genes": join_names(genes, "gene_name", top_n),
        f"top{top_n}_gene_distances": join_distances(genes, top_n),
        f"top{top_n}_syndromes": join_names(syndromes, "syndrome_name", top_n),
        f"top{top_n}_syndrome_distances": join_distances(syndromes, top_n),
    }


def main() -> None:
    args = parse_args()
    start = time.time()

    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    case_df = load_case_encodings(args.encoding_pkl)

    (
        gallery_df,
        images_synds_dict,
        images_genes_dict,
        genes_metadata_dict,
        synds_metadata_dict,
        synds_probabilities_dict,
    ) = load_metadata(args.metadata, args.probabilities)

    summary_rows: List[dict] = []

    grouped_cases = list(case_df.groupby("case_id", sort=False))

    for i, (case_id, single_case_df) in enumerate(grouped_cases):
        original_img_name = str(single_case_df["img_name"].iloc[0])

        # Keep one row per selected model/TTA encoding, usually rows 0, 4, 11
        # after your filtering step. Only the representations column is required
        # by evaluation.predict(), but keeping img_name is useful for debugging.
        single_case_df = single_case_df[["img_name", "representations"]].copy().reset_index(drop=True)

        try:
            result = predict(
                single_case_df,
                gallery_df,
                images_synds_dict,
                images_genes_dict,
                genes_metadata_dict,
                synds_metadata_dict,
                synds_probabilities_dict,
            )
            status = "ok"
            error_message = ""
        except Exception as e:
            result = {}
            status = "error"
            error_message = str(e)

        json_payload = {
            "case_id": case_id,
            "img_name": original_img_name,
            "status": status,
            "error_message": error_message,
            "result": trim_result(result, args.json_top_n),
        }

        json_path = json_dir / f"{case_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)

        summary_rows.append(
            make_summary_row(
                case_id=case_id,
                original_img_name=original_img_name,
                result=result,
                status=status,
                error_message=error_message,
                top_n=args.top_n_summary,
            )
        )

        top_gene = summary_rows[-1].get("top_gene", "")
        top_syndrome = summary_rows[-1].get("top_syndrome", "")
        print(f"Finished {i + 1}/{len(grouped_cases)}: {case_id} | top_gene={top_gene} | top_syndrome={top_syndrome}")

    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "gm_prediction_summary.csv"
    xlsx_path = output_dir / "gm_prediction_summary.xlsx"
    summary_df.to_csv(csv_path, index=False)
    try:
        summary_df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"Warning: could not write Excel output ({e}). CSV was still written.", file=sys.stderr)

    print("\nFinished GM prediction from encoding pickle")
    print(f"Input pickle: {args.encoding_pkl}")
    print(f"Cases evaluated: {len(summary_df)}")
    print(f"Per-case JSON folder: {json_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary XLSX: {xlsx_path}")
    print(f"Total running time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
