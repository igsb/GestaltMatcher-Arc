"""Configuration loading for the cohort analysis pipeline.

Replaces the hand-edited CONFIG dict in notebook cells 4 and 5. The loaded
config is a plain dict with exactly the same keys as the notebook CONFIG, so
downstream code can keep using config["..."] / config.get("...", fallback) with
identical semantics.

Reproducibility note: the notebook reads several keys it never defines
(pairwise_match_threshold, pairwise_match_rank, output_dir) and defines several
keys it never reads (heatmap_threshold, heatmap_match_rank,
heatmap_linkage_method, comparison_groups, exclude_family_members). Both sets
are kept untouched. No defaults are injected here: the notebook's .get()
fallbacks at the call sites are what actually determine the numbers, so adding
defaults here would silently change results. See the TODOs in the YAML files.
"""

import copy
import json
from pathlib import Path

import yaml

# Config keys holding a single path to an input file or directory.
PATH_KEYS = (
    "gallery_embedding_file",
    "photo_metadata_file",
    "random_distribution_file",
    "same_different_distribution_file",
    "input_crops_path",
    "output_root",
)

# Config keys holding a list of paths.
PATH_LIST_KEYS = ("gallery_metadata_files",)

# Path key inside each entry of the "cohorts" mapping.
COHORT_PATH_KEY = "embedding_file"


def load_config(path):
    """Load a cohort analysis YAML config into a dict.

    The returned dict is used exactly like the notebook's CONFIG. No defaults
    are injected and no keys are renamed.
    """
    path = Path(path)

    with open(str(path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            "Config file {} must contain a YAML mapping, got {}".format(
                path, type(config).__name__
            )
        )

    return config


def _resolve_one(value, base_dir):
    """Resolve a single path value against base_dir, leaving None untouched.

    Absolute paths and ~ are honoured as given; relative paths are taken
    relative to base_dir (the repository root), not the current working
    directory. Returns a str so the config stays JSON-serializable.
    """
    if value is None:
        return None

    path = Path(str(value)).expanduser()

    if not path.is_absolute():
        path = Path(base_dir) / path

    return str(path)


def resolve_paths(config, base_dir):
    """Resolve relative input paths in the config against base_dir.

    The notebook used paths relative to the current working directory, which is
    why running it from notebooks/ created a second analysis_output/ tree.
    Resolving against the repository root removes that ambiguity without
    changing which files are read.

    Only keys that are actually present are touched; missing keys are not
    invented, so the notebook's .get() fallbacks still behave the same way.
    Returns a new dict; the input is not mutated.
    """
    config = copy.deepcopy(config)

    for key in PATH_KEYS:
        if key in config:
            config[key] = _resolve_one(config[key], base_dir)

    for key in PATH_LIST_KEYS:
        if key in config:
            config[key] = [_resolve_one(v, base_dir) for v in config[key]]

    for cohort in config.get("cohorts", {}).values():
        if COHORT_PATH_KEY in cohort:
            cohort[COHORT_PATH_KEY] = _resolve_one(cohort[COHORT_PATH_KEY], base_dir)

    return config


def get_output_dir(config):
    """Return the run output directory as a Path, without creating it.

    Mirrors notebook cell 5:
        OUTPUT_DIR = Path(output_root) / f"{target_name}_{run_date}"
    """
    return Path(config["output_root"]) / "{}_{}".format(
        config["target_name"], config["run_date"]
    )


def prepare_output_dir(config):
    """Create the run output directory and return it as a Path.

    Notebook cell 5: mkdir(parents=True, exist_ok=True).
    """
    output_dir = get_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def dump_run_config(config, output_dir):
    """Write the effective config to output_dir/run_config.json.

    Notebook cell 5 used json.dump(CONFIG, f, indent=2). What is written here
    is the config *after* path resolution, i.e. what the run actually used, so
    the recorded paths are absolute where the notebook recorded them relative.
    That is a provenance difference only; it does not affect any computed
    result or figure.
    """
    output_dir = Path(output_dir)
    output_path = output_dir / "run_config.json"

    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return output_path
