"""Run metadata and config traceability for the cohort analysis pipeline.

Writes one extra artifact per run, run_metadata.json, next to the existing
run_config.json. This is provenance only: nothing here feeds back into the
numerical analysis, the TSV/plot outputs, or the config semantics, and
run_config.json is left exactly as dump_run_config writes it.

run_metadata.json records, for a single run:

    created_at              ISO-8601 UTC timestamp
    config_path             config path as passed on the command line
    config_path_abs         that path resolved to an absolute path
    config_sha256           sha256 of the original (unresolved) config file bytes
    resolved_config_sha256  sha256 of the run_config.json bytes written this run
    git                     commit / branch / dirty, or nulls if unavailable
    python                  version, implementation, executable
    platform                OS / release / version / machine / node
    packages                key package versions (None if not importable)
    output_dir              absolute run output directory
    target_name             target cohort name
    comparison_cohorts      cross_cohort_comparison.comparison_cohorts, if set
    base_cohort             cross_cohort_comparison.base_cohort, if set

Every collector degrades gracefully: a missing git binary, a non-repo checkout
or an absent package yields null rather than an error, so adding this step can
never break a run.
"""

import hashlib
import json
import platform as platform_module
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

RUN_METADATA_FILENAME = "run_metadata.json"

# Output key -> candidate distribution names, most specific first. Recorded via
# importlib.metadata so no heavy module is imported just to read its version.
_PACKAGE_DISTRIBUTIONS = {
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "scipy": ("scipy",),
    "sklearn": ("scikit-learn", "sklearn"),
    "matplotlib": ("matplotlib",),
    "seaborn": ("seaborn",),
    "Pillow": ("Pillow", "PIL"),
    "pyyaml": ("PyYAML", "pyyaml"),
}


def sha256_file(path):
    """Return the hex sha256 of a file's bytes, or None if it cannot be read."""
    if path is None:
        return None

    try:
        digest = hashlib.sha256()
        with open(str(path), "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _run_git(args, repo_root):
    """Run `git <args>` in repo_root, returning stripped stdout or None."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def collect_git_info(repo_root):
    """Collect commit hash, branch and dirty flag for repo_root.

    `dirty` reflects tracked changes only (`git status --porcelain
    --untracked-files=no`): untracked scratch files such as fresh output
    directories should not mark an otherwise-clean checkout as dirty. Any field
    that cannot be determined is None.
    """
    inside_work_tree = _run_git(
        ["rev-parse", "--is-inside-work-tree"], repo_root
    )
    if inside_work_tree != "true":
        return {"commit": None, "branch": None, "dirty": None}

    commit = _run_git(["rev-parse", "HEAD"], repo_root)

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if branch == "HEAD":
        # Detached head; keep the literal so it is not mistaken for a branch.
        branch = "HEAD (detached)"

    status = _run_git(
        ["status", "--porcelain", "--untracked-files=no"], repo_root
    )
    dirty = None if status is None else bool(status)

    return {"commit": commit, "branch": branch, "dirty": dirty}


def collect_python_info():
    """Collect the running interpreter's version and location."""
    return {
        "version": platform_module.python_version(),
        "implementation": platform_module.python_implementation(),
        "executable": sys.executable,
    }


def collect_platform_info():
    """Collect coarse OS / machine information."""
    return {
        "system": platform_module.system(),
        "release": platform_module.release(),
        "version": platform_module.version(),
        "machine": platform_module.machine(),
        "node": platform_module.node(),
        "platform": platform_module.platform(),
    }


def collect_package_versions():
    """Return {key: version_or_None} for the pipeline's key dependencies."""
    versions = {}
    for key, dist_names in _PACKAGE_DISTRIBUTIONS.items():
        version = None
        for dist_name in dist_names:
            try:
                version = importlib_metadata.version(dist_name)
                break
            except importlib_metadata.PackageNotFoundError:
                continue
        versions[key] = version
    return versions


def build_run_metadata(
    config,
    output_dir,
    run_config_path=None,
    config_path=None,
    repo_root=None,
):
    """Assemble the run_metadata dict. Pure: nothing is written here."""
    config_path_abs = None
    if config_path is not None:
        try:
            config_path_abs = str(Path(config_path).expanduser().resolve())
        except OSError:
            config_path_abs = None

    cross_cohort = config.get("cross_cohort_comparison") or {}

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": None if config_path is None else str(config_path),
        "config_path_abs": config_path_abs,
        "config_sha256": sha256_file(config_path),
        "resolved_config_sha256": sha256_file(run_config_path),
        "git": collect_git_info(repo_root if repo_root is not None else Path.cwd()),
        "python": collect_python_info(),
        "platform": collect_platform_info(),
        "packages": collect_package_versions(),
        "output_dir": str(output_dir),
        "target_name": config.get("target_name"),
        "base_cohort": cross_cohort.get("base_cohort"),
        "comparison_cohorts": cross_cohort.get("comparison_cohorts"),
    }

    return metadata


def dump_run_metadata(metadata, output_dir):
    """Write metadata to output_dir/run_metadata.json and return its path."""
    output_path = Path(output_dir) / RUN_METADATA_FILENAME

    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=False)
        f.write("\n")

    return output_path


def write_run_metadata(
    config,
    output_dir,
    run_config_path=None,
    config_path=None,
    repo_root=None,
):
    """Build and write run_metadata.json in one call; return its path."""
    metadata = build_run_metadata(
        config,
        output_dir,
        run_config_path=run_config_path,
        config_path=config_path,
        repo_root=repo_root,
    )
    return dump_run_metadata(metadata, output_dir)
