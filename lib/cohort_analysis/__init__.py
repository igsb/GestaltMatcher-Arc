"""Cohort analysis pipeline extracted from
notebooks/GestaltMatcher_cohort_analysis_template.ipynb

Refactor goal (first pass): reproduce the notebook's current outputs exactly.
Behaviour is deliberately preserved as-is, including config keys the notebook
defines but never reads. Those are marked with TODO comments and must not be
"fixed" until reproducibility against the notebook has been confirmed.

Module layout mirrors the notebook sections:

    config             YAML config loading, output directory
    run_metadata       run_metadata.json provenance sidecar
    analysis_plan      multi-cohort config detection + plan expansion (planning only)
    multi_runner       multi-cohort execution: pairwise + combined + comparisons
    data_loading       metadata, gallery IDs, cohort embedding pickles
    embeddings         TTA reshaping, embedding lookup, distance matrices
    pairwise           target-vs-target distance matrix + rank matrix
    random_validation  one-image-per-patient sampling, random percentiles
    cohort_comparison  cross-cohort sampling, PPV reports
    plotting           heatmap, box plots, KDE plot
    reports            TSV writers, output listing

Submodules are intentionally not imported here, so that importing the package
does not pull in matplotlib/seaborn/sklearn. Import the submodule you need.
"""

__all__ = [
    "config",
    "run_metadata",
    "analysis_plan",
    "multi_runner",
    "data_loading",
    "embeddings",
    "pairwise",
    "random_validation",
    "cohort_comparison",
    "plotting",
    "reports",
]
