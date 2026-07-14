"""TSV writers and the final output listing.

Collects the scattered to_csv calls from notebook cells 9, 12, 13, 19 and 27,
and the output listing from cell 32, so that every artifact this pipeline writes
goes through one place.

Artifacts written by a full run:

    run_config.json                                           (cell 5)
    target_cohort_metadata.tsv                                (cell 9)
    pairwise_distance_matrix.tsv                              (cell 12)
    pairwise_rank_matrix.tsv                                  (cell 13)
    {target}_validation_pairwise_{source}_{linkage}.svg       (cell 16)
    target_random_percentile_summary.tsv                      (cell 19)
    target_random_percentile_distributions.tsv                (cell 19)
    {target}_random_vs_target_boxplot.{png,jpeg,svg}          (cell 21)
    {target}_random_vs_target_kde.{png,jpeg,svg}              (cell 21)
    {target}_random_vs_target_kde_percentile_report.tsv       (cell 21)
    {base}_cohort_comparison_summary.tsv                      (cell 27)
    {base}_cohort_comparison_distributions.tsv                (cell 27)
    {base}_cohort_comparison_boxplot.{png,svg,jpeg}           (cell 28)

The KDE percentile report is written inside the plotting function, as in the
notebook, so it is not routed through write_table.
"""

from pathlib import Path


def write_table(df, output_dir, filename, index=False):
    """Write a DataFrame as a tab-separated file and return its path.

    index=False for the record tables; index=True for the distance and rank
    matrices, which carry their image IDs in the index.
    """
    output_path = Path(output_dir) / filename
    df.to_csv(output_path, sep="\t", index=index)

    return output_path


def list_outputs(output_dir):
    """Return every file in the run directory, sorted (notebook cell 32)."""
    return sorted([str(p) for p in Path(output_dir).glob("*")])


def print_outputs(output_dir):
    """Print the run directory listing, as the notebook's final cell does."""
    outputs = list_outputs(output_dir)
    print("\n".join(outputs))

    return outputs
