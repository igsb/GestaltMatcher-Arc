"""All figures for the cohort analysis pipeline.

Ported from notebook cells 15, 16, 20, 21 and 28. Figure sizes, font sizes,
palettes, axis limits and file formats are kept exactly as the notebook had
them, hardcoded. Do not promote them to config yet.

The only deliberate deviation: matplotlib is forced to the "Agg" backend. The
notebook ran under the inline backend, but in a CLI on Windows plt.show() would
otherwise try to open a blocking GUI window. Agg is non-interactive and writes
the same files; plt.show() becomes a no-op.

Reproducibility TODOs at the heatmap call site (see plot_pairwise_heatmap):
  - The notebook reads CONFIG["pairwise_match_threshold"] and
    CONFIG["pairwise_match_rank"], which the config does NOT define, so the
    hardcoded fallbacks 0.748 and 300 are what actually apply. The config keys
    heatmap_threshold and heatmap_match_rank are dead (same values by chance).
  - The notebook hardcodes linkage_method="single", so config
    heatmap_linkage_method ("complete") is dead. Wiring it up WOULD CHANGE the
    figure.
  - The notebook hardcodes the heatmap output format to SVG only.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.cluster.hierarchy as hc  # noqa: E402
import scipy.spatial as sp  # noqa: E402
import seaborn as sns  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402
from PIL import Image  # noqa: E402


# ----------------------------------------------------------------------
# Heatmap (notebook cells 15 and 16)
# ----------------------------------------------------------------------

def get_image_by_name(label, input_crops_path):
    """Find a face crop image by label, trying the common extensions.

    Raises FileNotFoundError when the crop is missing. That is the notebook's
    behaviour: it does NOT skip missing crops, it aborts the whole figure. Only
    reachable when input_crops_path is set, which it is not in the current
    configs (input_crops_path: null), so this path is currently unexercised.
    """
    possible_exts = ["", ".png", ".jpg", ".jpeg", ".JPG", ".PNG"]

    for ext in possible_exts:
        candidate = os.path.join(input_crops_path, str(label) + ext)
        if os.path.exists(candidate):
            return Image.open(candidate)

    raise FileNotFoundError(
        "Cannot find crop image for label: {} in {}".format(label, input_crops_path)
    )


def auto_heatmap_style(n_images):
    """Annotation/tick font size and rotation, scaled to the image count."""
    ann_size = 20
    tick_size = 18
    rotation = 0

    if n_images >= 40:
        ann_size = 6
        tick_size = 8
        rotation = 30
    elif n_images >= 20:
        ann_size = 10
        tick_size = 10
        rotation = 30
    elif n_images >= 15:
        ann_size = 16
        tick_size = 14

    return ann_size, tick_size, rotation


def plot_clustering_heatmap(
    dist_df,
    rank_df,
    cnames,
    family_labels,
    gene_name,
    output_path=".",
    file_format="png",
    ann_size=None,
    tick_size=None,
    rotation=None,
    title_size=36,
    label_size=36,
    fig_size=(16, 16),
    threshold=0.65,
    match_rank=30,
    display_match_box=False,
    map_dict=None,
    source_type="distance",
    input_crops_path=None,
    row_cluster=True,
    col_cluster=True,
    file_suffix="",
    linkage_method="single",
    display_number=True,
    default_x_offset=None,
    default_y_offset=None,
    xaxis_ticks_x_offset=0.5,
    xaxis_ticks_y_offset=0.01,
    rotation_mode="anchor",
    ha="right",
):
    """Heatmap + clustering dendrogram for the pairwise cohort comparison.

    source_type:
        "distance" -> show pairwise distance values
        "rank"     -> show pairwise rank values

    Clustering linkage is always computed from the distance matrix, even when
    the rank matrix is displayed.
    """
    os.makedirs(str(output_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Harmonize input order
    # ------------------------------------------------------------------
    cnames = list(cnames)

    dist_df = dist_df.loc[cnames, cnames].copy()
    rank_df = rank_df.loc[cnames, cnames].copy()

    if len(family_labels) != len(cnames):
        raise ValueError(
            "family_labels length ({}) does not match cnames length ({})".format(
                len(family_labels), len(cnames)
            )
        )

    family_labels = np.array(family_labels)

    # ------------------------------------------------------------------
    # Auto style
    # ------------------------------------------------------------------
    if ann_size is None or tick_size is None or rotation is None:
        auto_ann_size, auto_tick_size, auto_rotation = auto_heatmap_style(len(cnames))
        ann_size = auto_ann_size if ann_size is None else ann_size
        tick_size = auto_tick_size if tick_size is None else tick_size
        rotation = auto_rotation if rotation is None else rotation

    # ------------------------------------------------------------------
    # Select matrix to show
    # ------------------------------------------------------------------
    if source_type == "distance":
        fmt = ".2g"
        center = 0.62
        vmax = None
        df = dist_df
        title = "{} pairwise distance".format(gene_name)
    elif source_type == "rank":
        fmt = "d"
        center = None
        vmax = 100
        df = rank_df.astype(int)
        title = "{} pairwise rank".format(gene_name)
    else:
        raise ValueError("source_type must be either 'distance' or 'rank'")

    # ------------------------------------------------------------------
    # Clustering linkage is always calculated from distance matrix
    #
    # The diagonal is zeroed on an explicitly owned array. The notebook wrote
    # np.fill_diagonal(dist_for_linkage.values, 0), which mutates the frame in
    # place; under pandas >= 3.0 DataFrame.values is a read-only view
    # (copy-on-write is mandatory) and that raises
    # "ValueError: underlying array is read-only". Same values either way.
    # ------------------------------------------------------------------
    dist_for_linkage = dist_df.copy()
    dist_for_linkage_array = dist_for_linkage.to_numpy(copy=True)
    np.fill_diagonal(dist_for_linkage_array, 0)

    linkage = hc.linkage(
        sp.distance.squareform(dist_for_linkage_array, checks=False),
        method=linkage_method,
    )

    # ------------------------------------------------------------------
    # Plot clustering heatmap
    # ------------------------------------------------------------------
    cg = sns.clustermap(
        df,
        figsize=fig_size,
        annot=display_number,
        fmt=fmt,
        center=center,
        annot_kws={"size": ann_size},
        cmap="Blues_r",
        vmax=vmax,
        cbar_kws={"extend": "max"},
        yticklabels=cnames,
        xticklabels=cnames,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_linkage=linkage if row_cluster else None,
        col_linkage=linkage if col_cluster else None,
    )

    plt.suptitle(title, fontsize=title_size, y=1.02)

    ax = cg.ax_heatmap
    ax.figure.axes[-1].tick_params(labelsize=20)

    # ------------------------------------------------------------------
    # Optional red boxes for cross-family matches
    # ------------------------------------------------------------------
    diff_matrix = np.array([(i != family_labels) for i in family_labels])

    sorted_diff_matrix = pd.DataFrame(
        diff_matrix,
        index=dist_df.index,
        columns=dist_df.columns,
    )

    match_matrix = (
        (dist_df <= threshold)
        & sorted_diff_matrix
        & (rank_df <= match_rank)
    )

    if display_match_box:
        # Need to account for clustering-reordered heatmap positions
        reordered_y = [i.get_text() for i in ax.get_yticklabels()]
        reordered_x = [i.get_text() for i in ax.get_xticklabels()]

        reordered_match_matrix = match_matrix.loc[reordered_y, reordered_x]

        for i in range(reordered_match_matrix.shape[0]):
            for j in range(reordered_match_matrix.shape[1]):
                if reordered_match_matrix.iloc[i, j]:
                    ax.add_patch(
                        Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=3)
                    )

    # ------------------------------------------------------------------
    # Optional patient crop images on both axes
    # ------------------------------------------------------------------
    labels = [i.get_text() for i in ax.get_ymajorticklabels()]

    if input_crops_path:
        ax_pos = ax.get_position()
        num_of_images = len(family_labels)

        if num_of_images < 4:
            x_offset = 1.14
            y_offset = -0.10
        elif num_of_images < 10:
            x_offset = 1.16
            y_offset = -0.08
        elif num_of_images >= 20:
            x_offset = 1.09
            y_offset = -0.07
        else:
            x_offset = 1.11
            y_offset = -0.07

        if default_x_offset is not None:
            x_offset = default_x_offset
        if default_y_offset is not None:
            y_offset = default_y_offset

        x_size = (ax_pos.y1 - ax_pos.y0) / num_of_images
        size = x_size * fig_size[0] * 0.8

        x_pos = np.linspace(0, 1, 2 * num_of_images + 1)[np.arange(1, 2 * num_of_images, 2)]
        y_pos = np.linspace(1, 0, 2 * num_of_images + 1)[np.arange(1, 2 * num_of_images, 2)]

        for which_axis in ["x", "y"]:
            for idx, label in enumerate(labels):
                if "x" in which_axis:
                    pos = x_pos[idx]
                    y = y_offset - x_size * 1.1
                    anchor, loc = (pos, y), 8
                else:
                    pos = y_pos[idx]
                    x = x_offset + x_size * 1.4
                    anchor, loc = (x, pos), 7

                _ax = inset_axes(
                    ax,
                    width=size,
                    height=size,
                    bbox_transform=ax.transAxes,
                    bbox_to_anchor=anchor,
                    loc=loc,
                )
                _ax.axison = False

                image_label = label
                if map_dict and image_label in map_dict:
                    image_label = map_dict[image_label]

                img = get_image_by_name(image_label, input_crops_path)
                _ax.imshow(img, cmap="gray")

        ax.xaxis.set_label_coords(xaxis_ticks_x_offset, y - xaxis_ticks_y_offset)
        ax.yaxis.set_label_coords(x + 0.01, 0.5)

    # ------------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------------
    ax.set_ylabel("Gallery images", fontsize=label_size)
    ax.set_xlabel("Test images", fontsize=label_size)

    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        fontsize=tick_size,
        rotation_mode=rotation_mode,
        ha=ha,
    )
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_size)

    output_figure = os.path.join(
        str(output_path),
        "{}_validation_pairwise_{}{}.{}".format(
            gene_name, source_type, file_suffix, file_format
        ),
    )

    plt.savefig(output_figure, format=file_format, bbox_inches="tight")
    plt.close()

    print("Saved heatmap: {}".format(output_figure))

    return output_figure


def plot_pairwise_heatmap(distance_df, rank_df, output_dir, config):
    """The heatmap call site from notebook cell 16.

    Everything here mirrors that cell literally, including the hardcoded values
    that override or bypass config keys. See the module docstring.
    """
    IMAGE_TYPE = "svg"
    type_cluster = "single"

    # Use all images in the current pairwise matrix
    rename_image_ids = distance_df.index.astype(str).tolist()

    # Make sure distance and rank matrices have the same order
    df = distance_df.loc[rename_image_ids, rename_image_ids]
    target_ranks_df = rank_df.loc[rename_image_ids, rename_image_ids]

    ann_size, tick_size, rotation = auto_heatmap_style(len(rename_image_ids))

    # Match old notebook style
    rotation = "vertical"

    return plot_clustering_heatmap(
        dist_df=df,
        rank_df=target_ranks_df,
        cnames=rename_image_ids,
        family_labels=rename_image_ids,
        gene_name=config.get("target_name", "cohort"),
        output_path=output_dir,
        file_format=IMAGE_TYPE,
        ann_size=ann_size,
        tick_size=tick_size,
        rotation=rotation,
        threshold=config.get("pairwise_match_threshold", 0.748),
        match_rank=config.get("pairwise_match_rank", 300),
        display_match_box=config.get("display_match_box", False),
        source_type=config.get("heatmap_source", "rank"),
        map_dict=None,
        input_crops_path=config.get("input_crops_path", None),
        linkage_method=type_cluster,
        row_cluster=True,
        col_cluster=True,
        file_suffix="_{}".format(type_cluster),
        default_x_offset=config.get("default_x_offset", 1.16),
        default_y_offset=config.get("default_y_offset", -0.15),
    )


# ----------------------------------------------------------------------
# Random vs target plots (notebook cells 20 and 21)
# ----------------------------------------------------------------------

def get_random_and_target_distribution(distribution_df, random_label="random"):
    """Keep the random distribution plus every non-random (target) distribution."""
    labels = list(distribution_df["distribution"].dropna().unique())
    target_labels = [x for x in labels if x != random_label]

    if len(target_labels) == 0:
        raise ValueError("No target distribution found. Only random exists.")

    keep_labels = [random_label] + target_labels

    plot_df = distribution_df[distribution_df["distribution"].isin(keep_labels)].copy()

    plot_df["distribution"] = pd.Categorical(
        plot_df["distribution"],
        categories=keep_labels,
        ordered=True,
    )

    return plot_df, target_labels


def plot_random_vs_target_boxplot(distribution_df, output_dir, cohort_name,
                                  value_col="mean_pw_dist", random_label="random"):
    """Box plot: random vs target. Writes .jpeg, .png and .svg."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df, target_labels = get_random_and_target_distribution(
        distribution_df, random_label=random_label
    )

    color_palette = {
        random_label: (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
    }

    for target_label in target_labels:
        color_palette[target_label] = (
            0.9949711649365629,
            0.5974778931180315,
            0.15949250288350636,
        )

    plt.style.use("classic")
    fig, ax = plt.subplots(
        figsize=(max(10, 3.5 * len(plot_df["distribution"].cat.categories)), 10)
    )

    sns.boxplot(
        data=plot_df,
        x="distribution",
        y=value_col,
        order=list(plot_df["distribution"].cat.categories),
        palette=color_palette,
        width=0.55,
        showfliers=False,
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x="distribution",
        y=value_col,
        order=list(plot_df["distribution"].cat.categories),
        color="black",
        size=4,
        alpha=0.25,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Mean pairwise distance", fontsize=36)
    ax.tick_params(axis="x", labelsize=32, rotation=30)
    ax.tick_params(axis="y", labelsize=32)
    ax.grid(True)

    ax.set_title("Cohort mean pairwise distance vs random", fontsize=38)

    plt.tight_layout()

    filename_prefix = output_dir / "{}_random_vs_target_boxplot".format(cohort_name)

    fig.savefig("{}.jpeg".format(filename_prefix), dpi=300)
    fig.savefig("{}.png".format(filename_prefix), dpi=300)
    fig.savefig("{}.svg".format(filename_prefix), dpi=300)

    plt.show()
    plt.close(fig)

    print("Saved box plot: {}.png".format(filename_prefix))


def plot_random_vs_target_kde(distribution_df, output_dir, cohort_name,
                              value_col="mean_pw_dist", random_label="random",
                              xlim=(0.4, 1.3)):
    """KDE density plot: random vs target.

    Also writes the percentile report TSV, as the notebook does inside this
    function. Returns that report as a DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df, target_labels = get_random_and_target_distribution(
        distribution_df, random_label=random_label
    )

    if len(target_labels) > 1:
        print("Multiple target groups detected:")
        print(target_labels)
        print("KDE will overlay all target groups against random.")

    random_vals = plot_df.loc[
        plot_df["distribution"] == random_label, value_col
    ].dropna().to_numpy()

    if len(random_vals) == 0:
        raise ValueError("No random values found.")

    color_palette = {
        "random_fill": (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
        "random_line": (0.84, 0.15, 0.16),
    }

    target_color = (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)

    plt.style.use("classic")
    plt.figure(figsize=(24, 16))

    # Random KDE
    sns.kdeplot(
        data=plot_df[plot_df["distribution"] == random_label],
        x=value_col,
        fill=True,
        common_norm=False,
        color=color_palette["random_fill"],
        alpha=0.55,
        zorder=1,
    )

    # Random percentiles
    p05 = np.quantile(random_vals, 0.05)
    p50 = np.quantile(random_vals, 0.50)

    plt.axvline(p05, color=color_palette["random_line"], linestyle="--", linewidth=3, zorder=4)
    plt.axvline(p50, color=color_palette["random_line"], linestyle="-", linewidth=3, zorder=4)

    ax = plt.gca()

    def label_right_of_line(x, text, color="black"):
        ax.annotate(
            text,
            xy=(x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(6, -4),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=32,
            color=color,
            clip_on=False,
        )

    label_right_of_line(p05, "5th", color_palette["random_line"])
    label_right_of_line(p50, "50th", color_palette["random_line"])

    handles = [
        Line2D([0], [0], color=color_palette["random_fill"], lw=10, label=random_label),
        Line2D([0], [0], color=color_palette["random_line"], lw=3, linestyle="--",
               label="random 5th percentile"),
        Line2D([0], [0], color=color_palette["random_line"], lw=3, linestyle="-",
               label="random 50th percentile"),
    ]

    r_sorted = np.sort(random_vals)

    percentile_rows = []

    # Target KDE(s)
    for target_label in target_labels:
        target_vals = plot_df.loc[
            plot_df["distribution"] == target_label, value_col
        ].dropna().to_numpy()

        if len(target_vals) == 0:
            continue

        sns.kdeplot(
            data=plot_df[plot_df["distribution"] == target_label],
            x=value_col,
            fill=True,
            common_norm=False,
            color=target_color,
            alpha=0.7,
            zorder=2,
        )

        target_median = float(np.median(target_vals))

        plt.axvline(target_median, color=target_color, linestyle="-", linewidth=4, zorder=5)

        label_right_of_line(target_median, "{} median".format(target_label), target_color)

        target_percentiles = (
            100.0 * np.searchsorted(r_sorted, target_vals, side="right") / r_sorted.size
        )
        target_similarity_percentiles = 100.0 - target_percentiles

        median_percentile_in_random = (
            100.0 * np.searchsorted(r_sorted, target_median, side="right") / r_sorted.size
        )
        median_similarity_percentile = 100.0 - median_percentile_in_random

        percentile_rows.append({
            "target": target_label,
            "n_random": len(random_vals),
            "n_target": len(target_vals),
            "random_5th": p05,
            "random_median": p50,
            "target_mean": float(np.mean(target_vals)),
            "target_median": target_median,
            "target_median_percentile_in_random_lower_is_more_similar": median_percentile_in_random,
            "target_median_similarity_percentile_higher_is_more_similar": median_similarity_percentile,
            "target_sample_percentile_median_lower_is_more_similar": float(np.median(target_percentiles)),
            "target_sample_similarity_percentile_median_higher_is_more_similar": float(np.median(target_similarity_percentiles)),
        })

        print("{} percentiles within RANDOM empirical distribution:".format(target_label))
        print("  n_random = {}, n_{} = {}".format(len(random_vals), target_label, len(target_vals)))
        print("  random 5th = {:.4f}, 50th = {:.4f}".format(p05, p50))
        print("  {} mean = {:.4f}, median = {:.4f}".format(
            target_label, np.mean(target_vals), target_median))
        print("  {} median percentile in random (lower distance = more similar) = {:.2f}%".format(
            target_label, median_percentile_in_random))
        print("  {} median similarity percentile (higher = more similar) = {:.2f}%".format(
            target_label, median_similarity_percentile))

        handles.extend([
            Line2D([0], [0], color=target_color, lw=10, label=target_label),
            Line2D([0], [0], color=target_color, lw=4, linestyle="-",
                   label="{} median".format(target_label)),
        ])

    # Axes
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)

    plt.xlabel("Mean pairwise distance", fontsize=50)
    plt.ylabel("Density", fontsize=50)
    plt.grid(True)

    plt.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        fontsize=36,
        title_fontsize=40,
    )

    plt.tight_layout()

    filename_prefix = output_dir / "{}_random_vs_target_kde".format(cohort_name)

    plt.savefig("{}.jpeg".format(filename_prefix), dpi=300)
    plt.savefig("{}.png".format(filename_prefix), dpi=300)
    plt.savefig("{}.svg".format(filename_prefix), dpi=300)

    plt.show()
    plt.close()

    print("Saved KDE plot: {}.png".format(filename_prefix))

    percentile_report_df = pd.DataFrame(percentile_rows)
    percentile_report_path = output_dir / "{}_random_vs_target_kde_percentile_report.tsv".format(
        cohort_name
    )
    percentile_report_df.to_csv(percentile_report_path, sep="\t", index=False)

    print("Saved KDE percentile report: {}".format(percentile_report_path))

    return percentile_report_df


# ----------------------------------------------------------------------
# Cohort comparison box plot (notebook cell 28)
# ----------------------------------------------------------------------

def plot_cohort_comparison_boxplot(same_diff_df, comparison_distributions, summary_df,
                                   output_dir, base_key, threshold, top_k=8):
    """Horizontal box plot of the cohort comparisons vs the same/different controls."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distributions_plot = pd.concat(
        [same_diff_df, comparison_distributions],
        ignore_index=True,
    )

    if len(summary_df) > 0:
        show_names = (
            summary_df.head(top_k)["comparison"].tolist()
            + summary_df.tail(top_k)["comparison"].tolist()
        )
        show_names = list(dict.fromkeys(show_names))
    else:
        show_names = []

    plot_names = ["Different syndromes", "Same syndromes"] + show_names

    plot_df = distributions_plot[
        distributions_plot["distribution"].isin(plot_names)
    ].copy()

    comparison_order = (
        summary_df[summary_df["comparison"].isin(show_names)]
        .sort_values("mean_pw_distance_all_pairs")["comparison"]
        .tolist()
    )

    plot_order = ["Different syndromes", "Same syndromes"] + comparison_order

    plt.style.use("classic")
    plt.figure(figsize=(18, max(8, 0.7 * len(plot_order))))

    palette = {
        "Different syndromes": (0.894, 0.102, 0.110),
        "Same syndromes": (0.216, 0.494, 0.722),
    }

    default_comp_color = (0.9949711649365629, 0.5974778931180315, 0.15949250288350636)

    row_colors = [palette.get(name, default_comp_color) for name in plot_order]

    sns.boxplot(
        data=plot_df,
        y="distribution",
        x="mean_pw_dist",
        order=plot_order,
        palette=row_colors,
        orient="h",
        showfliers=False,
        width=0.6,
    )

    plt.axvline(threshold, color="black", linewidth=1.5, linestyle="--")

    plt.text(threshold + 0.002, -0.4, "Threshold", fontsize=18, ha="left", va="center")

    plt.xlabel("Mean cosine distance", fontsize=24)
    plt.ylabel("")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    prefix = output_dir / "{}_cohort_comparison_boxplot".format(base_key)

    plt.savefig("{}.png".format(prefix), dpi=300, bbox_inches="tight")
    plt.savefig("{}.svg".format(prefix), dpi=300, bbox_inches="tight")
    plt.savefig("{}.jpeg".format(prefix), dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    print("Saved figure: {}.[png/svg/jpeg]".format(prefix))
