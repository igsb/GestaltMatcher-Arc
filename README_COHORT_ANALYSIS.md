# Cohort analysis CLI (`cohort-analysis-refactor` branch)

This document describes the cohort-analysis workflow that lives on the
`cohort-analysis-refactor` branch: what it is, how to run it, what it writes,
and what is still open.

It is scoped to `run_cohort_analysis.py` + `lib/cohort_analysis/`. Nothing else
in the repository (training, prediction, face alignment, the REST service) is
affected by this branch.

---

## 1. Purpose of this branch

The cohort analysis was originally an exploratory Jupyter notebook,
`notebooks/GestaltMatcher_cohort_analysis_template.ipynb`. It took a hand-edited
`CONFIG` dict and produced, for one "target" cohort:

- a target-vs-target pairwise cosine-distance matrix,
- a target-vs-gallery rank matrix,
- a clustering heatmap,
- a random-percentile validation (is the cohort tighter than random?),
- a cross-cohort comparison + PPV report (does cohort A look *different* from
  cohort B?).

This branch turns that notebook into a **reproducible command-line pipeline**
driven by YAML config files, and then extends it to a **multi-cohort** mode that
can plan and run many pairwise / combined / comparison analyses from a single
config.

### Notebook vs. CLI

| | Notebook | CLI (`run_cohort_analysis.py`) |
|---|---|---|
| Config | hand-edited `CONFIG` dict in a cell | YAML file passed with `--config` |
| Reproducibility | manual | `run_config.json` + `run_metadata.json` written every run |
| Scope | one target cohort | one target cohort **or** a multi-cohort plan |
| Paths | relative to the notebook's working dir | resolved against the repo root |
| Status | kept as the exploratory front-end | the pipeline of record |

The notebook is **not deleted**. It stays as the exploratory reference. The CLI
was validated to reproduce the notebook's tables byte-for-byte under the
original Python 3.8 environment (see *Validation status*).

---

## 2. Current status of the implementation

| Capability | Status |
|---|---|
| Single-analysis CLI (notebook parity) | done, validated |
| Python 3.13 compatibility | done, tested end to end |
| `run_metadata.json` provenance / config hash | done |
| Multi-cohort **dry-run planning** (`--dry-run-plan`) | done |
| Multi-cohort **pairwise** execution | done |
| Multi-cohort **combined** (pooled) execution | done |
| Multi-cohort **cohort-vs-cohort comparisons** execution | done |
| Random-percentile validation in multi-cohort mode | **not implemented** (single-analysis mode only) |
| `Dockerfile` for the cohort-analysis CLI | **not added** (the existing `Dockerfile` is untouched) |

---

## 3. Environments

Two environments, both supported. Neither replaces the other.

### Python 3.8 baseline — `environment_gestalt_no_rpy2.yml` (conda env `gestalt`, Python 3.8.19)

The **original reproducibility baseline**. The pipeline was validated
byte-for-byte against the notebook under this environment, and it remains the
reference. The rest of the repo (training, prediction, alignment) also needs it.
`environment_gestalt_no_rpy2.yml` is the file tracked on this branch; an
identical `environment_gestalt.yml` may also be present locally.

```bash
conda env create -f environment_gestalt_no_rpy2.yml
conda activate gestalt
```

### Python 3.13 CLI environment — `environment_py313.yml` (conda `gestalt-py313`)

An **addition** for running the refactored CLI on a modern interpreter. Scope:
`run_cohort_analysis.py` + `lib/cohort_analysis/` only. It deliberately excludes
`torch`, `opencv`, `rpy2`, `tensorboard`, jupyter — the CLI never imports them.

```bash
conda env create -f environment_py313.yml
conda activate gestalt-py313
```

If conda solving is unavailable, the equivalent pip set is:

```bash
python -m venv venv
venv/Scripts/pip install numpy pandas scipy scikit-learn matplotlib seaborn pillow pyyaml
```

Python 3.13 has been tested end to end: all outputs are produced, and TSV data
is either byte-identical to the 3.8 baseline or differs only by
floating-point noise that does not change ranks, PPV, thresholds, or
`evidence_different`. Python 3.14 is **not** claimed (never exercised).

---

## 4. Expected input files

The configs point at files under `analysis_metadata/` (which is git-ignored — the
data is obtained separately, not committed). A full run needs:

| Config key | File (example) | Notes |
|---|---|---|
| `gallery_embedding_file` | `analysis_metadata/all_gmdb_encodings_v1.1.0_23052026.pkl` | ~1.9 GB GMDB gallery pickle. Read in full, then filtered. Needed for the **rank matrix** (pairwise + combined). `--dry-run-plan` never loads it; the comparison computation does not use it, but a multi-cohort **execution** loads it once up front regardless of what the plan contains. |
| `gallery_metadata_files` | 3 CSVs under `analysis_metadata/gmdb_metadata/` | Gallery image-ID lists. |
| `photo_metadata_file` | `analysis_metadata/patient_metadata_23052026.tsv` | image_id → patient_id lookup. |
| cohort `embedding_file` (per cohort) | `analysis_metadata/PRMT7_embeddings_04102025_v1.1.0.pkl`, `analysis_metadata/CTNND2_gmdb_encodings_v1.1.0.pkl` | Per-cohort embedding pickles. |
| `same_different_distribution_file` | `analysis_metadata/roc_same_different_distribution.csv` | Same/different-syndrome reference distributions for PPV. |
| `random_distribution_file` | `analysis_metadata/roc_same_random_distribution.csv` | Single-analysis mode only (random-percentile validation). |

Cost of a full run with the rank matrix: several GB of RAM and roughly a few
minutes just to read the gallery pickle.

---

## 5. How to run

All commands are run from the repository root, with one of the environments
above active.

### 5.1 Single-analysis configs (notebook parity)

These configs have top-level `target_name` / `run_date` / `active_cohorts` /
`cross_cohort_comparison`. Behaviour is unchanged from the notebook port.

```bash
# Full run (tables + all figures)
python run_cohort_analysis.py --config configs/cohort_analysis/prmt7_ctnnd2.yaml

# Tables only, no figures
python run_cohort_analysis.py --config configs/cohort_analysis/prmt7_ctnnd2.yaml --skip-plots
```

Output goes to `<output_root>/<target_name>_<run_date>/`, e.g.
`analysis_output/PRMT7_2026_07_14/`, and contains `run_config.json`,
`run_metadata.json`, `target_cohort_metadata.tsv`,
`pairwise_distance_matrix.tsv`, `pairwise_rank_matrix.tsv`,
`target_random_percentile_summary.tsv`,
`target_random_percentile_distributions.tsv`,
`<base>_cohort_comparison_summary.tsv`,
`<base>_cohort_comparison_distributions.tsv`. Without `--skip-plots` it also
writes the heatmap, the random-vs-target box/KDE plots, the cohort-comparison
box plot, and `<target>_random_vs_target_kde_percentile_report.tsv` (that
report is produced by the plotting step, so `--skip-plots` omits it too).

The second config, `configs/cohort_analysis/lins1_ctnnd2.yaml`, is a verbatim
port that exists to reproduce an older `analysis_output/LINS1_2026_07_01` run;
see the caveats section.

### 5.2 Multi-cohort config — dry-run planning

A config with a top-level `analysis_plan` key is a **multi-cohort** config. It is
auto-detected; there is no separate flag to opt in.

`--dry-run-plan` expands the plan and prints it. It resolves no paths, loads no
embeddings, and writes nothing.

```bash
python run_cohort_analysis.py --config configs/cohort_analysis/multi_cohort_example.yaml --dry-run-plan
```

This prints: the available cohorts and their embedding files, the pairwise
analyses that would run, the combined groups, the expanded directed comparison
pairs, the number of directed comparisons, and any warnings (LINS1 / shared
embedding files).

### 5.3 Multi-cohort config — full execution

Running a multi-cohort config **without** `--dry-run-plan` executes it:
`analysis_plan.pairwise`, then `analysis_plan.combined`, then
`analysis_plan.comparisons`. Random-percentile validation is **not** run in
multi-cohort mode.

```bash
python run_cohort_analysis.py --config configs/cohort_analysis/multi_cohort_example.yaml --skip-plots --output-root analysis_output_multi_test
```

`--skip-plots` writes all tables and metadata but no figures — the fastest way
to get a complete, reviewable result.

`--output-root` overrides `output_root` from the config (relative paths are
taken relative to the current working directory).

---

## 6. Output folder structure (multi-cohort)

```
analysis_output_multi_test/
  run_<timestamp>_<config_hash>/
    run_config.json
    run_metadata.json
    analysis_plan_resolved.json
    pairwise/
      PRMT7/
      LINS1/
      CTNND2/
    combined/
      PRMT7_LINS1/
    comparisons/
      PRMT7_vs_LINS1/
      PRMT7_vs_CTNND2/
      LINS1_vs_PRMT7/
      LINS1_vs_CTNND2/
      CTNND2_vs_PRMT7/
      CTNND2_vs_LINS1/
```

- The **run folder** is `run_<timestamp>_<config_hash>`, where `<timestamp>` is
  local `YYYYmmdd_HHMMSS` and `<config_hash>` is the first 12 hex characters of
  the SHA-256 of the original config file. Every run gets its own folder.
- `<timestamp>` and the exact folder name will differ between runs; the
  structure below it is stable.

### `pairwise/<COHORT>/`

Within-cohort analysis for one cohort, using the same functions as the
single-analysis pipeline. Contains:

- `target_cohort_metadata.tsv` — one row per image (image_id, cohort,
  subject_id, family_id).
- `pairwise_distance_matrix.tsv` — target-vs-target mean cosine distance
  (image IDs on both axes).
- `pairwise_rank_matrix.tsv` — each target image's rank among the combined
  gallery + target pool.
- `analysis_metadata.json` — see section 7.
- `<COHORT>_validation_pairwise_rank_single.svg` — heatmap, unless
  `--skip-plots`.

### `combined/<GROUP>/`

Same three matrices, but computed over the **pooled** images of several member
cohorts (defined by `analysis_plan.combined`). Same file names as `pairwise/`,
plus `analysis_metadata.json`. A `<GROUP>_validation_pairwise_rank_single.svg`
heatmap is written unless `--skip-plots`; it is **additionally** skipped when the
group contains **duplicate image IDs** (see caveats), since a lookup on repeated
labels would distort the figure. The distance/rank matrices are always written,
unmodified.

### `comparisons/<BASE>_vs_<COMPARISON>/`

One directed cohort-vs-cohort comparison (`BASE` sampled against `COMPARISON`),
using the existing cross-cohort comparison + PPV code. Contains:

- `cohort_comparison_summary.tsv` — one row: mean pairwise distance, sampled
  distance quantiles, `threshold_c`, `prop_sampled_above_c`,
  `evidence_different`, PPV quantiles, image/patient counts.
- `cohort_comparison_distributions.tsv` — the sampled mean distances
  (one row per sample).
- `analysis_metadata.json` — see section 7. `status` is `ok` or `skipped`.
- `<BASE>_vs_<COMPARISON>_cohort_comparison_boxplot.{png,svg,jpeg}` — unless
  `--skip-plots`.

---

## 7. Metadata / provenance files

### `run_config.json`

The **effective config after path resolution** — i.e. exactly what the run used,
with input paths made absolute. This is the notebook's old "dump the CONFIG
dict" behaviour, kept unchanged. It does not affect any computed result.

### `run_metadata.json`

Provenance for the whole run. Written once at the top of the run folder. Fields:

- `created_at` — ISO-8601 UTC timestamp.
- `config_path`, `config_path_abs` — the config as passed / resolved.
- `config_sha256` — SHA-256 of the **original** (unresolved) config file bytes.
  Its first 12 hex chars are the `<config_hash>` in the run folder name.
- `resolved_config_sha256` — SHA-256 of the `run_config.json` written this run.
- `git` — `commit`, `branch`, `dirty` (tracked changes only), or nulls if git
  is unavailable.
- `python` — version, implementation, executable.
- `platform` — OS / release / version / machine / node.
- `packages` — versions of numpy, pandas, scipy, sklearn, matplotlib, seaborn,
  Pillow, pyyaml (null if not importable).
- `output_dir`.
- `target_name`, `base_cohort`, `comparison_cohorts` — populated in
  single-analysis mode; `null` in multi-cohort mode (per-analysis identity lives
  in each folder's `analysis_metadata.json` instead).

Every collector degrades gracefully: a missing git binary or absent package
yields `null`, never an error.

### `analysis_plan_resolved.json` (multi-cohort only)

The **expanded plan plus what actually ran**. Written once early (crash
resilience) and rewritten at the end with `status: "complete"`. Fields:

- `available_cohorts` — every cohort defined in the config, with its embedding
  file and any warning.
- `executed.pairwise` — mode + the list of cohort names that ran.
- `executed.combined` — the groups that ran (name + members).
- `executed.comparisons` — mode, `n_planned`, `n_executed`, `n_skipped`, and a
  `pairs` list with per-pair `status` (`ok` / `skipped`, with `reason`).
- `not_executed_yet` — `{}`; every plan section (pairwise, combined,
  comparisons) is executed.
- `warnings` — the plan-level warnings.

### `analysis_metadata.json` (one per analysis folder)

A small record inside each `pairwise/`, `combined/`, and `comparisons/` folder.

Common fields: `analysis_type`, `skip_plots`, `warnings`, `outputs` (the files
in the folder), and image / patient counts.

- **pairwise** (`analysis_type: pairwise`): `cohort`, `label`,
  `embedding_file` (resolved) + `embedding_file_config` (as written),
  `img_name_parser`, `n_images`, `n_patients`, `distance_metric`, `n_tta`,
  `heatmap`.
- **combined** (`analysis_type: combined_pairwise`): `group`, `member_cohorts`,
  `embedding_files` (per member), `n_images` (unique), `n_images_in_matrix`
  (rows actually written), `n_unique_image_ids`, `n_duplicate_image_ids`,
  `has_duplicate_image_ids`, `per_member_counts`, `matrix_shape`,
  `heatmap_skipped_reason`.
- **comparison** (`analysis_type: cohort_comparison`): `base_cohort`,
  `comparison_cohort`, `comparison_label`, `status` (`ok` / `skipped` with
  `reason`), `embedding_files`, `shares_embedding_file`, `base_n_images` /
  `base_n_patients` / `comparison_n_images` / `comparison_n_patients`,
  `settings` (`threshold_c`, `pretest_probability`, `n_samples`,
  `min_base_images`, `min_comparison_images`, `top_k`, `random_seed`,
  `id_column`), `ppv_summary` (`mean_pw_distance_all_pairs`,
  `prop_sampled_above_c`, `evidence_different`, PPV quantiles), `plot`.

### Config hash / reproducibility tracking

To reproduce a run you need: the config file whose SHA-256 matches
`run_metadata.json.config_sha256`, the same input data under
`analysis_metadata/`, and ideally the same `git.commit` and `packages`. The run
folder name embeds the config hash so two runs of two different configs never
collide, and re-running the same config is obvious at a glance.

---

## 8. `analysis_plan` schema (multi-cohort configs)

```yaml
cohorts:
  <NAME>:
    embedding_file: analysis_metadata/<file>.pkl
    img_name_parser: drop_last_token      # none | first_token | drop_last_token
    label: <NAME>
    exclude_image_ids: []
    warning: <optional free-text caveat, surfaced in console + metadata>

analysis_plan:
  pairwise:
    mode: all            # all | none | explicit (+ cohorts: [...])
  combined:
    - name: <GROUP>
      cohorts: [<NAME>, <NAME>, ...]
  comparisons:
    mode: all_directed    # all_directed | all_undirected | none | explicit
```

Plus the shared analysis settings the execution path needs (mirrored from
`prmt7_ctnnd2.yaml`): `gallery_embedding_file`, `gallery_metadata_files`,
`photo_metadata_file`, `photo_metadata_sep`, `frontal_face_only`,
`distance_metric`, `n_tta`, `random_seed`, `id_column`,
`same_different_distribution_file`, `ppv_threshold`, and a
`cross_cohort_comparison:` block of tunables (`n_samples`, `min_base_images`,
`min_comparison_images`, `threshold`, `pretest_probability`, `top_k`). These are
ignored by `--dry-run-plan`.

### Comparison modes

| Mode | Meaning |
|---|---|
| `all_directed` | every **ordered** pair `(a, b)` with `a != b` |
| `all_undirected` | every **unordered** pair `{a, b}` |
| `explicit` | only the pairs listed under `pairs:` (each `[base, comparison]` or `{base: X, comparison: Y}`) |
| `none` | no comparisons |

For **n** cohorts:

- `all_directed` → **n × (n − 1)** ordered comparisons
  (e.g. 3 cohorts → 6: `A_vs_B`, `A_vs_C`, `B_vs_A`, `B_vs_C`, `C_vs_A`, `C_vs_B`).
- `all_undirected` → **n × (n − 1) / 2** unordered comparisons
  (e.g. 3 cohorts → 3: `A_vs_B`, `A_vs_C`, `B_vs_C`).

Direction matters numerically: the comparison samples sub-blocks of the
`base × comparison` distance matrix, so `A_vs_B` and `B_vs_A` are related but
not identical.

`pairwise.mode` works the same way: `all` = one folder per cohort, `explicit` =
only the cohorts listed under `cohorts:`, `none` = no pairwise.

### Example: a selected / explicit config

```yaml
analysis_plan:
  pairwise:
    mode: explicit
    cohorts: [PRMT7, CTNND2]
  combined:
    - name: PRMT7_CTNND2
      cohorts: [PRMT7, CTNND2]
  comparisons:
    mode: explicit
    pairs:
      - [PRMT7, CTNND2]
```

This runs pairwise for PRMT7 and CTNND2, one combined `PRMT7_CTNND2` group, and
exactly one comparison (`PRMT7_vs_CTNND2`). The snippet shows only the
`analysis_plan` block — a working config also needs the `cohorts:` block and the
shared analysis settings listed above.

---

## 9. Known warnings and caveats

### LINS1 currently uses the PRMT7 embedding file

In `configs/cohort_analysis/multi_cohort_example.yaml` (and
`lins1_ctnnd2.yaml`), the `LINS1` cohort's `embedding_file` points at
`PRMT7_embeddings_04102025_v1.1.0.pkl`. This is carried over from the notebook,
which flagged it "TODO: check this filename". **It is not a validated LINS1
embedding.**

Consequences, all surfaced loudly in the console and in
`analysis_metadata.json`:

- **Pairwise `LINS1/`** is numerically identical to `PRMT7/` (same 8 images).
- **Combined `PRMT7_LINS1/`** concatenates the same 8 images twice → 16 rows,
  8 unique image IDs. The distance / rank matrices are written **16×16 with
  repeated index labels, not de-duplicated**. `has_duplicate_image_ids: true`,
  and the heatmap is skipped for that group.
- **Any comparison whose base and comparison resolve to the same embedding
  file** (`PRMT7_vs_LINS1`, `LINS1_vs_PRMT7`) is effectively a cohort against a
  copy of itself: near-zero distances, `evidence_different: false`. These carry
  an explicit "NOT biologically meaningful" warning and
  `shares_embedding_file` is set.
- `LINS1_vs_CTNND2` / `CTNND2_vs_LINS1` numerically equal
  `PRMT7_vs_CTNND2` / `CTNND2_vs_PRMT7`.

**Do not biologically interpret any LINS1 result until a correct LINS1
embedding file is provided.** The tooling does not silently "fix" this — it runs
what it can and warns.

### Cohort image IDs do not map to patient metadata IDs

`build_target_df` falls back to using the image ID as the subject ID when an
image is not found in `photo_metadata_file`. For the current cohorts every image
falls back this way, so **`n_patients` equals `n_images`** in the metadata and
the per-patient logic is effectively a no-op. A real image → patient/subject
mapping file is needed before per-patient counts mean anything.

### Other

- `random_seed` is fixed (0) so sampling is deterministic across runs.
- Some config keys are dead / ignored (carried over from the notebook for
  parity) — see the comments in the YAML files.
- Tiny floating-point differences can appear between Python 3.8 and 3.13 for the
  distance matrices; they do not change ranks, PPV, thresholds, or
  `evidence_different`.

---

## 10. Validation status

- **Single-analysis behaviour preserved.** Running
  `configs/cohort_analysis/prmt7_ctnnd2.yaml --skip-plots` still produces the
  same 9 files in `<output_root>/PRMT7_2026_07_14/`.
- **PRMT7 single config TSV outputs are SHA-256 identical** to the previously
  validated reference for: `pairwise_distance_matrix.tsv`,
  `pairwise_rank_matrix.tsv`, `target_cohort_metadata.tsv`,
  `target_random_percentile_summary.tsv`,
  `target_random_percentile_distributions.tsv`,
  `PRMT7_cohort_comparison_summary.tsv`,
  `PRMT7_cohort_comparison_distributions.tsv`. `run_config.json` differs only in
  the `output_root` line.
- **Multi-cohort dry-run works** (`--dry-run-plan` prints the expanded plan,
  loads nothing, writes nothing).
- **Multi-cohort pairwise / combined / comparison execution works** end to end
  with `--skip-plots`: all folders and `analysis_metadata.json` files are
  produced, `analysis_plan_resolved.json` reaches `status: "complete"` with an
  empty `not_executed_yet`.
- **`PRMT7_vs_CTNND2` from the multi-cohort run matches the old single-analysis
  reference exactly**: `cohort_comparison_summary.tsv` is byte-identical and
  `cohort_comparison_distributions.tsv` is SHA-256 identical to the single
  pipeline's `PRMT7_cohort_comparison_*` outputs.
- Cross-check: per-cohort pairwise distance blocks in multi-cohort mode are
  exactly equal (max abs diff 0.0) to the corresponding sub-blocks of the
  single pipeline's pooled distance matrix.

---

## 11. Known open questions

- **A correct LINS1 embedding file is still needed.** Until then LINS1 results
  are not interpretable (see caveats).
- **A subject / image mapping file is still needed** so that `n_patients` is a
  real patient count rather than an image count.
- **Default comparison mode for production is undecided** — `all_directed`
  (keeps direction, `n(n−1)` folders) vs `all_undirected` (`n(n−1)/2` folders).
  The example config uses `all_directed`.
- **Random-percentile validation is not generalized** to multi-cohort mode. It
  currently runs only in single-analysis mode. Whether it should also run for
  each `pairwise/` and `combined/` result is open.
- **Docker support for the cohort-analysis CLI is not added.** The existing
  `Dockerfile` is untouched. If containerization is wanted for this workflow it
  should probably be a separate `Dockerfile.cohort-analysis` built on the
  Python 3.13 environment, not a change to the existing image.

---

## 12. Using this repository as a submodule

`GestaltMatcher-Arc` can be consumed as a git submodule of the main
GestaltMatcher repository. The cohort-analysis CLI is self-contained under
`run_cohort_analysis.py` + `lib/cohort_analysis/`, so from the parent repo:

```bash
# add once, on the desired branch
git submodule add -b cohort-analysis-refactor <GestaltMatcher-Arc-url> GestaltMatcher-Arc
git submodule update --init --recursive

# later, to pull the latest cohort-analysis work
git -C GestaltMatcher-Arc fetch
git -C GestaltMatcher-Arc checkout cohort-analysis-refactor
git -C GestaltMatcher-Arc pull
```

Run the CLI from **inside the submodule directory** so that relative paths
(`configs/…`, `analysis_metadata/…`) resolve, and with one of the environments
from section 3 active:

```bash
cd GestaltMatcher-Arc
python run_cohort_analysis.py --config configs/cohort_analysis/prmt7_ctnnd2.yaml --skip-plots
```

Input data under `analysis_metadata/` and the run outputs (`.json` / `.tsv` /
`.csv` / `.pkl` / image files, and the `analysis_output/` directory) are
git-ignored, so they are **not** carried by the submodule — they must be
provided in the submodule working tree. Pin the submodule to a specific commit
in the parent repo for reproducibility, and record which commit was used
alongside the `run_metadata.json` of any run you keep.
