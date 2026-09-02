"""Multi-cohort analysis-plan parsing and expansion (planning only).

Step 14 groundwork. This module reads a config, decides whether it is the
existing single-analysis schema or the new multi-cohort schema, and expands the
requested work into an explicit plan (which pairwise runs, which combined
groups, which directed comparison pairs).

Nothing here runs an analysis, loads an embedding pickle, or writes an output
file. It is pure config -> plan, used by ``run_cohort_analysis.py --dry-run-plan``.
Imports are stdlib only so the dry run stays cheap.

Single-analysis schema (unchanged, see configs/cohort_analysis/prmt7_ctnnd2.yaml):

    target_name, run_date, active_cohorts, cross_cohort_comparison, ...

Multi-cohort schema (new, see configs/cohort_analysis/multi_cohort_example.yaml):

    cohorts:
      <NAME>:
        embedding_file: ...
        img_name_parser: drop_last_token
        label: <NAME>
        exclude_image_ids: []
        warning: <optional free-text caveat, surfaced in the dry run>

    analysis_plan:
      pairwise:
        mode: all            # all | none | explicit (+ cohorts: [...])
      comparisons:
        mode: all_directed    # all_directed | all_undirected | none | explicit
      combined:
        - name: PRMT7_LINS1
          cohorts: [PRMT7, LINS1]

A config is treated as multi-cohort iff it has a top-level ``analysis_plan`` key.
"""

SINGLE_ANALYSIS = "single-analysis"
MULTI_COHORT = "multi-cohort"

_PAIRWISE_MODES = ("all", "none", "explicit")
_COMPARISON_MODES = ("all_directed", "all_undirected", "none", "explicit")


def detect_format(config):
    """Return SINGLE_ANALYSIS or MULTI_COHORT for a loaded config dict."""
    if not isinstance(config, dict):
        raise ValueError(
            "Config must be a YAML mapping, got {}".format(type(config).__name__)
        )
    if "analysis_plan" in config:
        return MULTI_COHORT
    return SINGLE_ANALYSIS


# ---------------------------------------------------------------------------
# Cohort listing
# ---------------------------------------------------------------------------
def _cohort_entries(cohorts_map):
    """Normalise the ``cohorts`` mapping into an ordered list of dicts."""
    if not isinstance(cohorts_map, dict) or not cohorts_map:
        raise ValueError("Config 'cohorts' must be a non-empty mapping.")

    entries = []
    for name, spec in cohorts_map.items():
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ValueError(
                "cohorts.{} must be a mapping, got {}".format(
                    name, type(spec).__name__
                )
            )
        entries.append(
            {
                "name": str(name),
                "embedding_file": spec.get("embedding_file"),
                "label": spec.get("label", str(name)),
                "img_name_parser": spec.get("img_name_parser", "drop_last_token"),
                "exclude_image_ids": list(spec.get("exclude_image_ids", []) or []),
                "warning": spec.get("warning"),
            }
        )
    return entries


def _validate_members(members, known_names, where):
    if not isinstance(members, (list, tuple)) or not members:
        raise ValueError("{} must be a non-empty list of cohort names.".format(where))
    unknown = [m for m in members if m not in known_names]
    if unknown:
        raise ValueError(
            "{} refers to undefined cohort(s): {} (known: {})".format(
                where, unknown, list(known_names)
            )
        )


# ---------------------------------------------------------------------------
# Plan-section expansion
# ---------------------------------------------------------------------------
def expand_pairwise(cohort_names, spec):
    """Return the list of cohort names to run a within-cohort pairwise analysis for."""
    if spec is None:
        return {"mode": "none", "cohorts": []}
    if not isinstance(spec, dict):
        raise ValueError("analysis_plan.pairwise must be a mapping.")

    mode = spec.get("mode", "explicit")
    if mode not in _PAIRWISE_MODES:
        raise ValueError(
            "analysis_plan.pairwise.mode must be one of {}, got {!r}".format(
                _PAIRWISE_MODES, mode
            )
        )

    if mode == "all":
        return {"mode": "all", "cohorts": list(cohort_names)}
    if mode == "none":
        return {"mode": "none", "cohorts": []}

    cohorts = spec.get("cohorts", [])
    _validate_members(cohorts, cohort_names, "analysis_plan.pairwise.cohorts")
    return {"mode": "explicit", "cohorts": list(cohorts)}


def expand_combined(cohort_names, spec):
    """Return the list of named combined (pooled) analyses."""
    if not spec:
        return []
    if not isinstance(spec, list):
        raise ValueError("analysis_plan.combined must be a list of groups.")

    combined = []
    seen_names = set()
    for i, entry in enumerate(spec):
        where = "analysis_plan.combined[{}]".format(i)
        if not isinstance(entry, dict) or "cohorts" not in entry:
            raise ValueError("{} must be a mapping with a 'cohorts' list.".format(where))

        members = entry["cohorts"]
        _validate_members(members, cohort_names, where + ".cohorts")

        name = str(entry.get("name") or "+".join(members))
        if name in seen_names:
            raise ValueError("{}: duplicate combined analysis name {!r}".format(where, name))
        seen_names.add(name)

        combined.append({"name": name, "cohorts": list(members)})
    return combined


def expand_comparisons(cohort_names, spec):
    """Return the directed comparison pairs (base, comparison)."""
    if spec is None:
        return {"mode": "none", "directed_pairs": []}
    if not isinstance(spec, dict):
        raise ValueError("analysis_plan.comparisons must be a mapping.")

    mode = spec.get("mode", "explicit")
    if mode not in _COMPARISON_MODES:
        raise ValueError(
            "analysis_plan.comparisons.mode must be one of {}, got {!r}".format(
                _COMPARISON_MODES, mode
            )
        )

    names = list(cohort_names)

    if mode == "none":
        pairs = []
    elif mode == "all_directed":
        pairs = [(a, b) for a in names for b in names if a != b]
    elif mode == "all_undirected":
        pairs = [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
    else:  # explicit
        raw = spec.get("pairs", [])
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                "analysis_plan.comparisons.mode: explicit needs a non-empty 'pairs' list."
            )
        pairs = []
        for i, item in enumerate(raw):
            where = "analysis_plan.comparisons.pairs[{}]".format(i)
            if isinstance(item, dict):
                base, comp = item.get("base"), item.get("comparison")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                base, comp = item[0], item[1]
            else:
                raise ValueError(
                    "{} must be [base, comparison] or {{base:, comparison:}}.".format(where)
                )
            _validate_members([base, comp], names, where)
            pairs.append((str(base), str(comp)))

    return {"mode": mode, "directed_pairs": pairs}


# ---------------------------------------------------------------------------
# Top-level plan builders
# ---------------------------------------------------------------------------
def _build_multi_plan(config):
    entries = _cohort_entries(config.get("cohorts"))
    names = [e["name"] for e in entries]

    plan_spec = config.get("analysis_plan") or {}
    if not isinstance(plan_spec, dict):
        raise ValueError("'analysis_plan' must be a mapping.")

    pairwise = expand_pairwise(names, plan_spec.get("pairwise"))
    combined = expand_combined(names, plan_spec.get("combined"))
    comparisons = expand_comparisons(names, plan_spec.get("comparisons"))

    warnings = [
        "{}: {}".format(e["name"], e["warning"])
        for e in entries
        if e["warning"]
    ]
    warnings.extend(_shared_embedding_warnings(entries))

    return {
        "format": MULTI_COHORT,
        "available_cohorts": entries,
        "pairwise": pairwise,
        "combined": combined,
        "comparisons": comparisons,
        "warnings": warnings,
    }


def _build_single_plan(config):
    """Present an existing single-analysis config as a plan, for the dry run.

    The single pipeline computes one pooled pairwise analysis over
    ``active_cohorts`` plus the ``cross_cohort_comparison`` pairs. Nothing about
    the pipeline changes; this is a read-only view.
    """
    cohorts_map = config.get("cohorts", {}) or {}
    entries = _cohort_entries(cohorts_map) if cohorts_map else []
    known = [e["name"] for e in entries]

    active = list(config.get("active_cohorts", known))

    combined = []
    if active:
        combined.append(
            {"name": str(config.get("target_name", "target")), "cohorts": active}
        )

    ccc = config.get("cross_cohort_comparison") or {}
    base = ccc.get("base_cohort")
    comp = ccc.get("comparison_cohorts", [])
    if comp == "all":
        comp = [c for c in active if c != base]
    directed = [(str(base), str(c)) for c in (comp or [])] if base else []

    warnings = [
        "{}: {}".format(e["name"], e["warning"]) for e in entries if e["warning"]
    ]

    return {
        "format": SINGLE_ANALYSIS,
        "available_cohorts": entries,
        "pairwise": {"mode": "pooled", "cohorts": []},
        "combined": combined,
        "comparisons": {"mode": "single-config", "directed_pairs": directed},
        "warnings": warnings,
    }


def _shared_embedding_warnings(entries):
    """Flag cohorts that resolve to the same embedding_file."""
    by_file = {}
    for e in entries:
        if e["embedding_file"]:
            by_file.setdefault(str(e["embedding_file"]), []).append(e["name"])
    out = []
    for path, names in by_file.items():
        if len(names) > 1:
            out.append(
                "cohorts {} share the same embedding_file ({}).".format(names, path)
            )
    return out


def build_plan(config, config_path=None):
    """Detect the config format and return the expanded plan dict."""
    fmt = detect_format(config)
    plan = _build_multi_plan(config) if fmt == MULTI_COHORT else _build_single_plan(config)
    plan["config_path"] = None if config_path is None else str(config_path)
    return plan


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def format_plan(plan):
    """Render a plan dict as a human-readable dry-run report string."""
    lines = []
    lines.append("Analysis plan (dry run)")
    if plan.get("config_path"):
        lines.append("config: {}".format(plan["config_path"]))
    lines.append("format: {}".format(plan["format"]))
    lines.append("")

    entries = plan["available_cohorts"]
    lines.append("Available cohorts ({}):".format(len(entries)))
    if entries:
        width = max(len(e["name"]) for e in entries)
        for e in entries:
            lines.append(
                "  - {:<{w}}  embedding: {}".format(
                    e["name"], e["embedding_file"] or "<none>", w=width
                )
            )
    else:
        lines.append("  (none defined)")
    lines.append("")

    pairwise = plan["pairwise"]
    if pairwise.get("mode") == "pooled":
        lines.append(
            "Pairwise analyses: 1 pooled run (single-analysis config; see Combined)"
        )
    else:
        pw = pairwise["cohorts"]
        lines.append(
            "Pairwise analyses (mode: {}) -> {}:".format(pairwise["mode"], len(pw))
        )
        for name in pw:
            lines.append("  - {}".format(name))
    lines.append("")

    combined = plan["combined"]
    lines.append("Combined analyses -> {}:".format(len(combined)))
    for group in combined:
        lines.append(
            "  - {}: [{}]".format(group["name"], ", ".join(group["cohorts"]))
        )
    if not combined:
        lines.append("  (none)")
    lines.append("")

    comparisons = plan["comparisons"]
    pairs = comparisons["directed_pairs"]
    lines.append(
        "Comparisons (mode: {}) -> {} directed pair(s):".format(
            comparisons["mode"], len(pairs)
        )
    )
    for base, comp in pairs:
        lines.append("  - {} vs {}".format(base, comp))
    if not pairs:
        lines.append("  (none)")
    lines.append("")
    lines.append("Number of directed comparisons: {}".format(len(pairs)))

    if plan.get("warnings"):
        lines.append("")
        lines.append("WARNINGS:")
        for w in plan["warnings"]:
            lines.append("  - {}".format(w))

    lines.append("")
    lines.append(
        "(dry run: no config paths resolved, no embeddings loaded, "
        "no outputs written, no analysis executed)"
    )
    return "\n".join(lines)
