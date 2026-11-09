"""Self-contained HTML report, parameter-filled METHODS.md and manifest for a run.

``<run>/report/index.html`` is one file (inline CSS, base64 PNGs, no scripts, no external
requests) with a section per stage whose ``complete.json`` exists; any subset of stages works.
Everything shown is read from the run directory, never recomputed. The report describes what
ran; it does not interpret the biology and it does not upgrade any statistic: percentile
scores are ranks (not probabilities), cross-validated performance after size selection is
optimistic, and associations with model-input genes are attribution, not independent discovery.

The report stage rebuilds unconditionally (``--force`` is accepted and has no effect).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from degas3 import __version__
from degas3.config import RunConfig
from degas3.contracts import LAYOUT, STAGES
from degas3.runs import RunDir, environment, git_commit, sha256_json
from degas3.validate import read_selected_sizes

log = logging.getLogger(__name__)

MAX_ROWS = 40
EXPLAIN_RESIDUAL_WARN = 0.05
CSS = """
:root{--fg:#1b1b1b;--bg:#fff;--muted:#555;--line:#ddd;--accent:#0072B2;--warn:#fdf3e1}
@media (prefers-color-scheme: dark){:root{--fg:#e8e8e8;--bg:#161616;--muted:#aaa;--line:#444;--accent:#56B4E9;--warn:#3a3020}}
body{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--fg);background:var(--bg);margin:0}
main{max-width:1100px;margin:0 auto;padding:12px 16px 48px}
h1{font-size:1.6em}h2{font-size:1.25em;border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2em}
h3{font-size:1.05em}a{color:var(--accent)}nav a{margin-right:10px;white-space:nowrap}
.scroll{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;font-size:13px}
th,td{border:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}th{background:rgba(128,128,128,.12)}
img{max-width:100%;height:auto;border:1px solid var(--line)}figure{margin:10px 0}figcaption{font-size:12px;color:var(--muted)}
.note{background:var(--warn);padding:6px 10px;border-radius:4px;margin:8px 0}.muted{color:var(--muted)}
details{margin:6px 0}summary{cursor:pointer;font-weight:600}code{font-size:12px}
"""

CITATIONS = {
    "degas": "Johnson TS et al. Genome Med 2022;14:11. doi:10.1186/s13073-022-01012-2",
    "degasv2": "Liu Z, Li S, Couetil J, Zhang J, Huang K, Johnson TS. bioRxiv 2025. doi:10.1101/2025.02.03.636036",
    "pydeseq2": "Muzellec B et al. Bioinformatics 2023;39(9):btad547. doi:10.1093/bioinformatics/btad547",
    "edger": "Robinson MD, McCarthy DJ, Smyth GK. Bioinformatics 2010;26:139-140. doi:10.1093/bioinformatics/btp616",
    "gsea": "Subramanian A et al. PNAS 2005;102:15545-15550. doi:10.1073/pnas.0506580102",
    "gseapy": "Fang Z, Liu X, Peltz G. Bioinformatics 2023;39(1):btac757. doi:10.1093/bioinformatics/btac757",
    "msigdb": "Liberzon A et al. Cell Syst 2015;1:417-425. doi:10.1016/j.cels.2015.12.004",
    "shap": "Lundberg SM, Lee SI. NeurIPS 2017 (arXiv:1705.07874)",
    "rctd": "Cable DM et al. Nat Biotechnol 2022;40:517-526. doi:10.1038/s41587-021-00830-w",
    "cside": "Cable DM et al. Nat Methods 2022;19:1076-1087. doi:10.1038/s41592-022-01575-3",
    "scanpy": "Wolf FA, Angerer P, Theis FJ. Genome Biol 2018;19:15. doi:10.1186/s13059-017-1382-0",
    "leiden": "Traag VA, Waltman L, van Eck NJ. Sci Rep 2019;9:5233. doi:10.1038/s41598-019-41695-z",
}


# -- small helpers -------------------------------------------------------------------------
def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


def _csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if path.is_file() else None
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "" if pd.isna(v) else f"{v:.4g}"
    return "" if v is None else str(v)


def table(df: pd.DataFrame, max_rows: int = MAX_ROWS, note_path: str = "") -> str:
    """HTML table (escaped, horizontally scrollable), truncated to ``max_rows`` rows."""
    if df is None or df.empty:
        return '<p class="muted">(empty table)</p>'
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{esc(_fmt(v))}</td>" for v in r) + "</tr>"
        for r in df.head(max_rows).itertuples(index=False, name=None)
    )
    more = ""
    if len(df) > max_rows:
        more = f'<p class="muted">Showing {max_rows} of {len(df)} rows{esc(" (full table: " + note_path + ")") if note_path else ""}.</p>'
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>{more}'


def kv_table(d: dict, keys: Optional[list] = None) -> str:
    """Two-column table of ``d`` (restricted to ``keys``); containers are shown as JSON."""
    items = [(k, d[k]) for k in (keys if keys is not None else list(d)) if k in d]
    df = pd.DataFrame({"item": [k for k, _ in items],
                       "value": [json.dumps(v, default=str) if isinstance(v, (dict, list)) else v for _, v in items]})
    return table(df, max_rows=100)


class Builder:
    """Collects sections, limitations, copied figures, and the stage summary for one report."""

    def __init__(self, rd: RunDir, cfg: RunConfig):
        self.rd, self.cfg = rd, cfg
        self.figdir = rd.path(LAYOUT.REPORT, "figures")
        self.sections: list[tuple[str, str, str]] = []  # (anchor, title, html)
        self.limits: list[str] = []
        self.figures: list[str] = []

    def img(self, path: Path, caption: str) -> str:
        """Copy ``path`` to report/figures and return a <figure> with the PNG embedded as base64."""
        if not path.is_file():
            return ""
        rel = path.relative_to(self.rd.root).as_posix() if self.rd.root in path.parents else path.name
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", rel.replace("/", "__"))
        self.figdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, self.figdir / name)
        self.figures.append(name)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return (f'<figure><img alt="{esc(caption)}" src="data:image/png;base64,{data}">'
                f"<figcaption>{esc(caption)} <code>{esc(rel)}</code></figcaption></figure>")

    def add(self, anchor: str, title: str, body: str) -> None:
        self.sections.append((anchor, title, body))


# -- sections ------------------------------------------------------------------------------
def selected_sizes(rd: RunDir) -> list[int]:
    return read_selected_sizes(rd, required=False)


def sec_overview(b: Builder) -> str:
    rd, cfg = b.rd, b.cfg
    env = environment()
    keys = ["outcome", "modality", "model", "sizes", "feature_selection", "cv", "folds", "cv_seeds", "final_seeds",
            "iterations", "rank_scope", "split", "de_method", "de_engine", "min_cells", "donor_col", "cluster_col",
            "label_col", "sample_col", "dataset_col"]
    cdict = {k: getattr(cfg, k) for k in keys}
    done = []
    for st in STAGES:
        rec = rd.read_complete(st) if st not in ("init", "report") else None
        if rec:
            done.append({"stage": st, "completed_at": rec.get("completed_at"), "slurm_job_id": rec.get("slurm_job_id")})
    parts = [
        kv_table({"run": rd.root.name, "run directory": str(rd.root), "generated (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "degas3 version": __version__, "git commit": git_commit() or "n/a"}),
        "<h3>Environment</h3>", kv_table(env), "<h3>Configuration</h3>", kv_table(cdict, keys),
        "<h3>Completed stages</h3>", table(pd.DataFrame(done)) if done else '<p class="muted">No stage completion files found.</p>',
    ]
    return "".join(parts)


def sec_inputs(b: Builder, m: dict) -> str:
    bulk, cells, sg = m.get("bulk", {}), m.get("cells", {}), m.get("shared_genes", {})
    oc = bulk.get("outcome_counts", {})
    rows = {
        "modality": m.get("modality"), "outcome type": m.get("outcome_type"),
        ("outcome balance (class: patients)" if m.get("outcome_type") == "binary" else "survival events / censored"): json.dumps(oc),
        "bulk samples x genes": " x ".join(map(str, bulk.get("shape_samples_genes", []))), "patients": bulk.get("n_patients"),
        "bulk datasets": ", ".join(bulk.get("datasets", [])) or "none",
        "cells x genes": " x ".join(map(str, cells.get("shape_cells_genes", []))), "donors": cells.get("n_donors"),
        "clusters": cells.get("n_clusters"), "shared genes": sg.get("n_shared"),
    }
    out = [kv_table(rows)]
    warns = m.get("warnings", [])
    out.append("<h3>Intake warnings</h3>" + ("<ul>" + "".join(f"<li>{esc(w)}</li>" for w in warns) + "</ul>" if warns else "<p>none</p>"))
    return "".join(out)


def sec_cluster(b: Builder) -> str:
    rec = b.rd.read_complete("cluster") or {}
    out = [kv_table({k: rec.get(k) for k in ("resolution", "n_clusters", "outcome_blind")}, ["resolution", "n_clusters", "outcome_blind"])]
    mk = _csv(b.rd.path(LAYOUT.CLUSTER, "markers.csv"))
    if mk is not None and {"cluster", "gene"} <= set(mk.columns):
        top = mk.sort_values(["cluster", "rank"] if "rank" in mk else ["cluster"]).groupby("cluster", sort=True).head(5)
        agg = top.groupby("cluster", sort=True)["gene"].apply(lambda s: ", ".join(map(str, s))).reset_index()
        out.append("<h3>Top 5 marker genes per cluster</h3>" + table(agg) +
                   '<p class="muted">Markers describe clusters (one-vs-rest Wilcoxon); they are not a disease association test.</p>')
    return "".join(out)


def sec_validation(b: Builder) -> str:
    rd = b.rd
    out = []
    met = _csv(rd.path(LAYOUT.CV, "metrics.csv"))
    sel = _json(rd.path(LAYOUT.CV, "selection.json")) or {}
    if met is not None and "size" in met:
        names = ("auroc", "average_precision", "log_loss", "brier", "c_index")
        num = [c for c in names if c in met.columns]
        g = met.groupby("size")[num]
        mean, sd, n = g.mean(), g.std(), g.size()

        def _pm(s, c):
            return f"{mean.loc[s, c]:.3f} ± {0 if pd.isna(sd.loc[s, c]) else sd.loc[s, c]:.3f}"

        tab = pd.DataFrame({c: [_pm(s, c) for s in mean.index] for c in num})
        tab.insert(0, "size", [int(s) for s in mean.index])
        tab["n folds"] = n.to_numpy()
        out.append("<h3>Held-out metrics by feature-set size (mean ± SD across folds)</h3>" + table(tab, 100))
        out.append("<details><summary>Per-fold metrics</summary>" + table(met, 200) + "</details>")
    if sel:
        sizes = selected_sizes(rd)
        out.append(
            f"<p><b>Selected sizes:</b> {esc(sizes)}. One-standard-error rule: every size whose mean {esc(sel.get('metric', 'metric'))} "
            f"is at least the best size's mean minus its bootstrap standard error is retained (best size {esc(sel.get('best_size'))}, "
            f"SE {esc(_fmt(sel.get('se_best', sel.get('SE'))))}, cutoff {esc(_fmt(sel.get('cutoff')))}).</p>")
    out.append('<p class="note">Performance after selection is optimistic: the same folds that report validation choose the feature-set sizes. '
               "The standard error is not a confidence interval.</p>")
    out.append(b.img(rd.path(LAYOUT.CV, "selection.png"), "Cross-validated metric by feature-set size"))
    return "".join(out)


def sec_scores(b: Builder) -> str:
    rd = b.rd
    out = []
    summ = _json(rd.path(LAYOUT.SCORES, "score_summary.json"))
    if summ:
        out.append("<h3>Score summary</h3>" + kv_table(summ))
    out.append('<p class="note">The ensemble score is a percentile rank among scored cells, not a probability.</p>')
    out.append(b.img(rd.path(LAYOUT.SCORES, "violins_by_cluster.png"), "Ensemble score by cluster (donor points over violins)"))
    out.append(b.img(rd.path(LAYOUT.SCORES, "umap_score.png"), "Ensemble score on the embedding"))
    ps = _csv(rd.path(LAYOUT.SCORES, "patient_scores.csv"))
    if ps is not None:
        out.append(f"<h3>Patient scores ({len(ps)} rows)</h3>" + table(ps, 15, "scores/patient_scores.csv") +
                   '<p class="muted">Bulk-sample scores from final models are in-sample; out-of-fold values are the honest ones.</p>')
    return "".join(out)


def sec_strata(b: Builder) -> str:
    rd = b.rd
    out = []
    for name, title in (("cutoffs.csv", "Within-cluster cutoffs"), ("coverage.csv", "Coverage per cluster and donor")):
        df = _csv(rd.path(LAYOUT.STRATA, name))
        if df is not None:
            out.append(f"<h3>{title}</h3>" + table(df, 60, f"strata/{name}"))
    return "".join(out)


def _col(df: pd.DataFrame, names: tuple) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _top_updown(df: pd.DataFrame, n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene = _col(df, ("gene", "names", "gene_name", "Unnamed: 0")) or df.columns[0]
    fc = _col(df, ("log2FoldChange", "log2fc", "logfoldchange", "log2_fold_change"))
    stat = _col(df, ("stat", "score"))
    rank = stat or fc
    keep = [c for c in (gene, fc, stat, _col(df, ("pvalue", "pval", "p")), _col(df, ("padj", "fdr", "pvals_adj"))) if c]
    keep = list(dict.fromkeys(keep))
    if rank is None:
        return df[keep].head(n), df[keep].iloc[0:0]
    d = df.dropna(subset=[rank])
    return d.nlargest(n, rank)[keep], d.nsmallest(n, rank)[keep]


def sec_de(b: Builder) -> str:
    rd = b.rd
    base = rd.path(LAYOUT.DE)
    out = []
    summ = _csv(base / "summary.csv")
    if summ is not None:
        out.append("<h3>Summary by cluster</h3>" + table(summ, 100, "de/summary.csv"))
    wil, skipped = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        meth = _json(d / "method.json") or {}
        name = str(meth.get("method", meth.get("name", "unknown")))
        if "wilcoxon" in name.lower():
            wil.append(d.name)
        if meth.get("skipped") or meth.get("skip_reason"):
            skipped.append(f"{d.name}: {meth.get('skip_reason', 'skipped')}")
        res = _csv(d / "results.csv")
        inner = [f"<p>Method: <b>{esc(name)}</b></p>"]
        if meth:
            inner.append(kv_table({k: v for k, v in meth.items() if k not in ("method", "name")}))
        if res is not None and len(res):
            up, down = _top_updown(res)
            inner.append("<h4>Top 15 up (high-score side)</h4>" + table(up, 15) + "<h4>Top 15 down</h4>" + table(down, 15))
        inner.append(b.img(d / "volcano.png", f"Volcano, cluster {d.name}"))
        inner.append(b.img(d / "clinical_vs_score.png", f"Clinical-label vs score contrast, cluster {d.name}"))
        out.append(f"<details><summary>Cluster {esc(d.name)}: {esc(name)}</summary>{''.join(inner)}</details>")
    if wil:
        b.limits.append(f"No donor replication: Cell-level Wilcoxon was used in clusters {', '.join(wil)}; its p-values describe cells, not donors.")
    b.limits += [f"Differential expression skipped or restricted in cluster {s}." for s in skipped]
    return "".join(out)


def sec_pathways(b: Builder) -> str:
    rd = b.rd
    out = []
    summ = _csv(rd.path(LAYOUT.PATHWAYS, "summary.csv"))
    if summ is not None:
        out.append(table(summ, 60, "pathways/summary.csv"))
    out.append(b.img(rd.path(LAYOUT.PATHWAYS, "gsea_dotplot.png"), "Pathway enrichment across clusters"))
    out.append('<p class="muted">Pathway rankings come from the model-derived contrast; they are not independent confirmation.</p>')
    return "".join(out)


def sec_explain(b: Builder) -> str:
    rd = b.rd
    base = rd.path(LAYOUT.EXPLAIN)
    out = ["<p class=\"note\">Model attribution relative to low-score same-cluster cells; not causal, not independent discovery. "
           "The explained quantity is the mean raw score across retained models, not the percentile.</p>"]
    summ = _csv(base / "summary.csv")
    if summ is not None:
        keep = [c for c in ("cluster", "status", "n_targets", "n_reference", "n_donors", "n_models", "max_abs_residual", "prediction_range", "reason") if c in summ]
        out.append("<h3>Additivity check</h3>" + table(summ[keep], 100, "explain/summary.csv"))
        if {"max_abs_residual", "prediction_range"} <= set(summ):
            bad = summ[(summ["max_abs_residual"] > EXPLAIN_RESIDUAL_WARN * summ["prediction_range"].fillna(1.0).clip(lower=0.1))]
            if len(bad):
                b.limits.append("SHAP additivity residual exceeds 5% of the prediction range in clusters "
                                f"{', '.join(map(str, bad['cluster']))}; treat those attributions as approximate.")
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        figs = b.img(d / "top_genes_abs.png", f"Top genes by absolute attribution, cluster {d.name}") + \
            b.img(d / "top_genes_signed.png", f"Top genes by signed attribution, cluster {d.name}")
        if figs:
            out.append(f"<details><summary>Cluster {esc(d.name)}</summary>{figs}</details>")
    return "".join(out)


def sec_spatial(b: Builder) -> str:
    rd = b.rd
    base = rd.path(LAYOUT.SPATIAL)
    out = []
    counts = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        nb = d / "neighborhoods.csv.gz"
        if nb.is_file():
            try:
                counts[d.name] = len(pd.read_csv(nb))
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                pass
        maps = sorted((d / "maps").glob("*.png")) if (d / "maps").is_dir() else []
        if maps:
            out.append(f"<details><summary>Maps, {esc(d.name)} ({len(maps)})</summary>"
                       + "".join(b.img(m, f"{d.name}: {m.stem}") for m in maps[:8]) + "</details>")
    if counts:
        out.insert(0, "<h3>Neighborhoods per sample</h3>" + table(pd.DataFrame({"sample": list(counts), "neighborhoods": list(counts.values())}))
                   + _neighborhood_figure(b, counts))
        b.limits.append("Neighborhoods overlap and are not independent observations; do not treat them as replicates.")
    cs = _csv(rd.path(LAYOUT.CSIDE, "summary.csv"))
    if cs is not None:
        out.append("<h3>C-SIDE summary</h3>" + table(cs, 60, "cside/summary.csv"))
    return "".join(out)


def _neighborhood_figure(b: Builder, counts: dict) -> str:
    from degas3.plotting.style import save_panel, square_panel

    fig, ax = square_panel("Neighborhoods per sample", "Sample", "Number of neighborhoods")
    ax.bar(range(len(counts)), list(counts.values()), color="#555555")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(list(counts), rotation=45, ha="right")
    stem = b.rd.path(LAYOUT.REPORT, "figures", "neighborhood_counts")
    save_panel(fig, ax, stem, formats=("png",))
    return b.img(stem.with_name("neighborhood_counts.png"), "Neighborhoods per sample")


def generated_limitations(b: Builder, manifest: Optional[dict]) -> list[str]:
    rd, lim = b.rd, []
    if rd.stage_complete("select"):
        lim.append("Feature-set sizes were selected on the same folds used to report validation; reported performance is optimistic.")
    if rd.stage_complete("score"):
        lim.append("Percentile scores are ranks among scored cells, not probabilities.")
    if rd.stage_complete("de") or rd.stage_complete("pathways") or rd.stage_complete("explain"):
        lim.append("Model-input genes are attribution, not discovery: genes used to build the score are expected to associate with it.")
    if manifest:
        nd = manifest.get("cells", {}).get("n_donors")
        if nd is not None and nd < 4:
            lim.append(f"Only {nd} donors: donor-level inference has little power and intervals are unstable.")
        lim += [f"Intake warning: {w}" for w in manifest.get("warnings", [])]
    cov = _csv(rd.path(LAYOUT.STRATA, "coverage.csv"))
    if cov is not None and {"cluster", "donor", "n_high", "n_low"} <= set(cov.columns):
        cov["cluster"] = cov["cluster"].astype(str)
        ok = cov[(cov["n_high"] >= b.cfg.min_cells) & (cov["n_low"] >= b.cfg.min_cells)].groupby("cluster")["donor"].nunique()
        for c in sorted(cov["cluster"].unique()):
            n = int(ok.get(c, 0))
            if n < 2:
                lim.append(f"Cluster {c}: only {n} donor(s) have at least {b.cfg.min_cells} cells in both the high and low group; no donor replication.")
    if rd.stage_complete("neighborhoods") or rd.stage_complete("cside"):
        lim.append("Spatial scores and C-SIDE results are associations with the model score, not independent confirmation.")
    return lim + [x for x in b.limits if x not in lim]


# -- METHODS.md -----------------------------------------------------------------------------
def methods_text(rd: RunDir, cfg: RunConfig, manifest_in: Optional[dict]) -> str:
    """Methods paragraph per stage that ran, filled with this run's parameters; ends with citations."""
    sizes = selected_sizes(rd)
    sel = _json(rd.path(LAYOUT.CV, "selection.json")) or {}
    cites = ["degas", "degasv2"]
    m = ["# Methods", "",
         f"Generated by degas3 {__version__} for run `{rd.root.name}`. Parameters below are read from `config.json` "
         "and the stage completion files.", ""]
    shared = (manifest_in or {}).get("shared_genes", {}).get("n_shared")
    m += ["## Data and preprocessing", "",
          f"Bulk RNA-seq ({cfg.outcome or 'unspecified'} outcome) and {cfg.modality or 'single-cell'} data were intersected on gene identifiers"
          + (f" ({shared} shared genes)" if shared else "") + ". Counts were transformed per sample with 1.5^log2(x+1), z-scored across the "
          "selected genes and min-max scaled, after gene selection and separately for bulk and cells"
          + ("; bulk input was already log2(x+1)" if cfg.already_log else "") + ".", ""]
    if rd.stage_complete("cluster"):
        cites += ["scanpy", "leiden"]
        c = rd.read_complete("cluster") or {}
        m += ["## Clustering", "", f"Cells were clustered without outcome or score information: library-size normalization (10,000), log1p, 2,000 highly "
              f"variable genes, PCA (50), 15 neighbors, Leiden resolution {c.get('resolution', cfg.cluster_resolution)} "
              f"({c.get('n_clusters', 'n/a')} clusters) and UMAP.", ""]
    if rd.stage_complete("prepare") or rd.stage_complete("select") or rd.stage_complete("fit"):
        m += ["## Model", "",
              f"DEGAS ({cfg.resolved_model()} variant) with a 3-layer sigmoid feature extractor (feature dimension {cfg.feature_dim}), "
              f"a Wasserstein discriminator with gradient penalty, and an outcome head; Adam, learning rate 0.01, {cfg.iterations} updates, "
              f"batches of {cfg.batch_size}. Genes were ranked inside each training fold by `{cfg.feature_selection}`; candidate set sizes "
              f"were {cfg.sizes}.", ""]
    if rd.stage_complete("select"):
        m += ["## Validation and size selection", "",
              f"{cfg.cv} cross-validation ({cfg.folds} folds, {cfg.cv_seeds} seeds per fit) with {sel.get('metric', 'the primary metric')} as the primary metric. "
              f"The one-standard-error rule ({cfg.bootstraps} patient bootstraps, seed {cfg.selection_seed}) retained sizes {sizes}. "
              "Performance after selection is optimistic because selection and reporting use the same folds.", ""]
    if rd.stage_complete("score"):
        m += ["## Scoring", "",
              f"Final models were fitted on all bulk samples for each retained size ({sizes}) with {cfg.final_seeds} seeds. Raw scores were averaged over seeds, "
              f"converted to average-tie percentile ranks (rank scope: {cfg.rank_scope}) and averaged equally over sizes to give the ensemble "
              "score. It is a percentile rank, not a probability.", ""]
    if rd.stage_complete("stratify"):
        split = "the median" if cfg.split == "median" else f"quantiles {cfg.quantiles}"
        m += ["## Stratification", "", f"Within each cluster, cells were split at {split} of the ensemble score; cells equal to a cutoff were excluded as ties.", ""]
    if rd.stage_complete("de"):
        cites.append("scanpy")
        if cfg.de_engine == "edger":
            cites.append("edger")
        else:
            cites.append("pydeseq2")
        m += ["## Differential expression", "",
              f"High versus low groups were compared within each cluster (method setting `{cfg.de_method}`, engine `{cfg.de_engine}`). Pseudobulk profiles "
              f"(raw counts summed per donor and group; at least {cfg.min_cells} cells) were tested with design ~ donor + group and Benjamini-Hochberg FDR; "
              "where donor replication was absent, a cell-level Wilcoxon test was used and is not donor-aware. Genes in the model input are attributed, not independently discovered.", ""]
    if rd.stage_complete("pathways"):
        cites += ["gsea", "gseapy", "msigdb"]
        m += ["## Pathways", "",
              f"Preranked GSEA (gseapy) on the differential-expression statistic with `{cfg.gmt_path or cfg.msigdb_collection}` "
              f"({cfg.species}{', version ' + cfg.msigdb_version if cfg.msigdb_version else ''}), gene sets of {cfg.pathway_min_size} to {cfg.pathway_max_size} genes, "
              f"{cfg.pathway_permutations} permutations, seed {cfg.pathway_seed}, plus hypergeometric over-representation of significant up and down genes.", ""]
    if rd.stage_complete("explain"):
        cites.append("shap")
        m += ["## Model attribution", "",
              f"Gradient SHAP (expected gradients, 200 samples per target) was computed for each retained model. The reference was a donor-balanced sample "
              f"(up to {cfg.explain_reference_cells} cells) of same-cluster cells below the {cfg.explain_reference_quantile:.0%} score quantile; targets were at most one cell per donor "
              f"and score quartile Q2 to Q4. Attributions were averaged over models, so the explained quantity is the mean raw score, not the percentile. Pooled and equal-donor "
              f"signed and absolute means are reported with {cfg.explain_bootstraps} donor bootstraps (seed {cfg.explain_seed}) and additivity residuals. "
              "This is model attribution relative to low-score same-cluster cells, not a causal effect.", ""]
    if rd.stage_complete("neighborhoods") or rd.stage_complete("maps"):
        m += ["## Spatial analysis", "",
              f"Scores were mapped per section; for single-cell spatial data, neighborhood pooling targeted {cfg.neighborhood_targets} cells with {cfg.n_seeds} seed cells "
              f"({cfg.calibration_seeds} calibration and {cfg.evaluation_seeds} evaluation seeds).", ""]
    if rd.stage_complete("cside"):
        cites += ["rctd", "cside"]
        m += ["## RCTD and C-SIDE", "", "Cell types were decomposed with RCTD and the scaled ensemble score was used as the C-SIDE explanatory variable across sections.", ""]
    m += ["## Citations", ""] + [f"- {CITATIONS[c]}" for c in dict.fromkeys(cites)] + [""]
    return "\n".join(m)


# -- manifest -------------------------------------------------------------------------------
def build_manifest(rd: RunDir, cfg: RunConfig, manifest_in: Optional[dict], figures: list[str]) -> dict:
    stages = {}
    for st in STAGES:
        rec = rd.read_complete(st) if st not in ("init", "report") else None
        if rec:
            stages[st] = rec
    return {
        "run": rd.root.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "git_commit": git_commit(),
        "environment": environment(),
        "config_sha256": sha256_json(cfg.to_dict()),
        "input_hashes": {k: v.get("sha256") for k, v in (manifest_in or {}).get("sources", {}).items()},
        "selected_sizes": selected_sizes(rd),
        "seeds": {"cv_seeds": cfg.cv_seeds, "final_seeds": cfg.final_seeds, "selection_seed": cfg.selection_seed,
                  "explain_seed": cfg.explain_seed, "pathway_seed": cfg.pathway_seed},
        "stages": stages,
        "figures": figures,
    }


def build_report(run: str | Path) -> Path:
    """Write ``report/index.html``, ``METHODS.md``, ``manifest.json`` and ``figures/``; returns the HTML path."""
    rd = RunDir(run)
    if not rd.config_path.is_file():
        raise ValueError(f"{rd.root} has no config.json; run `degas3 init` first.")
    cfg = RunConfig.load(rd.root)
    out = rd.ensure(LAYOUT.REPORT)
    shutil.rmtree(out / "figures", ignore_errors=True)  # figures are copies; rebuild from the sources
    b = Builder(rd, cfg)
    mi = _json(rd.input_manifest)
    b.add("overview", "Overview", sec_overview(b))
    if mi:
        b.add("inputs", "Inputs", sec_inputs(b, mi))
    plan = [
        ("cluster", "clustering", "Clustering", sec_cluster), ("select", "validation", "Validation", sec_validation),
        ("score", "scores", "Scores", sec_scores), ("stratify", "stratification", "Stratification", sec_strata),
        ("de", "de", "Differential expression", sec_de), ("pathways", "pathways", "Pathways", sec_pathways),
        ("explain", "explanations", "Explanations", sec_explain),
    ]
    for stage, anchor, title, fn in plan:
        if rd.stage_complete(stage):
            b.add(anchor, title, fn(b))
    if any(rd.stage_complete(s) for s in ("neighborhoods", "maps", "cside")):
        b.add("spatial", "Spatial", sec_spatial(b))
    limits = generated_limitations(b, mi)
    b.add("limitations", "Limitations", "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in limits) + "</ul>" if limits else "<p>None generated.</p>")
    manifest = build_manifest(rd, cfg, mi, b.figures)
    b.add("provenance", "Provenance", kv_table({k: manifest[k] for k in ("config_sha256", "package_version", "git_commit")})
          + "<h3>Input hashes (sha256)</h3>" + (kv_table(manifest["input_hashes"]) if manifest["input_hashes"] else "<p>none recorded</p>")
          + '<p class="muted">Full record: <code>report/manifest.json</code>. Methods text: <code>report/METHODS.md</code>.</p>')
    nav = "".join(f'<a href="#{a}">{esc(t)}</a>' for a, t, _ in b.sections)
    body = "".join(f'<section id="{a}"><h2>{esc(t)}</h2>{h}</section>' for a, t, h in b.sections)
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>DEGAS3 report - {esc(rd.root.name)}</title><style>{CSS}</style></head><body><main>"
           f"<h1>DEGAS3 report: {esc(rd.root.name)}</h1><nav>{nav}</nav>{body}</main></body></html>")
    (out / "index.html").write_text(doc, encoding="utf-8")
    (out / "METHODS.md").write_text(methods_text(rd, cfg, mi), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    rd.write_complete("report", {"sections": [a for a, _, _ in b.sections], "n_figures": len(b.figures)})
    log.info("report written to %s", out / "index.html")
    return out / "index.html"


def cli_report(args: argparse.Namespace) -> int:
    """CLI entry point for ``degas3 report RUN`` (always rebuilds)."""
    build_report(args.run)
    return 0
