"""Pathway enrichment of the within-cluster high-vs-low DE results.

* GSEA (``gseapy.prerank``): a permutation-based enrichment of gene sets along a ranked gene list
  (DESeq2/edgeR ``stat``, or sign(log2FC) * -log10(p) for Cell-level Wilcoxon). NES is an
  enrichment score normalised across permutations; it is not an effect size of expression and
  its FDR is across gene sets of one cluster only.
* ORA: a one-sided hypergeometric test of the significant up (or down) gene list against the
  tested genes of the same contrast. It ignores the magnitude of change and the dependence
  between genes; the BH correction is within one list.

Both are descriptive summaries of a contrast whose groups were defined by the DEGAS score;
genes the model used make the enrichment model attribution rather than independent discovery.
Pseudobulk contrasts support donor-level inference; Cell-level Wilcoxon rankings do not.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from degas3.config import RunConfig
from degas3.contracts import LAYOUT
from degas3.de import ALPHA, bh_adjust
from degas3.plotting import style
from degas3.runs import RunDir, sha256_file

log = logging.getLogger(__name__)

CACHE_DIR = Path("~/.cache/degas3/msigdb").expanduser()
ORA_COLUMNS = ["term", "set_size", "overlap", "list_size", "background_size", "fold_enrichment",
               "pvalue", "padj", "genes"]
SUMMARY_COLUMNS = ["cluster", "ranking", "n_ranked_genes", "n_gene_sets_tested", "n_gsea_fdr_lt_0.25",
                   "top_term", "top_nes", "n_ora_high_sets", "n_ora_low_sets", "skipped_reason"]


def read_gmt(path: str | Path) -> dict[str, list[str]]:
    """Parse a GMT file (name, description, genes; tab separated)."""
    sets: dict[str, list[str]] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[0]:
            sets[parts[0]] = [g for g in parts[2:] if g]
    if not sets:
        raise ValueError(f"no gene sets parsed from {path}; a GMT line is 'name<TAB>description<TAB>gene...'.")
    return sets


def write_gmt(sets: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}\tNA\t" + "\t".join(v) + "\n" for k, v in sets.items()))


def latest_msigdb_version(dbver, tag: str) -> str:
    """Newest release like ``2026.2.Hs`` for species tag ``Hs``/``Mm`` from ``Msigdb().list_dbver()``."""
    names = dbver["Name"] if hasattr(dbver, "columns") else dbver
    pat = re.compile(r"^(\d{4})\.(\d+)\.(Hs|Mm)$")
    found = []
    for n in map(str, names):
        m = pat.match(n)
        if m and m.group(3) == tag:
            found.append(((int(m.group(1)), int(m.group(2))), n))
    if not found:
        raise RuntimeError(f"no MSigDB release for species tag {tag}")
    return max(found)[1]


def resolve_gene_sets(cfg: RunConfig, gmt: Optional[str]) -> tuple[dict[str, list[str]], dict]:
    """Gene sets from a GMT file, else MSigDB via gseapy (cached under ``~/.cache/degas3/msigdb``)."""
    path = gmt or cfg.gmt_path
    if path:
        if not Path(path).is_file():
            raise ValueError(f"GMT file {path} not found; pass an existing file with --gmt.")
        return read_gmt(path), {"source": "gmt", "gmt_path": str(path), "collection": Path(path).stem,
                                "version": None, "gmt_sha256": sha256_file(path)}
    tag = "Hs" if cfg.species == "human" else "Mm"
    coll = cfg.msigdb_collection
    version = cfg.msigdb_version
    sets = None
    try:
        import gseapy

        msig = gseapy.Msigdb()
        if version is None:
            version = latest_msigdb_version(msig.list_dbver(), tag)
        cached = CACHE_DIR / f"{coll}.v{version}.gmt"
        if cached.is_file():
            sets = read_gmt(cached)
        else:
            sets = {k: list(v) for k, v in msig.get_gmt(category=coll, dbver=version).items()}
            write_gmt(sets, cached)
    except Exception as exc:  # noqa: BLE001 - network/package failures are all "offline"
        cached_any = sorted(CACHE_DIR.glob(f"{coll}.v*.{tag}.gmt")) if CACHE_DIR.is_dir() else []
        if version is None and cached_any:
            cached = cached_any[-1]
            version = cached.name[len(coll) + 2:-4]
            sets = read_gmt(cached)
            log.warning("MSigDB download failed (%s); using cached %s", exc, cached)
        else:
            raise ValueError(
                f"could not obtain MSigDB collection {coll!r} ({exc}); pass a local gene-set file with "
                "`--gmt PATH` (GMT format) or run once with network access."
            ) from exc
    return sets, {"source": "msigdb", "collection": coll, "version": version, "species": cfg.species,
                  "gmt_sha256": sha256_file(cached), "cache_file": str(cached)}


def ranking_from_results(res: pd.DataFrame, method: str) -> tuple[pd.DataFrame, str]:
    """Deduplicated gene ranking: ``stat`` for pseudobulk, signed -log10 p for Cell-level Wilcoxon."""
    if method and "Wilcoxon" in method:
        p = np.clip(res["pvalue"].to_numpy(float), 1e-300, 1.0)
        score = np.sign(res["log2FoldChange"].to_numpy(float)) * -np.log10(p)
        name = "sign(log2FC) * -log10(p)"
    else:
        score = res["stat"].to_numpy(float)
        name = "stat"
    rnk = pd.DataFrame({"gene": res["gene"].astype(str), "score": score}).dropna()
    rnk = rnk.reindex(rnk["score"].abs().sort_values(ascending=False).index).drop_duplicates("gene")
    return rnk.sort_values(["score", "gene"], ascending=[False, True]).reset_index(drop=True), name


def ora(genes, background, gene_sets: dict[str, list[str]], min_size: int = 15, max_size: int = 500) -> pd.DataFrame:
    """Hypergeometric over-representation of ``genes`` in each set, within ``background``.

    ``pvalue = P(X >= overlap)`` for X ~ Hypergeom(N=|background|, K=|set & background|, n=|list|);
    ``padj`` is Benjamini-Hochberg across the sets tested. Sets whose size inside the
    background is outside ``[min_size, max_size]`` are not tested.
    """
    bg = set(map(str, background))
    lst = set(map(str, genes)) & bg
    rows = []
    for term, members in gene_sets.items():
        s = set(members) & bg
        if not (min_size <= len(s) <= max_size):
            continue
        k = len(lst & s)
        p = float(hypergeom.sf(k - 1, len(bg), len(s), len(lst))) if lst else 1.0
        fold = (k / len(lst)) / (len(s) / len(bg)) if lst else np.nan
        rows.append({"term": term, "set_size": len(s), "overlap": k, "list_size": len(lst),
                     "background_size": len(bg), "fold_enrichment": fold, "pvalue": p,
                     "genes": ";".join(sorted(lst & s))})
    if not rows:
        return pd.DataFrame(columns=ORA_COLUMNS)
    out = pd.DataFrame(rows)
    out["padj"] = bh_adjust(out["pvalue"].to_numpy())
    return out[ORA_COLUMNS].sort_values("pvalue").reset_index(drop=True)


def _plot_dotplot(all_gsea: pd.DataFrame, stem: Path, top: int = 10) -> None:
    from matplotlib.colors import TwoSlopeNorm

    d = all_gsea.copy()
    terms = d.assign(a=d["NES"].abs()).groupby("Term")["a"].max().nlargest(top).index.tolist()
    d = d[d["Term"].isin(terms)]
    clusters = sorted(d["cluster"].unique())
    fig, ax = style.square_panel("GSEA: high vs low", "Cluster", "")
    ypos = {t: i for i, t in enumerate(terms[::-1])}
    xpos = {c: i for i, c in enumerate(clusters)}
    lim = max(float(d["NES"].abs().max()), 1e-6)
    sc = ax.scatter([xpos[c] for c in d["cluster"]], [ypos[t] for t in d["Term"]],
                    s=20 + 40 * np.minimum(-np.log10(np.clip(d["FDR"], 1e-4, 1)), 4), c=d["NES"],
                    cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), edgecolor="k", linewidth=0.3)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([t[:28] for t in terms[::-1]], fontsize=7)
    ax.set_xlim(-0.6, len(clusters) - 0.4)
    style.colorbar(fig, ax, sc, "NES")
    fig.text(0.19, 0.04, "Dot size: -log10 FDR (capped at 4). Terms: top by |NES| across clusters.", fontsize=6.5)
    style.save_panel(fig, ax, stem)


def run_pathways(run: RunDir, cfg: RunConfig, gmt: Optional[str]) -> dict:
    import gseapy

    if not run.stage_complete("de"):
        raise ValueError(f"de is not complete; run `degas3 de {run.root}` first.")
    sets, src = resolve_gene_sets(cfg, gmt)
    summary = pd.read_csv(run.path(LAYOUT.DE, "summary.csv"), keep_default_na=False, dtype={"cluster": str})
    out = run.ensure(LAYOUT.PATHWAYS)
    rows, gsea_all = [], []
    for r in summary.itertuples():
        row = {c: None for c in SUMMARY_COLUMNS}
        row.update(cluster=r.cluster, skipped_reason="")
        rows.append(row)
        res_path = run.path(LAYOUT.DE, r.directory, "results.csv")
        if r.skipped_reason or not res_path.is_file():
            row["skipped_reason"] = f"no DE results ({r.skipped_reason or 'results.csv missing'})"
            continue
        res = pd.read_csv(res_path)
        rnk, rank_name = ranking_from_results(res, str(r.method))
        row.update(ranking=rank_name, n_ranked_genes=len(rnk))
        dest = out / r.directory
        dest.mkdir(parents=True, exist_ok=True)
        rnk.to_csv(dest / "ranking.csv", index=False)
        background = res["gene"].astype(str).tolist()
        bg = set(background)
        n_elig = sum(cfg.pathway_min_size <= len(bg & set(m)) <= cfg.pathway_max_size for m in sets.values())
        if n_elig == 0:
            msg = (f"no gene set has {cfg.pathway_min_size}-{cfg.pathway_max_size} genes among the "
                   f"{len(bg)} tested genes (gene identifiers may not match the collection)")
            (dest / "gsea_status.json").write_text(json.dumps(
                {"status": "no_eligible_gene_sets", "detail": msg, "n_tested_genes": len(bg),
                 "min_size": cfg.pathway_min_size, "max_size": cfg.pathway_max_size}, indent=2) + "\n")
            row["skipped_reason"] = "no_eligible_gene_sets"
            log.warning("pathways cluster %s skipped: %s", r.cluster, msg)
            continue
        sig = res[res["padj"] < ALPHA]
        for name, genes in (("high", sig[sig["log2FoldChange"] > 0]["gene"]), ("low", sig[sig["log2FoldChange"] < 0]["gene"])):
            tab = ora(genes, background, sets, cfg.pathway_min_size, cfg.pathway_max_size)
            tab.to_csv(dest / f"ora_{name}.csv", index=False)
            row[f"n_ora_{name}_sets"] = int((tab["padj"] < ALPHA).sum()) if len(tab) else 0
        try:
            pre = gseapy.prerank(rnk=rnk, gene_sets=sets, min_size=cfg.pathway_min_size,
                                 max_size=cfg.pathway_max_size, permutation_num=cfg.pathway_permutations,
                                 seed=cfg.pathway_seed, outdir=None, threads=2, verbose=False)
        except Exception as exc:  # noqa: BLE001 - e.g. no set passes the size filter
            row["skipped_reason"] = f"GSEA failed: {exc}"
            log.warning("GSEA failed for cluster %s: %s", r.cluster, exc)
            continue
        g = pre.res2d.copy()
        for col in ("ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"):
            g[col] = pd.to_numeric(g[col], errors="coerce")
        g = g.sort_values("NES", ascending=False).reset_index(drop=True)
        g.to_csv(dest / "high_vs_low_gsea.csv", index=False)
        top = g.iloc[g["NES"].abs().argmax()] if len(g) else None
        row.update(n_gene_sets_tested=len(g), top_term=None if top is None else top["Term"],
                   top_nes=None if top is None else float(top["NES"]))
        row["n_gsea_fdr_lt_0.25"] = int((g["FDR q-val"] < 0.25).sum())
        gsea_all.append(g.assign(cluster=r.cluster, FDR=g["FDR q-val"]))
    pd.DataFrame(rows)[SUMMARY_COLUMNS].to_csv(out / "summary.csv", index=False)
    if gsea_all:
        _plot_dotplot(pd.concat(gsea_all, ignore_index=True), out / "gsea_dotplot")
    return {
        **src,
        "settings": {"min_size": cfg.pathway_min_size, "max_size": cfg.pathway_max_size,
                     "permutations": cfg.pathway_permutations, "seed": cfg.pathway_seed,
                     "ora_alpha": ALPHA, "ora_background": "genes tested in the DE contrast"},
        "n_gene_sets": len(sets), "n_clusters": len(rows), "n_clusters_with_gsea": len(gsea_all),
        "inference_note": "enrichment of a DE ranking; not independent of genes used by the model; "
                          "donor-level only where the DE method was pseudobulk",
    }


def cli_pathways(args: argparse.Namespace) -> int:
    """``degas3 pathways RUN [--gmt PATH | --collection h.all --species human]``."""
    run = RunDir(args.run)
    cfg = RunConfig.load(run.root, msigdb_collection=getattr(args, "collection", None),
                         species=getattr(args, "species", None))
    if getattr(args, "force", False):
        run.complete_path("pathways").unlink(missing_ok=True)
    if run.stage_complete("pathways"):
        log.info("pathways already complete; skipping (delete %s to rerun)", run.complete_path("pathways"))
        return 0
    payload = run_pathways(run, cfg, getattr(args, "gmt", None))
    run.write_complete("pathways", payload)
    return 0
