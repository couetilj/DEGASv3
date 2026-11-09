import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from degas3.io import (intersect_genes, load_bulk, load_bulk_npz, load_cells, save_bulk_npz,
                       strip_ensembl_version)


def _bulk_files(tmp_path, transpose):
    rng = np.random.default_rng(0)
    genes = [f"ENSG0000000{i}.{i}" for i in range(10, 30)]
    ids = [f"s{i}" for i in range(8)]
    df = pd.DataFrame(rng.poisson(20, (20, 8)), index=genes, columns=ids)
    pheno = pd.DataFrame({"sample_id": ids, "outcome": [0, 1] * 4})
    path = tmp_path / ("t.csv" if transpose else "g.csv")
    (df.T if transpose else df).to_csv(path)
    pheno.to_csv(tmp_path / "pheno.csv", index=False)
    return path, tmp_path / "pheno.csv", df


@pytest.mark.parametrize("transpose", [False, True])
def test_csv_orientation_detected(tmp_path, transpose):
    path, pheno, df = _bulk_files(tmp_path, transpose)
    b = load_bulk(path, pheno)
    assert b.counts.shape == (8, 20)
    assert b.sample_ids == list(df.columns)
    assert b.genes[0] == "ENSG000000010"  # version stripped
    np.testing.assert_array_equal(b.counts, df.T.to_numpy())
    assert b.outcome_type == "binary"


def test_bulk_npz_round_trip_and_survival(tmp_path):
    path, pheno, _ = _bulk_files(tmp_path, False)
    p = pd.read_csv(pheno)
    p["time"], p["event"] = np.arange(1, 9) * 1.5, [1, 0] * 4
    p["dataset"] = ["a", "b"] * 4
    p.to_csv(pheno, index=False)
    b = load_bulk(path, pheno, outcome="survival")
    assert b.outcome_type == "survival" and list(b.pheno.columns) == ["time", "event", "dataset"]
    save_bulk_npz(b, tmp_path / "b.npz")
    b2 = load_bulk_npz(tmp_path / "b.npz")
    np.testing.assert_array_equal(b.counts, b2.counts)
    assert b2.genes == b.genes and b2.outcome_type == "survival"
    pd.testing.assert_frame_equal(b.pheno, b2.pheno, check_dtype=False, check_names=False)


def test_bulk_errors(tmp_path):
    path, pheno, _ = _bulk_files(tmp_path, False)
    p = pd.read_csv(pheno)
    p["outcome"] = 1
    p.to_csv(pheno, index=False)
    with pytest.raises(ValueError, match="single class"):
        load_bulk(path, pheno)
    p["sample_id"] = [f"x{i}" for i in range(8)]
    p.to_csv(pheno, index=False)
    with pytest.raises(ValueError, match="sample"):
        load_bulk(path, pheno)


def test_strip_ensembl():
    assert strip_ensembl_version("ENSG00000123456.12") == "ENSG00000123456"
    assert strip_ensembl_version("ENSMUSG00000000001.3") == "ENSMUSG00000000001"
    assert strip_ensembl_version("MT-CO1.1") == "MT-CO1.1"


def _adata(X, genes=None):
    genes = genes or [f"ENSG{i:011d}.2" for i in range(X.shape[1])]
    return ad.AnnData(X=X, obs=pd.DataFrame({"donor": ["a", "b"] * (X.shape[0] // 2)}, index=[f"c{i}" for i in range(X.shape[0])]),
                      var=pd.DataFrame(index=genes))


def test_h5ad_counts_validation(tmp_path):
    rng = np.random.default_rng(0)
    good = _adata(sparse.csr_matrix(rng.poisson(1, (10, 6)).astype(np.float32)))
    good.write_h5ad(tmp_path / "good.h5ad")
    cd = load_cells(tmp_path / "good.h5ad", donor_col="donor")
    assert cd.modality == "sc" and cd.adata.var_names[0] == "ENSG00000000000"
    bad = _adata(rng.random((10, 6)).astype(np.float32))
    bad.write_h5ad(tmp_path / "bad.h5ad")
    with pytest.raises(ValueError, match="integer"):
        load_cells(tmp_path / "bad.h5ad")
    bad.layers["counts"] = rng.poisson(1, (10, 6)).astype(np.float32)
    bad.write_h5ad(tmp_path / "layer.h5ad")
    assert load_cells(tmp_path / "layer.h5ad").adata.n_obs == 10  # counts layer wins
    with pytest.raises(ValueError, match="donor"):
        load_cells(tmp_path / "good.h5ad", donor_col="missing")


def test_intersect_genes(tmp_path):
    path, pheno, _ = _bulk_files(tmp_path, False)
    b = load_bulk(path, pheno)
    a = ad.AnnData(X=np.ones((2, 5), np.float32), var=pd.DataFrame(index=b.genes[:5] + ["other"] * 0))
    res = intersect_genes(b, a, extra_panels=[b.genes[:3]], out_path=tmp_path / "n.json")
    assert res.genes == b.genes[:3]
    assert res.counts["bulk_and_cells"] == 5 and (tmp_path / "n.json").exists()
