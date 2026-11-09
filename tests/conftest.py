import pytest

from degas3.simulate import simulate


@pytest.fixture(scope="session")
def sim_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    simulate(out, seed=1, n_patients=60, n_cells=800, n_genes=400, n_donors=8)
    return out
