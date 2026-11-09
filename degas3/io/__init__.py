"""Input loaders: bulk expression + phenotype, single-cell, and spatial data."""

from degas3.io.bulk import (  # noqa: F401
    load_bulk, load_bulk_npz, save_bulk_npz, strip_ensembl_version,
)
from degas3.io.sc import (  # noqa: F401
    GeneIntersection, get_counts, intersect_genes, load_cells,
)
from degas3.io.spatial import load_visium, load_xenium  # noqa: F401
