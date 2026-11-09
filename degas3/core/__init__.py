"""Vendored DEGAS model code, cleaned for DEGASv3.

CREDITS
-------
This package is a cleaned port of the DEGASv2 PyTorch backend written by
Ziyu Liu, Travis S. Johnson, Sihong Li, and Jiahui Liu. The method is described in:

    Johnson TS, et al. "Diagnostic Evidence GAuge of Single cells (DEGAS): a flexible
    deep transfer learning framework for prioritizing cells in relation to disease."
    Genome Medicine 2022. doi:10.1186/s13073-022-01012-2

Architecture, losses, WGAN-GP transfer, Adam settings, balance sampling, and
preprocessing are unchanged; logging, explicit devices, and side-effect-free
constructors are the differences.

Importing this package does not import torch; ``degas3.core.preprocess`` needs numpy only.
"""

CREDITS = (
    "DEGASv2 backend: Ziyu Liu, Travis S. Johnson, Sihong Li, Jiahui Liu. "
    "Johnson et al., Genome Medicine 2022, doi:10.1186/s13073-022-01012-2."
)

from degas3.core.preprocess import normalize_scale, preprocess_counts  # noqa: E402,F401
