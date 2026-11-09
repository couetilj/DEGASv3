"""Model option dictionaries (DEGASv2 values). ``tot_iters`` etc. are overridable per run."""

from __future__ import annotations

from typing import Any

_COMMON: dict[str, Any] = {
    "input_shape": 25, "feature_dim": 25, "dropout_rate": 0.5, "lr": 0.01, "beta1": 0.9,
    "transfer_type": "Wasserstein", "lambda2": 3.0, "lambda3": 3.0, "gp_weight": 10.0,
    "n_critic": 2, "tot_iters": 300, "batch_size": 200, "pat_batch_size": 200,
    "sample_method": "balance", "fold": -1, "seed": 0, "tot_folds": 1,
}

BlankClass_opt: dict[str, Any] = {
    **_COMMON, "model_type": "BlankClass", "loss_type": "cross_entropy",
    "low_reso_output_shape": 2, "high_reso_output_shape": 1,  # high_reso unused by Blank*
}
ClassClass_opt: dict[str, Any] = {
    **_COMMON, "model_type": "ClassClass", "loss_type": "cross_entropy",
    "low_reso_output_shape": 2, "high_reso_output_shape": -1, "lambda1": 1.0,
}
BlankCox_opt: dict[str, Any] = {
    **_COMMON, "model_type": "BlankCox", "loss_type": "log_neg",
    "low_reso_output_shape": 1, "high_reso_output_shape": 1,
}
ClassCox_opt: dict[str, Any] = {
    **_COMMON, "model_type": "ClassCox", "loss_type": "log_neg",
    "low_reso_output_shape": 1, "high_reso_output_shape": -1, "lambda1": 1.0,
}

MODEL_TYPES = {
    "BlankClass": BlankClass_opt, "ClassClass": ClassClass_opt,
    "BlankCox": BlankCox_opt, "ClassCox": ClassCox_opt,
}


def make_options(outcome_type: str, with_cell_labels: bool, **overrides: Any) -> dict[str, Any]:
    """Option dict for ``outcome_type`` in {"binary","survival"}; Class* when ``with_cell_labels``.

    Class* models need ``high_reso_output_shape`` (number of cell classes) in ``overrides``.
    """
    if outcome_type not in ("binary", "survival"):
        raise ValueError(f"outcome_type must be 'binary' or 'survival', got {outcome_type!r}.")
    name = ("Class" if with_cell_labels else "Blank") + ("Class" if outcome_type == "binary" else "Cox")
    opt = dict(MODEL_TYPES[name])
    opt.update(overrides)
    return opt
