"""DEGAS networks, WGAN-GP transfer training step, prediction, and checkpoints.

Architecture and optimisation follow DEGASv2: a three-layer sigmoid feature extractor with
dense skip connections, a Wasserstein discriminator trained with gradient penalty, an
outcome head on the patient embedding and, for ``Class*`` models, a cell-label head on the
cell embedding. Constructors never touch disk and never read global device state.

A predicted score is the outcome head output applied to a cell embedding: softmax
probability of class 1 for binary outcomes, a sigmoid hazard for Cox. It is a model
output, not a calibrated probability.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad

from degas3.core.losses import MMD_loss, NegativeLogLikelihood, rank_loss
from degas3.core.sampling import CellBatchSampler, PatientBatchSampler

log = logging.getLogger(__name__)


def resolve_device(device: str | torch.device = "auto", allow_cpu: bool = False) -> torch.device:
    """Explicit device choice. ``auto`` uses CUDA when available, else CPU only if ``allow_cpu``."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if allow_cpu:
            log.warning("CUDA is unavailable; training on CPU is slow (identical math).")
            return torch.device("cpu")
        raise ValueError(
            "No CUDA device found. Run on a GPU node, or pass allow_cpu=True / --allow-cpu to train on CPU."
        )
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {device!r} requested but CUDA is unavailable; use 'cpu' with allow_cpu.")
    return dev


def set_seeds(seed: int) -> None:
    """Seed python, numpy, torch (CPU and CUDA) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -- layers ------------------------------------------------------------------
def xavier_init(m: nn.Module, seed: int) -> None:
    """Xavier-uniform weights and zero bias for Linear layers, re-seeding torch first (DEGASv2)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def define_linear(input_shape, output_shape, dropout_rate=0.5, activation=nn.Sigmoid(), seed=42):
    """Dropout -> Linear -> BatchNorm -> activation block; a bare Linear when ``activation`` is None."""
    if activation is None:
        layer = nn.Linear(input_shape, output_shape, bias=True)
    elif dropout_rate > 0.0:
        layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_shape, output_shape, bias=True),
            nn.BatchNorm1d(output_shape),
            activation,
        )
    else:
        layer = nn.Sequential(nn.Linear(input_shape, output_shape, bias=True), activation)
    layer.apply(lambda x: xavier_init(x, seed))
    return layer


class FeatureExtractor(nn.Module):
    """Three dense-skip sigmoid layers; returns (second, third) embeddings."""

    def __init__(self, opt: dict):
        super().__init__()
        d, f, r, s = opt["input_shape"], opt["feature_dim"], opt["dropout_rate"], opt["seed"]
        self.emb1_layer = define_linear(d, f, r, nn.Sigmoid(), s)
        self.emb2_layer = define_linear(f + d, f, r, nn.Sigmoid(), s + 1)
        self.emb3_layer = define_linear(f * 2 + d, f, r, nn.Sigmoid(), s + 2)

    def forward(self, x):
        emb1 = self.emb1_layer(x)
        emb2 = self.emb2_layer(torch.cat((x, emb1), dim=1))
        emb3 = self.emb3_layer(torch.cat((x, emb1, emb2), dim=1))
        return emb2, emb3


class Discriminator(nn.Module):
    """Wasserstein critic on the third embedding."""

    def __init__(self, opt: dict):
        super().__init__()
        f, r, s = opt["feature_dim"], opt["dropout_rate"], opt["seed"]
        self.model = nn.Sequential(
            define_linear(f, f // 2, r, nn.LeakyReLU(0.2, inplace=False), s),
            define_linear(f // 2, f // 2, r, nn.LeakyReLU(0.2, inplace=False), s + 1),
            define_linear(f // 2, 1, 0.0, None, s + 2),
        )

    def forward(self, x):
        return self.model(x)


# -- model ---------------------------------------------------------------------
class DegasModel:
    """One of BlankClass, ClassClass, BlankCox, ClassCox (``opt['model_type']``).

    "High resolution" is the cell side, "low resolution" the patient side, as in DEGASv2.
    """

    def __init__(self, opt: dict, device: str | torch.device = "cpu"):
        mt = opt["model_type"]
        if mt not in ("BlankClass", "ClassClass", "BlankCox", "ClassCox"):
            raise ValueError(f"unknown model_type {mt!r}; use BlankClass, ClassClass, BlankCox or ClassCox.")
        self.opt = dict(opt)
        for key, value in opt.items():
            setattr(self, key, value)
        self.device = torch.device(device)
        self.cox = "Cox" in mt
        self.has_cell_head = mt.startswith("Class")
        if self.has_cell_head and opt["high_reso_output_shape"] < 1:
            raise ValueError(
                "Class* models need high_reso_output_shape = number of cell classes; set it in the options."
            )
        f, r = opt["feature_dim"], opt["dropout_rate"]

        self.feature_extractor_layer = FeatureExtractor(opt).to(self.device)
        self.discrim_layer = Discriminator(opt).to(self.device)
        if self.cox:
            self.low_reso_pred_layer = define_linear(f, opt["low_reso_output_shape"], r, nn.Sigmoid())
        else:
            self.low_reso_pred_layer = define_linear(f, opt["low_reso_output_shape"], activation=None)
        self.low_reso_pred_layer.to(self.device)
        self.net_name_list = ["feature_extractor_layer", "discrim_layer", "low_reso_pred_layer"]
        heads = [self.feature_extractor_layer, self.low_reso_pred_layer]
        if self.has_cell_head:
            self.high_reso_pred_layer = define_linear(f, opt["high_reso_output_shape"], activation=None).to(self.device)
            self.net_name_list.append("high_reso_pred_layer")
            heads.append(self.high_reso_pred_layer)
            self.criterion_high_reso = nn.CrossEntropyLoss().to(self.device)

        if self.cox:
            if opt["loss_type"] == "log_neg":
                self.criterion_low_reso = NegativeLogLikelihood(self.device)
            elif opt["loss_type"] == "rank_loss":
                self.criterion_low_reso = rank_loss
            else:
                raise ValueError("loss_type must be 'rank_loss' or 'log_neg' for Cox models.")
        else:
            self.criterion_low_reso = nn.CrossEntropyLoss().to(self.device)

        self.loss_names = (["high_reso_loss"] if self.has_cell_head else []) + ["low_reso_loss", "transfer_loss"]
        self._loss_rows: list[dict] = []
        params1 = [p for m in heads for p in m.parameters()]
        self.optimizer1 = torch.optim.Adam(params1, lr=opt["lr"], betas=(opt["beta1"], 0.999))
        self.optimizer2 = torch.optim.Adam(self.discrim_layer.parameters(), lr=opt["lr"], betas=(opt["beta1"], 0.999))

    # -- bookkeeping --------------------------------------------------------
    @property
    def loss_rec(self) -> pd.DataFrame:
        return pd.DataFrame(self._loss_rows, columns=self.loss_names)

    def set_required_grad(self, nets, requires_grad: bool = True) -> None:
        for net in nets if isinstance(nets, list) else [nets]:
            if net is not None:
                for p in net.parameters():
                    p.requires_grad = requires_grad

    def set_evaluate_mode(self) -> None:
        for name in self.net_name_list:
            getattr(self, name).eval()

    def set_train_mode(self) -> None:
        for name in self.net_name_list:
            getattr(self, name).train()

    # -- one optimisation step -------------------------------------------------
    def set_input(self, cell_batch: dict, patient_batch: dict) -> None:
        dev = self.device
        self.high_reso_input = torch.as_tensor(cell_batch["data"], dtype=torch.float32, device=dev)
        self.low_reso_input = torch.as_tensor(patient_batch["data"], dtype=torch.float32, device=dev)
        if self.has_cell_head:
            self.high_reso_label = torch.as_tensor(cell_batch["label"], dtype=torch.long, device=dev)
        if self.cox:
            self.low_reso_time = torch.as_tensor(patient_batch["time"], dtype=torch.long, device=dev)
            self.low_reso_status = torch.as_tensor(patient_batch["status"], dtype=torch.float32, device=dev)
        else:
            self.low_reso_label = torch.as_tensor(patient_batch["label"], dtype=torch.long, device=dev)

    def forward(self) -> None:
        n_cells = self.high_reso_input.shape[0]
        _, self.emb3 = self.feature_extractor_layer(torch.cat((self.high_reso_input, self.low_reso_input), 0))
        self.high_reso_emb3 = self.emb3[:n_cells, ...]
        self.low_reso_emb3 = self.emb3[n_cells:, ...]
        self.low_low_pred = self.low_reso_pred_layer(self.low_reso_emb3)
        if self.has_cell_head:
            self.high_high_pred = self.high_reso_pred_layer(self.high_reso_emb3)

    def calculate_losses(self):
        if self.cox:
            self.low_reso_loss = self.criterion_low_reso(
                self.low_low_pred, self.low_reso_time.unsqueeze(dim=1),
                self.low_reso_status.unsqueeze(dim=1), self.device,
            )
        else:
            self.low_reso_loss = self.criterion_low_reso(self.low_low_pred, self.low_reso_label)
        total = self.lambda2 * self.low_reso_loss
        if self.has_cell_head:
            self.high_reso_loss = self.criterion_high_reso(self.high_high_pred, self.high_reso_label)
            total = self.lambda1 * self.high_reso_loss + total
        return total

    def _gradient_penalty(self, real_feat, fake_feat):
        n = min(real_feat.size()[0], fake_feat.size()[0])
        alpha = torch.rand(n, 1).to(self.device).expand_as(real_feat[:n, ...])
        interpolated = (alpha * real_feat[:n, ...] + (1 - alpha) * fake_feat[:n, ...]).detach()
        interpolated.requires_grad_(True)
        prob = self.discrim_layer(interpolated)
        grads = torch_grad(outputs=prob, inputs=interpolated,
                           grad_outputs=torch.ones(prob.size()).to(self.device),
                           create_graph=True, retain_graph=True)[0].view(n, -1)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def backward_side(self) -> None:
        """Critic update (Wasserstein + gradient penalty)."""
        self.set_required_grad(self.discrim_layer, True)
        self.optimizer2.zero_grad()
        real = self.low_reso_emb3.data   # patient embedding is the "real" side
        fake = self.high_reso_emb3.data  # cell embedding is pushed toward it
        self.transfer_loss = -torch.mean(self.discrim_layer(real)) + torch.mean(self.discrim_layer(fake))
        self.transfer_loss += self.gp_weight * self._gradient_penalty(self.high_reso_emb3.data, self.low_reso_emb3.data)
        self.transfer_loss.backward()
        self.optimizer2.step()
        self.set_required_grad(self.discrim_layer, False)

    def backward_main(self) -> None:
        """Encoder and head update."""
        self.optimizer1.zero_grad()
        self.loss = self.calculate_losses()
        if self.transfer_type == "Wasserstein":
            self.transfer_loss = -torch.mean(self.discrim_layer(self.high_reso_emb3))
        elif self.transfer_type == "MMD":
            self.transfer_loss = MMD_loss(self.high_reso_emb3, self.low_reso_emb3)
        else:
            raise ValueError("transfer_type must be 'Wasserstein' or 'MMD'.")
        self.loss += self.lambda3 * self.transfer_loss
        self._loss_rows.append({name: float(getattr(self, name).detach()) for name in self.loss_names})
        self.loss.backward()
        self.optimizer1.step()

    def optimize_parameters(self, step: int) -> None:
        self.forward()
        self.backward_main()
        if self.transfer_type == "Wasserstein" and (step % self.n_critic == 0):
            self.forward()
            self.backward_side()


def build_model(opt: dict, device: str | torch.device = "cpu") -> DegasModel:
    """Construct a model from an option dict (``input_shape`` must be the gene count)."""
    return DegasModel(opt, device)


# -- training and prediction -------------------------------------------------------
def fit(
    model: DegasModel,
    bulk_X: np.ndarray,
    bulk_y: np.ndarray,
    cell_X: np.ndarray,
    cell_y: Optional[np.ndarray] = None,
    iterations: Optional[int] = None,
    seed: Optional[int] = None,
    *,
    batch_size: Optional[int] = None,
    pat_batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """Train ``model`` for ``iterations`` paired balanced batches; return the per-step loss table.

    ``bulk_X`` and ``cell_X`` are preprocessed (n, genes) arrays over the same genes.
    ``bulk_y`` is 1-D class codes (binary) or an (n, 2) array of time and event (Cox).
    ``cell_y`` holds integer cell classes and is required for ``Class*`` models.
    Raises ValueError when any recorded loss is not finite.
    """
    if model.has_cell_head and cell_y is None:
        raise ValueError("Class* models need cell_y (integer cluster codes); use a Blank* model otherwise.")
    iterations = int(model.opt["tot_iters"] if iterations is None else iterations)
    seed = int(model.opt["seed"] if seed is None else seed)
    bulk_X = np.asarray(bulk_X, dtype=np.float32)
    cell_X = np.asarray(cell_X, dtype=np.float32)
    if bulk_X.shape[1] != cell_X.shape[1]:
        raise ValueError(f"bulk has {bulk_X.shape[1]} genes but cells have {cell_X.shape[1]}; use one gene list.")
    set_seeds(seed)
    cells = CellBatchSampler(cell_X, cell_y if model.has_cell_head else None, seed=seed,
                             batch_size=batch_size or model.opt["batch_size"])
    patients = PatientBatchSampler(bulk_X, bulk_y, seed=seed, model_type=model.opt["model_type"],
                                   batch_size=pat_batch_size or model.opt["pat_batch_size"])
    model.set_train_mode()
    for step in range(1, iterations + 1):
        model.set_input(cells.batch(step - 1), patients.batch(step - 1))
        model.optimize_parameters(step)
    losses = model.loss_rec
    if not np.isfinite(losses.to_numpy()).all():
        raise ValueError("non-finite training loss; lower the learning rate or check the inputs for NaN/constant genes.")
    return losses


def predict_scores(model: DegasModel, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Outcome-head output for each row of ``X``: softmax P(class 1) (binary) or sigmoid hazard (Cox).

    Uses the feature extractor and outcome head only (no discriminator, no cell head).
    Values lie in [0, 1] but are not calibrated probabilities.
    """
    out = np.empty(len(X), dtype=np.float32)
    model.set_evaluate_mode()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(np.asarray(X[start:start + batch_size]), dtype=torch.float32, device=model.device)
            _, emb = model.feature_extractor_layer(x)
            pred = model.low_reso_pred_layer(emb)
            val = pred.squeeze(1) if model.cox else torch.softmax(pred, dim=1)[:, 1]
            out[start:start + len(x)] = val.cpu().numpy()
    if not np.isfinite(out).all() or (out < 0).any() or (out > 1).any():
        raise ValueError("invalid predictions (NaN or outside [0, 1]); check the inputs and training losses.")
    return out


# -- checkpoints -------------------------------------------------------------------
def save_checkpoint(model: DegasModel, path: str | Path, *, genes: list[str], seed: int,
                    size: Optional[int] = None, fold: Optional[int] = None,
                    extra: Optional[dict] = None) -> Path:
    """Save the inference networks (extractor, outcome head, optional cell head) as CPU tensors.

    The discriminator is not saved; it is only needed for training.
    """
    cpu = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}  # noqa: E731
    ckpt: dict[str, Any] = {
        "feature_extractor": cpu(model.feature_extractor_layer),
        "outcome_head": cpu(model.low_reso_pred_layer),
        "opt": dict(model.opt),
        "genes": list(genes),
        "outcome_type": "survival" if model.cox else "binary",
        "model_type": model.opt["model_type"],
        "seed": int(seed),
        "size": None if size is None else int(size),
        "fold": None if fold is None else int(fold),
    }
    if model.has_cell_head:
        ckpt["cell_head"] = cpu(model.high_reso_pred_layer)
    if extra:
        ckpt["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[DegasModel, dict]:
    """Rebuild a model from a checkpoint (loadable without CUDA); returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if len(ckpt["genes"]) != ckpt["opt"]["input_shape"]:
        raise ValueError("checkpoint gene list and input_shape disagree; the file is corrupt or from another version.")
    model = build_model(ckpt["opt"], device)
    model.feature_extractor_layer.load_state_dict(ckpt["feature_extractor"])
    model.low_reso_pred_layer.load_state_dict(ckpt["outcome_head"])
    if model.has_cell_head:
        model.high_reso_pred_layer.load_state_dict(ckpt["cell_head"])
    model.set_evaluate_mode()
    return model, ckpt
