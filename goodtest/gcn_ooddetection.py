import os
import json
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import Accuracy
import snntorch as snn
from torch_geometric.nn import GCNConv
import PredictiveCoding.predictive_coding as pc
from sklearn.metrics import roc_auc_score, log_loss
from torchmetrics.classification import MulticlassCalibrationError

# Hyperparameters
T = 25
beta = 0.9
pc_infer_iters = 20
eta_x = 0.01
eta_p = 0.001
hidden_dims = [128, 128, 64]
epochs = 1000
patience = 200

# GCN layer
class StandardGCNLayer(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.conv = GCNConv(in_dim, hid_dim)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        return F.relu(h)

# Multi-layer GCN
class StandardGNN(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, edge_index):
        super().__init__()
        self.edge_index = edge_index
        dims = [in_dim] + hidden_dims
        self.layers = nn.ModuleList([
            StandardGCNLayer(dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        ])
        self.classifier = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.edge_index)
        return self.classifier(x)

# GOOD dataset loader
def load_structured_dataset_good(dataset_name, shift_type, domain=None, root="datasets"):
    mapping = {
        "cora": ("GOOD.data.good_datasets.good_cora", "GOODCora", "degree"),
        "arxiv": ("GOOD.data.good_datasets.good_arxiv", "GOODArxiv", "time"),
        "webkb": ("GOOD.data.good_datasets.good_webkb", "GOODWebKB", "university"),
        "twitch": ("GOOD.data.good_datasets.good_twitch", "GOODTwitch", "language"),
        "cbas":   ("GOOD.data.good_datasets.good_cbas",   "GOODCBAS",   "color"),
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {list(mapping.keys())}")

    module_path, class_name, default_domain = mapping[dataset_name]
    domain = domain or default_domain
    print(f"[Info] Dataset: {dataset_name}, Domain: {domain}, Shift: {shift_type}")

    mod = __import__(module_path, fromlist=[class_name])
    dataset_class = getattr(mod, class_name)
    data = dataset_class(root=root, domain=domain, shift=shift_type)[0]

    # unify mask names
    data.valid_in_mask = getattr(data, "val_mask", None)
    data.test_in_mask = getattr(data, "id_test_mask", None)
    data.test_out_mask = getattr(data, "test_mask", None)
    return data

# Helper functions
def get_multiclass_index_labels(y, C):
    if y.ndim > 1 and y.shape[1] > 1:
        y_idx = y.argmax(dim=1).long()
    else:
        y_idx = y.long()
    invalid = (y_idx < 0) | (y_idx >= C)
    return y_idx, invalid

def make_onehot(y_idx, C, device):
    safe_idx = torch.where((y_idx >= 0) & (y_idx < C), y_idx, torch.zeros_like(y_idx))
    y_oh = F.one_hot(safe_idx, C).float().to(device)
    return y_oh

# Temperature scaling for calibration
def temperature_scale(logits, labels, T_init=1.0, max_iter=200, lr=0.1, patience=5):
    Tt = torch.tensor([T_init], requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([Tt], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_T = Tt.clone().detach()
    no_improve = 0

    def eval_closure():
        nonlocal best_loss, best_T, no_improve
        optimizer.zero_grad()
        scaled = logits / Tt.clamp(min=1e-3, max=1e3)
        loss = loss_fn(scaled, labels)
        loss.backward()
        if loss.item() + 1e-6 < best_loss:
            best_loss = loss.item()
            best_T = Tt.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
        return loss

    for _ in range(3):
        optimizer.step(eval_closure)
        if no_improve >= patience:
            break
    return best_T.clamp(min=1e-2, max=100)

# Uncertainty metrics (NLL, Brier, ECE)
@torch.no_grad()
def compute_uncertainty_metrics(model, x, data, mask, num_classes, task_type, T_cal=None):
    logits = model(x)
    val_mask = data.valid_in_mask if data.valid_in_mask is not None else data.test_in_mask

    if task_type == "Binary classification":
        labels_all = data.y.view(-1).long()
    else:
        labels_all, _ = get_multiclass_index_labels(data.y, num_classes)

    if T_cal is None:
        T_cal = temperature_scale(logits[val_mask], labels_all[val_mask])

    calibrated_logits = logits / T_cal
    probs = F.softmax(calibrated_logits, dim=1)

    region_labels = labels_all[mask]
    region_probs = probs[mask]

    try:
        y_np = region_labels.cpu().numpy()
        p_np = region_probs.cpu().numpy()
        nll = log_loss(y_np, p_np, labels=list(range(num_classes)))
    except ValueError:
        nll = float('nan')

    y_onehot = F.one_hot(region_labels, num_classes).float()
    brier = torch.mean(torch.sum((region_probs - y_onehot) ** 2, dim=1)).item()

    if task_type == "Binary classification":
        conf, pred = region_probs.max(dim=1)
        correct = (pred == region_labels).float()
        n_bins = 15
        bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.tensor(0.0, device=probs.device)
        for i in range(n_bins):
            m = (conf >= bins[i]) & (conf < bins[i+1])
            if m.any():
                acc_bin = correct[m].mean()
                conf_bin = conf[m].mean()
                ece += (m.float().mean()) * torch.abs(acc_bin - conf_bin)
        ece = ece.item()
    else:
        ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1').to(probs.device)
        ece = ece_metric(region_probs, region_labels).item()

    return {"nll": float(nll), "brier": float(brier), "ece": float(ece), "temperature": float(T_cal.item())}

# Uncorr AUROC
@torch.no_grad()
def compute_uncorr_auroc(logits, labels, T=None, score_type='entropy'):
    if T is not None:
        logits = logits / (T if isinstance(T, torch.Tensor) else float(T))
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    correct_mask = (preds == labels).cpu().numpy()

    if score_type == 'msp':
        uncertainty = (1.0 - probs.max(dim=1).values).cpu().numpy()
    elif score_type == 'energy':
        uncertainty = (-torch.logsumexp(logits, dim=1)).cpu().numpy()
    else:  # entropy
        uncertainty = (-probs * probs.clamp_min(1e-12).log()).sum(dim=1).cpu().numpy()

    y_binary = 1 - correct_mask.astype(np.int32)
    if y_binary.sum() == 0 or y_binary.sum() == y_binary.size:
        return float('nan')
    return float(roc_auc_score(y_binary, uncertainty))

# Accuracy evaluation
@torch.no_grad()
def evaluate_accuracy(model, x, data, mask, task_type, C):
    out = model(x)
    if mask is None or mask.sum() == 0:
        return 0.0
    if task_type == "Binary classification":
        y_idx = data.y.view(-1).long()
    else:
        y_idx, invalid = get_multiclass_index_labels(data.y, C)
        mask = mask & (~invalid)
        if mask.sum() == 0:
            return 0.0
    metric_fn = Accuracy(task='multiclass', num_classes=C).to(out.device)
    return metric_fn(out[mask], y_idx[mask]).item()

def clean_state_dict(state_dict):
    return {k: v for k, v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params")}

# Main script
if __name__ == "__main__":
    ...
    # (rest of your code unchanged, only comments simplified)
