import os
import json
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import Accuracy, BinaryAUROC
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_auc_score, log_loss
from torchmetrics.classification import MulticlassCalibrationError

# ─── Hyperparameters ───
hidden_dims = [128, 128, 64]
epochs = 1000
patience = 200
gat_heads = 4  # GAT attention heads
learning_rate = 0.0005

# ─── GAT Layer ───
class GATLayer(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.conv = GATConv(in_dim, hid_dim // gat_heads, heads=gat_heads, concat=True)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        return h

# ─── GAT Model ───
class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, edge_index):
        super().__init__()
        self.edge_index = edge_index
        dims = [in_dim] + hidden_dims
        self.layers = nn.ModuleList([
            GATLayer(dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        ])
        self.classifier = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x, self.edge_index))
        return self.classifier(x)

# ─── GOOD Dataset Loader ───
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

# ─── Helpers  ───
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

# ─── Calibration & Uncertainty ───
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

@torch.no_grad()
def compute_uncertainty_metrics(model, x, data, mask, num_classes, task_type, T_cal=None):
    logits = model(x)

    # Choose calibration set: use validation in-distribution
    val_mask = data.valid_in_mask
    if val_mask is None or val_mask.sum() == 0:
        # fallback: use ID test mask for T (rare)
        val_mask = data.test_in_mask

    # labels for calibration must be class indices
    if task_type == "Binary classification":
        labels_all = data.y.view(-1).long()
    else:
        labels_all, _ = get_multiclass_index_labels(data.y, num_classes)

    # learn temperature if not provided
    if T_cal is None:
        T_cal = temperature_scale(logits[val_mask], labels_all[val_mask])

    calibrated_logits = logits / T_cal
    probs = F.softmax(calibrated_logits, dim=1)

    # Select region (ID or OOD) via mask
    region_labels = labels_all[mask]
    region_probs = probs[mask]

    # NLL
    try:
        y_np = region_labels.cpu().numpy()
        p_np = region_probs.cpu().numpy()
        nll = log_loss(y_np, p_np, labels=list(range(num_classes)))
    except ValueError:
        nll = float('nan')

    # Brier
    y_onehot = F.one_hot(region_labels, num_classes).float()
    brier = torch.mean(torch.sum((region_probs - y_onehot) ** 2, dim=1)).item()

    # ECE
    if task_type == "Binary classification":
        # manual binary ECE
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

    y_binary = 1 - correct_mask.astype(np.int32)  # miscls=1, correct=0
    if y_binary.sum() == 0 or y_binary.sum() == y_binary.size:
        return float('nan')
    return float(roc_auc_score(y_binary, uncertainty))

# ─── Evaluation helpers ───
@torch.no_grad()
def evaluate_accuracy(model, x, data, mask, task_type, C):
    out = model(x)
    if mask is None or mask.sum() == 0:
        return 0.0
    # For both binary (C=2) and multiclass, compute standard accuracy
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

# ─── Main ───
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="cora / arxiv / webkb / twitch/cbas")
    parser.add_argument("--domain", type=str, default=None, help="optional domain, e.g. webkb: university; twitch: language; cbas: color")
    parser.add_argument("--shift", type=str, default="concept", help="no_shift / covariate / concept")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    args = parser.parse_args()

    seeds = [0, 1, 2, 3, 4]

    # aggregate across seeds
    all_id_acc, all_ood_acc = [], []
    all_id_nll, all_id_brier, all_id_ece = [], [], []
    all_ood_nll, all_ood_brier, all_ood_ece = [], [], []
    all_temps, all_id_uncorr, all_ood_uncorr = [], [], []
    #合并(ID+OOD)后的 uncorr AUROC 统计
    all_combined_uncorr = []

    for seed in seeds:
        print(f"\n[Info] Running seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = load_structured_dataset_good(args.dataset, args.shift, args.domain, args.dataset_dir).to(device)

        dataset_name = args.dataset.lower()
        if dataset_name == "twitch":
            task_type = "Binary classification"
        else:
            task_type = data.task if hasattr(data, "task") else "Multi-class classification"
        print(f"[Info] Task: {task_type}")

        # determine C & labels format
        if task_type == "Binary classification":
            C = 2
            data.y = data.y.squeeze(1).long() if data.y.ndim > 1 else data.y.long()
            invalid = torch.zeros_like(data.y, dtype=torch.bool)
        else:
            if data.y.ndim == 1:
                C = int(data.y.max().item()) + 1 if data.y.numel() > 0 else 0
            else:
                C = data.y.shape[1]
            if C <= 1:
                raise ValueError(f"[Error] Invalid number of classes C={C} for multi-class task")
            # 提前算好 index 标签与 invalid
            y_idx_all, invalid = get_multiclass_index_labels(data.y, C)

        # 使用原始特征，不需要脉冲转换
        x = data.x

        # training targets/masks
        if task_type == "Binary classification":
            y_idx = data.y.long().view(-1)
            train_mask_valid = data.train_mask
        else:
            y_idx = y_idx_all
            train_mask_valid = data.train_mask & (~invalid)
            y_onehot = make_onehot(y_idx, C, device)

        model = GATModel(data.num_node_features, hidden_dims, C, data.edge_index).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # ====== New: best/日志命名（参照原脚本风格） ======
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"GAT_ablation_{args.dataset}_{args.shift}_seed{seed}_{timestamp}"
        best_ckpt_path = f"best_weights_{tag}.pt"
        best_model_path = f"best_model_{tag}.pt"
        cost_log_path = f"cost_log_{tag}.json"

        cost_log = {
            "dataset": args.dataset,
            "domain": args.domain,
            "shift": args.shift,
            "seed": seed,
            "parameters": {
                "hidden_dims": hidden_dims,
                "epochs": epochs,
                "patience": patience,
                "gat_heads": gat_heads,
                "learning_rate": learning_rate
            },
            "training": []
        }

        # ====== Early-Stopping Vars ======
        best_val_score = float("-inf")
        best_epoch = -1
        counter = 0

        print(f">>> Seed {seed}: GAT Ablation Training started...")

        for epoch in range(1, epochs + 1):
            model.train()
            start_t = time.time()

            # 标准训练流程
            optimizer.zero_grad()
            output = model(x)
            
            if task_type == "Binary classification":
                loss = F.cross_entropy(output[train_mask_valid], y_idx[train_mask_valid])
            else:
                loss = F.cross_entropy(output[train_mask_valid], y_idx[train_mask_valid])
            
            loss.backward()
            optimizer.step()

            val_score = evaluate_accuracy(model, x, data, data.valid_in_mask, task_type, C)
            dur = time.time() - start_t

            # 10-epoch 打点打印
            if (epoch % 10) == 0:
                print(f"[seed {seed}] epoch {epoch:03d} | val_score={val_score:.4f} | time={dur:.2f}s")

            # —— 提升即即时落盘 —— 
            if (val_score > best_val_score) or (best_epoch < 0):
                best_val_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), best_ckpt_path)
                counter = 0
                print(f"[Best][seed {seed}] epoch {epoch} | val={val_score:.4f} | saved -> {best_ckpt_path}")
            else:
                counter += 1

            cost_log["training"].append({
                "epoch": epoch,
                "val_score": float(val_score),
                "time_sec": float(dur)
            })

            # 早停判定
            if counter >= patience:
                print(f"[seed {seed}] Early Stopping at epoch {epoch}. Best val={best_val_score:.4f} @ epoch {best_epoch}.")
                break

        # —— 评估前：确保载入"验证最优"的模型权重（从磁盘） —— 
        assert os.path.exists(best_ckpt_path), "Best checkpoint not found (no improvement saved?)."
        state_dict_best = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state_dict_best, strict=True)
        print(f"[Info][seed {seed}] Use best epoch {best_epoch} (val={best_val_score:.4f}) for evaluation")

        # 也可打包成更完整的 best_model（含优化器/epoch等）
        torch.save({
            "epoch": best_epoch,
            "model_state_dict": state_dict_best,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_score": best_val_score
        }, best_model_path)

        # collect logits once
        with torch.no_grad():
            logits_all = model(x)

        # accuracy metrics
        acc_id = evaluate_accuracy(model, x, data, data.test_in_mask, task_type, C)
        acc_ood = evaluate_accuracy(model, x, data, data.test_out_mask, task_type, C)

        # temperature from ID validation
        if task_type == "Binary classification":
            labels_all = data.y.view(-1).long()
        else:
            labels_all, _ = get_multiclass_index_labels(data.y, C)
        val_mask = data.valid_in_mask if data.valid_in_mask is not None else data.test_in_mask
        T_used = temperature_scale(logits_all[val_mask], labels_all[val_mask])

        # uncertainty metrics (ID/OOD)
        metrics_id = compute_uncertainty_metrics(model, x, data, data.test_in_mask, C, task_type, T_cal=T_used)
        metrics_ood = compute_uncertainty_metrics(model, x, data, data.test_out_mask, C, task_type, T_cal=T_used)

        # uncorr AUROC (ID/OOD)
        id_uncorr = compute_uncorr_auroc(logits_all[data.test_in_mask], labels_all[data.test_in_mask], T=T_used, score_type='entropy')
        ood_uncorr = compute_uncorr_auroc(logits_all[data.test_out_mask], labels_all[data.test_out_mask], T=T_used, score_type='entropy')

        # ：Combined(ID+OOD) —— 用"联合掩码（按位 OR）"，并在多分类时剔除 invalid
        combined_mask = (data.test_in_mask.bool() | data.test_out_mask.bool())
        if task_type != "Binary classification":
            # 只在多分类中可能有 invalid 标签（比如 -1），需要剔除
            _, invalid_all = get_multiclass_index_labels(data.y, C)
            combined_mask = combined_mask & (~invalid_all)
        combined_uncorr = compute_uncorr_auroc(
            logits_all[combined_mask],
            labels_all[combined_mask],
            T=T_used,
            score_type='entropy'
        )

        all_id_acc.append(acc_id); all_ood_acc.append(acc_ood)
        all_id_nll.append(metrics_id['nll']); all_ood_nll.append(metrics_ood['nll'])
        all_id_brier.append(metrics_id['brier']); all_ood_brier.append(metrics_ood['brier'])
        all_id_ece.append(metrics_id['ece']); all_ood_ece.append(metrics_ood['ece'])
        all_temps.append(metrics_id['temperature'])
        all_id_uncorr.append(id_uncorr); all_ood_uncorr.append(ood_uncorr)
        all_combined_uncorr.append(combined_uncorr)  

        # per-seed printout
        print("=== Seed {} Results ===".format(seed))
        print(f"ID Accuracy: {acc_id:.4f} | OOD Accuracy: {acc_ood:.4f} | Gap: {acc_id - acc_ood:.4f}")
        print(f"Temperature: {float(T_used.item()):.4f}")
        print(f"ID NLL: {metrics_id['nll']:.4f} | Brier: {metrics_id['brier']:.4f} | ECE: {metrics_id['ece']:.4f}")
        print(f"OOD NLL: {metrics_ood['nll']:.4f} | Brier: {metrics_ood['brier']:.4f} | ECE: {metrics_ood['ece']:.4f}")
        print(f"Uncorr AUROC (ID): {id_uncorr:.4f} | (OOD): {ood_uncorr:.4f}")
        print(f"Uncorr AUROC (ID+OOD Combined): {combined_uncorr:.4f}")  

        # 写入本 seed 的 cost_log
        cost_log["test_results"] = {
            "id_acc": acc_id,
            "id_nll": metrics_id['nll'],
            "id_brier": metrics_id['brier'],
            "id_ece": metrics_id['ece'],
            "ood_acc": acc_ood,
            "ood_nll": metrics_ood['nll'],
            "ood_brier": metrics_ood['brier'],
            "ood_ece": metrics_ood['ece'],
            "temperature": float(T_used.item()),
            "uncorr_auroc_id": id_uncorr,
            "uncorr_auroc_ood": ood_uncorr,
            "uncorr_auroc_combined": combined_uncorr,  
            "best_epoch": best_epoch,
            "best_val_score": best_val_score
        }
        with open(cost_log_path, "w") as f:
            json.dump(cost_log, f, indent=2)
        print(f"[seed {seed}] Log saved to: {cost_log_path}")
        print(f"[seed {seed}] Best checkpoint: {best_ckpt_path}")
        print(f"[seed {seed}] Best model pack: {best_model_path}")

    # ====== Final summary across seeds ======
    def mean_std(arr):
        arr = np.array(arr, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    print("\n" + "="*50)
    print("========== GAT Ablation Final Summary Across Seeds ==========")
    print("="*50)

    id_mean, id_std = mean_std(all_id_acc)
    ood_mean, ood_std = mean_std(all_ood_acc)
    gap_mean, gap_std = mean_std(np.array(all_id_acc) - np.array(all_ood_acc))

    print("=== In-Distribution (ID) ===")
    print(f"Accuracy:  {id_mean:.4f} ± {id_std:.4f}")
    print(f"NLL:       {mean_std(all_id_nll)[0]:.4f} ± {mean_std(all_id_nll)[1]:.4f}")
    print(f"Brier:     {mean_std(all_id_brier)[0]:.4f} ± {mean_std(all_id_brier)[1]:.4f}")
    print(f"ECE:       {mean_std(all_id_ece)[0]:.4f} ± {mean_std(all_id_ece)[1]:.4f}")
    print(f"Temp:      {mean_std(all_temps)[0]:.4f} ± {mean_std(all_temps)[1]:.4f}")
    print(f"Uncorr AUC:{mean_std(all_id_uncorr)[0]:.4f} ± {mean_std(all_id_uncorr)[1]:.4f}")

    print("=== Out-of-Distribution (OOD) ===")
    print(f"Accuracy:  {ood_mean:.4f} ± {ood_std:.4f}")
    print(f"NLL:       {mean_std(all_ood_nll)[0]:.4f} ± {mean_std(all_ood_nll)[1]:.4f}")
    print(f"Brier:     {mean_std(all_ood_brier)[0]:.4f} ± {mean_std(all_ood_brier)[1]:.4f}")
    print(f"ECE:       {mean_std(all_ood_ece)[0]:.4f} ± {mean_std(all_ood_ece)[1]:.4f}")
    print(f"Uncorr AUC:{mean_std(all_ood_uncorr)[0]:.4f} ± {mean_std(all_ood_uncorr)[1]:.4f}")

    # ★ 新增：Combined(ID+OOD)
    print("=== Combined (ID + OOD) Correct-vs-Wrong ===")
    print(f"Uncorr AUC:{mean_std(all_combined_uncorr)[0]:.4f} ± {mean_std(all_combined_uncorr)[1]:.4f}")

    print("=== Performance Gap (ID - OOD) ===")
    print(f"Accuracy Gap: {gap_mean:.4f} ± {gap_std:.4f}")

    # dump JSON summary（保持原有结构，新增 combined 字段）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"gat_ablation_results_{args.dataset}_{args.shift}_metrics_{timestamp}.json"
    summary = {
        "dataset": args.dataset,
        "domain": args.domain,
        "shift": args.shift,
        "seeds": seeds,
        "metrics": {
            "id": {
                "acc": all_id_acc,
                "nll": all_id_nll,
                "brier": all_id_brier,
                "ece": all_id_ece,
                "uncorr_auroc": all_id_uncorr,
            },
            "ood": {
                "acc": all_ood_acc,
                "nll": all_ood_nll,
                "brier": all_ood_brier,
                "ece": all_ood_ece,
                "uncorr_auroc": all_ood_uncorr,
            },
            "combined": { 
                "uncorr_auroc": all_combined_uncorr
            },
            "temperature": all_temps,
        }
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Saved] Summary JSON -> {out_path}")
