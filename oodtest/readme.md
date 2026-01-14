# oodtest

This repository contains experiments for **Out-of-Distribution (OOD) robustness** and **uncertainty quantification** under the datasets like Cora/Citeseer/PubMed.

---

## 📂 Dataset Selection

All experiments can be run on **Cora**, **Citeseer**, or **Pubmed**.

The experiment will automatically download the dataset.

To change the dataset, edit the following line inside each Python file:

```python
dataset_name = "cora"   # or "citeseer", "pubmed"
```
---

## ⚙️ Key Hyperparameters

* `eta_x` : learning rate for **Predictive Coding (PC) dynamics**
* `eta_p` : learning rate for the **main network parameters**
* `random_seeds` : list of random seeds for reproducibility
* `T` : number of time steps
* `pc_infer_iters` or `K` : number of inference steps in Predictive Coding

---

## 🚀 Running Standard Experiments

Run the following scripts from the `oodtest/` directory:

### GCN + Spiking + Predictive Coding

```bash
python test_gcn_ooddetection_final.py
```

### Plain GCN baseline

```bash
python gcn_ooddetection_final.py
```

---

## 🔬 Advanced Experiments

### 1. Ablation Studies

Navigate to the `ablation/` folder:

```bash
cd ablation
# Example
python ablation_gcn_nsppc.py
```

### 2. G-ΔUQ (Anchoring-based)

Navigate to the `gduq/` folder:

```bash
cd gduq
# Example
python gcn_gduq_final.py
```

### 3. Post-hoc Calibration

Navigate to the `posthoc/` folder:

```bash
cd posthoc
# Example
python gcn_posthoc.py --posthoc ts
```
### 4. Parameter sensitivity analysis

Navigate to the `hpyer-se/` folder:

```bash
cd hyper-se
# Example
python sweep_and_plot.py
```
### 5. Predictive Coding Explainability Experiments

Navigate to the `pc-ex/` folder:

```bash
cd pc-ex
# Example
python ood-vis.py
```
Each folder contains usage examples named "eg.txt" for running the respective experiments.

---

## 📊 Notes

* Each experiment reports standard metrics such as **Accuracy, NLL, Brier score, ECE, AUROC**, and computational cost.
* You may need to install the dependencies listed in `requirements.txt` before running experiments.

---
