# SGPC Experiments

This repository provides experiments for **SGPC (Spiking Graph Predictive Coding)** models, focusing on both **Out-of-Distribution (OOD) robustness** and **OOD detection** across multiple benchmark datasets.

## Structure

The experiments are divided into two main parts based on the datasets used:

* **`oodtest/`** – Contains experiments for the **Cora, Citeseer, and PubMed** datasets. These experiments evaluate OOD robustness and detection under various settings.
* **`goodtest/`** – Contains experiments for the **Twitch and CBAS** datasets from the **GOOD benchmark**. These experiments also use the SGPC model for evaluation.

Both parts follow the same experimental framework, with dataset-specific details and instructions.

## Environment

To reproduce the experiments, please use the following environment setup:

* **Python**: 3.8
* **PyTorch**: 2.0.0
* **CUDA**: 11.8
* Additional dependencies: see **`requirements.txt`** for the complete list.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

1. Navigate to the corresponding folder depending on the dataset you want to experiment on:

   ```bash
   cd oodtest
   ```

   or

   ```bash
   cd goodtest
   ```

2. Inside each folder, you will find a **README file** with detailed instructions for running the experiments on that dataset.

3. Example (running an OOD test with GCN + SGPC on Cora):

   ```bash
   dataset_name = "cora"
   python test_gcn_ooddetection_final.py 
   ```

4. Example (running a GOOD test with GCN + SGPC on Twitch, domain = language, shift = concept):

   ```bash
   python test_gcn_goodnode_robustness_final.py --dataset twitch --domain language --shift concept
   ```

## Notes

* All experiments are conducted using the **SGPC model**.
* The structure allows you to easily switch between **OODTest** (Cora/Citeseer/PubMed) and **GOODTest** (Twitch/CBAS).
* Please refer to each subfolder’s README for dataset-specific hyperparameters and additional instructions.

---

This setup ensures clear separation of experiments for different datasets while maintaining consistency across the SGPC framework.

