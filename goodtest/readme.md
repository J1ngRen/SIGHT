# README: Experiments on the GOOD Dataset (Twitch & CBAS)

This repository provides experimental scripts for Out-of-Distribution (OOD) detection tasks on the **GOOD dataset**, focusing on the **Twitch** and **CBAS** subsets.

## Main Experiment Scripts

* `gcn_goodnode_ooddetection.py`: Experiments using a plain GCN model.
* `gat_goodnode_ooddetection.py`: Experiments using a plain GAT model.
* `test_goodnode_gcn_ooddetection.py`: Experiments using **Spiking + GCN + Predictive Coding**.
* `test_goodnode_gat_ooddetection.py`: Experiments using **Spiking + GAT + Predictive Coding**.

### Example Command

```bash
python test_goodnode_gcn_ooddetection.py --dataset twitch --domain language --shift concept
python test_goodnode_gcn_ooddetection.py --dataset cbas --domain color --shift concept
```

**Notes:**

* For **Twitch**, set `--domain language`.
* For **CBAS**, set `--domain color`.
* For both datasets, the shift type should be `concept`.
* Make sure to include these arguments explicitly in the command line.

## Recommended Hyperparameters

* **Hidden dimensions**:

  * Twitch: `[128, 128, 64]`
  * CBAS: `[64, 64, 32]`
* **Learning rates**:

  * Twitch: `eta_x = 0.01`, `eta_p (network learning rate) = 0.001`
  * CBAS: `eta_x = 0.005`, `eta_p (network learning rate) = 0.0005`

## Additional Experiments

For other experiment types, navigate to the corresponding folder:

* **Ablation studies** → `cd ablation`
* **Post-hoc calibration methods** → `cd posthoc`
* **GDUQ methods** → `cd gduq`
* **Hyperparameter sensitivity experiments** → `cd hyper-se`
* **Predictive coding explainability experiments** → `cd pc-ex`

The script can automatically download the dataset. If the download fails, you can manually download it and place it in the datasets folder for use.
*****
For Twitch dataset you can download from this url :  https://drive.google.com/file/d/1wii9CWmtTAUofNTgg-GkpRz_iECcbQzK/view?usp=sharing
For WebKB dataset you can download from this url : https://drive.google.com/file/d/1DOdUOzAMBtcHXTphrWrKhNWPxzMDNvnb/view?usp=sharing
*****


Each folder contains its own usage instructions. The general guidelines described above apply across all experiment types.
