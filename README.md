
# 📊 SegMetric: A Video Object Segmentation Evaluation Toolkit

**SegMetric** is a fast, flexible Python toolkit for evaluating Video Object Segmentation (VOS) models. It supports standard **Jaccard (IoU)** and **Boundary F-score** metrics, with full statistics including **Mean**, **Recall**, and **Decay**.  

Built on the official [DAVIS 2017](https://davischallenge.org/) evaluation code and extended for datasets like **VOST** and long video sequences.

---

## ✨ Features

- ✅ **Standard Metrics**  
  Computes **J-Score (IoU)** and **F-Score (boundary accuracy)** per sequence and globally.

- 📈 **Detailed Statistics**  
  Automatically computes **Mean**, **Recall**, and **Decay** for both IoU and boundary metrics.

- 🚀 **Parallel Evaluation**  
  Uses Python multiprocessing for fast evaluation—even for large datasets.

- 🧰 **Dual Interface**  
  Use via **command-line** or **Python API** in your own scripts.

- 📂 **Clean Output**  
  Prints results to terminal and saves to `.csv` files for easy analysis.

---

## 🛠️ Installation

**Requirements:** Python ≥ 3.8  
We recommend using a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-username/segmetric.git
cd segmetric/

# Install in editable (development) mode
pip install -e .
```

---

## 🚀 Usage

### 🔹 Option 1: Command-Line

```bash
run-seg-eval   --results_path ./results/my_model_predictions   --dataset_path ./datasets/long_videos   --set val   --re
```

#### Required Arguments

| Argument         | Description                                                             |
|------------------|-------------------------------------------------------------------------|
| `--results_path` | **(Required)** Path to predicted segmentation masks.                   |
| `--dataset_path` | **(Required)** Root directory of the dataset (e.g., `.../VOST/`).       |

#### Optional Flags

| Flag   | Description                                                |
|--------|------------------------------------------------------------|
| `--set` | Dataset split to evaluate (`train`, `val`, etc.). Default: `val`. |
| `--re`  | Re-evaluate and overwrite any cached `.csv` result files. |

---

### 🔹 Option 2: Python API

```python
from pathlib import Path
from segmetric.evaluation import Evaluation

results_path = Path("/path/to/predictions")
dataset_path = Path("/path/to/dataset")
subset_to_eval = "val"

evaluator = Evaluation(dataset_root=dataset_path, gt_set=subset_to_eval)
metrics = evaluator.evaluate(results_path)

# Access individual results
j_mean_global = metrics['J']['M']
print(f"Global J-Score Mean: {sum(j_mean_global) / len(j_mean_global):.4f}")
```

---

## 🧾 Sample Output

After running `run-seg-eval`, you will see something like:

```
Evaluating /.../long_video_prediction
Evaluating sequences...
Evaluating on dataset: /.../long_videos
Starting evaluation on 3 sequences with 32 workers...
100%|██████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 43842.90it/s]
Global results saved to .../global_results-val.csv
Per-sequence results saved to .../per-sequence_results-val.csv

================================================================================
                            Global results for 'val'                            
================================================================================
  J-Mean  J-Recall   J-Decay   F-Mean  F-Recall   F-Decay  J_last-Mean  J_last-Recall  J_last-Decay  F_last-Mean  F_last-Recall  F_last-Decay
0.857282  0.982456 -0.032032 0.914792  0.982456 -0.024466     0.864079            1.0      0.122738     0.919923            1.0      0.134012

================================================================================
                         Per-sequence results for 'val'                         
================================================================================
  Sequence   J-Mean   F-Mean  J_last-Mean  F_last-Mean
     rat_1 0.920132 0.976721     0.908191     0.991858
dressage_1 0.766742 0.867963     0.762747     0.887131
 blueboy_1 0.884973 0.899693     0.921300     0.880778

Total time: 1.43 seconds
```

---

## 📁 Output Files

| File                              | Description                          |
|-----------------------------------|--------------------------------------|
| `global_results-<set>.csv`        | Summary scores across all sequences. |
| `per-sequence_results-<set>.csv`  | Detailed metrics for each sequence.  |

---

## 📘 Citation

If SegMetric helps your research, please cite the [DAVIS 2017 challenge](https://davischallenge.org/) and mention SegMetric in your acknowledgments. A custom BibTeX entry is coming soon.

---

## 🤝 Contributing

We welcome PRs, bug reports, and feature suggestions!  
To contribute:

```bash
# Make changes in a feature branch
git checkout -b feature/your-feature-name
```

Then open a pull request 🚀

---

## 📬 Contact

For questions, please reach out to: **[yourname@domain.edu]**

---

## 🧪 License

This project is licensed under the **MIT License**.

