# IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment

## 📖 Introduction

**IDEAL** is an innovative data equilibrium adaptation framework designed to optimize the mixture of multi-domain datasets during Supervised Fine-tuning (SFT).

While most research focuses on data *quality*, IDEAL explores the impact of data *quantity* across different domains. It dynamically adjusts the proportions of domain-specific data based on their influence on downstream tasks, resolving potential capability conflicts and ensuring a well-balanced model.

### Key Highlights

* **Dynamic Balancing**: Optimizes data mixing ratios iteratively using gradient-based information.
* **Efficient Computation**: Employs K-FAC to approximate Hessian matrices, making it feasible for large-scale model alignment.
* **Scalable**: Proven effective on models ranging from Llama-3.2-1B to Llama-3.1-8B.
* **Superior Performance**: Outperforms uniform mixing and existing reweighting methods (e.g., DoReMi, DOGE) by significant margins.

---

## 🛠 Methodology

IDEAL models the data mixing problem as a **Bi-level Optimization** task:

1. **Inner Loop**: Optimizes model parameters  on the mixture training set.
2. **Outer Loop**: Optimizes the mixing ratio  by minimizing loss on a small, high-quality reference set .

---

## 💻 Usage/

### 1. Install

```bash
conda env create -n 'yourenv' -f environment.yml
```
or
```bash
pip install -r requirements.txt
```

### 2. Run

```bash
torchrun --nproc_per_node=8 main.py
```

Note: Please replace the args in `main.py` with your own.


---

## 📜 Citation

If you find our work useful in your research, please cite:

```bibtex
@article{ming2025ideal,
  title={IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment},
  author={Ming, Chenlin and Qu, Chendi and Cai, Mengzhang and Pei, Qizhi and Pan, Zhuoshi and Li, Yu and Duan, Xiaoming and Wu, Lijun and He, Conghui},
  journal={arXiv preprint arXiv:2505.12762},
  year={2025}
}

```

---

*For more details, please refer to the full paper.*
