# Kolmogorov-Arnold Networks (KAN) From Scratch – PyTorch

This project implements Kolmogorov-Arnold Networks (KAN) from scratch in PyTorch
and benchmarks them against a baseline MLP on MNIST and Fashion-MNIST datasets.

The implementation is modular, reproducible, and CLI-controlled.

---

## 1. Motivation

Kolmogorov-Arnold Networks replace traditional linear weights in neural networks
with learnable univariate spline functions on edges.

Instead of:

    y = Wx + b

KAN approximates mappings using:

    f(x) = Σ φ_i(x_i)

where φ_i are learnable spline-based functions.

This allows:

- Edge-wise function learning
- Higher functional expressivity
- Interpretable transformations
- Flexible approximation capacity

This project investigates whether KAN provides measurable performance gains
over classical MLP architectures on image classification tasks.

---

## 2. Architecture Overview

### Baseline: MLP

- Architecture: 784 → 64 → 10
- ReLU activation
- BatchNorm
- CrossEntropy loss
- AdamW optimizer

### KAN

Each KAN layer consists of:

- Base linear transformation (SiLU activation)
- B-Spline basis expansion
- Learnable spline coefficients
- Adaptive grid resolution
- Learnable scaling factor

Forward computation:

    output = base_out + spline_scaler * spline_out

Where:

- base_out = Linear(SiLU(x))
- spline_out = Linear(B-Spline(x))

---

## 3. Experimental Setup

Datasets:

- MNIST
- Fashion-MNIST

Training:

- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Epochs: 10
- Batch size: 64
- Device auto-detection (CPU / CUDA / MPS)

Models compared:

- MLP (~51K parameters)
- KAN (Grid=5)
- KAN (Grid=10)

---

## 4. Results

### MNIST

| Model      | Parameters | Best Test Accuracy |
| ---------- | ---------- | ------------------ |
| MLP        | ~51K       | 95.36%             |
| KAN (G=5)  | ~457K      | 96.56%             |
| KAN (G=10) | ~711K      | 95.60%             |

Observations:

- KAN (G=5) outperforms MLP.
- Increasing grid resolution increases parameters significantly.
- Higher resolution does not always improve accuracy.
- Trade-off exists between capacity and generalization.

---

## 5. Key Technical Components

### B-Spline Basis Computation

Implemented manually without external spline libraries.

- Grid construction
- Recursive spline basis calculation
- Clamped input normalization [-1,1]
- Efficient tensor reshaping for linear projection

### Device Compatibility

Automatic hardware selection:

- Apple Silicon (MPS)
- CUDA
- CPU fallback

### Parameter Analysis

Parameter count utility implemented to compare model complexity.

---

## 6. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train KAN on MNIST:

```bash
python src/train.py --model kan --grid 5 --dataset MNIST
```

Train MLP:

```bash
python src/train.py --model mlp --dataset FashionMNIST
```
