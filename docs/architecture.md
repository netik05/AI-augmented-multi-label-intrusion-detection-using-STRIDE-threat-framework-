# Model Architecture

The core decision is to treat intrusion detection as a **structured multi-label prediction problem** rather than single-label classification. This reframing cascades through the loss function, output layer, evaluation metrics, and fusion strategy.

## Problem Formulation

Let $\mathbf{x}_i \in \mathbb{R}^{202}$ be the feature vector for flow $i$, and $\mathbf{y}_i = [y_S, y_T, y_R, y_I, y_D, y_E] \in \{0,1\}^6$ its STRIDE label vector. The objective is to learn:

$$f : \mathbb{R}^{202} \rightarrow [0,1]^6$$

mapping each flow to six **independent** probabilities. Unlike multi-class softmax (mutually exclusive, sums to one), each output is an independent sigmoid trained with binary cross-entropy:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{6} \left[ y_{ij}\log(\hat{p}_{ij}) + (1 - y_{ij})\log(1 - \hat{p}_{ij}) \right]$$

This allows overlapping activations — a backdoor flow can fire Spoofing, Repudiation, and Elevation of Privilege simultaneously.

---

## Base Model 1 — Static XGBoost

Six independent `XGBClassifier` instances wrapped in a `MultiOutputClassifier` (one-vs-rest per STRIDE dimension — appropriate because the dimensions are semantically independent).

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `tree_method` | `hist` (GPU-accelerated) |
| `eval_metric` | `logloss` |
| `scale_pos_weight` | per-dimension inverse class-frequency |

**Why XGBoost:** consistently strong on UNSW-NB15 in prior literature; natively compatible with SHAP `TreeExplainer`; handles class imbalance via `scale_pos_weight` (critical for Repudiation, ~476 positive samples). Trains all six classifiers in ~84 s.

---

## Base Model 2 — Temporal BiLSTM + Attention

A single flow is a snapshot; a real attack is a sequence. A sliding window of length $L = 10$ produces subsequences $\mathbf{X}^{(t)} \in \mathbb{R}^{10 \times 202}$, labelled by the final flow in the window.

| Layer | Configuration |
|---|---|
| Input | (10, 202) |
| BiLSTM 1 | 128 units (64 fwd + 64 bwd), `return_sequences=True` |
| Dropout | 0.30 |
| BiLSTM 2 | 64 units (32 fwd + 32 bwd), `return_sequences=True` |
| Dropout | 0.20 |
| Attention | soft-attention over 10 timesteps |
| Dense | 64 units, ReLU |
| Output | 6 units, sigmoid |

**Attention mechanism.** A soft weighting over BiLSTM hidden states $\mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_{10}]$:

$$\alpha_t = \frac{\exp(s_t)}{\sum_{k=1}^{10}\exp(s_k)}, \quad s_t = \sum_d h_{td}, \qquad \mathbf{c} = \sum_{t=1}^{10}\alpha_t \mathbf{h}_t$$

letting the model focus on the most threat-relevant flows within a window.

**Training.** Adam ($\eta = 10^{-3}$), batch size 512, up to 20 epochs, early stopping (patience 5), LR reduction (patience 3, factor 0.5), 15% validation split. Converges in ~20 min on GPU.

---

## Hybrid Fusion Strategies

**1. Late Averaging** — element-wise mean of the two probability vectors.

$$\hat{\mathbf{P}}_{\text{avg}} = \frac{\hat{\mathbf{P}}_{\text{XGB}} + \hat{\mathbf{P}}_{\text{LSTM}}}{2}$$

**2. Attention-Weighted Averaging** — each model weighted by inverse output variance (confidence), per dimension.

**3. Meta-Stack Classifier** — a `MultiOutputClassifier` of six Logistic Regressions trained on the concatenated 12-dim probability outputs of both base models. Achieves the lowest Hamming Loss (0.0723).

---

## Per-Label Threshold Optimisation

A uniform 0.5 threshold is suboptimal when class balance varies (DoS activates 29.3% of flows; Repudiation only 0.9%). A grid search over $\tau \in \{0.10, 0.15, \dots, 0.85\}$ maximises per-dimension F1:

$$\tau_j^\* = \arg\max_\tau F_1\left(\mathbf{y}_j, \mathbb{1}[\hat{\mathbf{P}}_j \ge \tau]\right)$$

This lifts Macro F1 from 0.6592 to 0.6869 (+0.0277). Repudiation's threshold optimises to 0.10, reflecting severe imbalance.

---

## Design Constraints

1. **Dataset-agnostic** — every stage applies to both UNSW-NB15 and CIC-IDS-2018 without modification, making cross-dataset validation a genuine generalisation test.
2. **Near-real-time** — XGBoost < 1 ms/flow, BiLSTM < 5 ms/flow, both within operational IDS requirements.
