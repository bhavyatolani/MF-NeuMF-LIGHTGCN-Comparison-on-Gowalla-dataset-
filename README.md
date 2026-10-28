# MF-NeuMF-LIGHTGCN-omparison-on-Gowalla-dataset-
MF vs NeuMF vs LightGCN-3 on Gowalla (TensorFlow, BPR loss, Recall@20/NDCG@20): LightGCN-3 0.1285/0.0997 vs MF 0.1211/0.0952 and NeuMF 0.1112/0.0828, random 80/20 

A TensorFlow comparison of three collaborative-filtering models — Matrix Factorization (MF), Neural Matrix Factorization (NeuMF), and LightGCN with 3 propagation layers (LightGCN-3) — on implicit user–location check-in data from Gowalla. All three models are trained with Bayesian Personalized Ranking (BPR) loss and evaluated on Recall@20 and NDCG@20.

## Read this before citing the results

Two facts about the run in this notebook change how the numbers should be described:

1. **The split is a random 80/20 split, not the original NGCF split.** The code pools the downloaded `train.txt` and `test.txt`, shuffles all interactions (seed 2019), and splits them 80/20. The original train/test files are therefore not used as-is. Do not compare these numbers directly with the LightGCN paper's Table 2, which used a different protocol. See [Split](#split) below.
2. **There is a one-line bug in LightGCN's training step.** In `LightGCN.bpr_forward`, the negative-item embedding is gathered from the wrong offset (see [Known issue](#known-issue)). The evaluation code uses the correct offset, so the reported Recall/NDCG scores were computed correctly — but the BPR negative term during training used shifted embeddings. The results below are from the code exactly as run, bug included. Fix the line and re-run before treating LightGCN-3's lead as final.

## About

Gowalla is a location-based social network: a user "checking in" at a place is treated as an implicit positive interaction. There are no ratings and no user/item features in this notebook — every model learns only from which user interacted with which item.

The question this notebook answers is a narrow one: under the same embedding size, loss, optimizer, and evaluation, does graph propagation (LightGCN-3) beat a plain factorization model (MF) and a neural factorization model (NeuMF) on this dataset and split?

## Dataset

- **Dataset:** Gowalla, as distributed with the NGCF repository.
- **Source:** downloaded automatically at run time from the NGCF repository zip (`https://codeload.github.com/xiangwang1223/neural_graph_collaborative_filtering/zip/refs/heads/master`), files `Data/gowalla/train.txt` and `Data/gowalla/test.txt`. The data itself is **not** included in this repo; it is downloaded into `./data/gowalla/` on first run.
- **Format:** one user per line — user id followed by the ids of the items they interacted with, space-separated.
- **Feedback:** implicit only (interacted / not interacted). No ratings, timestamps, or side features are used.

User/item counts are not printed by the notebook and are not stated here; they are determined at run time from the downloaded files.

## Split

The run in this notebook uses an **interaction-level random 80/20 split**:

- All interactions from `train.txt` and `test.txt` are pooled, shuffled with seed 2019, and split 80% train / 20% test.
- The same user can therefore appear in both train and test (with different items). This is a random-split evaluation, not a leave-one-out or time-based evaluation.

The code contains a second path that uses the downloaded files as-is (`split_ratio is None` in `Dataset`), but the argument default is `0.8`, so a default run never takes that path, and it cannot be reached from the Colab command line without editing the code. The numbers in [Results](#results) are the random-split numbers.

## Models

All models use an embedding size of 64 and are trained with BPR loss. LightGCN-3's final representation is 256-dimensional because it concatenates the embeddings from all 4 layers (layer 0 plus 3 propagation layers).

| Model | Idea | Key details in this notebook |
|---|---|---|
| **MF** | Matrix Factorization | One 64-dim user embedding and one 64-dim item embedding per id; score is the dot product. |
| **NeuMF** | Neural Matrix Factorization | Two parallel branches: a GMF branch (element-wise product of 64-dim embeddings) and an MLP branch (separate 64-dim embeddings, concatenated to 128, through two ReLU layers of width 128). The two branch outputs are concatenated and passed through a single output layer. |
| **LightGCN-3** | Light Graph Convolutional Network, 3 layers | One 64-dim embedding per user/item node. Embeddings are propagated 3 times over the symmetrically normalized user–item adjacency (`D^(-1/2) A D^(-1/2)`), with no feature transformation and no non-linearity. Each layer's output is L2-normalized, and all layers (0–3) are concatenated. Score is the dot product of the final user and item vectors. Message dropout 0.1 is applied during training; node dropout is 0 (inactive). |

## Training setup

These are the defaults the notebook ran with. In Colab they are the `argparse` defaults in the code — edit them there to change a run; there is no separate command-line invocation in the notebook.

| Setting | Value |
|---|---|
| Embedding size | 64 |
| Loss | BPR: `-mean(log sigmoid(pos - neg))` + L2 regularization |
| Optimizer | Adam, learning rate 0.001 |
| L2 regularization | 1e-5 |
| Batch size | 1024 (one positive + one uniformly sampled negative per example) |
| Epochs | 10 |
| Gradient clipping | global norm 5.0 |
| Seed | 2019 (NumPy and TensorFlow) |
| LightGCN layers | 3 |
| LightGCN node dropout / message dropout | 0.0 / 0.1 |
| Evaluation | only at epoch 10 (`eval_every = 10`); early stopping is configured but cannot trigger with a single evaluation |
| Hardware in the recorded run | Colab GPU, T4 |

Negative items are sampled uniformly at random from items the user did not interact with in training. LightGCN re-runs full-graph propagation on every training batch, which is why its epochs are much slower than MF/NeuMF (see [Results](#results)).

## Evaluation

- **Metrics:** Recall@20 and NDCG@20, averaged over test users with at least one test item.
- **Ranking:** for each test user, all items are scored; items the user already interacted with in training are masked out (score set to `-inf`) before the top 20 are taken.
- **What the metrics mean here:** Recall@20 is the fraction of a user's test items found in their top 20; NDCG@20 additionally rewards test items ranked higher in the top 20.

## Results

Recorded run (Colab T4, 10 epochs, random 80/20 split, seed 2019). Values are the epoch-10 evaluation printed by the notebook.

| Model | Recall@20 | NDCG@20 |
|---|---|---|
| MF | 0.1211 | 0.0952 |
| NeuMF | 0.1112 | 0.0828 |
| **LightGCN-3** | **0.1285** | **0.0997** |
| % improvement of LightGCN-3 over the best other model | 6.14% | 4.76% |

The "% improvement" row is printed by the notebook from unrounded scores. Recomputing it from the 4-decimal values in the table gives 6.11% and 4.73% — the difference is rounding only.

Reading the table honestly:

- LightGCN-3 is best on both metrics on this split, but its lead over MF is modest (about 0.007 Recall@20).
- NeuMF is worst on both metrics despite reaching the lowest BPR training loss (see below) — lower BPR loss did not mean better top-20 ranking here.
- This is one run, one seed, 10 epochs, with no hyperparameter tuning. It does not show that LightGCN is better in general, and the [known issue](#known-issue) means LightGCN-3 should be re-run after the fix before the lead is relied on.

### Training loss by epoch

BPR loss printed at each epoch. Note that LightGCN-3 starts at a much lower loss than MF/NeuMF in epoch 1 and improves slowly, while MF and NeuMF fall steeply from a higher starting loss.

| Epoch | MF | NeuMF | LightGCN-3 |
|---:|---:|---:|---:|
| 1 | 0.6583 | 0.5419 | 0.1500 |
| 2 | 0.4100 | 0.2968 | 0.1411 |
| 3 | 0.2502 | 0.1253 | 0.1381 |
| 4 | 0.1761 | 0.0757 | 0.1347 |
| 5 | 0.1354 | 0.0561 | 0.1299 |
| 6 | 0.1099 | 0.0450 | 0.1214 |
| 7 | 0.0932 | 0.0390 | 0.1080 |
| 8 | 0.0813 | 0.0341 | 0.0907 |
| 9 | 0.0724 | 0.0309 | 0.0739 |
| 10 | 0.0659 | 0.0279 | 0.0596 |

### Run time

Time per epoch from the recorded run (first epoch includes graph/model setup for LightGCN and warm-up for the others): MF about 7.2 s/epoch (76.6 s total), NeuMF about 10.6 s/epoch (109.4 s total), LightGCN-3 about 32.5 s/epoch (327.0 s total) — about 8.5 minutes of epoch time in total, plus data download and the final evaluation.

## How to run

1. Open the notebook in Google Colab with a GPU runtime (the recorded run used a T4: Runtime → Change runtime type → GPU).
2. Run all cells. The notebook is a single code cell; running it runs the whole comparison.
3. On first run it downloads the Gowalla files into `./data/gowalla/` and trains all three models in order (MF, NeuMF, LightGCN-3), printing per-epoch loss and the final comparison table.

To change a setting (epochs, embedding size, models, split), edit the corresponding `argparse` default in the code — the notebook uses `parse_known_args()`, so Colab's own arguments are ignored and there is no command line to pass flags through. To run only one model, edit the `--models` default list. To try the original downloaded split instead of the random split, the `split_ratio` default must be changed in code (see [Split](#split)).

No output files are written: there are no saved models, checkpoints, or results CSVs. The printed log is the result — save a copy of the notebook with outputs if you need the numbers later.

## Files

```
.
├── README.md
├── MF_NeuMF_LIGHTGCN_omparison_on_Gowalla_dataset.ipynb   # the notebook (single code cell; filename as uploaded)
├── requirements.txt
└── .gitignore
```

The downloaded data (`data/`) is excluded by `.gitignore` and is not part of the repo. The notebook file name keeps its original spelling (`omparison`).

## Requirements

- Python 3 (as provided by Google Colab)
- TensorFlow (Keras API, `tf.keras`)
- NumPy
- SciPy (sparse matrices for the graph Laplacian)

Everything else used (`io`, `os`, `zipfile`, `urllib.request`, `argparse`, `time`) is Python standard library. No version pins are given because the notebook does not pin any; it was run in Colab on a T4 GPU. CPU execution is possible but will be substantially slower, especially for LightGCN.

Install locally with:

```bash
pip install -r requirements.txt
```

## Limitations

- **Random split, not the paper protocol.** See [Split](#split). Results are not directly comparable with published LightGCN/NGCF numbers.
- **One run, one seed, 10 epochs, no tuning.** No repeated runs, no confidence intervals, and no search over embedding size, learning rate, regularization, or number of layers. Ten epochs with evaluation only at the end also means no best-epoch selection was possible.
- **Known issue in LightGCN training** (below) — re-run after fixing before relying on the LightGCN-3 result.
- **No per-user or per-item analysis.** Only mean Recall@20 / NDCG@20 are reported; nothing is said about cold-start users, popular-item bias, or which users LightGCN helps.
- **No saved artifacts.** No trained embeddings, checkpoints, or predictions are stored, so the run must be repeated to inspect anything beyond the printed log.
- **Only Gowalla.** The code is written for one dataset (`DATASETS = ("gowalla",)`); running another dataset would need code changes, including the download paths.

## Known issue

In `LightGCN.bpr_forward`, the negative-item embedding is gathered with the wrong slice offset:

```python
eu, ei, ej = tf.gather(E[:self.n_users], u), tf.gather(E[self.n_users:], i), tf.gather(E[self.n_items:], j)
```

`E` stacks user embeddings first and item embeddings second, so the item slice must start at `self.n_users`. The negative term uses `self.n_items` instead, shifting the gathered embeddings whenever the number of users differs from the number of items. The fix is:

```python
eu, ei, ej = tf.gather(E[:self.n_users], u), tf.gather(E[self.n_users:], i), tf.gather(E[self.n_users:], j)
```

`full_scores` (used for evaluation) already uses `E[self.n_users:]`, so Recall@20 / NDCG@20 were scored with the correct embeddings. Only the BPR negative term during training was affected. This README reports the run as executed, with the bug present.

## References

- He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering.
- Wang, X., He, X., Wang, M., Feng, F., & Chua, T.-S. (2019). Neural Graph Collaborative Filtering (NGCF) — the repository this notebook downloads the Gowalla data from.
- Gowalla dataset, via the NGCF repository: `https://github.com/xiangwang1223/neural_graph_collaborative_filtering`

## Checked

Verified against the uploaded notebook before upload: 1 code cell (352 lines), Colab T4/GPU metadata, outputs present for all three models with the per-epoch losses and final table quoted in [Results](#results), dataset URL and `gowalla`-only dataset list read from the code, split behaviour read from the `Dataset` class and the `--split_ratio` default of 0.8, hyperparameters read from the `argparse` defaults, and the LightGCN offset line read from `bpr_forward`. The notebook copy in this repo is byte-identical to the uploaded file (md5 `c8d3f2648353c43c6cd0a8602ec88644`). No results were re-run and no numbers were taken from any other source.
