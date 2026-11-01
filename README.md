# DA-AVQA: Degradation-Aware No-Reference Quality Assessment for Animated Video

DA-AVQA predicts the mean opinion score (MOS, 1–5) of a distorted animated video **from the video alone**
(no clean reference). It averages four complementary predictors:

1. an SVR on 57 handcrafted features (temporal, sharpness, blockiness, 3D entropies, NSS, and two new line-art cues);
2. an SVR on frozen ImageNet EfficientNetB0 features (after PCA);
3. degradation-aware experts: a distortion-type classifier plus one SVR expert per distortion type;
4. DA-TransQA, a multi-stream transformer trained with ranking, distortion-type and self-supervised losses.

Evaluation uses **content-disjoint splits**: test cartoons are never seen in training.

## Results (BeDAMQA, held-out test cartoons, mean over 10 iterations, seed 1000)

| Method | PLCC ↑ | SROCC ↑ | KRCC ↑ | RMSE ↓ (MOS) | ms/video ↓ |
|---|---|---|---|---|---|
| RF-AVQA (reported in its paper) | 0.879 | 0.863 | – | 0.509 | – |
| SVR (57 features) | 0.876 | 0.840 | 0.666 | 0.506 | 1282 |
| PCA + SVR (backbone) | 0.864 | 0.848 | 0.669 | 0.523 | 196 |
| Experts | 0.904 | 0.889 | 0.723 | 0.443 | 1479 |
| DA-TransQA | 0.878 | 0.852 | 0.681 | 0.497 | 1304 |
| **DA-AVQA (ensemble)** | **0.927** | **0.912** | **0.751** | **0.390** | 1502 |

Per-distortion SROCC (within test splits): Blur 0.803, Frame freeze 0.938, Frame hang 0.947, H.264 0.941, H.265 0.715
(RF-AVQA: 0.614, 0.931, 0.907, 0.943, 0.688). Runtime: Apple MacBook Air (M4), one video at a time, including decoding.
The raw result files are in `results/`.

## Repository contents

```
da_transqa_v5_confirm.py      main experiment: 10 content-disjoint iterations (produces the reported results)
da_transqa_ablation_fast.py   ablations without the transformer (feature groups, expert routing, ensemble)
measure_runtime_v5.py         inference time per video for every component (no retraining)
make_plots_rfavqa_v2.py       figures from the saved result files (no retraining)
results/                      result CSVs of the confirmation run and the runtime measurement
plots/                        figures used in the report
requirements.txt              Python dependencies
```

## 1. Setup

Python 3.9. On an Apple-silicon Mac (as used for the reported results):

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Linux/Windows, `tensorflow-metal` is skipped automatically and TensorFlow runs on CPU or CUDA.
EfficientNetB0 ImageNet weights are downloaded by Keras on first use (internet needed once).

## 2. Dataset (not included)

The BeDAMQA dataset is **not redistributed** here. Obtain it from its authors / official source
(S. Ganavdiya, B. Appina, R. B. Pachori, *IEEE Data Descriptions*, vol. 2, pp. 441–451, 2025).
Arrange it as:

```
DatasetMain/
├── Reference/      15 pristine videos (01 … 15)
├── Blur/
├── FrameFreeze/
├── FrameHang/
├── H264/
├── H265/
└── <MOS spreadsheet>.xlsx   (file name + MOS columns)
```

Point the code to it:

```bash
export AVQA_ROOT=/path/to/DatasetMain
```

## 3. Run

```bash
# quick crash test: 1 iteration, 2 epochs (the first run also builds the feature caches)
SMOKE=1 python da_transqa_v5_confirm.py

# full experiment: 10 iterations -> reported results
python da_transqa_v5_confirm.py
```

Frames, handcrafted features and backbone features are extracted once and cached in `~/.avqa_cache`;
later runs reuse the cache. The full run writes:

- `da_transqa_results_v5_confirm_seed1000.csv` – metrics per iteration and predictor
- `da_transqa_per_distortion_within_v5_confirm_seed1000.csv` – per-distortion metrics within each test split
- `da_transqa_per_distortion_pooled_v5_confirm_seed1000.csv` – per-distortion metrics pooled per round

Optional, after the full run (both reuse the caches):

```bash
python measure_runtime_v5.py            # -> runtime_v5_confirm_seed1000.csv
python make_plots_rfavqa_v2.py results  # -> results/plots/*.pdf and *.png
python da_transqa_ablation_fast.py      # ablation tables (results pending in the report)
```

## 4. Protocol and reproducibility

- **Split:** all versions of a cartoon stay in one subset. The 15 cartoons are shuffled into 5 groups of 3;
  each group is the test set once per round; 2 reshuffled rounds give 10 iterations of
  9 training (225 videos) / 3 validation (75) / 3 test (75) cartoons.
- **No leakage:** normalisation uses training statistics only; hyperparameters and early stopping use the
  validation cartoons only; the pristine model uses only reference videos of non-test cartoons.
- **Seeds:** base seed 1000 (development runs used 42). Round *r* draws cartoon groups with seed 1000 + *r*;
  iteration *i* uses seed 1000 + *i* for the validation draw, transformer weights, dropout, augmentation and data order.
- **Metrics:** SROCC and KRCC on raw predictions; PLCC and RMSE after the VQEG five-parameter logistic mapping.
- **Determinism:** the splits and the SVR / PCA+SVR / expert predictors are exactly reproducible. Transformer
  reruns on a GPU (Apple Metal) can differ slightly, because GPU arithmetic and parallel data loading are not bit-exact.

## 5. Main settings

Transformer: token width 32, 4 heads, 8×8 patches, 1 spatial block per stream, 2 temporal blocks, dropout 0.2;
batch of 16 clips + 16 synthetic degraded copies; AdamW (lr 3e-4, weight decay 0.05, gradient clipping 1.0);
at most 100 epochs, early stopping on validation SROCC (patience 15), learning rate halved after 5 stalled epochs.
Loss = MSE + 0.5 rank + 0.5 same-cartoon rank + 0.3 distortion type + 0.5 self-supervised.
SVR C ∈ {0.1, 0.3, 1, 3, 10}; classifier C ∈ {0.3, 1, 3, 10}; backbone PCA size ∈ {8, 16, 32}, chosen on validation.

## Baseline

RF-AVQA: R. S. Mogasati, S. Ganavdiya, A. K. R. Poreddy, B. Appina, R. B. Pachori, "A no-reference framework for
animated video quality prediction in consumer multimedia applications," *IEEE Trans. Consum. Electron.*, 2026.
Its numbers above are taken from that paper; it was not re-run here.

## Authors

D. Kanishkarane and Guru Divya Darshini U — Department of Computer Science and Engineering,
Shiv Nadar University Chennai.
