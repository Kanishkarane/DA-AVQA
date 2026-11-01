# make_plots_rfavqa_v2.py — (v2: picks the distorted-video cache, fixing empty plots 08-10) comparison plots against RF-AVQA WITHOUT re-running any training.
#
# Reads only files that already exist after your confirmation run:
#   da_transqa_results_v5_confirm_seed1000.csv                  (per-iteration metrics)
#   da_transqa_per_distortion_within_v5_confirm_seed1000.csv    (per-distortion, within splits)
#   da_transqa_per_distortion_pooled_v5_confirm_seed1000.csv    (per-distortion, pooled)
#   ~/.avqa_cache/v3_frames_N16_S64_C5_*.npz                    (MOS + 57 features; optional)
# RF-AVQA numbers are the ones reported in its paper (same values the confirm script prints).
#
# Run (from the folder that holds the three CSVs):
#   python make_plots_rfavqa.py
# Output: plots/*.pdf (vector, use these in LaTeX) and plots/*.png (300 dpi, for slides).
# Needs: pandas, numpy, matplotlib, scipy  (all already installed for the main code)

import os, re, glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RUN = "v5_confirm_seed1000"
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(DATA_DIR, "plots")
CACHE_GLOB = os.path.join(os.path.expanduser("~"), ".avqa_cache", "v3_frames_N16_S64_C5_*.npz")
os.makedirs(OUT, exist_ok=True)

# ---- RF-AVQA, as reported in its paper ----
RF_OVERALL = {"PLCC": 0.879, "SROCC": 0.863, "RMSE": 0.509}
RF_DIST = {"Blur": (0.680, 0.614), "FrameFreeze": (0.961, 0.931), "FrameHang": (0.941, 0.907),
           "H264": (0.955, 0.943), "H265": (0.774, 0.688)}            # (PLCC, SROCC)
DISTS = ["Blur", "FrameFreeze", "FrameHang", "H264", "H265"]
DLAB = {"Blur": "Blur", "FrameFreeze": "Freeze", "FrameHang": "Hang", "H264": "H.264", "H265": "H.265"}
MLAB = {"unsup": "Pristine distance only", "svr": "SVR (57 features)", "cnn": "PCA+SVR (backbone)",
        "experts": "Experts", "deep": "DA-TransQA", "ensemble": "DA-AVQA (ensemble)",
        "hybrid3": "Without experts", "ens_no_deep": "Without transformer"}
PREDICTORS = ["svr", "cnn", "experts", "deep"]

# ---- look: thin marks, recessive grid, text in ink colours (never series colours) ----
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
GRAY = "#a3a29c"                                             # RF-AVQA = reference, kept neutral
LIGHT_BLUE = "#9ec5f4"                                       # single predictors (recede)
GAIN, LOSS = "#2a78d6", "#e34948"                            # diverging poles
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
COL1, COL2 = 3.5, 7.16          # IEEE single- and double-column widths (inches)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("  saved", name)


def load_csv(kind):
    p = os.path.join(DATA_DIR, f"da_transqa_{kind}_{RUN}.csv")
    if not os.path.exists(p):
        print(f"  missing {p} -> plots that need it are skipped")
        return None
    return pd.read_csv(p)


res = load_csv("results")
within = load_csv("per_distortion_within")
pooled = load_csv("per_distortion_pooled")
print("Making plots in", OUT)

# =====================================================================================
# 1. Overall SROCC / PLCC / RMSE: RF-AVQA vs each predictor vs the ensemble (±SD over 10 splits)
# =====================================================================================
if res is not None:
    rows = [("RF-AVQA (reported)", None)] + [(MLAB[m], m) for m in PREDICTORS + ["ensemble"]]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.9), sharey=True)
    for ax, metric in zip(axes, ["SROCC", "PLCC", "RMSE"]):
        vals = [RF_OVERALL[metric] if m is None else res[f"{m}_{metric}"].mean() for _, m in rows]
        sds = [0 if m is None else res[f"{m}_{metric}"].std() for _, m in rows]
        cols = [GRAY] + [LIGHT_BLUE] * len(PREDICTORS) + [BLUE]
        y = np.arange(len(rows))
        ax.barh(y, vals, xerr=sds, color=cols, height=0.62, error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
        lo = 0.3 if metric == "RMSE" else 0.75
        hi = max(v + s for v, s in zip(vals, sds)) * 1.17 if metric == "RMSE" else 1.04
        for yi, v, s in zip(y, vals, sds):                      # label sits right of the error bar
            ax.text(v + s + (hi - lo) * 0.015, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK2)
        ax.axvline(RF_OVERALL[metric], color=INK2, lw=0.8, ls="--")
        ax.set_title(f"{metric} ({'lower' if metric == 'RMSE' else 'higher'} is better)")
        ax.set_xlim(lo, hi)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(rows)), [r for r, _ in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Overall results on unseen cartoons (mean ± SD over 10 test splits); dashed = RF-AVQA",
                 fontsize=8.5, y=1.08)
    save(fig, "01_overall_vs_rfavqa")

# =====================================================================================
# 2-3. Per-distortion SROCC and PLCC: RF-AVQA vs ours (within splits) vs ours (pooled)
# =====================================================================================
if within is not None:
    w = within[within.method == "ensemble"].groupby("distortion")[["SROCC", "PLCC"]].mean()
    p = pooled[pooled.method == "ensemble"].set_index("distortion") if pooled is not None else None
    for metric, idx, name in [("SROCC", 1, "02_per_distortion_srocc"), ("PLCC", 0, "03_per_distortion_plcc")]:
        series = [("RF-AVQA (reported)", [RF_DIST[d][idx] for d in DISTS], GRAY),
                  ("DA-AVQA, within splits", [w.loc[d, metric] for d in DISTS], BLUE)]
        if p is not None:
            series.append(("DA-AVQA, pooled", [p.loc[d, metric] for d in DISTS], ORANGE))
        fig, ax = plt.subplots(figsize=(COL1, 2.1))
        x, bw = np.arange(len(DISTS)), 0.8 / len(series)
        for k, (lab, vals, c) in enumerate(series):
            xs = x - 0.4 + bw * (k + 0.5)
            ax.bar(xs, vals, bw * 0.9, color=c, label=lab)
            if lab.endswith("within splits"):                       # label only our main series
                for xi, v in zip(xs, vals):
                    ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=6, color=INK2)
        ax.set_xticks(x, [DLAB[d] for d in DISTS])
        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel(metric)
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, 1.2))
        save(fig, name)

# =====================================================================================
# 4. Gain over RF-AVQA per distortion (SROCC difference): blue = better, red = worse
# =====================================================================================
if within is not None:
    gw = [w.loc[d, "SROCC"] - RF_DIST[d][1] for d in DISTS]
    gp = [p.loc[d, "SROCC"] - RF_DIST[d][1] for d in DISTS] if p is not None else None
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    x = np.arange(len(DISTS))
    bw = 0.38 if gp is not None else 0.6
    ax.bar(x - (bw / 2 if gp is not None else 0), gw, bw * 0.9,
           color=[GAIN if g >= 0 else LOSS for g in gw], label="within splits")
    if gp is not None:
        ax.bar(x + bw / 2, gp, bw * 0.9, color="white", hatch="////", label="pooled",
               edgecolor=[GAIN if g >= 0 else LOSS for g in gp], linewidth=0.8)
    for xi, g in zip(x, gw):
        ax.text(xi - (bw / 2 if gp is not None else 0), g + (0.006 if g >= 0 else -0.018),
                f"{g:+.3f}", ha="center", fontsize=6, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [DLAB[d] for d in DISTS])
    ax.set_ylabel("SROCC gain over RF-AVQA")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "04_gain_over_rfavqa")

# =====================================================================================
# 5. Stability: SROCC of every predictor in each of the 10 test splits (box + points)
# =====================================================================================
if res is not None:
    ms = PREDICTORS + ["ensemble"]
    data = [res[f"{m}_SROCC"].values for m in ms]
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1), whiskerprops=dict(color=INK2, lw=0.8),
                    capprops=dict(color=INK2, lw=0.8), boxprops=dict(lw=0.8, color=INK2))
    for patch, m in zip(bp["boxes"], ms):
        patch.set_facecolor(BLUE if m == "ensemble" else LIGHT_BLUE)
    rng = np.random.default_rng(0)
    for k, d in enumerate(data, start=1):
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(d)), d, s=9, color=INK2, zorder=3,
                   edgecolor="white", linewidth=0.4)
    ax.axhline(RF_OVERALL["SROCC"], color=INK2, ls="--", lw=0.8)
    ax.text(0.55, RF_OVERALL["SROCC"] + 0.003, "RF-AVQA 0.863", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(1, len(ms) + 1), ["SVR", "PCA+\nSVR", "Experts", "DA-\nTransQA", "Ensemble"])
    ax.set_ylabel("SROCC per test split")
    ax.grid(axis="x", visible=False)
    save(fig, "05_stability_across_splits")

# =====================================================================================
# 6. Heatmap: SROCC of each predictor per distortion (within splits), RF-AVQA as the top row
# =====================================================================================
if within is not None:
    ms = PREDICTORS + ["ensemble"]
    tab = within[within.method.isin(ms)].pivot_table(index="method", columns="distortion",
                                                     values="SROCC", aggfunc="mean").reindex(index=ms, columns=DISTS)
    M = np.vstack([[RF_DIST[d][1] for d in DISTS], tab.values])
    labels = ["RF-AVQA (reported)"] + [MLAB[m] for m in ms]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    im = ax.imshow(M, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.8 else INK,
                    fontweight="bold" if M[i, j] == M[:, j].max() else "normal")
    ax.set_xticks(range(len(DISTS)), [DLAB[d] for d in DISTS])
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0.5, color="white", lw=2)                           # separate the reference row
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("SROCC", fontsize=7)
    cb.outline.set_visible(False)
    save(fig, "06_heatmap_predictor_by_distortion")

# =====================================================================================
# 7. Ablation: each part alone, the ensemble without one part, and the full ensemble
# =====================================================================================
if res is not None:
    ms = [m for m in ["unsup", "svr", "cnn", "deep", "experts", "hybrid3", "ens_no_deep", "ensemble"]
          if f"{m}_SROCC" in res.columns]
    vals = [res[f"{m}_SROCC"].mean() for m in ms]
    sds = [res[f"{m}_SROCC"].std() for m in ms]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    y = np.arange(len(ms))
    cols = [BLUE if m == "ensemble" else ("#6da7ec" if m in ("hybrid3", "ens_no_deep") else LIGHT_BLUE) for m in ms]
    ax.barh(y, vals, xerr=sds, color=cols, height=0.62, error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    for yi, v, s in zip(y, vals, sds):
        ax.text(v + s + 0.01, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.axvline(RF_OVERALL["SROCC"], color=INK2, ls="--", lw=0.8)
    ax.text(RF_OVERALL["SROCC"], -0.9, "RF-AVQA 0.863", ha="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [MLAB[m] for m in ms])
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("SROCC (mean ± SD over 10 test splits)")
    ax.grid(axis="y", visible=False)
    save(fig, "07_ablation")

# =====================================================================================
# 8-10. Dataset and feature plots from the cache (MOS + 57 features; no training)
# =====================================================================================
# The cache folder holds TWO files with this pattern: the 374 distorted videos and the 15 clean
# Reference videos. v1 took the newest one (the Reference cache) -> empty plots. Take the largest.
_n = lambda f: int(m.group(1)) if (m := re.search(r"_C5_(\d+)_", os.path.basename(f))) else 0
caches = sorted(glob.glob(CACHE_GLOB), key=_n)
if not caches:
    print(f"  no cache found at {CACHE_GLOB} -> plots 08-10 skipped")
else:
    z = np.load(caches[-1], allow_pickle=True)                # lazy: the big frame array is NOT loaded
    F, mos = z["F"], z["y"]
    paths = [str(s) for s in z["paths"]]
    dist = np.array([os.path.basename(os.path.dirname(s)) for s in paths])
    stem = [os.path.splitext(os.path.basename(s))[0] for s in paths]
    print(f"  cache: {os.path.basename(caches[-1])}  ({len(mos)} videos)")

    # 8. MOS distribution per distortion type
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    data = [mos[dist == d] for d in DISTS]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1), whiskerprops=dict(color=INK2, lw=0.8),
                    capprops=dict(color=INK2, lw=0.8), boxprops=dict(lw=0.8, color=INK2))
    for patch, d in zip(bp["boxes"], DISTS):
        patch.set_facecolor(BLUE if d == "H265" else LIGHT_BLUE)
    rng = np.random.default_rng(1)
    for k, dd in enumerate(data, start=1):
        ax.scatter(k + rng.uniform(-0.14, 0.14, len(dd)), dd, s=6, color=INK2, zorder=3,
                   edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(1, len(DISTS) + 1), [f"{DLAB[d]}\n(n={len(x)})" for d, x in zip(DISTS, data)])
    ax.set_ylabel("MOS (1-5)")
    ax.grid(axis="x", visible=False)
    save(fig, "08_mos_distribution")

    # 9. Mean MOS vs distortion level (level = last number in the file name, e.g. 05_H264_0200 -> 200)
    level = np.array([float(m.group(1)) if (m := re.search(r"(\d+)\D*$", s)) else np.nan for s in stem])
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.62, 1.9))
    for d, c in [("H264", BLUE), ("H265", ORANGE)]:
        mk = (dist == d) & ~np.isnan(level)
        lv = np.unique(level[mk])
        mu = [mos[mk & (level == l)].mean() for l in lv]
        sd = [mos[mk & (level == l)].std() for l in lv]
        axes[0].errorbar(lv, mu, yerr=sd, color=c, lw=2, marker="o", ms=4, capsize=2, label=DLAB[d])
        print(f"  {d} levels found: {lv.tolist()}")
    axes[0].set_xscale("log")
    axes[0].set_xticks([200, 500, 1000, 2000, 4000], ["200", "500", "1000", "2000", "4000"])
    axes[0].minorticks_off()
    axes[0].set_xlabel("Bitrate (kbps)")
    axes[0].set_ylabel("Mean MOS ± SD")
    axes[0].legend(loc="lower right")
    mk = (dist == "Blur") & ~np.isnan(level)
    lv = np.unique(level[mk])
    print(f"  Blur levels found: {lv.tolist()}  (check these are the blur radii 3-7)")
    axes[1].errorbar(lv, [mos[mk & (level == l)].mean() for l in lv],
                     yerr=[mos[mk & (level == l)].std() for l in lv],
                     color=BLUE, lw=2, marker="o", ms=4, capsize=2)
    axes[1].set_xlabel("Blur level (from file name)")
    for ax in axes:
        ax.set_ylim(1, 5)
    save(fig, "09_mos_vs_level")

    # 10. Do the new line-art cues track quality? (Spearman correlation with MOS, no training)
    i_sh, i_ri = 12, 13                                     # FEATURE_NAMES: outline_sharpness, outline_ringing
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.62, 1.9))
    for ax, idx, ds, name in [(axes[0], i_sh, ["Blur"], "Outline sharpness (Blur videos)"),
                              (axes[1], i_ri, ["H264", "H265"], "Outline ringing (H.264 + H.265)")]:
        mk = np.isin(dist, ds)
        rho = spearmanr(F[mk, idx], mos[mk])[0]
        for d, c in zip(ds, [BLUE, ORANGE]):
            m2 = dist == d
            ax.scatter(F[m2, idx], mos[m2], s=10, color=c, edgecolor="white", linewidth=0.4,
                       label=DLAB[d] if len(ds) > 1 else None)
        ax.set_title(f"{name}\nSpearman ρ with MOS = {rho:+.2f}", fontsize=7.5)
        ax.set_xlabel("feature value")
        if len(ds) > 1:
            ax.legend(loc="best")
    axes[0].set_ylabel("MOS")
    save(fig, "10_lineart_cues_vs_mos")

print("Done. Use the PDFs in LaTeX:  \\includegraphics[width=\\columnwidth]{plots/02_per_distortion_srocc.pdf}")