# ============================================================================
# DA-TransQA v5 — CONFIRMATION RUN (fresh seed) on BeDAMQA  —  cartoon split
#   Identical to da_transqa_v5_unsup.py except: SEED = 1000 (new cartoon groups, new validation
#   cartoons, new weight init), new output file names, and one extra ablation row
#   'ens_no_deep' (ensemble WITHOUT the transformer). The main result stays 'ensemble'.
#   Report THIS run's numbers in the paper: nothing was tuned on these splits.
#   config -> MOS table -> frames + handcrafted features -> pristine Reference videos
#   -> pretrained backbone features -> transformer -> losses/metrics -> 10x cartoon-split loop
#
# Everything from v4 (pretrained backbone, same-cartoon ranking loss, both per-distortion
# tables), plus:
#   1. UNSUPERVISED pristine model ("unsup"). The clean Reference videos have no human score,
#      but they show what "perfect" looks like. Per iteration we fit a statistical model of
#      "pristine" from the Reference videos of the NON-test cartoons only, and score every
#      video by how far it is from pristine (handcrafted + backbone space). No human scores
#      are used: this is our opinion-unaware score, the same family as RF-AVQA, but leak-free.
#   2. DEGRADATION-AWARE EXPERTS ("experts"), in the spirit of DIIVINE: stage 1 guesses the type
#      of damage (blur / freeze / hang / H.264 / H.265), stage 2 has one expert per damage type.
#      Score = experts weighted by the stage-1 guess. One expert sees only H.265 videos, so it
#      learns H.265 severity instead of being drowned out by the easy distortions.
#   3. SELF-SUPERVISED ranking for the transformer (RankIQA-style). Each training clip gets an
#      extra-damaged copy made on the fly (blur / JPEG / down-up-scaling / frame freeze), and the
#      copy must score lower than the original. No labels needed -> lots of extra training signal.
#   4. Degradation-aware MIXTURE-OF-EXPERTS head in the transformer: it predicts the damage type
#      (trained with the folder names, which are free labels, not human scores) and mixes one
#      small expert per type.
#   5. SMOKE TEST mode: run once with SMOKE=1 (1 iteration, 2 epochs) to catch errors in minutes.
#
#   Rows printed:  unsup (no human scores) | svr | cnn | experts | deep | hybrid3 (v4's main)
#                  ensemble = average of svr, cnn, experts, deep   <- MAIN, fixed in advance
#
# Saves to NEW files (older results are not touched):
#   da_transqa_results_v5_confirm_seed1000.csv
#   da_transqa_per_distortion_within_v5_confirm_seed1000.csv
#   da_transqa_per_distortion_pooled_v5_confirm_seed1000.csv
#
# Run:
#   SMOKE=1 python da_transqa_v5_confirm.py      # quick crash test (~10-20 min first time: fills caches)
#   python da_transqa_v5_confirm.py              # full 10-iteration run
# (VS Code / Jupyter: each "# %%" block is a cell; set SMOKE in the config cell instead.)
# ============================================================================

# %% 1. Config ---------------------------------------------------------------
import os, re, glob, time, hashlib
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage
from scipy.special import gamma as gamma_fn
from sklearn.svm import SVR, SVC
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

RUN_NAME = "v5_confirm_seed1000"    # used in the output file names (old CSVs are not overwritten)

ROOT_DIR = os.environ.get(
    "AVQA_ROOT", "/Users/krd/Documents/Semester 5/Animated VQA/DatasetMain")
DISTORTION_FOLDERS = ["Reference", "Blur", "FrameFreeze", "FrameHang", "H264", "H265"]
MAX_VIDEOS_PER_FOLDER = None        # must stay None: the cartoon split needs all 15 cartoons

# data shape
IMG_SIZE       = 64                 # crop size, cut at NATIVE resolution (no resizing, so blur and
                                    # H.265 artefacts stay visible). PATCH must divide it.
N_CROPS        = 5                  # crops per frame: centre + 4 quadrant centres (max 5)
N_FRAMES       = 16
K_SEGMENTS     = 4
FRAMES_PER_SEG = N_FRAMES // K_SEGMENTS
FRAME_STRIDE   = 2                  # gap between frames *inside* a segment -> short clips, so
                                    # frame diffs show real motion / freezes instead of 30-frame jumps
FREEZE_EPS     = 2e-5               # mean |diff| at 160x90 below this = repeated frame (measured on BeDAMQA)

# extraction
N_WORKERS      = min(8, os.cpu_count() or 4)   # videos decoded in parallel
EXTRACT_BATCH  = 32                            # videos per batch (progress + memory)
CACHE_DIR      = os.path.join(os.path.expanduser("~"), ".avqa_cache")
USE_CACHE      = True

# training
BATCH_SIZE     = 16                 # bigger batch = more pairs for the ranking loss
EPOCHS         = 100                # per iteration; early stopping on Valid SROCC
PATIENCE       = 15
LEARNING_RATE  = 3e-4
WEIGHT_DECAY   = 0.05               # was 1e-4, which was too small to stop memorising
RANK_WEIGHT    = 0.5
VERBOSE_EPOCHS = True               # True = print val PLCC/SROCC every epoch
N_FOLDS        = 5                  # 15 cartoons -> 5 groups of 3; each group is the test set once
N_VAL_CARTOONS = 3                  # cartoons held out for validation (early stopping, SVR's C)
N_ITERATIONS   = 10                 # = 2 rounds x N_FOLDS
SEED           = 1000               # fresh seed for the confirmation run (was 42)
SVR_C_GRID     = [0.1, 0.3, 1.0, 3.0, 10.0]    # chosen on the validation cartoons, never on test
GROUP_RANK_WEIGHT = 0.5             # NEW: extra ranking loss on pairs from the SAME cartoon (0 = off)
N_DEEP_SEEDS   = 1                  # NEW: transformers trained per iteration and averaged (3 = smoother, 3x slower)

# pretrained backbone (NEW)
USE_BACKBONE   = True               # frozen ImageNet EfficientNetB0 features
BB_CROP        = 224                # native-resolution crop size fed to the backbone (no resizing)
BB_BATCH       = 32                 # images per backbone forward pass
PCA_GRID       = [8, 16, 32]        # backbone features are squeezed with PCA; size picked on validation

# v5 additions
SMOKE          = os.environ.get("SMOKE", "0") == "1"   # SMOKE=1 -> 1 iteration, 2 epochs (crash test)
SSL_WEIGHT     = 0.5                # self-supervised: extra-damaged copy must score lower (0 = off)
TYPE_WEIGHT    = 0.3                # damage-type classification (labels = folder names, not human scores)
UNSUP_PCA      = 16                 # size of the "pristine" Gaussian model in backbone space
EXPERT_PCA     = 16                 # backbone PCA size used by the experts
EXPERT_CLF_C   = [0.3, 1.0, 3.0, 10.0]   # damage-type classifier strength (picked on validation)

# model
D_MODEL           = 32              # was 64 -> roughly 4x fewer weights, less room to memorise
N_HEADS           = 4
PATCH             = 8               # IMG_SIZE must be divisible by this
N_SPATIAL_LAYERS  = 1
N_TEMPORAL_LAYERS = 2
FF_MULT           = 2
DROPOUT           = 0.2

assert N_FRAMES == K_SEGMENTS * FRAMES_PER_SEG, "N_FRAMES must be divisible by K_SEGMENTS"
assert IMG_SIZE % PATCH == 0, "IMG_SIZE must be divisible by PATCH"
assert 1 <= N_CROPS <= 5, "N_CROPS must be between 1 and 5"
assert N_ITERATIONS % N_FOLDS == 0, "N_ITERATIONS must be a multiple of N_FOLDS"
assert MAX_VIDEOS_PER_FOLDER is None, "Cartoon split needs all videos (set MAX_VIDEOS_PER_FOLDER = None)"
if SMOKE:
    EPOCHS, PATIENCE = 2, 1
    print(">>> SMOKE TEST: 1 iteration, 2 epochs — only checks that everything runs <<<")
print("TF", tf.__version__, "| devices:", [d.device_type for d in tf.config.list_physical_devices()])


# %% 2. MOS table + video list ----------------------------------------------
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".y4m")

def find_mos_file(root_dir):
    c = glob.glob(os.path.join(root_dir, "*.xlsx")) + glob.glob(os.path.join(root_dir, "*.xls"))
    if not c:
        raise FileNotFoundError(f"No .xlsx/.xls MOS file found under {root_dir}")
    return c[0]

def load_mos_table(root_dir):
    df = pd.read_excel(find_mos_file(root_dir))
    low = {c.lower(): c for c in df.columns}
    fname_col = next((low[c] for c in low if "file" in c or "name" in c or "video" in c), None)
    score_col = next((low[c] for c in low if "mos" in c or "score" in c), None)
    if fname_col is None or score_col is None:
        raise ValueError(f"Could not detect filename/score columns. Found: {list(df.columns)}")
    df = df[[fname_col, score_col]].rename(columns={fname_col: "filename", score_col: "mos"})
    df["stem"] = df["filename"].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
    return df

def list_videos(root_dir, folders, max_per_folder=None):
    rows = []
    for folder in folders:
        fp = os.path.join(root_dir, folder)
        if not os.path.isdir(fp):
            print(f"  (skipping '{folder}' — not found at {fp})")
            continue
        paths = sorted(p for p in glob.glob(os.path.join(fp, "*")) if p.lower().endswith(VIDEO_EXTS))
        if max_per_folder is not None:
            paths = paths[:max_per_folder]
        rows += [{"path": p, "stem": os.path.splitext(os.path.basename(p))[0], "distortion": folder}
                 for p in paths]
    return pd.DataFrame(rows)

def build_video_mos_table(root_dir, folders, max_per_folder=None):
    videos, mos = list_videos(root_dir, folders, max_per_folder), load_mos_table(root_dir)
    merged = videos.merge(mos[["stem", "mos"]], on="stem", how="inner")
    if len(videos) - len(merged):
        print(f"Warning: {len(videos) - len(merged)}/{len(videos)} videos had no MOS row.")
        no_mos = videos.loc[~videos["stem"].isin(mos["stem"]) & (videos["distortion"] != "Reference"), "stem"]
        if len(no_mos):                                  # Reference videos have no MOS in BeDAMQA, that's fine
            print("   distorted videos without a MOS row:", no_mos.tolist())
    no_vid = mos.loc[~mos["stem"].isin(videos["stem"]), "filename"]
    if len(no_vid):
        print(f"Warning: {len(no_vid)} MOS rows have no matching video file:", no_vid.tolist()[:10])
    dup = merged["stem"].duplicated().sum()
    if dup:
        print(f"Warning: {dup} duplicate file stems across folders — MOS may be matched to the wrong video.")
    return merged.reset_index(drop=True)

table = build_video_mos_table(ROOT_DIR, DISTORTION_FOLDERS, MAX_VIDEOS_PER_FOLDER)
print(f"Matched {len(table)} videos with MOS scores.")
print(table["distortion"].value_counts().to_string())


# %% 3. Frame + feature extraction (batched, parallel, cached) ----------------
def sample_indices(total):
    """K_SEGMENTS short clips spread over the video, each FRAMES_PER_SEG frames FRAME_STRIDE apart."""
    span = (FRAMES_PER_SEG - 1) * FRAME_STRIDE
    starts = np.linspace(0, max(total - 1 - span, 0), K_SEGMENTS).astype(int)
    return np.minimum((starts[:, None] + np.arange(FRAMES_PER_SEG) * FRAME_STRIDE).reshape(-1), total - 1)


def crop_positions(h, w, size=IMG_SIZE):
    """Top-left corners of the N_CROPS native-resolution crops (size x size): centre + 4 quadrant
    centres, snapped to the 16-px grid so the model's patches line up with codec blocks."""
    centres = [(0.5, 0.5), (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)][:N_CROPS]
    snap = lambda v, hi: int(min(max(round(v / 16) * 16, 0), hi))
    return [(snap(fy * h - size / 2, h - size), snap(fx * w - size / 2, w - size))
            for fy, fx in centres]


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ---- inter-frame SSIM (the paper's V_t) ----
def _ssim(a, b):                                    # a, b: gray float32 0..1, same size
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda z: cv2.GaussianBlur(z, (7, 7), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    va, vb = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


# ---- 3D Gaussian second-derivative entropies (the paper's E_S, without their pristine model) ----
_G3_ORDERS = [(0, 0, 2), (0, 2, 0), (2, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1)]   # xx yy tt xy yt tx (axes t,y,x)
_G3_RANGE = 0.25                                    # fixed histogram range, so "flat" really means low entropy

def _entropy(v, bins=64):
    h, _ = np.histogram(np.clip(v, -_G3_RANGE, _G3_RANGE), bins=bins, range=(-_G3_RANGE, _G3_RANGE))
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())

def gauss3d_feats(vol):                             # vol: (T, H, W) float32 0..1, one crop over time
    H = [_entropy(ndimage.gaussian_filter(vol, sigma=1.0, order=o)) for o in _G3_ORDERS]
    hxx, hyy, htt, hxy, hyt, htx = H
    M = np.array([[hxx, hxy, htx], [hxy, hyy, hyt], [htx, hyt, htt]])
    return np.array(H + [np.linalg.eigvalsh(M)[-1]], np.float32)     # 6 entropies + lambda_max


# ---- natural-scene statistics (BRISQUE features; NIQE is built on the same ones) ----
_ALPHAS = np.arange(0.05, 10.0, 0.001)              # starts low: flat anime colour makes MSCN very peaky
_R_GGD = gamma_fn(1 / _ALPHAS) * gamma_fn(3 / _ALPHAS) / gamma_fn(2 / _ALPHAS) ** 2
_R_AGGD = gamma_fn(2 / _ALPHAS) ** 2 / (gamma_fn(1 / _ALPHAS) * gamma_fn(3 / _ALPHAS))

def _mscn(g):                                       # mean-subtracted contrast-normalised map
    mu = cv2.GaussianBlur(g, (7, 7), 7 / 6)
    sd = np.sqrt(np.abs(cv2.GaussianBlur(g * g, (7, 7), 7 / 6) - mu * mu))
    return (g - mu) / (sd + 1.0 / 255)

def _ggd(x):
    rho = np.mean(x ** 2) / (np.mean(np.abs(x)) ** 2 + 1e-12)
    return _ALPHAS[np.argmin((_R_GGD - rho) ** 2)], np.mean(x ** 2)

def _aggd(x):
    left, right = x[x < 0], x[x >= 0]
    sl = np.sqrt(np.mean(left ** 2)) if left.size else 1e-6
    sr = np.sqrt(np.mean(right ** 2)) if right.size else 1e-6
    g = sl / (sr + 1e-12)
    r = np.mean(np.abs(x)) ** 2 / (np.mean(x ** 2) + 1e-12)
    R = r * (g ** 3 + 1) * (g + 1) / ((g ** 2 + 1) ** 2)
    a = _ALPHAS[np.argmin((_R_AGGD - R) ** 2)]
    eta = (sr - sl) * gamma_fn(2 / a) / gamma_fn(1 / a) * np.sqrt(gamma_fn(1 / a) / gamma_fn(3 / a))
    return [a, eta, sl ** 2, sr ** 2]

def nss_features(g):                                # full-res gray float32 0..1 -> 36 numbers
    out = []
    for _ in range(2):                              # 2 scales
        m = _mscn(g)
        out += list(_ggd(m.ravel()))
        for sh in [(0, 1), (1, 0), (1, 1), (1, -1)]:   # horizontal, vertical, 2 diagonal neighbours
            out += _aggd((m * np.roll(m, sh, axis=(0, 1))).ravel())
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return np.array(out, np.float32)


# ---- animation line-art cues (ours): blur softens outlines, compression rings next to them ----
_K3, _K9 = np.ones((3, 3), np.uint8), np.ones((9, 9), np.uint8)

def _grad(img):
    return cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1))

def line_art_cues(g):                               # full-res gray float32 0..1
    mag = _grad(g)
    edges = (mag > np.percentile(mag, 98)).astype(np.uint8)          # strongest 2% = outlines
    on = edges.astype(bool)
    if not on.any():
        return 0.0, float(np.log(1e-8))
    mag_b = _grad(cv2.blur(g, (9, 9)))
    sharp = float(np.mean(np.maximum(mag[on] - mag_b[on], 0) / (mag[on] + 1e-6)))   # blurry -> ~0
    band = (cv2.dilate(edges, _K9) - cv2.dilate(edges, _K3)).astype(bool)          # flat strip beside outlines
    if not band.any():
        return sharp, float(np.log(1e-8))
    var = np.maximum(cv2.blur(g * g, (3, 3)) - cv2.blur(g, (3, 3)) ** 2, 0)
    return sharp, float(np.log(np.median(var[band]) + 1e-8))          # ringing / speckle


FEATURE_NAMES = [
    "freeze_ratio", "longest_freeze", "freeze_events", "hang_jump", "motion_mean", "motion_cv",
    "ssim_mean", "ssim_p10",
    "lap_mean", "lap_std", "grad_mean", "blockiness",
    "outline_sharpness", "outline_ringing",
    "H_xx", "H_yy", "H_tt", "H_xy", "H_yt", "H_tx", "lambda_max",
] + [f"nss{s}_{k}" for s in range(2) for k in range(18)]
N_FEATS = len(FEATURE_NAMES)                        # 57


def video_features(motion, ssim, grays, vol):
    """Whole-video, label-free cues.
    motion: (n_all-1,) mean |diff| of consecutive frames at 160x90
    ssim:   (n_all-1,) SSIM of consecutive frames at 320x180
    grays:  a few full-res gray frames (one per segment), float32 0..1
    vol:    (n_all, N_CROPS, IMG_SIZE, IMG_SIZE) native-res gray crops of EVERY frame, float32 0..1"""
    frozen = motion < FREEZE_EPS
    moving = motion[~frozen]
    med = float(np.median(moving)) if len(moving) else 1e-4
    ends = np.where(frozen[:-1] & ~frozen[1:])[0] + 1          # freeze ends; hang = big jump after
    jump = max([motion[e] / (med + 1e-6) for e in ends], default=1.0)
    lap = [np.log(cv2.Laplacian(g, cv2.CV_32F).var() + 1e-8) for g in grays]
    grad = [np.mean(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)))
            for g in grays]
    block = []
    for g in grays:                                 # |diff| on 8px block borders vs. elsewhere
        dx = np.abs(np.diff(g, axis=1)).mean(0)
        dy = np.abs(np.diff(g, axis=0)).mean(1)
        bx = dx[7::8].mean() / (np.delete(dx, np.s_[7::8]).mean() + 1e-6)
        by = dy[7::8].mean() / (np.delete(dy, np.s_[7::8]).mean() + 1e-6)
        block.append((bx + by) / 2)
    la = np.array([line_art_cues(g) for g in grays])
    g3 = np.mean([gauss3d_feats(vol[:, c]) for c in range(vol.shape[1])], axis=0)
    nss = np.mean([nss_features(g) for g in grays], axis=0)
    n = len(motion) + 1
    f = np.concatenate([[
        frozen.mean(), _longest_run(frozen) / n, len(ends) / n * 100, np.log(jump + 1e-6),
        np.log1p(1000 * motion.mean()), np.log1p(motion.std() / (motion.mean() + 1e-6)),
        np.mean(ssim), np.percentile(ssim, 10),
        np.mean(lap), np.std(lap), np.mean(grad) * 10, np.mean(block),
        la[:, 0].mean(), la[:, 1].mean(),
    ], g3, nss]).astype(np.float32)
    assert len(f) == N_FEATS, (len(f), N_FEATS)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def extract_video(video_path):
    """Decodes the whole video once. Returns
       frames (N_CROPS, N_FRAMES, IMG_SIZE, IMG_SIZE, 3) uint8 RGB -- native-res crops, NOT resized
       feats  (N_FEATS,) float32                                  -- whole-video handcrafted cues"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(sample_indices(total).tolist()) if total > 0 else None

    pos, kept, grays, crops_all, motion, ssim = None, {}, {}, [], [], []
    prev_small = prev_mid = None
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if pos is None:
            pos = crop_positions(*f.shape[:2])
        g8 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g8, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mid = cv2.resize(g8, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if prev_small is not None:
            motion.append(float(np.abs(small - prev_small).mean()))
            ssim.append(_ssim(mid, prev_mid))
        prev_small, prev_mid = small, mid
        crops_all.append(np.stack([g8[y0:y0 + IMG_SIZE, x0:x0 + IMG_SIZE] for y0, x0 in pos]))
        if wanted is None or i in wanted:           # unknown length -> keep all, subsample below
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            kept[i] = np.stack([rgb[y0:y0 + IMG_SIZE, x0:x0 + IMG_SIZE] for y0, x0 in pos])
            grays[i] = g8
        i += 1
    cap.release()
    if len(motion) < 2:
        raise ValueError("too few frames could be read")
    idx = sample_indices(total if total > 0 else i)
    out, prev = [], next(iter(kept.values()))
    for t in idx:                                   # missing frame -> reuse previous
        prev = kept.get(int(t), prev)
        out.append(prev)
    frames = np.stack(out, axis=1)                  # (N_CROPS, N_FRAMES, S, S, 3)
    g = [grays[int(t)].astype(np.float32) / 255.0 for t in idx if int(t) in grays][::FRAMES_PER_SEG]
    vol = np.stack(crops_all).astype(np.float32) / 255.0                   # (n_all, N_CROPS, S, S)
    return frames, video_features(np.array(motion), np.array(ssim), g, vol)


def _cache_path(tbl):
    cfg = f"|{N_FRAMES}|{IMG_SIZE}|{N_CROPS}|{K_SEGMENTS}|{FRAME_STRIDE}|{FREEZE_EPS}|{N_FEATS}|v3"
    key = hashlib.md5(("|".join(sorted(tbl["path"])) + cfg).encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"v3_frames_N{N_FRAMES}_S{IMG_SIZE}_C{N_CROPS}_{len(tbl)}_{key}.npz")


def extract_dataset(tbl):
    """Batch-process every video. Returns X (uint8), F (float32), y (float32), kept table. Cached on disk."""
    cache = _cache_path(tbl)
    if USE_CACHE and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"Loaded cached frames from {cache}")
        return z["X"], z["F"], z["y"], tbl[tbl["path"].isin(set(z["paths"].tolist()))].reset_index(drop=True)

    paths = tbl["path"].tolist()
    X = np.zeros((len(paths), N_CROPS, N_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    F = np.zeros((len(paths), N_FEATS), dtype=np.float32)
    ok_mask = np.zeros(len(paths), dtype=bool)
    failed, t0 = [], time.time()

    def work(i):
        try:
            return (i, *extract_video(paths[i]), None)
        except Exception as e:
            return i, None, None, str(e)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for start in range(0, len(paths), EXTRACT_BATCH):         # one batch at a time
            idxs = range(start, min(start + EXTRACT_BATCH, len(paths)))
            for i, frames, feats, err in ex.map(work, idxs):
                if err is None:
                    X[i], F[i], ok_mask[i] = frames, feats, True
                else:
                    failed.append((paths[i], err))
            done = min(start + EXTRACT_BATCH, len(paths))
            print(f"  extracted {done}/{len(paths)} videos  ({time.time() - t0:.0f}s)")

    for p, e in failed:
        print(f"  skipped {p}: {e}")
    kept = tbl[ok_mask].reset_index(drop=True)
    X, F, y = X[ok_mask], F[ok_mask], kept["mos"].to_numpy(dtype=np.float32)
    if USE_CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, X=X, F=F, y=y, paths=np.array(kept["path"].tolist()))
        print(f"Cached to {cache}")
    return X, F, y, kept


X_u8, F_raw, y, table = extract_dataset(table)     # X stays uint8 (saves ~1.5 GB); converted per batch
print(f"Loaded {len(X_u8)} videos. X: {X_u8.shape}  feats: {F_raw.shape}  y range: {y.min():.2f}-{y.max():.2f}")

# ---- which original cartoon does each video come from? ----
# BeDAMQA names START with the cartoon number: "05_H264_0200" -> cartoon "05".
# Only the START of the name is used: the old code searched the whole name, and "0200" contains "02".
_norm = lambda s: s.lower().replace(" ", "").replace("_", "").replace("-", "")
_ref_dir = os.path.join(ROOT_DIR, "Reference")
_refs = sorted((os.path.splitext(f)[0] for f in os.listdir(_ref_dir) if f.lower().endswith(VIDEO_EXTS)),
               key=len, reverse=True)                      # longer names first ("SoftRain" before "Soft")
_ref_by_num = {int(r): r for r in _refs if r.isdigit()}    # "01".."15" -> 1..15

def cartoon_of(stem):
    m = re.match(r"\s*(\d+)", stem)                        # leading number, e.g. "05" in "05_H264_0200"
    if m and int(m.group(1)) in _ref_by_num:
        return _ref_by_num[int(m.group(1))]
    return next((r for r in _refs if _norm(stem).startswith(_norm(r))), None)   # text names: start only

table["content"] = table["stem"].map(cartoon_of)
if table["content"].isna().any():
    print("Could not match these videos to a cartoon (send me these names):")
    print(table.loc[table["content"].isna(), ["distortion", "stem"]].head(10).to_string())
    raise ValueError("cartoon matching failed")
_ct = pd.crosstab(table["content"], table["distortion"])
print(f"{table['content'].nunique()} cartoons found (should be 15). Videos per cartoon and distortion "
      f"(every cell should be 5):\n{_ct.to_string()}")
_bad = _ct[(_ct != 5).any(axis=1)]
if len(_bad):
    print(f"Warning: these cartoons don't have exactly 5 videos per distortion:\n{_bad.to_string()}")


# %% 3b. Pretrained backbone: definitions ------------------------------------
# A frozen ImageNet network already "knows" edges, textures and fine detail from millions of
# photos. We only READ its features (nothing is trained), so it can't memorise our videos.
# For each layer we keep the mean and std of every channel over the crop (VSFA-style pooling).
# v5 keeps one vector PER CROP (not just per video): the unsupervised pristine model needs them.

def bb_frame_indices(total):
    """One frame from the middle of each segment -> K_SEGMENTS frames."""
    return sample_indices(total).reshape(K_SEGMENTS, FRAMES_PER_SEG)[:, FRAMES_PER_SEG // 2]


def extract_bb_crops(video_path):
    """(K_SEGMENTS * N_CROPS, BB_CROP, BB_CROP, 3) uint8 RGB native-resolution crops."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(bb_frame_indices(total).tolist()) if total > 0 else None
    last = max(want) if want else None
    kept, i = {}, 0
    while cap.grab():
        if want is None or i in want:
            ok, f = cap.retrieve()
            if ok:
                kept[i] = f
        i += 1
        if last is not None and i > last:
            break
    cap.release()
    if not kept:
        raise ValueError("no frames could be read")
    out, prev = [], next(iter(kept.values()))
    for t in bb_frame_indices(total if total > 0 else i):
        prev = kept.get(int(t), prev)                    # missing frame -> reuse previous
        rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        c = min(BB_CROP, h, w)
        for y0, x0 in crop_positions(h, w, c):
            crop = rgb[y0:y0 + c, x0:x0 + c]
            if c != BB_CROP:                             # only for videos smaller than 224 px
                crop = cv2.resize(crop, (BB_CROP, BB_CROP), interpolation=cv2.INTER_LINEAR)
            out.append(crop)
    return np.stack(out)


_BB_NET = None

def build_backbone():
    """Frozen EfficientNetB0 (ImageNet weights). Returns a model that outputs the feature maps of
    the last residual layer of stages 2-6 plus the final layer (fine -> coarse detail)."""
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                             input_shape=(BB_CROP, BB_CROP, 3))
    base.trainable = False
    names = [l.name for l in base.layers]
    picks = []
    for stage in ("block2", "block3", "block4", "block5", "block6"):
        adds = [n for n in names if n.startswith(stage) and n.endswith("_add")]
        if adds:
            picks.append(adds[-1])
    picks.append(names[-1])
    print("  backbone layers used:", picks)
    return keras.Model(base.input, [base.get_layer(n).output for n in picks])


def pool_maps(maps):
    """List of (n, h, w, c) feature maps -> (n, sum of 2c): spatial mean and std of every channel."""
    return np.concatenate([np.concatenate([m.mean((1, 2)), m.std((1, 2))], -1) for m in maps], -1)


def _bb_cache_path(tbl):
    cfg = f"|effb0|{BB_CROP}|{N_CROPS}|{K_SEGMENTS}|{FRAMES_PER_SEG}|{FRAME_STRIDE}|bb2"
    key = hashlib.md5(("|".join(sorted(tbl["path"])) + cfg).encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"bb2_effb0_C{BB_CROP}_{len(tbl)}_{key}.npz")


def backbone_crops(tbl):
    """(n_videos, K_SEGMENTS * N_CROPS, n_features): one backbone vector per crop. Cached on disk."""
    global _BB_NET
    cache = _bb_cache_path(tbl)
    paths = tbl["path"].tolist()
    if USE_CACHE and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if [str(p) for p in z["paths"]] == paths:
            print(f"Loaded cached backbone features from {cache}")
            return z["B"]
    if _BB_NET is None:
        _BB_NET = build_backbone()
    B, failed, t0 = None, [], time.time()

    def work(i):
        try:
            return i, extract_bb_crops(paths[i]), None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for start in range(0, len(paths), EXTRACT_BATCH):         # decode in parallel, run net in order
            for i, crops, err in ex.map(work, range(start, min(start + EXTRACT_BATCH, len(paths)))):
                if err is not None:
                    failed.append((i, err))
                    continue
                maps = _BB_NET.predict(crops.astype(np.float32), batch_size=BB_BATCH, verbose=0)
                maps = list(maps) if isinstance(maps, (list, tuple)) else [maps]
                v = pool_maps([np.asarray(m, dtype=np.float32) for m in maps])   # (n_crops, n_features)
                if B is None:
                    B = np.zeros((len(paths),) + v.shape, np.float32)
                B[i] = v
            print(f"  backbone {min(start + EXTRACT_BATCH, len(paths))}/{len(paths)} videos  "
                  f"({time.time() - t0:.0f}s)")
    if B is None:
        raise ValueError("backbone failed on every video")
    ok_rows = np.setdiff1d(np.arange(len(paths)), [j for j, _ in failed])
    for i, e in failed:                                  # rare: fill with the average so nothing crashes
        print(f"  backbone skipped {paths[i]}: {e}")
        B[i] = B[ok_rows].mean(0)
    if USE_CACHE and not failed:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, B=B, paths=np.array(paths))
        print(f"Cached to {cache}")
    return B


# %% 3c. Pristine Reference videos + backbone features (once, then cached) ----
# The clean Reference videos have no human score, but they show what "perfect" looks like.
# Section 7 builds a model of "pristine" from them, per iteration, using ONLY the non-test
# cartoons, so the test cartoons stay completely unseen.
ref_table = list_videos(ROOT_DIR, ["Reference"])
F_ref, ref_cartoon = None, None
if len(ref_table):
    ref_table["mos"] = 5.0                               # dummy, never used
    ref_table["content"] = ref_table["stem"].map(cartoon_of)
    ref_table = ref_table[ref_table["content"].notna()].reset_index(drop=True)
if len(ref_table):
    try:
        _, F_ref, _, ref_table = extract_dataset(ref_table)
        ref_cartoon = ref_table["content"].values
        print(f"Reference (pristine) videos: {len(ref_table)}")
    except Exception as e:
        print(f"Reference videos skipped ({type(e).__name__}: {e}). Running without 'unsup'.")
        F_ref = None
HAS_REF = F_ref is not None and len(F_ref) > 0

B_crops, B_ref_crops = None, None
if USE_BACKBONE:
    try:
        B_crops = backbone_crops(table)
        if HAS_REF:
            B_ref_crops = backbone_crops(ref_table)
        print(f"Backbone features: {B_crops.shape}  (videos x crops x features)")
    except Exception as e:                               # e.g. no internet for the one-time weight download
        print(f"Backbone skipped ({type(e).__name__}: {e}). Running without the 'cnn' model.")
        B_crops, B_ref_crops = None, None
B_raw = B_crops.mean(1) if B_crops is not None else None   # one vector per video


# %% 4. Model: DA-TransQA ----------------------------------------------------
TYPES = sorted(table["distortion"].unique())             # damage types in the data (folder names)
N_TYPES = len(TYPES)
# 1 inside a clip, 0 on the first frame of each clip (diff across clips is a jump, not motion)
_IN_SEG = (np.arange(N_FRAMES) % FRAMES_PER_SEG != 0).astype(np.float32)

def _frame_diffs(frames):                       # (B,T,H,W,3) -> same shape
    d = tf.abs(frames[:, 1:] - frames[:, :-1])
    d = tf.concat([tf.zeros_like(d[:, :1]), d], axis=1)
    return d * tf.reshape(_IN_SEG, (1, -1, 1, 1, 1))

def _animation_maps(x):                         # (N,H,W,3) -> (N,H,W,3)
    """Anime cues: line-art edges, flat-colour regions, saturation."""
    gray = tf.reduce_mean(x, -1, keepdims=True)
    g = tf.image.sobel_edges(gray)
    edge = tf.sqrt(tf.reduce_sum(tf.square(g), -1) + 1e-6)
    mu = tf.nn.avg_pool2d(gray, 3, 1, "SAME")
    var = tf.nn.relu(tf.nn.avg_pool2d(gray * gray, 3, 1, "SAME") - mu * mu)
    flat = tf.exp(-50.0 * var)
    sat = tf.reduce_max(x, -1, keepdims=True) - tf.reduce_min(x, -1, keepdims=True)
    return tf.concat([edge, flat, sat], -1)


class TransformerBlock(layers.Layer):
    """Pre-norm block. With `context`, does cross-attention (x attends to context)."""
    def __init__(self, d, heads, cross=False, drop=DROPOUT, **kw):
        super().__init__(**kw)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln_ctx = layers.LayerNormalization(epsilon=1e-6) if cross else None
        self.attn = layers.MultiHeadAttention(heads, d // heads, dropout=drop)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = layers.Dense(d * FF_MULT, activation="gelu")
        self.ff2 = layers.Dense(d)
        self.drop = layers.Dropout(drop)

    def call(self, x, context=None, training=None):
        h = self.ln1(x)
        kv = h if context is None else self.ln_ctx(context)
        x = x + self.drop(self.attn(h, kv), training=training)
        return x + self.drop(self.ff2(self.ff1(self.ln2(x))), training=training)


class AttentionPool(layers.Layer):
    """Learned-query attention pooling: (N, L, d) -> (N, d)."""
    def __init__(self, d, heads, **kw):
        super().__init__(**kw)
        self.d = d
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(heads, d // heads)

    def build(self, input_shape):
        self.query = self.add_weight(name="query", shape=(1, 1, self.d), initializer="glorot_uniform")

    def call(self, x):
        q = tf.tile(tf.convert_to_tensor(self.query), [tf.shape(x)[0], 1, 1])
        return tf.squeeze(self.attn(q, self.ln(x)), 1)


class StreamEncoder(layers.Layer):
    """Patch embed -> spatial self-attention (-> optional cross-attn) -> attention pool."""
    def __init__(self, d, heads, n_layers, patch, img_size, cross=False, **kw):
        super().__init__(**kw)
        self.d, self.n_tok = d, (img_size // patch) ** 2
        self.embed = layers.Conv2D(d, patch, strides=patch)
        self.blocks = [TransformerBlock(d, heads) for _ in range(n_layers)]
        self.cross_block = TransformerBlock(d, heads, cross=True) if cross else None
        self.ln = layers.LayerNormalization(epsilon=1e-6)
        self.pool = AttentionPool(d, heads)

    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(1, self.n_tok, self.d),
                                   initializer=keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x, context=None):                     # x: (N,H,W,C)
        t = tf.reshape(self.embed(x), (-1, self.n_tok, self.d)) + self.pos
        for b in self.blocks:
            t = b(t)
        if self.cross_block is not None:
            t = self.cross_block(t, context=context)
        t = self.ln(t)
        return t, self.pool(t)


class TemporalDegradation(layers.Layer):
    """Per-frame temporal-degradation cues -> embedding + degradation score in [0,1].
    Cues: log motion, relative motion, jerk, freeze indicator, sharpness, sharpness change."""
    def __init__(self, d, **kw):
        super().__init__(**kw)
        self.embed = keras.Sequential([layers.Dense(d, activation="gelu"), layers.Dense(d)])
        self.score = layers.Dense(1, activation="sigmoid")

    def call(self, frames, edge_mean):                   # (B,T,H,W,3), (B,T)
        in_seg = tf.constant(_IN_SEG > 0)[None, :]
        m = tf.reduce_mean(tf.abs(frames[:, 1:] - frames[:, :-1]), axis=[2, 3, 4])
        m = tf.concat([m[:, :1], m], 1)
        m = tf.where(in_seg, m, tf.concat([m[:, 1:], m[:, -1:]], 1))   # clip start: use next diff
        m_avg = tf.reduce_mean(m, 1, keepdims=True) + 1e-4
        m_prev = tf.where(in_seg, tf.concat([m[:, :1], m[:, :-1]], 1), m)
        e_prev = tf.where(in_seg, tf.concat([edge_mean[:, :1], edge_mean[:, :-1]], 1), edge_mean)
        feats = tf.stack([
            tf.math.log1p(100.0 * m),
            tf.minimum(m / m_avg, 5.0),
            tf.minimum(tf.abs(m - m_prev) / m_avg, 5.0),
            tf.exp(-m / 0.005),
            edge_mean * 5.0,
            tf.abs(edge_mean - e_prev) * 20.0,
        ], axis=-1)
        emb = self.embed(feats)
        return emb, tf.squeeze(self.score(emb), -1)


class CrossStreamFusion(layers.Layer):
    """Per frame: S, M, A, D tokens attend to each other, then merge into one frame token."""
    def __init__(self, d, heads, T, **kw):
        super().__init__(**kw)
        self.d, self.T = d, T
        self.block = TransformerBlock(d, heads)
        self.proj = layers.Dense(d, activation="gelu")

    def build(self, input_shape):
        self.stream_emb = self.add_weight(name="stream_emb", shape=(1, 4, self.d),
                                          initializer=keras.initializers.RandomNormal(stddev=0.02))

    def call(self, s, m, a, dg):                         # each (B,T,d)
        x = tf.reshape(tf.stack([s, m, a, dg], axis=2), (-1, 4, self.d)) + self.stream_emb
        x = self.block(x)
        return self.proj(tf.reshape(x, (-1, self.T, 4 * self.d)))


class DegradationAwareAttention(layers.Layer):
    """Temporal self-attention with a learned (T x T) per-head bias and a per-key
    degradation bias, so frames flagged as degraded get more attention."""
    def __init__(self, d, heads, T, drop=DROPOUT, **kw):
        super().__init__(**kw)
        self.d, self.h, self.T = d, heads, T
        self.qkv = layers.Dense(3 * d)
        self.out = layers.Dense(d)
        self.drop = layers.Dropout(drop)

    def build(self, input_shape):
        self.temp_bias = self.add_weight(name="temporal_bias", shape=(self.h, self.T, self.T),
                                         initializer="zeros")
        self.deg_scale = self.add_weight(name="deg_scale", shape=(self.h,),
                                         initializer=keras.initializers.Constant(1.0))

    def call(self, x, deg, training=None):               # x (B,T,d), deg (B,T)
        dh = self.d // self.h
        qkv = tf.reshape(self.qkv(x), (-1, self.T, 3, self.h, dh))
        q, k, v = [tf.transpose(qkv[:, :, i], (0, 2, 1, 3)) for i in range(3)]
        logits = tf.matmul(q, k, transpose_b=True) / float(np.sqrt(dh))
        logits += tf.expand_dims(tf.convert_to_tensor(self.temp_bias), 0)
        logits += tf.reshape(self.deg_scale, (1, self.h, 1, 1)) * deg[:, None, None, :]
        attn = self.drop(tf.nn.softmax(logits, -1), training=training)
        o = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (-1, self.T, self.d))
        return self.out(o)


class TemporalBlock(layers.Layer):
    def __init__(self, d, heads, T, drop=DROPOUT, **kw):
        super().__init__(**kw)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = DegradationAwareAttention(d, heads, T, drop)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = layers.Dense(d * FF_MULT, activation="gelu")
        self.ff2 = layers.Dense(d)
        self.drop = layers.Dropout(drop)

    def call(self, x, deg, training=None):
        x = x + self.drop(self.attn(self.ln1(x), deg), training=training)
        return x + self.drop(self.ff2(self.ff1(self.ln2(x))), training=training)


class SegmentAggregator(layers.Layer):
    """Frames -> K segment tokens -> inter-segment attention -> two poolings:
    quality-weighted and worst-segment-weighted (bad moments hurt more)."""
    def __init__(self, d, heads, K, F, **kw):
        super().__init__(**kw)
        self.d, self.K, self.F = d, K, F
        self.intra = AttentionPool(d, heads)
        self.inter = TransformerBlock(d, heads)
        self.q_head = layers.Dense(1)
        self.w_head = layers.Dense(1)

    def build(self, input_shape):
        self.seg_pos = self.add_weight(name="seg_pos", shape=(1, self.K, self.d),
                                       initializer=keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x, deg):                              # (B,T,d), (B,T)
        seg = self.intra(tf.reshape(x, (-1, self.F, self.d)))
        seg = self.inter(tf.reshape(seg, (-1, self.K, self.d)) + self.seg_pos)
        seg_deg = tf.reduce_mean(tf.reshape(deg, (-1, self.K, self.F)), 2, keepdims=True)
        q = self.q_head(seg)
        w_main = tf.nn.softmax(self.w_head(tf.concat([seg, seg_deg], -1)), axis=1)
        w_bad = tf.nn.softmax(-q, axis=1)
        return tf.concat([tf.reduce_sum(w_main * seg, 1), tf.reduce_sum(w_bad * seg, 1)], -1)


class DATransQA(keras.Model):
    def __init__(self):
        super().__init__(name="DA_TransQA")
        d, h = D_MODEL, N_HEADS
        self.spatial = StreamEncoder(d, h, N_SPATIAL_LAYERS, PATCH, IMG_SIZE, name="spatial_stream")
        self.motion  = StreamEncoder(d, h, N_SPATIAL_LAYERS, PATCH, IMG_SIZE, cross=True, name="motion_stream")
        self.anim    = StreamEncoder(d, h, N_SPATIAL_LAYERS, PATCH, IMG_SIZE, name="animation_stream")
        self.degr    = TemporalDegradation(d, name="temporal_degradation")
        self.fusion  = CrossStreamFusion(d, h, N_FRAMES, name="cross_stream_fusion")
        self.t_blocks = [TemporalBlock(d, h, N_FRAMES) for _ in range(N_TEMPORAL_LAYERS)]
        self.t_ln    = layers.LayerNormalization(epsilon=1e-6)
        self.segments = SegmentAggregator(d, h, K_SEGMENTS, FRAMES_PER_SEG, name="segment_aggregator")
        self.feat_mlp = keras.Sequential([layers.Dense(d, activation="gelu"), layers.Dropout(DROPOUT),
                                          layers.Dense(d, activation="gelu")], name="global_cues")
        # degradation-aware mixture-of-experts head (v5): guess the damage type, and let one small
        # expert per damage type score the video. Final score = experts weighted by the guess.
        self.shared = keras.Sequential([layers.Dense(d, activation="gelu"), layers.Dropout(DROPOUT)],
                                       name="shared_head")
        self.type_head = layers.Dense(N_TYPES, name="damage_type_head")
        self.expert_heads = layers.Dense(N_TYPES, name="expert_heads")

    def build(self, input_shape):
        self.t_pos = self.add_weight(name="t_pos", shape=(1, N_FRAMES, D_MODEL),
                                     initializer=keras.initializers.RandomNormal(stddev=0.02))

    def call(self, inputs, training=None):               # ((B,T,H,W,3), (B,N_FEATS)) -> (B, 1+N_TYPES)
        frames, feats = inputs
        T, H, W, d = N_FRAMES, IMG_SIZE, IMG_SIZE, D_MODEL
        flat  = tf.reshape(frames, (-1, H, W, 3))
        diffs = tf.reshape(_frame_diffs(frames), (-1, H, W, 3))
        anim  = _animation_maps(flat)
        edge_mean = tf.reshape(tf.reduce_mean(anim[..., :1], axis=[1, 2, 3]), (-1, T))

        s_tok, s_vec = self.spatial(flat)
        _,     m_vec = self.motion(diffs, context=s_tok)
        _,     a_vec = self.anim(anim)
        dg_vec, deg  = self.degr(frames, edge_mean)

        r = lambda v: tf.reshape(v, (-1, T, d))
        x = self.fusion(r(s_vec), r(m_vec), r(a_vec), dg_vec) + self.t_pos
        for blk in self.t_blocks:
            x = blk(x, deg)
        video = self.segments(self.t_ln(x), deg)
        h = self.shared(tf.concat([video, self.feat_mlp(feats, training=training)], -1), training=training)
        logits = self.type_head(h)                       # damage-type guess
        q = tf.reduce_sum(tf.nn.softmax(logits, -1) * self.expert_heads(h), -1, keepdims=True)
        return tf.concat([q, logits], -1)                # (B, 1 + N_TYPES): score + type logits


def build_model_transformer():
    m = DATransQA()
    m((tf.zeros((1, N_FRAMES, IMG_SIZE, IMG_SIZE, 3)), tf.zeros((1, N_FEATS))))   # build weights
    return m

MODEL_FN = build_model_transformer
_m = MODEL_FN()
print(f"DA-TransQA parameters: {_m.count_params():,}")
del _m


# %% 5. Loss: MSE + ranking + same-cartoon ranking + damage type + self-supervised --
# y_true columns: score, cartoon id, damage type, is_real (1 = real video, 0 = synthetic copy)
# y_pred columns: score, then N_TYPES damage-type logits
#   mse, rank        : real videos only (as before)
#   same_rank        : real pairs from the SAME cartoon -> learn severity, not the cartoon
#   type             : guess the damage type (folder name = free label, not a human score)
#   self-supervised  : the extra-damaged copy must score lower than its original.
#                      Batches are laid out as (original, copy, original, copy, ...).
# Masks are used as weights (no dynamic shapes), which keeps it fast on the Metal GPU.
USE_SSL = SSL_WEIGHT > 0

def combined_loss(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 4))
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), (-1, 1 + N_TYPES))
    t, gid, real = y_true[:, 0], y_true[:, 1], y_true[:, 3]
    typ = tf.cast(y_true[:, 2], tf.int32)
    q, logits = y_pred[:, 0], y_pred[:, 1:]
    n_real = tf.reduce_sum(real) + 1e-8
    mse = tf.reduce_sum(real * tf.square(t - q)) / n_real
    margin = tf.sign(t[:, None] - t[None, :]) * (q[:, None] - q[None, :])
    pair_loss = tf.nn.softplus(-margin)                  # = log(1 + exp(-margin)), overflow-safe
    rr = real[:, None] * real[None, :] * (1.0 - tf.eye(tf.shape(t)[0]))   # pairs of real videos
    rank = tf.reduce_sum(pair_loss * rr) / (tf.reduce_sum(rr) + 1e-8)
    same = tf.cast(tf.equal(gid[:, None], gid[None, :]), tf.float32) * rr
    same_rank = tf.reduce_sum(pair_loss * same) / (tf.reduce_sum(same) + 1e-8)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=typ, logits=logits)
    type_loss = tf.reduce_sum(real * ce) / n_real
    loss = mse + RANK_WEIGHT * rank + GROUP_RANK_WEIGHT * same_rank + TYPE_WEIGHT * type_loss
    if USE_SSL:
        q2 = tf.reshape(q, (-1, 2))                      # (original, extra-damaged copy)
        loss += SSL_WEIGHT * tf.reduce_mean(tf.nn.softplus(-(q2[:, 0] - q2[:, 1])))
    return loss


# %% 6. Metrics: PLCC, SROCC, KRCC, RMSE ------------------------------------
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
warnings.filterwarnings("ignore", category=OptimizeWarning)   # logistic fit on 15 points is often loose
warnings.filterwarnings("ignore", category=FutureWarning)    # newer scikit-learn deprecation notices

def _logistic5(x, b1, b2, b3, b4, b5):
    z = np.clip(b2 * (x - b3), -50, 50)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + b4 * x + b5

def fit_logistic(preds, mos):
    p0 = [np.max(mos) - np.min(mos), 1.0, np.mean(preds), 0.0, np.mean(mos)]
    try:
        popt, _ = curve_fit(_logistic5, preds, mos, p0=p0, maxfev=20000)
        return _logistic5(preds, *popt)
    except Exception:
        return preds

def compute_metrics(preds, mos):
    fitted = fit_logistic(preds, mos)
    return {"PLCC": float(pearsonr(fitted, mos)[0]),
            "SROCC": float(spearmanr(preds, mos)[0]),
            "KRCC": float(kendalltau(preds, mos)[0]),
            "RMSE": float(np.sqrt(np.mean((fitted - mos) ** 2)))}


# %% 7. 10 cartoon-split Train/Valid/Test iterations -------------------------
# ---- synthetic extra damage for the self-supervised ranking (runs on the CPU inside tf.data) ----
_GK = tf.range(-4, 5, dtype=tf.float32)
_TT = tf.range(N_FRAMES)

def _blur(clip):                                         # clip: (T, H, W, 3) float 0..1
    s = tf.random.uniform((), 0.8, 2.5)
    g = tf.exp(-tf.square(_GK) / (2.0 * s * s))
    g = g / tf.reduce_sum(g)
    k = tf.tile((g[:, None] * g[None, :])[:, :, None, None], [1, 1, 3, 1])
    x = tf.pad(clip, [[0, 0], [4, 4], [4, 4], [0, 0]], mode="REFLECT")
    return tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], "VALID")

def _jpeg(clip):                                         # blocky compression artefacts
    q = tf.random.uniform((), 5, 40, dtype=tf.int32)
    return tf.map_fn(lambda fr: tf.image.adjust_jpeg_quality(fr, q), clip)

def _downup(clip):                                       # lost fine detail (H.265-like smoothing)
    f = tf.random.uniform((), 1.5, 3.0)
    s = tf.cast(tf.round(IMG_SIZE / f), tf.int32)
    small = tf.image.resize(clip, tf.stack([s, s]), method="area")
    return tf.image.resize(small, [IMG_SIZE, IMG_SIZE], method="bilinear")

def _freeze(clip):                                       # repeat the first frame inside 1-2 segments
    seg = _TT // FRAMES_PER_SEG
    s1 = tf.random.uniform((), 0, K_SEGMENTS, dtype=tf.int32)
    s2 = tf.random.uniform((), 0, K_SEGMENTS, dtype=tf.int32)
    frozen = (tf.equal(seg, s1) | tf.equal(seg, s2)) & tf.not_equal(_TT % FRAMES_PER_SEG, 0)
    return tf.gather(clip, tf.where(frozen, seg * FRAMES_PER_SEG, _TT))

def degrade(clip):
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    out = tf.switch_case(k, [lambda: _blur(clip), lambda: _jpeg(clip),
                             lambda: _downup(clip), lambda: _freeze(clip)])
    out = tf.clip_by_value(tf.cast(out, tf.float32), 0.0, 1.0)
    out.set_shape(clip.shape)
    return out


def make_train_ds(Xu8, Fs, ys, seed=0):
    """Each epoch every video shows ONE random crop (+ random horizontal flip).
    ys is (N, 4): normalised score, cartoon id, damage type, is_real.
    With the self-supervised loss on, every clip is followed by its extra-damaged copy."""
    ds = tf.data.Dataset.from_tensor_slices(((Xu8, Fs), ys)).shuffle(len(ys), seed=seed)
    def pick(xf, t):
        x, f = xf
        c = tf.random.uniform((), 0, N_CROPS, dtype=tf.int32)
        clip = tf.cast(tf.gather(x, c), tf.float32) / 255.0                # (T,H,W,3)
        clip = tf.cond(tf.random.uniform(()) < 0.5, lambda: tf.reverse(clip, [2]), lambda: clip)
        if not USE_SSL:
            return (clip, f), t
        t_copy = tf.concat([t[:3], tf.zeros((1,), tf.float32)], 0)         # same labels, is_real = 0
        return (tf.stack([clip, degrade(clip)]), tf.stack([f, f])), tf.stack([t, t_copy])
    ds = ds.map(pick, num_parallel_calls=tf.data.AUTOTUNE)
    if USE_SSL:
        ds = ds.unbatch()                                # -> original, copy, original, copy, ...
    return ds.batch(BATCH_SIZE * (2 if USE_SSL else 1)).prefetch(tf.data.AUTOTUNE)


def crop_datasets(Xu8, Fs):
    """One evaluation dataset per crop position (no augmentation)."""
    return [tf.data.Dataset.from_tensor_slices((Xu8[:, c], Fs))
              .map(lambda x, f: ((tf.cast(x, tf.float32) / 255.0, f), 0.0))   # dummy label, ignored
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) for c in range(N_CROPS)]


def predict_full(model, dss):
    """Average the model output over all crops -> (N, 1 + N_TYPES): score + damage-type logits."""
    return np.mean([model.predict(ds, verbose=0).reshape(-1, 1 + N_TYPES) for ds in dss], axis=0)


def fit_svr(F_tr, y_tr, F_va, y_va):
    """Handcrafted features only. C is picked on the VALIDATION cartoons, never on test."""
    best_s, best_m = -2.0, None
    for C in SVR_C_GRID:
        m = SVR(C=C, gamma="scale").fit(F_tr, y_tr)
        s = spearmanr(m.predict(F_va), y_va)[0]
        s = -1.0 if np.isnan(s) else s
        if s > best_s:
            best_s, best_m = s, m
    return best_m, best_s


def fit_cnn(B_tr, y_tr, B_va, y_va):
    """Backbone features -> PCA -> SVR. PCA size and C are picked on the VALIDATION cartoons."""
    best = (-2.0, None, None, None)
    for k in PCA_GRID:
        k = int(min(k, B_tr.shape[0] - 1, B_tr.shape[1]))
        pca = PCA(n_components=k, random_state=0).fit(B_tr)
        Z_tr = pca.transform(B_tr)
        sc = Z_tr.std(0) + 1e-6
        Z_tr, Z_va = Z_tr / sc, pca.transform(B_va) / sc
        for C in SVR_C_GRID:
            m = SVR(C=C, gamma="scale").fit(Z_tr, y_tr)
            s = spearmanr(m.predict(Z_va), y_va)[0]
            s = -1.0 if np.isnan(s) else s
            if s > best[0]:
                best = (s, m, pca, sc)
    s, m, pca, sc = best
    return (lambda B: m.predict(pca.transform(B) / sc)), s, f"PCA={pca.n_components_} C={m.C}"


def unsup_scores(train_idx, ref_ok):
    """Opinion-UNAWARE: how far every video is from 'pristine', learned only from the clean
    Reference videos of the non-test cartoons. No human scores are used anywhere in here.
      d_hand: average standardised distance of the 57 handcrafted features to the pristine mean
      d_deep: Mahalanobis distance of every backbone crop to a Gaussian fitted on pristine crops"""
    sd = F_raw[train_idx].std(0) + 1e-6                  # spread of training videos (no labels used)
    d_hand = np.mean(np.clip(np.abs(F_raw - F_ref[ref_ok].mean(0)) / sd, 0, 10), axis=1)
    d_deep = None
    if B_crops is not None and B_ref_crops is not None:
        D = B_crops.shape[-1]
        P = B_ref_crops[ref_ok].reshape(-1, D)           # all pristine crops
        m = P.mean(0)
        s = B_crops[train_idx].reshape(-1, D).std(0) + 1e-6
        norm = lambda A: np.clip((A - m) / s, -10, 10)
        pca = PCA(n_components=int(min(UNSUP_PCA, len(P) - 1, D)), random_state=0).fit(norm(P))
        lw = LedoitWolf().fit(pca.transform(norm(P)))
        md = lw.mahalanobis(pca.transform(norm(B_crops.reshape(-1, D))))
        d_deep = np.sqrt(np.maximum(md, 0)).reshape(len(B_crops), -1).mean(1)
    return d_hand, d_deep


def fit_experts(X_tr, y_tr, k_tr, X_va, y_va):
    """Degradation-aware two-stage model (in the spirit of DIIVINE):
    stage 1 = classifier that guesses the damage type, stage 2 = one SVR expert per damage type.
    Score = experts weighted by the stage-1 probabilities. Both C values picked on VALIDATION."""
    classes = np.unique(k_tr)
    experts = {Ce: [SVR(C=Ce, gamma="scale").fit(X_tr[k_tr == k], y_tr[k_tr == k]) for k in classes]
               for Ce in SVR_C_GRID}
    E_va = {Ce: np.stack([e.predict(X_va) for e in ex], 1) for Ce, ex in experts.items()}
    best = (-2.0, None, None)
    for Cc in EXPERT_CLF_C:
        clf = SVC(C=Cc, gamma="scale", probability=True, random_state=0).fit(X_tr, k_tr)
        P_va = clf.predict_proba(X_va)                   # columns follow clf.classes_ (= classes)
        for Ce in SVR_C_GRID:
            s = spearmanr((P_va * E_va[Ce]).sum(1), y_va)[0]
            s = -1.0 if np.isnan(s) else s
            if s > best[0]:
                best = (s, clf, Ce)
    s, clf, Ce = best
    ex = experts[Ce]
    predict = lambda X: (clf.predict_proba(X) * np.stack([e.predict(X) for e in ex], 1)).sum(1)
    return predict, s, f"clfC={clf.C} C={Ce}", clf


class ValCorrelation(keras.callbacks.Callback):
    """Adds val_loss (MSE) + val PLCC/SROCC/KRCC/RMSE to the logs each epoch (SROCC drives early
    stopping). Same metrics as the test set (all crops averaged, logistic fit, MOS scale)."""
    def __init__(self, Xu8, Fs, y_true, yn_true, y_mu, y_sd):
        super().__init__()
        self.dss = crop_datasets(Xu8, Fs)
        self.y_true, self.yn_true, self.history = y_true, yn_true, []
        self.y_mu, self.y_sd = y_mu, y_sd

    def on_epoch_end(self, epoch, logs=None):
        p = predict_full(self.model, self.dss)[:, 0]
        logs["val_loss"] = float(np.mean((self.yn_true - p) ** 2))
        if np.std(p) > 0:
            vm = compute_metrics(p * self.y_sd + self.y_mu, self.y_true)
        else:                                            # constant predictions -> correlations undefined
            vm = {"PLCC": np.nan, "SROCC": np.nan, "KRCC": np.nan, "RMSE": float(np.std(self.y_true))}
        for k, v in vm.items():
            logs[f"val_{k.lower()}"] = -1.0 if np.isnan(v) else float(v)
        self.history.append((logs.get("loss"), logs.get("val_loss"), logs["val_srocc"], logs["val_plcc"],
                             logs["val_krcc"], logs["val_rmse"]))
        if VERBOSE_EPOCHS:
            print(f"      epoch {epoch+1:3d}  loss={logs['loss']:.3f}  val_loss={logs['val_loss']:.3f}  "
                  f"val_PLCC={logs['val_plcc']:.3f}  val_SROCC={logs['val_srocc']:.3f}  "
                  f"val_KRCC={logs['val_krcc']:.3f}  val_RMSE={logs['val_rmse']:.3f}")


HAS_CNN = B_raw is not None
METHODS = (["unsup"] if HAS_REF else []) + ["svr"] + (["cnn"] if HAS_CNN else []) + \
          ["experts", "deep"] + (["hybrid3"] if HAS_CNN else []) + ["ensemble", "ens_no_deep"]
MAIN = "ensemble"
METRICS = ["PLCC", "SROCC", "KRCC", "RMSE"]
iteration_rows, iteration_cartoons, within_rows = [], [], []
cid = table["content"].values                              # cartoon name of every video
gid_all = pd.factorize(table["content"])[0].astype(np.float32)   # cartoon id (number) for the loss
type_all = table["distortion"].map({t: k for k, t in enumerate(TYPES)}).to_numpy(np.float32)
dist_all = table["distortion"].values
contents = np.array(sorted(set(cid)))                      # the 15 cartoons
N_ROUNDS = N_ITERATIONS // N_FOLDS
oof = {k: np.full((N_ROUNDS, len(X_u8)), np.nan) for k in METHODS}   # test predictions, per round

for i in range(N_ITERATIONS):
    if SMOKE and i >= 1:
        break
    rnd, grp = divmod(i, N_FOLDS)                           # round 0..1, group 0..4
    groups = np.array_split(np.random.default_rng(SEED + rnd).permutation(contents), N_FOLDS)
    test_c = groups[grp]                                    # 3 cartoons for testing
    rest_c = np.concatenate([g for j, g in enumerate(groups) if j != grp])
    val_c = np.random.default_rng(SEED + i).choice(rest_c, N_VAL_CARTOONS, replace=False)
    test_idx  = np.where(np.isin(cid, test_c))[0]          # ALL versions of the test cartoons
    val_idx   = np.where(np.isin(cid, val_c))[0]           # ALL versions of the val cartoons
    train_idx = np.where(~np.isin(cid, np.concatenate([test_c, val_c])))[0]   # the other 9 cartoons
    print(f"\nIteration {i+1}/{N_ITERATIONS}  test cartoons: {', '.join(test_c)}  |  val: {', '.join(val_c)}")

    # standardise targets + features with TRAIN statistics only (no test leakage)
    y_mu, y_sd = y[train_idx].mean(), y[train_idx].std() + 1e-6
    f_mu, f_sd = F_raw[train_idx].mean(0), F_raw[train_idx].std(0) + 1e-6
    Fn = np.clip((F_raw - f_mu) / f_sd, -5, 5).astype(np.float32)
    yn = ((y - y_mu) / y_sd).astype(np.float32)
    yg = np.stack([yn, gid_all, type_all, np.ones_like(yn)], axis=1).astype(np.float32)
    preds, info = {}, []

    # np.errstate: Apple M-chip math library raises FALSE "overflow / divide by zero in matmul"
    # warnings inside sklearn (inputs are clipped, so real overflows can't happen). Results are fine.
    with np.errstate(all="ignore"):
        # (0) unsupervised pristine distance (no human scores)
        Dz = None
        if HAS_REF:
            ref_ok = ~np.isin(ref_cartoon, test_c)           # pristine videos of NON-test cartoons only
            dists = [d for d in unsup_scores(train_idx, ref_ok) if d is not None]
            D = np.stack(dists, 1)
            Dz = (D - D[train_idx].mean(0)) / (D[train_idx].std(0) + 1e-6)
            preds["unsup"] = -Dz.mean(1)[test_idx]           # far from pristine = worse quality

        # (a) handcrafted features only
        svr, svr_val = fit_svr(Fn[train_idx], yn[train_idx], Fn[val_idx], yn[val_idx])
        preds["svr"] = svr.predict(Fn[test_idx])
        info.append(f"svr: C={svr.C} val={svr_val:.3f}")

        # (a2) pretrained backbone features only
        if HAS_CNN:
            b_mu, b_sd = B_raw[train_idx].mean(0), B_raw[train_idx].std(0) + 1e-6
            Bn = np.clip((B_raw - b_mu) / b_sd, -5, 5).astype(np.float32)
            cnn_predict, cnn_val, cnn_cfg = fit_cnn(Bn[train_idx], yn[train_idx], Bn[val_idx], yn[val_idx])
            preds["cnn"] = cnn_predict(Bn[test_idx])
            info.append(f"cnn: {cnn_cfg} val={cnn_val:.3f}")

        # (a3) degradation-aware experts on handcrafted + backbone + pristine-distance features
        Xe = [Fn]
        if HAS_CNN:
            pca_e = PCA(n_components=int(min(EXPERT_PCA, len(train_idx) - 1, Bn.shape[1])),
                        random_state=0).fit(Bn[train_idx])
            Zb = pca_e.transform(Bn)
            Xe.append(Zb / (Zb[train_idx].std(0) + 1e-6))
        if Dz is not None:
            Xe.append(np.clip(Dz, -5, 5))
        Xe = np.concatenate(Xe, 1).astype(np.float32)
        exp_predict, exp_val, exp_cfg, exp_clf = fit_experts(Xe[train_idx], yn[train_idx], type_all[train_idx],
                                                             Xe[val_idx], yn[val_idx])
        preds["experts"] = exp_predict(Xe[test_idx])
        exp_acc = float(np.mean(exp_clf.predict(Xe[test_idx]) == type_all[test_idx]))
        info.append(f"experts: {exp_cfg} val={exp_val:.3f} damage-type acc={exp_acc:.2f}")

    # (b) DA-TransQA  (N_DEEP_SEEDS models, averaged)
    outs, histories = [], []
    for s in range(N_DEEP_SEEDS):
        keras.utils.set_random_seed(SEED + i + 1000 * s)
        model = MODEL_FN()                               # fresh weights every iteration
        model.compile(optimizer=keras.optimizers.AdamW(LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                                                       clipnorm=1.0), loss=combined_loss)
        val_cb = ValCorrelation(X_u8[val_idx], Fn[val_idx], y[val_idx], yn[val_idx], y_mu, y_sd)
        model.fit(
            make_train_ds(X_u8[train_idx], Fn[train_idx], yg[train_idx], seed=SEED + i + 1000 * s),
            epochs=EPOCHS, verbose=0,
            callbacks=[val_cb,                           # must come first: it writes val_srocc
                       keras.callbacks.ReduceLROnPlateau(monitor="val_srocc", mode="max", factor=0.5,
                                                         patience=max(3, PATIENCE // 3), min_lr=1e-6),
                       keras.callbacks.EarlyStopping(monitor="val_srocc", mode="max", patience=PATIENCE,
                                                     restore_best_weights=True)],
        )
        outs.append(predict_full(model, crop_datasets(X_u8[test_idx], Fn[test_idx])))
        histories.append(np.array(val_cb.history, dtype=float))
        keras.backend.clear_session()
    out = np.mean(outs, axis=0)
    preds["deep"] = out[:, 0]
    deep_acc = float(np.mean(np.argmax(out[:, 1:], 1) == type_all[test_idx]))
    h = histories[0]
    best = int(np.argmax(h[:, 2]))
    info.insert(0, f"deep: epochs={len(h)}/{EPOCHS} best={best+1} val={h[best, 2]:.3f} damage-type acc={deep_acc:.2f}")

    # (c) combinations = plain averages (fixed in advance, NOT tuned on test)
    if HAS_CNN:
        preds["hybrid3"] = (preds["svr"] + preds["cnn"] + preds["deep"]) / 3.0
    members = ["svr"] + (["cnn"] if HAS_CNN else []) + ["experts", "deep"]
    preds["ensemble"] = np.mean([preds[k] for k in members], axis=0)
    preds["ens_no_deep"] = np.mean([preds[k] for k in members if k != "deep"], axis=0)   # ablation only

    row = {"experts_type_acc": exp_acc, "deep_type_acc": deep_acc}
    for name in METHODS:
        p = preds[name] * y_sd + y_mu                    # back to MOS scale (unsup: arbitrary scale)
        oof[name][rnd, test_idx] = p
        for k, v in compute_metrics(p, y[test_idx]).items():
            row[f"{name}_{k}"] = v
        for d in DISTORTION_FOLDERS[1:]:                 # standard per-distortion: inside THIS test split
            mk = dist_all[test_idx] == d
            if mk.sum() > 3:
                within_rows.append({"iteration": i + 1, "method": name, "distortion": d,
                                    **compute_metrics(p[mk], y[test_idx][mk])})
    iteration_rows.append(row)
    iteration_cartoons.append(", ".join(test_c))
    print(f"Iteration {i+1:2d}/{N_ITERATIONS}  (train={len(train_idx)}, valid={len(val_idx)}, "
          f"test={len(test_idx)})")
    for name in METHODS:
        print(f"    {name:8s}  PLCC={row[name + '_PLCC']:.4f}  SROCC={row[name + '_SROCC']:.4f}  "
              f"KRCC={row[name + '_KRCC']:.4f}  RMSE={row[name + '_RMSE']:.4f}" + ("   <- main" if name == MAIN else ""))
    print("    " + "  |  ".join(info))


# %% 8. Final results (mean ± std over iterations) --------------------------
results_df = pd.DataFrame(iteration_rows)
n_done = len(results_df)
print(f"\n=== Final metrics (cartoon split), mean ± std over {n_done} iterations"
      + ("  [SMOKE TEST: numbers mean nothing]" if SMOKE else "") + " ===")
for name in METHODS:
    print(f"  {name:8s}  " + "  ".join(
        f"{k}={results_df[f'{name}_{k}'].mean():.4f} (± {results_df[f'{name}_{k}'].std():.4f})" for k in METRICS)
        + ("   <- main" if name == MAIN else "") + ("   (no human scores used)" if name == "unsup" else ""))
print("  RF-AVQA paper (overall): PLCC 0.879  SROCC 0.863  RMSE 0.509")
print(f"  damage-type accuracy on test: experts {results_df['experts_type_acc'].mean():.3f}  "
      f"deep {results_df['deep_type_acc'].mean():.3f}")
results_df.assign(test_cartoons=iteration_cartoons).to_csv(
    f"da_transqa_results_{RUN_NAME}.csv", index_label="iteration")

PAPER = {"Blur": (0.680, 0.614), "FrameFreeze": (0.961, 0.931), "FrameHang": (0.941, 0.907),
         "H264": (0.955, 0.943), "H265": (0.774, 0.688)}           # RF-AVQA (LCC, SROCC)
order = [d for d in DISTORTION_FOLDERS[1:] if d in PAPER and d in set(dist_all)]

# ---- per distortion, standard way: inside each test split, averaged over the iterations ----
within_df = pd.DataFrame(within_rows)
within_df.to_csv(f"da_transqa_per_distortion_within_{RUN_NAME}.csv", index=False)
for metric, pi in [("SROCC", 1), ("PLCC", 0)]:
    t = within_df.pivot_table(index="distortion", columns="method", values=metric, aggfunc="mean")
    t = t.reindex(index=order, columns=METHODS)
    t["paper"] = [PAPER[d][pi] for d in t.index]
    print(f"\n=== Per distortion {metric} — within each test split, mean over {n_done} iterations "
          f"(standard) ===\n{t.round(3).to_string()}")

# ---- per distortion, pooled: all models' test predictions scored together (stricter) ----
pooled_rows = []
for d in order:
    for name in METHODS:
        per_round = []
        for k in range(N_ROUNDS):
            ok = (dist_all == d) & ~np.isnan(oof[name][k])
            if ok.sum() > 3:
                per_round.append(compute_metrics(oof[name][k, ok], y[ok]))
        if per_round:
            pooled_rows.append({"distortion": d, "method": name,
                                **{m: float(np.mean([x[m] for x in per_round])) for m in METRICS}})
pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv(f"da_transqa_per_distortion_pooled_{RUN_NAME}.csv", index=False)
if len(pooled_df):
    t = pooled_df.pivot_table(index="distortion", columns="method", values="SROCC").reindex(index=order, columns=METHODS)
    t["paper"] = [PAPER[d][1] for d in t.index]
    print(f"\n=== Per distortion SROCC — pooled across all test splits (stricter) ===\n{t.round(3).to_string()}")
print(f"\nSaved: da_transqa_results_{RUN_NAME}.csv, da_transqa_per_distortion_within_{RUN_NAME}.csv, "
      f"da_transqa_per_distortion_pooled_{RUN_NAME}.csv")
results_df