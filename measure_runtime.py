# measure_runtime_v5.py — inference time per video (ms/video) for every DA-AVQA component.
#
# Does NOT re-run the 10 iterations and does NOT train the transformer.
#   1. Loads the confirm script's code + your caches (same dataset path, same settings).
#   2. Fits the fast predictors (SVR, PCA+SVR, experts, pristine model) on iteration 1's split
#      (about a minute).
#   3. Times every step on 10 unseen test videos (2 per distortion), one video at a time,
#      after a warm-up video that is not counted.
# The transformer is timed with freshly initialised weights: same layers and same sizes, so the
# same compute as the trained model (only the weight values differ, not the speed).
#
# Run from the folder that has da_transqa_v5_confirm.py (same terminal / dataset path as before):
#   python measure_runtime_v5.py
# Output: a table in the terminal + runtime_v5_confirm_seed1000.csv

import os, sys, time
import numpy as np
import pandas as pd

CONFIRM = sys.argv[1] if len(sys.argv) > 1 else "da_transqa_confirm_v5.py"
N_PER_TYPE = 2                                     # timed videos per distortion type (10 in total)

# ---- 1. load everything the confirm script defines, stopping right before its training loop ----
src = open(CONFIRM, encoding="utf-8").read()
cut = src.index("\nfor i in range(N_ITERATIONS):")
g = {"__name__": "runtime_probe", "__file__": os.path.abspath(CONFIRM)}
print("Loading code, data table and caches (no training)...")
exec(compile(src[:cut], CONFIRM, "exec"), g)

tf, PCA, LedoitWolf = g["tf"], g["PCA"], g["LedoitWolf"]
table, y, F_raw, B_raw, B_crops = g["table"], g["y"], g["F_raw"], g["B_raw"], g["B_crops"]
if B_raw is None:
    sys.exit("Backbone features are missing, so the full pipeline cannot be timed.")
HAS_REF = g["HAS_REF"]

# ---- 2. iteration 1's split (identical to the confirmation run) ----
SEED, N_FOLDS, NV = g["SEED"], g["N_FOLDS"], g["N_VAL_CARTOONS"]
contents, cid, type_all = g["contents"], g["cid"], g["type_all"]
groups = np.array_split(np.random.default_rng(SEED + 0).permutation(contents), N_FOLDS)
test_c = groups[0]
rest_c = np.concatenate(groups[1:])
val_c = np.random.default_rng(SEED + 0).choice(rest_c, NV, replace=False)
test_idx = np.where(np.isin(cid, test_c))[0]
val_idx = np.where(np.isin(cid, val_c))[0]
train_idx = np.where(~np.isin(cid, np.concatenate([test_c, val_c])))[0]
print(f"Split: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)} videos")

y_mu, y_sd = y[train_idx].mean(), y[train_idx].std() + 1e-6
f_mu, f_sd = F_raw[train_idx].mean(0), F_raw[train_idx].std(0) + 1e-6
Fn = np.clip((F_raw - f_mu) / f_sd, -5, 5).astype(np.float32)
yn = ((y - y_mu) / y_sd).astype(np.float32)
b_mu, b_sd = B_raw[train_idx].mean(0), B_raw[train_idx].std(0) + 1e-6
Bn = np.clip((B_raw - b_mu) / b_sd, -5, 5).astype(np.float32)

# pristine model (same maths as unsup_scores, but keeping the fitted objects for a new video)
if HAS_REF:
    F_ref, B_ref_crops, ref_cartoon = g["F_ref"], g["B_ref_crops"], g["ref_cartoon"]
    ref_ok = ~np.isin(ref_cartoon, test_c)
    sd_h = F_raw[train_idx].std(0) + 1e-6
    mu_ref = F_ref[ref_ok].mean(0)
    Dd = B_crops.shape[-1]
    P = B_ref_crops[ref_ok].reshape(-1, Dd)
    m_ref, s_tr = P.mean(0), B_crops[train_idx].reshape(-1, Dd).std(0) + 1e-6
    norm = lambda A: np.clip((A - m_ref) / s_tr, -10, 10)
    pca_u = PCA(n_components=int(min(g["UNSUP_PCA"], len(P) - 1, Dd)), random_state=0).fit(norm(P))
    lw = LedoitWolf().fit(pca_u.transform(norm(P)))

    def pristine(feats, crops_vecs):
        d_hand = np.mean(np.clip(np.abs(feats - mu_ref) / sd_h, 0, 10))
        d_deep = np.sqrt(np.maximum(lw.mahalanobis(pca_u.transform(norm(crops_vecs))), 0)).mean()
        return np.array([d_hand, d_deep])

    D_all = np.stack([d for d in g["unsup_scores"](train_idx, ref_ok) if d is not None], 1)
    D_mu, D_sd = D_all[train_idx].mean(0), D_all[train_idx].std(0) + 1e-6
    Dz = (D_all - D_mu) / D_sd

print("Fitting the fast predictors on this split...")
with np.errstate(all="ignore"):
    svr, _ = g["fit_svr"](Fn[train_idx], yn[train_idx], Fn[val_idx], yn[val_idx])
    cnn_predict, _, _ = g["fit_cnn"](Bn[train_idx], yn[train_idx], Bn[val_idx], yn[val_idx])
    pca_e = PCA(n_components=int(min(g["EXPERT_PCA"], len(train_idx) - 1, Bn.shape[1])),
                random_state=0).fit(Bn[train_idx])
    Zb = pca_e.transform(Bn)
    z_sd = Zb[train_idx].std(0) + 1e-6
    Xe = [Fn, Zb / z_sd] + ([np.clip(Dz, -5, 5)] if HAS_REF else [])
    Xe = np.concatenate(Xe, 1).astype(np.float32)
    exp_predict, _, _, _ = g["fit_experts"](Xe[train_idx], yn[train_idx], type_all[train_idx],
                                            Xe[val_idx], yn[val_idx])

# networks: frozen backbone + transformer, wrapped in tf.function (as Keras does for predict)
bb_net = g["_BB_NET"] if g["_BB_NET"] is not None else g["build_backbone"]()
model = g["MODEL_FN"]()
bb_fn = tf.function(lambda x: bb_net(x, training=False))
tr_fn = tf.function(lambda x, f: model((x, f), training=False))
pool_maps, extract_video, extract_bb_crops = g["pool_maps"], g["extract_video"], g["extract_bb_crops"]


# ---- 3. one video, end to end, with a timer around every step ----
def run_one(path):
    T = {}
    t0 = time.perf_counter()
    frames, feats = extract_video(path)                                  # decode + 57 features + frame crops
    T["decode_and_57_features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    crops = extract_bb_crops(path)                                       # decode the 4 backbone frames
    T["backbone_decode_crops"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    maps = bb_fn(tf.constant(crops.astype(np.float32)))
    maps = list(maps) if isinstance(maps, (list, tuple)) else [maps]
    vecs = pool_maps([m.numpy() for m in maps])                          # (20, 3456)
    T["backbone_network"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dist = pristine(feats, vecs) if HAS_REF else None
    T["pristine_distances"] = time.perf_counter() - t0

    fn = np.clip((feats - f_mu) / f_sd, -5, 5).astype(np.float32)[None]
    bn = np.clip((vecs.mean(0) - b_mu) / b_sd, -5, 5).astype(np.float32)[None]
    t0 = time.perf_counter(); s_svr = svr.predict(fn)[0]; T["svr_predict"] = time.perf_counter() - t0
    t0 = time.perf_counter(); s_cnn = cnn_predict(bn)[0]; T["cnn_predict"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    xe = [fn, pca_e.transform(bn) / z_sd] + ([np.clip((dist - D_mu) / D_sd, -5, 5)[None]] if HAS_REF else [])
    s_exp = exp_predict(np.concatenate(xe, 1).astype(np.float32))[0]
    T["experts_predict"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = tf.constant(frames.astype(np.float32) / 255.0)                  # 5 crops x 16 frames
    s_deep = float(tr_fn(x, tf.constant(np.repeat(fn, len(frames), 0))).numpy()[:, 0].mean())
    T["transformer"] = time.perf_counter() - t0

    _ = np.mean([s_svr, s_cnn, s_exp, s_deep]) * y_sd + y_mu            # final score (negligible)
    return T


test_tbl = table.iloc[test_idx]
picked = test_tbl.groupby("distortion", group_keys=False).head(N_PER_TYPE)["path"].tolist()
print(f"Warm-up on 1 video (not counted), then timing {len(picked)} unseen test videos...")
run_one(test_tbl["path"].iloc[-1])
rows = []
for k, p in enumerate(picked, 1):
    T = run_one(p)
    rows.append({"video": os.path.basename(p), **{c: v * 1000 for c, v in T.items()}})
    print(f"  {k:2d}/{len(picked)}  {os.path.basename(p):28s}  total {sum(T.values()) * 1000:8.1f} ms")
R = pd.DataFrame(rows)

# ---- 4. per component and per method (ms per video) ----
comp = [c for c in R.columns if c != "video"]
bb = ["backbone_decode_crops", "backbone_network"]
methods = {
    "SVR (57 features)":   ["decode_and_57_features", "svr_predict"],
    "PCA + SVR (backbone)": bb + ["cnn_predict"],
    "Experts":             ["decode_and_57_features"] + bb + ["pristine_distances", "experts_predict"],
    "DA-TransQA":          ["decode_and_57_features", "transformer"],
    "DA-AVQA (ensemble)":  comp,
}
for m, parts in methods.items():
    R[m] = R[parts].sum(1)
R.to_csv("runtime_v5_confirm_seed1000.csv", index=False)

print("\n=== Per component, ms per video (mean ± SD over", len(R), "videos) ===")
for c in comp:
    print(f"  {c:26s} {R[c].mean():9.1f} ± {R[c].std():7.1f}")
print("\n=== Per method, ms per video (use these in the results table) ===")
for m in methods:
    print(f"  {m:26s} {R[m].mean():9.1f} ± {R[m].std():7.1f}")
print(f"\nTF {tf.__version__} | devices: {[d.device_type for d in tf.config.list_physical_devices()]}")
print("One video at a time, after warm-up; one-time loading and training are excluded.")
print("Saved runtime_v5_confirm_seed1000.csv")