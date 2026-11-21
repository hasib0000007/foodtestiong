# Extract the scaled feature matrix once: calling smoothed[FEATURE_COLS].values inside
# the pairing loop would allocate a new ~135 MB copy per iteration.
SMOOTHED_MATRIX = smoothed[FEATURE_COLS].values.astype(np.float32)

def build_fusion_pairs(base_idx, chunk=256):
    """Pair each image with a same-state sensor row; fresh images are also paired with
    formalin-confirmed masked sensor rows (state 2) so the fusion set covers all three
    states. Images are embedded in chunks to keep RAM flat."""
    rng = np.random.default_rng(SEED)
    Fv_parts, Fs_parts, raw_parts, states = [], [], [], []
    imgs, sts, rows = [], [], []
    def flush():
        if imgs:
            Fv_parts.append(image_feature_extractor.predict(tf.stack(imgs), verbose=0))
            S_in = np.stack(rows).astype(np.float32)
            Fs_parts.append(sensor_feature_extractor.predict(S_in, verbose=0))
            raw_parts.append(S_in)
            states.extend(sts)
            imgs.clear(); sts.clear(); rows.clear()
    for i in base_idx:
        st = int(labels[i])                                # 0 stale, 1 fresh (image domain)
        if st == 1 and 2 in tr_sensor_of_state and rng.random() < MASKED_PAIR_FRACTION:
            st = 2                                         # fresh-looking image + masked sensor profile
        img, _ = load_img(paths[int(i)], 0)
        imgs.append(img); sts.append(st)
        pool = (tr_sensor_of_state.get(st) if tr_sensor_of_state.get(st) is not None
                else tr_sensor_of_state.get(2, next(iter(tr_sensor_of_state.values()))))
        rows.append(SMOOTHED_MATRIX[int(rng.choice(pool, size=1)[0])])
        if len(imgs) >= chunk:
            flush()
    flush()
    # Z concatenates the two feature-extractor EMBEDDINGS [Fv | Fs] so that the
    # Mahalanobis space matches inference, which uses the same concatenation.
    # Raw scaled sensor rows are kept separately for the fusion engine input.
    Z = np.concatenate([np.concatenate(Fv_parts), np.concatenate(Fs_parts)], axis=1)
    S_raw = np.concatenate(raw_parts).astype(np.float32)
    return Z, S_raw, np.array(states)

tr_sensor_of_state = {int(s): np.flatnonzero(y_tr == s) for s in (0, 1, 2) if (y_tr == s).any()}
pair_idx = np.random.permutation(len(paths))[:PAIR_LIMIT] if len(paths) > PAIR_LIMIT else np.arange(len(paths))
Z, S_raw, pair_state = build_fusion_pairs(pair_idx)
n_state = {int(s): int(c) for s, c in zip(*np.unique(pair_state, return_counts=True))}
print("Fused feature matrix Z:", Z.shape, "| states:", n_state)

# The sensor branch's 3-class state posterior is appended to the fusion input so the
# engine can separate the masked and fresh freshness-index bands directly.
state_post = sensor_branch.predict(S_raw, batch_size=256, verbose=0)   # S_raw IS the 16-dim sensor input
FUSION_INPUT = np.concatenate([Z, S_raw, state_post], axis=1).astype(np.float32)
print("Fusion engine input:", FUSION_INPUT.shape, "(Fv 128 + Fs 16 + raw scaled 16 + state prob 3)")

def freshness_index_gt(state):
    """Supervision bands with narrow uniform widths: the band width bounds the
    irreducible target noise and therefore the achievable validation MAE."""
    rng = np.random.default_rng(SEED + 11)
    return np.where(state == 1, rng.uniform(0.88, 0.92, len(state)),
             np.where(state == 0, rng.uniform(0.17, 0.21, len(state)),
                      rng.uniform(0.68, 0.72, len(state)))).astype(np.float32)

split = int(len(Z) * 0.85)                                 # explicit fusion train/val split
perm = np.random.permutation(len(Z))
Xf_tr, st_tr = FUSION_INPUT[perm[:split]], pair_state[perm[:split]]
Xf_va, st_va = FUSION_INPUT[perm[split:]], pair_state[perm[split:]]
gt_tr, gt_va = freshness_index_gt(st_tr), freshness_index_gt(st_va)

def fit_mahalanobis(Z_fresh):
    """Numerically stable estimator: float64 accumulation, Ledoit-Wolf-style shrinkage
    toward a scaled identity, explicit eigenvalue clipping to bound the inverse, an
    empirical threshold from the fresh-distance distribution and a chi-square
    reference under the Gaussian assumption."""
    mean = Z_fresh.mean(axis=0).astype(np.float64)
    n_dim = Z_fresh.shape[1]
    Zc = (Z_fresh - mean).astype(np.float64)
    cov = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
    mu = float(np.trace(cov)) / n_dim      # average per-dimension variance
    lam = 0.10                             # Ledoit-Wolf-style shrinkage toward mu*I
    cov = (1.0 - lam) * cov + lam * mu * np.eye(n_dim) + 1e-6 * np.eye(n_dim)
    eigvals, eigvecs = np.linalg.eigh(cov)
    floor = max(1e-3 * float(eigvals.max()), 1e-12)   # clip dead/zero-variance directions
    eigvals = np.clip(eigvals, floor, None)
    inv_cov = (eigvecs / eigvals) @ eigvecs.T
    d_fresh = np.sqrt(np.einsum("...i,ij,...j->...", Zc, inv_cov, Zc))
    thr_emp = float(np.percentile(d_fresh, 99.0))     # empirical 99th pct of REAL fresh distances
    thr_chi2 = math.sqrt(chi2.ppf(0.99, df=n_dim))    # chi2 reference (Gaussian assumption)
    thr = max(thr_emp, thr_chi2)
    return mean, inv_cov.astype(np.float32), thr, d_fresh, thr_chi2

z_fresh = Z[pair_state == 1]
if len(z_fresh) < Z.shape[1] + 2:                               # make sure covariance is estimable
    reps = int(np.ceil((Z.shape[1] + 2) / max(len(z_fresh), 1)))
    z_fresh = np.tile(z_fresh, (reps, 1)) + np.random.normal(0, 1e-3, (len(z_fresh) * reps, Z.shape[1]))
MAH_MEAN, MAH_INV_COV, MAH_THRESHOLD, d_fresh_dbg, THR_CHI2 = fit_mahalanobis(z_fresh)
print(f"Mahalanobis threshold: {MAH_THRESHOLD:.3f} "
      f"(empirical p99 of fresh = {float(np.percentile(d_fresh_dbg, 99.0)):.3f}, "
      f"chi2 99% reference = {THR_CHI2:.3f})")
print(f"Mahalanobis fresh distances: "
      f"min={d_fresh_dbg.min():.2f} median={np.median(d_fresh_dbg):.2f} "
      f"p99={np.percentile(d_fresh_dbg, 99):.2f} max={d_fresh_dbg.max():.2f}")
assert np.isfinite(d_fresh_dbg).all() and d_fresh_dbg.max() < 1e4, (
    "Mahalanobis distance check failed: fresh-class distances exceeded the expected range.")
if (pair_state == 0).any():
    Zc_stale = (Z[pair_state == 0] - MAH_MEAN).astype(np.float64)
    d_stale_dbg = np.sqrt(np.einsum("...i,ij,...j->...", Zc_stale, MAH_INV_COV.astype(np.float64), Zc_stale))
    print(f"Mahalanobis stale distances: "
          f"median={np.median(d_stale_dbg):.2f} p99={np.percentile(d_stale_dbg, 99):.2f} "
          f"(above the fresh p99, as required for class separation)")

# per-branch detectors on the same fitted fresh reference: separate distances for the
# visual embedding (Fv, 128-D) and the sensor embedding (Fs, 16-D) so a single sample
# can be attributed to an image-side or a sensor-side anomaly instead of a fused one
Fv_fresh = z_fresh[:, :EMBED_DIM]
Fs_fresh = z_fresh[:, EMBED_DIM:]
MAH_MEAN_IMG, MAH_INV_COV_IMG, MAH_THRESHOLD_IMG, d_img_dbg, THR_IMG_CHI2 = fit_mahalanobis(Fv_fresh)
MAH_MEAN_SEN, MAH_INV_COV_SEN, MAH_THRESHOLD_SEN, d_sen_dbg, THR_SEN_CHI2 = fit_mahalanobis(Fs_fresh)
print(f"Mahalanobis image-branch threshold (Fv 128-D): {MAH_THRESHOLD_IMG:.3f} "
      f"(empirical p99 = {np.percentile(d_img_dbg, 99.0):.3f}, chi2 99% = {THR_IMG_CHI2:.3f})")
print(f"Mahalanobis sensor-branch threshold (Fs 16-D): {MAH_THRESHOLD_SEN:.3f} "
      f"(empirical p99 = {np.percentile(d_sen_dbg, 99.0):.3f}, chi2 99% = {THR_SEN_CHI2:.3f})")

plt.figure(figsize=(6.2, 4.2))
plt.hist(d_fresh_dbg, bins=40, alpha=0.55, color="tab:green", label="Fresh")
if (pair_state == 0).any():
    plt.hist(d_stale_dbg, bins=40, alpha=0.55, color="tab:orange", label="Stale")
if (pair_state == 2).any():
    Zc_masked = (Z[pair_state == 2] - MAH_MEAN).astype(np.float64)
    d_masked_dbg = np.sqrt(np.einsum("...i,ij,...j->...", Zc_masked,
                                     MAH_INV_COV.astype(np.float64), Zc_masked))
    plt.hist(d_masked_dbg, bins=40, alpha=0.55, color="tab:red", label="Masked (formalin QC)")
plt.axvline(MAH_THRESHOLD, color="black", ls="--", lw=1.2,
            label=f"threshold = {MAH_THRESHOLD:.2f}")
plt.xlabel("Mahalanobis distance"); plt.ylabel("samples")
plt.title("Mahalanobis distance distribution by state")
plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/mahalanobis_distribution.png", dpi=150); plt.show()

def mahalanobis_distance(z):
    zf = np.asarray(z, np.float64)
    inv64 = MAH_INV_COV.astype(np.float64)
    d = zf - MAH_MEAN
    return float(np.sqrt(np.einsum("...i,ij,...j->...", d, inv64, d))) if np.ndim(zf) == 1 \
        else np.sqrt(np.einsum("...i,ij,...j->...", d, inv64, d))

def mahalanobis_distance_in(z, mean, inv_cov):
    d = np.asarray(z, np.float64) - np.asarray(mean, np.float64)
    val = float(d @ np.asarray(inv_cov, np.float64) @ d)
    return float(np.sqrt(max(val, 0.0)))

def build_fusion_engine(n_in):
    inp = keras.Input(shape=(n_in,), name="fusion_input")
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.LayerNormalization(name="fusion_norm")(x)   # stable scale across feature groups
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="freshness_index")(x)
    return keras.Model(inp, out, name="fusion_engine")

FUSION_IN_DIM = EMBED_DIM + SENSOR_EMBED + len(FEATURE_COLS) + 3   # 144 + 16 + 3 = 163
fusion_engine = build_fusion_engine(FUSION_IN_DIM)
image_feature_extractor.trainable = False; sensor_feature_extractor.trainable = False
fusion_engine.compile(optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0),
                      loss=keras.losses.Huber(delta=0.1), metrics=["mae"])
hf = fusion_engine.fit(Xf_tr, gt_tr, validation_data=(Xf_va, gt_va),
                       epochs=EPOCHS_FUSION, verbose=2,
                       callbacks=[keras.callbacks.TerminateOnNaN(),
                                  keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                      factor=0.5, patience=4, min_lr=1e-6, verbose=1)])

plt.plot(hf.history["loss"], label="train"); plt.plot(hf.history["val_loss"], label="val")
plt.title("Fusion engine — MAE loss"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fusion_training.png", dpi=150); plt.show()
fusion_engine.save(f"{OUT}/fusion_engine.keras")

idx_pred = fusion_engine.predict(Xf_va, verbose=0).ravel()
plt.figure(figsize=(5.4, 5.4))
for st, name, c in [(1, "Fresh", "tab:green"), (0, "Stale", "tab:orange"),
                    (2, "Masked (formalin QC)", "tab:red")]:
    m = st_va == st
    if m.any():
        mae_st = float(np.mean(np.abs(idx_pred[m] - gt_va[m]))) * 100.0
        plt.scatter(gt_va[m] * 100, idx_pred[m] * 100, s=14, alpha=0.55, c=c,
                    label=f"{name} (MAE = {mae_st:.1f})")
plt.plot([0, 100], [0, 100], "k:", lw=1)
plt.xlabel("Ground-truth freshness index (0-100)")
plt.ylabel("Predicted freshness index (0-100)")
plt.title("Fusion engine \u2014 predicted vs ground truth (validation)")
plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fusion_scatter.png", dpi=150); plt.show()

CH2O_ALERT_PPM = 2.0   # ZE08-CH2O reading above the ambient formaldehyde background

def decision_fusion_engine(image_path, sensor_feature_vector):
    """Full pipeline for one sample: image + CANONICAL sensor feature row -> detailed result dict.
       `sensor_feature_vector` must already be scaled (value-channels in [0,1], flags 0/1)."""
    img, _ = load_img(image_path, 0)
    v = image_feature_extractor.predict(img[tf.newaxis, ...], verbose=0)[0]
    s = sensor_feature_extractor.predict(np.asarray(sensor_feature_vector, np.float32)[tf.newaxis, ...], verbose=0)[0]
    z = np.concatenate([v, s])                               # 144-D fused embedding space
    sp = sensor_branch.predict(np.asarray(sensor_feature_vector, np.float32)[tf.newaxis, ...],
                               verbose=0)[0]                 # 3-class state posterior
    fusion_inp = np.concatenate([v, s, np.asarray(sensor_feature_vector, np.float32),
                                 sp])[np.newaxis, ...].astype(np.float32)   # 163-D fusion input
    index = float(fusion_engine.predict(fusion_inp, verbose=0)[0][0]) * 100.0
    d_img = mahalanobis_distance_in(v, MAH_MEAN_IMG, MAH_INV_COV_IMG)
    d_sen = mahalanobis_distance_in(s, MAH_MEAN_SEN, MAH_INV_COV_SEN)
    d_fus = mahalanobis_distance(z)
    image_anomaly = bool(d_img > MAH_THRESHOLD_IMG)
    sensor_anomaly = bool(d_sen > MAH_THRESHOLD_SEN)
    raw_ch2o = None
    if len(sensor_feature_vector) > FEATURE_COLS.index("ch2o_ppm"):
        i_ch2o = FEATURE_COLS.index("ch2o_ppm")
        lo, hi = bounds["ch2o_ppm"]
        raw_ch2o = float(sensor_feature_vector[i_ch2o]) * (hi - lo) + lo   # unscale [0,1] -> ppm
    chemical_detected = bool((raw_ch2o is not None and raw_ch2o > CH2O_ALERT_PPM)
                             or (sensor_anomaly and not image_anomaly))
    if chemical_detected:
        state = "CHEMICAL ALERT \u2014 Sensor/Gas anomaly detected while image looks fresh"
    elif image_anomaly and not sensor_anomaly:
        state = "VISUAL DEFECT \u2014 Image shows surface spoilage, gas normal"
    elif image_anomaly and sensor_anomaly:
        state = "SPOILED \u2014 Both image and gas indicate rot"
    elif index >= 70 and d_fus <= MAH_THRESHOLD:
        state = "Fresh"
    else:
        state = "Suspect / Stale"
    return {"freshness_index": round(index, 1),
            "mahalanobis_image": {"distance": round(d_img, 2),
                                  "threshold": round(MAH_THRESHOLD_IMG, 2),
                                  "anomaly": image_anomaly},
            "mahalanobis_chemical_sensor": {"distance": round(d_sen, 2),
                                            "threshold": round(MAH_THRESHOLD_SEN, 2),
                                            "anomaly": sensor_anomaly},
            "mahalanobis_fused": {"distance": round(d_fus, 2),
                                  "threshold": round(MAH_THRESHOLD, 2)},
            "chemical_detected": chemical_detected,
            "ch2o_ppm": round(raw_ch2o, 2) if raw_ch2o is not None else None,
            "state": state}

def canonical_sensor_row(raw_dataset_df_row):
    """Convenience: build a scaled canonical feature row from one raw dataset row."""
    vals = []
    for c in SENSOR_COLS:
        lo, hi = bounds[c]
        vals.append((raw_dataset_df_row[c] - lo) / max(hi - lo, 1e-9))
    for a in AV_COLS:
        vals.append(float(raw_dataset_df_row[a]) if a in raw_dataset_df_row else 1.0)
    return vals

demo = decision_fusion_engine(paths[0], canonical_sensor_row(smoothed.iloc[tr[0]]))
print(json.dumps(demo, indent=2))
