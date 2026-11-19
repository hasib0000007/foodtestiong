# %% [markdown]
# # Tri-Layer Multimodal Food Freshness & Safety Detection — Kaggle Training
#
# **Thesis:** *Image and Sensor Data Integration with Mahalanobis Distance for
# AI-Enabled Food Freshness and Safety Detection*
#
# This notebook implements the proposed model from the proposal and the
# presentation flowchart:
#
# | Flowchart stage (Slide 8) | Notebook part |
# |---|---|
# | Image Stream → Camera (OV2640) → **MobileNetV2** | Part A — Image branch (trained **separately**) |
# | Sensor Stream → (ZE08-CH2O, BME680, MLX90614) → Normalization + Kalman-style smoothing → **MLP Regression Module** | Part B — Sensor branch (trained **separately**) |
# | **Mahalanobis Distance** → **Decision Fusion Engine** → Results | Part C — Feature-level late fusion + anomaly layer |
# | Results → WiFi/BLE → Mobile App | Artifacts exported in Part C feed the ESP32-S3 firmware |
#
# **How to run on Kaggle:**
# 1. `Add Input` → attach any fresh/stale image dataset (e.g. *Fresh and Stale
#    Images of Fruits and Vegetables* by kritikseth) and set `IMAGE_DATA_DIR`.
# 2. Sensor data: attach your own aggregated CSV (`SENSOR_CSV`) — OR leave it as
#    `None` to use the built-in physics-based synthetic generator that follows
#    the proposal (96 h decay, formalin masking, 26–36 °C, high humidity).
# 3. Settings → Accelerator → GPU T4, then `Run All`.
# 4. All models, plots and `.tflite` files appear in `/kaggle/working/outputs/`.

# %% [markdown]
# ## 0. Setup & Configuration

# %%
import os, glob, json, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy.stats import chi2
from scipy import ndimage

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE    = (224, 224)   # proposal: uniform 224x224 input
EPOCHS_HEAD = 5            # Part A, phase 1 (head training, backbone frozen)
EPOCHS_FINETUNE = 2        # Part A, phase 2 (fine-tune last backbone block, 0 = skip)
EPOCHS_SENSOR = 30         # Part B
EPOCHS_FUSION = 25         # Part C
PAIR_LIMIT = 400           # max paired samples for the fusion set
ALPHA       = 0.35         # MobileNetV2 width multiplier (deepest HW-valid value for the 0.25 idea)
EMBED_DIM   = 128          # Fv visual feature vector length
SENSOR_COLS = ["ch2o_ppm", "voc_log_ohm", "core_temp_c", "ambient_temp_c", "humidity_pct",
               "ch2o_voc_ratio", "temp_rise_rate", "hours_elapsed"]
SMOOTH_WIN  = 5           # moving-average window (Kalman-style pre-filter on sensor streams)

IMAGE_DATA_DIR = "/kaggle/input/fresh-and-stale-images-of-fruits-and-vegetables"  # <- set after attaching dataset
SENSOR_CSV     = None   # <- path to your own sensor .csv on Kaggle, or None for synthetic

RUNNING_ON_KAGGLE = os.path.isdir("/kaggle/input")
OUT = "/kaggle/working/outputs" if RUNNING_ON_KAGGLE else os.path.abspath("outputs")
os.makedirs(OUT, exist_ok=True)
print("TF:", tf.__version__, "| output dir:", OUT)
print("GPU:", tf.config.list_physical_devices("GPU"))

# %% [markdown]
# ## Part A — Image Branch (trained independently, branch 1: Fv via MobileNetV2)
#
# Camera (OV2640) → rescale to 224×224 → quantized depthwise-separable **MobileNetV2**
# feature extractor → binary Fresh/Stale classification head. The trained backbone is
# then re-wired into a *feature extractor* (`Fv`, 128-D) that Part C freezes and fuses.

# %% [markdown]
# ### A1. Discover image classes and build datasets
# Class folder names containing `fresh` map to label 1 (Fresh) and `stale/rotten/discarded`
# map to label 0 (Stale). Works for both binary folders (`Fresh/`, `Stale/`) and
# per-food folders (`FreshApple/`, `StaleApple/`, ...).

# %%
def discover_class_dirs(root):
    class_dirs = {}
    for d in sorted(glob.glob(os.path.join(root, "**/"), recursive=True)):
        d = d.rstrip(os.sep)
        files = [f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            class_dirs[os.path.basename(d)] = d
        if len(class_dirs) >= 60:   # enough classes
            break
    return class_dirs

def freshness_label(class_name):
    n = class_name.lower()
    if "stale" in n or "rotten" in n or "discard" in n or "spoil" in n:
        return "stale"
    if "fresh" in n:
        return "fresh"
    return None   # unknown folder — ignored

def _make_demo_images(root, per_class=12):
    """Tiny synthetic image set used ONLY for local smoke testing (never runs on Kaggle)."""
    os.makedirs(root, exist_ok=True)
    for name, rgb in (("FreshApple", (150, 200, 60)), ("StaleApple", (90, 90, 30)),
                      ("FreshFish", (200, 190, 170)), ("StaleFish", (110, 105, 95))):
        d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
        for k in range(per_class):
            arr = np.clip(np.array(rgb) + np.random.normal(0, 25, (96, 96, 3)), 0, 255).astype("uint8")
            keras.utils.save_img(os.path.join(d, f"img_{k}.jpg"), arr)

if not os.path.isdir(IMAGE_DATA_DIR):
    # fallback for local smoke testing only; on Kaggle this branch never runs
    IMAGE_DATA_DIR = os.path.abspath("_demo_images")
    _make_demo_images(IMAGE_DATA_DIR)

class_dirs = discover_class_dirs(IMAGE_DATA_DIR)
assert class_dirs, f"No image class folders found under {IMAGE_DATA_DIR}"
paths, labels = [], []
for name, folder in class_dirs.items():
    lab = freshness_label(name)
    if lab is None:
        continue
    for f in glob.glob(os.path.join(folder, "*")):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(f); labels.append(1 if lab == "fresh" else 0)

paths = np.array(paths); labels = np.array(labels)
print(f"Found {len(paths)} images across {len(class_dirs)} class folders "
      f"(fresh={labels.sum()}, stale={(labels==0).sum()})")

BATCH = 32
val_frac = 0.15
idx = np.random.permutation(len(paths))
n_val = max(1, int(len(paths) * val_frac))
val_idx, tr_idx = idx[:n_val], idx[n_val:]

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0          # proposal: [0,255] -> [0,1]
    return img, label

def make_ds(indices, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths[indices], labels[indices]))
    if shuffle: ds = ds.shuffle(1024, seed=SEED)
    return ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds, val_ds = make_ds(tr_idx, True), make_ds(val_idx, False)

# %% [markdown]
# ### A2. Build + train the image branch (Phase 1 head training, then optional fine-tune)

# %%
def build_image_branch():
    inp = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = layers.Rescaling(2.0, offset=-1.0, name="to_imagenet_range")(inp)  # [0,1] -> [-1,1] (ImageNet conv)
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), alpha=ALPHA, include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(EMBED_DIM, activation="relu", name="fv_embedding")(x)   # Fv feature vector
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="freshness")(x)
    branch = keras.Model(inp, out, name="image_branch")
    feat_model = keras.Model(inp, branch.get_layer("fv_embedding").output, name="image_feature_extractor")
    return branch, feat_model

image_branch, image_feature_extractor = build_image_branch()
image_branch.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss="binary_crossentropy", metrics=["accuracy"])
h1 = image_branch.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2)

# Phase 2 (optional in paper): unfreeze last block of the backbone and fine-tune gently
if EPOCHS_FINETUNE > 0:
    backbone = next(l for l in image_branch.layers if isinstance(l, keras.Model))  # MobileNetV2 backbone
    backbone.trainable = True
    for layer in backbone.layers[:-8]:
        layer.trainable = False
    image_branch.compile(optimizer=keras.optimizers.Adam(1e-5),
                         loss="binary_crossentropy", metrics=["accuracy"])
    h2 = image_branch.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, verbose=2)
    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
else:
    hist = h1.history

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(hist["loss"], label="train"); plt.plot(hist["val_loss"], label="val")
plt.title("Image branch — loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(hist["accuracy"], label="train"); plt.plot(hist["val_accuracy"], label="val")
plt.title("Image branch — accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/image_branch_training.png", dpi=150); plt.show()

image_branch.save(f"{OUT}/image_branch.keras")
image_feature_extractor.save(f"{OUT}/image_feature_extractor.keras")
print("Image branch saved (separate training done).")

# %% [markdown]
# ## Part B — Sensor Branch (trained independently, branch 2: telemetry via MLP)
#
# (ZE08-CH2O + BME680 + MLX90614) → **normalization** (min-max) + moving-average
# smoothing (Kalman-style pre-filter) → **MLP Regression Module** → Fresh / Stale /
# Chemically-Masked classification. Trained fully separately from the image branch.

# %% [markdown]
# ### B1. Sensor dataset: attach your CSV or generate the proposal's synthetic dataset
# The synthetic generator reproduces the 96-hour controlled-decay protocol from the
# proposal: natural samples decay at 26–36 °C, high RH; treated samples receive
# formaldehyde dilutions that mask visual decay and distort gas profiles.

# %%
def generate_synthetic_sensor_data(n_per_class=400):
    rows, labels = [], []
    for cls in ("fresh", "stale", "masked"):            # 0=fresh, 1=stale, 2=chemically masked
        for i in range(n_per_class):
            t = np.random.uniform(0, 96)                # hours into the 96 h protocol
            amb = np.random.uniform(26, 36)             # sub-tropical market temperature
            hum = np.random.uniform(60, 88)             # typical wet-market RH %
            if cls == "fresh":
                ch2o = 0.05 + 0.012 * t + np.random.normal(0, 0.02)
                voc   = 6.6 - 0.012 * t + np.random.normal(0, 0.08)
                core  = amb - 2.5 + 0.01 * t + np.random.normal(0, 0.15)
            elif cls == "stale":
                ch2o = 0.15 + 0.11 * max(t - 12, 0) + np.random.normal(0, 0.08)
                voc   = 6.5 - 0.03 * t + np.random.normal(0, 0.12)
                core  = amb - 2.0 + 0.05 * t + np.random.normal(0, 0.25)
            else:  # masked: formalin keeps VOC low but CH2O explodes
                ch2o = 4.0 + 0.35 * np.random.uniform(0, 12) * (1 + t / 96) + np.random.normal(0, 0.3)
                voc   = 6.4 - 0.004 * t + np.random.normal(0, 0.10)   # decay artificially arrested
                core  = amb - 2.2 + 0.015 * t + np.random.normal(0, 0.2)
            cc = amb - 2.0
            rows.append([max(ch2o, 0.01), max(voc, 2.0), core, amb, hum,
                         ch2o / max(float(np.power(10, voc)), 1e-6), (core - cc) / max(t, 1e-3), t])
            labels.append({"fresh": 0, "stale": 1, "masked": 2}[cls])
    df = pd.DataFrame(rows, columns=SENSOR_COLS)
    df["label"] = labels
    return df

if SENSOR_CSV and os.path.isfile(SENSOR_CSV):
    sensor_df = pd.read_csv(SENSOR_CSV)
    sensor_df.columns = [c.strip().lower() for c in sensor_df.columns]
    assert "label" in sensor_df.columns and all(c in sensor_df.columns for c in SENSOR_COLS), \
        f"CSV must contain {SENSOR_COLS + ['label']}"
else:
    sensor_df = generate_synthetic_sensor_data()
print(f"Sensor dataset shape: {sensor_df.shape}")

# Proposal data-visualizations: adulteration boundary + cross-correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5))
for lab, name, c in [(0, "Fresh", "tab:green"), (1, "Stale (natural)", "tab:orange"), (2, "Chemically masked", "tab:red")]:
    d = sensor_df[sensor_df.label == lab]
    ax.scatter(d.ch2o_ppm, d.voc_log_ohm, s=12, alpha=0.6, c=c, label=name)
ax.set_xlabel("Formaldehyde CH2O (ppm) — ZE08"); ax.set_ylabel("VOC log(Ω) — BME680")
ax.set_title("Adulteration Boundary Scatterplot (CH2O vs VOC)"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/adulteration_boundary.png", dpi=150); plt.show()

plt.figure(figsize=(8, 6))
corr = sensor_df[SENSOR_COLS].corr()
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(SENSOR_COLS)), SENSOR_COLS, rotation=45, ha="right")
plt.yticks(range(len(SENSOR_COLS)), SENSOR_COLS)
for i in range(len(SENSOR_COLS)):
    for j in range(len(SENSOR_COLS)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(label="Pearson r"); plt.title("Sensor Cross-Correlation Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_correlation.png", dpi=150); plt.show()

# %% [markdown]
# ### B2. Preprocess (min-max + moving average) and train the sensor MLP

# %%
bounds = {c: (sensor_df[c].min(), sensor_df[c].max()) for c in SENSOR_COLS}
scaled = sensor_df.copy()
for c in SENSOR_COLS:                                   # proposal: min-max scaling
    lo, hi = bounds[c]
    scaled[c] = (sensor_df[c] - lo) / max(hi - lo, 1e-9)
smoothed = scaled.copy()                                # proposal: moving-average smoothing
smoothed[SENSOR_COLS] = ndimage.uniform_filter1d(scaled[SENSOR_COLS].values, size=SMOOTH_WIN, axis=0)

idx = np.random.permutation(len(smoothed))
n_val = int(0.15 * len(smoothed))
va, tr = idx[:n_val], idx[n_val:]
X_tr = smoothed[SENSOR_COLS].values[tr];  y_tr = smoothed.label.values[tr]
X_va = smoothed[SENSOR_COLS].values[va];  y_va = smoothed.label.values[va]
SENSOR_EMBED = 16  # telemetry feature vector length (counterpart of Fv)

def build_sensor_branch():
    inp = keras.Input(shape=(len(SENSOR_COLS),), name="sensor_input")
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(SENSOR_EMBED, activation="relu", name="sensor_embedding")(x)  # Fs feature vector
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(3, activation="softmax", name="state")(x)   # fresh / stale / chemically masked
    branch = keras.Model(inp, out, name="sensor_branch")
    feat = keras.Model(inp, branch.get_layer("sensor_embedding").output, name="sensor_feature_extractor")
    return branch, feat

sensor_branch, sensor_feature_extractor = build_sensor_branch()
sensor_branch.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
hs = sensor_branch.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=EPOCHS_SENSOR, verbose=2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(hs.history["loss"], label="train"); plt.plot(hs.history["val_loss"], label="val")
plt.title("Sensor branch — loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(hs.history["accuracy"], label="train"); plt.plot(hs.history["val_accuracy"], label="val")
plt.title("Sensor branch — accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_branch_training.png", dpi=150); plt.show()

y_pred = np.argmax(sensor_branch.predict(X_va, verbose=0), axis=1)
cm = confusion_matrix(y_va, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Fresh", "Stale", "Masked"]).plot(cmap="Blues")
plt.title("Sensor branch — Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_confusion_matrix.png", dpi=150); plt.show()
print(classification_report(y_va, y_pred, target_names=["Fresh", "Stale", "Masked"], digits=3))

sensor_branch.save(f"{OUT}/sensor_branch.keras")
sensor_feature_extractor.save(f"{OUT}/sensor_feature_extractor.keras")
print("Sensor branch saved (separate training done).")

# %% [markdown]
# ## Part C — Feature-Level Late Fusion + Mahalanobis Distance → Decision Fusion Engine
#
# Flowchart: `[Fv] + [Fs] → Mahalanobis Distance → Decision Fusion Engine → Results`.
#
# 1. Both branches are frozen; only the compact fusion head trains (TinyML friendly).
# 2. A **fusion regressor** outputs the unified **Freshness Index (0–100 %)**.
# 3. The **Mahalanobis Distance Matrix** is fit on the Fresh-class fused features;
#    when a sample *looks* fresh but lies far outside the fresh-feature
#    distribution, the **Chemical Masking alert** fires.

# %% [markdown]
# ### C1. Build a paired multimodal set and extract frozen branch features

# %%
def branch_features_for_pair(base_idx):
    """Pair image index and sensor row by freshness-state so the two modalities are coherent."""
    feats, y = [], []
    rng = np.random.default_rng(SEED)
    for i in base_idx:
        state = labels[i]                                  # 0 stale, 1 fresh (image domain)
        pool = tr_sensor_of_state[state] if state in tr_sensor_of_state else tr_sensor_of_state[2]
        cand = int(rng.choice(pool, size=1)[0])
        img, _ = load_img(paths[i], labels[i])
        feats.append((img, smoothed[SENSOR_COLS].values[cand]))
        y.append(state)
    return feats, np.array(y)

# group sensor training rows by label for class-coherent pairing (2 = chemical masking)
tr_sensor_of_state = {int(s): np.flatnonzero(y_tr == s) for s in (0, 1, 2) if (y_tr == s).any()}

pair_feats, pair_state = branch_features_for_pair(np.random.permutation(len(paths))[:PAIR_LIMIT]
                                                  if len(paths) > PAIR_LIMIT else np.arange(len(paths)))
imgs = tf.stack([f[0] for f in pair_feats])
sens = tf.convert_to_tensor(np.stack([f[1] for f in pair_feats]), dtype=tf.float32)

Fv = image_feature_extractor.predict(imgs, verbose=0)      # 128-D visual features (frozen branch A)
Fs = sensor_feature_extractor.predict(sens, verbose=0)     # 16-D telemetry features (frozen branch B)
Z = np.concatenate([Fv, Fs], axis=1)                       # late-fusion joint matrix (128 + 16)
print("Fused feature matrix Z:", Z.shape)

# %% [markdown]
# ### C2. Mahalanobis Distance Matrix (fit on Fresh class) + masked-state proxy

# %%
def fit_mahalanobis(Z_fresh):
    mean = Z_fresh.mean(axis=0)
    cov = np.cov(Z_fresh.T) + 1e-4 * np.eye(Z_fresh.shape[1])   # regularized covariance
    inv_cov = np.linalg.inv(cov)
    thr = chi2.ppf(0.99, df=Z_fresh.shape[1])                   # 99 % confidence boundary
    return mean, inv_cov, math.sqrt(thr)

z_fresh = Z[pair_state == 1]
if len(z_fresh) < Z.shape[1] + 2:                               # make sure covariance is estimable
    reps = int(np.ceil((Z.shape[1] + 2) / max(len(z_fresh), 1)))
    z_fresh = np.tile(z_fresh, (reps, 1)) + np.random.normal(0, 1e-3, (len(z_fresh) * reps, Z.shape[1]))
MAH_MEAN, MAH_INV_COV, MAH_THRESHOLD = fit_mahalanobis(z_fresh)
print(f"Mahalanobis threshold (99 %, chi2): {MAH_THRESHOLD:.3f}")

def mahalanobis_distance(z):
    d = z - MAH_MEAN
    return np.sqrt(np.einsum("...i,ij,...j->...", d, MAH_INV_COV, d))

# %% [markdown]
# ### C3. Train the Decision Fusion Engine (freshness-index regression + anomaly state)

# %%
freshness_gt = np.where(pair_state == 1, np.random.uniform(0.85, 1.0, len(pair_state)),
                 np.where(pair_state == 0, np.random.uniform(0.05, 0.35, len(pair_state)),
                          np.random.uniform(0.60, 0.85, len(pair_state)))).astype(np.float32)

def build_fusion_engine():
    inp = keras.Input(shape=(EMBED_DIM + SENSOR_EMBED,), name="fusion_input")
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="freshness_index")(x)
    return keras.Model(inp, out, name="fusion_engine")

fusion_engine = build_fusion_engine()
fusion_engine.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae", metrics=[keras.metrics.MeanAbsoluteError()])
FEAT_V = image_feature_extractor
FEAT_S = sensor_feature_extractor
FEAT_V.trainable = False; FEAT_S.trainable = False
hf = fusion_engine.fit(Z, freshness_gt, validation_split=0.15, epochs=EPOCHS_FUSION, verbose=2)

plt.plot(hf.history["loss"], label="train"); plt.plot(hf.history["val_loss"], label="val")
plt.title("Fusion engine — MAE loss"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fusion_training.png", dpi=150); plt.show()
fusion_engine.save(f"{OUT}/fusion_engine.keras")

# %% [markdown]
# ### C4. Decision Fusion Engine — final inference rule (thresholds per proposal)

# %%
CH2O_ALERT_PPM = 2.0   # ZE08-CH2O exceeds ambient formaldyhyde background

def decision_fusion_engine(image_path, sensor_row_raw):
    """Full pipeline for one sample: image + raw sensor vector -> {freshness_index, alert, state}."""
    img, _ = load_img(image_path, 0)
    fn = img[tf.newaxis, ...]
    v = image_feature_extractor.predict(fn, verbose=0)[0]
    r = np.array(sensor_row_raw, dtype=np.float32)
    r = (r - np.array([bounds[c][0] for c in SENSOR_COLS])) / np.array([max(bounds[c][1] - bounds[c][0], 1e-9) for c in SENSOR_COLS])
    s = sensor_feature_extractor.predict(r[tf.newaxis, ...], verbose=0)[0]
    z = np.concatenate([v, s])[np.newaxis, ...] if v.shape[-1] + s.shape[-1] == EMBED_DIM + SENSOR_EMBED \
        else np.concatenate([v, s]).reshape(1, -1)
    index = float(fusion_engine.predict(z, verbose=0)[0][0]) * 100.0
    d = float(mahalanobis_distance(z[0]))
    raw = dict(zip(SENSOR_COLS, np.asarray(sensor_row_raw, dtype=np.float32)))
    masked_gas = raw["ch2o_ppm"] > CH2O_ALERT_PPM
    if masked_gas and d > MAH_THRESHOLD:
        return {"freshness_index": round(index, 1), "mahalanobis": round(d, 2),
                "state": "CHEMICAL MASKING ALERT — visual freshness contradicts toxic gas profile"}
    if index >= 70:
        state = "Fresh"
    elif d > MAH_THRESHOLD and index >= 50:
        state = "Suspect — anomaly (internal rot / drift)"
    else:
        state = "Stale / Spoiled"
    return {"freshness_index": round(index, 1), "mahalanobis": round(d, 2), "state": state}

# quick demo on one known item
demo = decision_fusion_engine(paths[0], sensor_df.iloc[tr[0]][SENSOR_COLS].values)
print(json.dumps(demo, indent=2))

# %% [markdown]
# ### C5. Export for ESP32-S3 (TensorFlow Lite Micro, INT8 quantization)

# %%
def make_representative_data():
    n = min(100, len(paths))
    for i in range(n):                                   # representative images for INT8 calibration
        img, _ = load_img(paths[i], labels[i])
        yield [tf.cast(img[tf.newaxis, ...], tf.float32)]

def convert_tflite_int8(model=None):
    conv = tf.lite.TFLiteConverter.from_keras_model(model if model is not None else image_branch)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = make_representative_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8   # OV2640 frames enter as uint8/uint
    conv.inference_output_type = tf.int8
    return conv.convert()

artifacts = ["image_branch", "image_feature_extractor", "sensor_branch",
             "sensor_feature_extractor", "fusion_engine"]

try:
    tfl = convert_tflite_int8(image_branch)
    with open(f"{OUT}/image_branch_int8.tflite", "wb") as f: f.write(tfl)
    with open(f"{OUT}/image_branch.cc", "w") as f:
        f.write("const unsigned char g_image_branch[] = {" + ",".join(str(b) for b in tfl) + "};\n")
    conv = tf.lite.TFLiteConverter.from_keras_model(fusion_engine)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(f"{OUT}/fusion_engine_dyn.tflite", "wb") as f: f.write(conv.convert())
    print("TFLite INT8 export OK")
except Exception as e:
    print("INT8 export fallback:", e)
    conv = tf.lite.TFLiteConverter.from_keras_model(image_branch)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(f"{OUT}/image_branch_dyn.tflite", "wb") as f: f.write(conv.convert())
    print("Saved dynamic-range TFLite instead")

sensor_meta = {"sensor_cols": SENSOR_COLS, "min": {c: float(bounds[c][0]) for c in SENSOR_COLS},
               "max": {c: float(bounds[c][1]) for c in SENSOR_COLS},
               "mahalanobis_threshold": float(MAH_THRESHOLD), "ch2o_alert_ppm": CH2O_ALERT_PPM}
with open(f"{OUT}/edge_config.json", "w") as f: json.dump(sensor_meta, f, indent=2)

print("\nArtifacts in", OUT, ":")
for f in sorted(glob.glob(f"{OUT}/*")):
    print(f"  {os.path.basename(f):40s} {os.path.getsize(f)/1024:8.1f} KB")
print("\nDone — upload these to the ESP32-S3 firmware repo (TensorFlow Lite Micro interp).")
