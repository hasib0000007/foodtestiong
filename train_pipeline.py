# %% [markdown]
# # Tri-Layer Multimodal Food Freshness & Safety Detection — Kaggle-Only Training
#
# **Thesis:** *Image and Sensor Data Integration with Mahalanobis Distance for
# AI-Enabled Food Freshness and Safety Detection*
#
# **100% Kaggle strategy — no primary/self-collected data.** The image branch is
# trained across **3–4 public image datasets** and the sensor branch across
# **3–4 public sensor CSV datasets** (both branches fully separately). The
# multi-dataset loader detects **missing channels per sensor dataset**, imputes
# them, and produces a coverage report that tells you exactly which extra Kaggle
# dataset to attach to fill the gap.
#
# | Flowchart stage (Slide 8) | Notebook part |
# |---|---|
# | Image Stream → Camera (OV2640) → **MobileNetV2** | Part A — Image branch (3–4 datasets, trained **separately**) |
# | Sensor Stream → (ZE08-CH2O, BME680, MLX90614) → Normalization + Kalman-style smoothing → **MLP Regression Module** | Part B — Sensor branch (3–4 CSVs, trained **separately**) |
# | **Mahalanobis Distance** → **Decision Fusion Engine** → Results | Part C — Feature-level late fusion + anomaly layer |
# | Results → WiFi/BLE → Mobile App | Artifacts exported in Part C feed the ESP32-S3 firmware |
#
# **How to run on Kaggle:**
# 1. `Add Input` → attach 3–4 image datasets from `DATASETS.md` and fill `IMAGE_DATASETS`.
# 2. `Add Input` → attach 3–4 sensor CSV datasets and fill `SENSOR_CSVS` (with a
#    small column mapping per dataset — see the config cell).
# 3. Settings → Accelerator → GPU T4, then `Run All`.
# 4. All models, plots, coverage reports and `.tflite` files appear in
#    `/kaggle/working/outputs/`.

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
ALPHA       = 0.35         # MobileNetV2 width multiplier (deepest HW-valid value for the 0.25 idea)
EMBED_DIM   = 128          # Fv visual feature vector length
SENSOR_COLS = ["ch2o_ppm", "voc_log_ohm", "core_temp_c", "ambient_temp_c", "humidity_pct",
               "ch2o_voc_ratio", "temp_rise_rate", "hours_elapsed"]
SMOOTH_WIN  = 5           # moving-average window (Kalman-style pre-filter on sensor streams)

RUNNING_ON_KAGGLE = os.path.isdir("/kaggle/input")
OUT = "/kaggle/working/outputs" if RUNNING_ON_KAGGLE else os.path.abspath("outputs")
os.makedirs(OUT, exist_ok=True)
print("TF:", tf.__version__, "| output dir:", OUT)
print("GPU:", tf.config.list_physical_devices("GPU"))

# %% [markdown]
# ### 0.1 Multi-dataset configuration (EDIT THESE)
#
# **Image datasets (3–4):** list the Kaggle input paths after attaching them.
# Missing paths are skipped with a warning, so you can attach any subset.
#
# **Sensor datasets (3–4):** each entry = `path` (a CSV file or glob of CSVs) +
# `mapping` from canonical channels to that dataset's own column names. A mapping
# value of `None` means **this dataset has no data for that channel** — the loader
# records an availability flag and fills the value from cross-dataset statistics.
#
# `MASK_AUGMENT_FRACTION` controls the physics-based formalin-masking augmentation
# that creates the `chemically-masked` class (label 2). No public Kaggle dataset
# provides chemical-mask labels, so this controlled simulation is the honest way to
# obtain that class — state it as "simulated adulteration augmentation" in the thesis.

# %%
def find_ds(*keywords):
    """Return the top-level /kaggle/input/<slug> folder whose basename contains ALL keywords."""
    if not RUNNING_ON_KAGGLE:
        return None
    for root in sorted(glob.glob("/kaggle/input/*")):
        base = os.path.basename(root).lower()
        if all(k in base for k in keywords):
            return root
    return None

_IMAGE_FALLBACK = [
    "/kaggle/input/fresh-and-stale-images-of-fruits-and-vegetables",   # fruits (raghavrpotdar)
    "/kaggle/input/tilapia-fish-fresh-and-non-fresh-species",          # fish (haripriyasanga)
    "/kaggle/input/meat-freshness-image-dataset",                      # meat (vinayakshanawad)
    # "/kaggle/input/food-freshness-dataset",                          # optional 70k ablation set
]
IMAGE_DATASETS = [p for p in (find_ds("fresh-and-stale"), find_ds("tilapia"),
                              find_ds("meat-freshness-image")) if p] or _IMAGE_FALLBACK
print("Auto-detected image datasets:", IMAGE_DATASETS)
IMG_VAL_FRACTION = 0.15
BATCH = 32
BATCH_SENSOR = 256          # larger batch for the ~50k-row sensor table
EPOCHS_HEAD = 5            # Part A, phase 1 (head training, backbone frozen)
EPOCHS_FINETUNE = 2        # Part A, phase 2 (fine-tune last backbone block, 0 = skip)
EPOCHS_SENSOR = 30         # Part B
EPOCHS_FUSION = 25         # Part C
PAIR_LIMIT = 400           # max paired samples for the fusion set

MASK_AUGMENT_FRACTION = 0.25   # fraction of simulated chemical-masking rows added to Part B

IMG_AUGMENT = True            # random flip/rotation/zoom inside the image branch —
                              # multiplies the effective training-set size for free

# recommended extra Kaggle dataset per missing channel (printed by the coverage report)
MISSING_CHANNEL_GUIDE = {
    "ch2o_ppm":      "attach [Dataset of Formalin Tofu Aroma] https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin (formalin-labeled gas-sensor food data — closest public proxy for the ZE08-CH2O) or any MQ135 gas-sensor CSV",
    "voc_log_ohm":   "attach [Fruit Freshness Electronic Nose Dataset] https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data (MQ2-MQ135) or [BME688 Odours] https://www.kaggle.com/datasets/laurabasa/bme688-odours (real Bosch BME68x gas resistance — same family as our BME680)",
    "core_temp_c":   "attach [Formalin Tofu Aroma] https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin (Temperature(C) column) or any food-storage temperature CSV",
    "ambient_temp_c":"attach any IoT food-storage / cold-storage temperature-humidity CSV (e.g. 'IoT Agriculture 2024')",
    "humidity_pct":  "attach any IoT food-storage / cold-storage temperature-humidity CSV (e.g. 'IoT Agriculture 2024')",
    "ch2o_voc_ratio":"derived automatically from ch2o_ppm and voc_log_ohm — no extra dataset needed",
    "temp_rise_rate":"derived automatically from core_temp_c and hours_elapsed — no extra dataset needed",
    "hours_elapsed": "usually derivable from timestamps or file/day indices; else attach any time-series food CSV",
}

# each sensor dataset: path/glob (auto-resolved from /kaggle/input roots) + column mapping.
# The recursive '**' glob picks up EVERY CSV in the dataset, whatever the folder layout.
SENSOR_ROOTS = {
    "enose_fruits":   find_ds("electronic-nose") or find_ds("e-nose"),
    "chicken_sensor": find_ds("chicken-meat"),
    "tofu_formalin":  find_ds("aroma-tahu") or find_ds("formalin-tofu"),
}
print("Auto-resolved sensor roots:", SENSOR_ROOTS)
SENSOR_CSVS = [
    # --- e-nose MQ dataset (real hardware telemetry; decay day-series per fruit) ---
    {"name": "enose_fruits",
     "path": os.path.join(SENSOR_ROOTS["enose_fruits"], "**", "*.csv") if SENSOR_ROOTS["enose_fruits"] else "",
     "label_rule": {"type": "day_decay", "fresh_days": [1, 2], "stale_days": [3, 4, 5]},
     "mapping": {"voc_log_ohm": "MQ2", "ch2o_ppm": "MQ135",      # MQ135 responds to aldehyde-family gases
                 "core_temp_c": None, "ambient_temp_c": None,
                 "humidity_pct": None, "hours_elapsed": None}},
    # --- chicken meat dataset: 10 virtual VOC channels + mean/variance + Target ---
    # real columns: sensor-level VOC values, 'mean', 'variance', 'target'
    # ('Fresh_Day4' / 'FrozenThawed_Day6') — no temperature/humidity columns here.
    {"name": "chicken_sensor",
     "path": os.path.join(SENSOR_ROOTS["chicken_sensor"], "**", "*.csv") if SENSOR_ROOTS["chicken_sensor"] else "",
     "label_rule": {"type": "text", "col": "target"},
     "mapping": {"voc_log_ohm": "mean", "ch2o_ppm": None, "core_temp_c": None,
                 "ambient_temp_c": None, "humidity_pct": None, "hours_elapsed": None}},
    # --- formalin tofu aroma: only public food dataset with formalin-treatment labels ---
    # label is factory QC (1 = pass, 0 = fail); MQ channels carry the CH2O-family
    # signal — a documented surrogate for the ZE08-CH2O electrochemical ppm reading.
    {"name": "tofu_formalin",
     "path": os.path.join(SENSOR_ROOTS["tofu_formalin"], "**", "*.csv") if SENSOR_ROOTS["tofu_formalin"] else "",
     "label_rule": {"type": "column", "col": "label"},
     "mapping": {"ch2o_ppm": "CO_MQ2(ppm)", "voc_log_ohm": "Alcohol_MQ2(ppm)",
                 "core_temp_c": "Temperature(C)", "ambient_temp_c": None,
                 "humidity_pct": None, "hours_elapsed": None}},
    # optional 4th (gap-filler): any IoT temp/humidity CSV, e.g.
    # {"name": "storage_env",
    #  "path": "/kaggle/input/iot-agriculture-2024/*.csv",
    #  "label_rule": {"type": "column", "col": "<label-col>"},
    #  "mapping": {"ambient_temp_c": "<temp-col>", "humidity_pct": "<humidity-col>",
    #              "ch2o_ppm": None, "voc_log_ohm": None, "core_temp_c": None,
    #              "hours_elapsed": None}},
]
# if no CSV is configured (e.g. dry run), the notebook falls back to its physics
# simulator so every cell still executes; replace the list above for real runs.

# %% [markdown]
# ## Part A — Image Branch (trained independently, branch 1: Fv via MobileNetV2)
#
# Camera (OV2640) → rescale to 224×224 → quantized depthwise-separable **MobileNetV2**
# feature extractor → binary Fresh/Stale classification head. Trained across
# **3–4 Kaggle image datasets** (any subset of the list works). The trained backbone
# is then re-wired into a *feature extractor* (`Fv`, 128-D) that Part C freezes.

# %% [markdown]
# ### A1. Discover classes across all image datasets and build one merged dataset
# Class folder names containing `fresh` map to label 1, `stale/rotten/spoiled/discard`
# map to label 0 (`non-fresh` also counts as stale). Per-dataset counts are printed.

# %%
def discover_class_dirs(root, dataset_name):
    class_dirs = {}
    for d in sorted(glob.glob(os.path.join(root, "**/"), recursive=True)):
        d = d.rstrip(os.sep)
        files = [f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            class_dirs[os.path.join(dataset_name, os.path.relpath(d, root))] = d
        if len(class_dirs) >= 60:
            break
    return class_dirs

def freshness_label(class_name):
    n = class_name.lower()
    if "stale" in n or "rotten" in n or "discard" in n or "spoil" in n or "not fresh" in n or "non-fresh" in n or "nonfresh" in n or "half" in n:
        return 0
    if "fresh" in n:
        return 1
    return None

def _make_demo_images(root, per_class=12):
    """Tiny synthetic image set used ONLY for local dry runs (never on Kaggle)."""
    os.makedirs(root, exist_ok=True)
    for name, rgb in (("FreshApple", (150, 200, 60)), ("StaleApple", (90, 90, 30)),
                      ("FreshFish", (200, 190, 170)), ("StaleFish", (110, 105, 95))):
        d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
        for k in range(per_class):
            arr = np.clip(np.array(rgb) + np.random.normal(0, 25, (96, 96, 3)), 0, 255).astype("uint8")
            keras.utils.save_img(os.path.join(d, f"img_{k}.jpg"), arr)

paths, labels, provenance = [], [], []
for root in IMAGE_DATASETS:
    if not os.path.isdir(root):
        print(f"[skip] image dataset not attached: {root}")
        continue
    dataset_name = os.path.basename(root.rstrip("/"))
    n0 = len(paths)
    for name, folder in discover_class_dirs(root, dataset_name).items():
        lab = freshness_label(os.path.basename(name))
        if lab is None:
            continue
        for f in glob.glob(os.path.join(folder, "*")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(f); labels.append(lab)
    provenance.append({"dataset": dataset_name, "images": len(paths) - n0, "label": "fresh/stale"})

if not paths and not RUNNING_ON_KAGGLE:
    _make_demo_images("_demo_images")
    IMAGE_DATASETS = ["_demo_images"]
    for name, folder in discover_class_dirs("_demo_images", "_demo_images").items():
        lab = freshness_label(os.path.basename(name))
        if lab is None:
            continue
        for f in glob.glob(os.path.join(folder, "*")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(f); labels.append(lab)
    provenance.append({"dataset": "_demo_images", "images": len(paths), "label": "fresh/stale"})

assert paths, ("No image datasets found — attach at least one Kaggle image dataset and add its "
               "path to IMAGE_DATASETS in the config cell.")
paths = np.array(paths); labels = np.array(labels)
for prov in provenance:
    print(f"  {prov['dataset']:55s} {prov['images']:6d} images")
print(f"TOTAL: {len(paths)} images (fresh={labels.sum()}, stale={(labels==0).sum()})")

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

idx = np.random.permutation(len(paths))
n_val = max(1, int(len(paths) * IMG_VAL_FRACTION))
val_idx, tr_idx = idx[:n_val], idx[n_val:]
train_ds, val_ds = make_ds(tr_idx, True), make_ds(val_idx, False)

# %% [markdown]
# ### A2. Build + train the image branch (Phase 1 head training, then optional fine-tune)

# %%
def build_image_branch():
    inp = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = layers.Rescaling(2.0, offset=-1.0, name="to_imagenet_range")(inp)  # [0,1] -> [-1,1] (ImageNet conv)
    if IMG_AUGMENT:   # active only during .fit(); disabled at inference/INT8 export
        x = layers.RandomFlip("horizontal", name="aug_flip")(x)
        x = layers.RandomRotation(0.08, name="aug_rot")(x)
        x = layers.RandomZoom(0.08, name="aug_zoom")(x)
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
# 3–4 Kaggle sensor CSV datasets → per-dataset column mapping → **missing-channel
# detection** (availability flags + cross-dataset imputation) → min-max scaling +
# moving-average smoothing (Kalman-style pre-filter) → **channel-aware MLP** →
# Fresh / Stale / Chemically-Masked classification. Trained fully separately from
# the image branch.

# %% [markdown]
# ### B1. Load every sensor dataset, detect missing channels, build coverage report

# %%
def resolve_label(rule, df, fname):
    """Return a label array for one CSV via the dataset's label rule."""
    t = rule["type"]
    if t == "column":
        col = next((c for c in df.columns if c.strip().lower() == rule["col"].strip().lower()), None)
        assert col is not None, f"{fname}: label column '{rule['col']}' not found"
        y = pd.to_numeric(df[col], errors="coerce")
        return y.fillna(1).astype(int).values
    if t == "text":           # string labels like 'Fresh_Day4' vs 'FrozenThawed_Day6'
        col = next((c for c in df.columns if c.strip().lower() == rule["col"].strip().lower()), None)
        assert col is not None, f"{fname}: label column '{rule['col']}' not found"
        sl = df[col].astype(str).str.lower()
        y = sl.str.contains(rule.get("fresh_match", "fresh"), regex=False) \
            & ~sl.str.contains(rule.get("stale_match", "\x00"), regex=False)
        return y.astype(int).values
    if t == "day_decay":      # filenames like 'Banana D3.csv' — later days are staler
        seq = fname.rsplit(".", 1)[0].lower().split()[-1].lstrip("d")
        day = int(seq) if seq.isdigit() else 1
        return np.full(len(df), 1 if day in rule["fresh_days"] else 0)
    return np.zeros(len(df), dtype=int)   # "constant"

def load_sensor_datasets(cfgs):
    frames, coverage = {}, []
    for cfg in cfgs:
        if not cfg["path"]:
            print(f"[skip] sensor dataset not attached (no /kaggle/input root found): {cfg['name']}")
            continue
        files = sorted(glob.glob(cfg["path"], recursive=True)) if any(ch in cfg["path"] for ch in "*?[") \
                else ([cfg["path"]] if os.path.isfile(cfg["path"]) else [])
        if not files:
            print(f"[skip] sensor dataset not attached: {cfg['path']}")
            continue
        mapped = []
        for f in files:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            row = {}
            for canon, raw in cfg["mapping"].items():
                row[canon] = pd.to_numeric(df[raw.strip().lower()], errors="coerce") if raw \
                             and raw.strip().lower() in df.columns else np.full(len(df), np.nan)
            sub = pd.DataFrame(row)
            for canon in ("hours_elapsed", "ambient_temp_c"):      # still missing? neutral fill + flag
                if sub[canon].isna().all():
                    sub[canon] = 0.0 if canon == "hours_elapsed" else 25.0
                    sub[f"av_{canon}"] = 0.0
            base = os.path.basename(f)
            if sub["hours_elapsed"].nunique() <= 1:                 # unlogged time -> day index (e.g. 'Banana D3.csv')
                seq = base.rsplit(".", 1)[0].lower().split()[-1].lstrip("d")
                if seq.isdigit():
                    sub["hours_elapsed"] = int(seq) * 24.0
                    sub["av_hours_elapsed"] = 1.0
            sub["label"] = resolve_label(cfg["label_rule"], df, base)
            sub["dataset"] = cfg["name"]
            mapped.append(sub)
        d = pd.concat(mapped, ignore_index=True)
        for canon in SENSOR_COLS:                                        # derived channels start uncomputed
            if canon not in d:
                d[canon] = np.nan
        for canon in SENSOR_COLS:                                        # availability flag per dataset+channel
            d[f"av_{canon}"] = d[f"av_{canon}"] if f"av_{canon}" in d else (0.0 if d[canon].isna().all() else 1.0)
            d[canon] = d[canon].fillna(d[canon].median())                # sparse NaN -> dataset median
        frames[cfg["name"]] = d
    assert frames, ("No sensor datasets found — attach Kaggle sensor CSVs and add entries to "
                    "SENSOR_CSVS in the config cell.")
    comb = pd.concat(frames.values(), ignore_index=True)
    # fill still-missing whole channels from cross-dataset medians (flag 0 keeps the imputation marked)
    for canon in SENSOR_COLS:
        if comb[canon].isna().any():
            comb[canon] = comb[canon].fillna(comb[canon].median())
        if f"av_{canon}" not in comb:
            comb[f"av_{canon}"] = 1.0
    # derived channels where both parents exist per row (override imputation when possible)
    both = (comb["av_ch2o_ppm"] > 0) & (comb["av_voc_log_ohm"] > 0)
    comb.loc[both, "av_ch2o_voc_ratio"] = 1.0
    comb.loc[both, "ch2o_voc_ratio"] = (comb.loc[both, "ch2o_ppm"] /
                                        np.power(10.0, comb.loc[both, "voc_log_ohm"]).clip(lower=1e-6))
    for name, g in comb.groupby("dataset"):             # coverage table built AFTER derived channels
        coverage.append({"dataset": name, "rows": len(g),
                         **{c: ("YES" if g[f'av_{c}'].iloc[0] > 0 else "-") for c in SENSOR_COLS},
                         "fresh": int((g.label == 1).sum()), "stale": int((g.label == 0).sum())})
    return comb, pd.DataFrame(coverage)

# local dry-run fallback: fabricate three small CSVs with different coverage gaps
def _demo_sensor_csvs():
    base = os.path.abspath("_demo_sensor"); os.makedirs(base, exist_ok=True)
    if glob.glob(os.path.join(base, "*.csv")):
        return
    for name, cols in (("Fruit D1.csv", ["MQ2", "MQ135"]), ("Fruit D2.csv", ["MQ2", "MQ135"]),
                       ("Storage T1.csv", ["temperature", "humidity"]), ("Cold T2.csv", ["temperature", "humidity"])):
        n = 120
        df = pd.DataFrame({c: np.abs(np.random.normal(120, 30, n)) for c in cols})
        df["ambient_temperature"] = np.random.uniform(24, 32, n)
        df.to_csv(os.path.join(base, name), index=False)
    return base

def _config_files_exist(cfgs):
    for cfg in cfgs:
        if any(ch in cfg["path"] for ch in "*?["):
            if glob.glob(cfg["path"], recursive=True):
                return True
        elif os.path.isfile(cfg["path"]):
            return True
    return False

if not _config_files_exist(SENSOR_CSVS) and not RUNNING_ON_KAGGLE:
    # local dry-run only: two demo CSV groups with deliberately different coverage
    _demo_sensor_csvs()
    SENSOR_CSVS = [
        {"name": "enose_fruits", "path": os.path.abspath("_demo_sensor/Fruit D*.csv"),
         "label_rule": {"type": "day_decay", "fresh_days": [1], "stale_days": [2]},
         "mapping": {"voc_log_ohm": "MQ2", "ch2o_ppm": "MQ135", "core_temp_c": None,
                     "ambient_temp_c": None, "humidity_pct": None, "hours_elapsed": None}},
        {"name": "storage_iot", "path": os.path.abspath("_demo_sensor/* T*.csv"),
         "label_rule": {"type": "constant", "label": 0},
         "mapping": {"core_temp_c": "temperature", "ambient_temp_c": "ambient_temperature",
                     "humidity_pct": "humidity", "voc_log_ohm": None, "ch2o_ppm": None,
                     "hours_elapsed": None}},
    ]

sensor_df, coverage_df = load_sensor_datasets(SENSOR_CSVS)
AV_COLS = [f"av_{c}" for c in SENSOR_COLS]
ALL_COLS = SENSOR_COLS + AV_COLS
print(f"\nCombined sensor dataset: {sensor_df.shape[0]} rows x {len(ALL_COLS)} features "
      f"(8 channels + 8 availability flags)")
print(f"Class balance: fresh={int((sensor_df.label==1).sum())}, stale={int((sensor_df.label==0).sum())} "
      f"+ {MASK_AUGMENT_FRACTION:.0%} chemical-masking augmentation in B2")
print(coverage_df.to_string(index=False))   # per-dataset channel coverage table

missing = [c for c, a in zip(SENSOR_COLS, AV_COLS)
           if coverage_df.empty or (coverage_df[c] == "YES").sum() == 0]
if missing:
    print("\n[MISSING DATA] No attached dataset covers these channels:")
    for c in missing:
        print(f"  • {c}\n      → {MISSING_CHANNEL_GUIDE[c]}")
    coverage_df.to_csv(f"{OUT}/sensor_dataset_coverage.csv", index=False)
    print(f"\n(Coverage table saved to {OUT}/sensor_dataset_coverage.csv — attach the suggested "
          "extra datasets and re-run to close the gaps.)")
else:
    print("\nAll 8 channels covered by the attached sensor datasets ✅")
    coverage_df.to_csv(f"{OUT}/sensor_dataset_coverage.csv", index=False)

# %% [markdown]
# ### B2. Chemical-masking augmentation + proposal visualizations
# The masked rows simulate formalin preservation: CH2O spikes while VOC decay is
# arrested. This is the ONLY source of the chemically-masked class (no public dataset
# labels adulteration), added in a controlled, clearly-documented proportion.

# %%
def generate_masking_augment(df, fraction):
    n = int(len(df) * fraction)
    src = df.sample(n, random_state=SEED).copy()
    src["ch2o_ppm"] = np.random.uniform(4.0, 9.0, n)                 # toxic gas despite look
    src["voc_log_ohm"] = np.clip(src["voc_log_ohm"] * 0.4 + 3.6, 2.0, None)  # decay arrested
    src["core_temp_c"] = src["core_temp_c"] - 1.5                    # cooler: rot slowed
    src["label"] = 2
    for c in SENSOR_COLS:                                            # augment inherits channel availability
        av = f"av_{c}"
        if av in df.columns:
            src[av] = df.loc[src.index, av].values
    src["dataset"] = "simulated_masking"
    return pd.concat([df, src], ignore_index=True)

sensor_df = generate_masking_augment(sensor_df, MASK_AUGMENT_FRACTION)

for c in SENSOR_COLS:                       # final NaN/Inf cleanup (guards loss: nan)
    sensor_df[c] = np.nan_to_num(sensor_df[c].values, nan=0.0, posinf=0.0, neginf=0.0)
for a in AV_COLS:
    sensor_df[a] = np.nan_to_num(sensor_df[a].values, nan=0.0, posinf=1.0, neginf=0.0)

fig, ax = plt.subplots(figsize=(7, 5))
for lab, name, c in [(0, "Stale (natural)", "tab:orange"), (1, "Fresh", "tab:green"), (2, "Chemically masked (simulated)", "tab:red")]:
    d = sensor_df[sensor_df.label == lab]
    if len(d) > 2000:
        d = d.sample(2000, random_state=SEED)
    ax.scatter(d.ch2o_ppm, d.voc_log_ohm, s=12, alpha=0.6, c=c, label=name)
ax.set_xlabel("Formaldehyde-family CH2O (ppm) — ZE08/MQ135"); ax.set_ylabel("VOC log(Ω) — BME680/MQ2")
ax.set_title("Adulteration Boundary Scatterplot (CH2O vs VOC)"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/adulteration_boundary.png", dpi=150); plt.show()

plt.figure(figsize=(9, 7))
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
# ### B3. Preprocess (min-max + moving average + availability flags) and train the channel-aware MLP

# %%
bounds = {c: (sensor_df[c].min(), sensor_df[c].max()) for c in SENSOR_COLS}
scaled = sensor_df.copy()
for c in SENSOR_COLS:                                   # proposal: min-max scaling
    lo, hi = bounds[c]
    scaled[c] = (sensor_df[c] - lo) / max(hi - lo, 1e-9)
smoothed = scaled.copy()                                # proposal: moving-average smoothing
smoothed[SENSOR_COLS] = ndimage.uniform_filter1d(scaled[SENSOR_COLS].values, size=SMOOTH_WIN, axis=0)

FEATURE_COLS = SENSOR_COLS + AV_COLS                    # values + availability flags
idx = np.random.permutation(len(smoothed))
n_val = int(0.15 * len(smoothed))
va, tr = idx[:n_val], idx[n_val:]
X_tr = smoothed[FEATURE_COLS].values[tr].astype(np.float32);  y_tr = smoothed.label.values[tr]
X_va = smoothed[FEATURE_COLS].values[va].astype(np.float32);  y_va = smoothed.label.values[va]
SENSOR_EMBED = 16  # telemetry feature vector length (counterpart of Fv)

def build_sensor_branch(n_in):
    inp = keras.Input(shape=(n_in,), name="sensor_input")
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(SENSOR_EMBED, activation="relu", name="sensor_embedding")(x)  # Fs feature vector
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(3, activation="softmax", name="state")(x)   # fresh / stale / chemically masked
    branch = keras.Model(inp, out, name="sensor_branch")
    feat = keras.Model(inp, branch.get_layer("sensor_embedding").output, name="sensor_feature_extractor")
    return branch, feat

sensor_branch, sensor_feature_extractor = build_sensor_branch(len(FEATURE_COLS))
sensor_branch.compile(optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hs = sensor_branch.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                       batch_size=BATCH_SENSOR, epochs=EPOCHS_SENSOR, verbose=2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(hs.history["loss"], label="train"); plt.plot(hs.history["val_loss"], label="val")
plt.title("Sensor branch — loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(hs.history["accuracy"], label="train"); plt.plot(hs.history["val_accuracy"], label="val")
plt.title("Sensor branch — accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_branch_training.png", dpi=150); plt.show()

y_pred = np.argmax(sensor_branch.predict(X_va, batch_size=BATCH_SENSOR, verbose=0), axis=1)
cm = confusion_matrix(y_va, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Stale", "Fresh", "Masked"]).plot(cmap="Blues")
plt.title("Sensor branch — Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_confusion_matrix.png", dpi=150); plt.show()
print(classification_report(y_va, y_pred, target_names=["Stale", "Fresh", "Masked"], digits=3,
                            zero_division=0))

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
        feats.append((img, smoothed[FEATURE_COLS].values[cand]))
        y.append(state)
    return feats, np.array(y)

tr_sensor_of_state = {int(s): np.flatnonzero(y_tr == s) for s in (0, 1, 2) if (y_tr == s).any()}
pair_idx = np.random.permutation(len(paths))[:PAIR_LIMIT] if len(paths) > PAIR_LIMIT else np.arange(len(paths))
pair_feats, pair_state = branch_features_for_pair(pair_idx)

imgs = tf.stack([f[0] for f in pair_feats])
sens = tf.convert_to_tensor(np.stack([f[1] for f in pair_feats]), dtype=tf.float32)
Fv = image_feature_extractor.predict(imgs, verbose=0)      # 128-D visual features (frozen branch A)
Fs = sensor_feature_extractor.predict(sens, verbose=0)     # 16-D telemetry features (frozen branch B)
Z = np.concatenate([Fv, Fs], axis=1)                       # late-fusion joint matrix (128 + 16)
print("Fused feature matrix Z:", Z.shape)

# %% [markdown]
# ### C2. Mahalanobis Distance Matrix (fit on Fresh class)

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
# ### C3. Train the Decision Fusion Engine (freshness-index regression)

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
image_feature_extractor.trainable = False; sensor_feature_extractor.trainable = False
hf = fusion_engine.fit(Z, freshness_gt, validation_split=0.15, epochs=EPOCHS_FUSION, verbose=2)

plt.plot(hf.history["loss"], label="train"); plt.plot(hf.history["val_loss"], label="val")
plt.title("Fusion engine — MAE loss"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fusion_training.png", dpi=150); plt.show()
fusion_engine.save(f"{OUT}/fusion_engine.keras")

# %% [markdown]
# ### C4. Decision Fusion Engine — final inference rule (thresholds per proposal)

# %%
CH2O_ALERT_PPM = 2.0   # ZE08-CH2O exceeds ambient formaldehyde background

def decision_fusion_engine(image_path, sensor_feature_vector):
    """Full pipeline for one sample: image + CANONICAL sensor feature row -> result dict.
       `sensor_feature_vector` must already be scaled (value-channels in [0,1], flags 0/1)."""
    img, _ = load_img(image_path, 0)
    v = image_feature_extractor.predict(img[tf.newaxis, ...], verbose=0)[0]
    s = sensor_feature_extractor.predict(np.asarray(sensor_feature_vector, np.float32)[tf.newaxis, ...], verbose=0)[0]
    z = np.concatenate([v, s])[np.newaxis, ...]
    index = float(fusion_engine.predict(z, verbose=0)[0][0]) * 100.0
    d = float(mahalanobis_distance(z[0]))
    if d > MAH_THRESHOLD and (len(sensor_feature_vector) > FEATURE_COLS.index("ch2o_ppm")
                              and sensor_feature_vector[FEATURE_COLS.index("ch2o_ppm")] > CH2O_ALERT_PPM):
        return {"freshness_index": round(index, 1), "mahalanobis": round(d, 2),
                "state": "CHEMICAL MASKING ALERT — visual freshness contradicts toxic gas profile"}
    if index >= 70:
        state = "Fresh"
    elif d > MAH_THRESHOLD and index >= 50:
        state = "Suspect — anomaly (internal rot / drift)"
    else:
        state = "Stale / Spoiled"
    return {"freshness_index": round(index, 1), "mahalanobis": round(d, 2), "state": state}

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
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()

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

edge_meta = {"sensor_features": FEATURE_COLS, "min": {c: float(bounds[c][0]) for c in SENSOR_COLS},
             "max": {c: float(bounds[c][1]) for c in SENSOR_COLS},
             "mahalanobis_threshold": float(MAH_THRESHOLD), "ch2o_alert_ppm": CH2O_ALERT_PPM,
             "image_datasets": [p["dataset"] for p in provenance],
             "sensor_datasets": list(coverage_df.dataset) if len(coverage_df) else []}
with open(f"{OUT}/edge_config.json", "w") as f: json.dump(edge_meta, f, indent=2)

print("\nArtifacts in", OUT, ":")
for f in sorted(glob.glob(f"{OUT}/*")):
    print(f"  {os.path.basename(f):40s} {os.path.getsize(f)/1024:8.1f} KB")
print("\nDone — upload these to the ESP32-S3 firmware repo (TensorFlow Lite Micro interp).")
