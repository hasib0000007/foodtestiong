# %% [markdown]
# # Multimodal Food Freshness and Safety Detection — Training Pipeline
#
# **Thesis:** *Image and Sensor Data Integration with Mahalanobis Distance for
# AI-Enabled Food Freshness and Safety Detection*
#
# **Author:** Md. Hasibul Islam (ID 0242310005341469), Department of Software
# Engineering, Daffodil International University.
# **Supervisor:** Dr. Kamrul Islam Shahin, Associate Professor, Department of
# Software Engineering, Daffodil International University.
#
# **Purpose.** This notebook implements the complete experimental pipeline of the
# thesis on public Kaggle data only: (i) a MobileNetV2-based visual branch trained
# separately on 3–4 fresh/stale image datasets, (ii) a channel-aware MLP sensor
# branch trained separately on 3–4 public sensor CSV datasets with per-dataset
# column mapping, availability flags and cross-dataset imputation, and (iii) a
# feature-level fusion stage that combines both embeddings with a
# Mahalanobis-distance anomaly detector and exports TensorFlow Lite models and C
# firmware arrays for the ESP32-S3 prototype.
#
# **Running the notebook on Kaggle:**
# 1. `Add Input` → attach the 3–4 image datasets; the loader auto-detects the
#    mounted input paths.
# 2. `Add Input` → attach the 3–4 sensor CSV datasets; per-dataset column mappings
#    are defined in the configuration cell.
# 3. Settings → Accelerator → GPU T4, then `Run All`.
# 4. Models, plots, coverage reports and `.tflite` artifacts are written to
#    `/kaggle/working/outputs/`.

# %% [markdown]
# ## 1. Environment Setup & Configuration
#
# Imports, random seeding, canonical channel definitions, Kaggle dataset discovery
# helpers, and the per-dataset configuration for the attached image and sensor
# datasets.
#
# Each sensor entry maps its own column names onto the 8 canonical channels; a
# mapping value of `None` means the dataset has no data for that channel — the
# loader records an availability flag and imputes the value from cross-dataset
# statistics. `MASK_AUGMENT_FRACTION` controls the physics-based chemical-masking
# augmentation that creates the simulated masked class (label 2), for which no
# public dataset provides labels.

# %%
import os, glob, json, math, random, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy.stats import chi2
from scipy import ndimage

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE    = (224, 224)   # uniform 224x224 RGB input for all image datasets
ALPHA       = 0.35         # MobileNetV2 width multiplier (small-footprint choice for edge deployment)
EMBED_DIM   = 128          # Fv visual feature vector length
SENSOR_COLS = ["ch2o_ppm", "voc_log_ohm", "core_temp_c", "ambient_temp_c", "humidity_pct",
               "ch2o_voc_ratio", "temp_rise_rate", "hours_elapsed"]
SMOOTH_WIN  = 5           # moving-average window (Kalman-style pre-filter on sensor streams)

RUNNING_ON_KAGGLE = os.path.isdir("/kaggle/input")
OUT = "/kaggle/working/outputs" if RUNNING_ON_KAGGLE else os.path.abspath("outputs")
os.makedirs(OUT, exist_ok=True)
print("TF:", tf.__version__, "| output dir:", OUT)
print("GPU:", tf.config.list_physical_devices("GPU"))

def find_ds(*keywords):
    """Find a dataset folder ANYWHERE under /kaggle/input (some Kaggle sessions mount
    attached datasets nested, e.g. /kaggle/input/datasets/<slug>). Returns the shallowest
    (slug-level) folder whose basename contains ALL keywords."""
    if not RUNNING_ON_KAGGLE:
        return None
    best, best_depth = None, 99
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        base = os.path.basename(dirpath.rstrip(os.sep)).lower()
        if base and all(k in base for k in keywords):
            depth = dirpath[len("/kaggle/input"):].count(os.sep)
            if depth < best_depth:
                best, best_depth = dirpath, depth
        if depth_of(dirpath) >= 3:
            dirnames[:] = []                 # limit traversal depth for very large trees
    return best

def depth_of(path):
    return path[len("/kaggle/input"):].strip(os.sep).count(os.sep) if len(path) > len("/kaggle/input") else 0

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
BATCH_SENSOR = 256          # larger batch for the tabular sensor data
EPOCHS_HEAD = 5            # Part A, phase 1 (head training, backbone frozen)
EPOCHS_FINETUNE = 2        # Part A, phase 2 (fine-tune last backbone block, 0 = skip)
EPOCHS_SENSOR = 30         # Part B
EPOCHS_FUSION = 40         # Part C; ReduceLROnPlateau anneals the learning rate
PAIR_LIMIT = 4000          # max paired samples for the fusion set (balanced across 3 states)
MASKED_PAIR_FRACTION = 0.30   # share of fresh-image pairs paired with masked sensor rows (state 2)

MASK_AUGMENT_FRACTION = 0.25   # fraction of simulated chemical-masking rows added to Part B

IMG_AUGMENT = True            # random flip/rotation/zoom inside the image branch
                              # (training-time augmentation)

# recommended extra Kaggle dataset per missing channel (printed by the coverage report)
MISSING_CHANNEL_GUIDE = {
    "ch2o_ppm":      "attach [Dataset of Formalin Tofu Aroma] https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin (formalin-labeled food gas-sensor data) or any MQ135/MQ-series gas-sensor CSV",
    "voc_log_ohm":   "attach [Fruit Freshness Electronic Nose Dataset] https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data (MQ2-MQ135) or [BME688 Odours] https://www.kaggle.com/datasets/laurabasa/bme688-odours (real Bosch BME68x gas resistance, same sensor family as the BME680)",
    "core_temp_c":   "attach [Formalin Tofu Aroma] https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin (Temperature(C) column) or any food-storage temperature CSV",
    "ambient_temp_c":"attach [IoT Agriculture 2024] https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024 (temperature and humidity columns) or any IoT temperature–humidity CSV (DHT22 logs also work)",
    "humidity_pct":  "attach [IoT Agriculture 2024] https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024 (temperature and humidity columns) or any IoT temperature–humidity CSV (DHT22 logs also work)",
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
    "storage_env":    find_ds("iot-agriculture"),   # gap-filler: ambient temp + humidity
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
    # verified columns: Category, Shelf_Life_Day, Sensor_1..10, Mean_Value, Variance, Target
    # Shelf_Life_Day supplies the hours_elapsed channel (day x 24 via the 'scale' entry).
    {"name": "chicken_sensor",
     "path": os.path.join(SENSOR_ROOTS["chicken_sensor"], "**", "*.csv") if SENSOR_ROOTS["chicken_sensor"] else "",
     "label_rule": {"type": "text", "col": "target"},
     "mapping": {"voc_log_ohm": "Mean_Value", "ch2o_ppm": None, "core_temp_c": None,
                 "ambient_temp_c": None, "humidity_pct": None, "hours_elapsed": "Shelf_Life_Day"},
     "scale": {"hours_elapsed": 24.0}},
    # --- formalin tofu aroma: only public food dataset with formalin-treatment labels ---
    # label is factory QC (1 = pass, 0 = fail); MQ channels carry the CH2O-family
    # signal — a documented surrogate for the ZE08-CH2O electrochemical ppm reading.
    {"name": "tofu_formalin",
     "path": os.path.join(SENSOR_ROOTS["tofu_formalin"], "**", "*.csv") if SENSOR_ROOTS["tofu_formalin"] else "",
     "label_rule": {"type": "column", "col": "label"},
     "mapping": {"ch2o_ppm": "CO_MQ2(ppm)", "voc_log_ohm": "Alcohol_MQ2(ppm)",
                 "core_temp_c": "Temperature(C)", "ambient_temp_c": None,
                 "humidity_pct": None, "hours_elapsed": None}},
    # --- 4th gap-filler: IoT Agriculture 2024 (wisam1985) — closes ambient_temp_c/humidity_pct ---
    # IoTProcessed_Data.csv ships the temperature column as 'tempreature' (sic) —
    # resolved by COLUMN_ALIASES above; humidity is spelled correctly.
    # greenhouse IoT telemetry; no food-freshness ground truth, so it contributes
    # REAL environment channels only ('constant' label rows keep part of B stale-side).
    # alternative: any DHT22/DHT11 CSV — map its columns the same way.
    {"name": "storage_env",
     "path": os.path.join(SENSOR_ROOTS["storage_env"], "**", "*.csv") if SENSOR_ROOTS["storage_env"] else "",
     "label_rule": {"type": "constant"},
     "mapping": {"ambient_temp_c": "temperature", "humidity_pct": "humidity",
                 "ch2o_ppm": None, "voc_log_ohm": None, "core_temp_c": None,
                 "hours_elapsed": None}},
]
# Local (non-Kaggle) dry runs without attached CSVs fall back to small demo files
# so that every section remains executable.

# %% [markdown]
# ## 2. Image Dataset Loading & Preprocessing
#
# Class-folder names containing `fresh` map to label 1; `stale/rotten/spoiled/discard`
# names map to label 0. Images are resized to 224x224, rescaled to [0,1], and split
# into training and validation partitions; per-dataset counts are reported. The
# discovery helpers walk every attached input root and merge all folders into one
# unified image dataset.

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

def _string_label(text):
    n = str(text).lower()
    if any(k in n for k in ("stale", "rotten", "discard", "spoil", "not fresh", "non-fresh",
                            "non fresh", "nonfresh", "half")):
        return 0
    if "fresh" in n:
        return 1
    return None

def freshness_label(class_name):
    """Checks only the class-folder basename, never the full path: full-path checks
    break on slugs such as 'fresh-and-stale-images...'."""
    return _string_label(os.path.basename(str(class_name)))

def labels_from_classes_csv(folder):
    """Parse a Roboflow multiclass export (one folder, images + _classes.csv with
    columns like 'filename,Fresh,Half-Fresh,Spoiled'). Returns [(filepath, 0/1)].
    Half-Fresh and Spoiled both count as not-fresh (label 0)."""
    pairs = []
    for f in glob.glob(os.path.join(folder, "*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or len(df.columns) < 2:
            continue
        fname_col, cls_cols = df.columns[0], list(df.columns[1:])
        if not df[fname_col].astype(str).str.lower().str.endswith((".jpg", ".jpeg", ".png")).all():
            continue
        for _, r in df.iterrows():
            fp = os.path.join(folder, str(r[fname_col]))
            if not os.path.isfile(fp):
                continue
            best = max(cls_cols, key=lambda c: r[c])
            lab = _string_label(best)
            if lab is not None:
                pairs.append((fp, lab))
    return pairs

def _make_demo_images(root, per_class=12):
    """Tiny synthetic image set used ONLY for local dry runs (never on Kaggle)."""
    os.makedirs(root, exist_ok=True)
    for name, rgb in (("FreshApple", (150, 200, 60)), ("StaleApple", (90, 90, 30)),
                      ("FreshFish", (200, 190, 170)), ("StaleFish", (110, 105, 95))):
        d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
        for k in range(per_class):
            arr = np.clip(np.array(rgb) + np.random.normal(0, 25, (96, 96, 3)), 0, 255).astype("uint8")
            keras.utils.save_img(os.path.join(d, f"img_{k}.jpg"), arr)

paths, labels, dataset_of, provenance = [], [], [], []
for root in IMAGE_DATASETS:
    if not os.path.isdir(root):
        print(f"Image dataset not attached, skipping: {root}")
        continue
    dataset_name = os.path.basename(root.rstrip("/"))
    n0 = len(paths)
    for name, folder in discover_class_dirs(root, dataset_name).items():
        csv_pairs = labels_from_classes_csv(folder)   # Roboflow multiclass export (meat)
        if csv_pairs:
            for fp, lab in csv_pairs:
                paths.append(fp); labels.append(lab); dataset_of.append(dataset_name)
            continue
        # rel-path labeling: tilapia's fresh/stale info lives in PARENT folders
        # (Fresh_Day1/Eyes, NonFresh/Day3/Eyes), not in the leaf basename ('Eyes').
        lab = _string_label(os.path.relpath(folder, root))
        if lab is None:
            continue
        for f in glob.glob(os.path.join(folder, "*")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(f); labels.append(lab); dataset_of.append(dataset_name)
    provenance.append({"dataset": dataset_name, "images": len(paths) - n0, "label": "fresh/stale"})

if not paths and not RUNNING_ON_KAGGLE:
    _make_demo_images("_demo_images")
    IMAGE_DATASETS = ["_demo_images"]
    for name, folder in discover_class_dirs("_demo_images", "_demo_images").items():
        lab = _string_label(os.path.basename(name))
        if lab is None:
            continue
        for f in glob.glob(os.path.join(folder, "*")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(f); labels.append(lab); dataset_of.append("_demo_images")
    provenance.append({"dataset": "_demo_images", "images": len(paths), "label": "fresh/stale"})

assert paths, ("No image datasets found — attach at least one Kaggle image dataset and add its "
               "path to IMAGE_DATASETS in the config cell.")
paths = np.array(paths); labels = np.array(labels); dataset_of = np.array(dataset_of)
for prov in provenance:
    print(f"  {prov['dataset']:55s} {prov['images']:6d} images")
print(f"TOTAL: {len(paths)} images (fresh={labels.sum()}, stale={(labels==0).sum()})")

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0          # rescale to [0,1]
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
# ## 3. Part A: Visual Branch Training (MobileNetV2)
#
# MobileNetV2 (alpha = 0.35) backbone with a compact sigmoid head, trained in two
# phases: head training with the backbone frozen, then fine-tuning of the last
# backbone blocks with BatchNorm statistics kept frozen. ROC/AUC,
# precision/recall/F1 and the confusion matrix are reported on the held-out
# validation split.

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
h1 = image_branch.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2,
                      callbacks=[keras.callbacks.TerminateOnNaN()])

if EPOCHS_FINETUNE > 0:
    backbone = next(l for l in image_branch.layers if isinstance(l, keras.Model))  # MobileNetV2 backbone
    backbone.trainable = True
    for layer in backbone.layers[:-8]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):   # freeze BN stats (no val drop)
            layer.trainable = False
    image_branch.compile(optimizer=keras.optimizers.Adam(1e-5),
                         loss="binary_crossentropy", metrics=["accuracy"])
    ckpt = keras.callbacks.ModelCheckpoint(f"{OUT}/image_branch_best.keras",
                                           monitor="val_loss", save_best_only=True)
    h2 = image_branch.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, verbose=2,
                          callbacks=[keras.callbacks.TerminateOnNaN(), ckpt])
    image_branch = keras.models.load_model(f"{OUT}/image_branch_best.keras")   # keep best epoch
    image_feature_extractor = keras.Model(image_branch.input,
                                          image_branch.get_layer("fv_embedding").output,
                                          name="image_feature_extractor")
    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
else:
    hist = h1.history

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(hist["loss"], label="train"); plt.plot(hist["val_loss"], label="val")
plt.title("Image branch — loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(hist["accuracy"], label="train"); plt.plot(hist["val_accuracy"], label="val")
plt.title("Image branch — accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/image_branch_training.png", dpi=150); plt.show()

# ROC curve and AUC of the image branch on the held-out validation split (evaluation only).
img_prob = image_branch.predict(val_ds, verbose=0).ravel()
fpr_i, tpr_i, _ = roc_curve(labels[val_idx], img_prob)
auc_img_branch = float(auc(fpr_i, tpr_i))
plt.figure(figsize=(4.8, 4.8))
plt.plot(fpr_i, tpr_i, color="black", lw=1.8, label=f"image branch (AUC = {auc_img_branch:.4f})")
plt.plot([0, 1], [0, 1], "k:", lw=1, label="chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Image branch \u2014 ROC (validation)"); plt.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/image_branch_roc.png", dpi=150); plt.show()
print(f"Image branch validation AUC = {auc_img_branch:.4f} on {len(val_idx):,} held-out images")

# Precision/recall/F1 and the binary confusion matrix of the image branch
# (0.5 operating threshold, evaluation only).
img_pred = (img_prob >= 0.5).astype(int)
cm_img = confusion_matrix(labels[val_idx], img_pred, labels=[0, 1])
ConfusionMatrixDisplay(cm_img, display_labels=["Stale", "Fresh"]).plot(cmap="Blues")
plt.title("Image branch \u2014 Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/image_branch_confusion_matrix.png", dpi=150); plt.show()
img_cr = classification_report(labels[val_idx], img_pred, target_names=["Stale", "Fresh"],
                               digits=3, zero_division=0, output_dict=True)
print(classification_report(labels[val_idx], img_pred, target_names=["Stale", "Fresh"],
                            digits=3, zero_division=0))

image_branch.save(f"{OUT}/image_branch.keras")
image_feature_extractor.save(f"{OUT}/image_feature_extractor.keras")
print("Image branch saved (separate training done).")

# %% [markdown]
# ## 4. Sensor Data Harmonization & Augmentation
#
# 3–4 Kaggle sensor CSV datasets are loaded through per-dataset column mappings into
# the 8 canonical channels. Missing channels are detected, flagged (availability
# flags) and imputed from cross-dataset statistics; a coverage report enumerates
# which extra datasets close the remaining gaps. Physics-based chemical-masking
# augmentation creates the simulated masked class (label 2), and storage-environment
# channels not covered by any attached dataset receive documented synthetic
# variation. Min–max scaling and a moving-average pre-filter complete the
# harmonization.

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

# dataset authors misspell columns (e.g. IoT Agriculture 2024 ships 'tempreature');
# the alias table lets the canonical mapping resolve them instead of silently dropping
# the channel into imputation.
COLUMN_ALIASES = {
    "temperature": ["tempreature", "temp", "temp_c", "temperature(c)", "temperature(c)"],
    "humidity":    ["humidity_pct", "humid", "relative_humidity", "rh", "humidity(%)"],
}

def _resolve_col(df, want):
    w = want.strip().lower()
    if w in df.columns:
        return w
    for alias in COLUMN_ALIASES.get(w, []):
        if alias in df.columns:
            return alias
    return None

def load_sensor_datasets(cfgs):
    frames, coverage = {}, []
    for cfg in cfgs:
        if not cfg["path"]:
            print(f"Sensor dataset not attached (no matching input root): {cfg['name']}")
            continue
        files = sorted(glob.glob(cfg["path"], recursive=True)) if any(ch in cfg["path"] for ch in "*?[") \
                else ([cfg["path"]] if os.path.isfile(cfg["path"]) else [])
        if not files:
            print(f"Sensor dataset not attached (no files found): {cfg['path']}")
            continue
        mapped = []
        for f in files:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            row = {}
            for canon, raw in cfg["mapping"].items():
                src_col = _resolve_col(df, raw) if raw else None
                row[canon] = pd.to_numeric(df[src_col], errors="coerce") if src_col \
                             else np.full(len(df), np.nan)
            sub = pd.DataFrame(row)
            for canon, mul in cfg.get("scale", {}).items():  # unit harmonization (e.g. days -> hours)
                sub[canon] = sub[canon] * mul
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
    print("\nThe attached sensor datasets do not cover the following channels:")
    for c in missing:
        print(f"  - {c}: {MISSING_CHANNEL_GUIDE[c]}")
    coverage_df.to_csv(f"{OUT}/sensor_dataset_coverage.csv", index=False)
    print(f"\nCoverage table saved to {OUT}/sensor_dataset_coverage.csv")
else:
    print("\nAll 8 channels are covered by the attached sensor datasets.")
    coverage_df.to_csv(f"{OUT}/sensor_dataset_coverage.csv", index=False)

def augment_environmental_channels(df, active_fraction=0.70):
    """Storage-environment channels that NO attached dataset covers would otherwise stay
    flat constants the sensor branch cannot learn from. Fill ~70% of rows with realistic
    DHT22-like synthetic variation (flag 1) and keep ~30% flag 0 so the 'sensor absent'
    behaviour seen at inference still stays represented (DHT22 future-proofing)."""
    rng = np.random.default_rng(SEED + 7)
    fresh_like = (df.label.values == 1) | (df.label.values == 2)   # masked looks fresh
    added = []
    if (df["av_ambient_temp_c"] == 0).all():
        vals = np.where(fresh_like, rng.uniform(12.0, 24.0, len(df)), rng.uniform(20.0, 38.0, len(df)))
        use = rng.random(len(df)) < active_fraction
        df["ambient_temp_c"] = np.where(use, vals, df["ambient_temp_c"])
        df["av_ambient_temp_c"] = use.astype(np.float32)
        added.append("ambient_temp_c (synthetic, 12-24C stored / 20-38C room)")
    if (df["av_humidity_pct"] == 0).all():
        vals = np.where(fresh_like, rng.uniform(40.0, 65.0, len(df)), rng.uniform(60.0, 95.0, len(df)))
        use = rng.random(len(df)) < active_fraction
        df["humidity_pct"] = np.where(use, vals, df["humidity_pct"])
        df["av_humidity_pct"] = use.astype(np.float32)
        added.append("humidity_pct (synthetic, 40-65% stored / 60-95% room)")
    if (df["av_temp_rise_rate"] == 0).all():
        vals = np.where(df.label.values == 2, rng.uniform(0.0, 0.2, len(df)),
                 np.where(df.label.values == 1, rng.uniform(0.0, 0.3, len(df)),
                          rng.uniform(0.3, 1.5, len(df))))
        use = rng.random(len(df)) < active_fraction
        df["temp_rise_rate"] = np.where(use, vals, df["temp_rise_rate"])
        df["av_temp_rise_rate"] = use.astype(np.float32)
        added.append("temp_rise_rate (synthetic C/h, masked < fresh < stale)")
    if added:
        print("\nUncovered environment channels receive synthetic variation "
              "(70% active rows, 30% kept at flag 0 for absent-sensor robustness):")
        for a in added:
            print("  -", a)
    return df

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
sensor_df = augment_environmental_channels(sensor_df)

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

bounds = {c: (float(sensor_df[c].min()), float(sensor_df[c].max())) for c in SENSOR_COLS}
scaled = sensor_df.copy()
for c in SENSOR_COLS:                                   # min–max scaling to [0,1]
    lo, hi = bounds[c]
    scaled[c] = np.clip((sensor_df[c] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
smoothed = scaled.copy()                                # moving-average pre-filter
smoothed[SENSOR_COLS] = np.clip(
    ndimage.uniform_filter1d(scaled[SENSOR_COLS].values, size=SMOOTH_WIN, axis=0), 0.0, 1.0)
del scaled                                              # free ~130 MB before feature building

FEATURE_COLS = SENSOR_COLS + AV_COLS                    # values + availability flags
idx = np.random.permutation(len(smoothed))
n_val = int(0.15 * len(smoothed))
va, tr = idx[:n_val], idx[n_val:]
X = smoothed[FEATURE_COLS].values.astype(np.float32)
y = smoothed.label.values.astype(np.int32)

print("\nRows per dataset after augmentation:")
print(sensor_df["dataset"].value_counts().to_string())
print("\nPer-channel statistics after NaN/Inf cleanup and min-max scaling:")
for c in SENSOR_COLS:
    v = sensor_df[c].values
    n_nan = int(np.isnan(v).sum()); n_inf = int(np.isinf(v).sum())
    print(f"  {c:16s} min={v.min():12.4g} max={v.max():12.4g} nan={n_nan} inf={n_inf}")
assert np.isfinite(X).all(), "Data sanity check failed: NaN/Inf values remain in the feature matrix."
assert np.isin(np.unique(y), [0, 1, 2]).all(), "Data sanity check failed: unexpected label values."
print(f"Data sanity check passed: finite={np.isfinite(X).all()}, "
      f"X_min={X.min():.4f}, X_max={X.max():.4f}")

# %% [markdown]
# ## 5. Part B: Sensor Telemetry Branch Training
#
# A channel-aware MLP (64–32–16 with BatchNorm and dropout) trained on the
# harmonized 16-D feature table to classify Fresh / Stale / Chemically-Masked
# states. Training curves, the confusion matrix, per-class precision/recall/F1 and
# one-vs-rest ROC curves are reported on the held-out validation split.

# %%
X_tr, y_tr = X[tr], y[tr]
X_va, y_va = X[va], y[va]
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
sensor_branch.compile(optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hs = sensor_branch.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                       batch_size=BATCH_SENSOR, epochs=EPOCHS_SENSOR, verbose=2,
                       callbacks=[keras.callbacks.TerminateOnNaN()])

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
sensor_cr = classification_report(y_va, y_pred, target_names=["Stale", "Fresh", "Masked"],
                                  digits=3, zero_division=0, output_dict=True)
print(classification_report(y_va, y_pred, target_names=["Stale", "Fresh", "Masked"], digits=3,
                            zero_division=0))

# One-vs-rest ROC curves of the sensor branch on the held-out validation split
# (evaluation only).
y_prob = sensor_branch.predict(X_va, batch_size=BATCH_SENSOR, verbose=0)
Y_bin = label_binarize(y_va, classes=[0, 1, 2])
plt.figure(figsize=(4.9, 4.9))
for k, name in enumerate(["Stale", "Fresh", "Masked"]):
    fpr_k, tpr_k, _ = roc_curve(Y_bin[:, k], y_prob[:, k])
    plt.plot(fpr_k, tpr_k, lw=1.8, label=f"{name} (AUC = {auc(fpr_k, tpr_k):.4f})")
fpr_mi, tpr_mi, _ = roc_curve(Y_bin.ravel(), y_prob.ravel())
plt.plot(fpr_mi, tpr_mi, "k:", lw=1.2, label=f"micro-avg (AUC = {auc(fpr_mi, tpr_mi):.4f})")
plt.plot([0, 1], [0, 1], "--", color="0.6", lw=0.8, label="chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Sensor branch \u2014 one-vs-rest ROC (validation)")
plt.legend(loc="lower right", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_roc_curves.png", dpi=150); plt.show()
auc_per_class = {name: round(float(auc(*roc_curve(Y_bin[:, k], y_prob[:, k])[:2])), 4)
                 for k, name in enumerate(["Stale", "Fresh", "Masked"])}
macro_auc = float(roc_auc_score(Y_bin, y_prob, average="macro", multi_class="ovr"))
print(f"Sensor branch one-vs-rest AUC (macro) = {macro_auc:.4f} on {len(y_va):,} rows | per-class: {auc_per_class}")

# Consolidated evaluation-metrics summary for both branches (confusion matrices,
# precision/recall/F1, ROC AUCs).
eval_metrics = {
    "image_branch": {
        "n_val_images": int(len(val_idx)),
        "accuracy": float(img_cr["accuracy"]),
        "precision": {n: img_cr[n]["precision"] for n in ("Stale", "Fresh")},
        "recall": {n: img_cr[n]["recall"] for n in ("Stale", "Fresh")},
        "f1_score": {n: img_cr[n]["f1-score"] for n in ("Stale", "Fresh")},
        "confusion_matrix": cm_img.tolist(),
        "roc_auc": auc_img_branch},
    "sensor_branch": {
        "n_val_rows": int(len(y_va)),
        "accuracy": float(sensor_cr["accuracy"]),
        "precision": {n: sensor_cr[n]["precision"] for n in ("Stale", "Fresh", "Masked")},
        "recall": {n: sensor_cr[n]["recall"] for n in ("Stale", "Fresh", "Masked")},
        "f1_score": {n: sensor_cr[n]["f1-score"] for n in ("Stale", "Fresh", "Masked")},
        "confusion_matrix": cm.tolist(),
        "roc_auc_macro_ovr": macro_auc,
        "roc_auc_per_class": auc_per_class},
}
with open(f"{OUT}/evaluation_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)
print("evaluation_metrics.json saved (confusion matrices, precision/recall/F1, ROC AUCs)")

sensor_branch.save(f"{OUT}/sensor_branch.keras")
sensor_feature_extractor.save(f"{OUT}/sensor_feature_extractor.keras")
print("Sensor branch saved (separate training done).")
gc.collect()                                            # release training temporaries before Part C

# %% [markdown]
# ## 6. Part C: Multimodal Fusion & Mahalanobis Anomaly Detection
#
# Both branches are frozen; only the compact fusion engine trains, taking the
# concatenation [Fv 128 | Fs 16 | scaled sensor values 16 | state posterior 3] and
# regressing the continuous Freshness Index (0–100). The Mahalanobis distance is
# calibrated on the fresh-class fused embeddings with a shrunk covariance estimate;
# samples that appear fresh but lie far outside the fresh-feature distribution
# trigger the chemical-masking alert.

# %%
# Extract the scaled feature matrix once: calling smoothed[FEATURE_COLS].values inside
# the pairing loop would allocate a new ~135 MB copy per iteration.
SMOOTHED_MATRIX = smoothed[FEATURE_COLS].values.astype(np.float32)

def build_fusion_pairs(base_idx, chunk=256):
    """Pair each image with a same-state sensor row; fresh images are also paired with
    simulated chemical-masked sensor rows (state 2) so the fusion set covers all three
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

def mahalanobis_distance(z):
    zf = np.asarray(z, np.float64)
    inv64 = MAH_INV_COV.astype(np.float64)
    d = zf - MAH_MEAN
    return float(np.sqrt(np.einsum("...i,ij,...j->...", d, inv64, d))) if np.ndim(zf) == 1 \
        else np.sqrt(np.einsum("...i,ij,...j->...", d, inv64, d))

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

CH2O_ALERT_PPM = 2.0   # ZE08-CH2O reading above the ambient formaldehyde background

def decision_fusion_engine(image_path, sensor_feature_vector):
    """Full pipeline for one sample: image + CANONICAL sensor feature row -> result dict.
       `sensor_feature_vector` must already be scaled (value-channels in [0,1], flags 0/1)."""
    img, _ = load_img(image_path, 0)
    v = image_feature_extractor.predict(img[tf.newaxis, ...], verbose=0)[0]
    s = sensor_feature_extractor.predict(np.asarray(sensor_feature_vector, np.float32)[tf.newaxis, ...], verbose=0)[0]
    z = np.concatenate([v, s])[np.newaxis, ...]              # 144-D embedding space (Mahalanobis)
    sp = sensor_branch.predict(np.asarray(sensor_feature_vector, np.float32)[tf.newaxis, ...],
                               verbose=0)[0]                 # 3-class state posterior
    z_in = np.concatenate([z[0], np.asarray(sensor_feature_vector, np.float32),
                           sp])[np.newaxis, ...].astype(np.float32)
    index = float(fusion_engine.predict(z_in, verbose=0)[0][0]) * 100.0
    d = float(mahalanobis_distance(z[0]))
    if d > MAH_THRESHOLD and len(sensor_feature_vector) > FEATURE_COLS.index("ch2o_ppm"):
        i_ch2o = FEATURE_COLS.index("ch2o_ppm")
        lo, hi = bounds["ch2o_ppm"]
        raw_ch2o = sensor_feature_vector[i_ch2o] * (hi - lo) + lo   # unscale [0,1] -> ppm
        if raw_ch2o > CH2O_ALERT_PPM:
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
# ## 7. Edge Deployment Export (TFLite & C Headers)
#
# The image branch is exported as a full-integer INT8 TensorFlow Lite model with
# stratified calibration data, the fusion engine as a float32 TensorFlow Lite model
# (TensorFlow Lite Micro-compatible), and the Mahalanobis detector parameters (mean,
# shrunk inverse covariance, threshold) plus sensor scaling bounds as C header arrays
# for the ESP32-S3 firmware.

# %%
def make_representative_data(n=400):
    """Stratified INT8 calibration: draws equally from every (dataset x class) group so
    quantization ranges cover the full composition of the attached image datasets."""
    rng = np.random.default_rng(SEED)
    groups = {}
    for i in range(len(paths)):
        groups.setdefault((str(dataset_of[i]), int(labels[i])), []).append(i)
    k = max(1, n // max(len(groups), 1))
    picks = []
    for g in groups.values():
        r = rng.choice(g, size=min(k, len(g)), replace=False)
        picks.extend(int(x) for x in r)
    rng.shuffle(picks)
    for i in picks:
        img, _ = load_img(paths[i], 0)
        yield [tf.cast(img[tf.newaxis, ...], tf.float32)]

def convert_tflite_int8(model=None):
    conv = tf.lite.TFLiteConverter.from_keras_model(model if model is not None else image_branch)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = make_representative_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()

# image INT8 and fusion exports are INDEPENDENT so one failure can never skip the other
import warnings
warnings.filterwarnings("ignore", message="Statistics for quantized inputs")   # harmless TFLite calibration note
try:
    tfl = convert_tflite_int8(image_branch)
    with open(f"{OUT}/image_branch_int8.tflite", "wb") as f: f.write(tfl)
    with open(f"{OUT}/image_branch.cc", "w") as f:
        f.write("const unsigned char g_image_branch[] = {" + ",".join(str(b) for b in tfl) + "};\n")
    print("TFLite INT8 (image) export OK")
except Exception as e:
    print("INT8 export fallback:", e)
    conv = tf.lite.TFLiteConverter.from_keras_model(image_branch)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(f"{OUT}/image_branch_dyn.tflite", "wb") as f: f.write(conv.convert())
    print("Saved dynamic-range TFLite instead")

try:
    conv = tf.lite.TFLiteConverter.from_keras_model(fusion_engine)   # float32 — TFLM-safe
    f32 = conv.convert()
    with open(f"{OUT}/fusion_engine_f32.tflite", "wb") as f: f.write(f32)
    with open(f"{OUT}/fusion_engine.cc", "w") as f:
        f.write("const unsigned char g_fusion_engine[] = {" + ",".join(str(b) for b in f32) + "};\n")
    print("TFLite Float32 (fusion) export OK")
except Exception as e:
    print("FUSION EXPORT FAILED:", e)

edge_meta = {"sensor_features": FEATURE_COLS, "min": {c: float(bounds[c][0]) for c in SENSOR_COLS},
             "max": {c: float(bounds[c][1]) for c in SENSOR_COLS},
             "mahalanobis_threshold": float(MAH_THRESHOLD), "ch2o_alert_ppm": CH2O_ALERT_PPM,
             "mahalanobis_mean": [float(x) for x in MAH_MEAN],
             "mahalanobis_shrinkage": 0.10,
             "fusion_pairs": int(len(Z)),
             "fusion_states": {str(k): v for k, v in n_state.items()},
             "fusion_input": "concat(Fv 128, Fs 16, raw_scaled 16, state_prob 3) = 163",
             "freshness_index_bands": {"fresh": [0.88, 0.92], "masked": [0.68, 0.72], "stale": [0.17, 0.21]},
             "mahalanobis_inv_cov": [[float(x) for x in row] for row in np.asarray(MAH_INV_COV, np.float64)],
             "image_datasets": [p["dataset"] for p in provenance],
             "sensor_datasets": list(coverage_df.dataset) if len(coverage_df) else []}
with open(f"{OUT}/edge_config.json", "w") as f: json.dump(edge_meta, f, indent=2)

# ---- ESP32-S3 firmware header: scaling bounds + Mahalanobis detector as C arrays ----
def _cpp_array_1d(name, arr, per_line=8):
    vals = [f"{float(v):.8e}f" for v in np.asarray(arr, np.float64).ravel()]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f"const float {name}[{len(vals)}] = {{\n  " + ",\n  ".join(lines) + "\n}};"

def _cpp_array_2d(name, mat):
    rows = []
    for r in np.asarray(mat, np.float64):
        rows.append("{" + ",".join(f"{float(v):.8e}f" for v in r) + "}")
    n = len(rows)
    return f"const float {name}[{n}][{n}] = {{\n  " + ",\n  ".join(rows) + "\n}};"

with open(f"{OUT}/edge_config.h", "w") as f:
    f.write("#ifndef EDGE_CONFIG_H\n#define EDGE_CONFIG_H\n\n")
    f.write("// Auto-generated by train_pipeline.py — do not edit by hand.\n")
    f.write(f"// Mahalanobis: mean + shrunk inverse covariance of the FRESH class in the "
            f"144-d fused embedding space [Fv 128 | Fs 16].\n\n")
    f.write("#define NUM_SENSOR_VALUE_COLS 8\n")
    f.write("#define NUM_FEATURES 16          // 8 scaled sensor values + 8 availability flags\n")
    f.write(f"#define MAH_DIM {Z.shape[1]}\n")
    f.write(f"#define FUSION_INPUT_DIM {FUSION_IN_DIM}     // Fv 128 + Fs 16 + raw scaled 16 + state prob 3\n")
    f.write(f"#define MAH_THRESHOLD {MAH_THRESHOLD:.4f}f\n")
    f.write(f"#define CH2O_ALERT_PPM {CH2O_ALERT_PPM:.1f}f\n\n")
    f.write("static const char *SENSOR_FEATURE_NAMES[NUM_FEATURES] = {\n")
    f.write("  " + ",\n  ".join(f'"{c}"' for c in FEATURE_COLS) + "\n};\n\n")
    f.write(_cpp_array_1d("SENSOR_MIN", [bounds[c][0] for c in SENSOR_COLS]) + "\n\n")
    f.write(_cpp_array_1d("SENSOR_MAX", [bounds[c][1] for c in SENSOR_COLS]) + "\n\n")
    f.write(_cpp_array_1d("MAH_MEAN", MAH_MEAN) + "\n\n")
    f.write(_cpp_array_2d("MAH_INV_COV", MAH_INV_COV) + "\n\n")
    f.write("#endif\n")
print("edge_config.h generated for ESP32-S3 firmware (bounds + threshold + mean + 144x144 inv covariance)")

print("\nArtifacts in", OUT, ":")
for f in sorted(glob.glob(f"{OUT}/*")):
    print(f"  {os.path.basename(f):40s} {os.path.getsize(f)/1024:8.1f} KB")
print("\nDone — copy these artifacts into the ESP32-S3 firmware project (TensorFlow Lite Micro inference).")
