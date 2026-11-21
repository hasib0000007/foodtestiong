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

MASK_AUGMENT_FRACTION = 0.0    # optional simulated chemical-masking rows for Part B; 0 keeps the masked class 100% real (class_weight handles the imbalance)

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
    # factory QC label: 1 = pass (formalin-free), 0 = fail (formalin detected).
    # QC-fail rows are remapped to the masked class (label 2) inside load_sensor_datasets,
    # so the masked class trains on real adulteration-confirmed rows. MQ channels carry
    # the CH2O-family signal — a documented surrogate for the ZE08-CH2O ppm reading.
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
