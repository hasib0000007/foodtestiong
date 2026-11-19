# Tri-Layer Multimodal Food Freshness & Safety Detection — Kaggle Training Pipeline

Training code for the thesis *“Image and Sensor Data Integration with Mahalanobis
Distance for AI-Enabled Food Freshness and Safety Detection”* — built according to
the submitted proposal (DOCX) and the presentation flowchart (Slide 8).

Everything runs on Kaggle (GPU T4). The **image branch and the sensor branch are
trained completely separately**, exactly as required, and only meet at the
feature-level late-fusion stage.

## 1. Flowchart → Notebook mapping (Slide 8)

```
                 Food sample
        ┌────────────┴────────────┐
   IMAGE STREAM              SENSOR STREAM
   Camera (OV2640)           ZE08-CH2O │ BME680 │ MLX90614
        │                        │ Normalization
        │                        │ Kalman-style smoothing + drift handling
        │                        │ Min-Max scaling
        ▼                        ▼
   MobileNetV2 (α=0.35)     MLP Regression Module      ← Part A / Part B (separate training)
     → Fv (128-D)             → Fs (16-D)
        └────────────┬────────────┘
      concatenate (late fusion, joint matrix Z = [Fv | Fs])
                     ▼
            Mahalanobis Distance Matrix            ← Part C
                     ▼
            Decision Fusion Engine
      → Freshness Index 0–100 %  +  Chemical-Masking alert
                     ▼
            Results (…later: WiFi/BLE → Mobile App)
```

| Notebook part | Branch | Output |
|---|---|---|
| **Part A** | Image branch: MobileNetV2 (DepthwiseSeparable, 224×224, [0,1] input) → fresh/stale head | `image_branch.keras`, `image_feature_extractor.keras` (Fv = 128-D) |
| **Part B** | Sensor branch: CH2O/VOC/temp/RH → min-max + moving-average → MLP → fresh / stale / chemically-masked (3-state) | `sensor_branch.keras`, `sensor_feature_extractor.keras` (Fs = 16-D) |
| **Part C** | Freeze both branches → feature-level **late fusion** → **Mahalanobis Distance** (fit on Fresh class, regularized covariance, χ² 99 % threshold) → fusion engine (Freshness Index regression) → **Decision Fusion Engine** rule → TFLite **INT8** export | `fusion_engine.keras`, `image_branch_int8.tflite`, `fusion_engine_dyn.tflite`, `image_branch.cc` (ESP-NN C array), `edge_config.json` |

## 2. How to run on Kaggle (step by step)

1. Go to kaggle.com → **Create → New Notebook** → **File → Import Notebook** →
   upload `notebooks/food_freshness_kaggle_training.ipynb`.
2. **Add Input** → attach an image dataset. Verified working options:
   - [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) (folders: `fresh_apple`, `stale_apple`, …)
   - [Food Freshness Dataset](https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset) (70 k images, `Fresh/`, `Rotten/`)
   - [Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
3. In the **Setup cell**, set:
   - `IMAGE_DATA_DIR = "/kaggle/input/<dataset-folder-path>"` (point to the folder
     that directly contains the class sub-folders),
   - `SENSOR_CSV = None` → uses the built-in **synthetic 96-hour decay generator**
     that follows the proposal (natural decay at 26–36 °C, high RH, formalin
     dilution masking). When you collect your own multi-sensor CSV, set
     `SENSOR_CSV = "/kaggle/input/<your-dataset>/<file>.csv"` — the CSV must
     contain: `ch2o_ppm, voc_log_ohm, core_temp_c, ambient_temp_c, humidity_pct,
     ch2o_voc_ratio, temp_rise_rate, hours_elapsed, label (0=fresh,1=stale,2=masked)`.
4. **Settings → Accelerator → GPU T4** → **Run All**.
5. Outputs (models, plots, `.tflite`) land in **`/kaggle/working/outputs/`** →
   download them from the notebook's *Output* tab.

## 3. Generated results (usable in the thesis report)

| Plot / file | Thesis use |
|---|---|
| `adulteration_boundary.png` | Data Visualization §: CH2O-vs-VOC scatter separating natural vs chemically-preserved samples |
| `sensor_correlation.png` | Data Visualization §: sensor cross-correlation heatmap |
| `image_branch_training.png`, `sensor_branch_training.png` | Implementation §: training/validation curves |
| `sensor_confusion_matrix.png` + `classification_report` | Testing §: precision / recall / F1 for Fresh vs Stale vs Masked |
| `fusion_training.png` | Late-fusion convergence curve |
| `image_branch_int8.tflite`, `image_branch.cc`, `edge_config.json` | Hardware chapter: the files flashed into the ESP32-S3 (TFLite Micro tensor arena + sensor scaling constants + Mahalanobis threshold) |

## 4. Decision Fusion Engine rule (matches proposal §Methodology-5)

1. Compute visual features `Fv` (image branch) and telemetry features `Fs` (sensor branch),
   concatenate to `Z = [Fv | Fs]`.
2. `freshness_index = 100 × fusion_engine(Z)` → unified 0–100 % score.
3. `d = mahalanobis(Z)` — statistical distance from the Fresh-class feature distribution.
4. If `ch2o_ppm > 2.0 ppm` **and** `d > threshold` → **“Chemical Masking alert”**
   (looks fresh but gas profile contradicts → unsynthesizable by vision-only models).
5. Else freshness-index bands → *Fresh / Suspect anomaly / Stale-Spoiled*.

## 5. Notes

- MobileNetV2 `alpha=0.35` is used — Keras’s smallest hardware-valid width
  multiplier, implementing the proposal’s “0.25-class” memory idea; kernel
  supports 0.35, 0.5, 0.75, 1.0. 0.35 keeps the model well under the 250 KB
  tensor-arena strategy after INT8 quantization (~0.8 MB flatbuffer for α=0.35
  full backbone; the fusion engine itself is only 15 KB).
- The Mahalanobis covariance is ridge-regularized (+1e-4·I) and its threshold is
  the χ² 99 % quantile with df = feature dimension (144).
- Image scaling [0,255]→[0,1] per proposal; the branch maps it to ImageNet’s
  [-1,1] range internally with a single `Rescaling` layer (quantization-friendly).
