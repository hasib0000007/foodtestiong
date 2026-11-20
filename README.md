# Tri-Layer Multimodal Food Freshness & Safety Detection — Kaggle-Only Training Pipeline

Training code for the thesis *“Image and Sensor Data Integration with Mahalanobis
Distance for AI-Enabled Food Freshness and Safety Detection”* — built according to
the proposal and the presentation flowchart (Slide 8), using a **100 % Kaggle data
strategy** (no primary/self-collected data).

The **image branch is trained across 3–4 public image datasets** and the
**sensor branch across 3–4 public sensor-CSV datasets** — both branches fully
separately. When an attached sensor dataset lacks some canonical channels, the
pipeline detects the gap, imputes with cross-dataset statistics + availability
flags, and prints exactly which extra Kaggle dataset closes the remaining holes
(see `DATASETS.md` for the recommended combo).

## 1. Flowchart → Notebook mapping (Slide 8)

```
                 Food sample
        ┌────────────┴────────────┐
   IMAGE STREAM              SENSOR STREAM
   Camera (OV2640)           ZE08-CH2O │ BME680 │ MLX90614
   3–4 Kaggle image          3–4 Kaggle sensor CSVs
   datasets                  → Normalization + min-max
        │                    → Kalman-style smoothing
        │                    → missing-channel flags + imputation
        ▼                        ▼
   MobileNetV2 (α=0.35)     Channel-aware MLP           ← Part A / Part B (separate training)
     → Fv (128-D)             (8 channels + 8 availability flags)
        │                        → Fs (16-D)
        └────────────┬────────────┘
      concatenate (feature-level late fusion, Z = [Fv | Fs])
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
| **Part A** | Image branch: MobileNetV2 (224×224, [0,1]) trained on multiple fresh/stale corpora | `image_branch.keras`, `image_feature_extractor.keras` (Fv = 128-D) |
| **Part B** | Sensor branch: CSVs → mapping → flags → MLP → fresh / stale / chemically-masked | `sensor_branch.keras`, `sensor_feature_extractor.keras` (Fs = 16-D), `sensor_dataset_coverage.csv` |
| **Part C** | Late fusion + Mahalanobis (χ² 99 % threshold) + fusion engine + TFLite INT8 | `fusion_engine.keras`, `image_branch_int8.tflite`, `image_branch.cc`, `edge_config.json` |

## 2. How to run on Kaggle (step by step)

1. **Create → New Notebook → File → Import Notebook** → upload
   `notebooks/food_freshness_kaggle_training.ipynb`.
2. **Add Input** — attach 3–4 image datasets (verified options in `DATASETS.md`):
   - [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)
   - [Tilapia Fresh and Non Fresh (6,900)](https://www.kaggle.com/datasets/haripriyasanga/tilapia-fish-fresh-and-non-fresh-species)
   - [Meat Freshness Image Dataset](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset)
   - optional ablation: [Food Freshness Dataset (70k)](https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset)
3. **Add Input** — attach 3–4 sensor CSV datasets (the final combo in `DATASETS.md`):
   [Fruit Freshness Electronic Nose Dataset](https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data),
   [Chicken Meat Freshness Dataset](https://www.kaggle.com/datasets/ziya07/chicken-meat-freshness-dataset),
   [Dataset of Formalin Tofu Aroma](https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin)
   (closest public proxy for ZE08-CH2O), and
   [IoT Agriculture 2024](https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024) (temp + RH gap-filler).
   Fill the `SENSOR_CSVS` config with each dataset's column mapping.
   Unattached paths in both lists are skipped automatically — only run what you attach.
4. **Missing-data behaviour:** the *coverage report* cell prints a dataset × channel
   table (`sensor_dataset_coverage.csv`). Any channel no dataset covers triggers a
   `[MISSING DATA]` block that names the exact extra Kaggle dataset to attach for
   that channel (mapping in `MISSING_CHANNEL_GUIDE`). Derived channels
   (`ch2o_voc_ratio`, `temp_rise_rate`) compute themselves when their parents exist.
5. **Settings → Accelerator → GPU T4 → Run All** → outputs in
   `/kaggle/working/outputs/` (download from the notebook's Output tab).

## 3. Chemical-masking class (honesty note for the thesis)

No public dataset labels formalin-treated samples. The notebook adds **controlled
simulated adulteration augmentation** (`MASK_AUGMENT_FRACTION`, default 25 %) that
 spikes CH2O while arresting VOC decay — creating the 3rd class needed by the
Mahalanobis layer. Describe it in the thesis exactly as that: *simulated
adulteration augmentation over public sensor telemetry*.

## 4. Generated results (thesis Ch. 4–6 + defense slides)

| Plot / file | Thesis use |
|---|---|
| `sensor_dataset_coverage.csv` | Data §: multi-dataset channel coverage + missing-data strategy evidence |
| `adulteration_boundary.png` | Data Visualization §: CH2O-vs-VOC class separation |
| `sensor_correlation.png` | Data Visualization §: cross-correlation heatmap |
| `image_branch_training.png`, `sensor_branch_training.png` | Implementation §: curves |
| `sensor_confusion_matrix.png` + classification report | Testing §: precision/recall/F1 (3-state) |
| `fusion_training.png` | Late-fusion convergence |
| `image_branch_int8.tflite`, `image_branch.cc`, `edge_config.json` | Hardware chapter: ESP32-S3 flash artifacts + thresholds |

## 5. Notes

- MobileNetV2 `alpha=0.35` — Keras's smallest hardware-valid width multiplier,
  implementing the proposal's 0.25-class memory idea; INT8 flatbuffer ≈ 0.8 MB,
  fusion engine ≈ 15 KB.
- Mahalanobis covariance ridge-regularized (+1e-4·I); threshold = χ² 99 % quantile,
  df = 144.
- Image scaling [0,255]→[0,1] per proposal; internal rescaling maps to ImageNet range.

## Thesis Report (generated)

`Thesis_Report_0242310005341469.docx` is the complete Bachelor thesis report,
rebuilt from this repository's pipeline outputs by `build_thesis_final.py`.
It follows the DIU Thesis Guidelines and SWE Thesis Template 2024:

- A4, margins: left 4.0 cm, top/bottom/right 2.5 cm; Times New Roman 12 pt
- Double-spaced body text; single-spaced tables, captions, and references
- Preliminaries in Roman numerals (title page number unprinted), body in Arabic
- Exact page order: cover -> declaration & copyright -> supervisor's declaration
  -> student's declaration -> title/submission page -> acknowledgements ->
  dedication -> abstract -> table of contents -> lists of tables/figures/
  symbols/abbreviations/appendices -> Chapters 1-5 -> references -> appendices
- All figures (training curves, confusion matrix, correlation heatmap,
  adulteration boundary, fusion convergence) and the dataset coverage CSV are
  integrated with chapter-wise captions (table captions above, figure captions
  below)
- Open the file in Word and press Ctrl+A, F9 once to populate the Table of
  Contents and the Lists of Tables/Figures from the embedded caption styles.
