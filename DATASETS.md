# Perfect Dataset Combo — Tri-Layer Multimodal Food Freshness Model (100% Kaggle)

**Strategy: no primary/self-collected data.** The image branch trains across
**3–4 public Kaggle image datasets**; the sensor branch trains across **3–4 public
Kaggle sensor-CSV datasets**. If any sensor dataset has missing channels, the
pipeline's coverage report names the exact extra dataset to attach (gap-filling
table at the bottom). All links verified.

## Image Branch Combo — attach these 4 (Part A)

| # | Dataset | Group | Why |
|---|---|---|---|
| 1 | [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) | Fruits | Folder names auto-load with zero code change |
| 2 | [Tilapia Fresh and Non Fresh (6,900 imgs)](https://www.kaggle.com/datasets/haripriyasanga/tilapia-fish-fresh-and-non-fresh-species) | Fish | South-Asian species, day-wise decay, gill/eye close-ups |
| 3 | [Meat Freshness Image Dataset (2,266 imgs)](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset) | Meat | Fresh / Half-Fresh / Spoiled labels |
| 4 | [Food Freshness Dataset (70k imgs)](https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset) | Ablation | Volume run: accuracy-vs-dataset-size curve for the defense |

Optional swap: [Meat Quality Assessment (ASYU 2019)](https://www.kaggle.com/datasets/crowww/meat-quality-assessment-based-on-deep-learning)
— published baseline, citable.

## Sensor Branch Combo — attach these 4 CSV datasets (Part B)

| # | Dataset | Channels it covers | Notes |
|---|---|---|---|
| 1 | [Fruit Freshness Electronic Nose Dataset](https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data) | VOC (MQ2-class), CH2O-family gas (MQ135), day/time index | Real MOS e-nose telemetry; 59 per-fruit decay CSVs; CC BY 4.0; DOI: 10.5281/zenodo.17285312 |
| 2 | [Chicken Meat Freshness Dataset](https://www.kaggle.com/datasets/ziya07/chicken-meat-freshness-dataset) | 10 virtual VOC channels (+ mean/variance), fresh vs frozen-thawed shelf-life labels (1,000 rows) | Meat-side VOC coverage — matches a ZE08/BME680-style gas curve |
| 3 | [Dataset of Formalin Tofu Aroma](https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin) | Gas ppm array (MQ2/MQ4), product temperature, **formalin-treatment label** (45,000 rows, CC0) | Closest public proxy for ZE08-CH2O: real chemical-adulteration labels on gas-sensor food data |
| 4 | [IoT Agriculture 2024](https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024) | Ambient temperature + RH time series | Fills the BME680 environmental channels; strengthens the drift-compensation story |

**Chemically-masked class (label 2):** added by the notebook's controlled
simulated adulteration augmentation (`MASK_AUGMENT_FRACTION`) — no public dataset
labels formalin *treatment* together with freshness classes. The Formalin Tofu
dataset supplies the closest real-world evidence of the chemical-off-gassing
signature; state both facts as *simulated augmentation over public sensor
telemetry* in the thesis.

**Sensor-hardware note (for the defense):** no public dataset ships data from our
exact sensors (ZE08-CH2O electrochemical, BME680, MLX90614 IR). The combo covers
every *channel* through nearest-family hardware — MQ MOS arrays (BME680 gas
resistance follows the same MOS principle), food temperature probes, and humidity
logs. One extra reference dataset,
[BME688 Odours](https://www.kaggle.com/datasets/laurabasa/bme688-odours), contains
**real Bosch BME68x gas-resistance CSVs captured on an ESP32 + Adafruit stack**
(the same sensor family and MCU as our prototype) — cite it to validate the
`voc_log_ohm` channel. Its matrix-format CSVs (heater-step × cycle rows) need a
small reshape before use, so it is a reference set, not a `SENSOR_CSVS` entry.
The MLX90614 non-contact core-temperature channel has no public equivalent —
the pipeline's availability flags + imputation handle it, and the thesis frames
it as *hardware-validated at deployment*.

## Gap-filling table — missing channel → extra dataset to attach

The notebook prints this automatically whenever the attached CSVs leave channels
uncovered (see the notebook's missing-channel notice + `sensor_dataset_coverage.csv`):

| Missing channel | Fix |
|---|---|
| `ch2o_ppm` | attach the [Formalin Tofu Aroma dataset](https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin) or any MQ135 gas-sensor CSV |
| `voc_log_ohm` | attach the [Fruit E-Nose MQ dataset](https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data); [BME688 Odours](https://www.kaggle.com/datasets/laurabasa/bme688-odours) as the real-Bosch reference |
| `core_temp_c` | attach the Formalin Tofu `Temperature(C)` column or any food-storage temperature CSV |
| `ambient_temp_c`, `humidity_pct` | attach any IoT food-storage / cold-chain temp-RH CSV (e.g. [IoT Agriculture 2024](https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024)) |
| `hours_elapsed` | usually derivable from timestamps or per-file day indices — no dataset needed |
| `ch2o_voc_ratio`, `temp_rise_rate` | derived automatically from their parent channels — never need a dataset |

## Recommended data volumes for a standard training run

| Branch | Minimum (works) | Standard (recommended) | Ideal (best results) |
|---|---|---|---|
| Image (Part A) | 10,000 images | **24,000 — the 3-dataset combo above** | 95,000 — also attach the 70k set |
| Sensor (Part B) | 5,000 rows | **50,000 rows — tofu 45k + e-nose + chicken + IoT** | 100,000 rows (add any extra gas-sensor CSV) |
| Fusion pairs (Part C) | 100 pairs | **400 pairs (`PAIR_LIMIT = 400`)** | 1,000 pairs (raise `PAIR_LIMIT`) |

Notes for the thesis:
- Per class balance: keep fresh/stale roughly 50/50 (all four image datasets and the
  tofu label split are already near-balanced).
- With `IMG_AUGMENT = True`, each image acts as ~20 training views, so even the
  24,000-image standard set gives an effective ~500,000 training views — more than
  enough for MobileNetV2 (alpha=0.35) transfer learning.
- The sensor MLP has only 16 input features; 50k rows is already 10x the samples-per-feature
  rule of thumb — do not expect accuracy gains beyond ~100k rows.

## Kaggle assembly steps

1. Attach the 4 image datasets → fill `IMAGE_DATASETS` (unattached entries are skipped).
2. Attach the 4 sensor CSV datasets → fill each `SENSOR_CSVS` mapping after
   checking the actual CSV headers on Kaggle.
3. Run with 1 image + 1 sensor dataset first to validate, then scale to the full
   combo — the coverage report updates itself.

## Thesis citation line (Data chapter)

"Training data are drawn exclusively from public repositories: four Kaggle
fresh/stale image corpora covering fruits, fish, and meat, and four Kaggle
sensor-telemetry CSVs (MQ e-nose arrays, temperature and humidity logs, and a
formalin-treated tofu chemical dataset; DOI: 10.5281/zenodo.17285312). The
chemically-masked class is obtained through controlled simulated adulteration
augmentation, since no public dataset provides labeled formalin-treated
freshness samples. Missing sensor channels across datasets are handled with
per-dataset availability flags and cross-dataset statistical imputation
(reported in the coverage table)."
