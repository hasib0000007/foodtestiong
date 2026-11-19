# Perfect Dataset Combo — Tri-Layer Multimodal Food Freshness Model (100% Kaggle)

**Strategy: no primary/self-collected data.** The image branch trains across
**3–4 public Kaggle image datasets**; the sensor branch trains across **3–4 public
Kaggle sensor-CSV datasets**. If any sensor dataset has missing channels, the
pipeline's coverage report names the exact extra dataset to attach (gap-filling
table at the bottom). All links verified.

## Image Branch — attach these 4 (Part A)

| # | Dataset | Group | Why |
|---|---|---|---|
| 1 | [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) | Fruits | Folder names auto-load with zero code change |
| 2 | [Tilapia Fresh and Non Fresh (6,900 imgs)](https://www.kaggle.com/datasets/haripriyasanga/tilapia-fish-fresh-and-non-fresh-species) | Fish | South-Asian species, day-wise decay, gill/eye close-ups |
| 3 | [Meat Freshness Image Dataset (2,266 imgs)](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset) | Meat | Fresh / Half-Fresh / Spoiled labels |
| 4 | [Food Freshness Dataset (70k imgs)](https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset) | Ablation | Volume run: accuracy-vs-dataset-size curve for the defense |

Optional swap: [Meat Quality Assessment (ASYU 2019)](https://www.kaggle.com/datasets/crowww/meat-quality-assessment-based-on-deep-learning)
— published baseline, citable.

## Sensor Branch — attach these 3–4 CSV datasets (Part B)

| # | Dataset | Channels it covers | Notes |
|---|---|---|---|
| 1 | [Fruit Freshness Electronic Nose Dataset](https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data) | VOC (MQ2-class), CH2O-family gas (MQ135), day/time index | Real MOS e-nose telemetry; 59 per-fruit decay CSVs; CC BY 4.0; DOI: 10.5281/zenodo.17285312 |
| 2 | [Chicken Meat Freshness Dataset](https://www.kaggle.com/datasets/ziya07/chicken-meat-freshness-dataset) | Core/ambient temperature, humidity, quality-stage labels | Meat-side sensor coverage |
| 3 | Any temperature/humidity IoT CSV (e.g. [IoT Agriculture 2024](https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024)) | Ambient temp + RH baselines | Strengthens drift-compensation claims |
| 4 | [e-Nose Beef Quality (Mendeley, 8 MOS + DHT11 on ESP32)](https://data.mendeley.com/datasets/n8mc3nspfn/1) | Full gas + temp + humidity + label (4 quality classes) | Hardware mirrors our ESP32 design; free download; use as validation set |

**Chemically-masked class (label 2):** added by the notebook's controlled
simulated adulteration augmentation (`MASK_AUGMENT_FRACTION`) — no public dataset
labels formalin treatment. State it as *simulated augmentation over public sensor
telemetry* in the thesis.

## Gap-filling table — missing channel → extra dataset to attach

The notebook prints this automatically whenever the attached CSVs leave channels
uncovered (see `[MISSING DATA]` output + `sensor_dataset_coverage.csv`):

| Missing channel | Fix |
|---|---|
| `ch2o_ppm` | attach an MQ135/air-quality gas-sensor CSV (MQ135 responds to aldehyde-family gases) |
| `voc_log_ohm` | attach the [Fruit E-Nose MQ dataset](https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data) |
| `core_temp_c` | attach [Chicken Meat Freshness](https://www.kaggle.com/datasets/ziya07/chicken-meat-freshness-dataset) or any food-storage temp CSV |
| `ambient_temp_c`, `humidity_pct` | attach any IoT food-storage / cold-chain temp-RH CSV |
| `hours_elapsed` | usually derivable from timestamps or per-file day indices — no dataset needed |
| `ch2o_voc_ratio`, `temp_rise_rate` | derived automatically from their parent channels — never need a dataset |

## Thesis citation line (Data chapter)

"Training data are drawn exclusively from public repositories: three to four
Kaggle fresh/stale image corpora covering fruits, fish, and meat, and three to
four Kaggle/Mendeley sensor-telemetry CSVs (MQ e-nose arrays, temperature and
humidity logs, DOI: 10.5281/zenodo.17285312). The chemically-masked class is
obtained through controlled simulated adulteration augmentation, since no public
dataset provides labeled formalin-treated samples. Missing sensor channels across
datasets are handled with per-dataset availability flags and cross-dataset
statistical imputation (reported in the coverage table)."
