# Multimodal Food Freshness and Safety Detection — Training Pipeline

**Thesis:** *Image and Sensor Data Integration with Mahalanobis Distance for
AI-Enabled Food Freshness and Safety Detection*

**Author:** Md. Hasibul Islam (ID 0242310005341469), Department of Software
Engineering, Daffodil International University.
**Supervisor:** Dr. Kamrul Islam Shahin, Associate Professor, Department of
Software Engineering, Daffodil International University.

**Purpose.** This notebook implements the complete experimental pipeline of the
thesis on public Kaggle data only: (i) a MobileNetV2-based visual branch trained
separately on 3–4 fresh/stale image datasets, (ii) a channel-aware MLP sensor
branch trained separately on 3–4 public sensor CSV datasets with per-dataset
column mapping, availability flags and cross-dataset imputation, and (iii) a
feature-level fusion stage that combines both embeddings with a
Mahalanobis-distance anomaly detector and exports TensorFlow Lite models and C
firmware arrays for the ESP32-S3 prototype.

**Running the notebook on Kaggle:**
1. `Add Input` → attach the 3–4 image datasets; the loader auto-detects the
   mounted input paths.
2. `Add Input` → attach the 3–4 sensor CSV datasets; per-dataset column mappings
   are defined in the configuration cell.
3. Settings → Accelerator → GPU T4, then `Run All`.
4. Models, plots, coverage reports and `.tflite` artifacts are written to
   `/kaggle/working/outputs/`.
