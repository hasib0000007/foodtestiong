# Build the thesis-report skeleton DOCX (proposal structure + results tables)
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

PLACEHOLDER = RGBColor(0x99, 0x00, 0x00)

doc = Document()

def h(text, level):
    return doc.add_heading(text, level=level)

def p(text, italic=False, placeholder=False, align=None):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.italic = italic
    if placeholder:
        run.font.color.rgb = PLACEHOLDER
    if align:
        para.alignment = align
    return para

def ph(text):
    """Red placeholder instruction the student must replace."""
    p(text, italic=True, placeholder=True)

def table(headers, rows, caption=None):
    if caption:
        cap = doc.add_paragraph()
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, hdr in enumerate(headers):
        t.rows[0].cells[i].text = hdr
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            t.rows[ri].cells[ci].text = str(val)
    doc.add_paragraph()
    return t

# ---------- Title page ----------
for _ in range(3):
    doc.add_paragraph()
p("Image and Sensor Data Integration with Mahalanobis Distance for AI-Enabled Food Freshness and Safety Detection", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
p("A Thesis Report Submitted in Partial Fulfillment of the Requirements for the Degree of B.Sc. in Software Engineering", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
p("Presented By", align=WD_ALIGN_PARAGRAPH.CENTER)
p("Md. Hasibul Islam", align=WD_ALIGN_PARAGRAPH.CENTER)
p("ID: 0242310005341469", align=WD_ALIGN_PARAGRAPH.CENTER)
p("B.Sc. in SWE, Department of Software Engineering, Daffodil International University", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
p("Supervised By", align=WD_ALIGN_PARAGRAPH.CENTER)
p("Dr. Kamrul Islam Shahin", align=WD_ALIGN_PARAGRAPH.CENTER)
p("Associate Professor, Dept. of SWE, DIU", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
p("Course No: SE 431  |  Course Title: Final Year Thesis", align=WD_ALIGN_PARAGRAPH.CENTER)
ph("[Insert DIU logo above; adjust front-matter to the DIU thesis-report template margins/cover rules.]")
doc.add_page_break()

# ---------- Abstract ----------
h("Abstract", 1)
p("Food safety and the illegal addition of toxic chemical preservatives present critical public health threats in developing nations like Bangladesh. Traditional inspection relies on destructive laboratory assays or subjective human sensory evaluation, which are impractical for real-time, non-destructive testing at open-air point-of-purchase markets. Single-modality vision-based AI models are easily bypassed by superficial chemical masking or artificial color preservation.")
p("This research introduces a portable, non-destructive Tri-Layer Multimodal Sensor Fusion scanner operating at the edge. The system comprises an ESP32-S3 microcontroller with an OV2640 camera, a ZE08-CH2O formaldehyde sensor, a BME680 VOC/gas sensor, and an MLX90614 non-contact infrared thermometer. Image and sensor branches are trained separately: a quantized MobileNetV2 convolutional branch extracts visual features (Fv), while an MLP regression module processes normalized, filtered telemetry (Fs). A feature-level late-fusion engine computes a unified Freshness Index (0-100%), and a Mahalanobis Distance anomaly layer exposes illegal chemical preservation such as formalin masking. The trained hybrid network is optimized with INT8 quantization for TensorFlow Lite Micro deployment, and results stream over BLE to a companion mobile application.")
ph("[After the final Kaggle experiments, add 2-3 quantitative headline sentences here: e.g., image-branch accuracy, sensor-branch F1, masking-alert detection rate, and the INT8 model footprint. Keep abstract under 300 words.]")
doc.add_page_break()

# ---------- TOC ----------
h("Table of Contents", 1)
ph("[Insert an automatic Table of Contents here: References tab > Table of Contents in Word, after all headings are final. List of Figures and List of Tables should follow on separate pages.]")
doc.add_page_break()

# ---------- Chapter 1 ----------
h("Chapter 1: Introduction", 1)
h("1.1 Background", 2)
p("Rapid food spoilage and illegal chemical preservation severely impact public health, consumer trust, and agrarian economies. Non-destructive food evaluation using machine learning (ML) and Internet of Things (IoT) sensors has emerged as a promising alternative to destructive testing. However, deploying computationally expensive deep learning models onto resource-constrained, low-power microcontrollers (Edge AI / Tiny ML) remains a significant challenge. This research develops a portable, edge-powered multimodal scanner that simultaneously captures visual surface matrices, volatile organic chemical concentrations, and sub-surface thermal trends to evaluate food quality at the point of purchase.")
h("1.2 Problem Statement", 2)
for item in [
    "Detection deficit in chemical masking: formaldehyde (formalin) treatments preserve surface color and texture, creating false-positive freshness in vision-only systems.",
    "Invisibility of sub-surface and core decomposition: internal bacterial acceleration in thick-skinned produce and cold-stored fish is invisible to standard surface cameras.",
    "Destructive testing constraints: on-field chemical validation requires cutting, crushing, or solvent dilution, causing food waste and vendor conflict.",
    "High infrastructure cost: gold-standard instruments (GC-MS) are bulky and laboratory-confined, leaving consumers without point-of-purchase tools.",
    "Cloud dependency and power: cloud-reliant IoT devices fail in connectivity-opaque wet markets and drain batteries quickly.",
]:
    p(item)
h("1.3 Research Objectives", 2)
for obj in [
    "RO1 (Hardware): construct a low-cost, portable, non-destructive embedded scanner using the ESP32-S3 and integrated multi-sensor modules.",
    "RO2 (Sensing): design a multi-modal data fusion framework integrating visual, chemical (VOC / CH2O), and thermal modalities.",
    "RO3 (AI Model): develop an intelligent detection model using Mahalanobis Distance feature fusion to isolate chemical-masking anomalies.",
    "RO4 (Edge AI): implement TinyML acceleration via TensorFlow Lite Micro and INT8-quantized MobileNetV2 for offline inference.",
    "RO5 (Dataset): build and validate a synchronized multi-sensor dataset capturing decay dynamics across indigenous Bangladeshi foods.",
]:
    p(obj)
h("1.4 Research Contributions", 2)
for c in [
    "Low-cost, non-destructive handheld scanning hardware (3D-printable, <450 g).",
    "Novel tri-layer data fusion framework across visual, chemical, and thermal domains.",
    "Intelligent adulteration proxying via cross-domain statistical anomaly analysis.",
    "On-device TinyML acceleration using INT8 quantization and ESP-NN vector instructions.",
    "Localized multi-sensor dataset creation for regional Bangladeshi food items.",
]:
    p(c)
doc.add_page_break()

# ---------- Chapter 2 ----------
h("Chapter 2: Literature Review", 1)
h("2.1 Review of Related Works", 2)
table(
    ["Paper", "Method / Dataset", "Contribution", "Observed Limitation"],
    [
        ["Tan & Xu (2020) [1]", "E-Nose & E-Tongue arrays / food-quality samples", "Non-destructive gas sensing; multi-sensor volatile analysis", "Gas-only modality; no visual/thermal coverage"],
        ["Ru (2021) [2]", "TinyML on low-power devices / embedded benchmarks", "Ultra-low-power MCU inference", "Benchmark-only; no food-safety application"],
        ["Gu et al. (2021) [3]", "Computer vision + ML / food image corpus", "Non-destructive visual inspection", "Vulnerable to chemical masking; ignores internal decay"],
        ["Ooi & Wong (2021) [4]", "Electrochemical formaldehyde detection / fishery products", "Real-time CH2O quantification", "Single-gas; no fusion"],
        ["Santos et al. (2022) [5]", "Multi-sensor fusion on MCUs / sensor arrays", "On-chip integration, low power", "Not tailored to food-safety tasks; no anti-adulteration"],
        ["Zhao & Zhang (2022) [6]", "Post-training quantization survey / DNNs on MCUs", "Memory and latency reduction", "General survey; no multimodal pipeline"],
        ["Wanniarachchi & Liyanage (2023) [7]", "Hybrid DL + multi-sensor / edge food freshness", "Improved freshness scoring", "No anti-adulteration proxying"],
        ["Shukla et al. (2023) [8]", "Non-contact IR thermography / thermal images", "Sub-surface decay detection", "Fails on iced/moist surfaces; thermal-only"],
        ["Shuvo et al. (2024) [9]", "Smart gas scanner / wet-market samples", "Chemical masking detection, low cost", "No synchronized visual+thermal fusion"],
        ["Banik & Sahu (2024) [10]", "Feature-level late fusion / vision + gas arrays", "Cross-modal classification accuracy", "High complexity; not TinyML-optimized"],
    ],
)
h("2.2 Research Gaps Identified", 2)
for g in [
    "Lack of public multi-modal datasets combining vision, formaldehyde, and thermal streams for South Asian foods.",
    "Chemical-masking vulnerability of prior single-modality vision models.",
    "Absence of sensor-fusion frameworks optimized for sub-250 KB TFLite tensor arenas.",
    "Under-explored cross-sensitivity and humidity-drift compensation in field environments.",
]:
    p(g)
ph("[Optionally add 2-3 paragraphs summarizing the state of the art and positioning this work relative to the reviewed papers.]")
doc.add_page_break()

# ---------- Chapter 3 ----------
h("Chapter 3: Proposed Methodology", 1)
h("3.1 System Overview", 2)
p("The system operates through a sequential multi-stage sensor fusion paradigm across five layers:")
for item in [
    "Layer 1 - Synchronized Ingestion: a 3D-printed isolation hood establishes a micro-climate chamber; camera, gas sensors, and IR thermometer sample simultaneously at a common timestamp (t0).",
    "Layer 2 - Cross-Domain Feature Extraction: visual frames are compressed into compact spatial vectors (Fv) by an INT8-quantized CNN; telemetry undergoes min-max scaling and moving-average smoothing.",
    "Layer 3 - State-Space Noise Filtering and Drift Compensation: an adaptive Kalman-style filter with baseline tracking isolates true decomposition gases; live RH/temperature corrects the CH2O calibration curve.",
    "Layer 4 - Late-Fusion Deep Learning Inference: frozen branch features (Fv, Fs) are concatenated; a compact fusion network regresses a unified Freshness Index (0-100%).",
    "Layer 5 - Anomaly Detection (Anti-Adulteration): a Mahalanobis Distance matrix on the fused Fresh-class feature distribution flags contradictions between visual freshness and toxic gas profiles.",
]:
    p(item)
ph("[Insert the system flowchart figure here (the Slide-8 flowchart exported at 300+ DPI). Add a hardware block diagram (OV2640, ZE08-CH2O, BME680, MLX90614, ESP32-S3, TP4056, BLE) as a second figure.]")
h("3.2 Model Architecture", 2)
table(
    ["Branch", "Input", "Core layers", "Feature output", "Output head"],
    [
        ["A - Image (MobileNetV2)", "224x224x3, scaled [0,1]", "MobileNetV2 backbone (alpha 0.35, ImageNet transfer)", "Fv: 128-D dense embedding", "Sigmoid fresh/stale probability"],
        ["B - Sensor (MLP)", "8 telemetry features, min-max normalized", "Dense 64 -> Dense 32 -> Dense 16 with dropout", "Fs: 16-D embedding", "Softmax: Fresh / Stale / Chemically-Masked"],
        ["C - Fusion (Decision engine)", "Z = [Fv | Fs] (144-D)", "Dense 64 -> Dense 32", "-", "Sigmoid Freshness Index (0-100%) + Mahalanobis distance thresholding"],
    ],
)
p("The image and sensor branches are trained completely separately. Both feature extractors are then frozen, and only the compact fusion head is retrained on the concatenated joint matrix Z, matching the Slide-8 flow of the presentation (MobileNetV2 and MLP paths converging at the Mahalanobis Distance and Decision Fusion Engine stages).")
h("3.3 Mahalanobis Anomaly Formulation", 2)
p("For fused feature vectors z sampled from the Fresh class, the mean vector \u03bc and regularized covariance \u03a3 = Cov(Z) + \u03bbI are estimated. The distance d(z) = sqrt((z-\u03bc)\u1d40 \u03a3\u207b\u00b9 (z-\u03bc)) follows a chi-square distribution; the alarm threshold is set at the 99% quantile with df = feature dimension (144). A CH2O reading above the 2 ppm ambient background combined with d(z) exceeding the threshold triggers the Chemical Masking alert.")
ph("[Add the mathematical derivation from the proposal (Kalman filtering equations and the Mahalanobis definition) in proper equation formatting here.]")
doc.add_page_break()

# ---------- Chapter 4 ----------
h("Chapter 4: Data", 1)
h("4.1 Data Sources", 2)
p("Two modality-specific datasets are used and processed separately, mirroring the decoupled training design:")
for item in [
    "Image data: public Kaggle freshness datasets (e.g., Fresh and Stale Images of Fruits and Vegetables; Food Freshness Dataset; Fruits fresh and rotten for classification). Class folders containing 'fresh', 'stale', or 'rotten' are mapped to binary labels.",
    "Sensor data: an aggregated CSV of CH2O (ppm), BME680 VOC (log-ohm), core/ambient temperature (\u00b0C), humidity (%), the CH2O/VOC ratio, core temperature rise rate, and elapsed hours, labeled 0 = fresh, 1 = stale (natural decay), 2 = chemically masked.",
    "When no physical sensor CSV is available yet, a physics-based synthetic generator replicating the proposal's 96-hour controlled-decay protocol (26-36 \u00b0C, 60-88% RH, formalin-dilution masking of visual decay) is used for pipeline development and validation.",
]:
    p(item)
h("4.2 Data Preprocessing and Handling", 2)
for item in [
    "Images resized to a uniform 224x224 and normalized from [0,255] to [0,1]; an internal rescaling layer maps them to the ImageNet range.",
    "Sensor streams: min-max scaling followed by a windowed moving average (Kalman-style pre-filtering) to suppress market-environment noise.",
    "15% stratified validation split for both branches; class-coherent pairing of image and sensor samples for the fusion stage.",
    "INT8 post-training quantization reduces the trained network footprint by roughly 75% while preserving over 95% of FP32 accuracy.",
]:
    p(item)
ph("[Record the final dataset attachment names/sizes on Kaggle and the exact train/validation counts here once runs are complete.]")
h("4.3 Data Visualization", 2)
for item in [
    "Adulteration boundary scatterplot: CH2O-vs-VOC separation between natural decay and chemically preserved samples (outputs/adulteration_boundary.png).",
    "Sensor cross-correlation heatmap: dependency structure across telemetry channels (outputs/sensor_correlation.png).",
    "Training curves and confusion matrices for both branches (outputs/*_training.png, outputs/sensor_confusion_matrix.png).",
]:
    p(item)
ph("[Insert the generated PNGs from the Kaggle outputs as numbered figures and reference them in the text.]")
doc.add_page_break()

# ---------- Chapter 5 ----------
h("Chapter 5: Implementation", 1)
h("5.1 Training Environment (Kaggle)", 2)
p("Both branches and the fusion engine are trained on Kaggle free GPU notebooks (GPU T4). The complete pipeline is reproducible from notebooks/food_freshness_kaggle_training.ipynb: attach the image dataset, set IMAGE_DATA_DIR, optionally point SENSOR_CSV to the collected CSV, and run all cells. Artifacts (models, plots, .tflite binaries) are written to /kaggle/working/outputs/.")
table(
    ["Hyper-parameter", "Part A (image)", "Part B (sensor)", "Part C (fusion)"],
    [
        ["Backbone / structure", "MobileNetV2 \u03b1=0.35, GAP + Dense 128", "Dense 64-32-16 MLP", "Dense 64-32 regressor"],
        ["Optimizer / LR", "Adam, 1e-3 (head); 1e-5 (fine-tune)", "Adam, 1e-3", "Adam, 1e-3"],
        ["Epochs", "[XX] + [XX] fine-tune", "[XX]", "[XX]"],
        ["Batch size", "32", "32 (table rows)", "32"],
        ["Loss", "Binary cross-entropy", "Sparse categorical cross-entropy", "Mean absolute error"],
    ],
)
h("5.2 Hardware and Firmware Pipeline", 2)
for item in [
    "ESP32-S3 dual-core MCU with ESP-NN vector acceleration; the INT8 flatbuffer is compiled into a C array (image_branch.cc) flashed with TensorFlow Lite Micro.",
    "Sensor scaling constants and the Mahalanobis threshold are exported to edge_config.json for firmware use.",
    "Custom PCB (KiCad) with star grounding, analog isolation, and decoupling caps; 3D-printed enclosure with micro-climate sampling hood; TP4056-regulated 18650/2000 mAh LiPo power with light-sleep duty cycling.",
    "BLE (low-energy) tranceiver streams scan results to a Flutter mobile application.",
]:
    p(item)
ph("[Insert ESP32-S3 tensor-arena-size measurement and measured per-scan latency here after firmware benchmarking (target: arena < 250 KB, scan < 1.5 s).]")
doc.add_page_break()

# ---------- Chapter 6 ----------
h("Chapter 6: Results and Discussion", 1)
ph("All bracketed [XX] cells below must be filled from the Kaggle outputs after the final training run; keep the table numbering and add discussion paragraphs under each table.")
h("6.1 Image Branch (Part A)", 2)
table(
    ["Metric", "Train", "Validation"],
    [
        ["Accuracy (%)", "[XX.XX]", "[XX.XX]"],
        ["Loss", "[XX.XX]", "[XX.XX]"],
        ["Precision (%)", "[XX.XX]", "[XX.XX]"],
        ["Recall (%)", "[XX.XX]", "[XX.XX]"],
        ["F1-score (%)", "[XX.XX]", "[XX.XX]"],
    ],
    "Table 6.1 - MobileNetV2 image-branch performance (fresh vs stale).",
)
p("[Figure 6.1: image_branch_training.png - training/validation curves.]")
h("6.2 Sensor Branch (Part B)", 2)
table(
    ["Class", "Precision (%)", "Recall (%)", "F1-score (%)", "Support"],
    [
        ["Fresh", "[XX.XX]", "[XX.XX]", "[XX.XX]", "[XX]"],
        ["Stale (natural)", "[XX.XX]", "[XX.XX]", "[XX.XX]", "[XX]"],
        ["Chemically masked", "[XX.XX]", "[XX.XX]", "[XX.XX]", "[XX]"],
        ["Macro average", "[XX.XX]", "[XX.XX]", "[XX.XX]", "[XX]"],
    ],
    "Table 6.2 - Sensor-branch three-state classification report.",
)
p("[Figure 6.2: sensor_confusion_matrix.png; Figure 6.3: sensor_correlation.png; Figure 6.4: adulteration_boundary.png.]")
h("6.3 Fusion Engine and Mahalanobis Anomaly Detection (Part C)", 2)
table(
    ["Evaluation item", "Result"],
    [
        ["Fusion regression MAE (freshness index, 0-1 scale)", "[XX.XXXX]"],
        ["Freshness-index error (\u00b1 percentage points)", "[XX.X]"],
        ["Mahalanobis threshold (99% chi-square, df=144)", "[XX.XX]"],
        ["Chemical-masking detection rate (TPR)", "[XX.XX%]"],
        ["False-alarm rate on genuinely fresh samples (FPR)", "[XX.XX%]"],
    ],
    "Table 6.3 - Decision Fusion Engine and anti-adulteration performance.",
)
p("[Describe 2-3 representative test cases here: (i) genuinely fresh sample scored high with low Mahalanobis distance; (ii) naturally stale sample; (iii) formalin-treated sample that appears fresh visually but triggers the CHEMICAL MASKING ALERT.]")
p("[Figure 6.5: fusion_training.png - fusion convergence curve.]")
h("6.4 Edge Deployment Footprint", 2)
table(
    ["Artifact", "FP32 size (KB)", "INT8 size (KB)", "Reduction (%)", "Target"],
    [
        ["Image branch (MobileNetV2 \u03b1=0.35)", "[XXXX]", "[XXX]", "[XX.X]", "TFLite Micro flatbuffer"],
        ["Fusion engine", "[XX.X]", "[XX.X]", "[XX.X]", "< 250 KB tensor arena"],
        ["Per-scan end-to-end latency (s)", "-", "measured on ESP32-S3: [X.XX]", "-", "< 1.5 s"],
    ],
    "Table 6.4 - Quantization and edge deployment metrics.",
)
h("6.5 Discussion", 2)
p("[Discuss: (i) why separate branch training is preferable for TinyML (modularity, memory budgeting, branch reuse); (ii) how the Mahalanobis layer defeats chemical masking that bypasses vision-only baselines; (iii) limitations - calibration on specific local species, environmental drift, and moisture-related IR distortion; (iv) comparison against the reviewed literature.]")
doc.add_page_break()

# ---------- Chapter 7 ----------
h("Chapter 7: Conclusion and Future Perspectives", 1)
h("7.1 Conclusion", 2)
p("This research develops a robust, unspoofable, and cost-effective handheld food-defense device that brings laboratory-grade food inspection to everyday consumers. By deploying a multi-modal, decoupled TinyML pipeline onto a low-power ESP32-S3, the system operates entirely at the edge, bypassing connectivity barriers of local wet markets. Its principal contribution is exposing artificial chemical preservation by simultaneously cross-referencing visual surface degradation, VOC profiles, and internal thermal anomalies through Kalman-style filtering and a Mahalanobis anomaly-detection pipeline.")
ph("[Rewrite the conclusion once final numbers exist, mirroring the abstract's quantitative claims.]")
h("7.2 Future Perspectives", 2)
for item in [
    "Multi-spectral (NIR) sensing for pesticide and fertilizer residue screening.",
    "Privacy-preserving federated learning for on-device model improvement.",
    "Robotic smart-bins on enterprise edge boards for real-time inventory decay management.",
    "Decentralized, geo-tagged public-health food-safety heatmaps aggregated via BLE.",
]:
    p(item)
doc.add_page_break()

# ---------- References ----------
h("References", 1)
refs = [
    "J. Tan and J. Xu, \"Applications of Electronic Nose (E-Nose) and Electronic Tongue (E-Tongue) in Food Quality and Safety Assessment,\" Sensors, vol. 20, no. 18, p. 5077, 2020.",
    "L. Ru, \"TinyML: Machine Learning on Low-Power Devices,\" IEEE Cloud Computing, vol. 8, no. 4, pp. 12-19, 2021.",
    "X. Gu, J. Sun, and Y. Wang, \"Computer Vision and Machine Learning for Non-Destructive Food Quality Assessment,\" Food Engineering Reviews, vol. 13, no. 2, pp. 311-325, 2021.",
    "S. K. Ooi and T. J. Wong, \"A Low-Cost IoT-Based System for Real-Time Formaldehyde Detection in Fishery Products Using Electrochemical Sensors,\" Measurement, vol. 182, Art. no. 109675, 2021.",
    "M. S. Santos, J. P. Silva, and R. Castro, \"Multi-Sensor Data Fusion inside Low-Power Microcontrollers for Edge-AI Applications,\" IEEE Sensors Letters, vol. 6, no. 3, pp. 1-4, 2022.",
    "Y. Zhao and J. Zhang, \"Post-Training Quantization for Deep Neural Networks on Microcontrollers: A Survey,\" ACM Computing Surveys, vol. 55, no. 6, pp. 112-135, 2022.",
    "T. R. Wanniarachchi and S. R. Liyanage, \"Real-Time Food Freshness Detection Using a Hybrid Deep Learning and Multi-Sensor System on Edge Devices,\" IEEE Access, vol. 11, pp. 45112-45124, 2023.",
    "A. K. Shukla, M. Chandra, and P. K. Singh, \"Non-Contact Infrared Thermography for Sub-surface Food Decay Detection: A Review,\" Infrared Physics & Technology, vol. 128, Art. no. 104495, 2023.",
    "S. Shuvo, M. N. Islam, and T. Hasan, \"Low-Cost Smart Scanner for Formalin and Chemical Masking Detection in South Asian Wet Markets,\" International Journal of Food Contamination, vol. 11, no. 1, pp. 5-14, 2024.",
    "M. Banik and R. K. Sahu, \"Feature-Level Late Fusion of Multimodal Visual and Gas Sensor Arrays for Food Degradation Classification,\" Analytica Chimica Acta, vol. 1290, pp. 135-148, 2024.",
]
for i, r in enumerate(refs, 1):
    p(f"[{i}] {r}")

doc.save("Thesis_Report_Skeleton.docx")
print("saved")
