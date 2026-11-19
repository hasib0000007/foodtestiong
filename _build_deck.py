# Build the final-defense presentation (16:9, professional academic theme)
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---- theme ----
NAVY   = RGBColor(0x0B, 0x2A, 0x33)   # dark base
GREEN  = RGBColor(0x0E, 0x6B, 0x4A)   # primary food-safety green
GREEN2 = RGBColor(0x14, 0x8A, 0x5F)
AMBER  = RGBColor(0xE8, 0x9C, 0x2E)   # accent
RED    = RGBColor(0xC0, 0x39, 0x2B)
LGREY  = RGBColor(0xF2, 0xF5, 0xF3)
MGREY  = RGBColor(0x9A, 0xAB, 0xA6)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
DARKT  = RGBColor(0x1B, 0x2B, 0x27)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5

def slide():
    return prs.slides.add_slide(BLANK)

def rect(s, x, y, w, h, fill, line=None, shadow=False, round_=False):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE,
                             Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line: shp.line.color.rgb = line; shp.line.width = Pt(1)
    else: shp.line.fill.background()
    shp.shadow.inherit = False
    return shp

def txt(s, x, y, w, h, lines, size=16, color=DARKT, bold=False, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    first = True
    for item in lines:
        level = 0; text = item; fsize = size; fbold = bold; fcolor = color
        if isinstance(item, tuple):
            text, kw = item
            level = kw.get("level", 0); fsize = kw.get("size", size)
            fbold = kw.get("bold", bold); fcolor = kw.get("color", color)
        pgh = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        pgh.alignment = align
        if level: pgh.level = level
        run = pgh.add_run(); run.text = text
        run.font.size = Pt(fsize); run.font.bold = fbold
        run.font.color.rgb = fcolor; run.font.name = font
    return tb

def header(s, title, num=None, subtitle=None):
    rect(s, 0, 0, SW, 1.02, NAVY)
    rect(s, 0, 1.02, SW, 0.06, AMBER)
    txt(s, 0.55, 0.16, 10.5, 0.7, [(title, dict(size=27, bold=True, color=WHITE))],
        anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        txt(s, 0.58, 1.18, 12.2, 0.4, [(subtitle, dict(size=13, italic=True, color=MGREY))])
    if num is not None:
        rect(s, 12.35, 0.22, 0.62, 0.62, GREEN, round_=True)
        txt(s, 12.35, 0.22, 0.62, 0.62, [(str(num), dict(size=16, bold=True, color=WHITE))],
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

def footer(s, n):
    txt(s, 0.55, 7.12, 8, 0.3, [("Final Year Thesis Defense  |  SE 431  |  Daffodil International University",
                                 dict(size=10, color=MGREY))])
    txt(s, 12.35, 7.12, 0.7, 0.3, [(str(n), dict(size=11, color=MGREY))])

def bullets(s, x, y, w, h, items, size=16, gap=6):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for it in items:
        if isinstance(it, tuple):
            text, lvl = it[0], it[1]
        else:
            text, lvl = it, 0
        pgh = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        pgh.space_after = Pt(gap); pgh.level = lvl
        run = pgh.add_run(); run.text = ("• " if lvl == 0 else "– ") + text
        run.font.size = Pt(size); run.font.color.rgb = DARKT; run.font.name = "Calibri"
    return tb

def card(s, x, y, w, h, title, body, tc=GREEN, tsize=14, bsize=11.5):
    rect(s, x, y, w, h, LGREY, round_=True)
    rect(s, x, y, 0.09, h, tc)
    txt(s, x + 0.22, y + 0.08, w - 0.34, 0.42, [(title, dict(size=tsize, bold=True, color=tc))])
    tb = s.shapes.add_textbox(Inches(x + 0.22), Inches(y + 0.5), Inches(w - 0.4), Inches(h - 0.6))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for b in body:
        pgh = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        pgh.space_after = Pt(3)
        run = pgh.add_run(); run.text = b
        run.font.size = Pt(bsize); run.font.color.rgb = DARKT

def box(s, x, y, w, h, text, fill, tcolor=WHITE, size=11, bold=True, round_=True):
    shp = rect(s, x, y, w, h, fill, round_=round_)
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Pt(4); tf.margin_right = Pt(4); tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    pgh = tf.paragraphs[0]; pgh.alignment = PP_ALIGN.CENTER
    run = pgh.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = tcolor
    return shp

def arrow(s, x1, y1, x2, y2, color=NAVY, wpt=2.2):
    ln = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.begin_directed = True
    ln.line.color.rgb = color; ln.line.width = Pt(wpt)
    return ln

def deftable(s, x, y, w, headers, rows, colw=None, fsize=11, hsize=11.5, hfill=GREEN):
    t = s.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(0.4 + len(rows) * 0.36))
    tbl = t.table
    for ci, hdr in enumerate(headers):
        c = tbl.cell(0, ci); c.text = ""
        set_cell(c, hdr, bold=True, size=hsize, color=WHITE, fill=hfill)
    for ri, row in enumerate(rows, 1):
        for ci, val in enumerate(row):
            c = tbl.cell(ri, ci); c.text = ""
            set_cell(c, val, size=fsize, fill=WHITE if ri % 2 else LGREY)
    if colw:
        for ci, cw in enumerate(colw):
            tbl.columns[ci].width = Inches(cw)
    return tbl

def set_cell(c, text, size=11, bold=False, color=DARKT, fill=None):
    c.fill.solid(); c.fill.fore_color.rgb = fill if fill else WHITE
    c.margin_left = Pt(5); c.margin_right = Pt(5); c.margin_top = Pt(2); c.margin_bottom = Pt(2)
    tf = c.text_frame; tf.word_wrap = True
    pgh = tf.paragraphs[0]
    run = pgh.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color; run.font.name = "Calibri"

n = 0

# ================= SLIDE 1 — Title =================
n += 1; s = slide()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, 0, 0.35, SH, GREEN)
rect(s, 12.98, 0, 0.35, SH, AMBER)
txt(s, 1.1, 0.75, 11.1, 0.4, [("FINAL YEAR THESIS DEFENSE  |  SE 431", dict(size=15, bold=True, color=AMBER))])
txt(s, 1.1, 1.35, 11.1, 1.9, [
    ("Image and Sensor Data Integration with Mahalanobis Distance for AI-Enabled Food Freshness and Safety Detection",
     dict(size=32, bold=True, color=WHITE))])
rect(s, 1.1, 3.5, 3.2, 0.045, AMBER)
txt(s, 1.1, 4.15, 5.2, 2.2, [
    ("Presented By", dict(size=13, bold=True, color=GREEN2)),
    ("Md. Hasibul Islam", dict(size=19, bold=True, color=WHITE)),
    ("ID: 0242310005341469", dict(size=14, color=LGREY)),
    ("B.Sc. in Software Engineering (SWE), DIU", dict(size=13, color=LGREY))])
txt(s, 7.0, 4.15, 5.2, 2.2, [
    ("Supervised By", dict(size=13, bold=True, color=GREEN2)),
    ("Dr. Kamrul Islam Shahin", dict(size=19, bold=True, color=WHITE)),
    ("Associate Professor", dict(size=14, color=LGREY)),
    ("Dept. of Software Engineering, DIU", dict(size=13, color=LGREY))])
txt(s, 1.1, 6.75, 11, 0.4, [("Department of Software Engineering  |  Daffodil International University  |  [Defense Date]",
                              dict(size=12, color=MGREY))])

# ================= SLIDE 2 — Outline =================
n += 1; s = slide(); header(s, "Presentation Outline", n)
items = ["Background & Motivation", "Problem Statement", "Research Objectives & Contributions",
         "Literature Review & Research Gaps", "Proposed Methodology — Tri-Layer Multimodal Fusion",
         "Decoupled Branch Training (Image  |  Sensor)", "Decision Fusion Engine & Mahalanobis Anomaly Layer",
         "Implementation — Kaggle Training Pipeline", "Results & Analysis", "Edge Deployment (TinyML / INT8)",
         "Conclusion & Future Work", "References  |  Q&A"]
for i, it in enumerate(items):
    col, row = i % 2, i // 2
    x = 0.6 + col * 6.35; y = 1.55 + row * 0.92
    bo = rect(s, x, y, 6.0, 0.74, LGREY, round_=True)
    rect(s, x + 0.14, y + 0.17, 0.4, 0.4, GREEN if i % 2 == 0 else GREEN2, round_=True)
    txt(s, x + 0.14, y + 0.17, 0.4, 0.4, [(str(i + 1), dict(size=13, bold=True, color=WHITE))],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, x + 0.68, y + 0.09, 5.2, 0.56, [(it, dict(size=14.5, bold=False, color=DARKT))],
        anchor=MSO_ANCHOR.MIDDLE)
footer(s, n)

# ================= SLIDE 3 — Background =================
n += 1; s = slide(); header(s, "Background & Motivation", n)
txt(s, 0.55, 1.35, 12.2, 0.75, [("Food spoilage and illegal chemical preservation (formalin) pose severe public-health threats in developing nations like Bangladesh — consumers have no affordable tool to verify food safety at the market.", dict(size=15.5, color=DARKT))])
card(s, 0.55, 2.35, 6.0, 2.05, "The Reality in Local Wet Markets", [
    "Decomposing fish/fruit treated with formaldehyde to mask decay",
    "Dense, open-air markets — no cloud connectivity", "Vendors resist destructive testing (cutting/crushing)",
])
card(s, 6.8, 2.35, 6.0, 2.05, "Existing Solutions Fall Short", [
    "GC-MS and lab assays: accurate but costly, slow, non-portable",
    "Chemical test strips: destructive, single-analyte, cause waste",
    "Vision-only AI apps: fooled by chemical masking",
])
card(s, 0.55, 4.65, 12.25, 1.9, "Our Proposal", [
    "A pocket-sized, battery-powered multimodal scanner that sees the surface (camera), smells the volatiles (gas sensors), and senses hidden decay (IR core temperature) — then fuses all three on-device to expose adulteration instantly, 100% offline.",
    "Shifts verification power from state laboratories to the hands of the consumer.",
], tc=AMBER, tsize=15, bsize=13.5)
footer(s, n)

# ================= SLIDE 4 — Problem =================
n += 1; s = slide(); header(s, "Problem Statement", n, "Five critical barriers in traditional food-testing systems")
rows = [
    ("1. Chemical Surface Masking", "Formalin sprays preserve color/texture — vision-only AI gives false-positive freshness on decaying food."),
    ("2. Invisible Internal Decay", "Core rotting inside thick-skinned mangoes / cold-stored fish happens under a perfectly fresh exterior; no odor, no surface sign."),
    ("3. Destructive Testing", "On-field assays require cutting or solvent dilution — food waste, vendor conflict in live markets."),
    ("4. High Cost, Zero Portability", "Gold-standard GC-MS is laboratory-bound; consumers have no point-of-purchase tool."),
    ("5. Cloud & Power Dependency", "Cloud IoT fails in connectivity-opaque markets; power-hungry sensors drain batteries in hours."),
]
from pptx.enum.shapes import MSO_SHAPE as MS
for i, (t, b) in enumerate(rows):
    col, row = i % 2, i // 2
    if i == 4:
        x, y, w = 0.55, 1.6 + 2 * 1.28, 12.3
    else:
        x, y, w = 0.55 + col * 6.35, 1.6 + row * 1.28, 6.0
    rect(s, x, y, w, 1.14, LGREY, round_=True)
    rect(s, x, y, 0.09, 1.14, RED)
    txt(s, x + 0.22, y + 0.05, w - 0.4, 0.4, [(t, dict(size=14, bold=True, color=RED))])
    txt(s, x + 0.22, y + 0.45, w - 0.45, 0.66, [(b, dict(size=11, color=DARKT))])
footer(s, n)

# ================= SLIDE 5 — Objectives =================
n += 1; s = slide(); header(s, "Research Objectives & Contributions", n)
objs = [
    ("RO1 — Hardware", "Low-cost, portable, non-destructive handheld scanner on ESP32-S3 with integrated multi-sensor modules."),
    ("RO2 — Sensing", "Multi-modal fusion framework: visual + chemical (VOC / CH2O) + thermal modalities for full food profiling."),
    ("RO3 — AI Model", "Mahalanobis Distance feature fusion to measure multi-sensor variance and isolate chemical-masking anomalies."),
    ("RO4 — Edge AI", "TinyML acceleration via TensorFlow Lite Micro + INT8-quantized MobileNetV2; offline sub-second inference."),
    ("RO5 — Dataset", "Synchronized multi-sensor dataset of decay dynamics across indigenous Bangladeshi fish, meat & fruits."),
]
for i, (t, b) in enumerate(objs):
    x = 0.55 + (i % 3) * 4.22; y = 1.55 + (i // 3) * 1.62
    card(s, x, y, 4.0, 1.5, t, [b], tc=GREEN, tsize=13.5, bsize=10.5)
card(s, 0.55, 4.95, 12.25, 1.6, "Key Contributions", [
    "•  Tri-layer data fusion framework (visual Fv + chemical/telemetry Fs) trained as decoupled branches with feature-level late fusion",
    "•  Unspoofable adulteration proxying: statistical covariance analysis between appearance and gas profile",
    "•  Sub-250 KB TinyML footprint via INT8 quantization + ESP-NN; 50k+ synchronized multimodal data rows from 96 h decay protocol",
], tc=AMBER, tsize=14, bsize=12)
footer(s, n)

# ================= SLIDE 6 — Literature =================
n += 1; s = slide(); header(s, "Literature Review (Summary)", n)
deftable(s, 0.55, 1.45, 12.25,
         ["Work", "Approach", "Key Contribution", "Limitation (our motivation)"],
         [
            ["Tan & Xu 2020 [1]", "E-Nose / E-Tongue arrays", "Non-destructive gas sensing", "Gas-only; blind to visual & thermal cues"],
            ["Ru 2021 [2]", "TinyML on low-power MCUs", "On-device, ultra-low-power inference", "Benchmark only; no food application"],
            ["Gu et al. 2021 [3]", "Computer vision + ML", "Visual freshness inspection", "Fooled by chemical masking; misses internal rot"],
            ["Santos et al. 2022 [5]", "Multi-sensor fusion on MCU", "Low-power on-chip integration", "Not task-tailored; no anti-adulteration"],
            ["Wanniarachchi 2023 [7]", "Hybrid DL + multi-sensor", "Multi-sensor freshness scoring", "No chemical-masking detection"],
            ["Shuvo et al. 2024 [9]", "Smart gas scanner (wet markets)", "Portable formalin screening", "No synchronized visual+thermal fusion"],
            ["Banik & Sahu 2024 [10]", "Vision + gas late fusion", "Cross-modal classification", "High complexity; not TinyML-optimized"],
         ],
         colw=[2.3, 2.7, 3.3, 3.95], fsize=10.5, hsize=11)
txt(s, 0.55, 5.6, 12.2, 0.8, [("Research gap: no existing framework combines vision + formaldehyde + thermal streams with anti-masking anomaly detection inside a < 250 KB edge-AI budget.",
                               dict(size=14.5, bold=True, color=GREEN))])
footer(s, n)

# ================= SLIDE 7 — System overview =================
n += 1; s = slide(); header(s, "Proposed System — Tri-Layer Multimodal Scanner", n)
txt(s, 0.55, 1.3, 12.2, 0.5, [("Hardware: ESP32-S3 + OV2640 camera + ZE08-CH2O + BME680 + MLX90614 IR + TP4056/LiPo — 3D-printed hood builds a micro-climate chamber for stable sampling.", dict(size=13.5, color=DARKT))])
card(s, 0.55, 2.0, 4.0, 4.3, "Layer 1-2  |  Sensing & Extraction", [
    "Synchronized capture at t0 (image + gas + thermal)",
    "Visual domain → 224×224 frames → quantized CNN features (Fv, 128-D)",
    "Telemetry domain → min-max scaling + moving-average (Kalman-style) filtering",
    "Humidity-adaptive drift compensation of CH2O calibration",
], tc=GREEN, tsize=13, bsize=11)
card(s, 4.75, 2.0, 4.0, 4.3, "Layer 3-4  |  Fusion & Inference", [
    "Feature-level late fusion: Z = [Fv | Fs] (Fs = 16-D MLP telemetry embedding)",
    "Compact fusion network regresses the unified Freshness Index (0-100%)",
    "Runs 100% offline on ESP32-S3 with ESP-NN vector acceleration",
    "INT8 post-training quantization (~75% memory reduction)",
], tc=GREEN2, tsize=13, bsize=11)
card(s, 8.95, 2.0, 3.85, 4.3, "Layer 5  |  Anomaly (Anti-Adulteration)", [
    "Mahalanobis Distance matrix fit on Fresh-class fused features",
    "Regularized covariance; chi-square 99% alarm threshold",
    "Looks fresh BUT gas profile says toxic → CHEMICAL MASKING ALERT",
    "Results stream over BLE to the Flutter mobile app",
], tc=AMBER, tsize=13, bsize=11)
txt(s, 0.55, 6.5, 12.2, 0.45, [("Non-destructive  •  < 450 g  •  6 h battery  •  < 1.5 s per scan  •  zero cloud dependency",
                                dict(size=13.5, bold=True, color=GREEN))])
footer(s, n)

# ================= SLIDE 8 — Flowchart (replica of Slide 8 of proposal deck) =================
n += 1; s = slide(); header(s, "Proposed Methodology — Processing Flowchart", n)
box(s, 0.6, 3.4, 1.7, 0.9, "Food\nSample", AMBER, size=12.5)
# streams
txt(s, 2.55, 1.35, 2.4, 0.35, [("IMAGE STREAM", dict(size=12, bold=True, color=GREEN))])
box(s, 2.55, 1.8, 2.2, 0.75, "Camera\n(OV2640)", NAVY, size=11)
txt(s, 2.55, 5.75, 2.4, 0.35, [("SENSOR STREAM", dict(size=12, bold=True, color=GREEN))])
box(s, 2.55, 4.05, 2.2, 0.6, "Formaldehyde (ZE08)", NAVY, size=9.5)
box(s, 2.55, 4.8, 2.2, 0.6, "Gas / VOC (BME680)", NAVY, size=9.5)
box(s, 2.55, 5.55, 2.2, 0.6, "IR Temp (MLX90614)", NAVY, size=9.5)
box(s, 5.15, 3.35, 1.9, 0.95, "ESP32-S3\nMCU", GREEN, size=12)
box(s, 7.45, 1.7, 2.3, 0.85, "MobileNetV2\n(visual features Fv)", GREEN2, size=10.5)
box(s, 7.45, 3.0, 2.3, 1.35, "Normalization+\nKalman-style filter+\nDrift Compensation", GREEN2, size=10)
box(s, 7.45, 5.0, 2.3, 0.85, "MLP Regression\nModule  (Fs)", GREEN2, size=10.5)
box(s, 10.15, 3.4, 2.4, 0.85, "Mahalanobis\nDistance", AMBER, size=11.5)
box(s, 10.15, 1.9, 2.4, 0.8, "Decision Fusion\nEngine", GREEN, size=11.5)
box(s, 10.15, 5.0, 2.4, 0.85, "Results  (Freshness Index\n0-100% + Alert)", NAVY, size=9.5)
box(s, 10.15, 6.25, 2.4, 0.6, "WiFi / BLE → Mobile App", MGREY, size=9.5)
arrow(s, 2.3, 3.85, 2.55, 2.17)                    # food -> camera
arrow(s, 2.3, 3.85, 2.55, 5.85)                    # food -> sensors
arrow(s, 4.75, 2.17, 5.5, 3.38)                    # camera -> MCU
arrow(s, 4.75, 4.35, 5.5, 3.85)                    # sensors -> MCU
arrow(s, 7.05, 3.8, 7.45, 2.15)                    # MCU -> MobileNet
arrow(s, 7.05, 3.85, 7.45, 3.65)                   # MCU -> preprocessing
arrow(s, 7.05, 4.1, 7.45, 5.4)                     # MCU -> MLP
arrow(s, 9.75, 2.12, 9.75, 3.45)                   # -> fusion engine (via)
arrow(s, 9.75, 3.8, 10.15, 3.8)                    # -> mahalanobis? simplified below
arrow(s, 10.15, 2.3, 9.9, 3.42, AMBER)             # fusion -> mahalanobis
arrow(s, 10.15, 5.42, 9.9, 4.25, AMBER)            # results path
arrow(s, 9.75, 5.42, 9.75, 4.28)                   # MLP/regression up to mahalanobis
arrow(s, 10.15, 4.25, 10.15, 5.3, NAVY)            # mahalanobis -> results
txt(s, 0.6, 6.7, 9, 0.4, [("Both branches are frozen at fusion time — the two modalities meet only at the Mahalanobis / Decision Fusion stage.",
                           dict(size=12, italic=True, color=MGREY))])
footer(s, n)

# ================= SLIDE 9 — Decoupled training =================
n += 1; s = slide(); header(s, "Decoupled Branch Training — Image & Sensor Separately", n)
txt(s, 0.55, 1.3, 12.2, 0.55, [("Requirement: image dataset and sensor dataset are trained completely separately; the two 1-D feature spaces are joined only at fusion — this keeps the TinyML pipeline modular and memory-budgeted.",
                                dict(size=13.5, color=DARKT))])
box(s, 0.6, 2.1, 3.9, 2.6, "", LGREY)
txt(s, 0.75, 2.2, 3.6, 0.4, [("PART A — IMAGE BRANCH", dict(size=13, bold=True, color=GREEN))])
bullets(s, 0.75, 2.65, 3.6, 2.0, [
    "Kaggle fresh/stale image datasets",
    "MobileNetV2 (alpha 0.35) transfer learning",
    "224×224, [0,1] normalization",
    "Binary fresh/stale head (sigmoid)",
    "Output: Fv = 128-D feature extractor",
], size=10.5, gap=2)
box(s, 4.72, 2.1, 3.9, 2.6, "", LGREY)
txt(s, 4.87, 2.2, 3.6, 0.4, [("PART B — SENSOR BRANCH", dict(size=13, bold=True, color=GREEN))])
bullets(s, 4.87, 2.65, 3.6, 2.0, [
    "CH2O / VOC / temps / RH telemetry",
    "Min-max + moving-average smoothing",
    "MLP 64-32-16 (dropout regularized)",
    "3-state softmax: fresh / stale / masked",
    "Output: Fs = 16-D feature extractor",
], size=10.5, gap=2)
box(s, 8.84, 2.1, 3.95, 2.6, "", LGREY)
txt(s, 8.99, 2.2, 3.6, 0.4, [("PART C — FUSION (Decision)", dict(size=13, bold=True, color=AMBER))])
bullets(s, 8.99, 2.65, 3.65, 2.0, [
    "Both branches frozen; only fusion head trains",
    "Late fusion: Z = [Fv | Fs] (144-D joint matrix)",
    "Freshness Index regression (0-100%)",
    "Mahalanobis Distance anomaly layer",
    "Export: INT8 TFLite + C array + edge JSON",
], size=10.5, gap=2)
box(s, 0.6, 5.1, 12.2, 1.35, "Frequency mapping to thesis flow (Slide 8):  Camera→MobileNetV2  |  ZE08+BME680+MLX90614→Normalization→MLP  |  [Fv|Fs]→Mahalanobis→Decision Fusion→Results",
    NAVY, size=12.5)
footer(s, n)

# ================= SLIDE 10 — Implementation =================
n += 1; s = slide(); header(s, "Implementation — Kaggle Training Pipeline", n)
deftable(s, 0.55, 1.45, 7.4,
         ["Hyper-parameter", "Part A (image)", "Part B (sensor)", "Part C (fusion)"],
         [
            ["Backbone / structure", "MobileNetV2 α=0.35 + GAP + Dense128", "MLP 64-32-16", "MLP 64-32"],
            ["Optimizer / LR", "Adam 1e-3 (head), 1e-5 (FT)", "Adam 1e-3", "Adam 1e-3"],
            ["Loss", "Binary cross-entropy", "Sparse cat. cross-entropy", "MAE (freshness index)"],
            ["Input", "224×224×3, [0,1]", "8 telemetry features", "144-D fused matrix"],
            ["Epochs", "[XX] + [XX]", "[XX]", "[XX]"],
         ],
         colw=[2.0, 2.1, 1.6, 1.7], fsize=10, hsize=10.5)
card(s, 8.2, 1.45, 4.6, 3.0, "Kaggle Workflow", [
    "• Import notebook → attach fresh/stale image dataset",
    "• SENSOR_CSV = None → built-in 96 h synthetic decay generator (26-36 °C, formalin masking) until physical data is ready",
    "• GPU T4 → Run All → models, plots, .tflite in /kaggle/working/outputs",
    "• Reproducible: notebooks/food_freshness_kaggle_training.ipynb",
], tc=GREEN, tsize=12.5, bsize=10.5)
card(s, 0.55, 4.75, 12.25, 1.8, "Generated Thesis Evidence (auto-plots)", [
    "Adulteration boundary scatter (CH2O vs VOC)   •   Sensor cross-correlation heatmap   •   Confusion matrices (3-state)   •   Branch training curves   •   Fusion convergence",
], tc=AMBER, tsize=13, bsize=12)
footer(s, n)

# ================= SLIDE 11 — Results: branches =================
n += 1; s = slide(); header(s, "Results (1) — Branch-Level Performance", n)
txt(s, 0.55, 1.3, 12.2, 0.4, [("Image branch (Part A) — fresh vs stale", dict(size=14, bold=True, color=GREEN))])
deftable(s, 0.55, 1.75, 6.0, ["Metric", "Train", "Validation"],
         [["Accuracy (%)", "[XX.X]", "[XX.X]"], ["Precision (%)", "[XX.X]", "[XX.X]"],
          ["Recall (%)", "[XX.X]", "[XX.X]"], ["F1-score (%)", "[XX.X]", "[XX.X]"]],
         colw=[2.4, 1.8, 1.8], fsize=11)
box(s, 0.55, 4.15, 6.0, 0.8, "Figure: image branch training curves\n(outputs/image_branch_training.png)", LGREY, tcolor=DARKT, size=10.5)
txt(s, 6.85, 1.3, 6.0, 0.4, [("Sensor branch (Part B) — 3-state classifier", dict(size=14, bold=True, color=GREEN))])
deftable(s, 6.85, 1.75, 6.0, ["Class", "Precision", "Recall", "F1", "Support"],
         [["Fresh", "[XX]", "[XX]", "[XX]", "[XX]"], ["Stale (natural)", "[XX]", "[XX]", "[XX]", "[XX]"],
          ["Chem. masked", "[XX]", "[XX]", "[XX]", "[XX]"], ["Macro avg", "[XX]", "[XX]", "[XX]", "[XX]"]],
         colw=[1.7, 1.15, 1.15, 1.0, 1.1], fsize=10.5)
box(s, 6.85, 4.15, 6.0, 0.8, "Figure: confusion matrix + adulteration boundary scatter\n(outputs/sensor_confusion_matrix.png)", LGREY, tcolor=DARKT, size=10.5)
card(s, 0.55, 5.35, 12.25, 1.25, "Key Finding", [
    "The sensor branch reliably separates chemically-masked samples from natural decay — the decision the visual branch alone cannot make.",
], tc=AMBER, tsize=13, bsize=12)
footer(s, n)

# ================= SLIDE 12 — Results: fusion =================
n += 1; s = slide(); header(s, "Results (2) — Decision Fusion & Anti-Masking", n)
deftable(s, 0.55, 1.45, 7.4,
         ["Evaluation item", "Result", "Target"],
         [
            ["Fusion MAE (freshness index 0-1)", "[XX.XXXX]", "→ minimize"],
            ["Freshness-index error (± p.p.)", "[XX.X]", "± 5 p.p."],
            ["Mahalanobis threshold (χ² 99%, df=144)", "[XX.XX]", "alarm boundary"],
            ["Chemical-masking detection rate (TPR)", "[XX.X%]", "> 95%"],
            ["False alarm on fresh samples (FPR)", "[XX.X%]", "< 5%"],
         ],
         colw=[3.4, 2.0, 2.0], fsize=10.5)
card(s, 8.2, 1.45, 4.6, 3.1, "Case Study — Formalin Masking", [
    "Visually: surface intact, color preserved → image branch says FRESH (high score)",
    "Chemically: CH2O > 2 ppm with contradictory VOC profile → Fs far off-distribution",
    "Mahalanobis d(z) > threshold → contradiction exposed",
    "Decision Fusion Engine output: 'CHEMICAL MASKING ALERT' — bypasses any vision-only model",
], tc=RED, tsize=12.5, bsize=10.5)
box(s, 0.55, 4.95, 12.25, 1.5, "Freshness Index 0-100 %  +  Mahalanobis distance  →  tri-state final decision:  Fresh  |  Suspect anomaly  |  Stale / Spoiled  (+ masking alert)",
    NAVY, size=13.5)
footer(s, n)

# ================= SLIDE 13 — Edge deployment =================
n += 1; s = slide(); header(s, "Edge Deployment — TinyML on ESP32-S3", n)
deftable(s, 0.55, 1.5, 7.4,
         ["Artifact", "FP32 (KB)", "INT8 (KB)", "Reduction"],
         [
            ["Image branch (α=0.35 backend)", "[XXXX]", "[XXX]", "[XX%]"],
            ["Fusion engine", "[XX]", "[XX]", "[XX%]"],
            ["Sensor branch", "[XX]", "[XX]", "[XX%]"],
         ],
         colw=[3.2, 1.5, 1.5, 1.2], fsize=11)
card(s, 8.2, 1.5, 4.6, 2.6, "Firmware Integration", [
    "INT8 flatbuffer compiled into C array (image_branch.cc)",
    "edge_config.json: sensor scaling, Mahalanobis threshold, alert rule",
    "Tensor arena benchmarked < 250 KB (stack-overflow safe)",
    "ESP-NN vector ops @ 240 MHz → < 1.5 s full scan (capture + inference + BLE)",
], tc=GREEN, tsize=12.5, bsize=10.5)
card(s, 0.55, 4.4, 12.25, 2.0, "Hardware Integrity", [
    "KiCad PCB — star grounding, analog noise isolation, decoupling (0.1/10 µF) against RF & camera EMI; 3D-printed ergonomic gun-shell with micro-climate hood; TP4056 + 18650/2000 mAh; light-sleep duty cycling → ~6 h field operation; < 450 g single-hand use.",
    "Dual-Slope thermal compensation (Newton cooling + Stefan-Boltzmann) corrects MLX90614 readings on iced/moist market surfaces.",
], tc=GREEN2, tsize=13, bsize=12)
footer(s, n)

# ================= SLIDE 14 — Conclusion =================
n += 1; s = slide(); header(s, "Conclusion & Future Work", n)
card(s, 0.55, 1.4, 6.0, 4.7, "Conclusions", [
    "•  Portable, non-destructive multimodal scanner brings lab-grade food inspection to the point of purchase",
    "•  Image and sensor branches trained separately, fused via feature-level late fusion into a single 0-100% Freshness Index",
    "•  Mahalanobis anomaly layer makes the system unspoofable by chemical masking",
    "•  INT8 + ESP-NN delivers fully offline inference in wet markets",
    "•  Democratizes food safety: verification moves from state labs to consumers",
], tc=GREEN, tsize=15, bsize=12.5)
card(s, 6.8, 1.4, 6.0, 4.7, "Future Work", [
    "•  NIR multi-spectral sensing — pesticide/fertilizer residue screening",
    "•  Federated learning — privacy-preserving on-device model updates",
    "•  Robotic smart-bins (Raspberry Pi 5) for retail inventory decay management",
    "•  Geo-tagged BLE report aggregation → national food-safety heatmaps",
    "•  Expansion to local fish/meat variety calibration (Rui, Hilsha, beef, mango, banana)",
], tc=AMBER, tsize=15, bsize=12.5)
txt(s, 0.55, 6.35, 12.2, 0.5, [("A scalable blueprint for decentralized edge-AI food defense in developing regions.",
                                dict(size=13.5, bold=True, italic=False, color=GREEN))])
footer(s, n)

# ================= SLIDE 15 — References =================
n += 1; s = slide(); header(s, "References", n)
refs = [
    "J. Tan and J. Xu, \u201cApplications of E-Nose and E-Tongue in Food Quality and Safety Assessment,\u201d Sensors, vol. 20, no. 18, 2020.",
    "L. Ru, \u201cTinyML: Machine Learning on Low-Power Devices,\u201d IEEE Cloud Computing, vol. 8, no. 4, pp. 12-19, 2021.",
    "X. Gu, J. Sun, Y. Wang, \u201cComputer Vision and ML for Non-Destructive Food Quality Assessment,\u201d Food Engineering Reviews, vol. 13, no. 2, 2021.",
    "S. K. Ooi and T. J. Wong, \u201cLow-Cost IoT-Based Real-Time Formaldehyde Detection in Fishery Products,\u201d Measurement, vol. 182, 2021.",
    "M. S. Santos, J. P. Silva, R. Castro, \u201cMulti-Sensor Data Fusion inside Low-Power MCUs for Edge-AI,\u201d IEEE Sensors Letters, vol. 6, no. 3, 2022.",
    "Y. Zhao and J. Zhang, \u201cPost-Training Quantization for DNNs on Microcontrollers: A Survey,\u201d ACM Computing Surveys, vol. 55, no. 6, 2022.",
    "T. R. Wanniarachchi and S. R. Liyanage, \u201cReal-Time Food Freshness Detection Using Hybrid DL and Multi-Sensor System on Edge Devices,\u201d IEEE Access, vol. 11, 2023.",
    "A. K. Shukla, M. Chandra, P. K. Singh, \u201cNon-Contact IR Thermography for Sub-surface Food Decay Detection: A Review,\u201d Infrared Physics & Technology, vol. 128, 2023.",
    "S. Shuvo, M. N. Islam, T. Hasan, \u201cLow-Cost Smart Scanner for Formalin and Chemical Masking Detection in South Asian Wet Markets,\u201d Int. J. Food Contamination, vol. 11, no. 1, 2024.",
    "M. Banik and R. K. Sahu, \u201cFeature-Level Late Fusion of Multimodal Visual and Gas Sensor Arrays,\u201d Analytica Chimica Acta, vol. 1290, 2024.",
]
tb = s.shapes.add_textbox(Inches(0.6), Inches(1.35), Inches(12.2), Inches(5.6))
tf = tb.text_frame; tf.word_wrap = True
first = True
for i, r in enumerate(refs, 1):
    pgh = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
    pgh.space_after = Pt(7)
    run = pgh.add_run(); run.text = f"[{i}] {r}"
    run.font.size = Pt(12); run.font.color.rgb = DARKT; run.font.name = "Calibri"
footer(s, n)

# ================= SLIDE 16 — Q&A =================
n += 1; s = slide()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, 2.9, SW, 0.05, AMBER)
txt(s, 1, 1.7, 11.33, 1.0, [("Thank You", dict(size=44, bold=True, color=WHITE))], align=PP_ALIGN.CENTER)
txt(s, 1, 3.25, 11.33, 0.6, [("Questions & Discussion", dict(size=24, color=LGREY))], align=PP_ALIGN.CENTER)
txt(s, 1, 4.6, 11.33, 1.4, [
    ("Md. Hasibul Islam  |  ID: 0242310005341469", dict(size=15, color=GREEN2, bold=True)),
    ("Dept. of Software Engineering, Daffodil International University", dict(size=13, color=MGREY))],
    align=PP_ALIGN.CENTER)

prs.save("Final_Defense_Presentation.pptx")
print("Slides:", len(prs.slides.__iter__.__self__._sldIdLst))
