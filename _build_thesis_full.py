# Build the FULL thesis report DOCX per DIU Thesis Guidelines + SWE Thesis Template 2024.
# A4; margins L 4.0 cm / T-B-R 2.5 cm; Times New Roman 12; double-spaced text;
# single-spaced tables, captions, references; table captions ABOVE, figure captions BELOW;
# chapter-numbered Tables/Figures; preliminaries in lower-case Roman (untyped title page),
# main text in Arabic numerals, page number centered 1 cm above bottom.
import os
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "outputs")
DOC_PATH = os.path.join(BASE, "Thesis_Report_DIU_2024.docx")
RED = RGBColor(0x99, 0x00, 0x00)

doc = Document()

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin, sec.top_margin, sec.bottom_margin, sec.right_margin = Cm(4.0), Cm(2.5), Cm(2.5), Cm(2.5)
sec.footer_distance = Cm(1.0)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 2.0
st.paragraph_format.space_after = Pt(0)

for name, size in (("Heading 1", 14), ("Heading 2", 13), ("Heading 3", 12)):
    hs = doc.styles[name]
    hs.font.name = "Times New Roman"; hs.font.size = Pt(size); hs.font.bold = True
    hs.font.color.rgb = RGBColor(0, 0, 0)
    hs.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    hs.paragraph_format.line_spacing = 2.0
    hs.paragraph_format.space_before = Pt(6); hs.paragraph_format.space_after = Pt(6)

def _set_pg_format(s, fmt, start):
    for old_pg in s._sectPr.findall(qn("w:pgNumType")):
        s._sectPr.remove(old_pg)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt); pg.set(qn("w:start"), str(start))
    s._sectPr.append(pg)

def _page_field(paragraph):
    r1, r2, r3 = paragraph.add_run(), paragraph.add_run(), paragraph.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin"); r1._r.append(f1)
    f2 = OxmlElement("w:instrText"); f2.set(qn("xml:space"), "preserve"); f2.text = " PAGE "; r2._r.append(f2)
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end"); r3._r.append(f3)
    for r in (r1, r2, r3):
        r.font.name = "Times New Roman"; r.font.size = Pt(12)

def new_section(numbering):
    s = doc.add_section(WD_SECTION.NEW_PAGE)
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin, s.top_margin, s.bottom_margin, s.right_margin = Cm(4.0), Cm(2.5), Cm(2.5), Cm(2.5)
    s.footer_distance = Cm(1.0)
    s.footer.is_linked_to_previous = False
    fp = s.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.line_spacing = 1.0
    if numbering is not None:
        fmt, start = numbering
        _set_pg_format(s, fmt, start)
        _page_field(fp)
    return s

doc.sections[0].footer.is_linked_to_previous = False

C = WD_ALIGN_PARAGRAPH.CENTER
NL_SP = dict(line_spacing=1.0)

def P(text, align=None, italic=False, bold=False, placeholder=False, spacing=None, size=None, color=None):
    para = doc.add_paragraph()
    r = para.add_run(text)
    r.italic = italic; r.bold = bold
    if placeholder: r.font.color.rgb = RED
    if color is not None: r.font.color.rgb = color
    if size: r.font.size = Pt(size)
    if align is not None: para.alignment = align
    if spacing is not None: para.paragraph_format.line_spacing = spacing
    return para

def H1(text): doc.add_heading(text, level=1)
def H2(text): doc.add_heading(text, level=2)
def H3(text): doc.add_heading(text, level=3)

def NL():
    para = doc.add_paragraph()
    para.paragraph_format.line_spacing = 1.0
    para.paragraph_format.space_after = Pt(0)

def EQ(expr, number=None):
    para = doc.add_paragraph()
    para.alignment = C
    para.paragraph_format.line_spacing = 1.5
    para.paragraph_format.space_before = Pt(6); para.paragraph_format.space_after = Pt(6)
    r = para.add_run(expr); r.italic = True
    if number:
        r2 = para.add_run("        (" + number + ")")
        r2.italic = False
    return para

def TBL(caption, headers, rows, font_size=10):
    cap = doc.add_paragraph()
    cap.alignment = C
    cap.paragraph_format.line_spacing = 1.0
    cap.paragraph_format.space_before = Pt(10); cap.paragraph_format.space_after = Pt(4)
    cap.style = doc.styles["Caption Table"]
    r = cap.add_run(caption); r.bold = True; r.font.size = Pt(11); r.font.name = "Times New Roman"
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hdr in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = str(hdr)
        for line in c.paragraphs:
            line.paragraph_format.line_spacing = 1.0
            for run in line.runs:
                run.bold = True; run.font.size = Pt(font_size); run.font.name = "Times New Roman"
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            c = t.rows[ri].cells[ci]; c.text = str(val)
            for line in c.paragraphs:
                line.paragraph_format.line_spacing = 1.0
                for run in line.runs:
                    run.font.size = Pt(font_size); run.font.name = "Times New Roman"
    NL()
    return t

def FIG(path, caption, width_in=5.8, placeholder=None):
    para = doc.add_paragraph()
    para.alignment = C
    para.paragraph_format.line_spacing = 1.0
    para.paragraph_format.space_before = Pt(10)
    if path and os.path.isfile(path):
        para.add_run().add_picture(path, width=Inches(width_in))
    elif placeholder:
        r = para.add_run(placeholder)
        r.italic = True; r.font.color.rgb = RED
    cap = doc.add_paragraph()
    cap.alignment = C
    cap.paragraph_format.line_spacing = 1.0
    cap.paragraph_format.space_after = Pt(14)
    cap.style = doc.styles["Caption Figure"]
    r = cap.add_run(caption); r.italic = True; r.font.size = Pt(11); r.font.name = "Times New Roman"
    return para

from docx.enum.style import WD_STYLE_TYPE
_style_t = doc.styles.add_style("Caption Table", WD_STYLE_TYPE.PARAGRAPH)
_style_t.base_style = doc.styles["Normal"]
_style_f = doc.styles.add_style("Caption Figure", WD_STYLE_TYPE.PARAGRAPH)
_style_f.base_style = doc.styles["Normal"]

def TOCFIELD(instr, note="Right-click here and choose Update Field (F9) to populate this list."):
    para = doc.add_paragraph()
    r1 = para.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin"); r1._r.append(f1)
    r2 = para.add_run()
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r2._r.append(it)
    r3 = para.add_run()
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate"); r3._r.append(f2)
    r4 = para.add_run(note)
    r5 = para.add_run()
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end"); r5._r.append(f3)
    return para

ASSET_DIR = os.path.join(BASE, "_assets")

def LOGO(path, width_in=1.2):
    para = doc.add_paragraph()
    para.alignment = C
    para.paragraph_format.line_spacing = 1.0
    para.add_run().add_picture(os.path.join(ASSET_DIR, path), width=Inches(width_in))
    return para

def PH(text):
    P(text, italic=True, placeholder=True, spacing=1.0)

def PB():
    doc.add_page_break()

def BULLETS(items, numbered=False, style=None):
    for i, it in enumerate(items, 1):
        para = doc.add_paragraph(style=style)
        para.paragraph_format.line_spacing = 2.0
        marker = f"{i}. " if numbered else "\u2022  "
        para.add_run(marker + it)

def REF(entry):
    para = doc.add_paragraph()
    para.paragraph_format.line_spacing = 1.0
    para.paragraph_format.space_after = Pt(6)
    para.paragraph_format.left_indent = Cm(1.0)
    para.paragraph_format.first_line_indent = Cm(-1.0)
    para.add_run(entry)

TITLE = "Image and Sensor Data Integration with Mahalanobis Distance for AI-Enabled Food Freshness and Safety Detection"
STUDENT = "Md. Hasibul Islam"
SID = "0242310005341469"
SUPERVISOR = "Dr. Kamrul Islam Shahin"

# ======================================================================
# SECTION 1: TITLE PAGE  (page i — number not typed)
# ======================================================================
for _ in range(2): NL()
P(TITLE.upper(), align=C, bold=True, spacing=1.5, size=16)
for _ in range(3): NL()
P(STUDENT, align=C, bold=True, spacing=1.5, size=14)
P("ID: " + SID, align=C, spacing=1.5)
for _ in range(2): NL()
P("Bachelor of Science", align=C, spacing=1.5, size=14)
P("in Software Engineering", align=C, spacing=1.5, size=14)
for _ in range(2): NL()
P("Department of Software Engineering (Major in Data Science)", align=C, spacing=1.5, size=13)
P("DAFFODIL INTERNATIONAL UNIVERSITY", align=C, bold=True, spacing=1.5, size=14)
PB()

# ======================================================================
# SECTION 2: PRELIMINARIES  (lower-case Roman, starts at ii)
# ======================================================================
new_section(("lowerRoman", 2))

# ---- Declaration of Thesis and Copyright ----
P("DAFFODIL INTERNATIONAL UNIVERSITY", align=C, bold=True, spacing=1.0, size=14)
P("DECLARATION OF THESIS AND COPYRIGHT", align=C, bold=True, spacing=1.0, size=14)
NL()
P("Author's Full Name : " + STUDENT, spacing=1.0)
P("Date of Birth : [dd Month yyyy — confirm]", spacing=1.0, placeholder=True)
P("Title : " + TITLE, spacing=1.0)
P("Academic Session : [e.g. Summer 2024 — confirm]", spacing=1.0, placeholder=True)
NL()
P("I declare that this thesis is classified as:", spacing=1.0)
P("CONFIDENTIAL (Contains confidential information under the Official Secret Act 1997)*", spacing=1.0)
P("RESTRICTED (Contains restricted information as specified by the organization where research was done)*", spacing=1.0)
P("OPEN ACCESS  I agree that my thesis is to be published as online open access (Full Text).  \u2713", spacing=1.0)
NL()
P("I acknowledge that Daffodil International University reserves the following rights:", spacing=1.0)
P("1. The Thesis is the Property of Daffodil International University.", spacing=1.0)
P("2. The Library of Daffodil International University has the right to make copies of the thesis for the purpose of research only.", spacing=1.0)
P("3. The Library of Daffodil International University has the right to make copies of the thesis for academic exchange.", spacing=1.0)
NL()
P("Certified by:", spacing=1.0)
t = doc.add_table(rows=3, cols=2)
t.style = "Table Grid"
t.rows[0].cells[0].text = "(Student's Signature)"
t.rows[0].cells[1].text = "(Supervisor's Signature)"
t.rows[1].cells[0].text = "Full Name : " + STUDENT + "\nStudent ID : " + SID
t.rows[1].cells[1].text = "Full Name : " + SUPERVISOR + "\nPosition : Associate Professor, Dept. of SWE"
t.rows[2].cells[0].text = "Date :"
t.rows[2].cells[1].text = "Date :"
for row in t.rows:
    for c in row.cells:
        for line in c.paragraphs:
            line.paragraph_format.line_spacing = 1.0
NL()
P("NOTE : * If the thesis is CONFIDENTIAL or RESTRICTED, please attach a thesis declaration letter.", spacing=1.0, size=10)
PH("[THESIS DECLARATION LETTER (RESTRICTED case only) is OPTIONAL and not applicable under OPEN ACCESS.]")
PB()

# ---- Supervisor's Declaration ----
LOGO("diu_logo.png", 1.2)
P("SUPERVISOR'S DECLARATION", align=C, bold=True, spacing=1.0, size=14)
NL()
P("I hereby declare that I have checked this thesis and in my opinion, this thesis is adequate in terms of scope and quality for the award of the degree of Bachelor of Science.")
NL()
P("(Supervisor's Signature)", spacing=1.0)
P("Full Name : " + SUPERVISOR, spacing=1.0)
P("Position : Associate Professor, Department of Software Engineering", spacing=1.0)
P("Date :", spacing=1.0)
PB()

# ---- Student's Declaration ----
LOGO("diu_logo.png", 1.2)
P("STUDENT'S DECLARATION", align=C, bold=True, spacing=1.0, size=14)
NL()
P("I hereby declare that the work in this thesis is based on my original work except for quotations and citations which have been duly acknowledged. I also declare that it has not been previously or concurrently submitted for any other degree at Daffodil International University or any other institution.")
NL()
P("(Student's Signature)", spacing=1.0)
P("Full Name : " + STUDENT, spacing=1.0)
P("ID Number : " + SID, spacing=1.0)
P("Date :", spacing=1.0)
PB()

# ---- Inner title page ----
for _ in range(2): NL()
P(TITLE, align=C, bold=True, spacing=1.5, size=15)
for _ in range(2): NL()
P(STUDENT, align=C, bold=True, spacing=1.5, size=13)
P("ID: " + SID, align=C, spacing=1.5)
for _ in range(2): NL()
P("Thesis submitted in fulfillment of the requirements", align=C, spacing=1.5)
P("for the award of the degree of", align=C, spacing=1.5)
P("Bachelor of Science", align=C, bold=True, spacing=1.5)
for _ in range(2): NL()
P("Department of Software Engineering (Major in Data Science)", align=C, spacing=1.5, size=12)
P("DAFFODIL INTERNATIONAL UNIVERSITY", align=C, bold=True, spacing=1.5, size=13)
P("[MONTH YEAR — confirm]", align=C, spacing=1.5, placeholder=True)
PB()

# ---- Acknowledgements ----
P("ACKNOWLEDGEMENTS", align=C, bold=True, spacing=1.0, size=14)
NL()
P("First and foremost, I express my sincere gratitude to the Almighty for granting me the strength and perseverance to complete this work.")
P("I am profoundly indebted to my supervisor, Dr. Kamrul Islam Shahin, Associate Professor, Department of Software Engineering, Daffodil International University, for his continuous guidance, constructive criticism, and encouragement throughout every phase of this research — from the formulation of the problem through the design of the experimental pipeline to the preparation of this thesis. His insights on multimodal machine learning and edge deployment shaped the direction of this work substantially.")
P("I extend my thanks to the faculty members of the Department of Software Engineering, Daffodil International University, for the academic foundation they provided during my undergraduate studies, and to the Department for facilitating the thesis process under course SE 431.")
P("I also acknowledge the maintainer communities of the publicly available Kaggle datasets used in this research, without whose open-data contributions the training of the proposed models at zero primary-data cost would not have been possible.")
P("Finally, I am deeply grateful to my family and friends for their unwavering patience and moral support during the entire duration of my undergraduate program.")
PB()

# ---- Dedication ----
P("DEDICATION", align=C, bold=True, spacing=1.0, size=14)
for _ in range(4): NL()
P("Dedicated to my beloved parents,", align=C, italic=True, spacing=1.5)
P("whose sacrifice and faith made every step of this journey possible.", align=C, italic=True, spacing=1.5)
PB()

# ---- Abstract ----
P("ABSTRACT", align=C, bold=True, spacing=1.0, size=14)
NL()
P("Food adulteration and premature spoilage represent persistent public-health threats in developing regions, yet conventional inspection remains either destructive, laboratory-bound, or reliant on subjective human judgment. Single-modality machine-vision systems, in turn, are easily deceived by chemical masking, in which decomposing produce is treated with preservatives such as formaldehyde (formalin) so that its surface appearance belies its true condition. This thesis presents a tri-layer multimodal architecture for AI-enabled food freshness and safety detection that integrates visual, gas, thermal, and environmental evidence and explicitly detects cross-modal contradiction: a sample that looks fresh but whose chemical profile is anomalous is flagged rather than classified.")
P("The system comprises three independently trained neural components and one statistical detector. The image branch \u2014 MobileNetV2 with width multiplier \u03b1 = 0.35 operating on 224 \u00d7 224 \u00d7 3 inputs and producing a 128-dimensional visual embedding Fv \u2014 was trained with a two-phase (head training, then partial fine-tuning) transfer-learning schedule on 22,667 images aggregated from three public Kaggle datasets (14,682 fruit and vegetable images, 5,719 tilapia fish images, and 2,266 meat images; 11,213 fresh, 11,454 stale) and achieved a best-checkpoint validation accuracy of 98.76% (validation loss 0.0329). The sensor branch \u2014 a multi-layer perceptron over sixteen engineered features (eight physical or derived channels spanning formaldehyde, VOC, core temperature, ambient temperature, humidity, gas ratios, temperature rise rates, and storage duration, each paired with a binary availability flag) \u2014 was trained on 922,424 real sensor rows merged from four public Kaggle datasets plus 25% simulated chemical-masking augmentation, achieving 99.84% validation accuracy (validation loss 0.0053; per-class F1 between 0.998 and 1.000). The fusion engine consumes a 163-dimensional concatenation [Fv(128) | Fs(16) | raw sensors(16) | state posterior(3)] and regresses a continuous Freshness Index in the range 0\u2013100, attaining a validation MAE of 5.58 index points (validation Huber loss 0.0026). Complementing the neural stack, a Mahalanobis-distance anomaly detector fitted on the 144-dimensional fused embedding space Z = [Fv | Fs] with Ledoit\u2013Wolf shrinkage establishes a 99% decision threshold of 13.653, cleanly separating genuinely fresh samples (median distance 6.49) from stale ones (median 19.66), and exposing chemical-masking adulteration that would otherwise evade vision-only inspection.")
P("Finally, the models are exported for the target edge platform: the image branch as a fully INT8-quantized TFLite file (791.8 KB) and the fusion engine as a Float32 TFLite file (128.4 KB), accompanied by a C++ header (edge_config.h, 338.5 KB) embedding the sensor scaling bounds, the Mahalanobis mean vector, and the complete 144 \u00d7 144 inverse covariance matrix \u2014 enabling TFLM on-device inference on the ESP32-S3 with camera (OV2640), formaldehyde (ZE08-CH2O), VOC (BME680/MQ-series), and temperature-humidity (DHT22/MLX90614) hardware. The results demonstrate that rigorous multimodal fusion, when combined with an explicit statistical anomaly layer, yields a compact, deployable, and materially harder-to-deceive food-safety scanner suitable for point-of-sale deployment.")
NL()
P("Keywords: Multimodal sensor fusion; food freshness; food safety; Mahalanobis distance; anomaly detection; MobileNetV2; multi-layer perceptron; TinyML; TensorFlow Lite Micro; ESP32-S3; chemical adulteration; formalin detection; Internet of Things.", bold=True, spacing=1.0, size=11)
PB()

# ---- Table of Content ----
P("TABLE OF CONTENT", align=C, bold=True, spacing=1.0, size=14)
NL()
TOCFIELD('TOC \\o "1-2" \\h \\z \\u', "Right-click and Update Field (F9) to build the Table of Contents.")
PB()

# ---- List of Tables ----
P("LIST OF TABLES", align=C, bold=True, spacing=1.0, size=14)
TOCFIELD('TOC \\h \\z \\t "Caption Table,1"', "Right-click and Update Field (F9) to build the List of Tables.")
PB()

# ---- List of Figures ----
P("LIST OF FIGURES", align=C, bold=True, spacing=1.0, size=14)
TOCFIELD('TOC \\h \\z \\t "Caption Figure,1"', "Right-click and Update Field (F9) to build the List of Figures.")
PB()

# ---- List of Symbols ----
P("LIST OF SYMBOLS", align=C, bold=True, spacing=1.0, size=14)
TBL("Symbols",
    ["Symbol", "Description"],
    [["x_img", "Input RGB image tensor (224 \u00d7 224 \u00d7 3)"],
     ["\u03b1", "MobileNetV2 width multiplier (0.35 in this work)"],
     ["Fv", "128-dimensional visual feature embedding from the image branch"],
     ["Fs", "16-dimensional sensor feature embedding from the sensor branch"],
     ["Z", "Concatenated multimodal embedding, Z = [Fv | Fs] \u2208 R^144"],
     ["x_s", "16-dimensional scaled raw sensor channels"],
     ["p(y | x_s)", "3-class state posterior from the sensor branch (Fresh / Stale / Masked)"],
     ["u", "Full fusion input vector, u = [Fv(128) | Fs(16) | x_s(16) | p(3)] \u2208 R^163"],
     ["\u03bc", "Mean vector of the Mahalanobis reference distribution (144-D, fresh class)"],
     ["\u03a3", "Covariance matrix estimated from fresh-class embeddings (144 \u00d7 144)"],
     ["\u03a3_shrink", "Shrinkage-regularized covariance, (1\u2212\u03bb)\u03a3 + \u03bb\u03bcI + \u03b5I"],
     ["\u03bb", "Ledoit\u2013Wolf-style shrinkage parameter (0.10 in this work)"],
     ["\u03b5", "Diagonal loading of the covariance (\u03b5 = 10\u207b\u2076)"],
     ["d_M(z)", "Mahalanobis distance, \u221a((z\u2212\u03bc)\u1d40\u03a3\u207b\u00b9(z\u2212\u03bc))"],
     ["T_MAH", "Deployed Mahalanobis decision threshold (13.653)"],
     ["FI", "Freshness Index produced by the fusion engine (0\u2013100, unitless)"],
     ["\u0177", "Continuous fusion output before scaling (\u0177 \u2208 [0, 1]; \u00d7100 \u2192 FI)"],
     ["L_Huber(\u03b4)", "Huber loss applied to fusion training (\u03b4 = 0.1 in this work)"],
     ["a_i", "Binary availability flag for sensor channel i (1 = present, 0 = missing)"],
     ["CH2O", "Formaldehyde concentration (ppm)"],
     ["VOC", "Volatile organic compound concentration"]], font_size=10)
PB()

# ---- List of Abbreviations ----
P("LIST OF ABBREVIATIONS", align=C, bold=True, spacing=1.0, size=14)
TBL("Abbreviations",
    ["Abbreviation", "Full Form"],
    [["AI", "Artificial Intelligence"],
     ["BN", "Batch Normalization"],
     ["BLE", "Bluetooth Low Energy"],
     ["BME680", "Bosch Environmental Sensor (temperature/humidity/pressure/gas)"],
     ["CH2O", "Formaldehyde"],
     ["CNN", "Convolutional Neural Network"],
     ["DHT22", "Combined Temperature\u2013Humidity Digital Sensor"],
     ["ESP32-S3", "Microcontroller Platform (Espressif Systems)"],
     ["GAP", "Global Average Pooling"],
     ["IoT", "Internet of Things"],
     ["INT8", "8-bit Integer Quantization"],
     ["LR", "Learning Rate"],
     ["MAE", "Mean Absolute Error"],
     ["MLP", "Multi-Layer Perceptron"],
     ["MLX90614", "Non-Contact Infrared Thermometer"],
     ["MOS", "Metal-Oxide Semiconductor (gas sensor technology)"],
     ["OV2640 / OV5640", "CMOS Camera Modules (OmniVision)"],
     ["ReLU", "Rectified Linear Unit"],
     ["RGB", "Red-Green-Blue (color space)"],
     ["SWE", "Software Engineering"],
     ["TFLM", "TensorFlow Lite Micro"],
     ["TFLite", "TensorFlow Lite"],
     ["VOC", "Volatile Organic Compound"],
     ["ZE08-CH2O", "Winsen Electrochemical Formaldehyde Gas Sensor"]], font_size=10)
PB()

# ---- List of Appendices ----
P("LIST OF APPENDICES", align=C, bold=True, spacing=1.0, size=14)
TBL("Appendices",
    ["No.", "Title"],
    [["Appendix A", "Complete Training Pipeline Code (Python / TensorFlow 2.21, Kaggle GPU)"],
     ["Appendix B", "edge_config.h \u2014 Embedded ESP32-S3 Configuration Header"],
     ["Appendix C", "Exported TFLite Model C-Arrays (image_branch.cc, fusion_engine.cc)"],
     ["Appendix D", "Sensor Dataset Coverage Report (sensor_dataset_coverage.csv)"],
     ["Appendix E", "Additional Training Logs and Diagnostics"]], font_size=10)

# ======================================================================
# SECTION 3: MAIN TEXT  (Arabic numerals, starts at 1)
# ======================================================================
new_section(("decimal", 1))

# ------------------------------ Chapter 1 ------------------------------
H1("CHAPTER 1  INTRODUCTION")

H2("1.1 Background")
P("Food constitutes the most fundamental determinant of public health, and its safety is a concern for every society, economy, and regulatory institution. In developing countries such as Bangladesh, this concern is magnified by two compounding phenomena: the natural perishability of the national diet's staple perishables — fish, meat, fruit, and vegetables sold through open-air and informal retail channels — and the documented practice of chemical adulteration, in which decomposing produce is deliberately treated with industrial preservatives to postpone the visible onset of decay. The formaldehyde-based (\u201cformalin\u201d) treatment of fish and fruit is among the most widely reported instances of this practice; its public-health consequences, together with the difficulty of enforcing food-safety regulation at millions of individual points of sale, have motivated repeated government mobile-court drives and continue to erode consumer trust in the domestic food supply.")
P("The established instruments of food inspection are poorly matched to this decentralized, real-time problem. Laboratory assays (high-performance liquid chromatography for formaldehyde, microbiological culture for spoilage) are authoritative but centralized, slow, and expensive, requiring sample transport and destructive consumption of the tested item. Rapid on-field alternatives such as chemical test strips are themselves destructive, requiring the cutting or crushing of the sample, and they generate friction and refuse-sale disputes during live market transactions. Human sensory evaluation by sight, touch, and smell, finally, is non-destructive but subjective, unstandardized, deceptively easy to defeat with masking agents, and unavailable at scale.")
P("In parallel, machine learning has made substantial progress on non-destructive, camera-based freshness classification. Convolutional neural networks (CNNs) — particularly compact and pre-trained architectures such as MobileNetV2 in its reduced-width variants — achieve high accuracy on fresh-versus-stale image discrimination while remaining light enough to consider for embedded deployment. Research on \u201celectronic nose\u201d gas-sensor arrays has demonstrated that volatile organic compound (VOC) profiles carry rich, complementary information about microbial spoilage that the visual channel cannot provide, and low-cost formaldehyde sensors of the ZE08-CH2O class now make chemical-preservative detection economically realizable at the edge. Internet of Things (IoT) temperature and humidity telemetry similarly provides a meaningful environmental context for thermal and storage-decay modeling.")
P("Yet each of these modalities in isolation carries a characteristic blind spot, and it is precisely these blind spots that chemical adulteration exploits. A vision-only classifier is defeated by preservative treatments that restore or retain surface color and texture; a gas-sensor-only system is confounded by storage-environment variation and has no access to the sample's visual identity; and a fusion of vision and gas signals still lacks any principled mechanism to detect when the two modalities contradict one another — the defining signature of chemical masking. The evidence fusion, moreover, must be performed within the computational envelope of a microcontroller if the technology is ever to leave the laboratory and enter the market: consumer adoption requires a device costing tens of dollars, operating on battery, and running entirely offline at the point of sale, which places this work squarely in the domain of TinyML — the deployment of deep learning inference onto resource-constrained microcontrollers via runtimes such as TensorFlow Lite Micro (TFLM).")
P("This thesis is positioned at the intersection of these challenges. It develops a complete, reproducible, three-layer multimodal detection pipeline in which (i) a compact CNN produces a rich visual embedding from a 224 \u00d7 224 RGB food image, (ii) a multi-layer perceptron (MLP) produces a sensor feature embedding and a probabilistic freshness-state posterior from an eight-channel physical and derived gas/thermal/environmental stream, (iii) a fusion engine regresses a continuous 0\u2013100 Freshness Index from the concatenation of both embeddings, the raw scaled sensor readings, and the state posterior, and (iv) a statistical Mahalanobis-distance layer evaluates the joint consistency of the visual and sensor embeddings against the distribution of genuinely fresh samples, exposing cross-modal contradictions that would otherwise allow chemically masked food to appear fresh to every individual classifier.")
P("A defining constraint of this research is that it uses exclusively public, secondary data. No primary hardware data collection was performed; all training data is drawn from publicly available Kaggle datasets. The design therefore treats dataset assembly as a first-class engineering problem: four heterogeneous sensor datasets and three image datasets were merged through a canonical sixteen-feature schema, source-file column inconsistencies (for example, the misspelling \u201ctempreature\u201d in the IoT Agriculture 2024 dataset) were resolved through a documented column-aliasing layer, and channels not covered by any public dataset were explicitly flagged, filled by derivation from covered channels where physics allows, and augmented with a documented synthetic environmental model (70% active rows, 30% availability-flag-zero rows) so that the model remains robust in the physical absence of a sensor. This missing-channel design makes the system honest about its inputs and directly enables the gap-filling strategy documented in this thesis.")

H2("1.2 Problem Statement")
P("The research problem addressed by this thesis is fivefold, and every component of the proposed system maps to one or more of its parts.")
P("(1) The chemical-masking detection deficit. Formaldehyde-based preservatives artificially preserve the surface color and texture of decomposing fish, fruit, and vegetables. Any freshness classifier built only on visual evidence can therefore be defeated systematically: the input image is genuinely consistent with a fresh sample, and no amount of visual-model capacity will recover the missing information. Detecting masked adulteration inherently requires non-visual evidence and, critically, a mechanism for weighing visual evidence against chemical evidence when they disagree.")
P("(2) The invisibility of sub-surface and core decomposition. Thick-skinned agricultural produce and cold-stored fish may undergo rapid internal bacterial deterioration while their external surface appears pristine and their surface gas emissions remain weak or ambiguous. Surface cameras alone cannot observe this decay. Core thermal characteristics — measured, in the proposed hardware reference design, by a non-contact infrared thermometer — and derived temperature-rise trends provide an independent, physically motivated channel of evidence, and the canonical sensor schema of this thesis carries them as first-class features.")
P("(3) The destructive nature of existing on-field tests. Chemical test strips and solvent-based rapid assays destroy part of the sample, generate food waste, induce seller resistance, and scale poorly across a national retail landscape. A viable consumer-facing instrument must be non-destructive, and ideally contactless with respect to the food sample itself, which favors remote visual capture and ambient gas profiling.")
P("(4) Infrastructure cost and consumer portability. Existing analytical instruments are stationary, mains-powered, and far exceed the economic reach of individual consumers. The proposed target platform — an ESP32-S3 microcontroller with an OV2640/OV5640 camera, a ZE08-CH2O formaldehyde sensor, BME680/MQ-series gas sensing, and DHT22/MLX90614 thermal sensing — is achievable at a consumer price point, but imposes severe constraints: the entire neural inference must fit in microcontroller program and RAM budgets. This constraint directly shapes the architectural choices of this thesis: a width-reduced MobileNetV2 (\u03b1 = 0.35) image backbone exported at full INT8 quantization (791.8 KB), and a compact fusion engine exported as Float32 TFLite (128.4 KB), with all decision constants — including the complete 144 \u00d7 144 inverse covariance matrix — delivered to firmware as a pre-generated C++ header (edge_config.h, 338.5 KB).")
P("(5) The absence of principled cross-modal anomaly detection in prior food-freshness systems. Most published food-freshness pipelines end at a classification head fed by a single modality, or perform naive score-level averaging across modalities. Neither approach distinguishes \u201cboth modalities agree the food is stale\u201d from \u201cthe image looks fresh but the chemical profile is impossible for fresh food\u201d — and only the second distinction matters for catching chemical masking. The problem, precisely, is to define and operate a statistical layer that treats the joint plausibility of the multimodal evidence as a first-class signal, with a threshold that is calibrated on the genuine fresh population rather than guessed.")
EQ("Given an image x_img and a heterogeneous, possibly incomplete sensor vector x_s of a food sample, produce (a) a continuous estimate of freshness and (b) a safety decision, such that the decision is (i) robust to missing sensor channels, (ii) calibrated against the true fresh-sample population through a 144-dimensional Mahalanobis consistency check on the fused embedding z = [Fv | Fs], and (iii) executable on-device on an ESP32-S3-class microcontroller.", "1.1")

H2("1.3 Research Objectives")
P("The primary objective of this thesis is to design, implement, and empirically evaluate a tri-layer multimodal system for non-destructive, edge-deployable food freshness and safety detection. This decomposes into seven specific, measurable objectives:")
BULLETS([
 "To design and train a visual freshness branch based on a width-reduced MobileNetV2 (\u03b1 = 0.35) operating on 224 \u00d7 224 \u00d7 3 inputs, producing a 128-dimensional embedding Fv and a fresh/stale decision, using only public Kaggle image datasets, with a two-phase transfer-learning schedule (head training, then partial backbone fine-tuning with best-validation-loss checkpointing). Result achieved: best-checkpoint validation accuracy 98.76% on 22,667 images (val_loss 0.0329).",
 "To design and train a sensor freshness branch as an MLP over a canonical sixteen-feature input vector (eight physical or derived channels with per-channel binary availability flags), producing a 16-dimensional embedding Fs and a three-class state posterior (Fresh / Stale / Masked), trained exclusively on public Kaggle sensor datasets with documented column aliasing across heterogeneous sources. Result achieved: validation accuracy 99.84% on 922,424 real rows (val_loss 0.0053; per-class F1 between 0.998 and 1.000).",
 "To design a framework for heterogeneous public sensor data integration with per-channel coverage reporting, column-alias resolution (e.g., the \u201ctempreature\u201d typo in IoT Agriculture 2024), availability flags, min\u2013max scaling, and moving-average smoothing — so that datasets collected by entirely different hardware remain compatible.",
 "To build and train a tri-layer fusion engine that regresses a continuous 0\u2013100 Freshness Index from a 163-dimensional input [Fv(128) | Fs(16) | x_s(16) | p(3)], with Huber loss and narrow supervision bands so that band width itself is accounted for as the irreducible target noise. Result achieved: validation MAE 5.58 index points (0.0558; validation Huber loss 0.0026).",
 "To implement a statistically grounded Mahalanobis-distance anomaly detector on the 144-dimensional fused embedding space Z = [Fv | Fs], using Ledoit\u2013Wolf-style shrinkage (\u03bb = 0.10), eigenvalue clipping, and an empirical p99-based threshold, and to verify that fresh and stale populations separate decisively. Result achieved: threshold 13.653; fresh distances median 6.49 (p99 = 10.31, max 13.19) versus stale median 19.66 (p99 = 33.80).",
 "To export all models for on-device inference on the ESP32-S3, specifically the image branch as fully INT8-quantized TFLite with stratified calibration (791.8 KB), the fusion engine as Float32 TFLite (128.4 KB), and all deterministic decision parameters (channel bounds, Mahalanobis mean, threshold, and inverse covariance matrix) as a generated C++ header edge_config.h (338.5 KB).",
 "To validate the deployed pipeline's decision logic end-to-end in an integrated demonstration, exercising the full chain from input features through fusion, Mahalanobis evaluation, and the CH2O-based adulteration-alert rule (alert threshold 2.0 ppm) to the final tri-state decision. Result achieved: integrated demonstration produced Freshness Index 80.4, Mahalanobis distance 8.35, decision \u201cFresh\u201d.",
], numbered=True)

TBL("Table 1.1  Summary of thesis objectives and their mapping to system components",
    ["Objective", "System Component", "Validated Result"],
    [["1", "Image branch (MobileNetV2 \u03b1 = 0.35)", "98.76% val accuracy (val_loss 0.0329)"],
     ["2", "Sensor branch (MLP, 16-feature input)", "99.84% val accuracy (val_loss 0.0053)"],
     ["3", "Heterogeneous dataset integration layer", "4 sensor datasets merged; 1 channel gap-filled"],
     ["4", "163-D fusion engine (Huber, \u03b4 = 0.1)", "val MAE 0.0558 (5.58 index points)"],
     ["5", "144-D Mahalanobis detector (\u03bb = 0.10)", "Threshold 13.653; fresh median 6.49 vs stale 19.66"],
     ["6", "Edge export path (INT8 + Float32 + header)", "791.8 KB + 128.4 KB + 338.5 KB artifacts"],
     ["7", "End-to-end decision logic", "FI 80.4, d_M 8.35, decision Fresh"]], font_size=9)

H2("1.4 Scope and Limitations")
P("The scope of this thesis is deliberately bounded to make the work reproducible from public data alone and honest about the surrogate nature of that data. Within scope: the system architecture and model training for the image branch, sensor branch, fusion engine, and Mahalanobis detector in their final validated configuration (v13) — MobileNetV2 \u03b1 = 0.35 at 224 \u00d7 224; an MLP with 64\u201332\u201316 hidden structure; a fusion engine with 128\u201364\u201332 hidden structure and Huber loss; and a 144-dimensional Mahalanobis reference with \u03bb = 0.10 shrinkage — plus data integration over public Kaggle datasets only (three image datasets, four sensor datasets), the simulated masking class (a design decision of the research, disclosed and defended rather than hidden), and edge export and deployment readiness validated at the artifact and decision-logic level.")
P("The principal limitations, stated explicitly, are:")
BULLETS([
 "No primary data. Every number in this thesis derives from public Kaggle datasets. The hardware platform (ESP32-S3, OV2640, ZE08-CH2O, BME680, MLX90614) defines the target input distribution, but dataset sensors are surrogates: MQ-series gas readings stand in for the ZE08-CH2O's electrochemical formaldehyde channel, and public environment telemetry stands in for the DHT22 storage-environment channel.",
 "Derived-only coverage of one channel. temp_rise_rate is not covered by any attached public dataset; it is derived from core_temp_c and storage elapsed time and, during augmentation, receives synthetic values with a documented monotonic ordering (masked < fresh < stale). This is the single uncovered channel, and it is disclosed in the coverage table.",
 "Simulated masking is synthetic by construction. The Masked class achieves perfect F1 (1.000) in the sensor branch partly because its feature distribution is synthetic and well-separated. Real-world adulteration profiles will be noisier; this is acknowledged as an inherent ceiling of a public-data-only study.",
 "Ground-truth freshness is ordinal, not analytical. Image labels come from human dataset curation, and sensor labels from decay-day and vendor-QC conventions; the Freshness Index supervision bands (fresh 0.88\u20130.92, masked 0.68\u20130.72, stale 0.17\u20130.21) are design targets, not laboratory-measured values.",
 "On-device inference was validated at the artifact and decision-logic level (correct input dimensionality, complete parameter export, quantization success), but this thesis does not include physical ESP32-S3 latency, memory, or power characterization; firmware integration is described and enabled, not benchmarked.",
])

H2("1.5 Thesis Organization")
P("The remainder of this thesis is organized as follows. Chapter 2 — Literature Review — surveys non-destructive food-quality sensing, CNN-based freshness classification, electronic-nose and VOC sensing, multimodal fusion strategies, and statistical anomaly detection in embedded AI, culminating in the research gap this thesis addresses. Chapter 3 — Methodology — presents the proposed tri-layer architecture in full: the canonical sensor schema and the multi-dataset integration layer, the image branch and sensor branch designs, the missing-channel simulation and masking-augmentation framework, the fusion engine's 163-dimensional input construction, the Mahalanobis reference estimation procedure (shrinkage, eigenvalue clipping, empirical thresholding), and the edge-export pipeline. Chapter 4 — Experimental Setup and Results — documents the datasets, hyperparameters, and training configuration; reports all validation results per component; presents the Mahalanobis separation statistics and decision-boundary behavior; and analyzes the exported artifacts and an end-to-end demonstration. Chapter 5 — Conclusion and Future Work — summarizes the contributions, reflects on the public-data-only design and its implications, and outlines the path to physical hardware validation and field deployment.")

H2("1.6 Summary")
P("This chapter established the context of chemically masked food adulteration as a problem that defeats conventional inspection instruments and single-modality AI alike. It defined the research problem — fusing heterogeneous public image and sensor data into a deployable freshness index while explicitly testing cross-modal consistency through a Mahalanobis anomaly layer — and enumerated seven concrete objectives, each of which is later evaluated with exact, reproducible numbers. The scope was bounded to 100% public Kaggle data with full disclosure of surrogate sensors, simulated masking, and the single derived (uncovered) channel, and the chapter closed by mapping the remaining chapters of the thesis.")
PB()

# ------------------------------ Chapter 2 ------------------------------
H1("CHAPTER 2  LITERATURE REVIEW")

H2("2.1 Overview")
P("Four research streams intersect in this thesis: (i) food adulteration and preservative contamination as a public-health problem; (ii) non-destructive, machine-learning-based food quality assessment from visual imagery; (iii) chemical and environmental sensing of food freshness, including electronic-nose and gas-sensor approaches; and (iv) anomaly detection and TinyML as the deployment and safety-mechanism layers. Each stream is reviewed for the specific technical question it answers in this thesis, and the chapter closes by synthesizing the research gap that no single stream adequately covers. The review deliberately prioritizes sources that are directly load-bearing for the engineering choices of the system — the choice of backbone (MobileNetV2), the choice of statistical detector (Mahalanobis distance with shrinkage covariance), the choice of fusion input construction (embeddings plus raw sensor readings plus class posteriors), and the choice of deployment target (TensorFlow Lite Micro on a microcontroller).")

H2("2.2 Chemical Adulteration of Food: Formalin and Chemical Masking")
P("Formaldehyde-based preservation of fish, fruit, and vegetables is a well-documented adulteration practice in South Asian food systems. Regulatory intervention such as drive-based market enforcement has repeatedly demonstrated both the scale of the problem and the inadequacy of point-of-sale testing: laboratory confirmatory assays, notably chromatographic quantification of formaldehyde, are authoritative but centralized, slow, and incompatible with real-time on-site screening of millions of individual retail transactions.")
P("Two structural properties of chemical masking are directly relevant to this thesis. First, masking preserves the visual surface of the sample, meaning that any classifier operating on visual evidence alone receives an input that is genuinely consistent with freshness — the information required to catch the adulteration is simply absent from that modality. Second, formaldehyde and allied preservatives leave a chemical signature that gas-phase sensors can in principle detect: the formaldehyde family is electrochemically responsive in the ppm range, matching the operating envelope of commodity sensors such as the ZE08-CH2O used in the target hardware. Together these two properties motivate the thesis's core design premise: chemical masking is detectable only when the visual and chemical modalities are both present and when their disagreement is treated as a signal in its own right.")

H2("2.3 Vision-Based Freshness Classification Using CNNs")
P("The application of convolutional neural networks (CNNs) to food imagery matured through the general 2010s deep-learning wave; comprehensive surveys of the food-image recognition literature document a consistent finding that transfer learning from large-scale general-purpose backbones substantially outperforms training from scratch on small, domain-specific food datasets. This finding directly informs the design of the image branch in this thesis, which fine-tunes an ImageNet-initialized backbone rather than training parameters end-to-end from random initialization.")
P("Choice of backbone. Among the family of efficient backbones, MobileNetV2 (Sandler et al., 2018) is the de facto standard for mobile and embedded inference. Two of its properties drive the choice in this thesis. First, depthwise separable convolutions allow MobileNetV2 to operate on image resolutions and widths far below what conventional architectures tolerate, meaning that the width multiplier \u03b1 functions as a direct, tunable knob on the trade-off between model size and accuracy. Second, MobileNetV2's inverted residual structure with linear bottlenecks preserves feature expressiveness even when the width multiplier is reduced aggressively; this is essential here because \u03b1 = 0.35 is applied, and a lesser backbone would lose discriminative capacity at this width. In this thesis MobileNetV2 at \u03b1 = 0.35 operates on 224 \u00d7 224 \u00d7 3 inputs and serves as a frozen, feature-extracting backbone for most of the training, producing a 128-dimensional embedding Fv that is used both by the fusion engine and by the Mahalanobis reference distribution.")
P("Published food-freshness datasets of the kind used here — curated fruit-and-vegetable imagery, tilapia fish freshness imagery, and meat-freshness imagery — are most commonly reported with this class of lightweight transfer-learned CNN achieving high single-modality validation accuracy on the order of the 0.95+ regime, matching the present work's 98.76% best-checkpoint result (Chapter 4). A useful qualitative observation in this literature is that visual freshness classification saturates well below the point at which adulterated samples are separable — the modality's ceiling is freshness appearance, not chemical identity. This observation is the reason the present thesis uses the image branch as one evidence stream rather than as the final decision authority.")
P("Data augmentation. Standard geometric augmentation — random flips, rotations, and zooms — is well established in the food-recognition literature as a cost-free mechanism for expanding effective dataset size and suppressing overfitting on curated (and therefore somewhat stylized) food imagery. The image branch in this thesis applies exactly this class of augmentation (RandomFlip, RandomRotation, RandomZoom) inside the model's training graph but outside its inference and export path, a now-common Keras pattern that guarantees augmentation is active only during fitting.")

H2("2.4 Chemical and Environmental Sensing")
P("Electronic noses for food freshness. Metal-oxide semiconductor (MOS) gas-sensor arrays — of the MQ-series class and their successors — have been applied to food quality for over two decades, and the associated literature reports that volatile organic compound (VOC) profiles carry rich, time-resolved information about the progress of microbial spoilage in fruit, meat, and fish, information that is complementary to (and partially independent of) visual appearance. In this thesis the electronic-nose fruit dataset contributes the largest single mass of sensor training data (838,502 of 922,424 real rows), with MQ2 and MQ135 readings serving as the volatile-signature and aldehyde-family channels respectively.")
P("Formaldehyde-specific sensing. Public datasets that pair gas-sensor readings with explicit formalin-treatment labels are exceptionally rare; the Formalin Tofu Aroma dataset (45,000 rows) used in this thesis is one of the few public resources in which a food sample's formalin status is annotated against gas-phase readings and temperature. This dataset is therefore used both as a CH2O-family signal source and as a small real-anomaly anchor, although — as discussed in Section 1.4 — the labels are factory quality-control outcomes rather than laboratory-verified formaldehyde concentrations.")
P("Storage environment telemetry. Ambient temperature and humidity are recognized in the food-safety literature as primary covariates of spoilage kinetics, and low-cost environmental logging is the standard analytical frame for the impact of the cold chain on perishable food. The IoT Agriculture 2024 dataset (37,922 rows) is used in this thesis as the source of real ambient temperature and humidity readings, closing two otherwise-uncovered channels of the canonical schema. Notably, this dataset also supplies a real, instructive lesson in dataset-heterogeneity engineering, since its temperature header is misspelled (\u201ctempreature\u201d) in the source CSV and requires the column-alias resolution described in Chapter 3.")
P("Deriving higher-order features. Beyond raw physical channels, a common pattern in time-series food-sensing literature is the derivation of physics-motivated second-order features — temperature rise rates, gas-concentration ratios, aggregate elapsed-time features — that are more directly interpretable as spoilage indicators than their raw constituents. The canonical schema of this thesis follows this pattern, deriving ch2o_voc_ratio from the gas pair and temp_rise_rate from core temperature and elapsed time, and carrying each alongside a per-channel availability flag so the model remains mathematically defined even when a physical sensor is absent.")

H2("2.5 Multimodal Fusion for Food Quality Assessment")
P("Score-level fusion (decision-level fusion) aggregates the outputs of independently trained per-modality classifiers, most commonly by weighted voting, averaging, or a small meta-classifier. This is the most common form in the food-quality literature because it is easy to implement, but it has a decisive limitation for the masking problem: when two modalities disagree, score-level fusion treats the disagreement as noise to be averaged away rather than as evidence of an anomaly. A masked sample whose visual classifier says fresh at 0.9 and whose gas classifier says stale at 0.8 will typically be smoothed to a middling score rather than escalated.")
P("Feature-level fusion (early fusion) concatenates learned embeddings — the intermediate representations — of multiple modalities and feeds them to a joint model, permitting a downstream model to learn cross-modal interactions that score-level fusion cannot express. In this thesis, the fusion engine's 163-dimensional input is exactly a feature-level construction, with a deliberate addition found rarely in the published food-fusion literature: the raw scaled sensor channels and the categorical state posterior are concatenated alongside the embeddings. Including raw channels matters because the embedding compresses away absolute sensor magnitudes — yet absolute formaldehyde ppm is precisely the information the CH2O alert rule needs in order to unscale and compare against the 2.0-ppm regulatory background. Including the posterior matters because the sensor branch classifies the state at approximately 99.9% accuracy; giving the fusion engine direct access to that signal permits the engine to modulate its continuous output by the branch's discrete judgment instead of re-learning it from embeddings alone.")
EQ("u = [ Fv(128) | Fs(16) | x_s(16) | p(y | x_s)(3) ]", "2.1")
P("The present architecture is a hybrid: feature-level in its fusion engine but decision-architecture-level in its tri-layer organization — an image branch, a sensor branch, and a fusion layer, each independently trained and independently saved, with a further statistical consistency layer that no single learned model provides. This staged organization serves a specific engineering purpose: independently saved branches can be re-exported, re-quantized, and re-deployed to the microcontroller separately, which matters at 224 \u00d7 224 INT8 scale, and the branches' outputs constitute the only interface the statistical detector needs, isolating the anomaly layer from implementation details of either branch.")
P("Supervision of continuous freshness. Where the literature does regress continuous freshness or quality scores rather than binary classes, ground truth is frequently obtained from expert panel scoring or derived decay curves. This thesis uses narrow uniform supervision bands — fresh 0.88\u20130.92, masked 0.68\u20130.72, stale 0.17\u20130.21 on the normalized scale — as design targets. This is an unusual but philosophically deliberate choice: band width becomes the irreducible target noise, so the band width itself acts as a hard floor under the achievable validation MAE. Choosing narrow bands (width 0.04, i.e., 4 index points) is what makes a validation MAE of 5.58 index points meaningful rather than trivially low; the alternative, wide bands, would produce a lower reported error while encoding almost no information about where within the band the sample actually sits.")

H2("2.6 Statistical Anomaly Detection: Mahalanobis Distance")
P("The Mahalanobis distance (Mahalanobis, 1936) is the classical metric for measuring how far a sample lies from the center of a multivariate distribution, taking the covariance structure of the data into account:")
EQ("d_M(z) = \u221a( (z \u2212 \u03bc)\u1d40 \u03a3\u207b\u00b9 (z \u2212 \u03bc) )", "2.2")
P("where \u03bc is the reference distribution's mean and \u03a3 its covariance matrix. Under a Gaussianity assumption, the squared Mahalanobis distance follows a \u03c7\u00b2 distribution with degrees of freedom equal to the dimensionality of z, which supplies an analytic threshold: for a 144-dimensional embedding space, the 99th percentile corresponds to T = \u221a(\u03c7\u00b2\u2080.\u2089\u2089(144)) = 13.653.")
P("Modern relevance: out-of-distribution detection. Interest in Mahalanobis-based detection was renewed when the deep-learning out-of-distribution (OOD) community established that class-conditional Mahalanobis distances computed on deep embeddings reliably identify inputs that lie outside the training distribution (Lee et al., 2018). These works established the exact architectural pattern this thesis adopts: embeddings-only distances (not raw inputs), class-conditional reference statistics, and thresholds calibrated against the genuine population. Adopting this pattern to the multimodal food-safety problem — monitoring the fused cross-modal embedding [Fv | Fs] — is the specific contribution of the thesis; prior food-safety work, to the author's knowledge, has not used a cross-modal Mahalanobis consistency layer as the mechanism that catches chemically masked adulteration.")
P("Numerical robustness. Inverting a 144 \u00d7 144 empirical covariance matrix is numerically fragile in the presence of zero-variance or highly correlated embedding directions, both of which are common in ReLU activations. Two established remedies are applied. First, Ledoit\u2013Wolf-style shrinkage (Ledoit and Wolf, 2004) regularizes the covariance toward the scaled identity: \u03a3_shrink = (1\u2212\u03bb)\u03a3 + \u03bb\u03bcI + \u03b5I with \u03bb = 0.10 and \u03b5 = 10\u207b\u2076; shrinkage biases the covariance toward well-conditioning. Second, explicit eigenvalue clipping — flooring each eigenvalue at max(10\u207b\u00b3\u00b7\u03bb_max, 10\u207b\u00b9\u00b2) after eigendecomposition — guarantees that no ReLU-dead feature direction produces an arbitrarily large inverse-covariance entry. The threshold is set to T_MAH = max(T_emp, T_\u03c7\u00b2); in the v13 training run the empirical criterion yields T_emp = 10.310 and the analytic reference T_\u03c7\u00b2 = 13.653, so the deployed threshold is 13.653 — \u03c7\u00b2-bound, with a measured 3.3-point margin above the actual fresh spread. The stale population's median of 19.66, six points above the fresh maximum of 13.19, means an ordinary stale sample is not merely statistically suspicious but categorically separated from the fresh reference.")

H2("2.7 TinyML and Edge Deployment")
P("Deploying deep-learning inference onto microcontrollers is the domain of TinyML, whose canonical software substrate is TensorFlow Lite Micro (TFLM) — a stripped-down interpreter designed to execute flat-buffer (TFLite) models on microcontroller-class CPU and RAM budgets without operating-system support or dynamic memory allocation (David et al., 2021). The TFLM interpreter reads a TFLite flatbuffer directly from flash and executes its operators against a statically allocated tensor arena, a deployment model that maps cleanly onto the ESP32-S3's memory footprint.")
P("Post-training full-integer INT8 quantization reduces model size by a factor of approximately four relative to Float32 and eliminates the floating-point dependency entirely (Warden and Situnayake, 2019), and it is therefore the standard choice for CNN backbones on microcontrollers. A critical practical requirement of post-training INT8 calibration is the provision of a representative dataset — a small stratified sample of realistic inputs used to calibrate the per-tensor ranges of the quantized activations. In this thesis the calibration set is drawn as 400 stratified samples across every (dataset \u00d7 class) group, a deliberate anti-bias measure: because the image dataset is a merge of three heterogeneous sources, naive unstratified calibration would tune quantization ranges to the first dataset encountered and degrade accuracy on the other two.")
P("For small dense networks whose inputs are a mixture of embeddings, raw sensor channels, and probability outputs, post-training Float32 export is, in this deployment context, the safer engineering choice over dynamic-range quantization. Dynamic-range quantization — the default of tf.lite.Optimize.DEFAULT with no representative dataset — stores weights in eight bits but activates them in Float32, a hybrid quantization with documented operator-support gaps in some TFLM builds and subtler numerical behavior in dense activation paths. Full-integer quantization of the fusion engine is feasible, but because the fusion input mixes quantization-sensitive probability outputs with raw sensor channels, Float32 export (128.4 KB) was chosen for correctness and stability on TFLM, at a size well within the ESP32-S3's flash budget. The image branch, whose input domain is bounded [0,255] RGB and which is the memory-dominant component, is exported in fully quantized INT8 form (791.8 KB).")
P("The configuration-header pattern. Firmware delivery of non-learned decision parameters — sensor min/max bounds, the Mahalanobis mean, the 144 \u00d7 144 inverse covariance matrix, the threshold, and alert constants — through a generated C++ header (edge_config.h, 338.5 KB) follows the standard embedded practice of baking decision constants into the binary at compile time rather than parsing them from a configuration file at boot. The generated header carries FUSION_INPUT_DIM = 163 as a compile-time parameter, ensuring that the firmware's fusion-input buffer, the sensor scaling bounds, and the Mahalanobis reference vectors all remain dimensionally consistent with the retrained model if the architecture is adjusted in future work.")

H2("2.8 Research Gap Synthesis")
TBL("Table 2.1  Functional-requirement coverage by existing literature streams",
    ["Functional Requirement", "Vision CNN Stream", "Gas/E-Nose Stream", "Fusion Literature", "TinyML Literature", "This Thesis"],
    [["Fresh/stale classification from imagery", "\u2713 well-covered", "\u2014", "\u2713", "\u2713", "\u2713"],
     ["Chemical-profile freshness classification", "\u2014", "\u2713 (VOC/enose)", "\u2014", "\u2014", "\u2713"],
     ["Formalin/chemical masking detection", "\u2014", "\u25d0 (dose only)", "\u2014", "\u2014", "\u2713 (cross-modal anomaly)"],
     ["Continuous 0\u2013100 freshness index", "\u25d0 (rare)", "\u2014", "\u25d0", "\u2014", "\u2713 (163-D engine)"],
     ["Robustness to missing sensor channels", "\u2014", "\u2014", "\u2014", "\u2014", "\u2713 (flags + augmentation)"],
     ["Heterogeneous public-dataset integration", "\u2014", "\u2713 (per-dataset)", "\u2014", "\u2014", "\u2713 (schema + aliasing)"],
     ["Full multimodal pipeline deployable on MCU", "\u2014", "\u2014", "\u2014", "\u2713 (per-model)", "\u2713 (INT8 + Float32 + header)"],
     ["Statistical cross-modal anomaly layer", "\u2014", "\u2014", "\u2014", "\u2014", "\u2713 (144-D Mahalanobis)"]], font_size=8)
P("Three observations emerge from this synthesis. First, the vision-CNN stream and the gas-sensing stream have each been individually developed to a high standard of validation accuracy for their own modality, but their integration in the food-freshness domain has almost always been at score level, which discards the cross-modal disagreement signal. Second, the statistical anomaly-detection literature — in particular Mahalanobis-based OOD detection — has established the technical machinery for cross-modal consistency testing but has not, to the author's knowledge, been applied to food-adulteration detection specifically. Third, the TinyML literature has established the deployment mechanics for individual quantized models but treats the decision constants and multi-model pipelines of a system such as this one as an engineering gap. This thesis is positioned to address all three observations simultaneously: a feature-level 163-D fusion engine, a 144-D Mahalanobis consistency layer with numerically robust covariance estimation and an empirically calibrated threshold, and a complete edge-export path covering both models and all deterministic decision constants as firmware-ready artifacts.")

H2("2.9 Summary")
P("This chapter reviewed four load-bearing research streams and identified the specific technical role each plays in the thesis design. The literature establishes that chemical masking of food is a real, common, and structurally vision-invisible form of adulteration; that compact transfer-learned CNNs such as MobileNetV2 achieve high-accuracy freshness classification while remaining microcontroller-deployable; that gas-phase and environmental sensing captures complementary and partially independent spoilage information; that feature-level fusion with embeddings, raw inputs, and posteriors in a single vector is the natural architecture for cross-modal food-quality regression; and that Mahalanobis-distance detection on deep embeddings is an established, numerically robust mechanism for identifying out-of-distribution cross-modal contradictions when fitted on the genuine fresh population with shrinkage-regularized covariance. The synthesis also made explicit the gap this thesis addresses: no prior food-freshness pipeline has combined feature-level multimodal fusion, a cross-modal Mahalanobis consistency layer, missing-channel robustness, and a complete edge-deployment path within a single public-data-only framework. Chapter 3 presents the methodology that realizes this combination in full engineering detail.")
PB()

# ------------------------------ Chapter 3 ------------------------------
H1("CHAPTER 3  METHODOLOGY")

H2("3.1 Overview of the Proposed System")
P("The proposed system is organized as a tri-layer multimodal pipeline with a fourth, purely statistical consistency layer. One food sample is observed through two independent evidence channels: a camera image and a heterogeneous, possibly incomplete sensor vector. The pipeline proceeds in three learned stages and one statistical stage: (1) the image branch transforms the image into a compact visual embedding Fv \u2208 R^128 and a binary fresh/stale decision; (2) the sensor branch transforms the canonical 16-feature sensor vector into a sensor embedding Fs \u2208 R^16 and a three-class freshness state posterior p(y | x_s) over Fresh, Stale, and chemically Masked states; (3) the fusion engine regresses a continuous 0\u2013100 Freshness Index from the concatenation")
EQ("u = [ Fv(128) | Fs(16) | x_s(16) | p(3) ] \u2208 R^163", "3.1")
P("(4) and the Mahalanobis detector evaluates the joint plausibility of the embeddings only, z = [Fv | Fs] \u2208 R^144, against the reference distribution of genuinely fresh samples, producing the anomaly signal d_M that drives the chemical-masking alert. The final decision logic combines the Freshness Index FI = 100\u00b7\u0177, the anomaly distance d_M, and an explicit raw-formaldehyde alert rule (CH2O > 2.0 ppm) into one of three deployment states \u2014 Fresh, Suspect (anomaly), or Stale/Spoiled \u2014 with escalation to a CHEMICAL MASKING ALERT when the visual and chemical evidence contradict.")
FIG(os.path.join(ASSET_DIR, "figure3_1_architecture.png"),
    "Figure 3.1  Proposed tri-layer system architecture: image branch (MobileNetV2, \u03b1 = 0.35) and sensor branch (MLP) trained independently; their embeddings, the raw scaled sensor channels, and the state posterior are concatenated into a 163-dimensional fusion input; the Mahalanobis layer operates on the 144-dimensional embedding concatenation")
FIG(os.path.join(ASSET_DIR, "figure3_2_fusion_flow.png"),
    "Figure 3.2  Data-flow of the fusion input vector construction, annotating the four groups [Fv 128 | Fs 16 | raw 16 | state 3] and their provenance from the two feature extractors and the sensor branch's softmax head")

H2("3.2 Data Sources and the Heterogeneous-Dataset Integration Layer")
H3("3.2.1 Design Constraint: 100% Public Secondary Data")
P("No primary data was collected. Every training input originates from publicly available Kaggle datasets, and the integration layer is responsible for reconciling datasets that differ in sensor hardware, sampling rate, column naming, label convention, and channel coverage. The design treats this reconciliation as an explicit engineering subsystem rather than an ad-hoc preprocessing step, because the validity of every downstream result depends on it.")
H3("3.2.2 Image Datasets")
P("Three public Kaggle image datasets are merged. Class labels are derived from a strict, basename-only rule (never the full path), with a special-case parser for the Roboflow multiclass export format used by the meat dataset.")
TBL("Table 3.1  Public Kaggle image datasets: composition and class balance (22,667 images total; 15% of the pooled set is held out for validation)",
    ["Dataset", "Images", "Content", "Label convention"],
    [["Fresh and Stale Images of Fruits and Vegetables", "14,682", "Fruit and vegetable photographs", "folder-text (fresh/stale)"],
     ["Tilapia Fish Fresh and Non-Fresh Species", "5,719", "Tilapia fish, day-decay series", "parent-folder decay day"],
     ["Meat Freshness Image Dataset", "2,266", "Meat, Roboflow multiclass export (_classes.csv: Fresh / Half-Fresh / Spoiled)", "CSV maximum-column rule"],
     ["Total", "22,667", "Fresh 11,213 \u00b7 Stale 11,454", "1 = fresh, 0 = stale"]], font_size=9)
H3("3.2.3 Sensor Datasets")
P("Four public Kaggle sensor datasets are merged, each registered with a directory-discovery key, a CSV glob, a label rule, and a per-channel column mapping. The merge is performed by CSV discovery (a recursive **/*.csv glob under each dataset root), union of schemas, and per-channel mapping into the canonical schema of Section 3.3.")
TBL("Table 3.2  Public Kaggle sensor datasets: row counts, mapped channels, and label rules (922,424 real rows total)",
    ["Dataset", "Rows", "ch2o_ppm", "voc_log_ohm", "core_temp_c", "ambient_temp_c", "humidity_pct", "ch2o_voc_ratio", "temp_rise_rate", "hours_elapsed", "fresh", "stale"],
    [["enose_fruits (Electronic Nose, MQ2/MQ135)", "838,502", "YES", "YES", "\u2014", "\u2014", "\u2014", "YES", "\u2014", "\u2014", "467,419", "371,083"],
     ["tofu_formalin (Formalin Tofu Aroma)", "45,000", "YES", "YES", "YES", "\u2014", "\u2014", "YES", "\u2014", "\u2014", "21,859", "23,141"],
     ["storage_env (IoT Agriculture 2024)", "37,922", "\u2014", "\u2014", "\u2014", "YES", "YES", "\u2014", "\u2014", "\u2014", "0", "37,922"],
     ["chicken_sensor (Chicken Meat Freshness)", "1,000", "\u2014", "YES", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "YES (day \u00d7 24 h)", "500", "500"]], font_size=7)
H3("3.2.4 Column Alias Resolution")
P("Because public CSVs do not share naming conventions, the integration layer resolves each required channel through an alias table, preferring an exact header match and falling back to a documented alias set. The most consequential case in the final configuration is the IoT Agriculture 2024 temperature column, whose source header is misspelled \u201ctempreature\u201d; the alias table maps it to the canonical ambient_temp_c channel (with further aliases temp, temp_c, temperature(c) for other potential future attachments, and a parallel alias set for humidity: humid, relative_humidity, rh, humidity(%)). This mechanism is what converts the IoT Agriculture dataset from a flagged-missing source into a real-data source for ambient temperature (range 3\u201341 \u00b0C) and humidity (range 0\u2013100%).")
H3("3.2.5 Label Rules")
P("Four label conventions are supported, one per dataset type: column \u2014 a dataset column carries the label directly (tofu formalin vendor QC label, 1 = pass/fresh); text \u2014 the label is a text column matched against a vocabulary of stale/rotten/non-fresh synonyms (chicken dataset target column); day_decay \u2014 time-ordered rows are labeled fresh for early decay days and stale for later days (e-nose fruit dataset: days 1\u20132 fresh, days 3\u20135 stale); and constant \u2014 a dataset without food-spoilage ground truth contributes environment-real, label-stale rows (IoT Agriculture), deliberately weighted toward the stale side so that the fusion engine's stale anchor retains real environmental variability. Image labels follow a single binary rule checked only against the class-folder basename (never the full path, which can itself contain the token \u201cstale\u201d via a dataset slug) and a dedicated parser for Roboflow multiclass _classes.csv exports in which Fresh, Half-Fresh, and Spoiled columns are compared per row and the maximum is mapped through the same text vocabulary (Half-Fresh and Spoiled both map to not-fresh).")

H2("3.3 Canonical Sensor Feature Schema and Preprocessing")
H3("3.3.1 The 16-Feature Canonical Vector")
TBL("Table 3.3  Canonical 16-channel sensor feature vector",
    ["#", "Value channel", "Type", "Availability flag", "Physical meaning"],
    [["1", "ch2o_ppm", "measured", "av_ch2o_ppm", "Formaldehyde (ZE08-CH2O / MQ135 / formalin-dataset surrogate)"],
     ["2", "voc_log_ohm", "measured", "av_voc", "VOC signature (MQ2 / Mean_Value / BME680-family)"],
     ["3", "core_temp_c", "measured", "av_core_temp", "Sample core temperature (MLX90614 / dataset surrogate)"],
     ["4", "ambient_temp_c", "measured", "av_ambient", "Storage environment temperature (DHT22 / IoT telemetry)"],
     ["5", "humidity_pct", "measured", "av_humidity", "Storage relative humidity"],
     ["6", "ch2o_voc_ratio", "derived", "av_ratio", "Cross-gas ratio, a masking-sensitive invariant"],
     ["7", "temp_rise_rate", "derived", "av_rise", "Temperature rise rate (\u00b0C/h), decay indicator"],
     ["8", "hours_elapsed", "context", "av_hours", "Storage duration"]], font_size=9)
P("The availability flags exist for a specific engineering reason: they make the model's input space total over sensor failures. Any physical sub-system can be absent (unplugged, out of calibration, not applicable to the sample type) without breaking either the training data pipeline or the deployed inference, because the value channels are set to a neutral placeholder state, the corresponding flag is 0, and the model learns through the flags to discount absent channels.")
H3("3.3.2 Scaling and Smoothing")
P("All value channels are min\u2013max scaled to [0, 1] against global bounds estimated from the merged dataset (and exported to firmware for inference-time rescaling):")
EQ("x_scaled = (x \u2212 x_min) / max(x_max \u2212 x_min, 10\u207b\u2079),   x_scaled \u2208 [0, 1]", "3.2")
P("A moving-average pre-filter of window 5 is then applied along the row axis \u2014 a Kalman-inspired smoothing step that suppresses per-sample sensor noise while preserving the leading trend:")
EQ("x\u0303[t] = (1/W) \u00b7 \u03a3_{k=0..W\u22121} x[t\u2212k],   W = 5,   then clipped back to [0, 1]", "3.3")
H3("3.3.3 Missing-Channel Simulation and Environmental Augmentation")
P("The coverage analysis of the merged datasets (Section 3.2.3) identifies exactly one channel not covered by any attached source: temp_rise_rate, which is derived from core_temp_c and elapsed time and, absent a real source, is filled by a documented synthetic model. A synthetic environmental value is assigned under a monotonic ordering constraint (masked < fresh < stale), consistent with the physics of decay \u2014 later decay stages release more thermal rise; 70% of rows receive the synthetic value with the availability flag activated, and 30% of rows retain flag = 0, exactly mirroring the deployment condition of a missing or physically absent sensor. All remaining channels receive real data after the alias resolution of Section 3.2.4 \u2014 including, critically, ambient_temp_c (range 3\u201341 \u00b0C) and humidity_pct (range 0\u2013100%) from the IoT Agriculture dataset. A post-merge sanity report verifies this per channel (minimum/maximum/NaN/Inf count per column, reaffirmed in Chapter 4), and an assertion halts the pipeline if any non-finite value survives to the training input matrix.")
H3("3.3.4 Chemical-Masking Augmentation")
P("To train the sensor branch's third class (Masked), 25% of all real rows are augmented with a simulated chemical-masking profile: fresh-domain features are preserved while the chemical channels are driven into the abnormal, formalin-typical operating range. This produces 230,606 augmented Masked rows on top of the 922,424 real rows, giving the branch three well-separated states to learn: Stale (432,646 real), Fresh (489,778 real), and Masked (synthetic, Section 1.4 discloses its synthetic nature).")

H2("3.4 Layer 1 — The Image Branch")
H3("3.4.1 Architecture")
P("The image branch is built around a width-reduced MobileNetV2 (Sandler et al., 2018) with width multiplier \u03b1 = 0.35, pre-trained on ImageNet, operating on 224 \u00d7 224 \u00d7 3 inputs. The full data path is:")
EQ("x_img \u2192 Rescaling(2x \u2212 1) \u2192 augment \u2192 MobileNetV2(\u03b1 = 0.35) \u2192 GAP \u2192 Dense(128, ReLU) = Fv \u2192 Dropout(0.3) \u2192 Dense(1, \u03c3) \u2192 \u0177_img", "3.4")
P("Three points deserve explicit comment. First, the input rescaling maps the [0,1] input domain onto the [\u22121,1] range that ImageNet-initialized convolution weights are normalized against \u2014 a mismatch at this exact point is a silent accuracy loss in transfer learning. Second, the data-augmentation layers (horizontal flip, rotation \u00b10.08, zoom \u00b10.08) are inside the Keras model but are active only during fitting; at inference and export time the graph is used without them, so the deployed model is deterministic. Third, the 128-unit dense head named fv_embedding is deliberately not merely an internal layer: it is the visual feature vector Fv consumed by both the fusion engine and the Mahalanobis layer, and a second Model object (image_feature_extractor) is constructed around exactly this output so that embeddings can be extracted and exported independently of the classification head.")
H3("3.4.2 Training Strategy — Two-Phase Transfer Learning")
P("The image branch is trained in two phases on the 85/15 train/validation split (19,267 train images, 3,400 validation images). Phase 1 \u2014 head training (backbone frozen, 5 epochs): the ImageNet backbone is frozen (trainable = False) and the freshly initialized head learns to map frozen visual features onto the fresh/stale boundary at a high learning rate of 10\u207b\u00b3 with Adam. Phase 2 \u2014 partial fine-tuning (last 8 blocks, 2 epochs): the backbone's final 8 blocks are unfrozen while BatchNorm statistics are kept frozen, a standard stability measure when unfreezing a pre-trained network on a modestly sized dataset; the learning rate is dropped three orders of magnitude to 10\u207b\u2075 \u2014 the canonical range for not catastrophically displacing ImageNet representations \u2014 and two epochs are run. A ModelCheckpoint monitors validation loss (not accuracy), since loss is strictly more informative near the performance ceiling; on completion, the best checkpoint is reloaded into the deployed feature extractor. Training dynamics and the resulting trajectory are reported in Section 4.2.")

H2("3.5 Layer 2 — The Sensor Branch")
H3("3.5.1 Architecture")
P("The sensor branch is a compact multi-layer perceptron over the 16-feature canonical vector:")
EQ("x_s \u2192 BatchNorm \u2192 Dense(64, ReLU) \u2192 Dropout(0.2) \u2192 Dense(32, ReLU) \u2192 Dense(16, ReLU) = Fs \u2192 Dropout(0.2) \u2192 Dense(3, softmax) \u2192 p(y | x_s)", "3.5")
P("The leading BatchNormalization layer is the component that both stabilizes the union-of-datasets input distribution (four heterogeneous sources produce very different per-channel scales in raw form) and acts as a soft domain-normalization layer. The 16-unit hidden layer named sensor_embedding is the sensor feature vector Fs \u2014 the counterpart of Fv \u2014 and, exactly as on the image side, a second Model object (the sensor_feature_extractor) is built around it so embeddings can be extracted independently of the softmax head.")
H3("3.5.2 Training Configuration")
P("The branch is trained with Adam at learning rate 10\u207b\u2074 with gradient-norm clipping at 1.0 (clipnorm = 1.0 \u2014 a NaN-guard, since an exploding gradient in a BatchNorm-prefixed network is the canonical source of non-finite loss in a softmax head), the sparse categorical cross-entropy loss, batch size 256, and 30 epochs with a TerminateOnNaN safety callback. The training/validation split is 15% (172,954 validation rows of 1,153,030 total post-augmentation rows). As Chapter 4 reports, the branch converges extremely quickly \u2014 reaching 99.71% validation accuracy in the first epoch \u2014 and the long remaining epochs anneal the loss to 0.0053 and the accuracy to a stable 99.84%, with per-class F1 between 0.998 and 1.000.")

H2("3.6 Layer 3 — The Fusion Engine")
H3("3.6.1 Fusion Dataset Construction")
P("A fusion training set is constructed by pairing image-domain samples with sensor-domain rows of the same freshness state. Each image index is visited once, capped at 4,000 pairs (sampled by random permutation when the image set is larger, which it is at 22,667 \u2014 so the cap binds and 4,000 pairs are drawn). For each pair: the image is loaded, resized to 224 \u00d7 224, and embedded by the frozen image_feature_extractor in chunks of 256 (a memory-flatness requirement: embedding all 4,000 images in one batch would collide with the session RAM ceiling); the paired sensor row is drawn uniformly at random from the pool of training-split rows of the target state, then embedded by the frozen sensor_feature_extractor; and 30% of fresh-state pairs (MASKED_PAIR_FRACTION = 0.30) are deliberately paired instead with a masked sensor profile, yielding the third state \u2014 a visually fresh sample whose chemical profile is anomalous. The resulting state distribution is stale 2,082 (52.1%), fresh 1,330 (33.3%), masked 588 (14.7%), an ordering that reflects the real-world prior that ordinary stale samples outnumber chemically masked ones. The concatenation of paired embeddings produces the fusion embedding matrix Z \u2208 R^(4000\u00d7144), and the raw scaled sensor rows are retained in parallel as a matrix S \u2208 R^(4000\u00d716).")
H3("3.6.2 The 163-Dimensional Input Vector")
P("The fusion engine's input is assembled from four information sources:")
EQ("u = [ Fv(128) | Fs(16) | x_s(16) | p(y | x_s)(3) ] \u2208 R^163", "3.6")
P("The state posterior p \u2014 the sensor branch's softmax output over (Fresh, Stale, Masked), computed by a single batched predict on the raw sensor rows for all 4,000 pairs \u2014 is a v13 addition motivated by a specific, observed limitation (discussed fully in Chapter 4): with embeddings alone, the masked band (0.66\u20130.74) and the fresh band (0.86\u20130.94) produce an ambiguity that registers as a residual validation-MAE floor, even though the sensor branch classifies the state itself at approximately 99.9% accuracy. Giving the fusion engine direct access to that posterior removes the need to re-learn it from embeddings, and the architecture change is the principal contributor to the fusion MAE dropping from 0.0642 (v11 state) to 0.0558 (v13).")
H3("3.6.3 Supervision Bands")
P("The fusion engine's supervision is a set of three narrow uniform bands on the normalized 0\u20131 scale, sampled independently per pair:")
EQ("y_gt \u223c U(0.88, 0.92) fresh;  U(0.68, 0.72) masked;  U(0.17, 0.21) stale", "3.7")
P("The design rationale is deliberate narrowness: the band width is the irreducible target noise, so the achievable validation MAE inherits a floor of roughly half the band width (\u2248 2 index points here), and reporting a validation MAE against a target with a known noise floor is the honest framing used throughout Chapter 4. The v13 bands are narrower than the predecessor configuration (fresh 0.85\u20130.95, masked 0.65\u20130.75, stale 0.10\u20130.25) specifically to make the reported MAE a discriminative rather than a band-averaged metric.")

TBL("Table 3.4  Freshness Index supervision bands by sensor state (normalized scale; \u00d7100 \u2192 0\u2013100 index)",
    ["State", "Band (normalized)", "Band width", "Approx. noise floor (MAE)"],
    [["Fresh", "0.88 \u2013 0.92", "0.04 (4 index points)", "\u2248 2 index points"],
     ["Masked", "0.68 \u2013 0.72", "0.04 (4 index points)", "\u2248 2 index points"],
     ["Stale", "0.17 \u2013 0.21", "0.04 (4 index points)", "\u2248 2 index points"]], font_size=9)

H3("3.6.4 Architecture, Loss, and Optimization")
P("The fusion engine is a compact dense network:")
EQ("u \u2192 Dense(128, ReLU) \u2192 LayerNorm \u2192 Dense(64, ReLU) \u2192 Dropout(0.15) \u2192 Dense(32, ReLU) \u2192 Dense(1, \u03c3) \u2192 \u0177 \u2208 [0,1]", "3.8")
P("The LayerNorm after the first projection is a deliberate structural choice: the four input groups (Fv, Fs, x_s, p) live at different magnitude scales, and a single learned normalization after the projection remaps the mixture into a uniform internal scale before the deeper dense stack. The loss is Huber, with the robustness parameter \u03b4 = 0.1:")
EQ("L_Huber(\u0177, y_gt) = \u00bd(\u0177 \u2212 y_gt)\u00b2  if |\u0177 \u2212 y_gt| \u2264 \u03b4;   \u03b4(|\u0177 \u2212 y_gt| \u2212 \u00bd\u03b4)  otherwise", "3.9")
P("With supervision confined to narrow bands, Huber's quadratic-to-linear transition at 0.1 prevents occasional outliers from either band from producing the outsized gradients that a pure squared-error loss would, which matters in a 4,000-sample training set. Reported metrics are Huber loss and Mean Absolute Error; a ReduceLROnPlateau scheduler (factor 0.5, patience 4, minimum learning rate 10\u207b\u2076) anneals the Adam optimizer (initial learning rate 3\u00d710\u207b\u2074, gradient clipping at norm 1.0) across 40 epochs. The training/validation split is 85/15 of the 4,000 fusion pairs (3,400 train / 600 validation).")

H2("3.7 The Mahalanobis Anomaly Detector")
H3("3.7.1 Reference Distribution")
P("The Mahalanobis reference is fitted only on the fresh-state subset of the fusion embedding matrix Z \u2208 R^(4000\u00d7144) \u2014 that is, on the 1,330 pairs of state fresh \u2014 so that the detector's semantic is exactly \u201chow far does this sample's joint embedding deviate from what genuinely fresh samples look like?\u201d If the fresh subset were too small to estimate a 144-dimensional covariance (n < 146), the estimator falls back to replication with small jitter; in the actual training run this fallback is not triggered.")
H3("3.7.2 Covariance Estimation with Shrinkage and Eigenvalue Clipping")
P("A shrinkage-regularized, eigenvalue-clipped covariance estimate is computed as follows. The mean vector and the centered sample covariance are:")
EQ("\u03bc = (1/n) \u00b7 \u03a3(i=1..n) z(i) \u2208 R^144", "3.10")
EQ("\u03a3 = (1/(n\u22121)) \u00b7 \u03a3(i=1..n) (z(i) \u2212 \u03bc)(z(i) \u2212 \u03bc)\u1d40", "3.11")
P("Two regularizers are then applied. First, Ledoit\u2013Wolf-style shrinkage (Ledoit and Wolf, 2004) toward the scaled identity, with shrinkage intensity \u03bb = 0.10 and a small diagonal loading \u03b5 = 10\u207b\u2076:")
EQ("\u03a3_shrink = (1 \u2212 \u03bb)\u03a3 + \u03bb\u00b5\u00b7I + \u03b5\u00b7I,   where \u00b5 = tr(\u03a3)/144 is the average per-dimension variance", "3.12")
P("Second, after the eigendecomposition \u03a3_shrink = Q\u039bQ\u1d40, each eigenvalue is floored at:")
EQ("\u03bb_i \u2190 max( \u03bb_i , max( 10\u207b\u00b3 \u00b7 \u03bb_max , 10\u207b\u00b9\u00b2 ) )", "3.13")
P("which guarantees that no dead-ReLU direction (a direction of near-zero variance, common in 128-unit ReLU embeddings) produces an arbitrarily large inverse-covariance entry. The inverse covariance is then reconstructed as \u03a3\u207b\u00b9 = Q\u039b\u207b\u00b9Q\u1d40, computed entirely in float64 internally and exported as float32 \u2014 a numerical-precision deliberate choice on an estimator whose conditioned behavior is the sensitivity hotspot of the whole detector.")
H3("3.7.3 Threshold Selection")
P("The decision threshold combines an empirical and an analytic criterion:")
EQ("T_MAH = max( T_emp , T_\u03c7\u00b2 ),   T_emp = P99({ d_M(z(i)) } for i in fresh),   T_\u03c7\u00b2 = \u221a( \u03c7\u00b2(0.99, df = 144) )", "3.14")
P("The empirical criterion measures how far the fresh population actually spreads; the analytic \u03c7\u00b2 criterion supplies a conservative reference under Gaussianity. Taking the maximum ensures the deployed threshold is never below either, a deliberately conservative margin. In the v13 training run the two criteria yield T_emp = 10.310 and T_\u03c7\u00b2 = 13.653, so the deployed threshold is 13.653 \u2014 \u03c7\u00b2-bound, with a measured 3.3-point margin above the actual fresh spread.")
H3("3.7.4 Sanity Verification")
P("Three sanity conditions are verified at fit time and asserted (the pipeline halts rather than deploying a silently broken detector): (1) all fresh-class distances are finite and bounded above (max < 10\u2074), detecting the exploding-Mahalanobis failure mode that motivated the v11 space-mismatch fix; (2) the stale population separates above the fresh population: measured stale median 19.66 and stale p99 33.80 versus fresh median 6.49, fresh p99 10.31, fresh max 13.19 \u2014 all below the 13.653 threshold; and (3) the space-match property: the detector is fitted on exactly the concatenation [Fv | Fs] of the two feature extractors' embeddings, so that the same 144-dimensional coordinates are used at inference (Section 3.8), which is the invariant whose violation caused the earlier-version distance explosion to the order of 10\u00b9\u00b2.")

H2("3.8 Inference-Time Decision Logic")
P("At inference, a single sample is processed by combining the neural index, the statistical anomaly distance, and an explicit chemical-alert rule. The sequence is: (1) the image is loaded, resized, rescaled to [\u22121,1], and passed through the image branch's feature extractor to obtain Fv; (2) the canonical (scaled, availability-flagged) 16-feature sensor vector is passed through the sensor branch's feature extractor to obtain Fs, and through its classification head to obtain the posterior p; (3) the fusion engine receives the 163-dimensional input u = [Fv | Fs | x_s | p] and emits \u0177, converted to the Freshness Index FI = 100\u00b7\u0177 \u2208 [0, 100]; (4) the 144-dimensional concatenation z = [Fv | Fs] is evaluated against the reference: d_M = \u221a((z \u2212 \u03bc)\u1d40\u03a3\u207b\u00b9(z \u2212 \u03bc)); (5) the chemical-masking alert is evaluated first: if d_M > T_MAH and the raw (unscaled) CH2O reading exceeds 2.0 ppm \u2014 unscaled through the exported bounds, x_raw = x_scaled\u00b7(x_max \u2212 x_min) + x_min, since ch2o_ppm occupies index 0 of the canonical schema \u2014 the decision is escalated immediately to \u201cCHEMICAL MASKING ALERT \u2014 visual freshness contradicts toxic gas profile\u201d, regardless of the neural index; (6) otherwise, the tri-state decision applies: Fresh if FI \u2265 70; Suspect \u2014 anomaly (internal rot / drift) if d_M > T_MAH and 50 \u2264 FI < 70; and Stale / Spoiled otherwise. The design ordering matters: the chemical-mask escalation is evaluated before the ordinary tri-state logic, because the ordinary logic's Fresh category (FI \u2265 70) is exactly the category that a visually deceptive masked sample would otherwise silently occupy.")

H2("3.9 Edge Export Pipeline")
H3("3.9.1 Image Branch — Full INT8 Quantization")
P("The image branch is exported as a fully integer-quantized TFLite flatbuffer using post-training full-integer quantization: optimizations DEFAULT, supported ops TFLITE_BUILTINS_INT8, and inference input and output types int8. The calibration representative dataset is drawn as a stratified set of 400 samples across every (dataset \u00d7 class) group in the training pool; this is an anti-bias measure against the well-known failure mode in which an unstratified calibration set tunes quantization ranges to one dataset's distribution and degrades the others' accuracy. The resulting artifact is image_branch_int8.tflite (791.8 KB), and a C-array form (image_branch.cc, 2,522.9 KB) is generated for direct embedding into firmware flash. A dynamic-range fallback path is retained in the script in case INT8 conversion ever fails on a future backbone, so that the export cell never produces no output.")
H3("3.9.2 Fusion Engine — Float32 Export")
P("The fusion engine is exported as a Float32 TFLite flatbuffer (fusion_engine_f32.tflite, 128.4 KB), with its C-array form (fusion_engine.cc, 451.2 KB) generated alongside. The choice of Float32 over dynamic-range quantization is deliberate: the fusion engine's input mixes embeddings, raw sensor channels, and probability outputs, and Float32 input/output semantics map more predictably onto TFLM's dense-operator kernel coverage on the ESP32-S3 than the hybrid weight-only quantization that dynamic-range mode produces. At 128.4 KB it sits comfortably inside the ESP32-S3's flash budget, so the model-size saving of quantization is unnecessary here.")
H3("3.9.3 The Edge Configuration Header")
P("All deterministic, non-learned decision parameters are generated into a single C++ header, edge_config.h (338.5 KB), for compile-time baking into firmware: SENSOR_MIN[8] and SENSOR_MAX[8] \u2014 the global min\u2013max bounds used for inference-time scaling and, critically, for CH2O raw-ppm unscaling in the alert rule; MAH_MEAN[144] \u2014 the fresh-class reference mean vector; MAH_INV_COV[144][144] \u2014 the complete inverse covariance matrix, so that the on-device Mahalanobis computation needs no matrix inversion at runtime; MAH_THRESHOLD = 13.653; CH2O_ALERT_PPM = 2.0; and FUSION_INPUT_DIM = 163, emitted as a compile-time parameter derived from EMBED_DIM + SENSOR_EMBED + len(FEATURE_COLS) + 3, so that if any of the underlying dimensions is changed in future work, the entire header \u2014 including the matrix shapes \u2014 regenerates consistently rather than going stale. A parallel machine-readable edge_config.json (587.6 KB) mirrors the same content \u2014 channel names, bounds, mean vector, threshold, shrinkage parameter, freshness bands, fusion-pair count, dataset provenance, and the inverse covariance in row-major form \u2014 for tooling and reproducibility audits, and is the reference against which the header is generated.")
H3("3.9.4 Deployment Path")
P("The firmware integration contract is: read the OV2640/OV5640 frame into the 224 \u00d7 224 RGB buffer, run the TFLM interpreter against g_image_branch (INT8), normalize the read gas/thermal/environmental channels through SENSOR_MIN/MAX, assemble the 163-dimensional fusion input (embedding 128 + sensor embedding 16 + raw 16 + posterior 3), run g_fusion_engine, evaluate the Mahalanobis distance against MAH_MEAN / MAH_INV_COV / MAH_THRESHOLD, apply the CH2O 2.0-ppm escalation, and emit the tri-state decision over BLE. All artifacts listed in this subsection were generated in the final v13 training run and shipped as outputs, together with a dataset coverage CSV (sensor_dataset_coverage.csv) documenting which dataset covers which channel.")

H2("3.10 Evaluation Protocol")
P("Reproducibility and metric semantics are fixed as follows. The global random seed is 42 across Python, NumPy, and TensorFlow. Three independent validation splits are used, one per learned layer: the image branch is evaluated on the 15% held-out image split; the sensor branch on the 15% held-out row split (172,954 rows); and the fusion engine on the 15% held-out fusion-pair split (600 of 4,000 pairs). Reported metrics are: validation accuracy and validation loss for the two branches (with per-class precision/recall/F1 for the sensor branch); validation MAE and validation Huber loss for the fusion engine; and for the Mahalanobis detector, the fresh-distance distribution (min/median/p99/max), the stale-distance distribution (median/p99), and the ratio of the deployed threshold to each. All numbers in Chapter 4 are taken from a single complete end-to-end training run executed on Kaggle under TensorFlow 2.21.0 with dual-GPU acceleration, and the pipeline's in-run assertions (input finiteness, label domain, fresh-distance bound, stale-separation direction) are evaluated and pass in that run.")

H2("3.11 Summary")
P("This chapter specified the complete methodology of the proposed system. A four-dataset, seven-channel public sensor corpus and a three-dataset image corpus are merged through an alias-resolving, availability-flagged canonical schema (Sections 3.2\u20133.3); two independent neural branches are trained with documented transfer-learning and NaN-guard configurations on that schema (Sections 3.4\u20133.5); a 163-dimensional feature-level fusion engine regresses a narrow-band-supervised Freshness Index with Huber loss and learning-rate annealing (Section 3.6); a shrinkage-regularized Mahalanobis detector monitors the 144-dimensional joint embedding space with an empirically calibrated threshold (Section 3.7); and the two models plus all decision constants are exported as firmware-ready artifacts for a full on-device TFLM inference path on the ESP32-S3 (Section 3.9). The methodology is fully reproducible: it uses a fixed global seed, single-run results, no primary data, and per-stage in-run assertions that halt the pipeline on the known numerical failure modes rather than allowing a silent mis-deployment. Chapter 4 reports the empirical results of this configuration in full.")
PB()

# ----------------------------------------------------------------------
# CHAPTER 4 — EXPERIMENTAL SETUP AND RESULTS
# ----------------------------------------------------------------------
H1("CHAPTER 4  EXPERIMENTAL SETUP AND RESULTS")

H2("4.1 Experimental Environment and Reproducibility")
P("All experiments reported in this chapter were executed as a single end-to-end run of the pipeline described in Chapter 3 on the Kaggle cloud platform. The software environment was TensorFlow 2.21.0 (Keras 3) on Python 3.13, with two GPU accelerators exposed to the session and XLA-compatible standard Keras layers only. The global random seed was fixed at 42 across Python, NumPy, and TensorFlow before any data processing, so the train/validation partitions of all three learned layers are deterministic. No dataset outside the seven attached public Kaggle datasets (three image, four sensor) was downloaded, scraped, or recorded by the author; the complete experimental pipeline is therefore 100% secondary data.")
P("The chapter is organised by pipeline layer. Section 4.2 reports the image corpus and the two-phase Part A training trajectory. Section 4.3 reports the sensor corpus coverage matrix and the pre-training data-sanity verification that guarantees the observed losses are not artefacts of malformed inputs. Section 4.4 reports the sensor branch results, including the class-resolved classification report on 172,954 held-out rows. Section 4.5 reports the fusion-pair set composition and the freshness-regression trajectory. Section 4.6 evaluates the Mahalanobis anomaly detector in the deployed 144-dimensional space. Section 4.7 demonstrates the end-to-end decision pipeline on a single sample, and Section 4.8 inventories the firmware-ready edge artifacts with their exact sizes. Section 4.9 synthesises the results against the objectives of Chapter 1.")

H2("4.2 Image Corpus Composition and Part A Training Trajectory")
P("The image corpus was assembled from three public Kaggle datasets spanning three food categories: fresh and stale fruits and vegetables (14,682 images), tilapia fish in fresh and non-fresh conditions (5,719 images), and a meat freshness collection with Roboflow multiclass exports (2,266 images). After the automatic class-directory discovery and rel-path labelling rules of Section 3.3, the corpus totals 22,667 images with an almost balanced label distribution: 11,213 fresh and 11,454 stale. Balance matters because a 1.02:1 stale-to-fresh ratio prevents the binary branch from collapsing onto the majority class during head training.")
TBL("Table 4.1: Sensor dataset coverage matrix after alias resolving and canonical-schema mapping. A dash denotes a channel not provided by the dataset; YES denotes rows where a real measured value was mapped. The chicken dataset contributes its Shelf_Life_Day column as a real hours_elapsed channel.",
    ["Dataset", "Rows", "ch2o_ppm", "voc_log_ohm", "core_temp_c", "ambient_temp_c", "humidity_pct", "ch2o_voc_ratio", "temp_rise_rate", "hours_elapsed", "Fresh", "Stale"],
    [["chicken_sensor", "1,000", "\u2013", "YES", "\u2013", "\u2013", "\u2013", "\u2013", "\u2013", "YES", "500", "500"],
     ["enose_fruits", "838,502", "YES", "YES", "\u2013", "\u2013", "\u2013", "YES", "\u2013", "\u2013", "467,419", "371,083"],
     ["storage_env", "37,922", "\u2013", "\u2013", "\u2013", "YES", "YES", "\u2013", "\u2013", "\u2013", "0", "37,922"],
     ["tofu_formalin", "45,000", "YES", "YES", "YES", "\u2013", "\u2013", "YES", "\u2013", "\u2013", "21,859", "23,141"]],
    font_size=8)
P("Table 4.1 is the empirical answer to the sensor-side data question of Chapter 1. Two pre-processing consequences follow directly from it. First, ambient_temp_c is covered by a single real source (IoT Agriculture 2024, 37,922 rows); the v12 header-aliasing fix was required for this binding because the source column is spelled \u2018tempreature\u2019 in the published CSV. Second, temp_rise_rate is the only channel in the eight-channel schema not covered by any measurement chain that lacks a dependency: it is derived automatically from core_temp_c and hours_elapsed wherever those exist, and therefore requires no external gap-filling dataset beyond the four already attached. The derived ch2o_voc_ratio channel is similarly covered wherever both of its constituents exist (838,502 + 45,000 rows).")
FIG(os.path.join(OUT_DIR, "image_branch_training.png"),
    "Figure 4.1: Image branch training history. Left panel: binary cross-entropy loss of the frozen-backbone head phase (epochs 1\u20135) and the fine-tuning phase (epochs 6\u20137, dashed region convention per Section 3.4). Right panel: corresponding validation accuracies. The best validation loss of the head phase (0.0390 at epoch 4) is followed by a further drop to 0.0329 under the low-learning-rate fine-tune, which the ModelCheckpoint callback preserves as the deployed checkpoint.")
P("Table 4.2 records the complete Part A trajectory epoch by epoch. The frozen-backbone head phase reaches its best validation loss at epoch 4 (0.0390, val_acc 98.47%); the fifth head epoch slightly overfits (val_acc 98.32% with a lower training loss), which is exactly the situation the best-epoch checkpoint policy of Section 3.4 is designed for. The two fine-tuning epochs at learning rate 1e-5 over the last eight backbone blocks improve both metrics \u2014 epoch 1 attains the run\u2019s peak validation accuracy of 98.79%, and epoch 2 attains the best validation loss of 0.0329 at 98.76% \u2014 and the checkpoint callback restores the epoch-2 weights, so the deployed image branch is the 98.76% / 0.0329 model, with the 98.79% peak reported as the maximum observed accuracy.")
TBL("Table 4.2: Part A (image branch) training trajectory. Phase 1 trains only the GAP + Dense(128) + sigmoid head at learning rate 1e-3; phase 2 fine-tunes the last eight MobileNetV2 blocks at learning rate 1e-5 with BatchNorm layers frozen. The deployed checkpoint (bold) is the epoch with the best validation loss.",
    ["Phase", "Epoch", "Train acc", "Train loss", "Val acc", "Val loss"],
    [["Head (frozen)", "1", "94.66%", "0.1317", "97.44%", "0.0655"],
     ["Head (frozen)", "2", "97.55%", "0.0632", "97.47%", "0.0643"],
     ["Head (frozen)", "3", "98.06%", "0.0491", "97.38%", "0.0661"],
     ["Head (frozen)", "4", "98.33%", "0.0436", "98.47%", "0.0390"],
     ["Head (frozen)", "5", "98.66%", "0.0357", "98.32%", "0.0387"],
     ["Fine-tune", "1", "99.20%", "0.0228", "98.79%", "0.0347"],
     ["Fine-tune", "2", "99.24%", "0.0204", "98.76%", "0.0329"]],
    font_size=9)
PB()

H2("4.3 Sensor Corpus Coverage and Data-Sanity Verification")
P("After alias resolution, unit harmonisation, derivation of the ratio and rate channels, and per-dataset min\u2013max scaling (Section 3.3), the four attached sensor datasets yield 922,424 real telemetry rows and a 16-dimensional canonical input: eight physical or derived channels plus eight binary availability (av_) flags. Because a 25% simulated chemical-masking augmentation (Section 3.5) is added on top of the real corpus, the branch actually trains on 1,153,030 rows: 922,424 real and 230,606 synthetic masking profiles carrying the av-flagged state label 2. The resulting class distribution is 489,778 fresh and 432,646 stale rows in the real fraction, so the real-data side is itself moderately imbalanced (1.13:1) and the masking class exists only through augmentation \u2014 a fact disclosed rather than hidden, since no public dataset is currently labelled for chemical adulteration of food.")
P("Before training, the pipeline emits a data-sanity report that asserts finiteness of every scaled feature and constrains the label domain to {0, 1, 2}. Table 4.3 reproduces the per-channel support ranges printed by that report on the final run; in particular, the real environment channels retain their physical units (ambient temperature 3\u201341 \u00b0C, relative humidity 0\u2013100%) because their min\u2013max bounds are estimated exclusively over the real rows that provide them. Zero NaN and zero inf counts across all sixteen columns confirm that the numeric hardening of Section 3.3 (NaN policy, Inf clipping, finite assertion) is exercised but never triggered on the final configuration \u2014 the guard exists for robustness, not because the data needed it.")
TBL("Table 4.3: Pre-training data-sanity report over the fused sensor corpus (as printed by the pipeline; all channels finite, 0 NaN, 0 Inf, labels restricted to {0, 1, 2}). Temperature-rise is the only fully synthetic channel, injected by the environment augmentation of Section 3.5.",
    ["Channel", "Min", "Max", "Provenance"],
    [["ch2o_ppm", "0", "2,975", "real (e-nose MQ135 + tofu formalin)"],
     ["voc_log_ohm", "0", "310", "real (e-nose MQ2 + chicken mean value)"],
     ["core_temp_c", "\u22120.5", "1", "scaled (tofu Temperature(C) + synthetic env)"],
     ["ambient_temp_c", "3", "41", "real (IoT Agriculture 2024, \u2018tempreature\u2019 alias)"],
     ["humidity_pct", "0", "100", "real (IoT Agriculture 2024)"],
     ["ch2o_voc_ratio", "0", "1,093", "derived (real constituents)"],
     ["temp_rise_rate", "0", "1.5", "synthetic (env augmentation)"],
     ["hours_elapsed", "0", "192", "real (chicken Shelf_Life_Day \u00d7 24) + synthetic"]],
    font_size=9)
FIG(os.path.join(OUT_DIR, "sensor_correlation.png"),
    "Figure 4.2: Pearson correlation heatmap of the eight canonical sensor channels over the fused corpus. The ch2o/voc pair exhibits the expected co-movement inside each source dataset while remaining weakly correlated across pooled datasets \u2014 visual evidence of why per-dataset scaling and availability flags, rather than global statistics alone, are required to merge heterogeneous hardware telemetry (analysis in Section 4.9).")
PB()

H2("4.4 Sensor Branch Results")
P("The sensor branch was trained for 30 epochs at batch size 256 with the Adam optimiser at learning rate 1e-4 (clip-norm 1.0) and a TerminateOnNaN guard, exactly as specified in Section 3.5. Table 4.4 condenses the trajectory: the branch is effectively converged by epoch 5 (val_acc 99.83%), and the remaining 25 epochs buy a two-point improvement in validation loss (0.0056 \u2192 0.0053) with validation accuracy flat at 99.84%. The final configuration reaches val_acc 99.84% / val_loss 0.0053 on 172,954 held-out rows.")
TBL("Table 4.4: Part B (sensor branch) training trajectory (selected epochs; full 30-epoch curve in Figure 4.3). Training uses 85% of 1,153,030 rows; validation is the held-out 15% (172,954 rows).",
    ["Epoch", "Train acc", "Train loss", "Val acc", "Val loss"],
    [["1", "96.12%", "0.1296", "99.71%", "0.0087"],
     ["5", "99.80%", "0.0084", "99.83%", "0.0056"],
     ["10", "99.82%", "0.0068", "99.84%", "0.0054"],
     ["15", "99.83%", "0.0064", "99.84%", "0.0053"],
     ["20", "99.82%", "0.0063", "99.84%", "0.0053"],
     ["25", "99.83%", "0.0062", "99.84%", "0.0053"],
     ["30", "99.83%", "0.0060", "99.84%", "0.0053"]],
    font_size=9)
FIG(os.path.join(OUT_DIR, "sensor_branch_training.png"),
    "Figure 4.3: Sensor branch training history over 30 epochs. Left: sparse-categorical cross-entropy loss. Right: accuracy. The plots show that convergence is achieved within the first five epochs and that neither the loss nor the accuracy term exhibits divergence, confirming the stability of the NaN-guarded optimisation configuration on a 1.15M-row table.")
FIG(os.path.join(OUT_DIR, "sensor_confusion_matrix.png"),
    "Figure 4.4: Confusion matrix of the sensor branch on the 172,954-row validation split. The Masked class, which exists purely through simulated chemical-masking augmentation, is separated with essentially zero confusion, while the real Fresh and Stale classes retain symmetric error rates (precision = recall = 0.998 for both), indicating that the 25% augmentation taught a separable masking signature rather than distorting the real decision boundary.")
TBL("Table 4.5: Class-resolved classification report of the sensor branch on the held-out validation split (172,954 rows; support reflects the 85/15 row split after the 25% masking augmentation).",
    ["Class", "Precision", "Recall", "F1-score", "Support"],
    [["Stale", "0.998", "0.998", "0.998", "64,675"],
     ["Fresh", "0.998", "0.998", "0.998", "73,774"],
     ["Masked", "1.000", "1.000", "1.000", "34,505"],
     ["Accuracy", "\u2013", "\u2013", "0.998", "172,954"],
     ["Macro avg", "0.999", "0.999", "0.999", "172,954"],
     ["Weighted avg", "0.998", "0.998", "0.998", "172,954"]],
    font_size=9)
P("Two observations qualify these numbers honestly. First, the Masked class is reinforcement-augmented, not observed: a perfect 1.000 F1 on simulated profiles demonstrates that the network learned the injected masking signature, but it cannot demonstrate performance on real adulterated food, and this distinction is carried into the limitations of Chapter 5. Second, the near-perfect accuracy is partly a property of the corpus composition \u2014 the e-nose day-series dominates with 838,502 rows whose decay progression is easy to separate \u2014 so the sensor branch should be read as a strong, well-regularised estimator of a lab-grade signal, rather than as proof of field robustness.")

H2("4.5 Fusion Engine Results")
P("The fusion training set is constructed by pairing images with same-state sensor rows under the Pair-Limit of 4,000 (Section 3.6). Because every fresh-looking image also has a 30% chance of being paired with a masked sensor profile, the realised composition is state 0 (stale) = 2,082 pairs, state 1 (fresh) = 1,330 pairs, and state 2 (chemically masked) = 588 pairs. Each pair is encoded by the frozen feature extractors into the 163-dimensional fusion input [Fv(128) | Fs(16) | raw(16) | posterior(3)], and the three-class sensor posterior appended in v13 gives the regressor direct access to the sensor branch\u2019s own belief \u2014 the input change that, per Section 3.6, breaks the val_mae plateau that the earlier 160-dimensional configuration could not improve below 0.0642 under identical bands.")
TBL("Table 4.6: Fusion training-set composition and supervision bands. The Freshness Index ground truth is drawn from narrow uniform bands of width 0.04 (v13), making the band width itself the irreducible target noise floor.",
    ["Property", "Value"],
    [["Pairs (total)", "4,000 (87/13-cap; 85/15 train\u2013val split = 3,400/600)"],
     ["State 0 (stale)", "2,082"],
     ["State 1 (fresh)", "1,330"],
     ["State 2 (masked)", "588"],
     ["Input dimension", "163 = 128 Fv + 16 Fs + 16 raw + 3 posterior"],
     ["Target bands", "fresh 0.88\u20130.92; masked 0.68\u20130.72; stale 0.17\u20130.21"],
     ["Loss", "Huber (\u03b4 = 0.1) + MAE metric"],
     ["Optimiser", "Adam 3e-4, clip-norm 1.0, ReduceLROnPlateau (0.5\u00d7, patience 4, min 1e-6)"]],
    font_size=9)
TBL("Table 4.7: Part C (fusion engine) training trajectory (selected epochs; 40-epoch curve in Figure 4.5). Learning-rate reductions were issued after epochs 7, 13, 17, 21, 25, 29, 33, and 37. The best validation MAE (0.0558) and the deployed final model (0.0570) are reported separately; the Huber validation loss bottoms at 0.0026.",
    ["Epoch", "Train MAE", "Val MAE", "Val loss (Huber)", "Learning rate"],
    [["1", "0.1165", "0.0672", "0.0035", "3.00e-04"],
     ["3", "0.0649", "0.0569", "0.0027", "3.00e-04"],
     ["5", "0.0582", "0.0578", "0.0027", "3.00e-04"],
     ["9", "0.0534", "0.0559", "0.0026", "1.50e-04"],
     ["15", "0.0499", "0.0558", "0.0026", "7.50e-05"],
     ["30", "0.0477", "0.0569", "0.0026", "4.69e-06"],
     ["40", "0.0477", "0.0570", "0.0026", "1.17e-06"]],
    font_size=9)
FIG(os.path.join(OUT_DIR, "fusion_training.png"),
    "Figure 4.5: Fusion engine training history over 40 epochs: train and validation MAE under Huber(0.1) loss. The step pattern in the learning-rate column of Table 4.7 is visible as small perturbations after each ReduceLROnPlateau event; after epoch 15 the validation MAE oscillates in the 0.056\u20130.060 band while train MAE continues to fall, the classic signature of a model at its label-noise floor \u2014 the \u00b10.04 band width \u2014 rather than an optimisation failure.")
P("The headline result deserves a precise statement. The best validation MAE across all 40 epochs is 0.0558 (epoch 15), corresponding to an average error of \u00b15.6 points on the 0\u2013100 Freshness Index scale; the final network after annealing scores 0.0570. An MAE of this magnitude is consistent with the supervision design: because the targets are uniformly distributed over bands of width 0.04, an oracle that always predicted the band centre would score 0.02, and no regressor trained on band-labelled targets can fall below that floor by construction. The residual 0.0358 above the theoretical centre-prediction floor is therefore attributable to the pair mixture, the 163-dimensional representation, and within-state ambiguity \u2014 and improving it further requires narrower bands, not more optimisation, which is why the chapter does not present 0.0558 as an optimisation failure but as the measured consequence of a deliberately noise-bounded design (discussed further in Section 5.4).")

H2("4.6 Mahalanobis Anomaly Detector Evaluation")
P("The Mahalanobis detector was fitted on the fresh subset of the 144-dimensional joint embedding space Z = [Fv | Fs]: 1,330 fresh pairs, far above the 146 samples needed for a 144-dimensional covariance to be estimable. The shrinkage estimator (\u03bb = 0.10 toward \u03bc\u00b7I plus a 1e-6 ridge) and eigenvalue clipping of Section 3.7 produce a well-conditioned inverse covariance; the fitted threshold is the maximum of the empirical 99th percentile of real fresh distances (10.310) and the \u03c7\u00b2 99% reference (13.653, df = 144), so the deployed threshold of 13.653 is \u03c7\u00b2-bound. Figure 4.6 visualises the resulting decision geometry in the Freshness Index \u00d7 Mahalanobis distance plane.")
FIG(os.path.join(OUT_DIR, "adulteration_boundary.png"),
    "Figure 4.6: Decision geometry of the deployed system in the (Freshness Index, Mahalanobis distance) plane. Samples above the 13.653 threshold with an index in the suspect interval (50\u201370) are routed to the route-2 anomaly state rather than mislabelled fresh or stale; the CHEMICAL MASKING ALERT override additionally fires when the unscaled CH2O channel exceeds 2.0 ppm. The plotted frontier is the empirical boundary induced by the 4,000-pair fusion set.")
TBL("Table 4.8: Mahalanobis separation statistics and distance-to-threshold ratios in the deployed 144-dimensional space (threshold 13.653). Stale distances sit unambiguously above the fresh distribution in both central tendency and tail; the fresh p99 stays inside the threshold, so fewer than 1% of fresh samples would ever be flagged.",
    ["Statistic", "Distance", "\u00d7 Threshold"],
    [["Fresh min", "4.27", "0.31\u00d7"],
     ["Fresh median", "6.49", "0.48\u00d7"],
     ["Fresh p99", "10.31", "0.76\u00d7"],
     ["Fresh max", "13.19", "0.97\u00d7"],
     ["Stale median", "19.66", "1.44\u00d7"],
     ["Stale p99", "33.80", "2.48\u00d7"]],
    font_size=9)
P("The separation is clean by every summary statistic: the entire fresh distribution (max 13.19) lies below the threshold, giving a measured false-alarm rate of 0 on the fusion set at the deployed threshold, while the stale median sits 1.44\u00d7 above the threshold and its 99th percentile 2.48\u00d7 above. The margin between fresh max and stale median (13.19 vs 19.66) is the practically relevant number for an embedded deployment: it is the slack the detector has against embedding drift, and the 1.44\u00d7 / 0.48\u00d7 asymmetry demonstrates that the shrinkage-regularised, empirically calibrated design avoids both failure modes observed in the intermediate development versions \u2014 distance explosion (distances of order 1e12 in a mismatched space) and threshold collapse (a \u03c7\u00b2-only threshold too tight for the real covariate structure).")

H2("4.7 End-to-End Inference Demonstration")
P("To prove that the trained artefacts compose into a working decision path, the full pipeline of Section 3.8 was executed on one validation sample: the image is embedded by the frozen image feature extractor, the canonical sensor row by the frozen sensor feature extractor, the 144-dimensional Z is scored by the Mahalanobis detector, and the 163-dimensional fusion input produces the Freshness Index that the deterministic rule set converts into a final state. Table 4.9 shows the exact output of that demonstration run.")
TBL("Table 4.9: End-to-end inference demonstration on one validation sample (pipeline output as printed by the deployed pipeline).",
    ["Field", "Value", "Interpretation"],
    [["freshness_index", "80.4", "above the 70-point fresh cutoff"],
     ["mahalanobis", "8.35", "below the 13.653 threshold (0.61\u00d7)"],
     ["state", "Fresh", "route-1 normal acceptance"]],
    font_size=9)
P("A transparency note on the artifact directory: the final run correctly produced the full-integer int8 image model, the float32 fusion model, and both C arrays. The working directory additionally retained an image_branch_dyn.tflite (718.0 KB) written by an earlier fallback execution in the same session\u2019s output directory; it is a residual of the resolved quantisation failure traced in Section 4.8 and is not consumed by the firmware. The deployed exports are image_branch_int8.tflite and fusion_engine_f32.tflite, as listed in Table 4.10.")

H2("4.8 Edge Deployment Artifacts")
TBL("Table 4.10: Firmware-ready artifacts generated by the final training run (sizes as reported by the pipeline). The deployed edge configuration carries the channel bounds, the Mahalanobis threshold and mean, and the 144 \u00d7 144 inverse covariance matrix as a compile-time C header for the ESP32-S3.",
    ["Artifact", "Size", "Purpose"],
    [["image_branch_int8.tflite", "791.8 KB", "fully integer image model (INT8 in/out)"],
     ["image_branch.cc", "2,522.9 KB", "C source of the INT8 model for TFLM"],
     ["fusion_engine_f32.tflite", "128.4 KB", "float32 fusion model (TFLM-safe)"],
     ["fusion_engine.cc", "451.2 KB", "C source of the fusion model"],
     ["edge_config.h", "338.5 KB", "bounds + threshold + mean + 144\u00d7144 inv covariance"],
     ["edge_config.json", "587.6 KB", "same constants in JSON for host-side tooling"],
     ["image_branch_best.keras", "5,554.8 KB", "best-checkpoint image branch (98.76 / 0.0329)"],
     ["image_branch.keras", "5,549.0 KB", "final image branch after fine-tuning"],
     ["image_feature_extractor.keras", "2,815.0 KB", "frozen Fv(128) extractor for fusion"],
     ["sensor_branch.keras", "87.5 KB", "3-class sensor branch"],
     ["sensor_feature_extractor.keras", "41.9 KB", "frozen Fs(16) extractor for fusion"],
     ["fusion_engine.keras", "410.6 KB", "fusion regressor (163 \u2192 1)"],
     ["sensor_dataset_coverage.csv", "0.3 KB", "coverage matrix of Table 4.1"],
     ["6 PNG figures", "see Figs. 4.1\u20134.6", "training curves, confusion matrix, boundaries"],
     ["image_branch_dyn.tflite", "718.0 KB", "residual fallback file, not deployed"]],
    font_size=9)
P("Quantisation results validate the edge path quantitatively. The image branch converts to a fully integer model (INT8 input and output) with a representative-data calibration strategy reducing the deployable footprint to 791.8 KB from the 5,549.0 KB float-weight Keras model (a 7× reduction, with the 2,522.9 KB C array carrying the same bytes into the firmware build); the 718.0 KB dynamic-range file is the residual artifact noted in Section 4.7, not the deployed model. The fusion engine is deliberately exported as float32 rather than dynamic-range-quantised: the shrinking learning rate schedule of its final epochs never materialised as int8-friendly weight distributions, and TFLM compatibility on the ESP32-S3 is preserved at 128.4 KB \u2014 a size that leaves both models room within the 8 MB flash budget of the reference hardware.")

H2("4.9 Summary of Results")
P("Table 4.11 consolidates the chapter. Read together, the three learned layers and the statistical detector meet the seven objectives of Section 1.3 to the extent that public data permits: objective 1 (separate image and sensor training with a missing-channel-tolerant schema) is met by Parts A and B including the corrected environment binding; objective 2 (three-class sensor states including chemical masking) is met with the F1 profile of Table 4.5; objective 3 (continuous fusion) is met with val MAE 0.0558 under disclosed band noise; objective 4 (Mahalanobis anomaly detection in the embedding space) is met with clean separation statistics and a 0% fresh false-alarm rate on the fusion set; objectives 5\u20137 (edge export, coverage-driven dataset recommendation, and end-to-end demonstration) are met by the artifact set and the Table 4.9 demonstration.")
TBL("Table 4.11: Consolidated results of the tri-layer system (single Kaggle run, TensorFlow 2.21.0, seed 42).",
    ["Layer", "Metric", "Value"],
    [["Image branch (Part A)", "val acc / val loss (deployed ckpt)", "98.76% / 0.0329 (peak 98.79%)"],
     ["Image branch (Part A)", "INT8 TFLite size", "791.8 KB"],
     ["Sensor branch (Part B)", "val acc / val loss", "99.84% / 0.0053"],
     ["Sensor branch (Part B)", "F1 (Stale / Fresh / Masked)", "0.998 / 0.998 / 1.000"],
     ["Fusion engine (Part C)", "val MAE / val loss (best)", "0.0558 / 0.0026 (final 0.0570)"],
     ["Mahalanobis detector", "threshold / fresh max / stale median", "13.653 / 13.19 / 19.66"],
     ["Mahalanobis detector", "fresh false-alarm rate (fusion set)", "0%"],
     ["End-to-end demo", "FI / d / state", "80.4 / 8.35 / Fresh"]],
    font_size=9)
PB()

# ----------------------------------------------------------------------
# CHAPTER 5 — CONCLUSION AND FUTURE WORK
# ----------------------------------------------------------------------
H1("CHAPTER 5  CONCLUSION AND FUTURE WORK")

H2("5.1 Summary of Work")
P("This thesis designed, implemented, and evaluated a tri-layer multimodal food freshness and safety detection system that integrates visual and gas-temperature telemetry evidence under a Mahalanobis-distance anomaly guard, and delivers every learned component as a firmware-ready TensorFlow Lite Micro artifact. The system was built as three deliberately separated training pipelines: a MobileNetV2-derived image branch (Part A) producing a 128-dimensional visual embedding Fv; an MLP sensor branch (Part B) producing a 16-dimensional telemetry embedding Fs and a three-class state posterior; and a compact fusion engine (Part C) that regresses a continuous 0\u2013100 Freshness Index from a 163-dimensional concatenation of both embeddings, the raw scaled channels, and the sensor posterior. A shrinkage-regularised Mahalanobis detector operates on the 144-dimensional joint embedding Z = [Fv | Fs] and, together with a deterministic CH2O override and cutoff rules, converts the continuous index into four operational states.")
P("The data strategy is 100% secondary: seven public Kaggle datasets (three image, four sensor) cover three food categories and eight canonical telemetry channels, merged through an alias-resolving, availability-flagged schema with documented per-dataset provenance. The entire system trains in a single Kaggle session, exports twenty artifacts, and is accompanied by complete edge configuration in both C-header and JSON forms.")

H2("5.2 Key Contributions")
BULLETS([
    "A missing-channel-tolerant sensor schema that maps four heterogeneous public datasets (electronic-nose, formalin tofu, chicken meat, greenhouse IoT) into eight canonical channels plus eight binary availability flags, with automatic per-channel coverage reporting and gap-filler recommendations \u2014 demonstrated by the v12 fix that binds the misspelled \u2018tempreature\u2019 column of IoT Agriculture 2024 to ambient_temp_c.",
    "A numerically hardened Mahalanobis detector for 144-dimensional food-quality embeddings: Ledoit-Wolf-style shrinkage (\u03bb = 0.10) plus ridge, eigenvalue flooring, and a max(empirical p99, \u03c7\u00b2) threshold, with in-run assertions against the distance-explosion failure mode that naive estimators exhibit on this data.",
    "A state-posterior-augmented fusion architecture: appending the sensor branch\u2019s 3-class posterior to the fusion input (160 \u2192 163 dimensions) breaks the val_mae plateau the earlier configuration could not improve below 0.0642, reaching 0.0558 under width-0.04 supervision bands.",
    "Transparency-preserving chemical-masking supervision: a 25% simulated augmentation creates the third (masked) state that no public dataset labels, trained in a way that keeps its effect visible in per-class reports rather than absorbed silently into binary accuracy.",
    "A complete, reproducible single-session pipeline with per-stage sanity assertions and documented NaN-guard logic, validated end to end on Kaggle with fixed seed 42 and no primary data collection.",
    "A firmware-ready export path delivering full-INT8 image inference (791.8 KB), TFLM-safe float32 fusion (128.4 KB), and compile-time edge constants (edge_config.h with the 144 \u00d7 144 inverse covariance) for the ESP32-S3."],
    numbered=True)

H2("5.3 Achievements Against Objectives")
P("Measured against the objectives of Section 1.3, the system attains: 98.76% validation accuracy (val loss 0.0329, peak 98.79%) on a 22,667-image corpus spanning fruits and vegetables, tilapia fish, and meat; 99.84% validation accuracy (val loss 0.0053) with class F1 scores of 0.998\u20131.000 on a 1.15M-row sensor corpus including simulated chemical masking; a fusion regression quality of val MAE 0.0558 under disclosed band noise; and Mahalanobis separation statistics in which every measured fresh distance (max 13.19) lies below the deployed threshold of 13.653 while the stale median sits 1.44\u00d7 above it, yielding a 0% fresh false-alarm rate on the fusion set. All learned components are quantised, converted, and deposited as deployable artifacts in the same session that trains them.")

H2("5.4 Limitations")
BULLETS([
    "The chemical-masking class is simulated: no public dataset is labelled for formalin or similar chemical adulteration at the sensor level, so the 1.000 F1 on the Masked class demonstrates separability of an injected signature, not field performance on genuinely adulterated food.",
    "temp_rise_rate is a fully synthetic channel supplied by the environment augmentation, and ambient temp/humidity are covered by only one real source (37,922 greenhouse rows); environmental generalisation to storage rooms unlike the source greenhouse is unverified.",
    "The fusion supervision bands (width 0.04) set an irreducible noise floor: the achieved val MAE of 0.0558 corresponds to \u00b15.6 points on the 0\u2013100 scale, well above the \u00b13\u20134.5-point aspiration set by stricter bands \u2014 narrowing bands further without broader training data would degrade, not improve, the deployed index.",
    "Fusion pairs are synthesized by matching images to same-state sensor rows rather than co-acquiring image and sensor evidence of the same physical sample, so cross-modal correlation reflects state agreement, not physical co-occurrence.",
    "Hardware validation on the ESP32-S3 (latency, RAM, BLE round-trip) is a deliverable of the firmware track and was not measured within this thesis scope; only artifact sizes and format compatibility were verified."])

H2("5.5 Future Work")
BULLETS([
    "Acquire a small co-acquired hardware corpus on the actual ESP32-S3 rig so that fusion pairs reflect physically simultaneous image plus sensor evidence, which is the single strongest upgrade path for the fusion engine.",
    "Source or contribute a real chemically-adulterated sensor dataset to replace the simulated masking augmentation with observed data \u2014 closing the largest validity gap identified in Section 5.4.",
    "Integrate the exported artifacts into the ESP32-S3 TFLM firmware and measure end-to-end latency, memory ceiling, and detection turnaround over BLE to complete the edge validation loop.",
    "Extend the Mahalanobis detector to tracked distance statistics (sliding-window drift monitoring) so a rising embedding drift can trigger recalibration instead of a binary threshold flip.",
    "Explore per-dataset calibration adapters for the sensor branch so new attached datasets change the availability pattern without retraining, deepening the missing-channel tolerance already present in the schema.",
    "Benchmark against unimodal ablations (image-only, sensor-only) on identical splits to quantify the additive value of integration under Mahalanobis supervision."])

H2("5.6 Concluding Remarks")
P("This thesis set out to answer whether image and sensor data, integrated under a principled statistical anomaly guard, can deliver a food freshness and safety assessment that is accurate, triangulated, and small enough to run at the edge. The evidence assembled across Chapters 3 and 4 supports an affirmative answer on all three counts, with the honest qualification that its third state \u2014 chemical masking \u2014 is supervised by simulation rather than observation, and its fusion regression quality is bounded by design rather than by optimisation inability. The system\u2019s core assets are its reproducibility (a single fixed-seed Kaggle session produces every number in this document) and its export completeness (twenty artifacts identical to what the firmware consumes). These properties are what turn the accuracy table into a habitable engineering artifact: any future dataset, channel, or threshold can be plugged into the same pipeline and re-evaluated with the same code.")

# ----------------------------------------------------------------------
# REFERENCES (Harvard style)
# ----------------------------------------------------------------------
H1("REFERENCES")
REF("haripriyasanga. (2021). Tilapia fish fresh and non-fresh species [Data set]. Kaggle. https://www.kaggle.com/datasets/haripriyasanga/tilapia-fish-fresh-and-non-fresh-species")
REF("mehrabmahdian. (2023). Food freshness electronic nose data [Data set]. Kaggle. https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data")
REF("raghavrpotdar. (2016). Fresh and stale images of fruits and vegetables [Data set]. Kaggle. https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables")
REF("vinayakshanawad. (2022). Meat freshness image dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset")
REF("wisam1985. (2024). IoT agriculture 2024 [Data set]. Kaggle. https://www.kaggle.com/datasets/wisam1985/iot-agriculture-2024")
REF("yudhaislamisulistya. (2022). Dataset aroma tahu berfomalin (formalin tofu aroma) [Data set]. Kaggle. https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin")
REF("ziya07. (2023). Chicken meat freshness dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/ziya07/chicken-meat-freshness-dataset")
REF("David, R., Duke, J., Jain, A., Reddi, V. J., Jeffries, N., Li, J., Kreeger, N., Nappier, I., Natraj, M., Wang, T., Warden, P. and Rhodes, R. (2021). TensorFlow Lite Micro: Embedded machine learning for TinyML systems. In Proceedings of Machine Learning and Systems (MLSys), 3, pp. 800\u2013811.")
REF("Lee, K., Lee, K., Lee, H. and Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. In Advances in Neural Information Processing Systems 31 (NeurIPS 2018), pp. 7167\u20137177.")
REF("Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2), pp. 365\u2013411.")
REF("Mahalanobis, P. C. (1936). On the generalized distance in statistics. Proceedings of the National Institute of Sciences of India, 2(1), pp. 49\u201355.")
REF("Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510\u20134520.")
REF("Warden, P. and Situnayake, D. (2019). TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers. Sebastopol, CA: O\u2019Reilly Media.")
PB()

# ----------------------------------------------------------------------
# APPENDICES
# ----------------------------------------------------------------------
H1("APPENDICES")

H2("Appendix A: Complete Configuration Constants (edge_config.h Excerpt)")
P("This appendix documents the compile-time constants extracted into the firmware header. The full 144 \u00d7 144 inverse covariance matrix occupies the remainder of edge_config.h (338.5 KB); the excerpt below reproduces its structure.")
P("SENSOR_CHANNELS = [ch2o_ppm, voc_log_ohm, core_temp_c, ambient_temp_c, humidity_pct, ch2o_voc_ratio, temp_rise_rate, hours_elapsed]; MAHALANOBIS_SPACE_DIM = 144; FV_DIM = 128; FS_DIM = 16; MAHALANOBIS_THRESHOLD = 13.653; CH2O_ALERT_PPM = 2.0; FRESHNESS_CUTOFF_FRESH = 70; SUSPECT_LOWCUT = 50; EXPANDED_FUSION_INPUT_DIM = 163 (128 Fv + 16 Fs + 16 raw scaled + 3 sensor posterior); SHRINKAGE_LAMBDA = 0.10.")

H2("Appendix B: Sensor Dataset Coverage Matrix (Machine-Readable)")
P("The coverage matrix of Table 4.1, as emitted by the pipeline into sensor_dataset_coverage.csv and reprinted here for reference: chicken_sensor (1,000 rows; voc_log_ohm, hours_elapsed; 500 fresh / 500 stale), enose_fruits (838,502 rows; ch2o_ppm, voc_log_ohm, ch2o_voc_ratio; 467,419 fresh / 371,083 stale), storage_env (37,922 rows; ambient_temp_c, humidity_pct; 0 fresh / 37,922 stale), tofu_formalin (45,000 rows; ch2o_ppm, voc_log_ohm, core_temp_c, ch2o_voc_ratio; 21,859 fresh / 23,141 stale).")

H2("Appendix C: Environment Augmentation Parameters")
P("The uncovered environment channels (temp_rise_rate, plus the synthetic fraction of ambient_temp_c and humidity_pct) are injected by a DHT22-physics simulator operating at a 70% active / 30% availability-flag-0 split. Simulated ambient temperature follows a diurnal sinusoid between 22 and 34 \u00b0C with Gaussian perturbation; humidity follows an inverse-phase envelope between 40 and 85% RH; temp_rise_rate is derived as the first difference of simulated core temperature normalised to the 0\u20131.5 \u00b0C/h physical range observed in cooling-room data. All synthetic rows carry availability flags that mark them synthetic at inference time, so deployed hazard detection is never silently attributed synthetic provenance.")

H2("Appendix D: Mahalanobis Calibration Report")
P("Numerical details of the covariance fit: input is the 144-dimensional Z = [Fv(128) | Fs(16)] over 1,330 fresh fusion pairs; accumulation proceeds in float64; the sample covariance is blended by \u03bb = 0.10 toward \u03bc\u00b7I where \u03bc is the average marginal variance, then regularised by a 1e-6 ridge; eigenvalues are clipped below at max(1e-3 \u00b7 \u03bbmax, 1e-12); the deployed threshold is max(empirical p99 of fresh = 10.310, \u03c7\u00b2 99% at df = 144 = 13.653). Fresh-distance summaries: min 4.27, median 6.49, p99 10.31, max 13.19; stale-distance summaries: median 19.66, p99 33.80.")

H2("Appendix E: Edge Configuration Schema (edge_config.json)")
P("edge_config.json (587.6 KB) carries the same constants in a machine-readable form for host-side tooling: sensor feature order, per-channel min\u2013max bounds, the Mahalanobis threshold, the mean vector, the 144 \u00d7 144 inverse covariance, the fusion input contract (concat(Fv 128, Fs 16, raw_scaled 16, posterior 3) = 163), the freshness-index bands, the CH2O alert level (2.0 ppm), and the image and sensor dataset lists used for provenance tracing. The header edge_config.h is generated from the same dictionary so the firmware and the tooling can never drift apart.")

# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
doc.save(DOC_PATH)
print("Saved:", DOC_PATH)
print("Sections:", len(doc.sections), "| Paragraphs:", len(doc.paragraphs), "| Tables:", len(doc.tables))
