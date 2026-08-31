# Project Vision-Edge

**Embedded AI-Based Real-Time Optical Defect Detection for Manufacturing Quality Control**

Final Year Project — Department of Electrical Engineering
National University of Sciences and Technology (NUST CEME)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-INT8_Quantized-FF6F00?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Software%20Pipeline%20Complete-10B981?style=flat-square)

| | |
|---|---|
| **Supervisor** | Dr Sidra Tul Muntaha |
| **Group Members** | Maria Hyder (486881), Raja Hassan Naeem (368667) |

Vision-Edge classifies manufactured parts as **good** or **defective** from a camera image. A MobileNetV2 model is trained on a PC, then compressed to TensorFlow Lite (including full INT8) so the same pipeline can later run on embedded / FPGA hardware.

This repository is the **software and AI pipeline**. FPGA deployment is a separate phase.

> 📘 First-time setup? See [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for the full folder-by-folder walkthrough. This README's Run Order table below is the quick reference once you're set up.

---

## Results (test set)

Evaluated on **715** held-out images with decision threshold **0.85**.

| Metric | Value |
|---|---|
| Overall accuracy | **98.60%** (705 / 715) |
| Defective parts | 453 images — recall **99.34%** (3 missed defects) |
| Good parts | 262 images — recall **97.33%** (7 false alarms) |
| Borderline scores (within 0.05 of 0.85) | 20 / 715 |

Class order from the data folders (alphabetical):

- class `0` → `defective_parts`
- class `1` → `good_parts`

A score **below 0.85** is called defective. That cutoff is stricter than 0.5 so missed defects stay low, which is the costly error in quality control.

---

## Pipeline

```
images (data/train, data/test)
        │
        ▼
   train.py  →  models/defect_model.keras
        │
        ├─► evaluate.py
        ├─► threshold_analysis.py
        └─► convert_tflite.py / convert_tflite_int8.py
                    │
                    ▼
         optimized_defect.tflite          (PC testing)
         optimized_defect_int8.tflite     (embedded / FPGA)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     infer.py   live_inspection.py   batch_inspect.py
     infer_int8.py                   reports/*.csv
```

---

## Repository layout

```
Project-Vision-Edge/
├── data/
│   ├── train/
│   │   ├── defective_parts/
│   │   └── good_parts/
│   ├── test/
│   │   ├── defective_parts/
│   │   └── good_parts/
│   └── batch_inspect/          # mixed images for the batch demo
├── models/
│   ├── defect_model.keras
│   ├── optimized_defect.tflite
│   └── optimized_defect_int8.tflite
├── reports/                    # CSV output from batch_inspect.py
├── src/
│   ├── train.py
│   ├── fine_tune.py            # optional; do not rerun if the current model is frozen
│   ├── evaluate.py
│   ├── threshold_analysis.py
│   ├── convert_tflite.py
│   ├── convert_tflite_int8.py
│   ├── preprocessing.py
│   ├── infer.py
│   ├── infer_int8.py
│   ├── live_inspection.py
│   ├── batch_inspect.py
│   └── benchmark_latency.py
├── requirements.txt
└── README.md
```

Folder names under `train/` and `test/` must match exactly. Keras reads class names from those folders.

---

## Setup

Python 3.11 recommended. From the project root:

```bash
pip install -r requirements.txt
```

Dependencies: `tensorflow-cpu` (or `tensorflow` if you have a GPU), `silence_tensorflow`, `opencv-python`, `numpy`, `scikit-learn`.

Run every command from the **project root**, not from inside `src/`:

```bash
python src/train.py
```

---

## Run order

| Step | Command | Output |
|---|---|---|
| 1 | `python src/train.py` | `models/defect_model.keras` |
| 2 | `python src/evaluate.py` | Accuracy, confusion matrix, precision / recall |
| 3 | `python src/threshold_analysis.py` | Table of cutoffs; **0.85** is the value used in this project |
| 4 | `python src/convert_tflite.py` | `models/optimized_defect.tflite` |
| 5 | `python src/convert_tflite_int8.py` | `models/optimized_defect_int8.tflite` |
| 6 | `python src/infer.py path/to/image.jpeg` | One-image verdict (float TFLite) |
| 7 | `python src/infer_int8.py path/to/image.jpeg` | One-image verdict (INT8 TFLite) |
| 8 | `python src/benchmark_latency.py` | Average and 95th-percentile latency (PC CPU baseline) |
| 9 | `python src/live_inspection.py` | Webcam demo: SPACE = inspect, Q = quit |
| 10 | `python src/batch_inspect.py` | CSV in `reports/` |

Optional:

```bash
python src/fine_tune.py
```

Fine-tuning overwrites `models/defect_model.keras`. Re-run steps 2–5 after it. The reported **98.60%** is from the current frozen Keras model — do not fine-tune again unless you intend to replace those numbers.

If the dataset images change, delete the `cache/` folder before training so it rebuilds.

---

## Decision threshold

`DEFECT_THRESHOLD = 0.85` in:

- `src/evaluate.py`
- `src/infer.py`
- `src/infer_int8.py`
- `src/live_inspection.py`
- `src/batch_inspect.py`

Keep these in sync. `threshold_analysis.py` is the script used to choose the cutoff.

---

## Models

| File | Role |
|---|---|
| `models/defect_model.keras` | Full trained Keras model (MobileNetV2 transfer learning) |
| `models/optimized_defect.tflite` | Dynamic-range quantized TFLite for PC testing |
| `models/optimized_defect_int8.tflite` | Full INT8 TFLite for embedded / FPGA toolchains |

`benchmark_latency.py` times inference on **this PC's CPU**. That is a baseline for the report, not an FPGA measurement.

---

## Live inspection

`live_inspection.py` opens the default webcam.

- **SPACE** — classify the current frame
- **Q** — quit
- Green border = good; red border = defective
- Defective frames also show a Grad-CAM heatmap
- The terminal prints a simulated gate command (`PASS` / `DIVERT`) where a real Arduino or FPGA actuator would be driven later

---

## Dataset notes

- Training uses on-disk caching (`cache/`) so epochs after the first are faster.
- Data augmentation (flip, small rotation, brightness, contrast, zoom) is applied **only during training**.
- The test folder is also used as validation during `model.fit`. That should be stated in the report; there is no separate validation split.

---

## Tech stack

- Python, TensorFlow / Keras
- MobileNetV2 (ImageNet pretrained, frozen base, custom sigmoid head)
- TensorFlow Lite (dynamic-range and full INT8)
- OpenCV
- scikit-learn (metrics)

---

## Project status

| Phase | Status |
|---|---|
| Dataset, training, evaluation | Complete |
| Threshold selection (0.85) | Complete |
| TFLite + INT8 conversion | Complete |
| Live webcam demo + batch CSV | Complete |
| PC latency baseline | Complete |
| FPGA / Vitis-AI deployment | Next phase |

---

## License

Academic / FYP use — NUST CEME.
