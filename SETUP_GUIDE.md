# Project Vision-Edge — Complete Setup and Run Guide

This is the full, final version of your software pipeline. Follow this
guide top to bottom the first time. After that, you'll only need the
"Run Order" table.

---

## 1. Folder structure to create

Inside your project folder (`Project-Vision-Edge`), you need this layout.
The `data/train` and `data/test` folders must already contain your
images before you run anything — everything else is created
automatically by the scripts.

```
Project-Vision-Edge/
├── data/
│   ├── train/
│   │   ├── defective_parts/      <- put your defective training images here
│   │   └── good_parts/           <- put your good training images here
│   ├── test/
│   │   ├── defective_parts/      <- put your defective test images here
│   │   └── good_parts/           <- put your good test images here
│   └── batch_inspect/            <- (optional) any images you want to test as a batch/demo
├── models/                       <- created automatically, holds trained models
├── cache/                        <- created automatically, speeds up training (safe to delete anytime)
├── reports/                      <- created automatically by batch_inspect.py
└── src/
    ├── preprocessing.py
    ├── train.py
    ├── fine_tune.py
    ├── convert_tflite.py
    ├── convert_tflite_int8.py
    ├── evaluate.py
    ├── threshold_analysis.py
    ├── infer.py
    ├── infer_int8.py
    ├── benchmark_latency.py
    ├── live_inspection.py
    └── batch_inspect.py
```

**Folder names matter.** The two class folders under `data/train` and
`data/test` must be spelled exactly the same in both places (e.g.
`defective_parts` and `good_parts`), because the code reads class names
directly from these folder names.

---

## 2. Libraries to install

Open a terminal in your project folder and run:

```
pip install -r requirements.txt --break-system-packages
```

(If you're using a virtual environment instead of your system Python,
drop the `--break-system-packages` part.)

This installs: `tensorflow-cpu`, `silence_tensorflow`, `opencv-python`,
`numpy`, and `scikit-learn`.

---

## 3. Run order

Run these from your project's root folder (not from inside `src/`), like:
`python src/train.py`

| # | Command | What it does | What you'll see |
|---|---|---|---|
| 1 | `python src/train.py` | Trains the model from scratch | 5 epochs of accuracy/loss printed; `models/defect_model.keras` created |
| 2 | `python src/evaluate.py` | Measures real accuracy on the test set | Accuracy, confusion matrix, precision/recall table |
| 3 | `python src/threshold_analysis.py` | Finds the best decision cutoff | A table comparing accuracy/missed-defects at different thresholds |
| 4 | *(optional)* `python src/fine_tune.py` | Tries to push accuracy higher | 5 more epochs; overwrites `defect_model.keras` — **re-run steps 2 and 3 after this** to check if it actually helped |
| 5 | `python src/convert_tflite.py` | Compresses the model for fast PC testing | `models/optimized_defect.tflite` created |
| 6 | `python src/convert_tflite_int8.py` | Compresses the model for embedded/FPGA deployment | `models/optimized_defect_int8.tflite` created, much smaller |
| 7 | `python src/infer.py` | Tests one image | Prints score + Good/Defective for one sample image |
| 8 | `python src/infer_int8.py` | Same, but for the INT8 model | Same output, using the embedded-ready model |
| 9 | `python src/benchmark_latency.py` | Measures speed | Milliseconds per inference for both models |
| 10 | `python src/live_inspection.py` | Live webcam demo with heatmap | A window opens; press SPACE to inspect, Q to quit |
| 11 | `python src/batch_inspect.py` | Inspects a whole folder at once | A CSV report in `reports/`, plus a printed summary |

**Before step 1**, make sure your images are already sorted into the
`data/train/...` and `data/test/...` folders described above.

**Before step 10 or 11**, make sure you've run steps 1 and 5 (or 6) at least once.

---

## 4. Updating the decision threshold

After running `threshold_analysis.py` (step 3), pick a threshold value
from its table and update the `DEFECT_THRESHOLD` constant near the top
of these three files, so they all agree:
- `src/infer.py`
- `src/infer_int8.py`
- `src/live_inspection.py`
- `src/batch_inspect.py`

---

## 5. Why training takes a while, and what's been done about it

Every epoch, TensorFlow has to re-read and decode every image from disk.
That decoding work is identical every single epoch, so `train.py` now
uses `.cache()` to save the decoded images to disk after the first pass.
**Epoch 1 will still be slow** (the cache is being built), but epochs 2
through 5 should be noticeably faster. If you ever change your dataset
images, delete the `cache/` folder first so it rebuilds with the new data.

---

## 6. About lighting/noise robustness

`train.py` now includes data augmentation (random flips, slight rotation,
brightness, contrast, and zoom changes) applied only during training.
This meaningfully improves how well the model handles different lighting
and camera angles in your live demo — but no vision system can be
completely unaffected by lighting or noise, so don't claim full immunity
in your report. "Improved robustness through data augmentation" is the
accurate and defensible claim.

---

## 7. What each output file actually is

| File | What it is |
|---|---|
| `models/defect_model.keras` | The full trained Keras model (float32) |
| `models/optimized_defect.tflite` | Compressed for quick PC testing (weights only quantized) |
| `models/optimized_defect_int8.tflite` | Fully quantized for embedded/FPGA deployment |
| `reports/inspection_report_*.csv` | Batch inspection results, one row per image |
| `cache/*` | Cached decoded images — speeds up retraining, safe to delete |
