# MNIST Handwritten Digit Recognition (CNN)

An internship-ready machine learning portfolio project implementing a Convolutional Neural Network (CNN) in PyTorch for handwritten digit classification ($0$ through $9$).

---

## Project Overview

* **Task**: Multiclass image classification of $28 \times 28$ grayscale handwritten digits.
* **Framework**: PyTorch & Torchvision.
* **Target Metric**: Out-of-sample test accuracy targeting $\ge 99.0\%$ (to be evaluated on canonical test set).
* **Architecture Highlights**:
  * Dual convolutional blocks with Batch Normalization and spatial Dropout.
  * Deterministic $50{,}000$ train / $10{,}000$ validation / $10{,}000$ test partition.
  * Conservative augmentation (rotation $\pm 10^\circ$, translation $\pm 2$ px, subtle scaling; no flipping).
  * Center-of-mass and aspect-ratio preserving inference preprocessing to overcome real-world canvas mismatch.
  * Interactive dual-mode Streamlit web application (live drawing canvas and file upload).

---

## Project Structure

```
MNIST-Digit-Recognition/
├── .gitignore               # Strict exclusion of dataset caches, envs, and temporary files
├── LICENSE                  # MIT License
├── README.md                # Project documentation and status
├── DEMO.md                  # Verification and demo instructions
├── requirements.txt         # Pinned project dependencies
├── data/                    # MNIST dataset directory (excluded from git)
├── models/                  # Checkpoint storage
├── reports/
│   └── figures/             # Evaluation plots and figures
├── config/
│   └── config.yaml          # Centralized hyperparameter configuration
├── src/
│   ├── __init__.py
│   ├── dataset.py           # MNIST download, transforms, splitting, and DataLoaders
│   ├── model.py             # CNN architecture definition (Phase 2)
│   ├── train.py             # Training loop, AdamW, scheduler, and early stopping (Phase 3)
│   ├── evaluate.py          # Benchmark metrics, confusion matrix, error analysis (Phase 4)
│   ├── inference.py         # Center-of-mass centering & prediction engine (Phase 5)
│   └── utils.py             # Deterministic seed, device selection, config helpers
├── tests/
│   ├── __init__.py
│   ├── test_app.py          # Streamlit UI & caching tests (Phase 6)
│   ├── test_dataset.py      # Unit tests for data shapes, bounds, and split isolation
│   ├── test_model.py        # Model architecture tests (Phase 2)
│   ├── test_train.py        # Pipeline & checkpoint tests (Phase 3)
│   ├── test_evaluate.py     # Evaluation & metrics tests (Phase 4)
│   └── test_inference.py    # Preprocessing and inference tests (Phase 5)
└── app.py                   # Streamlit web interface (Phase 6)
```

---

## Implementation Roadmap

| Phase | Milestone | Status |
| :--- | :--- | :--- |
| **Phase 1** | **Setup & Data Pipeline** (Structure, config, deterministic splits, augmentation, unit tests) | **COMPLETED** |
| **Phase 2** | **CNN Architecture** (Dual-block Conv2D, BatchNorm, Dropout, parameter profiling) | **COMPLETED** |
| **Phase 3** | **Training Pipeline** (AdamW, ReduceLROnPlateau, Early Stopping, Checkpointing) | **COMPLETED** |
| **Phase 4** | **Evaluation & Diagnostics** (Confusion matrix, classification report, error analysis) | **COMPLETED** |
| **Phase 5** | **Inference Engine** (Center-of-mass centering, bounding-box crop, normalization) | **COMPLETED** |
| **Phase 6** | **Interactive Streamlit UI** (Drawing canvas, upload, probability bar chart) | **COMPLETED** |

---

## Real-World Inference Preprocessing Pipeline (Phase 5)

> The model achieves 99.64% accuracy on the canonical MNIST test set. A dedicated preprocessing pipeline adapts external handwritten images toward the MNIST input format.

Real-world user drawings (from web canvases, mobile touchscreens, or document uploads) originate from a different data distribution than MNIST. The `DigitPredictor` class bridges this domain gap via a multi-stage computer vision pipeline:

1. **Grayscale & Format Normalization**: Converts PIL (L, RGB, RGBA) and NumPy arrays to a float32 $[0.0, 1.0]$ luminance channel.
2. **Automated Polarity Inversion**: Detects whether the input is dark strokes on light paper or light strokes on dark canvas via boundary sampling, ensuring the digit is foreground ($\approx 1.0$) on dark background ($\approx 0.0$).
3. **Blank & Uniform Image Rejection**: Checks contrast and active stroke pixel counts; raises a descriptive `ValueError` if the canvas is blank or no clear foreground digit is detected.
4. **Bounding-Box Cropping**: Detects active digit contours and tightly crops out extraneous white space.
5. **Aspect-Ratio-Preserving Resize**: Scales the largest digit dimension to $20\text{ pixels}$ using high-quality area/cubic interpolation without geometric distortion.
6. **Center-of-Mass Alignment**: Places the digit on a $28 \times 28$ canvas and shifts the centroid to $(13.5, 13.5)$ using image moments ($M_{10}/M_{00}, M_{01}/M_{00}$), matching Yann LeCun's canonical MNIST specification.
7. **Z-Score Normalization**: Scales tensor values by MNIST mean ($0.1307$) and standard deviation ($0.3081$).

### Python Usage Example

```python
from PIL import Image
from src.inference import DigitPredictor

predictor = DigitPredictor("models/best_model.pth")
image = Image.open("sample_digit.png")

result = predictor.predict(image)
print(f"Predicted Digit: {result['digit']}")
print(f"Confidence:      {result['confidence'] * 100:.2f}%")
print(f"Probabilities:   {result['probabilities']}")
```

---

## Test-Set Evaluation Results (Phase 4 Benchmark)

> A convolutional neural network trained on MNIST achieved **99.64% accuracy** on the untouched 10,000-image test set.

### 1. Overall Test Metrics
* **Total Test Samples**: $10{,}000$ (canonical MNIST test set, completely unaugmented)
* **Correct Predictions**: $9{,}964$
* **Incorrect Predictions**: $36$ (only $0.36\%$ error rate)
* **Test Cross-Entropy Loss**: `0.0098`
* **Macro Precision / Recall / F1**: `0.9964` / `0.9964` / `0.9964`
* **Weighted Precision / Recall / F1**: `0.9964` / `0.9964` / `0.9964`

### 2. Per-Class Performance
| Digit | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.9980 | 0.9990 | **0.9985** (Easiest) | 980 |
| **1** | 0.9974 | 0.9965 | 0.9969 | 1135 |
| **2** | 0.9942 | 0.9971 | 0.9956 | 1032 |
| **3** | 0.9960 | 0.9960 | 0.9960 | 1010 |
| **4** | 0.9959 | 0.9949 | **0.9954** (Hardest) | 982 |
| **5** | 0.9966 | 0.9955 | 0.9961 | 892 |
| **6** | 0.9958 | 0.9969 | 0.9963 | 958 |
| **7** | 0.9961 | 0.9951 | 0.9956 | 1028 |
| **8** | 0.9979 | 0.9969 | 0.9974 | 974 |
| **9** | 0.9960 | 0.9960 | 0.9960 | 1009 |

### 3. Diagnostic Visualizations
* **Confusion Matrix**: `reports/figures/confusion_matrix.png`
* **Normalized Confusion Matrix**: `reports/figures/confusion_matrix_normalized.png`
* **Training Curves**: `reports/figures/training_curves.png`
* **Error Analysis**: `reports/figures/error_analysis.png` (displays top 16 highest-confidence errors)

---

## Training Performance (Phase 3 Validation Benchmark)

The model was trained using AdamW on $50{,}000$ training images and monitored against $10{,}000$ validation images:

* **Hardware Acceleration**: NVIDIA GeForce RTX 2050 (CUDA)
* **Training Epochs Completed**: 15 epochs (Early stopping triggered after epoch 12 peak)
* **Best Epoch**: Epoch 12
* **Best Validation Loss**: `0.0151`
* **Best Validation Accuracy**: `99.55%`
* **Final Training Accuracy**: `98.54%` (incorporates online data augmentation)
* **Checkpoint File**: `models/best_model.pth` (5.39 MB, verified loadable)

---

## Interactive Web Application (Streamlit)

A dark-themed, dashboard interface built with **Streamlit** and **Altair** allows users to test the CNN in real-time.

### Key Features:
* **Interactive Drawing Canvas**: Powered by `streamlit-drawable-canvas` ($280 \times 280$ box, adjustable brush width, clear button).
* **Image Upload**: Supports uploading PNG, JPG, JPEG, and WEBP files (both dark-on-light paper sketches and light-on-dark digital sketches).
* **Prominent Metric Card**: Displays the dominant predicted digit with confidence badge (`>= 90%` High, `70-90%` Moderate, `< 70%` Low Confidence warning).
* **Full Class Probability Distribution**: Horizontal bar chart comparing probabilities across all 10 digit classes ($0$ through $9$) using Altair.
* **Preprocessed 28×28 Tensor Preview**: Visually exposes the exact normalized, centered 28×28 array fed to the CNN for model explainability.
* **Robust Error Handling**: Automatically catches blank canvases or faint noise, prompting users with guidance without application tracebacks.

### Running the Web Application:

```bash
streamlit run app.py
```

### Real-World Input Caveat & Scope:
* **Canonical vs. Real-World Accuracy**: The model achieved **99.64% accuracy** on the benchmark MNIST test set. Real-world user drawings originate from a different visual distribution; performance may vary based on stroke thickness, touch device latency, and handwriting style.
* **Single-Digit Limitation**: The classifier is intentionally designed for **one isolated digit at a time (0–9)**. Multi-digit sequences are not segmented.

---

## Quickstart & Verification Commands

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Full Test Suite** (37 Unit Tests across all phases):
   ```bash
   pytest tests/ -v
   ```

3. **Launch Streamlit Web UI**:
   ```bash
   streamlit run app.py
   ```


