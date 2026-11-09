# MNIST Digit Recognition — Demo & Verification Guide

This guide outlines how to verify the data ingestion and transformation pipeline established during **Phase 1**.

---

## 1. Environment Setup

Verify Python 3.10+ and install project dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset Pipeline Verification

Execute the automated test suite to confirm dataset download, deterministic splitting, conservative augmentations, and PyTorch DataLoader generation:

```bash
pytest tests/test_dataset.py -v
```

### Expected Output:
* `test_dataset_sizes`: Validates $50{,}000$ train, $10{,}000$ validation, and $10{,}000$ test samples.
* `test_tensor_shapes_and_label_bounds`: Confirms tensor shape `(1, 28, 28)` and label range $[0, 9]$.
* `test_normalized_tensor_values`: Confirms normalization with $\mu=0.1307$ and $\sigma=0.3081$.
* `test_train_val_split_disjoint`: Ensures zero index overlap between train and validation partitions.
* `test_augmentation_attached_only_to_train`: Verifies that `RandomAffine` is active only on the training set.
* `test_class_distributions_balanced`: Verifies balanced representation of digits $0$ through $9$ across all splits.
* `test_dataloaders_batch_generation`: Checks that batches yield `(64, 1, 28, 28)` tensors.

---

## 3. Direct Verification Script

You can run the dataset module directly to print dataset statistics and class distributions:

```bash
python -m src.dataset
```

---

## 4. CNN Model Architecture Verification (Phase 2)

Inspect model parameters and verify CPU/CUDA forward passes:

```bash
python -m src.model
```

Run model unit tests:

```bash
pytest tests/test_model.py -v
```

---

## 5. Model Training Execution (Phase 3)

Run the full training pipeline with AdamW, learning rate scheduling, and early stopping:

```bash
python -m src.train
```

Run training pipeline unit tests (configuration validity, weight updates, checkpoint reload):

```bash
pytest tests/test_train.py -v
```

---

## 6. Test-Set Evaluation & Benchmarking (Phase 4)

Execute formal evaluation against the untouched 10,000-sample test set:

```bash
python -m src.evaluate
```

This generates:
* `reports/evaluation_results.json` — Comprehensive serialized test metrics.
* `reports/classification_report.json` — Precision, recall, and F1 scores per digit class.
* `reports/figures/confusion_matrix.png` — Heatmap of prediction counts.
* `reports/figures/confusion_matrix_normalized.png` — Proportion heatmap.
* `reports/figures/training_curves.png` — Loss and accuracy vs epoch curves.
* `reports/figures/error_analysis.png` — Diagnostic grid of top misclassified test samples.

Run evaluation unit tests:

```bash
pytest tests/test_evaluate.py -v
```

---

## 7. Real-World Inference & Preprocessing (Phase 5)

Run standalone inference verification on synthetic external digits:

```bash
python -m src.inference
```

Run comprehensive inference unit tests (format handling, polarity inversion, blank image rejection, center of mass):

```bash
pytest tests/test_inference.py -v
```

---

## 8. Interactive Streamlit Web Application (Phase 6)

### 8.1 Launch the Web Application

```bash
streamlit run app.py
```

The application runs at `http://localhost:8501`.

### 8.2 End-to-End Application Workflow

1. **Verify Requirements**: Ensure `best_model.pth` is available in `models/`.
2. **Launch Streamlit**: Execute `streamlit run app.py`.
3. **Model Caching**: The model predictor loads onto GPU (CUDA) or CPU once upon startup via `@st.cache_resource`.
4. **Drawing a Digit**:
   * Navigate to the **✏️ Draw Digit** tab.
   * Adjust the brush stroke slider (default 18px).
   * Draw a bold digit (e.g. `3` or `7`) near the center of the canvas.
   * Click **✨ Predict Drawn Digit**.
5. **Inspect the Output**:
   * Observe the dominant predicted digit card and confidence percentage.
   * Review the horizontal probability bar chart across all 10 classes.
   * Inspect the preprocessed $28 \times 28$ image to see the cropped, centered digit.
6. **Uploading an Image**:
   * Navigate to the **📤 Upload Image** tab.
   * Upload an external PNG/JPG image containing a handwritten digit.
   * Click **✨ Predict Uploaded Digit**.

### 8.3 Recommended Interactive Demo Test Cases

| Test Case | Interaction Steps | Expected Application Behavior |
| :--- | :--- | :--- |
| **1. Clear Drawn Digit** | Draw a bold `7` on the black canvas and submit. | Displays `# 7` with high confidence ($>90\%$), top class bar for `7`, and a centered $28 \times 28$ preview. |
| **2. Clear Uploaded Digit** | Upload an image of a black digit `0` on white paper. | Polarity detection inverts the image to white-on-black; displays `# 0` with high confidence. |
| **3. Blank Canvas Submission** | Click predict without drawing anything on the canvas. | Gracefully displays warning: *"⚠️ No clear digit detected. Please draw a single digit with a stronger stroke."* (No error traceback). |
| **4. White-on-Black vs. Black-on-White** | Test drawings with both light and dark backgrounds. | Bounding-box and polarity routines automatically align both to the MNIST convention. |
| **5. Ambiguous Handwriting** | Draw an ambiguous digit (e.g. flat-topped `3` resembling `5`). | Prediction reflects the top class, while the probability chart visualizes secondary confidence distributed to `5`. |
| **6. Fast Multiple Predictions** | Predict multiple digits in succession. | Model is cached in memory via `@st.cache_resource` and never reloaded across runs. |

### 8.4 Run Full Automated Test Suite (All 37 Tests)

```bash
pytest tests/ -v
```






