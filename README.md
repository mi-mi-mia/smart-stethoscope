# Smart Stethoscope

Smart Stethoscope is a machine learning project exploring how respiratory conditions can be classified from lung sound recordings.

The project was developed as a team effort during a machine learning bootcamp, combining signal processing, feature engineering, and deep learning to build a working prototype for classifying breath sounds.

👉 **Live demo:** [Smart Stethoscope](https://smart-stethoscope.streamlit.app/)

This repository contains both the original project work and a later re-evaluation of the modelling pipeline.

---

## 🩺 What we built

As a team, we:

* Processed raw respiratory recordings into structured and visual representations
* Engineered acoustic features capturing intensity, frequency, and temporal variation
* Built multiple models, including:

  * Logistic regression baselines
  * XGBoost (tabular model)
  * CNN on mel spectrograms
  * A hybrid (late fusion) model combining both
* Developed a Streamlit app that takes a short breath recording and outputs predicted conditions with confidence scores

The final demo used a **hybrid model**, combining XGBoost and CNN predictions.

---

## 🧠 Approach (high level)

The pipeline follows three main stages:

### 1. Preprocessing

* Audio resampled to 12 kHz
* Recordings segmented into 6-second windows
* Feature extraction:

  * Tabular acoustic features (e.g. MFCCs, spectral statistics)
  * Mel spectrograms for CNN input

### 2. Modelling

* Tabular models (logistic regression, XGBoost) trained on engineered features
* CNN trained on spectrogram images
* Hybrid model combining predictions via weighted probability fusion

### 3. Evaluation

* Grouped splitting by patient to avoid data leakage
* Evaluation at:

  * **Chunk level** (6s segments)
  * **Patient level** (aggregated predictions)
* Macro F1 used as the primary comparison metric

---

## 📊 Results (re-evaluation)

The pipeline was later reconstructed and evaluated using a consistent methodology.

| Model                | Chunk Macro F1 | Patient Macro F1 |
| -------------------- | -------------- | ---------------- |
| Logistic regression  | ~0.51          | ~0.45            |
| XGBoost              | ~0.41          | ~0.49            |
| CNN                  | ~0.33          | ~0.32            |
| Hybrid (late fusion) | ~0.41          | ~0.39            |

### Key takeaways

* XGBoost provided the strongest **patient-level performance**
* The CNN struggled as a standalone model and was sensitive to class imbalance
* The hybrid model did **not consistently improve performance** over XGBoost
* Performance appears limited primarily by:

  * Severe class imbalance
  * Overlapping acoustic characteristics between conditions

---

## 🔍 Re-evaluation

After the initial project, the pipeline was revisited to better understand which components were contributing to performance.

This re-evaluation involved:

* Reconstructing the full pipeline from raw audio to prediction
* Applying grouped splitting consistently across all models
* Comparing models using both chunk-level and patient-level metrics
* Running qualitative inference checks on example recordings

The results suggest that while the hybrid approach was a reasonable design choice during the project, a simpler tabular model (XGBoost) is more effective under a consistent evaluation setup.

---

## 🗂 Repository structure

```text
.
├── notebooks/
│   ├── 00_re_evaluation_pipeline.ipynb   # End-to-end reconstruction and evaluation
│   ├── README.md                        # Notebook guide
│   └── ...
│
├── docs/
│   ├── PIPELINE_AND_MODELS.md           # Detailed pipeline and modelling documentation
│   ├── FEATURES.md                      # Feature dictionary and acoustic rationale
│   ├── XGBOOST.md                       # XGBoost preprocessing and feature pipeline
│
├── models/                              # Saved model artefacts
├── app/                                 # Streamlit application
```

---

## 📓 Notebooks

* **Start here:** `notebooks/00_re_evaluation_pipeline.ipynb`

  * Rebuilds the pipeline end to end
  * Contains final evaluation and conclusions

Earlier notebooks are included for context and show the evolution of the project.

---

## 📚 Technical documentation

More detailed technical references are available in:

* `docs/FEATURES.md` — feature engineering and acoustic interpretation
* `docs/XGBOOST.md` — preprocessing and tabular pipeline
* `docs/PIPELINE_AND_MODELS.md` — full technical overview

---

## 📁 Dataset

This project uses the **Respiratory Sound Database**:

* Rocha, B. M., Filos, D., Mendes, L., et al. (2019). *An open access database for the evaluation of respiratory sound classification algorithms*. Physiological Measurement.
* Available via: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database

---

## ⚠️ Notes

* The dataset is highly imbalanced, with COPD dominating the majority of samples
* Results should be interpreted cautiously and are not clinically validated
* The application is intended as a prototype / exploration, not a diagnostic tool

---

## 🙌 Credits

This project was developed as a group effort, combining contributions across data processing, modelling, and application development.

This repository also includes a later re-evaluation of the modelling pipeline to better understand performance and limitations.
