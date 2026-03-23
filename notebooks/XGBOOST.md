# XGBOOST.md - Audio Processing & Feature Engineering

This document describes the technical pipeline used to transform raw respiratory recordings into a tabular dataset optimized for **XGBoost** classification.

---

## 🛠️ Preprocessing Workflow

The pipeline follows a specific order to ensure signal integrity and feature relevance. Processing the signal in this sequence prevents noise amplification before cleaning.

### 1. Data Loading & Standardization
* **Resampling:** Audio is loaded and decimated to a Target Sample Rate (SR) of **10,000 Hz**.
    * *Objective:* Reduces high-frequency noise and computational complexity while keeping the clinical range of breath sounds (typically < 4kHz).
* **Amplitude Normalization:** `librosa.load` automatically scales the raw 16-bit PCM signal to a **float32** range between $[-1.0, 1.0]$.
    * *Requirement:* This normalization is mandatory for the subsequent $\mu$-law compression step.

### 2. Temporal Cleaning
* **Silence Removal (Trimming):** `librosa.effects.trim` is applied to the full recording.
    * *Logic:* We remove leading and trailing silence **before** compression. If applied after, the compression would amplify background noise, making it harder for the algorithm to detect the true start of the breath sound.

### 3. Segmentation (Windowing)
* **6-Second Slicing:** The trimmed signal is divided into fixed **6-second chunks**.
* **Constraint:** Only complete 6s segments are kept. This ensures that the statistical means calculated in the next step are based on the same amount of temporal information.

### 4. Signal Enhancement ($\mu$-Law Compression)
Each 6s chunk is transformed using the $\mu$-law companding algorithm within the `extract_features_raw` function:

$$y_{compressed} = \operatorname{sgn}(y) \frac{\ln(1 + \mu |y|)}{\ln(1 + \mu)}$$

Where:
* $y$ is the normalized input signal.
* $\mu = 255$ (standard ITU-T value).
* **Why:** Respiratory sounds like crackles and wheezes often have low energy compared to the main breath. $\mu$-law non-linearly amplifies these low-level details while compressing high-energy peaks, acting as a "magnifying glass" for subtle diagnostic artifacts.

---

## Feature Extraction (18 Features)

After compression, we extract a vector of 18 numerical features (statistical means) per chunk:

| Category | Features | Description |
| :--- | :--- | :--- |
| **Temporal** | `rms_mean`, `zcr_mean` | Root Mean Square (Energy/Volume) and Zero-Crossing Rate (Noise/Texture). |
| **Spectral** | `centroid`, `rolloff`, `flatness`, `flux` | Describes the "brightness," spectral shape, and onset strength of the sound. |
| **Cepstral** | `mfcc_1` to `mfcc_12` | Mel-Frequency Cepstral Coefficients. We ignore `mfcc_0` to avoid bias from constant energy levels. |

---

## Validation & Anti-Leakage Protocol

To ensure the **XGBoost** model generalizes to new patients and doesn't just "memorize" voices:

1.  **Patient-Level Split:** Training and Testing sets **must** be split by `patient_id`. 
2.  **No Overlap:** All 6s chunks from "Patient A" must reside in the same fold (either Train or Test), never both.
3.  **Validation:** Use `GroupKFold` during cross-validation to maintain this boundary.

