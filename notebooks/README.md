# Smart Stethoscope: Respiratory Disease Classification

This Bootcamp project consists of a hybrid **Deep Learning + Machine Learning** system for respiratory disease classification using audio signals captured from digital stethoscopes.

For more technical details regarding preprocessing and the models, the file **FEATURES.md** is available.

---

## Overview

The **Smart Stethoscope** automates the detection of pathologies such as Asthma, COPD, and Pneumonia using state-of-the-art Digital Signal Processing (DSP). The model's intelligence lies in the combination of two analysis streams:

1.  **XGBoost Stream (80% weight):** Focused on extracting statistical features and structured acoustic biomarkers.
2.  **CNN Stream (20% weight):** Focused on recognizing morphological patterns and textures in Mel-Spectrograms.

---

## Technical Pipeline (Preprocessing)

To ensure data consistency across models, we apply a rigorous preprocessing pipeline:

* **Resampling:** All audio files are converted to **12,000 Hz** to optimize the capture of the respiratory range (20Hz - 6kHz).
* **Mu-Law Compression:** Non-linear compression is applied to enhance low-amplitude details (breathing nuances) and compress loud peaks.
    * **Formula:** $y_{compressed} = \text{sign}(y) \cdot \frac{\ln(1 + \mu|y|)}{\ln(1 + \mu)}$ where $\mu=255$.
* **Standardization:** Segmentation into fixed 6-second windows (**72,000 samples**).
* **Data Integrity:** The train/test split is strictly partitioned by **Patient ID** to prevent **Data Leakage** (preventing the model from memorizing a patient's unique "acoustic fingerprint").

---

## Model Architecture

### 1. XGBoost: Structured Acoustic Biomarkers
The tabular model analyzes over 32 features extracted from both time and frequency domains:

* **Temporal Features:** RMS Energy (intensity/effort) and Zero Crossing Rate (turbulence/obstruction).
* **Spectral Features:** MFCCs (timbre/resonance), Spectral Centroid (sound "brightness"), and Spectral Flux (detection of sudden energy shifts).
* **Statistical Shape:** Skewness and Kurtosis to identify transient peaks (such as fine crackles).

### 2. CNN: Deep Spatial Patterns
Audio is converted into a **Mel-Spectrogram**, allowing the convolutional network to identify visual textures:
* **Vertical Spikes:** Represent *Crackles* (Pneumonia/LRTI).
* **Horizontal Streaks:** Represent *Wheezes* (Asthma/COPD).
* **Coarse Grain/Clouds:** Indicate secretion noise in large airways (Rhonchi). - DID THE MODEL DO THIS? 

---

## Features & Clinical Patterns

| Diagnosis | Key Acoustic Pattern | Clinical Rationale |
| :--- | :--- | :--- |
| **Asthma** | High Centroid Std + High ZCR | Wheezing and turbulence in narrowed bronchi. |
| **Pneumonia** | High Spectral Flux + High RMS Std | Fluid "pops" (crackles) causing explosive energy peaks. |
| **COPD** | Low Rolloff + Specific MFCCs | High frequencies muffled by chronic inflammation. |
| **Healthy** | Low ZCR + Stable RMS | Rhythmic laminar flow without adventitious sounds. |

---

## Implementation Specs

```python
# Digital Signal Processing (DSP) constants
SAMPLE_RATE = 12000
DURATION = 6          # seconds
WINDOW_SIZE = 2048    # n_fft
HOP_LENGTH = 512
MU = 255              # Mu-law factor

---

## Results

| Metric | Value |
|--------|------|
| Chunk level Accuracy | 72% |
| Weighted F1 Score | .55 |
| Patient-level accuracy| 63% |

The hybrid model outperformed standalone approaches, especially in distinguishing COPD vs Asthma - it did start to prioritize Pneumonia, so class imbalance still needs to be improved.