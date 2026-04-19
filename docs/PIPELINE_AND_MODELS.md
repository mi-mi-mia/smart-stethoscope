features_content = """# Audio Feature Dictionary & Technical Documentation

This document serves as reference for the features extracted from the respiratory sound recordings for the **Smart Stethoscope** project.

## Preprocessing Standards Shared Pipeline)
* **Resampling:** 12,000 Hz.
* **Temporal Formatting/Standardization:** 6-second standardized windows (aligned w starting breathing cycle).
* **Peak Normalization:** Amplitude scaled to **[-1.0, 1.0]** during loading. (automatic with librosa)
* **Non-Linear Compression (Mu-Law):** Applied to enhance low-level signals (like breathing details) and compress loud peaks.
    * *Formula:* $y_{compressed} = \text{sign}(y) \cdot \frac{\ln(1 + \mu|y|)}{\ln(1 + \mu)}$ where $\mu=255$.
* **Data Integrity:** Train/test split is strictly partitioned by **Patient ID** to prevent "fingerprint memorization" (**Data Leakage**).
* **Demo Blacklist:** Patient IDs **142, 191, and 182** are excluded from all training to serve as 100% unseen data for live validation.

TBC:
* **Statistical Aggregation:** Each feature is summarized by its **Mean** and **Standard Deviation** across the 6-second window to capture both the average sound and its temporal variance.

## Temporal Alignment & Splitting Strategy (Our 6s Rule - 1 breath)
A strict 6-second temporal constraint is applied to all audio samples to ensure uniform input for the XGBoost model:

* **Fixed Duration:** Every audio segment is standardized to **72,000 samples** (6s @ 12,000 Hz).
* **Overlapping Sliding Window (Data Augmentation):**
    * **COPD Patients:** Sequential splitting (**no overlap**) to avoid data redundancy in the majority class.
    * **Other Pathologies:** **2-second step size (Overlap)**, augmenting minority classes to improve model balance and exposure to transient events.
* **Pure Audio Focus:** Demographic data (age, BMI, sex) was excluded after testing showed it did not significantly improve predictive performance. The model relies exclusively on acoustic biomarkers.

---

## XGBoost Stream: Structured Acoustic Biomarkers
The **XGBoost** model (Weight: 0.8) identifies respiratory conditions by detecting complex statistical relationships between 32+ engineered features.

### Clinical Diagnosis & Acoustic Patterns
Instead of analyzing features in isolation, the model identifies specific respiratory conditions by detecting **Feature Patterns**. This approach mimics clinical auscultation by correlating sound characteristics with physical lung states.

| Diagnosis | Key Acoustic Pattern (Combinations) | Clinical Rationale |
| :--- | :--- | :--- |
| **Asthma** | **High Centroid Std + High ZCR** | Wheezing causes rapid tone shifts (**Centroid Std**) and high-frequency turbulence (**ZCR**) in narrowed bronchi. |
| **Pneumonia** | **High Spectral Flux + High RMS Std** | Sudden energy spikes (**Flux**) represent **Crackles** (fluid pops in alveoli) and "sound explosions" (**RMS Std**). |
| **COPD** | **Low Rolloff + Specific MFCCs** | Chronic inflammation "muffles" high frequencies (**Rolloff**) and alters the lung's unique resonance signature (**MFCCs**). |
| **Healthy** | **Low ZCR + Stable RMS** | Smooth, laminar airflow with no turbulence and a regular, rhythmic breathing effort. |
| **URTI** | **Skewness/Kurtosis + MFCC Variance** | Upper infections create sound asymmetry (**Skewness**) and timbre shifts due to throat/nasal congestion. |
| **Bronchiectasis**| **High Bandwidth + High Flatness Std** | Excess mucus creates "noisy" signals (**Bandwidth**) with intermittent secretion-related energy shifts (**Flatness Std**). |
| **Bronchiolitis** | **High Centroid + High Flux** | Common in small airways; represents a mix of high-pitched whistling and inflammatory fluid pops. |
| **LRTI** | **High Flux + Spectral Flatness Shifts** | Lower infections show complex patterns of deep congestion and irregular "crackling" noise. |

## Feature Glossary

### 1. Temporal Features (Time-Domain)
* **RMS Energy (`rms_mean`, `rms_std`):**
    * *Description:* Represents the physical intensity (volume) of the sound.
    * *Clinical Use:* Tracks breathing effort. High `rms_std` indicates "sound explosions" or gasping.
* **Zero Crossing Rate (`zcr_mean`):**
    * *Description:* The rate at which the signal changes sign.
    * *Clinical Use:* Measures turbulence. High ZCR values indicate airway obstructions or high-frequency friction.

### 2. Spectral Features (Frequency-Domain)
* **MFCCs (`mfcc_1_mean` to `mfcc_15_mean` and `mfcc_1_std` to `mfcc_15_std`):**
    * *Description:* Mel-Frequency Cepstral Coefficients (15 coefficients, ignoring the 0-th index). A compact representation of the "timbre.
    * * *Clinical Use:* The "Lung Fingerprint." Captures the unique resonance and timbre of the patient's respiratory tract.
* **Spectral Centroid (`centroid_mean`, `centroid_std`):**
    * *Description:* The "center of mass" of the spectrum.
    * *Clinical Use:* Identifies "brightness." High `centroid_std` indicates rapid **tone changes** (common in wheezing).
* **Spectral Flatness (`flatness_mean`, `flatness_std`):**
    * *Description:* Measures how "noise-like" vs. "tone-like" a sound is.
    * *Clinical Use:* Distinguishes constant background noise from intermittent pathological sounds.
* **Spectral Flux (`flux_mean`):**
    * *Description:* Measures the rate of change in the power spectrum.
    * *Clinical Use:* Primary detector for **Crackles** (sudden, explosive energy shifts).
* **Spectral Bandwidth (`bandwidth_mean`):**
    * *Description:* The spectral "spread."
    * *Clinical Use:* Distinguishes clean airflows (narrow) from congested/noisy sounds (wide).

### 3. Shape Statistics (Advanced Metrics)
* **Skewness (`skewness_mean`):** Measures the asymmetry of the Spectral Centroid distribution.
    * *Description:* Measures the asymmetry of the spectral distribution (how much the sound leans toward lower or higher frequencies).
    * *Clinical Use:* High skewness helps the model identify "bursty" or impulsive events. It is a key differentiator for isolated crackles (short, sudden pops) versus continuous background noise.
* **Kurtosis (`kurtosis_mean`):** Measures the "tailedness," identifying if the sound has frequent extreme outliers or peaks.
    * *Description:*  Measures the "tailedness" or the presence of outliers in the sound wave.
    * *Clinical Use:* Identifies the "sharpness" of acoustic peaks. In respiratory analysis, high kurtosis pinpoint extreme, transient sounds (like sharp, fine crackles) that deviate significantly from the normal breathing baseline.

## Statistical Summarization
Since each recording lasts 6 seconds, features are extracted frame-by-frame and then aggregated into single values for the tabular model:
* **Mean (`_mean`):** Captures the average acoustic characteristic of the breath.
* **Standard Deviation (`_std`):** Measures the stability of the sound. High `std` in RMS or Flux often indicates irregular breathing or crackles.
* **Skewness (`_skew`):** Indicates the asymmetry of the sound distribution (useful for distinguishing short "pops" from continuous noise).

--- Still needs improvement:

## CNN Stream: Deep Spatial Patterns
The **CNN (Convolutional Neural Network)** (Weight: 0.2) treats audio as a **2D Mel-Spectrogram**, identifying visual "textures" and temporal-frequency correlations that statistical averages might miss.

### 1. The Mel-Spectrogram Transformation
Raw audio is converted into a visual heatmap (Image Tensor) where:
* **X-Axis:** Time (Temporal progression).
* **Y-Axis:** Mel-Scale Frequency (Human-centric pitch perception).
* **Intensity:** Decibels (Energy/Loudness at a specific moment).

Unlike a standard spectrogram, the **Mel-Scale** compresses frequencies to mimic human auditory perception.
* **Clinical Focus:** It provides higher resolution for low frequencies (**20Hz - 2,000Hz**), where the majority of pathological pulmonary events occur.
* **Dimensions:** The 6-second audio is converted into a 2D matrix (Image) where the **X-axis** is Time and the **Y-axis** is the logarithmic Mel-frequency.
* **Normalization:** Logarithmic decibel (dB) scaling is applied to emphasize low-intensity respiratory details against background silence.


### 2. What the CNN "Sees" (Pattern Recognition)
While XGBoost analyzes global statistical trends, the CNN acts as a **morphological specialist**:
* **Convolutional Filters:** These detect "edges" in the sound. A sudden vertical edge on the spectrogram is the exact visual record of a **Crackle** (explosive pop).
* **Temporal Continuity:** Long, bright horizontal lines represent the persistence of a musical tone, the classic hallmark of a **Wheeze**.
* **Local Robustness:** The CNN can identify a pathology even if it occurs for only 1 second of the total 6-second window, as it scans the image in small overlapping fragments (kernels).

### 3. Visual Pattern Recognition Guide
| Visual Feature | Acoustic Equivalent | Clinical Significance |
| :--- | :--- | :--- |
| **Vertical Spikes** | **Transient Bursts** | **Crackles (Pneumonia/LRTI/Fibrosis):** Short-duration, explosive pops across frequencies. |
| **Horizontal Streaks** | **Tonal/Musical Lines** | **Wheezes (Asthma/COPD):** Continuous musical sounds indicating airway narrowing. |
| **Coarse Grain / Clouds** | **Low-Freq Turbulence** | **Rhonchi (Bronchiectasis):** Deep, "snoring" patterns caused by secretions in large airways. |
| **Faint High-Freq Haze** | **Upper Airway Noise** | **Congestion (URTI):** Diffuse, unstructured energy from nasal or throat obstruction. |
| **Sparse Sharp Points** | **Fine Crackles** | **Bronchiolitis:** Tiny, high-pitched "pin-prick" pops typical of small airway inflammation. |
| **Smooth Gradients** | **Laminar Flow** | **Healthy:** Rhythmic "fading in/out" without jagged lines or sharp discontinuities. |

### 4. Hybrid Logic (Why 20%?)
The CNN serves as an **"Audit Vote."** If the XGBoost model detects statistics suggesting Asthma, but the CNN fails to visualize "horizontal streaks" (wheezes) in the spectrogram, the final output confidence is adjusted. This dual-layer approach reduces **False Positives** caused by external ambient noise that might distort statistical averages but does not replicate biological respiratory patterns.

---

## 🧠 Hybrid Model Fusion Logic
To maximize predictive reliability, the final output is a **Weighted Probability Fusion**:
* **XGBoost (80%):** Provides high-precision classification based on clinical acoustic biomarkers.
* **CNN (20%):** Acts as a secondary "specialist," validating the tabular prediction by recognizing deep visual textures in the sound.
* **Outcome:** A unified **Confidence Score** that leverages both mathematical rigor and spatial intuition.


### Feature Context 
* **Non-Linear Relationships:** The XGBoost model is expected to learn that while one feature might be noisy, the *relationship* between them (e.g., Flux jumping while RMS stays low) is the primary indicator of a **Crackle**. TBC - do we wanto to do this?
* **Temporal Stability:** Pathological sounds like **Wheezes** tend to be continuous (stable Spectral Centroid), while **Crackles** are transient (high variance in Spectral Flux).

## !!! Data Integrity & Demo Safety
* **Patient ID Partitioning:** Training and testing sets are strictly partitioned by **Patient ID** to prevent "fingerprint memorization" (Data Leakage).
    * **The Risk:** Since multiple recordings exist for the same patient, including the same patient in both sets will cause **Data Leakage**, inflating accuracy by memorizing the individual's unique respiratory "fingerprint".
* **Demo Blacklist:** Specific Patient IDs (**142, 191, 182**) are **excluded** from all preprocessing and training. These are reserved as "Unseen Data" for live validation during the Demo Day.

Maybe as improvement? next steps:
## Contextual Metadata: Chest Location - TBC
For the predictive model to distinguish between normal resonance and pathological sounds, the recording location is a critical categorical feature.

### 1. Data Mappin
Each audio sample in the dataset is tagged with one of the following locations:
* **Trachea (Tc):** High-flow, turbulent sounds (normally brighter).
* **Anterior (Al, Ar):** Front of the chest.
* **Posterior (Pl, Pr):** Back of the chest (ideal for detecting crackles in the lung base).
* **Lateral (Ll, Lr):** Sides of the rib cage.

### 2. Clinical Logic for the Model - TBC (see how medical accurate this is)
The model must treat `chest_location` as a **categorical input**. 
* **Rationale:** A "High Spectral Centroid" (high-pitched sound) might be normal in the **Trachea**, but is a strong indicator of **Wheezing** (Asthma/COPD) if detected in the **Posterior** lower lobes. 

### 3. Product Requirement (App Submission)
When a user submits a recording via the app, the interface **must** ask: *"Where are you placing the device?"*. This user input will be mapped to the `input_location` column in the inference pipeline to match the training data structure.

## 🛠️ Technical Implementation Specs
To ensure reproducibility, the following Digital Signal Processing (DSP) constants were used across the entire pipeline:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Sample Rate** | 12,000 Hz | Optimized to capture the 20Hz-6kHz respiratory range. |
| **Window Size** | 2048 (n_fft) | Spectral resolution for FFT calculation. |
| **Hop Length** | 512 | Step size between successive frames. |
| **Mu-Law ($\mu$)** | 255 | Non-linear compression for dynamic range enhancement. |