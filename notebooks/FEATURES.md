features_content = """# Audio Feature Dictionary & Technical Documentation

This document serves as reference for the features extracted from the respiratory sound recordings.

## Preprocessing Standards
* **Resampling:** 22,050 Hz.
* **Temporal Formatting/Standardization:** 6-second standardized windows (aligned w breathing cycles).
* **Peak Normalization:** Amplitude scaled to 1.0. TBC
TBC * **Statistical Aggregation:** Each spectral/temporal feature is summarized by its **Mean**, **Standard Deviation**, and **Skewness**?? (double check this one) across the 6-second window. TBC

## Temporal Alignment & Splitting Strategy (Our 6s Rule - 1 breath)
To align with the Tabular Preprocessing, a strict 6-second temporal constraint is applied to all audio samples using a **Sliding/Sequential Window** approach:

* **Fixed Duration:** Every audio segment is standardized to exactly 132,300 samples (6s @ 22,050 Hz).
* **Sequential Splitting (The "Split" Logic):** Instead of only capturing the first 6 seconds, the entire recording is split into consecutive 6s blocks (0-6s, 6-12s, etc.). This ensures that pathological events occurring later in the recording (like a late-cycle crackle) are not lost.
* **Synchronization:** Feature extraction is performed per block. Any metadata or annotations provided by 'txt' (like `start` and `end` of cycles) are mapped to their respective 6s window. (check Lily's and Mia's code to make sure you are not missing anything)
* **Consistency:** This 6s window is the same used for the `cycle_length` calculation in the demographic pipeline, ensuring the model views a consistent "time slice" of the patient's respiratory health across both audio and tabular data. (again, double check code)
* **Remainder Handling:** If the final segment of a recording is shorter than 6s, it is zero-padded to maintain a uniform input shape ($1 \times 132,300$) for the feature extractor.

## Feature Glossary

### 1. Temporal Features (Time-Domain)
* **RMS Energy (`rms_mean`, `rms_std`):** * *Description:* Represents the physical intensity of the sound.
    * *Clinical Use:* Tracks breathing effort and rhythm. High variance (`std`) may indicate labored breathing.
* **Zero Crossing Rate (`zcr_mean`, `zcr_std`):** * *Description:* The rate at which the signal changes sign.
    * *Clinical Use:* Measures turbulence. High ZCR values are common indicators of airway obstructions (narrowed passages).

### 2. Spectral Features (Frequency-Domain)
* **MFCCs (`mfcc_0` to `mfcc_12`):** * *Description:* Mel-Frequency Cepstral Coefficients. A compact representation of the "timbre."
    * *Clinical Use:* Acts as a "lung fingerprint," capturing the unique resonance of the patient's respiratory tract.
* **Spectral Centroid (`centroid_mean`):** * *Description:* The "center of mass" of the frequency spectrum.
    * *Clinical Use:* Identifies the perceived "brightness" of the sound. Critical for detecting high-pitched **Wheezes**.
* **Spectral Flux (`flux_mean`, `flux_max`):** * *Description:* Measures the rate of change in the power spectrum.
    * *Clinical Use:* The primary detector for **Crackles** (sudden, explosive pops caused by fluid in the lungs).
* **Spectral Bandwidth (`bandwidth_mean`):** * *Description:* The spectral "spread." 
    * *Clinical Use:* Distinguishes clean airflows from "noisy" or congested sounds (mucus/secretions).
* **Spectral Rolloff (`rolloff_mean`):** * *Description:* The frequency below which 85% of the energy lies.
    * *Clinical Use:* Helps distinguish legitimate lung sounds from high-frequency environmental noise.
 
## Feature Dictionary & Clinical Relevance - TBD
We extract 7 key acoustic/sound features to map respiratory patologies:

| Feature | Clinical Target | Why it matters? |
| :--- | :--- | :--- |
| **MFCC** | COPD & Airways | Captures the "timbre" and structural signature of the lungs. |
| **Spectral Flux** | Pneumonia | Detects sudden "pops" or **Crackles** (fluid in the lungs). |
| **Spectral Centroid**| Asthma | Measures "brightness" to identify high-pitched **Wheezes**. |
| **RMS Energy** | Breathing Effort | Quantifies the intensity and physical effort of the cycle. |
| **ZCR** | Obstruction | Measures turbulence caused by narrowed air passages. |
| **Bandwidth** | Congestion | Tracks the "clarity" vs. "noise" spread of the breath. |
| **Rolloff** | Filter/Quality | Distinguishes lung sounds from environmental noise. |

## OR

## Clinical Diagnosis & Acoustic Patterns - TBD
Instead of analyzing features in isolation, the model identifies specific respiratory conditions by detecting **Feature Patterns**. This approach mimics clinical auscultation by correlating sound characteristics with physical lung states.

| Diagnosis | Key Acoustic Pattern (Combinations) | Clinical Rationale |
| :--- | :--- | :--- |
| **Asthma** | **High Centroid + High ZCR** | High-pitched "whistling" (Centroid) caused by extreme air turbulence in narrowed bronchi (ZCR). |
| **Pneumonia** | **High Spectral Flux + MFCC Shifts** | Sudden energy spikes (Flux) represent **Crackles** (fluid pops), while MFCCs capture the dense sound of consolidated lung tissue. |
| **COPD** | **Low Rolloff + Specific MFCCs** | Chronic inflammation "muffles" high frequencies (Rolloff) and permanently alters the lung's resonance "signature" (MFCC). |
| **Bronchiectasis**| **High Bandwidth + High Flux** | Excess mucus creates "noisy/dirty" signals (Bandwidth) with frequent bubbling pops (Flux). |
| **Healthy** | **Low ZCR + Stable RMS** | Smooth, laminar airflow with no turbulence and a regular, rhythmic breathing effort. |

### Feature Context - TBC
* **The "Location" Rule:** Acoustic patterns must be interpreted based on `chest_location`. For example, a **High Centroid** is normal in the **Trachea** (due to high-speed airflow) but indicates pathology (Wheezing) if found in the **Posterior** lobes.
* **Non-Linear Relationships:** The XGBoost model is expected to learn that while one feature might be noisy, the *relationship* between them (e.g., Flux jumping while RMS stays low) is the primary indicator of a **Crackle**. TBC - do we wanto to do this?
* **Temporal Stability:** Pathological sounds like **Wheezes** tend to be continuous (stable Spectral Centroid), while **Crackles** are transient (high variance in Spectral Flux).

## !!! Data Integrity
* **Critical Rule:** The training and testing sets are partitioned by **Patient ID** to prevent **Data Leakage**.
"""
**The Risk:** Since multiple recordings exist for the same patient, including the same patient in both sets will cause **Data Leakage**, inflating accuracy by memorizing the individual's unique respiratory "fingerprint".

## Statistical Summarization
Since each recording lasts 6 seconds, features are extracted frame-by-frame and then aggregated into single values for the tabular model:
* **Mean (`_mean`):** Captures the average acoustic characteristic of the breath.
* **Standard Deviation (`_std`):** Measures the stability of the sound. High `std` in RMS or Flux often indicates irregular breathing or crackles.
* **Skewness (`_skew`):** Indicates the asymmetry of the sound distribution (useful for distinguishing short "pops" from continuous noise).

## Contextual Metadata: Chest Location
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

## 🛠️ Implementation Specs?
* **Library:** Librosa (v0.10.x)
* **Window Size (n_fft):** 2048
* **Hop Length:** 512