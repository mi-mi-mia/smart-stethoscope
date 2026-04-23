# Notebooks

This folder contains both the original project notebooks and a later re-evaluation of the pipeline.

## ⭐ Start here

- **0_project_revisit_eval.ipynb**

This notebook reconstructs the full pipeline end to end and re-evaluates model performance using a consistent methodology (grouped splitting, macro F1, and patient-level aggregation).

---

## Notebook Folder Structure

### Exploration
Early data exploration and understanding of class imbalance.

### Preprocessing
Initial work on audio segmentation and preprocessing pipelines that informed the final implementation.

### Modelling
Iterative model development:
- Logistic regression baselines
- XGBoost experiments
- CNN models
- Hybrid (late fusion) approaches

These notebooks reflect the development process and are not necessarily directly comparable due to differences in preprocessing, class setup, and evaluation.

### Evaluation
Later-stage attempts to evaluate models more rigorously, including early signs that performance was lower than initially expected (though still encouraging for a multiclass challenge with high class imbalance).

### Experiments
Additional exploratory work (augmentation, alternative features, transformers, architectures). Informed but not part of the final pipeline.

---

## Notes

- The re-evaluation notebook should be treated as the **canonical reference** for model performance.
- Earlier notebooks are retained for context and to show the evolution of the project.
