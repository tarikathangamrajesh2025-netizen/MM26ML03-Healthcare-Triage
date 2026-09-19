# Healthcare Triage Classification — MM26ML03

Machine learning project for emergency patient START triage classification using clinical and demographic features.

## 📌 Project Overview

This project uses machine learning to classify emergency patients into four START triage categories:

* **RED** — Immediate
* **YELLOW** — Delayed
* **GREEN** — Minor
* **BLACK** — Expectant/Deceased

The model uses patient vital signs, consciousness indicators, demographics, and other clinical features to predict the appropriate triage category.

## 📊 Dataset

The training dataset contains **664 patient records** and the test dataset contains **112 patient records**.

Features include:

* Vital signs: heart rate, respiratory rate, oxygen saturation, blood pressure, and temperature
* Glasgow Coma Scale (GCS)
* AVPU/consciousness information
* Pain assessment
* Arrival transport
* Demographic information
* Diagnosis, medication, and vital-sign count features

The patient identifier (`subject_id`) was excluded from model training.

## ⚙️ Preprocessing

The following preprocessing steps were applied:

* Numerical missing values → **median imputation**
* Categorical missing values → **most-frequent imputation**
* Categorical features → **one-hot encoding**
* Stratified **80/20 train-validation split**
* Class balancing using `class_weight='balanced'`

Preprocessing and model training were combined using scikit-learn pipelines.

## 🤖 Models Evaluated

| Model                        |   Accuracy |  Macro-F1 | Weighted-F1 | Misclassification Cost |
| ---------------------------- | ---------: | --------: | ----------: | ---------------------: |
| Random Forest                |     91.73% |     0.910 |       0.915 |                     43 |
| Logistic Regression          |     90.98% |     0.921 |       0.910 |                     46 |
| Extra Trees                  | **93.23%** | **0.939** |   **0.931** |                 **35** |
| Cost-Sensitive Random Forest |     85.71% |     0.877 |       0.852 |                     33 |

Five-fold cross-validation for the Extra Trees model produced a mean Macro-F1 of **0.9363**.

## 🏆 Final Model

The final model used an **Extra Trees Classifier** with:

* 300 trees
* `class_weight='balanced'`
* `random_state=42`
* Parallel processing

The final model was retrained on all **664 training records** before generating predictions for the test dataset.

### Final Validation Results

* **Accuracy:** 93.23%
* **Macro-F1:** 0.9393
* **Weighted-F1:** 0.9313
* **Misclassification Cost:** 35

The provided misclassification cost matrix was also used to evaluate the cost of different prediction errors.

## 📁 Project Files

```text
MM26ML03-Healthcare-Triage/
├── README.md
├── MM26ML03_Healthcare_Triage.ipynb
└── MM2650_MM26ML03.csv
```

### Files

* `README.md` — Project documentation
* `MM26ML03_Healthcare_Triage.ipynb` — Complete machine learning workflow
* `MM2650_MM26ML03.csv` — Final test predictions and class probabilities

## 📤 Submission

The final submission file contains:

* Patient ID
* Predicted Triage
* RED Probability
* YELLOW Probability
* GREEN Probability
* BLACK Probability

The submission contains **112 predictions**, with no missing values, and the four class probabilities sum to approximately 1 for each patient.
