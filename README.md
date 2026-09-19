# MM26ML03-Healthcare-Triage
Machine learning project for emergency patient START triage classification using clinical and demographic features.
# Healthcare Triage Classification — MM26ML03

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
