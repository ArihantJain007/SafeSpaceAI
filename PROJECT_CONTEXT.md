# Project Context — AI Mental Health Support Chatbot

This document serves as the authoritative state of the **AI Mental Health Support Chatbot** project across development environments and coding platforms.

---

## 1. Project Objective
Build an empathetic AI Mental Health Support Chatbot that categorizes user statements into mental health categories with high confidence and provides compassionate, non-diagnostic conversational responses.

---

## 2. Locked Architecture
The application architecture is strictly locked as follows:

```
User Statement
  │
  ▼
Streamlit UI
  │
  ▼
Fine-tuned DistilBERT Classifier  ───► Predicted mental health category + confidence score
  │
  ▼
Gemini API (Conversational Model) ───► Compassionate response contextualized by category
  │
  ▼
Streamlit UI
```

### Constraints & Architectural Boundaries:
- **DistilBERT**: Used ONLY as the text classification model.
- **Gemini API**: Used ONLY as the conversational response generation model.
- **No RAG**: Retrieval-Augmented Generation is NOT permitted.
- **No Embeddings / Vector Databases**: Vector search or custom embedding index is NOT permitted.
- **No Gemini Classification**: Gemini must not be used as the classifier.
- **No Model Swapping**: DistilBERT must not be replaced with another final classifier.

---

3. Current Phase

PHASE 3 — FINE-TUNE DISTILBERT CLASSIFIER (Completed & Verified)

Phase 3 implementation and full GPU training are complete.

src/models/distilbert_classifier.py — robust training & smoke test pipeline supporting class-weighted loss, CUDA/CPU fallback, checkpoint save/resume, validation Macro F1 selection, and single test set evaluation.
notebooks/03_distilbert_finetuning.ipynb — Colab-ready notebook with Google Drive artifact persistence.
Local Smoke Test: Passed successfully on CPU.
Full Training: Completed successfully on Google Colab using an NVIDIA Tesla T4 GPU.
Best Validation Macro F1: 0.7995
Final Test Macro F1: 0.7854
Final Test Accuracy: 0.8194
Final Test Weighted F1: 0.8203

---

4. Dataset Source & Status
Kaggle Dataset: Sentiment Analysis for Mental Health
Raw Path: data/raw/Combined Data.csv (53,043 rows)
Cleaned Path: data/processed/cleaned_mental_health_data.csv (51,055 rows)
Dataset Cleaning Summary

The raw dataset was conservatively cleaned before model development.

362 missing statement values were removed.
Exact duplicate statements were removed.
18 unique statements with conflicting labels were removed, corresponding to 49 rows.
No stopword removal, stemming, lemmatization, blanket punctuation removal, or emoji removal was performed.
Raw data was preserved separately from processed data.

---

5. Environment & Setup Details
Local Environment
Python Version: 3.13.3
Virtual Environment: .venv
Local Package Versions
pandas: 3.0.5
numpy: 2.5.2
scikit-learn: 1.9.0
matplotlib: 3.11.1
seaborn: 0.13.2
joblib: 1.5.3
jupyter: 1.1.1 (notebook 7.6.2)
torch: 2.13.0+cpu
transformers: 5.15.1
Colab Training Environment
Platform: Google Colab
GPU: NVIDIA Tesla T4
PyTorch: 2.11.0+cu128
CUDA: Available

The local machine was used for implementation and smoke testing. Full DistilBERT training was performed on the Colab GPU.

---

## 6. Project Structure & Files Created
```
mental-health-chatbot/
│
├── .venv/                         # Local virtual environment (ignored by git)
│
├── data/
│   ├── raw/
│   │   └── Combined Data.csv     # Raw Kaggle dataset (53,043 rows)
│   └── processed/
│       ├── cleaned_mental_health_data.csv # Processed dataset (51,055 rows)
│       └── splits/               # Persisted train/val/test CSV splits
│           ├── train.csv         # 40,844 rows (80%)
│           ├── validation.csv    # 5,105 rows (10%)
│           └── test.csv          # 5,106 rows (10%)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb # Data exploration & cleaning documentation
│   ├── 02_baseline_classifier.ipynb # Baseline classifier experiments & analysis
│   └── 03_distilbert_finetuning.ipynb # DistilBERT fine-tuning (Colab GPU)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── cleaning.py          # Modular, executable dataset cleaning pipeline
│   └── models/
│       ├── __init__.py
│       ├── baseline.py          # Modular baseline training & evaluation pipeline
│       └── distilbert_classifier.py # DistilBERT fine-tuning & smoke test pipeline
│
├── models/
│   ├── baseline/                # Persisted vectorizer, metadata & model artifacts
│   │   ├── metadata.json
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── linear_svc.joblib
│   │   ├── cm_val_logistic_regression.png
│   │   ├── cm_val_linear_svc.png
│   │   └── cm_test_selected_baseline.png
│   └── distilbert/              # DistilBERT artifacts (Colab output on Google Drive)
│       ├── checkpoints/
│       ├── best_model/
│       ├── tokenizer/
│       ├── label_mapping.json
│       ├── metadata.json
│       └── plots/
│
├── app/
│   └── .gitkeep                  # Placeholder for future Streamlit UI app
│
├── PROJECT_CONTEXT.md            # Authoritative project state document
├── README.md                     # Setup and execution guide
├── requirements.txt              # Phase 1–3 dependencies
└── .gitignore                    # Version control exclusion rules
```

---

## 7. Data Splits & Label Consistency

### Stratified Splits (80% / 10% / 10%, `random_state=42`)
- **Train Set**: `40,844` rows (`80.00%`)
- **Validation Set**: `5,105` rows (`10.00%`)
- **Test Set**: `5,106` rows (`10.00%`)
- **Total Rows**: `51,055` rows
- **Data Leakage Check**: Confirmed **zero statement overlap** across train, validation, and test splits.

### Canonical Label Set & Ordering (7 Classes)
Fixed canonical order used consistently across all evaluation reports, confusion matrices, and future phases:
1. `Anxiety`
2. `Bipolar`
3. `Depression`
4. `Normal`
5. `Personality disorder`
6. `Stress`
7. `Suicidal`

### Class Distribution Across Splits

| Target Label | Total Count | Train Count (80%) | Validation Count (10%) | Test Count (10%) |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 16,037 | 12,830 | 1,603 | 1,604 |
| **Depression** | 15,078 | 12,062 | 1,508 | 1,508 |
| **Suicidal** | 10,634 | 8,507 | 1,063 | 1,064 |
| **Anxiety** | 3,617 | 2,894 | 361 | 362 |
| **Bipolar** | 2,501 | 2,001 | 250 | 250 |
| **Stress** | 2,293 | 1,834 | 230 | 229 |
| **Personality disorder** | 895 | 716 | 90 | 89 |
| **TOTAL** | **51,055** | **40,844** | **5,105** | **5,106** |

---

## 8. Feature Extraction & Baseline Models Setup

### TF-IDF Vectorizer Configuration
- `ngram_range`: `(1, 2)` (unigrams + bigrams)
- `min_df`: `2`, `max_df`: `0.95`
- `sublinear_tf`: `True`
- **Fitted ONLY on Train set** text: Produced `274,360` vocabulary features.

### Classifier Models
1. **Logistic Regression**: `class_weight='balanced'`, `random_state=42`, `max_iter=1000`
2. **LinearSVC**: `class_weight='balanced'`, `random_state=42`, `max_iter=2000`

---

## 9. Validation Results & Baseline Selection

### Validation Metrics Comparison (Primary Metric: Macro F1)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7798 | 0.7218 | **0.7570** | 0.7349 | **0.7873** | 0.7798 | 0.7796 |
| **LinearSVC (WINNER)** | **0.7843** | **0.7697** | 0.7201 | **0.7392** | 0.7825 | **0.7843** | **0.7821** |

### Winning Baseline Selection:
- **Selected Winner**: **LinearSVC** (Validation Macro F1 = `0.7392` vs Logistic Regression = `0.7349`).

---

## 10. Per-Class Performance Breakdown

### Validation Set Per-Class Results (LinearSVC):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.8974 | 0.9495 | **0.9227** | 1,603 |
| **Anxiety** | 0.7932 | 0.8393 | **0.8156** | 361 |
| **Bipolar** | 0.8578 | 0.7720 | **0.8126** | 250 |
| **Depression** | 0.7424 | 0.6976 | **0.7193** | 1,508 |
| **Suicidal** | 0.6828 | 0.7168 | **0.6994** | 1,063 |
| **Personality disorder** | 0.8246 | 0.5222 | **0.6395** | 90 |
| **Stress** | 0.5896 | 0.5435 | **0.5656** | 230 |

### Final Test Set Evaluation (LinearSVC — Evaluated EXACTLY ONCE):
- **Test Accuracy**: `0.7763`
- **Test Macro Precision**: `0.7522`
- **Test Macro Recall**: `0.6959`
- **Test Macro F1**: `0.7165`
- **Test Weighted F1**: `0.7737`

Baseline Limitations
Strong performance on the majority Normal class.
Lower recall and F1 for Stress and Personality disorder.
Linear TF-IDF features have difficulty representing contextual distinctions between semantically overlapping categories such as Depression and Suicidal.

#### Test Set Per-Class Detailed Results:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.8986 | 0.9451 | **0.9213** | 1,604 |
| **Anxiety** | 0.7842 | 0.8232 | **0.8032** | 362 |
| **Bipolar** | 0.7675 | 0.7000 | **0.7322** | 250 |
| **Depression** | 0.7302 | 0.7036 | **0.7166** | 1,508 |
| **Suicidal** | 0.6730 | 0.7002 | **0.6863** | 1,064 |
| **Stress** | 0.6517 | 0.5721 | **0.6093** | 229 |
| **Personality disorder** | 0.7600 | 0.4270 | **0.5468** | 89 |

---

11. Phase 3 — DistilBERT Fine-Tuning
Model & Implementation
Model: distilbert-base-uncased
Module: src/models/distilbert_classifier.py
Notebook: notebooks/03_distilbert_finetuning.ipynb
Tokenizer: DistilBertTokenizerFast
Maximum Sequence Length: 256
Loss: Class-weighted CrossEntropyLoss
Class weights calculated using training labels only.
Checkpoint Selection: Validation Macro F1
Test Evaluation: Held-out test set evaluated once after best-model selection.
CUDA Guard: Full training exits safely when CUDA is unavailable.
Checkpoint Resume: Supported through the training pipeline.
Tokenization Statistics
Median token length: 75
90th percentile: 328
Maximum sequence length: 256
Training truncation rate: 15.17%
Validation truncation rate: 15.02%
Test truncation rate: 15.53%
Training Configuration
Training Platform: Google Colab
GPU: NVIDIA Tesla T4
PyTorch: 2.11.0+cu128
Epochs: 3
Batch Size: 16
Learning Rate: 2e-5
Weight Decay: 0.01
Warmup: 10%
Random Seed: 42
Class Weights

Calculated exclusively from the training set:

Label	Weight
Anxiety	2.0162
Bipolar	2.9160
Depression	0.4837
Normal	0.4548
Personality disorder	8.1492
Stress	3.1815
Suicidal	0.6859
Validation Results by Epoch
Epoch	Validation Accuracy	Validation Macro F1	Validation Weighted F1
1	0.7926	0.7594	0.7936
2	0.8176	0.7969	0.8190
3	0.8251	0.7995	0.8257
Best Checkpoint
Best Epoch: 3
Best Checkpoint: checkpoint-7659
Validation Macro F1: 0.7995
Final Test Results

The selected best checkpoint was evaluated once on the held-out test set.

Metric	DistilBERT
Test Accuracy	0.8194
Test Macro F1	0.7854
Test Weighted F1	0.8203
DistilBERT vs LinearSVC
Model	Validation Macro F1	Test Macro F1	Test Accuracy
LinearSVC	0.7392	0.7165	0.7763
DistilBERT	0.7995	0.7854	0.8194
Improvement Over Baseline
LinearSVC Test Macro F1: 0.7165
DistilBERT Test Macro F1: 0.7854
Absolute improvement: +0.0689
Relative improvement: approximately 9.62%
Training Runtime
Approximate full training runtime: 864 seconds (~14.4 minutes)
Local Inference Verification

The trained Colab model was downloaded locally and successfully loaded using the saved model and tokenizer.

Sample input:

I feel anxious and overwhelmed by everything today.

Prediction:

Predicted label: Anxiety
Prediction probability: approximately 99.71%

This verifies that the trained model can be loaded and used for local inference.

Saved Artifacts

The final model artifacts are stored:

Locally

models/distilbert/

Google Drive backup

MyDrive/mental-healthbot/models/distilbert/

Artifacts include:

best_model/
tokenizer/
label_mapping.json
metadata.json
checkpoints/
plots/

The trained model files are intentionally excluded from GitHub through .gitignore.

12. Exact Commands to Reproduce Phase 1 & Phase 2
Virtual Environment Activation
# Windows PowerShell
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
Run Data Cleaning Pipeline
python -m src.data.cleaning
Run Baseline Pipeline
python -m src.models.baseline
Run Baseline Notebook
jupyter notebook notebooks/02_baseline_classifier.ipynb
13. Phase 3 Reproduction & Colab Workflow
Local Smoke Test
python -m src.models.distilbert_classifier --smoke-test
Google Colab Training

Full training was performed successfully on an NVIDIA Tesla T4 GPU.

The Colab notebook is:

notebooks/03_distilbert_finetuning.ipynb

The Phase 2 split CSVs are loaded into the Colab workspace without recreating or reshuffling them.

Full Training Configuration
Model: distilbert-base-uncased
Epochs: 3
Batch Size: 16
Learning Rate: 2e-5
Max Length: 256
Loss: Class-weighted CrossEntropyLoss
Seed: 42
Selection Metric: Validation Macro F1
Google Drive Artifact Location
MyDrive/mental-healthbot/models/distilbert/

The Google Drive copy is the persistent backup of the trained model and checkpoints.

---

## 14. Phase 4 — MentalBERT Fine-Tuning & Evaluation

### Implementation Summary

- Model: `mental/mental-bert-base-uncased`
- Module: `src/models/mental_bert_classifier.py`
- Notebook: `notebooks/04_mental_bert_classifier.ipynb`
- Architecture: BERT-base / MentalBERT
- Task: 7-class mental-health text classification
- Max sequence length: `256`
- Epochs: `3`
- Learning rate: `2e-5`
- Per-device training batch size: `8` in the actual Colab run
- Weight decay: `0.01`
- Seed: `42`
- Loss: class-weighted `CrossEntropyLoss`
- Class weights: calculated from training labels only
- Checkpoint selection: validation Macro F1
- Test evaluation: held-out test set evaluated once after model selection
- Training environment: Google Colab with Tesla T4 GPU
- Transformers compatibility: `warmup_steps` was used with dynamic calculation as 10% of total training steps; `warmup_ratio` and `logging_dir` were not used because of the Colab Transformers API version

### Dataset / Splits

Use the existing Phase 2 splits without regeneration:

- Train: `40,844`
- Validation: `5,105`
- Test: `5,106`

Canonical class ordering:

1. Anxiety
2. Bipolar
3. Depression
4. Normal
5. Personality disorder
6. Stress
7. Suicidal

### MentalBERT Results

Validation:
- Macro F1: `0.8146602122`
- Accuracy: `0.8374142997`

Held-out Test:
- Macro F1: `0.8008539552`
- Accuracy: `0.8360752056`
- Weighted F1: `0.8366941635`

For concise reporting elsewhere, use:
- Validation Macro F1: `0.8147`
- Test Macro F1: `0.8009`
- Test Accuracy: `0.8361`
- Test Weighted F1: `0.8367`

### Test Set Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Anxiety | 0.8753 | 0.8729 | 0.8741 | 362 |
| Bipolar | 0.8595 | 0.8320 | 0.8455 | 250 |
| Depression | 0.8151 | 0.7513 | 0.7819 | 1508 |
| Normal | 0.9649 | 0.9595 | 0.9622 | 1604 |
| Personality disorder | 0.6875 | 0.6180 | 0.6509 | 89 |
| Stress | 0.7027 | 0.7948 | 0.7459 | 229 |
| Suicidal | 0.7091 | 0.7857 | 0.7454 | 1064 |

### Test Confusion Matrix Findings

1. The largest cross-class confusion is between Depression and Suicidal:
   - Actual Depression → Predicted Suicidal: `327`
   - Actual Suicidal → Predicted Depression: `213`

2. Personality disorder remains the lowest-F1 class:
   - F1: `0.6509`
   - Recall: `0.6180`
   - Test support: `89`

3. Normal has the highest F1:
   - F1: `0.9622`

4. Anxiety and Bipolar also show relatively strong test F1:
   - Anxiety: `0.8741`
   - Bipolar: `0.8455`

### Benchmark Comparison

| Model | Test Macro-F1 |
| :--- | :--- |
| LinearSVC + TF-IDF | 0.7165 |
| DistilBERT | 0.7854 |
| MentalBERT | 0.8009 |
| MentalRoBERTa | Not yet trained |

- MentalBERT test Macro-F1 is `0.0155` higher than the existing DistilBERT test Macro-F1.
- MentalBERT test Macro-F1 is `0.0844` higher than the LinearSVC baseline.

### Artifacts

MentalBERT artifacts are stored under:

`models/mental_bert/`

including:
- `checkpoints/`
- `best_model/`
- `tokenizer/`
- `plots/`
- `label_mapping.json`
- `metadata.json`

### Reproduction

```bash
python -m src.models.mental_bert_classifier --smoke-test
```

---

## 15. Current Results Summary
Final Model

Fine-tuned DistilBERT / MentalBERT

Benchmark

LinearSVC Test Macro F1: 0.7165

Final DistilBERT Performance
Validation Macro F1: 0.7995
Test Macro F1: 0.7854
Test Accuracy: 0.8194
Test Weighted F1: 0.8203

Final MentalBERT Performance
Validation Macro F1: 0.8147
Test Macro F1: 0.8009
Test Accuracy: 0.8361
Test Weighted F1: 0.8367

Conclusion

Fine-tuning DistilBERT and MentalBERT produced substantial improvements over the classical LinearSVC baseline on the held-out test set.

---

## 16. Next Phase

PHASE 5 — STREAMLIT UI + GEMINI INTEGRATION

Planned work:

Build the Streamlit chat interface.
Load the trained classifier model and tokenizer.
Accept user text input.
Generate a mental-health category prediction and prediction probability.
Pass the user statement and classifier context to the Gemini API.
Generate a compassionate, non-diagnostic conversational response.
Display the conversation through the Streamlit interface.

The architecture remains:

User Statement
      ↓
Streamlit UI
      ↓
Classifier (MentalBERT / DistilBERT)
      ↓
Category + Prediction Probability
      ↓
Gemini API
      ↓
Conversational Response
      ↓
Streamlit UI

