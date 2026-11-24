# Project 15: Extractive Question Answering System

**Course:** Natural Language Processing and Text Mining  
**Category:** Category 3 — Language Generation & Sequence-to-Sequence / Transformer Tasks  
**Dataset:** Full SQuAD v1.1 (`rajpurkar/squad` — 87,599 train / 10,570 validation)  
**Core Libraries:** Hugging Face (`transformers`, `datasets`), `NLTK`, `Scikit-learn`, `spaCy`

## Overview
This project implements an end-to-end **Extractive Question Answering (QA)** system following the standard 4-stage NLP pipeline across all four required course libraries:
1. **Data Loading & Exploratory Data Analysis (EDA):** Loads the complete SQuAD v1.1 dataset via Hugging Face `datasets`, computes passage/question/answer statistics with `nltk`, and performs Named Entity Recognition (NER) and Part-of-Speech (POS) linguistic profiling with `spaCy` (`en_core_web_sm`).
2. **Preprocessing:**
   - *Classical NLP:* Sentence segmentation (`nltk.sent_tokenize`), word tokenization, stop-word filtering, and TF-IDF vectorization (`scikit-learn`).
   - *Transformer:* WordPiece tokenization (`distilbert-base-uncased`) with sliding windows (`max_length=384`, `doc_stride=128`) and character-to-token span alignment (`start_positions`, `end_positions`).
3. **Model Implementation:**
   - **Baseline Model (`TfidfSentenceBaseline`):** Classical TF-IDF + cosine similarity sentence retrieval baseline.
   - **Extractive Transformer (`DistilBERT`):** Fine-tuned `distilbert-base-uncased` (`AutoModelForQuestionAnswering`) predicting token-level start and end logits with mixed precision (`fp16`) over all 87,599 training examples (88,524 sliding-window features).
4. **Evaluation & Interactive Demo:** Evaluates both models across all 10,570 validation examples using official SQuAD v1.1 **Exact Match (EM)** and **Token-Level F1**, plots training/validation loss curves, analyzes performance by WH-question type and **spaCy Named Entity type**, inspects qualitative successes and errors, and exposes an interactive **Gradio** web interface (`app.py`).

## Project Structure
```text
Proiect NLP/
├── Question_Answering_System.ipynb   # Executed end-to-end NLP pipeline notebook (Full SQuAD)
├── qa_pipeline.py                    # Reusable data, TF-IDF baseline, spaCy NER/POS & DistilBERT utilities
├── app.py                            # Interactive Gradio demo with span highlighting & spaCy entity badge
├── Project_15_QA_Report.docx         # Full academic technical report with tables & figures
├── Project_15_QA_Presentation.pptx   # 12-slide widescreen presentation with speaker notes
├── assets/                           # Exported high-resolution matplotlib figures
├── requirements.txt                  # Python dependencies
├── .gitignore
├── README.md
└── saved_model/
    └── distilbert-squad/             # Local full-SQuAD fine-tuned DistilBERT checkpoint & tokenizer
```

## Setup & Installation
1. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install PyTorch (CUDA 12.4 for NVIDIA GPUs) and the remaining dependencies:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
   *(For CPU-only environments, `pip install -r requirements.txt` installs PyTorch automatically.)*

## Running the Project
### 1. Jupyter Notebook
Open and run `Question_Answering_System.ipynb` in Jupyter or VS Code (select the `.venv` Python interpreter).
- Configured with `USE_FULL_DATASET = True` (`87,599` training examples, `10,570` validation examples, 2 epochs in `fp16` — 56.34 min on an NVIDIA GeForce RTX 3050 Laptop GPU, peak VRAM `1,759.0 MB`).

### 2. Interactive Gradio Demo
Launch the interactive web application:
```powershell
.\.venv\Scripts\python.exe app.py
```
Then open the local URL printed in the terminal (typically `http://127.0.0.1:7860`).

## Evaluation Results (Full SQuAD v1.1 Validation Set, `n = 10,570`)

### Overall Model Comparison
| Model | Exact Match (EM) | Token F1 Score | Notes |
| --- | ---: | ---: | --- |
| **TF-IDF Sentence Baseline** (`NLTK` + `Scikit-learn`) | 0.26% | 19.38% | Retrieves whole sentences (28.30s eval) |
| **Fine-Tuned DistilBERT** (`distilbert-base-uncased`, 2 epochs) | **77.08%** | **85.41%** | Full SQuAD v1.1 (`56.34 min` train, `1,759.0 MB` peak VRAM) |

### DistilBERT Performance by spaCy Answer Named Entity Type (`n = 10,570`)
| spaCy Entity Type | Examples | Exact Match (%) | F1 Score (%) |
| --- | ---: | ---: | ---: |
| `DATE` | 1,078 | 88.68% | 92.53% |
| `PERSON` | 1,132 | 83.92% | 88.93% |
| `CARDINAL` | 983 | 81.38% | 87.73% |
| `GPE` | 603 | 79.77% | 87.73% |
| `ORG` | 1,173 | 77.41% | 86.19% |
| `NORP` | 331 | 70.39% | 84.15% |
| `LOC` | 122 | 69.67% | 80.39% |
| `NON-ENTITY` | 4,591 | 71.88% | 81.84% |
