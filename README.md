# Multimodal Emotion Recognition System

A deep learning system that combines **CNN-based visual encoding** and **BERT-based textual encoding** to classify emotions from image-text pairs, augmented with a **RAG pipeline** for context-aware downstream responses.

Built on the [Memotion Dataset 7k](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k) — a real-world multimodal dataset of meme images paired with text and sentiment labels.

---

## Demo

Upload a meme image + enter its caption → get a predicted emotion + a RAG-generated contextual response.

```
Image + Text  →  CNN Encoder + BERT Encoder  →  Fusion Layer  →  Emotion Label
                                                                        ↓
                                                               FAISS Retriever
                                                                        ↓
                                                            Context-aware Response
```

---

## Features

- **Multimodal Fusion** — ResNet18 (image) and BERT (text) embeddings concatenated and passed through a classifier head
- **5-class Emotion Classification** — very_positive, positive, neutral, negative, very_negative
- **RAG Pipeline** — LangChain + FAISS + HuggingFace sentence-transformers for retrieval-augmented response generation
- **Interactive Demo** — Streamlit web app for real-time inference

---

## Project Structure

```
multimodal-emotion-recognition/
│
├── data/
│   ├── raw/                        # Memotion 7k dataset (not tracked in git)
│   ├── processed/                  # train.csv, val.csv, test.csv
│   └── data_prep.py                # cleaning, label encoding, train/val/test split
│
├── src/
│   ├── dataset.py                  # PyTorch Dataset class (image + text + label)
│   ├── models/
│   │   ├── image_encoder.py        # ResNet18 CNN encoder
│   │   ├── text_encoder.py         # BERT encoder
│   │   ├── fusion.py               # ConcatFusion and CrossAttentionFusion
│   │   └── classifier.py           # Full multimodal model
│   ├── train.py                    # Training loop with checkpointing
│   ├── evaluate.py                 # Test set evaluation, confusion matrix, F1
│   └── utils.py                    # Seed, logging helpers
│
├── rag/
│   ├── build_index.py              # Embed knowledge base, save FAISS index
│   ├── retriever.py                # Similarity search + response generation
│   └── knowledge_base/             # FAISS index files
│
├── app/
│   └── demo_app.py                 # Streamlit demo
│
├── experiments/
│   ├── logs/                       # Training loss/accuracy curves
│   └── checkpoints/                # Saved model weights
│
├── notebooks/
│   ├── 01_data_prep_eda.ipynb
│   ├── 02_train_baseline_model.ipynb
│   └── 03_rag_pipeline.ipynb
│
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_rag.py
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Visual Encoder | ResNet18 (PyTorch / torchvision) |
| Text Encoder | BERT base uncased (HuggingFace Transformers) |
| Fusion | Concat + MLP classifier head |
| RAG | LangChain + FAISS + sentence-transformers |
| Demo | Streamlit |
| Dataset | Memotion Dataset 7k (Kaggle) |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
```

**2. Create and activate conda environment**
```bash
conda create -n emotion_env python=3.10
conda activate emotion_env
```

**3. Install dependencies**
```bash
pip install torch torchvision transformers pandas numpy scikit-learn pillow matplotlib
pip install langchain langchain-community langchain-huggingface langchain-core faiss-cpu sentence-transformers
pip install streamlit pytest
```

**4. Download the dataset**

Download [Memotion Dataset 7k](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k) from Kaggle and unzip into:
```
data/raw/memotion_dataset_7k/
    images/
    labels.csv
```

---

## How to Run

All commands must be run from the **project root directory**.

**Step 1 — Data preparation**
```bash
python data/data_prep.py
```
Outputs `train.csv`, `val.csv`, `test.csv` into `data/processed/`.

**Step 2 — Train the model**
```bash
python -m src.train
```
Trains for 5 epochs, saves weights to `experiments/checkpoints/baseline_model.pt`.

**Step 3 — Evaluate on test set**
```bash
python -m src.evaluate
```
Prints per-class precision, recall, F1-score, and confusion matrix.

**Step 4 — Build RAG index**
```bash
python -m rag.build_index
```
Embeds the knowledge base and saves a FAISS index to `rag/knowledge_base/faiss_index/`.

**Step 5 — Launch the demo**
```bash
streamlit run app/demo_app.py
```
Opens at `http://localhost:8501`. Upload an image, enter text, get a predicted emotion and a RAG-generated response.

**Run tests**
```bash
pytest tests/
```

---

## Model Architecture

```
Input Image (224x224x3)
        │
   ResNet18 (pretrained, layer4 unfrozen)
        │
  image_feat (512-d)
        │
        ├───────────────────────────┐
        │                           │
Input Text (tokenized, max_len=32)  │
        │                           │
  BERT base uncased (frozen)        │
        │                           │
  text_feat (768-d)                 │
        │                           │
        └──────── concat ───────────┘
                     │
               fused (1280-d)
                     │
              Linear(1280→256)
                     │
                   ReLU
                     │
                Dropout(0.3)
                     │
              Linear(256→5)
                     │
              Emotion Label
```

---

## Training Results

| Epoch | Loss | Train Accuracy |
|---|---|---|
| 1 | 1.3075 | 42.7% |
| 2 | 1.2499 | 45.9% |
| 3 | 1.0847 | 52.9% |
| 4 | 0.7239 | 72.7% |
| 5 | 0.3766 | 86.9% |

---

## RAG Pipeline

Once an emotion is predicted, the system queries a FAISS vector index using a sentence-transformer embedding. The top-2 most relevant documents from the knowledge base are retrieved and returned as a contextual response.

```
Predicted Emotion
       │
  Query: "How to respond to someone feeling {emotion}"
       │
  sentence-transformers/all-MiniLM-L6-v2
       │
  FAISS similarity_search (k=2)
       │
  Retrieved Documents → generate_response()
```

---

## Dataset

**Memotion Dataset 7k**
- ~7,000 meme images paired with OCR-extracted and corrected text
- Labels: `very_positive`, `positive`, `neutral`, `negative`, `very_negative`
- Source: [Kaggle — williamscott701/memotion-dataset-7k](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k)

---

## Future Work

- Upgrade fusion from concat to **cross-attention** (CrossAttentionFusion already implemented in `src/models/fusion.py`)
- Fine-tune BERT encoder layers instead of freezing
- Evaluate on **CMU-MOSEI** for cross-dataset generalization
- Replace template RAG responses with an LLM call (Groq / Llama 3)
- Add confidence scores to the Streamlit demo

---

## Author

**Priyal Parmar**
