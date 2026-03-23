---
title: Arxiv Recommender
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
---

# Semantic Research Paper Recommendation System

An MSc project that compares three classical information-retrieval methods for recommending arXiv research papers. Search by natural-language query or select a paper from the dataset to find similar work.

---

## Methods Compared

| Method | Type | Description |
|---|---|---|
| **TF-IDF** | Statistical | Term-frequency weighting with cosine similarity |
| **BM25** | Probabilistic | Okapi BM25 ranking over tokenised paper text |
| **SBERT + FAISS** | Semantic | Dense embeddings via `all-MiniLM-L6-v2` indexed with FAISS |

---

## Interface

The app has two tabs:

- **Search by Query Text** — enter a free-text research query
- **Search by Paper Title** — pick a paper from the dataset; its full text is used as the query

Both tabs support running a single method or **Compare All Three** side-by-side.

---

## Project Structure

```
arxiv_recommender/
├── app.py               # Main Gradio application
├── requirements.txt     # Python dependencies
├── .gitignore
└── artifacts/           # Pre-built artifacts (not in repo — see below)
    ├── arxiv_df.parquet
    ├── tfidf_vectorizer.pkl
    ├── tfidf_matrix.npz
    ├── bm25_model.pkl
    ├── arxiv_embeddings.npy
    └── faiss_index.index
```

> **Note:** The artifact files (~256 MB total) are excluded from version control. Place them in the project root before running the app.

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/Uncommontayo/arxix_recommender.git
cd arxix_recommender
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the artifact files**

Download the pre-built artifacts and place them in the project root:

| File | Size | Description |
|---|---|---|
| `arxiv_df.parquet` | ~46 MB | Paper metadata (id, title, category, text) |
| `tfidf_vectorizer.pkl` | ~183 KB | Fitted `TfidfVectorizer` |
| `tfidf_matrix.npz` | ~30 MB | Sparse TF-IDF matrix |
| `bm25_model.pkl` | ~42 MB | Pre-trained `BM25Okapi` model |
| `arxiv_embeddings.npy` | ~69 MB | SBERT dense embeddings |
| `faiss_index.index` | ~69 MB | FAISS flat L2 index |

**5. Run the app**
```bash
python app.py
```

Open `http://localhost:7860` in your browser.

---

## Configuration

All runtime settings are controlled via environment variables — no code changes needed.

| Variable | Default | Description |
|---|---|---|
| `SERVER_HOST` | `127.0.0.1` | Host to bind the server to |
| `SERVER_PORT` | `7860` | Port to listen on |
| `DEBUG` | `false` | Enable Gradio debug mode |
| `SHARE` | `false` | Create a public Gradio tunnel |
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model name |

Example — expose a public share link:
```bash
SHARE=true python app.py
```

---

## Tech Stack

- [Gradio](https://gradio.app/) — web interface
- [Sentence-Transformers](https://www.sbert.net/) — SBERT embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 ranking
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorisation
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — data handling
