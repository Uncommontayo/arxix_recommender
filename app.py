import logging
import os
import pickle

import faiss
import gradio as gr
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS = {
    "df":         os.path.join(BASE_DIR, "arxiv_df.parquet"),
    "tfidf":      os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"),
    "tfidf_mat":  os.path.join(BASE_DIR, "tfidf_matrix.npz"),
    "bm25":       os.path.join(BASE_DIR, "bm25_model.pkl"),
    "embeddings": os.path.join(BASE_DIR, "arxiv_embeddings.npy"),
    "faiss":      os.path.join(BASE_DIR, "faiss_index.index"),
}

RESULT_COLS = ["Rank", "id", "title", "main_category", "score", "arXiv Link"]
MAX_QUERY_LENGTH = 1_000
TOP_K = 5

SBERT_MODEL = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "7860"))
DEBUG       = os.environ.get("DEBUG", "false").lower() == "true"
SHARE       = os.environ.get("SHARE", "false").lower() == "true"

# ── LOAD ARTIFACTS ───────────────────────────────────────────────
def _load_artifacts() -> tuple:
    """Load all pre-built artifacts from disk. Raises FileNotFoundError on missing files."""
    missing = [name for name, path in ARTIFACTS.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact file(s): {', '.join(missing)}. "
            "Ensure all pre-built artifacts are present in the project directory."
        )

    logger.info("Loading artifacts...")

    df = pd.read_parquet(ARTIFACTS["df"])

    with open(ARTIFACTS["tfidf"], "rb") as f:
        tfidf = pickle.load(f)

    tfidf_matrix = sp.load_npz(ARTIFACTS["tfidf_mat"])

    with open(ARTIFACTS["bm25"], "rb") as f:
        bm25 = pickle.load(f)

    embeddings = np.load(ARTIFACTS["embeddings"]).astype("float32")

    index = faiss.read_index(ARTIFACTS["faiss"])

    sbert = SentenceTransformer(SBERT_MODEL)

    logger.info("All artifacts loaded successfully (%d papers).", len(df))
    return df, tfidf, tfidf_matrix, bm25, embeddings, index, sbert


df, tfidf, tfidf_matrix, bm25, embeddings, index, sbert = _load_artifacts()


# ── HELPERS ──────────────────────────────────────────────────────
def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLS)


def _error_result(message: str) -> pd.DataFrame:
    return pd.DataFrame({"Error": [message]})


def _build_result_df(indices: list[int], scores: np.ndarray) -> pd.DataFrame:
    """Assemble the standard result DataFrame from row indices and scores."""
    results = df.iloc[indices][["id", "title", "main_category"]].copy()
    results["score"] = np.round(scores, 4)
    results["arXiv Link"] = "https://arxiv.org/abs/" + results["id"].astype(str)
    results.insert(0, "Rank", range(1, len(results) + 1))
    return results.reset_index(drop=True)


def _validate_query(query: str) -> str | None:
    """Return an error message if the query is invalid, otherwise None."""
    if not query or not query.strip():
        return "Please enter a search query."
    if len(query) > MAX_QUERY_LENGTH:
        return f"Query is too long (max {MAX_QUERY_LENGTH} characters)."
    return None


# ── SEARCH FUNCTIONS ─────────────────────────────────────────────
def tfidf_search(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    vec = tfidf.transform([query.lower()])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    idx = np.argsort(scores)[::-1][:top_k]
    return _build_result_df(idx.tolist(), scores[idx])


def bm25_search(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    tokens = query.lower().split()
    scores = np.array(bm25.get_scores(tokens))
    idx = np.argsort(scores)[::-1][:top_k]
    return _build_result_df(idx.tolist(), scores[idx])


def faiss_search(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    q_emb = sbert.encode([query.lower()], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    # Fetch top_k+1 so we can drop the source paper itself when searching by title
    # (the source paper's text is used verbatim as the query, so it often ranks #1).
    scores, indices = index.search(q_emb, top_k + 1)
    return _build_result_df(indices[0][:top_k].tolist(), scores[0][:top_k])


def run_search(
    query: str, method: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Route query to the selected search method(s) and return three result DataFrames."""
    error = _validate_query(query)
    if error:
        empty = _empty_result()
        return empty, empty, empty

    try:
        if method == "TF-IDF":
            return tfidf_search(query), _empty_result(), _empty_result()
        elif method == "BM25":
            return _empty_result(), bm25_search(query), _empty_result()
        elif method == "SBERT + FAISS":
            return _empty_result(), _empty_result(), faiss_search(query)
        else:  # "Compare All Three"
            return tfidf_search(query), bm25_search(query), faiss_search(query)
    except Exception:
        logger.exception("Search failed for query=%r method=%r", query, method)
        err = _error_result("An unexpected error occurred. Please try again.")
        return err, err, err


def search_by_title(
    title: str | None, method: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find papers similar to the selected paper using its full text as the query."""
    if not title:
        empty = _empty_result()
        return empty, empty, empty

    row = df[df["title"] == title]
    if row.empty:
        err = _error_result(f"Paper not found: {title!r}")
        return err, err, err

    return run_search(row.iloc[0]["text"], method)


# ── GRADIO INTERFACE ─────────────────────────────────────────────
METHODS = ["TF-IDF", "BM25", "SBERT + FAISS", "Compare All Three"]
TITLES  = sorted(df["title"].tolist())

with gr.Blocks(title="Semantic Paper Recommender") as demo:
    gr.Markdown("# Semantic Research Paper Recommendation System")
    gr.Markdown("### MSc Project — Comparing TF-IDF, BM25, and SBERT+FAISS")

    with gr.Tabs():

        # ── TAB 1: Search by Query ────────────────────────────────
        with gr.Tab("Search by Query Text"):
            query_input = gr.Textbox(
                label="Enter your research query",
                placeholder="e.g. semantic similarity in information retrieval",
                max_lines=3,
            )
            method_1 = gr.Radio(
                choices=METHODS,
                value="Compare All Three",
                label="Select Search Method",
            )
            search_btn = gr.Button("Search", variant="primary")

            gr.Markdown("#### TF-IDF Results")
            out_tfidf_1 = gr.Dataframe()
            gr.Markdown("#### BM25 Results")
            out_bm25_1  = gr.Dataframe()
            gr.Markdown("#### SBERT + FAISS Results")
            out_faiss_1 = gr.Dataframe()

            search_btn.click(
                fn=run_search,
                inputs=[query_input, method_1],
                outputs=[out_tfidf_1, out_bm25_1, out_faiss_1],
            )

        # ── TAB 2: Search by Paper Title ──────────────────────────
        with gr.Tab("Search by Paper Title"):
            title_input = gr.Dropdown(
                choices=TITLES,
                label="Select a paper from the dataset",
                filterable=True,
            )
            method_2 = gr.Radio(
                choices=METHODS,
                value="Compare All Three",
                label="Select Search Method",
            )
            title_btn = gr.Button("Find Similar Papers", variant="primary")

            gr.Markdown("#### TF-IDF Results")
            out_tfidf_2 = gr.Dataframe()
            gr.Markdown("#### BM25 Results")
            out_bm25_2  = gr.Dataframe()
            gr.Markdown("#### SBERT + FAISS Results")
            out_faiss_2 = gr.Dataframe()

            title_btn.click(
                fn=search_by_title,
                inputs=[title_input, method_2],
                outputs=[out_tfidf_2, out_bm25_2, out_faiss_2],
            )

# ── LAUNCH ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting app on %s:%d (share=%s, debug=%s)", SERVER_HOST, SERVER_PORT, SHARE, DEBUG)
    demo.launch(server_name=SERVER_HOST, server_port=SERVER_PORT, debug=DEBUG, share=SHARE)
