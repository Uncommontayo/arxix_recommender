"""
Mock all heavy dependencies (torch, faiss, sentence_transformers) at the
sys.modules level before app.py is imported. This lets tests run without
GPU drivers, CUDA DLLs, or the ~256 MB artifact files.
"""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ── MOCK HEAVY MODULES BEFORE ANY IMPORT ─────────────────────────
# Prevents torch / faiss DLL loading errors on Windows CI / test envs.
_faiss_mock = MagicMock()
_st_mock = MagicMock()

# torch.Tensor must be a real class so scipy's issubclass() checks don't crash.
class _FakeTensor:
    pass

_torch_mock = MagicMock()
_torch_mock.Tensor = _FakeTensor

sys.modules.setdefault("faiss", _faiss_mock)
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("sentence_transformers", _st_mock)

# ── MOCK DATASET ─────────────────────────────────────────────────
# 6 rows: top_k=5 and top_k+1=6 are both satisfiable.
MOCK_DF = pd.DataFrame(
    {
        "id": [
            "2101.00001",
            "2101.00002",
            "2101.00003",
            "2101.00004",
            "2101.00005",
            "2101.00006",
        ],
        "title": [
            "Deep Learning Survey",
            "Neural Networks Intro",
            "Machine Learning Basics",
            "Information Retrieval",
            "Natural Language Processing",
            "Computer Vision Review",
        ],
        "main_category": ["cs.AI", "cs.LG", "cs.LG", "cs.IR", "cs.CL", "cs.CV"],
        "text": [
            "deep learning neural network transformer attention",
            "neural network backpropagation gradient descent",
            "machine learning classification regression tree",
            "information retrieval search ranking bm25",
            "natural language processing text classification bert",
            "computer vision image recognition convolutional",
        ],
    }
)

N = len(MOCK_DF)   # 6 papers
EMBED_DIM = 384    # all-MiniLM-L6-v2 output dimension

# ── MOCK ARTIFACTS ───────────────────────────────────────────────
_mock_tfidf = MagicMock()
_mock_tfidf.transform.return_value = sp.random(1, N, density=0.5, format="csr")

_mock_bm25 = MagicMock()
_mock_bm25.get_scores.return_value = np.array(
    [0.5, 0.3, 0.8, 0.1, 0.6, 0.2], dtype="float32"
)

_mock_index = MagicMock()
_mock_index.search.return_value = (
    np.array([[0.95, 0.85, 0.75, 0.65, 0.55, 0.45]], dtype="float32"),
    np.array([[0, 1, 2, 3, 4, 5]], dtype="int64"),
)

_mock_sbert = MagicMock()
_mock_sbert.encode.return_value = np.random.rand(1, EMBED_DIM).astype("float32")

_mock_tfidf_matrix = sp.eye(N, format="csr", dtype="float32")
_mock_embeddings = np.random.rand(N, EMBED_DIM).astype("float32")

# ── PATCH ARTIFACT LOADING BEFORE IMPORT ─────────────────────────
# pickle.load is called twice: first for tfidf, then for bm25.
_pickle_calls = iter([_mock_tfidf, _mock_bm25])

_patches = [
    patch("os.path.exists", return_value=True),
    patch("pandas.read_parquet", return_value=MOCK_DF),
    patch("pickle.load", side_effect=lambda _: next(_pickle_calls)),
    patch("scipy.sparse.load_npz", return_value=_mock_tfidf_matrix),
    patch("numpy.load", return_value=_mock_embeddings),
]

for _p in _patches:
    _p.start()

import app  # noqa: E402 — must be imported after patches are active

for _p in _patches:
    _p.stop()

# Override module globals so tests use the mock objects directly.
app.df           = MOCK_DF
app.tfidf        = _mock_tfidf
app.tfidf_matrix = _mock_tfidf_matrix
app.bm25         = _mock_bm25
app.index        = _mock_index
app.sbert        = _mock_sbert

# Patch faiss.normalize_L2 to a no-op (modifies array in-place; mock is fine).
import faiss  # noqa: E402
faiss.normalize_L2 = lambda x: None
