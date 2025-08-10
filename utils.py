import os, pickle, numpy as np, faiss
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
os.makedirs(DB_DIR, exist_ok=True)
FAISS_PATH = os.path.join(DB_DIR, "index.faiss")
NAMES_PATH = os.path.join(DB_DIR, "names.pkl")
EMBED_DIM = 512

def make_faiss_index(dim=EMBED_DIM):
    index = faiss.IndexFlatL2(dim)
    id_index = faiss.IndexIDMap(index)
    return id_index

def load_faiss():
    if not os.path.exists(FAISS_PATH):
        idx = make_faiss_index()
        return idx
    return faiss.read_index(FAISS_PATH)

def save_faiss(index):
    faiss.write_index(index, FAISS_PATH)

def load_names():
    if not os.path.exists(NAMES_PATH):
        return {}
    with open(NAMES_PATH, "rb") as f:
        return pickle.load(f)

def save_names(names: Dict[int, str]):
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(names, f)

def next_id(names: Dict[int, str]) -> int:
    if not names:
        return 1
    return max(names.keys()) + 1

def l2_normalize(v: np.ndarray):
    v = np.array(v, dtype='float32')
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
