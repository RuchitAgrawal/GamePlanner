"""
Central config for paths, hyperparameters, and model settings.
Import from here rather than hardcoding values across files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_EMBEDDINGS_DIR = ROOT_DIR / "data" / "embeddings"
MODELS_DIR = ROOT_DIR / "models"

TRAIN_PATH = DATA_PROCESSED_DIR / "train.parquet"
VAL_PATH = DATA_PROCESSED_DIR / "val.parquet"
TEST_PATH = DATA_PROCESSED_DIR / "test.parquet"
ITEMS_META_PATH = DATA_PROCESSED_DIR / "items_metadata.parquet"

USER2IDX_PATH = DATA_PROCESSED_DIR / "user2idx.json"
ITEM2IDX_PATH = DATA_PROCESSED_DIR / "item2idx.json"

ITEM_VECTORS_PATH = DATA_EMBEDDINGS_DIR / "item_vectors.npy"
FAISS_INDEX_PATH = DATA_EMBEDDINGS_DIR / "faiss_index.bin"
IDX_TO_ITEMID_PATH = DATA_EMBEDDINGS_DIR / "idx_to_itemid.json"

EXPERIMENTS_PATH = ROOT_DIR / "data" / "processed" / "experiments.json"
CACHE_PATH = ROOT_DIR / "explanations_cache.json"

# ── Data preprocessing ────────────────────────────────────────────────────────

MIN_USER_INTERACTIONS = 5   # drop users below this threshold
MIN_ITEM_INTERACTIONS = 10  # drop items below this threshold
CONFIDENCE_ALPHA = 40       # weight for log-scaled playtime confidence scores

# ── Model hyperparameters ──────────────────────────────────────────────────────

EMBEDDING_DIM = 64
BATCH_SIZE = 512
NUM_NEGATIVES = 4       # negative samples per positive in training
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-4      # lower LR for NeuMF joint fine-tuning stage
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5  # epochs without val NDCG improvement before stopping
LR_SCHEDULER_PATIENCE = 3

MLP_LAYERS = [128, 64, 32]  # hidden layer sizes for NeuMF MLP branch
DROPOUT = 0.2

TOP_K = [5, 10]  # evaluation cutoffs

# ── RAG / LLM ─────────────────────────────────────────────────────────────────

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

LLM_MAX_TOKENS = 150
LLM_TEMPERATURE = 0.3

FAISS_CANDIDATES = 50   # retrieve this many candidates before CF re-ranking
COLD_START_MULTIPLIER = 3  # retrieve k * this many candidates for cold-start

# ── API ───────────────────────────────────────────────────────────────────────

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
DEFAULT_TOP_K = 10
