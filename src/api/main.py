"""
FastAPI application entrypoint.

Loads all models, indexes, and shared state into app.state at startup
so every request handler can access them without re-loading from disk.
"""

import json
import logging
from contextlib import asynccontextmanager

import pandas as pd
import torch
from fastapi import FastAPI

from src.api.routes import router
from src.models.matrix_factorization import GMF
from src.models.baseline import PopularityRecommender
from src.rag.vector_store import VectorStore
from src.rag.llm_client import LLMClient
from src.rag.explainability import ExplainabilityPipeline
from src.rag.coldstart import ColdStartHandler
from src.rag.conversational import ConversationalRecommender
from src.utils.cache import ExplanationCache
from src.utils.config import (
    API_PREFIX,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MODELS_DIR, USER2IDX_PATH, ITEM2IDX_PATH,
    TRAIN_PATH, ITEMS_META_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load all models and shared state once at startup, release at shutdown."""
    await _startup(application)
    yield


app = FastAPI(
    title="GamePlanner — Steam Recommendation Engine",
    description=(
        "ML-powered game recommendation engine trained on Steam gameplay data. "
        "Features collaborative filtering (GMF), explainable recommendations, "
        "cold-start handling, and a conversational interface."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix=API_PREFIX)


async def _startup(application: FastAPI):
    log.info("Starting up GamePlanner API...")

    state = application.state
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", state.device)

    # Load ID mappings
    with open(USER2IDX_PATH) as f:
        state.user2idx = json.load(f)
    with open(ITEM2IDX_PATH) as f:
        item2idx = json.load(f)

    state.num_users = len(state.user2idx)
    state.num_items = len(item2idx)
    log.info("Users: %d | Items: %d", state.num_users, state.num_items)

    # Load GMF model (best-performing: HR@10=0.3096)
    checkpoint_path = MODELS_DIR / "gmf_best.pt"
    if checkpoint_path.exists():
        state.model = GMF(state.num_users, state.num_items, emb_dim=EMBEDDING_DIM)
        state.model.load_state_dict(torch.load(checkpoint_path, map_location=state.device))
        state.model.to(state.device)
        state.model.eval()
        state.model_name = "GMF"
        log.info("Loaded GMF from %s", checkpoint_path)
    else:
        log.warning("GMF checkpoint not found at %s. /recommend will use popularity fallback.", checkpoint_path)
        state.model = None
        state.model_name = "popularity_fallback"

    # Load metadata
    state.items_meta = pd.read_parquet(ITEMS_META_PATH)

    # Build user->train items mapping (for excluding seen items from recs)
    train_df = pd.read_parquet(TRAIN_PATH)
    state.user_train_items = (
        train_df.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )
    state.user_train_items = {str(k): v for k, v in state.user_train_items.items()}

    # Playtime map for explanation context: "user_idx_item_idx" -> playtime
    if "playtime" in train_df.columns:
        state.user_train_playtime = {
            f"{row.user_idx}_{row.item_idx}": row.playtime
            for row in train_df[["user_idx", "item_idx", "playtime"]].itertuples()
        }
    else:
        state.user_train_playtime = {}

    # Item popularity map
    popularity = train_df.groupby("item_idx")["item_idx"].count().to_dict()
    state.item_popularity = {int(k): int(v) for k, v in popularity.items()}

    # Popularity baseline (fallback recommender)
    state.popularity_model = PopularityRecommender()
    state.popularity_model.fit(train_df, state.items_meta)

    # FAISS vector store
    state.vector_store = VectorStore()
    if FAISS_INDEX_PATH.exists():
        state.vector_store.load()
    else:
        log.warning("FAISS index not found. RAG features will be unavailable.")

    # Shared components
    state.cache = ExplanationCache()
    state.llm = LLMClient()

    # RAG feature instances
    state.explainability = ExplainabilityPipeline(
        vector_store=state.vector_store,
        llm_client=state.llm,
        cache=state.cache,
        items_meta=state.items_meta,
    )
    state.coldstart_handler = ColdStartHandler(
        vector_store=state.vector_store,
        items_meta=state.items_meta,
    )
    state.conversational = ConversationalRecommender(
        vector_store=state.vector_store,
        llm_client=state.llm,
        items_meta=state.items_meta,
        cf_model=state.model,
        num_items=state.num_items,
        device=str(state.device),
    )

    log.info("Startup complete. API ready.")
