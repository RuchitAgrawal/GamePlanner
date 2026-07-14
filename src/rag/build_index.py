"""
Build the FAISS vector store from pre-generated item embeddings.

Run this after src.rag.embeddings to create the searchable FAISS index.

Usage:
    python -m src.rag.build_index
"""

import logging
from src.rag.embeddings import generate_item_embeddings, load_item_embeddings
from src.rag.vector_store import VectorStore
from src.utils.config import ITEM_VECTORS_PATH, FAISS_INDEX_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def build() -> VectorStore:
    # Load or generate embeddings
    if ITEM_VECTORS_PATH.exists():
        log.info("Loading existing item vectors from disk...")
        vectors, idx_to_itemid = load_item_embeddings()
    else:
        log.info("No cached vectors found — generating now...")
        vectors, idx_to_itemid = generate_item_embeddings()

    log.info("Item vectors shape: %s", vectors.shape)

    # Build and save FAISS index
    store = VectorStore()
    store.build(vectors, idx_to_itemid)

    log.info("FAISS index built and saved to %s", FAISS_INDEX_PATH)
    log.info("Index contains %d vectors, dim=%d", store.index.ntotal, store.dim)

    # Quick sanity check — find games similar to "The Elder Scrolls V Skyrim"
    log.info("Sanity check — querying similar games to 'Skyrim'...")
    results = store.retrieve_by_text("The Elder Scrolls V Skyrim", k=5)
    log.info("Top 5 similar games:")
    for i, (item_id, score) in enumerate(results, 1):
        log.info("  %d. %s  (score=%.4f)", i, item_id, score)

    return store


if __name__ == "__main__":
    build()
