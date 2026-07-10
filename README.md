# GamePlanner - Steam Game Recommendation Engine

A recommendation engine built on the UCSD Steam dataset, using collaborative filtering (Matrix Factorization + NeuMF) as the core model, with a RAG layer on top for explainability, cold-start handling, and a conversational interface.

## Overview

Most recommendation engine projects use MovieLens. This one uses Steam gameplay data (playtime as implicit feedback) which is a more realistic and interesting signal than star ratings.

The system is structured in two main layers:

**Core ML layer** - trains and evaluates collaborative filtering models
- Popularity baseline and content-based baseline
- Matrix Factorization (GMF) with user/item bias
- Neural Collaborative Filtering (NeuMF) with pre-trained initialization
- Full ranking evaluation: Precision@K, Recall@K, NDCG@K, Hit Rate, Coverage

**RAG extension layer** - adds intelligence on top of the trained CF model
- Explainable recommendations: 1-2 sentence grounded explanations per recommendation
- Cold-start handling: new users/items with no interaction history
- Conversational recommender: natural language queries re-ranked by the CF model

## Tech Stack

- **ML**: PyTorch, scikit-learn, pandas, numpy
- **RAG**: sentence-transformers, FAISS, Gemini 1.5 Flash (free tier)
- **API**: FastAPI, Pydantic, Uvicorn
- **Tracking**: MLflow / experiments.json

## Project Structure

```
GamePlanner/
├── data/
│   ├── raw/            # downloaded dataset files (gitignored)
│   ├── processed/      # train/val/test parquet files
│   └── embeddings/     # FAISS index and item vectors
├── src/
│   ├── data/           # preprocessing and PyTorch dataset
│   ├── models/         # baseline, MF, NeuMF
│   ├── evaluation/     # ranking metrics
│   ├── rag/            # embeddings, vector store, LLM client, features
│   ├── api/            # FastAPI app
│   └── utils/          # config, cache
├── notebooks/          # EDA, baselines, model comparison
├── tests/
├── models/             # saved checkpoints
└── experiments.json
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys.

For PyTorch with CUDA (RTX 3050):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Dataset

This project uses the [UCSD Steam dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) by Julian McAuley et al. Download `steam_reviews.json.gz` and `steam_games.json.gz` and place them in `data/raw/`.

For quick development iteration, the [Kaggle Steam CSV](https://www.kaggle.com/datasets/tamber/steam-video-games) works as a drop-in starting point.

## Results

*(will be filled in after training)*

| Model | P@5 | P@10 | NDCG@10 | HR@10 | Coverage |
|---|---|---|---|---|---|
| Popularity | - | - | - | - | - |
| Content-Based | - | - | - | - | - |
| MF | - | - | - | - | - |
| NeuMF | - | - | - | - | - |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health check |
| `/api/v1/metrics` | GET | Latest evaluation metrics |
| `/api/v1/recommend/{user_id}` | GET | Top-K recommendations |
| `/api/v1/recommend/{user_id}/explain` | GET | Recommendations with explanations |
| `/api/v1/coldstart` | POST | Recs for new users from liked game list |
| `/api/v1/chat` | POST | Natural language query to recommendations |

## Design Notes

- Time-based train/val/test split (leave-one-last) to simulate real deployment conditions
- Playtime as implicit feedback, log-scaled confidence weighting
- NeuMF initialized from separately pre-trained GMF and MLP weights (per original paper)
- Gemini 1.5 Flash free tier with disk-backed explanation caching to stay within quota
- FAISS CPU index (sub-millisecond at 10K items, no GPU needed for retrieval)
