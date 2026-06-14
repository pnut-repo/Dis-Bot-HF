"""
app.py — HuggingFace Space ML Pipeline
========================================
FastAPI service deployed on HuggingFace Spaces (free CPU tier).

Receives all of a day's Discord messages in ONE request, runs:
    1. Sentiment inference (cardiffnlp/twitter-roberta-base-sentiment-analysis)
    2. Embedding generation (sentence-transformers/all-MiniLM-L6-v2)
    3. HDBSCAN topic clustering (on the full embedding matrix)

Returns ONLY lightweight labels — embeddings are ephemeral and never sent
over the wire. This keeps the response ~800 KB instead of ~25 MB.

Endpoints:
    GET  /         — info
    GET  /health   — readiness probe (Render polls this to wake the space)
    POST /analyze  — main ML pipeline
"""

import logging
import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import hdbscan

# ── Bearer Token Authentication ────────────────────────────────────────────────
# Set HF_SPACE_API_KEY as a Space secret in HuggingFace Settings → Variables.
# The Render orchestrator must send this in the Authorization header.
# If unset, authentication is disabled (for local dev only).
API_KEY = os.getenv("HF_SPACE_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Load models once at startup ────────────────────────────────────────────────
# Models are pre-downloaded during Docker build, so this is fast (< 10s).
# They live in RAM for the lifetime of the Space — no reload per request.

logger.info("Loading sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
sentiment_pipe = hf_pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    max_length=512,
    truncation=True,
    device=-1,   # CPU — free tier only
)
logger.info("Sentiment model loaded.")

logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model loaded. Ready to serve requests.")


# ── Request / Response Schemas ─────────────────────────────────────────────────

class Message(BaseModel):
    id: str          # message_id from Supabase (Discord snowflake as string)
    content: str
    user_id: str
    timestamp: str   # ISO 8601 UTC


class HDBSCANParams(BaseModel):
    min_cluster_size: int = 8            # minimum messages for a valid topic
    min_samples: int = 3                 # controls how conservative clustering is
    cluster_selection_epsilon: float = 0.3  # merges nearby micro-clusters


class AnalyzeRequest(BaseModel):
    messages: list[Message]
    hdbscan_params: HDBSCANParams = HDBSCANParams()


# ── Label normalization ────────────────────────────────────────────────────────
# cardiffnlp model returns LABEL_0/1/2 in some versions, named labels in others.
# This map handles both safely.

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral":  "neutral",
    "positive": "positive",
}


def normalize_label(raw: str) -> str:
    return LABEL_MAP.get(raw, "neutral")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Discord Sentiment + Clustering Pipeline",
    description=(
        "Receives a day's Discord messages, runs sentiment analysis + embeddings + HDBSCAN, "
        "returns per-message labels. Embeddings are ephemeral — never sent over the wire."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Discord Sentiment Pipeline — POST /analyze with {messages:[...], hdbscan_params:{...}}",
        "docs": "/docs",
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """
    Render polls this endpoint to check if the Space is awake before sending
    the full /analyze payload. Returns immediately — no ML involved.
    """
    return {
        "status": "ok",
        "models": ["roberta-sentiment", "minilm-embeddings", "hdbscan"],
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, authorization: str = Header(default="")):
    """
    Main pipeline endpoint. Called once per day by Render's midnight orchestrator.

    Authentication: If HF_SPACE_API_KEY env var is set, the request must include
    an 'Authorization: Bearer <key>' header. Unauthenticated requests get 401.

    Processing order (all on HF Space CPU):
        1. Sanitize empty content → [empty] placeholder to keep indices aligned
        2. Sentiment inference in internal batches of 128
        3. Embedding generation in internal batches of 64 (L2-normalized)
        4. HDBSCAN clustering on the full (n × 384) matrix
        5. Return sentiments + topic_labels — embeddings discarded

    Expected input size: 2,000–10,000 messages
    Expected processing time: ~5 min for 8,000 messages on free CPU tier
    """
    # ── Auth check ────────────────────────────────────────────────────────────
    if API_KEY:
        token = authorization.removeprefix("Bearer ").strip()
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    if not req.messages:
        raise HTTPException(status_code=400, detail="Empty messages list.")

    if len(req.messages) > 10_000:
        raise HTTPException(
            status_code=400,
            detail=f"Max 10,000 messages per request. Got {len(req.messages)}.",
        )

    t_start = time.time()
    n = len(req.messages)

    ids        = [m.id        for m in req.messages]
    texts      = [m.content   for m in req.messages]
    users      = [m.user_id   for m in req.messages]
    timestamps = [m.timestamp for m in req.messages]

    # Replace empty/whitespace-only content with a placeholder.
    # Keeps index alignment intact — these messages get label -1 (noise) from HDBSCAN.
    clean_texts = [t.strip() if t.strip() else "[empty]" for t in texts]

    logger.info(f"[analyze] Starting pipeline for {n} messages...")

    # ── Step 1: Sentiment inference ───────────────────────────────────────────
    t_sent = time.time()
    try:
        raw_sentiments = sentiment_pipe(clean_texts, batch_size=128)
    except Exception as e:
        logger.error(f"Sentiment inference failed: {e} — falling back to neutral/0.5")
        raw_sentiments = [{"label": "neutral", "score": 0.5}] * n

    sentiments = [
        {
            "id":    ids[i],
            "label": normalize_label(raw_sentiments[i]["label"]),
            "score": round(float(raw_sentiments[i]["score"]), 4),
        }
        for i in range(n)
    ]
    logger.info(f"[analyze] Sentiment done in {time.time() - t_sent:.1f}s")

    # ── Step 2: Embedding generation ──────────────────────────────────────────
    t_embed = time.time()
    try:
        vecs = embed_model.encode(
            clean_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalized → cosine similarity via dot product
        )
        embedding_matrix = vecs.astype(np.float32)   # shape: (n, 384)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e} — using zero matrix")
        embedding_matrix = np.zeros((n, 384), dtype=np.float32)

    logger.info(f"[analyze] Embeddings done in {time.time() - t_embed:.1f}s — shape: {embedding_matrix.shape}")

    # ── Step 3: HDBSCAN clustering ────────────────────────────────────────────
    t_cluster = time.time()
    try:
        p = req.hdbscan_params
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=p.min_cluster_size,
            min_samples=p.min_samples,
            metric="euclidean",
            cluster_selection_epsilon=p.cluster_selection_epsilon,
        )
        topic_labels: list[int] = clusterer.fit_predict(embedding_matrix).tolist()
    except Exception as e:
        logger.error(f"HDBSCAN clustering failed: {e} — returning all-noise labels")
        topic_labels = [-1] * n

    # Embeddings are intentionally NOT included in the response.
    # They served their purpose (clustering) and are discarded here.
    # This keeps the response payload ~800 KB vs ~25 MB if we returned them.
    del embedding_matrix

    n_topics = len(set(lbl for lbl in topic_labels if lbl != -1))
    n_noise  = sum(1 for lbl in topic_labels if lbl == -1)
    logger.info(
        f"[analyze] HDBSCAN done in {time.time() - t_cluster:.1f}s "
        f"— {n_topics} topics, {n_noise} noise ({100*n_noise/n:.1f}%)"
    )

    total_time = round(time.time() - t_start, 1)
    logger.info(f"[analyze] Full pipeline complete in {total_time}s")

    return {
        "count":                    n,
        "sentiments":               sentiments,     # [{id, label, score}, ...] × n
        "topic_labels":             topic_labels,   # [int, ...] × n — (-1 = noise)
        "n_topics":                 n_topics,
        "n_noise":                  n_noise,
        "processing_time_seconds":  total_time,
    }
