"""
app.py — Discord Insight & Sentiment Pipeline v2
===================================================
FastAPI service deployed on HuggingFace Spaces (free CPU tier).

Receives a day's Discord messages in ONE request, runs the full
BERTopic-style pipeline:

    Stage 0: Discord-specific preprocessing (emoji→text, @mention→[USER])
    Stage 1: Context enrichment (reply chains + temporal sliding window)
    Stage 2: Embedding generation (all-MiniLM-L6-v2, on enriched docs)
    Stage 3: UMAP dimensionality reduction (384D → 10D)
    Stage 4: HDBSCAN density-based clustering (on UMAP output)
    Stage 5: Outlier reassignment (centroid cosine similarity)
    Stage 6: c-TF-IDF topic keyword extraction
    Stage 7: Per-topic sentiment aggregation + tension scoring

Key fixes over v1:
    1. UMAP before HDBSCAN (critical — without this, no topics form)
    2. Context enrichment via reply chains + temporal sliding windows
    3. Emoji-to-text conversion (preserves sentiment signal)
    4. Outlier reassignment (recovers ~20% of noise into topics)
    5. c-TF-IDF keyword extraction per cluster
    6. Sentiment aggregated per topic (not just per message)

Endpoints:
    GET  /         — info
    GET  /health   — readiness probe
    POST /analyze  — main ML pipeline
"""

import logging
import os
import re
import time
from collections import Counter, defaultdict
from typing import Optional

import emoji as emoji_lib
import hdbscan
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline as hf_pipeline
from umap import UMAP

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Auth ───────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_SPACE_API_KEY", "")

# ── Load ML models at startup ─────────────────────────────────────────────────
# Pre-downloaded during Docker build so the first request isn't slow.

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

# ── Label normalization ────────────────────────────────────────────────────────
# cardiffnlp model returns LABEL_0/1/2 in some versions, named labels in others.
LABEL_MAP = {
    "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
    "negative": "negative", "neutral": "neutral", "positive": "positive",
}


def normalize_label(raw: str) -> str:
    return LABEL_MAP.get(raw, "neutral")


# ── Request / Response Schemas ─────────────────────────────────────────────────

class Message(BaseModel):
    id: str                              # Discord snowflake as string
    content: str
    user_id: str
    timestamp: str                       # ISO 8601 UTC
    reference_id: Optional[str] = None   # Discord reply reference — for context enrichment


class AnalyzeRequest(BaseModel):
    messages: list[Message]
    umap_n_neighbors: int = 15
    umap_n_components: int = 10
    hdbscan_min_cluster_size: Optional[int] = None    # auto-tuned if None
    hdbscan_min_samples: Optional[int] = None         # auto-tuned if None
    outlier_cosine_threshold: float = 0.35


# ── Stage 0: Discord-Specific Preprocessing ───────────────────────────────────

def preprocess_message(text: str) -> str:
    """
    Clean a raw Discord message for embedding.
    Preserve semantic content. Convert (don't delete) sentiment carriers.
    """
    if not text or not text.strip():
        return "[empty]"

    # 1. Remove Discord custom emojis (<:name:123456> or <a:name:123456>)
    #    These are server-specific and carry no semantic value for a model.
    text = re.sub(r'<a?:[a-zA-Z0-9_]+:\d+>', '', text)

    # 2. Convert Unicode emojis to text BEFORE any cleaning.
    #    😂 → "face with tears of joy" — critical for sentiment accuracy.
    text = emoji_lib.demojize(text, delimiters=(' ', ' '))
    text = re.sub(r':([a-zA-Z_]+):', lambda m: m.group(1).replace('_', ' '), text)

    # 3. Replace @mentions with [USER] — preserve social graph signal
    text = re.sub(r'<@!?\d+>', '[USER]', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '[USER]', text)

    # 4. Strip URLs — no topical signal for chat analysis
    text = re.sub(r'https?://\S+', '[LINK]', text)

    # 5. Remove Discord markdown formatting but keep the text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold** → bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)        # *italic* → italic
    text = re.sub(r'__(.+?)__', r'\1', text)        # __underline__ → underline
    text = re.sub(r'~~(.+?)~~', r'\1', text)        # ~~strike~~ → strike
    text = re.sub(r'`(.+?)`', r'\1', text)          # `code` → code

    # 6. Remove quote prefix (> text) but keep the content
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)

    # 7. Collapse repeated characters (Discord meme: "noooooo" → "noo")
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Final guard: nothing meaningful left
    return text if len(text) >= 2 else "[reaction]"


# ── Stage 1: Context Enrichment ────────────────────────────────────────────────

def build_enriched_documents(
    messages: list[Message],
    clean_texts: list[str],
) -> list[str]:
    """
    Creates context-enriched pseudo-documents for embedding.

    Priority:
      1. Reply chain parents (highest signal — Discord reference_id linkage)
      2. Temporal neighbors (±3 messages by index, fallback)
      3. Standalone message (isolated)

    The enriched document is what gets embedded. The original message
    is stored separately for display. Enriched docs are internal only.
    """
    msg_by_id = {m.id: (i, m, clean_texts[i]) for i, m in enumerate(messages)}
    enriched = []

    for i, (msg, clean) in enumerate(zip(messages, clean_texts)):
        # Try reply chain first — walk up to 3 parents
        chain = []
        current_id = msg.reference_id
        depth = 0
        while current_id and current_id in msg_by_id and depth < 3:
            _, parent_msg, parent_clean = msg_by_id[current_id]
            chain.insert(0, parent_clean)
            current_id = parent_msg.reference_id
            depth += 1

        if chain:
            context_str = ' | '.join(chain)
            enriched.append(f"[THREAD: {context_str}] [MSG: {clean}]")
        else:
            # Fallback: temporal neighbors (±3 messages by index)
            start = max(0, i - 3)
            end = min(len(messages), i + 4)
            neighbors = [clean_texts[j] for j in range(start, end) if j != i]
            if neighbors:
                enriched.append(f"[NEARBY: {' | '.join(neighbors[:4])}] [MSG: {clean}]")
            else:
                enriched.append(clean)

    return enriched


# ── Stage 5: Outlier Reassignment ──────────────────────────────────────────────

def reassign_outliers(
    embeddings: np.ndarray,
    topic_labels: list[int],
    cosine_threshold: float = 0.35,
) -> list[int]:
    """
    Reassign noise messages (label == -1) to the nearest non-noise cluster
    centroid using cosine similarity in the ORIGINAL 384D embedding space.

    Only reassign if cosine_sim >= threshold. Messages with no cluster
    reaching threshold stay as -1 (truly uncategorized).

    Recovers ~15-25% of noise messages into meaningful topics.
    """
    updated_labels = list(topic_labels)
    noise_indices = [i for i, l in enumerate(topic_labels) if l == -1]

    if not noise_indices:
        return updated_labels

    unique_topics = sorted(set(l for l in topic_labels if l != -1))
    if not unique_topics:
        return updated_labels

    # Build centroids: mean of all non-noise embeddings per cluster, re-normalized
    centroids = {}
    for tid in unique_topics:
        mask = np.array([l == tid for l in topic_labels])
        centroid = np.mean(embeddings[mask], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids[tid] = centroid

    topic_ids = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[t] for t in topic_ids])   # (n_topics, 384)

    # Cosine similarity via dot product (embeddings are L2-normalized)
    noise_embs = embeddings[noise_indices]                          # (n_noise, 384)
    sims = noise_embs @ centroid_matrix.T                           # (n_noise, n_topics)

    reassigned = 0
    for local_i, global_i in enumerate(noise_indices):
        best_j = np.argmax(sims[local_i])
        if sims[local_i][best_j] >= cosine_threshold:
            updated_labels[global_i] = topic_ids[best_j]
            reassigned += 1

    logger.info(f"Outlier reassignment: {reassigned}/{len(noise_indices)} noise messages reassigned")
    return updated_labels


# ── Stage 6: c-TF-IDF Topic Keywords ──────────────────────────────────────────

def extract_topic_keywords(
    clean_texts: list[str],
    topic_labels: list[int],
    n_top_words: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """
    Compute c-TF-IDF (class-based TF-IDF) for each topic cluster.

    Unlike standard TF-IDF, c-TF-IDF treats each cluster as one mega-document
    and finds words that are frequent WITHIN a cluster but rare ACROSS all
    other clusters — surfacing what makes each topic distinctive.

    Returns: {topic_id: [(word, score), ...]} sorted by relevance descending.
    """
    cluster_docs = defaultdict(list)
    for text, label in zip(clean_texts, topic_labels):
        if label != -1:
            cluster_docs[label].append(text)

    if not cluster_docs:
        return {}

    unique_topics = sorted(cluster_docs.keys())
    super_docs = [' '.join(cluster_docs[t]) for t in unique_topics]

    try:
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.9,            # Ignore words in >90% of clusters (too generic)
            max_features=5000,
            token_pattern=r'(?u)\b[a-zA-Z]\w+\b',   # No numbers, no single chars
        )
        count_matrix = vectorizer.fit_transform(super_docs).toarray().astype(np.float64)
        vocab = vectorizer.get_feature_names_out()

        # c-TF: term frequency normalized by cluster size
        tf = count_matrix / (count_matrix.sum(axis=1, keepdims=True) + 1e-8)

        # c-IDF: log(1 + avg_words_per_cluster / word_cluster_frequency)
        avg_words = count_matrix.sum() / len(unique_topics)
        word_cluster_freq = (count_matrix > 0).sum(axis=0)
        idf = np.log(1 + avg_words / (word_cluster_freq + 1e-8))

        c_tfidf = tf * idf

        result = {}
        for i, tid in enumerate(unique_topics):
            top_idx = np.argsort(c_tfidf[i])[::-1][:n_top_words]
            result[tid] = [
                (vocab[j], float(c_tfidf[i][j]))
                for j in top_idx if c_tfidf[i][j] > 0
            ]
        return result

    except Exception as e:
        logger.error(f"c-TF-IDF failed: {e}")
        return {}


# ── Stage 7: Sentiment Aggregation by Topic ───────────────────────────────────

def aggregate_sentiment_by_topic(
    per_msg_sentiment: list[dict],
    topic_labels: list[int],
    msg_ids: list[str],
) -> dict[int, dict]:
    """
    Group per-message sentiment by topic_label and compute cluster-level metrics.

    Returns: {topic_id: {positive, negative, neutral, pct_*, tension_score,
              dominant_sentiment, needs_moderation_review}}

    tension_score: 0.0 (fully positive) to 1.0 (fully negative).
    Flag topics with tension_score > 0.40 for moderator attention.
    """
    id_to_sent = {s['id']: s for s in per_msg_sentiment}

    topic_sentiments = defaultdict(list)
    for msg_id, label in zip(msg_ids, topic_labels):
        if label != -1 and msg_id in id_to_sent:
            topic_sentiments[label].append(id_to_sent[msg_id])

    aggregated = {}
    for topic_id, sent_list in topic_sentiments.items():
        counts = Counter(s['label'] for s in sent_list)
        total = len(sent_list)

        pos = counts.get('positive', 0)
        neg = counts.get('negative', 0)
        neu = counts.get('neutral', 0)

        tension = (neg + 0.3 * neu) / (total + 1e-8)
        dominant = max(counts, key=counts.get) if counts else 'neutral'

        aggregated[topic_id] = {
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'total': total,
            'pct_positive': round(100 * pos / (total + 1e-8), 1),
            'pct_negative': round(100 * neg / (total + 1e-8), 1),
            'pct_neutral': round(100 * neu / (total + 1e-8), 1),
            'dominant_sentiment': dominant,
            'tension_score': round(tension, 4),
            'needs_moderation_review': tension > 0.40,
        }

    return aggregated


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Discord Insight Pipeline v2",
    description=(
        "BERTopic-style pipeline: Preprocessing → Context Enrichment → "
        "Embeddings → UMAP → HDBSCAN → Outlier Reassignment → c-TF-IDF → "
        "Sentiment Aggregation. Free CPU tier compatible."
    ),
    version="2.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Discord Insight Pipeline v2 — POST /analyze",
        "docs": "/docs",
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "pipeline": [
            "preprocess", "context_enrich", "embed", "umap",
            "hdbscan", "outlier_reassign", "c_tfidf", "sentiment_agg",
        ],
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, authorization: str = Header(default="")):
    """
    Main pipeline endpoint. Called once per day by Render's midnight orchestrator.

    Processing order (all on HF Space CPU):
        0. Discord-specific preprocessing (emoji→text, @mention→[USER])
        1. Context enrichment (reply chains + temporal window)
        2. Sentiment inference on cleaned text (RoBERTa, batch=128)
        3. Embedding generation on enriched pseudo-documents (MiniLM, batch=64)
        4. UMAP dimensionality reduction (384D → 10D)
        5. HDBSCAN density-based clustering (on UMAP 10D output)
        6. Outlier reassignment (centroid cosine similarity, threshold=0.35)
        7. c-TF-IDF keyword extraction per topic cluster
        8. Per-topic sentiment aggregation + tension scoring

    Returns: topics, per_message_sentiment, day_summary, metadata
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
    ids = [m.id for m in req.messages]
    logger.info(f"Pipeline start: {n} messages")

    # ── Stage 0: Discord-specific preprocessing ──────────────────────────────
    t0 = time.time()
    clean_texts = [preprocess_message(m.content) for m in req.messages]
    logger.info(f"Stage 0 (preprocess): {time.time() - t0:.1f}s")

    # ── Stage 2a: Sentiment inference (on cleaned text, before enrichment) ───
    # Sentiment runs on clean individual messages, not enriched pseudo-docs,
    # because sentiment is a per-message property, not a per-context property.
    t_sent = time.time()
    try:
        raw_sents = sentiment_pipe(clean_texts, batch_size=128)
    except Exception as e:
        logger.error(f"Sentiment inference failed: {e} — falling back to neutral/0.5")
        raw_sents = [{"label": "neutral", "score": 0.5}] * n

    per_msg_sentiment = [
        {
            "id": ids[i],
            "label": normalize_label(raw_sents[i]["label"]),
            "score": round(float(raw_sents[i]["score"]), 4),
        }
        for i in range(n)
    ]
    logger.info(f"Stage 2a (sentiment): {time.time() - t_sent:.1f}s")

    # ── Stage 1: Context enrichment ──────────────────────────────────────────
    t1 = time.time()
    enriched_docs = build_enriched_documents(req.messages, clean_texts)
    logger.info(f"Stage 1 (context enrich): {time.time() - t1:.1f}s")

    # ── Stage 2b: Embedding generation (on enriched docs) ────────────────────
    t2 = time.time()
    try:
        embeddings = embed_model.encode(
            enriched_docs,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
            convert_to_numpy=True,
        ).astype(np.float32)
    except Exception as e:
        logger.error(f"Embedding failed: {e} — using zero matrix")
        embeddings = np.zeros((n, 384), dtype=np.float32)

    logger.info(f"Stage 2b (embed): {time.time() - t2:.1f}s — shape: {embeddings.shape}")

    # ── Stage 3: UMAP dimensionality reduction (384D → 10D) ──────────────────
    # THE CRITICAL FIX: HDBSCAN cannot find density in 384D space.
    # UMAP reduces to 10D by learning the manifold structure.
    t3 = time.time()
    try:
        umap_model = UMAP(
            n_neighbors=req.umap_n_neighbors,
            n_components=req.umap_n_components,
            min_dist=0.0,        # Pack points for density detection (clustering mode)
            metric='cosine',     # Correct for L2-normalized sentence embeddings
            random_state=42,     # Reproducibility
            low_memory=False,    # Fine for ≤10k messages on 16GB
        )
        reduced = umap_model.fit_transform(embeddings).astype(np.float32)

        # Sanity check: UMAP output should have non-trivial variance
        variance = np.mean(np.var(reduced, axis=0))
        logger.info(
            f"Stage 3 (UMAP): {time.time() - t3:.1f}s — "
            f"shape: {reduced.shape}, mean_variance: {variance:.4f}"
        )
        if variance < 0.01:
            logger.warning("UMAP output has near-zero variance — check input embeddings")
    except Exception as e:
        logger.error(f"UMAP failed: {e} — falling back to raw embeddings (clustering will be poor)")
        reduced = embeddings  # Known bad — logged for debugging

    # ── Stage 4: HDBSCAN clustering (on UMAP 10D output) ─────────────────────
    t4 = time.time()
    min_cs = req.hdbscan_min_cluster_size or max(5, n // 500)
    min_samp = req.hdbscan_min_samples or max(3, min_cs // 2)

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_samp,
            metric='euclidean',               # Correct for UMAP output (not cosine!)
            cluster_selection_method='eom',    # Excess of Mass — standard
            cluster_selection_epsilon=0.0,
            prediction_data=True,              # Needed for membership probabilities
        )
        topic_labels = clusterer.fit_predict(reduced).tolist()
        membership_probs = clusterer.probabilities_
    except Exception as e:
        logger.error(f"HDBSCAN failed: {e} — returning all-noise labels")
        topic_labels = [-1] * n
        membership_probs = np.zeros(n)

    n_topics_raw = len(set(l for l in topic_labels if l != -1))
    n_noise_raw = sum(1 for l in topic_labels if l == -1)
    noise_pct = 100 * n_noise_raw / n
    logger.info(
        f"Stage 4 (HDBSCAN): {time.time() - t4:.1f}s — "
        f"{n_topics_raw} topics, {n_noise_raw} noise ({noise_pct:.1f}%)"
    )
    if noise_pct > 60:
        logger.warning(
            f"High noise rate ({noise_pct:.1f}%). Consider lowering "
            f"min_cluster_size to {max(3, min_cs - 2)} or increasing "
            f"UMAP n_neighbors to 25."
        )

    # ── Stage 5: Outlier reassignment ────────────────────────────────────────
    t5 = time.time()
    updated_labels = reassign_outliers(
        embeddings, topic_labels, req.outlier_cosine_threshold,
    )
    logger.info(f"Stage 5 (outlier reassign): {time.time() - t5:.1f}s")

    # ── Stage 6: c-TF-IDF keyword extraction ─────────────────────────────────
    t6 = time.time()
    topic_keywords = extract_topic_keywords(clean_texts, updated_labels)
    logger.info(f"Stage 6 (c-TF-IDF): {time.time() - t6:.1f}s")

    # ── Stage 7: Per-topic sentiment aggregation ─────────────────────────────
    t7 = time.time()
    topic_sentiment_agg = aggregate_sentiment_by_topic(
        per_msg_sentiment, updated_labels, ids,
    )
    logger.info(f"Stage 7 (sentiment agg): {time.time() - t7:.1f}s")

    # ── Build representative messages per topic ──────────────────────────────
    # Select messages with highest HDBSCAN membership probability (most central)
    topic_rep_msgs = defaultdict(list)
    indexed = sorted(
        zip(req.messages, updated_labels, membership_probs),
        key=lambda x: float(x[2]),
        reverse=True,
    )
    for msg, label, prob in indexed:
        if label != -1 and len(topic_rep_msgs[label]) < 6:
            topic_rep_msgs[label].append(msg.content)

    # ── Build final response ─────────────────────────────────────────────────
    # Embeddings served their purpose — discard to free memory.
    del embeddings, reduced

    final_topic_ids = sorted(set(l for l in updated_labels if l != -1))
    topics_output = []
    for tid in final_topic_ids:
        sent_agg = topic_sentiment_agg.get(tid, {})
        total_in_topic = sent_agg.get('total', 0)
        keywords = topic_keywords.get(tid, [])

        topics_output.append({
            "topic_id": tid,
            "topic_name": f"Topic {tid}",     # Named by Groq on Render side
            "keywords": [kw for kw, _ in keywords[:10]],
            "keyword_scores": keywords[:10],   # [(word, score), ...]
            "message_count": total_in_topic,
            "pct_of_day": round(100 * total_in_topic / (n + 1e-8), 1),
            "sentiment": {
                "positive": sent_agg.get('positive', 0),
                "negative": sent_agg.get('negative', 0),
                "neutral": sent_agg.get('neutral', 0),
                "pct_positive": sent_agg.get('pct_positive', 0),
                "pct_negative": sent_agg.get('pct_negative', 0),
                "pct_neutral": sent_agg.get('pct_neutral', 0),
                "dominant_sentiment": sent_agg.get('dominant_sentiment', 'neutral'),
                "tension_score": sent_agg.get('tension_score', 0),
                "needs_moderation_review": sent_agg.get('needs_moderation_review', False),
            },
            "representative_messages": topic_rep_msgs[tid],
        })

    topics_output.sort(key=lambda x: x['message_count'], reverse=True)

    n_final_topics = len(topics_output)
    n_uncategorized = sum(1 for l in updated_labels if l == -1)
    total_time = round(time.time() - t_start, 1)

    # Day-level sentiment summary
    all_sent_labels = [s['label'] for s in per_msg_sentiment]
    day_sent_counts = Counter(all_sent_labels)
    day_total = len(all_sent_labels) or 1
    day_tension = day_sent_counts.get('negative', 0) / day_total

    dominant_topic = topics_output[0]['keywords'][:3] if topics_output else []
    most_negative = max(
        topics_output,
        key=lambda t: t['sentiment']['tension_score'],
        default={},
    )

    logger.info(
        f"Pipeline complete in {total_time}s — "
        f"{n_final_topics} topics, {n_uncategorized} uncategorized"
    )

    return {
        "count": n,
        "processing_time_seconds": total_time,
        "n_topics": n_final_topics,
        "uncategorized_count": n_uncategorized,
        "uncategorized_pct": round(100 * n_uncategorized / (n + 1e-8), 1),
        "topics": topics_output,
        "topic_labels": updated_labels,                # [int, ...] × n (for message-level mapping)
        "per_message_sentiment": per_msg_sentiment,    # [{id, label, score}, ...] × n
        "day_summary": {
            "total_topics": n_final_topics,
            "dominant_topic_keywords": dominant_topic,
            "most_negative_topic_keywords": (
                most_negative.get('keywords', [])[:3] if most_negative else []
            ),
            "day_tension_score": round(day_tension, 4),
            "day_sentiment_distribution": {
                "positive": day_sent_counts.get('positive', 0),
                "neutral": day_sent_counts.get('neutral', 0),
                "negative": day_sent_counts.get('negative', 0),
            },
            "overall_dominant_sentiment": (
                day_sent_counts.most_common(1)[0][0] if day_sent_counts else 'neutral'
            ),
        },
    }
