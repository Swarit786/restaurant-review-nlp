"""
=============================================================================
APPENDIX A: Analysis Pipeline — Restaurant Review Topic Modelling
             and Sentiment Analysis
=============================================================================
Thesis: Identifying Business Improvement Priorities in Restaurant Reviews
         Through Topic Modelling and Sentiment Analysis
Dataset: Google Maps Reviews — Alaska (approx. 200,000 restaurant reviews)

Pipeline Overview:
    1. Data Extraction   — Filter restaurant businesses and their reviews
    2. Text Cleaning     — Handle Google-translated reviews
    3. Sentiment Analysis — DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
    4. Topic Modelling   — BERTopic with UMAP dimensionality reduction and
                           HDBSCAN clustering
    5. Export            — CSV outputs for analysis and thesis reporting

Dependencies:
    pip install transformers sentence-transformers bertopic umap-learn hdbscan
=============================================================================
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import gzip
import json
import re
from typing import Generator, List, Literal, Optional

import pandas as pd
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from transformers import pipeline
from umap import UMAP


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# REQUIRED INPUT FILES
# ---------------------------------------------------------------------------
# This pipeline requires two JSON dataset files from the Google Maps Reviews
# dataset (available at: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/)
#
#   1. meta-Alaska.json   — Business metadata file containing business IDs,
#                           names, and category tags. Used to identify which
#                           businesses are restaurants.
#
#   2. review-Alaska.json — Review data file containing review text, star
#                           ratings, and business IDs. Used to extract all
#                           reviews belonging to restaurant businesses.
#
# Both files can be plain .json (one JSON object per line) or gzip-compressed
# .json.gz — the parser handles both formats automatically.
#
# OUTPUT DIRECTORY
#   Set OUTPUT_DIR to any existing folder on your machine where the five
#   output CSV files will be saved after the pipeline completes.
#
# UPDATE THE THREE PATHS BELOW BEFORE RUNNING:
# ---------------------------------------------------------------------------

META_PATH    = ""   # e.g. "C:/data/meta-Alaska.json"
REVIEWS_PATH = ""   # e.g. "C:/data/review-Alaska.json"
OUTPUT_DIR   = ""   # e.g. "C:/data/output/"


def _detect_device() -> Literal["cuda", "mps", "cpu"]:
    """
    Detect the fastest available compute device for PyTorch inference.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


SENT_DEVICE: Literal["cuda", "mps", "cpu"] = _detect_device()
SENT_BATCH_SIZE: int = 16
SENT_MAX_LENGTH: int = 256    # Token truncation limit for DistilBERT

# BERTopic settings
MIN_TOPIC_SIZE: int = 200     # Minimum reviews required to form a topic
RANDOM_SEED: int    = 42      # Ensures reproducibility across UMAP runs

# Redundant word forms excluded from topic representations to reduce noise
# (plurals and verb forms that add no semantic value to topic labels)
REDUNDANT_WORD_FORMS: List[str] = [
    # Plurals
    "pizzas", "burgers", "rolls", "steaks", "orders", "tacos",
    "wings", "beers", "fries", "drinks", "nachos", "sandwiches",
    "burritos", "salads", "pancakes", "waffles",
    # Verb forms
    "ordering", "ordered", "waiting", "waited", "asking", "asked",
    "closing", "closed", "opens", "opening",
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def parse_json_file(path: str) -> Generator[dict, None, None]:
    """
    Yield JSON objects line-by-line from either a gzip-compressed or
    plain-text JSON file.

    Parameters
    ----------
    path : str
        Path to the .json or .json.gz file.

    Yields
    ------
    dict
        Parsed JSON object for each non-empty line.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except gzip.BadGzipFile:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def clean_translated_text(raw_text: str) -> str:
    """
    Extract the English translation from Google-translated review text.

    Google Maps stores machine-translated reviews in the format:
        "(Translated by Google) <English text> (Original) <original text>"
    This function extracts only the English portion. If no translation
    marker is found, the original text is returned unchanged.

    Parameters
    ----------
    raw_text : str
        Raw review text, potentially containing Google translation markers.

    Returns
    -------
    str
        Cleaned English text.
    """
    if not raw_text or pd.isna(raw_text):
        return ""

    text: str = str(raw_text)

    if "(translated by google)" in text.lower():
        # Attempt to extract text between markers using regex
        pattern = r"\(Translated by Google\)(.*?)\(Original\)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: split on the translation marker
        parts = re.split(r"\(Translated by Google\)", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            after_translation = parts[1]
            original_split = re.split(r"\(Original\)", after_translation, flags=re.IGNORECASE)
            return original_split[0].strip()

    return text.strip()


def get_topic_label(fitted_model: BERTopic, topic_num: int, top_n: int = 8) -> str:
    """
    Retrieve the top keywords for a given topic as a comma-separated string.

    Parameters
    ----------
    fitted_model : BERTopic
        A fitted BERTopic model instance.
    topic_num : int
        Topic index to retrieve keywords for.
    top_n : int
        Number of top keywords to include (default: 8).

    Returns
    -------
    str
        Comma-separated top keywords, or "N/A" if the topic has no words.
    """
    words = fitted_model.get_topic(topic_num)
    if not words:
        return "N/A"
    word_list: List[tuple] = list(words)
    return ", ".join([word for word, _ in word_list[:top_n]])


def stratified_sample_reviews(
    topic_df: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Draw a stratified sample of reviews from a topic: 2 negative,
    2 positive, and 1 neutral, based on star rating.

    Sampling thresholds:
        Negative  — ratings 1-2
        Neutral   — rating 3
        Positive  — ratings 4-5

    If a sentiment group has fewer reviews than requested, all available
    reviews from that group are included.

    Parameters
    ----------
    topic_df : pd.DataFrame
        DataFrame filtered to a single topic, containing a "rating" column.
    random_state : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    pd.DataFrame
        Deduplicated sample of up to 5 representative reviews.
    """
    neg_reviews = topic_df[topic_df["rating"] <= 2]
    neu_reviews = topic_df[topic_df["rating"] == 3]
    pos_reviews = topic_df[topic_df["rating"] >= 4]

    samples = []

    if len(neg_reviews) >= 2:
        samples.append(neg_reviews.sample(n=2, random_state=random_state))
    elif len(neg_reviews) == 1:
        samples.append(neg_reviews)

    if len(pos_reviews) >= 2:
        samples.append(pos_reviews.sample(n=2, random_state=random_state))
    elif len(pos_reviews) == 1:
        samples.append(pos_reviews)

    if len(neu_reviews) >= 1:
        samples.append(neu_reviews.sample(n=1, random_state=random_state))

    if samples:
        return pd.concat(samples).drop_duplicates()

    return topic_df.head(5)


def build_topic_summary(
    reviews_df: pd.DataFrame,
    fitted_model: BERTopic
) -> pd.DataFrame:
    """
    Compute per-topic statistics: review count, sentiment breakdown,
    star-rating extremes, and top keywords.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        Full reviews DataFrame with "topic", "sentiment", and "rating" columns.
    fitted_model : BERTopic
        Fitted BERTopic model used to retrieve topic keywords.

    Returns
    -------
    pd.DataFrame
        One row per topic (excluding outlier topic -1) with columns:
        topic, num_reviews, top_words, positive_count, positive_pct,
        negative_count, negative_pct, star_5, star_1.
    """
    records = []

    for topic_num in sorted(reviews_df["topic"].unique()):
        if topic_num == -1:
            continue  # Skip outlier cluster

        single_topic_df = reviews_df[reviews_df["topic"] == topic_num]
        total           = len(single_topic_df)

        n_pos = int((single_topic_df["sentiment"] == "POSITIVE").sum())
        n_neg = int((single_topic_df["sentiment"] == "NEGATIVE").sum())

        records.append({
            "topic":          topic_num,
            "num_reviews":    total,
            "top_words":      get_topic_label(fitted_model, topic_num),
            "positive_count": n_pos,
            "positive_pct":   round((n_pos / total) * 100, 1),
            "negative_count": n_neg,
            "negative_pct":   round((n_neg / total) * 100, 1),
            "star_5":         int((single_topic_df["rating"] == 5).sum()),
            "star_1":         int((single_topic_df["rating"] == 1).sum()),
        })

    return pd.DataFrame(records)


def export_topics_to_csv(
    topic_list_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    filename: str,
    label: str
) -> None:
    """
    Export a set of topics to CSV, interleaving each topic's statistics
    with up to 5 stratified example reviews (2 negative, 2 positive,
    1 neutral).

    Output CSV columns:
        type, topic, num_reviews, top_words, positive_count, positive_pct,
        negative_count, negative_pct, star_5, star_1,
        example_num, rating, sentiment, review_text

    Row types:
        "TOPIC"   — Aggregate statistics row for the topic
        "EXAMPLE" — Individual review example row

    Parameters
    ----------
    topic_list_df : pd.DataFrame
        Subset of the topic summary DataFrame (e.g., top 20 largest topics).
    reviews_df : pd.DataFrame
        Full reviews DataFrame used to draw example reviews.
    filename : str
        Full output file path for the CSV.
    label : str
        Descriptive label printed to console upon successful export.
    """
    rows = []

    # itertuples() yields typed namedtuple rows, avoiding pandas Series
    # type ambiguity that causes PyCharm inspection warnings with iterrows()
    for row in topic_list_df.itertuples(index=False):
        topic_num     = int(row.topic)
        topic_reviews = reviews_df[reviews_df["topic"] == topic_num]
        examples      = stratified_sample_reviews(topic_reviews)

        # Topic-level statistics row
        rows.append({
            "type":           "TOPIC",
            "topic":          topic_num,
            "num_reviews":    int(row.num_reviews),
            "top_words":      str(row.top_words),
            "positive_count": int(row.positive_count),
            "positive_pct":   float(row.positive_pct),
            "negative_count": int(row.negative_count),
            "negative_pct":   float(row.negative_pct),
            "star_5":         int(row.star_5),
            "star_1":         int(row.star_1),
            "example_num":    "",
            "rating":         "",
            "sentiment":      "",
            "review_text":    "",
        })

        # Individual example review rows
        for i, ex_row in enumerate(examples.itertuples(index=False), start=1):
            rows.append({
                "type":           "EXAMPLE",
                "topic":          topic_num,
                "num_reviews":    "",
                "top_words":      "",
                "positive_count": "",
                "positive_pct":   "",
                "negative_count": "",
                "negative_pct":   "",
                "star_5":         "",
                "star_1":         "",
                "example_num":    i,
                "rating":         int(ex_row.rating),
                "sentiment":      str(ex_row.sentiment),
                "review_text":    str(ex_row.text),
            })

    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  Saved {label}: {filename}")


# ---------------------------------------------------------------------------
# Step 1: Identify Restaurant Businesses
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Identifying restaurant businesses...")
print("=" * 60)

restaurant_ids: set = set()

for business in parse_json_file(META_PATH):
    categories: Optional[list] = business.get("category", [])
    is_restaurant = categories and any(
        isinstance(cat, str) and "restaurant" in cat.lower()
        for cat in categories
    )
    if is_restaurant:
        gmap_id = business.get("gmap_id")
        if gmap_id:
            restaurant_ids.add(gmap_id)

print(f"  Found {len(restaurant_ids):,} restaurant businesses\n")


# ---------------------------------------------------------------------------
# Step 2: Extract and Clean Reviews
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 2: Extracting reviews...")
print("=" * 60)

reviews_list: list = []
processed: int     = 0

for review_record in parse_json_file(REVIEWS_PATH):
    processed += 1

    if review_record.get("gmap_id") not in restaurant_ids:
        continue

    review_text   = review_record.get("text", "")
    review_rating = review_record.get("rating")

    # Retain only reviews with non-empty text and a valid star rating
    if review_text and str(review_text).strip() and review_rating is not None:
        cleaned = clean_translated_text(str(review_text))
        if cleaned:
            reviews_list.append({"text": cleaned, "rating": review_rating})

    if processed % 100_000 == 0:
        print(f"  Processed {processed:,} rows | Kept {len(reviews_list):,} reviews")

df = pd.DataFrame(reviews_list).dropna(subset=["rating"])

print(f"\n  Total reviews retained: {len(df):,}")
print("\n  Rating distribution:")
print(df["rating"].value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# Step 3: Sentiment Analysis (DistilBERT)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Sentiment analysis (DistilBERT)...")
print("=" * 60)
print(f"  Device selected: {SENT_DEVICE.upper()}")

# Free unused GPU memory before loading the model
if SENT_DEVICE == "cuda":
    torch.cuda.empty_cache()
elif SENT_DEVICE == "mps":
    torch.mps.empty_cache()

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=SENT_MAX_LENGTH,
    device=SENT_DEVICE,
)

all_texts: List[str] = df["text"].astype(str).tolist()
results              = sentiment_analyzer(all_texts, batch_size=SENT_BATCH_SIZE)

df["sentiment"] = [r["label"] for r in results]

n_positive = int((df["sentiment"] == "POSITIVE").sum())
n_negative = int((df["sentiment"] == "NEGATIVE").sum())

print(f"\n  Positive: {n_positive:,} ({n_positive / len(df) * 100:.1f}%)")
print(f"  Negative: {n_negative:,} ({n_negative / len(df) * 100:.1f}%)")

# Sentiment-Star Rating correlation table
print("\n  Sentiment vs. Star Rating:")
correlation_data: list = []

for star_rating in sorted(df["rating"].unique()):
    rating_df  = df[df["rating"] == star_rating]
    r_positive = int((rating_df["sentiment"] == "POSITIVE").sum())
    r_negative = int((rating_df["sentiment"] == "NEGATIVE").sum())
    pos_pct    = (r_positive / len(rating_df)) * 100
    neg_pct    = (r_negative / len(rating_df)) * 100

    correlation_data.append({
        "rating":         int(star_rating),
        "total_reviews":  len(rating_df),
        "positive_count": r_positive,
        "positive_pct":   round(pos_pct, 1),
        "negative_count": r_negative,
        "negative_pct":   round(neg_pct, 1),
    })

    print(f"  {int(star_rating)}* ({len(rating_df):,} reviews): "
          f"Positive {pos_pct:.1f}% | Negative {neg_pct:.1f}%")


# ---------------------------------------------------------------------------
# Step 4: Topic Modelling (BERTopic)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Topic modelling (BERTopic)...")
print("=" * 60)

# Sentence embeddings: maps each review to a 384-dimensional vector
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# UMAP: reduces 384-dimensional embeddings to 5 dimensions for clustering
# random_state=42 ensures reproducible dimensionality reduction
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=RANDOM_SEED,
)

# HDBSCAN: density-based clustering of the reduced embedding space
hdbscan_model = HDBSCAN(
    min_cluster_size=MIN_TOPIC_SIZE,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# CountVectorizer: generates topic keyword representations via c-TF-IDF
# Standard English stop words and redundant word forms are excluded
vectorizer_model = CountVectorizer(
    stop_words=list(ENGLISH_STOP_WORDS) + REDUNDANT_WORD_FORMS,
    min_df=10,
    max_df=0.95,
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=MIN_TOPIC_SIZE,
    calculate_probabilities=True,
    verbose=True,
)

topic_assignments, _ = topic_model.fit_transform(all_texts)
df["topic"]          = topic_assignments

num_topics = len(set(topic_assignments)) - (1 if -1 in topic_assignments else 0)
n_outliers = int((df["topic"] == -1).sum())

print(f"\n  Topics identified (excl. outliers): {num_topics}")
print(f"  Outlier reviews (topic = -1):       {n_outliers:,} ({n_outliers / len(df) * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Step 5: Compute Topic Statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Computing topic statistics...")
print("=" * 60)

topic_sentiment_df = build_topic_summary(df, topic_model)
print(f"  Topics summarised: {len(topic_sentiment_df)}")


# ---------------------------------------------------------------------------
# Step 6: Export Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Exporting results...")
print("=" * 60)

# All topics — aggregate statistics only
all_topics_path = OUTPUT_DIR + "all_topics.csv"
topic_sentiment_df.to_csv(all_topics_path, index=False)
print(f"  Saved all topics summary: {all_topics_path}")

# Sentiment-rating correlation
correlation_path = OUTPUT_DIR + "sentiment_rating_correlation.csv"
pd.DataFrame(correlation_data).to_csv(correlation_path, index=False)
print(f"  Saved correlation table: {correlation_path}")

# Top 20 largest topics — with stratified examples
export_topics_to_csv(
    topic_sentiment_df.nlargest(20, "num_reviews"),
    df,
    OUTPUT_DIR + "top_20_largest_topics.csv",
    "Top 20 Largest Topics",
)

# Top 10 most positive topics — with stratified examples
export_topics_to_csv(
    topic_sentiment_df.nlargest(10, "positive_pct"),
    df,
    OUTPUT_DIR + "top_10_positive_topics.csv",
    "Top 10 Most Positive Topics",
)

# Top 10 most negative topics — with stratified examples
export_topics_to_csv(
    topic_sentiment_df.nlargest(10, "negative_pct"),
    df,
    OUTPUT_DIR + "top_10_negative_topics.csv",
    "Top 10 Most Negative Topics",
)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Reviews analysed : {len(df):,}")
print(f"  Topics found     : {num_topics}")
print(f"  Outlier rate     : {n_outliers / len(df) * 100:.1f}%")
print(f"  Output directory : {OUTPUT_DIR}")