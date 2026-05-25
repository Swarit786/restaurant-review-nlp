"""
Generate visualizations for the Master's Thesis README.
Reads the pipeline output CSVs and produces publication-ready charts.

Usage:
    python generate_visuals.py

Output:
    visuals/negative_topics.png
    visuals/positive_topics.png
    visuals/sentiment_rating.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Style configuration — dark theme with accent colors
# ---------------------------------------------------------------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "savefig.facecolor": "#0d1117",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.edgecolor": "#21262d",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 11,
})

RED = "#f85149"
GREEN = "#3fb950"
TEAL = "#1D9E75"
CORAL = "#E24B4A"
PURPLE = "#a5aaff"
GRAY_BAR = "#21262d"

OUTPUT_DIR = "visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Topic labels (manually assigned from thesis analysis)
# ---------------------------------------------------------------------------
NEGATIVE_LABELS = {
    124: "Food poisoning & illness",
    6:   "Poor service & staff issues",
    104: "Order delays",
    32:  "Incorrect / wrong orders",
    91:  "Strong negative taste reactions",
    36:  "Opening hours confusion",
    85:  "Drive-thru wait times",
    83:  "COVID / mask policy",
    47:  "Star rating commentary",
    16:  "McDonald's experience",
}

POSITIVE_LABELS = {
    30:  "General positive food feedback",
    80:  "Enthusiastic taste reactions",
    99:  "Positive food experience",
    112: "Positive service & food",
    118: "Casual positive expressions",
    128: "Social / place-oriented mentions",
    137: "Mixed food & service evaluation",
    31:  "Pizza & casual dining feedback",
    81:  "Overall positive dining",
    20:  "Excellent service & food quality",
}


# ===================================================================
# Chart 1: Top 10 Negative Topics
# ===================================================================
def plot_negative_topics(csv_path: str):
    df = pd.read_csv(csv_path)
    topics = df[df["type"] == "TOPIC"].copy()
    topics["negative_pct"] = topics["negative_pct"].astype(float)
    topics["topic"] = topics["topic"].astype(int)
    topics["label"] = topics["topic"].map(NEGATIVE_LABELS)
    topics = topics.sort_values("negative_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(topics["label"], topics["negative_pct"], color=CORAL,
                   height=0.65, edgecolor="none", zorder=3)

    for bar, val in zip(bars, topics["negative_pct"]):
        ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="right", va="center", fontsize=10,
                fontweight="bold", color="#fff")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Negative sentiment (%)", fontsize=11, labelpad=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/negative_topics.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ negative_topics.png")


# ===================================================================
# Chart 2: Top 10 Positive Topics
# ===================================================================
def plot_positive_topics(csv_path: str):
    df = pd.read_csv(csv_path)
    topics = df[df["type"] == "TOPIC"].copy()
    topics["positive_pct"] = topics["positive_pct"].astype(float)
    topics["topic"] = topics["topic"].astype(int)
    topics["label"] = topics["topic"].map(POSITIVE_LABELS)
    topics = topics.sort_values("positive_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(topics["label"], topics["positive_pct"], color=TEAL,
                   height=0.65, edgecolor="none", zorder=3)

    for bar, val in zip(bars, topics["positive_pct"]):
        ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="right", va="center", fontsize=10,
                fontweight="bold", color="#fff")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Positive sentiment (%)", fontsize=11, labelpad=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/positive_topics.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ positive_topics.png")


# ===================================================================
# Chart 3: Sentiment–Rating Alignment (stacked horizontal bars)
# ===================================================================
def plot_sentiment_alignment(csv_path: str):
    df = pd.read_csv(csv_path)
    df["star_label"] = df["rating"].apply(lambda r: "★" * r)
    df = df.sort_values("rating", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Positive portion
    ax.barh(df["star_label"], df["positive_pct"], color=TEAL,
            height=0.55, label="Positive", zorder=3)
    # Negative portion stacked
    ax.barh(df["star_label"], df["negative_pct"], left=df["positive_pct"],
            color=CORAL, height=0.55, label="Negative", zorder=3)

    # Percentage labels on both sides
    for _, row in df.iterrows():
        y = row["star_label"]
        # Positive label
        if row["positive_pct"] > 12:
            ax.text(row["positive_pct"] / 2, y,
                    f'{row["positive_pct"]:.1f}%',
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#fff")
        # Negative label
        if row["negative_pct"] > 8:
            ax.text(row["positive_pct"] + row["negative_pct"] / 2, y,
                    f'{row["negative_pct"]:.1f}%',
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#fff")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Sentiment distribution (%)", fontsize=11, labelpad=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=2, framealpha=0, edgecolor="none", fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/sentiment_rating.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print("  ✓ sentiment_rating.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating visualizations...")
    plot_negative_topics("top_10_negative_topics.csv")
    plot_positive_topics("top_10_positive_topics.csv")
    plot_sentiment_alignment("sentiment_rating_correlation.csv")
    print(f"\nDone — all charts saved to {OUTPUT_DIR}/")
