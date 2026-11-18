
import ast
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# 1. CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CUTOFFS = [5, 10]

# ==================================================
# 2. LOAD EXISTING DATA AND MODEL
# ==================================================
print("Loading dataset and trained model...")

metadata = pd.read_csv(BASE_DIR / "talk_metadata.csv")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
tfidf_matrix = load_npz(BASE_DIR / "tfidf_matrix.npz")

if len(metadata) != tfidf_matrix.shape[0]:
    raise ValueError("Metadata and TF-IDF row counts do not match.")

required_columns = {"topics", "views", "title"}
missing_columns = required_columns - set(metadata.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Dataset shape:", metadata.shape)
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))

# ==================================================
# 3. PREPARE TOPIC TAGS
# ==================================================
def parse_topics(value):
    if isinstance(value, list):
        return [str(x).strip().lower() for x in value]

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed]
    except (ValueError, SyntaxError):
        pass

    return []


metadata["topic_list"] = metadata["topics"].apply(parse_topics)
metadata["views"] = pd.to_numeric(
    metadata["views"], errors="coerce"
).fillna(0)

topic_sets = metadata["topic_list"].apply(set).tolist()
n_talks = len(metadata)

# ==================================================
# 4. DEFINE RELEVANCE
# ==================================================
# A talk is relevant when it shares at least one topic
# with the selected talk. The selected talk itself is
# excluded from the relevance set.
def get_relevance(query_idx):
    query_topics = topic_sets[query_idx]

    return np.array([
        int(
            idx != query_idx
            and bool(query_topics & candidate_topics)
        )
        for idx, candidate_topics in enumerate(topic_sets)
    ], dtype=int)


# ==================================================
# 5. EVALUATION METRICS
# ==================================================
def precision_at_k(ranking, relevant, k):
    top_k = ranking[:k]
    return sum(relevant[idx] for idx in top_k) / k


def recall_at_k(ranking, relevant, k):
    total_relevant = int(relevant.sum())

    if total_relevant == 0:
        return np.nan

    hits = sum(relevant[idx] for idx in ranking[:k])
    return hits / total_relevant


def dcg_at_k(relevance_values, k):
    values = np.asarray(relevance_values[:k], dtype=float)

    if len(values) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(values) + 2))
    gains = (2 ** values) - 1

    return float(np.sum(gains / discounts))


def ndcg_at_k(ranking, relevant, k):
    actual = [relevant[idx] for idx in ranking[:k]]
    ideal = sorted(relevant, reverse=True)[:k]

    ideal_dcg = dcg_at_k(ideal, k)

    if ideal_dcg == 0:
        return np.nan

    return dcg_at_k(actual, k) / ideal_dcg


def map_at_k(ranking, relevant, k):
    hits = 0
    precision_sum = 0.0

    for rank, idx in enumerate(ranking[:k], start=1):
        if relevant[idx]:
            hits += 1
            precision_sum += hits / rank

    # AP@K normalized by min(total relevant, K)
    denominator = min(int(relevant.sum()), k)

    if denominator == 0:
        return np.nan

    return precision_sum / denominator


def evaluate_ranking(ranking, relevant, k):
    return {
        "Precision": precision_at_k(ranking, relevant, k),
        "Recall": recall_at_k(ranking, relevant, k),
        "NDCG": ndcg_at_k(ranking, relevant, k),
        "MAP": map_at_k(ranking, relevant, k),
    }


# ==================================================
# 6. PREPARE RECOMMENDATION RANKINGS
# ==================================================
print("\nCalculating cosine similarities...")
similarity_matrix = cosine_similarity(tfidf_matrix)

popularity_order = np.argsort(
    -metadata["views"].to_numpy(),
    kind="stable"
)

# ==================================================
# 7. EVALUATE TOP 5 AND TOP 10
# ==================================================
results = []
evaluated_queries = 0
skipped_queries = 0

print("\nEvaluating recommendation models...")

for query_idx in range(n_talks):
    relevant = get_relevance(query_idx)

    # Queries without any relevant alternatives cannot
    # contribute meaningfully to these ranking metrics.
    if relevant.sum() == 0:
        skipped_queries += 1
        continue

    evaluated_queries += 1

    # Content-based ranking
    content_scores = similarity_matrix[query_idx].copy()
    content_scores[query_idx] = -np.inf

    content_order = np.argsort(
        -content_scores, kind="stable"
    )

    # Popularity ranking; exclude the selected talk
    popular_order = popularity_order[
        popularity_order != query_idx
    ]

    rankings = {
        "TF-IDF Content-Based": content_order,
        "Popularity Baseline": popular_order,
    }

    for model_name, ranking in rankings.items():
        for k in CUTOFFS:
            scores = evaluate_ranking(ranking, relevant, k)

            results.append({
                "Model": model_name,
                "K": k,
                "Precision": scores["Precision"],
                "Recall": scores["Recall"],
                "NDCG": scores["NDCG"],
                "MAP": scores["MAP"],
            })

# ==================================================
# 8. AGGREGATE RESULTS
# ==================================================
if not results:
    raise ValueError(
        "No queries with relevant talks were found. "
        "Check the topic tags in the dataset."
    )

per_query_df = pd.DataFrame(results)

summary = (
    per_query_df
    .groupby(["Model", "K"], as_index=False)[
        ["Precision", "Recall", "NDCG", "MAP"]
    ]
    .mean()
)

summary.to_csv(
    OUTPUT_DIR / "model_comparison_top5_top10.csv",
    index=False
)

print("\n" + "=" * 65)
print("TED TALKS RECOMMENDATION MODEL EVALUATION")
print("=" * 65)
print(f"Total talks: {n_talks}")
print(f"Queries evaluated: {evaluated_queries}")
print(f"Queries skipped: {skipped_queries}")
print("\nAverage metrics:")
print(summary.round(6).to_string(index=False))

# Also save a wide comparison table
wide_summary = summary.pivot(
    index="Model",
    columns="K",
    values=["Precision", "Recall", "NDCG", "MAP"]
)

wide_summary.to_csv(
    OUTPUT_DIR / "model_comparison_wide.csv"
)

# Save individual query results for further analysis
per_query_df.to_csv(
    OUTPUT_DIR / "per_query_evaluation.csv",
    index=False
)

# ==================================================
# 9. GENERATE COMPARISON PLOTS
# ==================================================
metric_names = ["Precision", "Recall", "NDCG", "MAP"]
model_names = [
    "TF-IDF Content-Based",
    "Popularity Baseline"
]

for k in CUTOFFS:
    subset = summary[summary["K"] == k].set_index("Model")

    ax = subset[metric_names].plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1),
        width=0.75
    )

    ax.set_title(
        f"TED Talks Recommendation Model Comparison (Top {k})"
    )
    ax.set_xlabel("Recommendation model")
    ax.set_ylabel("Mean metric score")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Evaluation metric")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"model_comparison_top{k}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

# Compare Top 5 and Top 10 for each metric separately.
# Separate plots make small recall values easier to inspect.
for metric in metric_names:
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name in model_names:
        model_data = summary[summary["Model"] == model_name]

        ax.plot(
            model_data["K"],
            model_data[metric],
            marker="o",
            linewidth=2,
            label=model_name
        )

    ax.set_title(f"{metric}: Top 5 vs Top 10")
    ax.set_xlabel("Number of recommendations (K)")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_xticks(CUTOFFS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{metric.lower()}_top5_vs_top10.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

# ==================================================
# 10. GENERATE EVALUATION REPORT
# ==================================================
report_lines = [
    "TED TALKS RECOMMENDATION SYSTEM",
    "MILESTONE 3: MODEL EVALUATION REPORT",
    "",
    "DATASET",
    "-------",
    f"Total talks: {n_talks}",
    f"TF-IDF matrix shape: {tfidf_matrix.shape}",
    f"Vocabulary size: {len(vectorizer.vocabulary_)}",
    f"Queries evaluated: {evaluated_queries}",
    f"Queries skipped: {skipped_queries}",
    "Cutoffs evaluated: Top 5 and Top 10",
    "",
    "METHODOLOGY",
    "-----------",
    "Each talk is used as a query.",
    "A different talk is considered relevant if it shares at least",
    "one topic tag with the query talk.",
    "The query talk itself is excluded from recommendations.",
    "TF-IDF ranks candidates using cosine similarity.",
    "The popularity baseline ranks candidates by view count.",
    "",
    "METRIC DEFINITIONS",
    "------------------",
    "Precision@K: relevant recommendations divided by K.",
    "Recall@K: retrieved relevant talks divided by all relevant talks.",
    "NDCG@K: ranking quality with greater weight for higher ranks.",
    "MAP@K: mean average precision, normalized here by",
    "min(total relevant talks, K).",
    "",
    "RESULTS",
    "-------",
    summary.round(6).to_string(index=False),
    "",
    "LIMITATIONS",
    "-----------",
    "Topic overlap is a proxy for relevance, not real user feedback.",
    "A talk sharing one topic is counted as relevant even if the",
    "overall semantic similarity is weak.",
    "The popularity baseline may favor older or widely viewed talks.",
    "Collaborative filtering was not evaluated because user-talk",
    "interaction data was not available.",
    "",
    "MODEL SELECTION",
    "---------------",
    "Compare all four metrics at both cutoffs before selecting the",
    "final model. Consider interpretability and suitability as well",
    "as measured ranking performance.",
    "Do not treat proxy-label metrics as evidence of user satisfaction.",
]

# Add a compact interpretation of the measured change.
for model_name in model_names:
    model_data = summary[summary["Model"] == model_name].set_index("K")

    report_lines.extend([
        "",
        f"{model_name.upper()} — TOP 5 VS TOP 10",
        "-" * (len(model_name) + 21),
    ])

    for metric in metric_names:
        top5 = model_data.loc[5, metric]
        top10 = model_data.loc[10, metric]
        change = top10 - top5

        report_lines.append(
            f"{metric}: Top 5={top5:.6f}; "
            f"Top 10={top10:.6f}; change={change:+.6f}"
        )

report_path = OUTPUT_DIR / "evaluation_report_top5_top10.txt"
report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nEvaluation completed successfully.")
print("Output folder:", OUTPUT_DIR.resolve())
print("\nGenerated files:")
print("- model_comparison_top5_top10.csv")
print("- model_comparison_wide.csv")
print("- per_query_evaluation.csv")
print("- model_comparison_top5.png")
print("- model_comparison_top10.png")
print("- precision_top5_vs_top10.png")
print("- recall_top5_vs_top10.png")
print("- ndcg_top5_vs_top10.png")
print("- map_top5_vs_top10.png")
print("- evaluation_report_top5_top10.txt")
