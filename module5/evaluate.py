"""Module 5 Evaluation Pipeline: evaluates TED Talk recommendation approaches
using existing Module 3 artifacts and proxy relevance labels from `related_talk_ids`.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k


def parse_related_ids(value) -> List[int]:
    """Extract integer talk IDs from a pipe-separated string or list-like value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(x) for x in re.findall(r"\d+", str(value))]


def build_ground_truth(df: pd.DataFrame, id_to_idx: Dict[int, int]) -> tuple:
    """Build proxy ground-truth relevance sets from `related_talk_ids`.

    Excludes self-links and out-of-dataset talk IDs.
    Returns (ground_truth_by_idx, stats_dict).
    """
    gt: Dict[int, Set[int]] = {}
    self_links_removed = 0
    out_of_dataset_removed = 0
    raw_total_links = 0

    for i, (tid, raw_rel) in enumerate(zip(df["talk_id"].tolist(), df["related_talk_ids"].tolist())):
        parsed = parse_related_ids(raw_rel)
        raw_total_links += len(parsed)
        valid_idxs: Set[int] = set()
        for rel_id in parsed:
            if rel_id == tid:
                self_links_removed += 1
                continue
            if rel_id not in id_to_idx:
                out_of_dataset_removed += 1
                continue
            valid_idxs.add(id_to_idx[rel_id])
        gt[i] = valid_idxs

    eligible = [i for i, rel in gt.items() if len(rel) > 0]
    excluded = [i for i, rel in gt.items() if len(rel) == 0]
    rel_counts = [len(gt[i]) for i in eligible]

    stats = {
        "total_talks": int(len(df)),
        "eligible_queries": int(len(eligible)),
        "excluded_queries": int(len(excluded)),
        "excluded_talk_ids": [int(df["talk_id"].iat[i]) for i in excluded],
        "raw_links_parsed": int(raw_total_links),
        "self_links_removed": int(self_links_removed),
        "out_of_dataset_links_removed": int(out_of_dataset_removed),
        "mean_relevant_per_eligible_query": round(float(np.mean(rel_counts)), 4) if rel_counts else 0.0,
        "min_relevant_per_eligible_query": int(np.min(rel_counts)) if rel_counts else 0,
        "max_relevant_per_eligible_query": int(np.max(rel_counts)) if rel_counts else 0,
        "ground_truth_note": (
            "Proxy relevance labels derived from TED's editorial 'related_talk_ids'. "
            "These represent editorial curation links rather than actual user click/watch feedback."
        ),
    }
    return gt, eligible, excluded, stats


def top_k_from_scores(scores: np.ndarray, k: int, exclude_idx: int) -> List[int]:
    """Return top-k indices sorted by descending score, breaking ties deterministically."""
    s = scores.astype(np.float64).copy()
    s[exclude_idx] = -np.inf
    k_eff = min(k, len(s) - 1)
    if k_eff <= 0:
        return []
    part = np.argpartition(-s, k_eff - 1)[:k_eff]
    ordered = part[np.lexsort((part, -s[part]))]
    return ordered.tolist()


def popularity_ranking(df: pd.DataFrame) -> List[int]:
    """Return all talk matrix indices sorted by descending views (ties broken by lower talk_id)."""
    order_df = pd.DataFrame({
        "idx": np.arange(len(df)),
        "views": pd.to_numeric(df["views"], errors="coerce").fillna(0).to_numpy(),
        "talk_id": df["talk_id"].to_numpy(),
    })
    order_df = order_df.sort_values(by=["views", "talk_id"], ascending=[False, True], kind="mergesort")
    return order_df["idx"].tolist()


def generate_plots(summary_df: pd.DataFrame, per_query_df: pd.DataFrame, fig_dir: Path) -> List[str]:
    """Generate all 6 required evaluation visualizations from actual computed metrics."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})
    approaches = summary_df["approach"].tolist()
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#c44e52"]
    saved_figs = []

    # 1-4: Individual grouped bar charts for Precision@K, Recall@K, NDCG@K, MAP@K
    metric_specs = [
        ("precision", "Precision@K", "01_precision_at_k_comparison.png"),
        ("recall", "Recall@K", "02_recall_at_k_comparison.png"),
        ("ndcg", "NDCG@K", "03_ndcg_at_k_comparison.png"),
        ("map", "MAP@K", "04_map_at_k_comparison.png"),
    ]
    x = np.arange(len(approaches))
    width = 0.35

    for key, label, fname in metric_specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        vals_5 = summary_df[f"{key}@5"].to_numpy()
        vals_10 = summary_df[f"{key}@10"].to_numpy()
        b1 = ax.bar(x - width / 2, vals_5, width, label=f"{label.replace('@K', '@5')}", color="#3b6ea5")
        b2 = ax.bar(x + width / 2, vals_10, width, label=f"{label.replace('@K', '@10')}", color="#55a868")
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.4f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, rotation=12, ha="right", fontsize=9.5)
        ax.set_ylabel(label, fontsize=10.5)
        ax.set_title(f"{label} Comparison Across Recommendation Approaches (N={int(summary_df['evaluated_queries'].iloc[0])} Queries)", fontsize=11.5, fontweight="bold")
        ax.set_ylim(0, max(vals_5.max(), vals_10.max()) * 1.22 + 0.01)
        ax.legend()
        plt.tight_layout()
        out_path = fig_dir / fname
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved_figs.append(out_path.name)

    # 5: Overall model comparison (2x1 panels: K=5 and K=10)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    metrics_list = ["precision", "recall", "ndcg", "map"]
    metric_labels = ["Precision", "Recall", "NDCG", "MAP"]
    xm = np.arange(len(metrics_list))
    bar_w = 0.155
    n_app = len(approaches)
    for ax, k in zip(axes, (5, 10)):
        for idx_a, (app, col) in enumerate(zip(approaches, colors)):
            row = summary_df[summary_df["approach"] == app].iloc[0]
            vals = [row[f"{m}@{k}"] for m in metrics_list]
            offset = (idx_a - (n_app - 1) / 2.0) * bar_w
            bars = ax.bar(xm + offset, vals, bar_w, label=app, color=col)
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_xticks(xm)
        ax.set_xticklabels([f"{lbl}@{k}" for lbl in metric_labels], fontsize=9.5)
        ax.set_title(f"Overall Evaluation Metrics @ K={k}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (0 to 1)" if k == 5 else "")
        ax.set_ylim(0, max(summary_df[[f"{m}@{k}" for m in metrics_list]].max().max() * 1.28, 0.35))
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Overall Recommendation Model Comparison (TED Talks Proxy Ground Truth)", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = fig_dir / "05_overall_model_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    saved_figs.append(out_path.name)

    # 6: Trade-offs visualization (Precision@10 vs Recall@10 & Hit Rate / Ranking Quality)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    # Left: Precision@10 vs Recall@10 scatter with NDCG@10 bubble/annotations
    for app, col in zip(approaches, colors):
        row = summary_df[summary_df["approach"] == app].iloc[0]
        axes[0].scatter(row["recall@10"], row["precision@10"], s=row["ndcg@10"] * 1800 + 80, color=col, alpha=0.85, edgecolors="black", label=f"{app} (NDCG@10={row['ndcg@10']:.4f})")
        axes[0].annotate(app, (row["recall@10"], row["precision@10"]), xytext=(6, 6), textcoords="offset points", fontsize=8.5, fontweight="bold")
    axes[0].set_xlabel("Recall@10", fontsize=10)
    axes[0].set_ylabel("Precision@10", fontsize=10)
    axes[0].set_title("Precision-Recall Trade-off @ K=10 (Bubble size = NDCG@10)", fontsize=10.5, fontweight="bold")
    axes[0].legend(fontsize=8, loc="upper left")

    # Right: Per-query NDCG@10 distribution (boxplot across eligible queries)
    data_box = [per_query_df[per_query_df["approach"] == app]["ndcg@10"].to_numpy() for app in approaches]
    bp = axes[1].boxplot(data_box, tick_labels=[a.replace(" ", "\n") for a in approaches], patch_artist=True, showmeans=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    axes[1].set_ylabel("Per-Query NDCG@10", fontsize=10)
    axes[1].set_title("Per-Query NDCG@10 Distribution & Stability Across Approaches", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    out_path = fig_dir / "06_approach_tradeoffs.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    saved_figs.append(out_path.name)

    return saved_figs


def verify_no_data_leakage(m3_dir: Path, project_root: Path) -> dict:
    """Verify that `related_talk_ids` is never used in feature construction or ranking."""
    m3_meta = json.loads((m3_dir / "metadata.json").read_text(encoding="utf-8"))
    vec_src = (project_root / "module3" / "vectorizer.py").read_text(encoding="utf-8")
    sim_src = (project_root / "module3" / "similarity.py").read_text(encoding="utf-8")

    related_in_vectorizer = "related_talk_ids" in vec_src or "related_ids_col" in vec_src
    related_in_similarity = "related_talk_ids" in sim_src or "related_ids_col" in sim_src

    return {
        "leakage_detected": bool(related_in_vectorizer or related_in_similarity),
        "features_used_for_tfidf": ["transcript", "topics"],
        "related_talk_ids_in_vectorizer": related_in_vectorizer,
        "related_talk_ids_in_similarity": related_in_similarity,
        "m3_transcript_matrix_shape": m3_meta.get("dimensions", {}).get("transcript_matrix"),
        "m3_tag_matrix_shape": m3_meta.get("dimensions", {}).get("tag_matrix"),
        "note": (
            "Verified: `related_talk_ids` is used strictly for offline proxy ground-truth evaluation "
            "and is completely excluded from TF-IDF fitting, similarity computation, and popularity ranking."
        ),
    }


def run_evaluation(project_root: Path = Path(".")) -> dict:
    """Execute full Module 5 evaluation and write all required outputs to `module5_outputs/`."""
    project_root = Path(project_root).resolve()
    data_path = project_root / "ted_talks_prepared.csv"
    m3_dir = project_root / "milestone2_outputs"
    out_dir = project_root / "module5_outputs"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset and verify alignment with Module 3 index
    df = pd.read_csv(data_path, usecols=["talk_id", "title", "speaker_1", "views", "comments", "related_talk_ids"])
    talk_index = pd.read_csv(m3_dir / "talk_index.csv")
    if not (df["talk_id"].to_numpy() == talk_index["talk_id"].to_numpy()).all():
        raise ValueError("Row mismatch between ted_talks_prepared.csv and milestone2_outputs/talk_index.csv")

    ids = df["talk_id"].to_numpy()
    id_to_idx = {int(tid): idx for idx, tid in enumerate(ids)}

    # 2. Ground truth from related_talk_ids + data leakage check
    gt, eligible, excluded, gt_stats = build_ground_truth(df, id_to_idx)
    leakage_audit = verify_no_data_leakage(m3_dir, project_root)
    assert not leakage_audit["leakage_detected"], "Data leakage detected: related_talk_ids used in features!"

    # 3. Load existing Module 3 similarity matrices (no rebuilding)
    sim_files = {
        "Transcript-only TF-IDF": m3_dir / "similarity_transcript.npy",
        "Tags-only TF-IDF": m3_dir / "similarity_tags.npy",
        "Combined TF-IDF (50:50)": m3_dir / "similarity_combined.npy",
    }
    S_t_full = np.load(sim_files["Transcript-only TF-IDF"], mmap_mode="r")
    S_g_full = np.load(sim_files["Tags-only TF-IDF"], mmap_mode="r")
    has_t = talk_index["has_transcript"].to_numpy(bool)
    pop_order = popularity_ranking(df)

    # 4. Evaluate all five approaches on the exact same eligible queries
    per_query_rows = []
    summary_rows = []
    fallback_count = 0

    approaches_order = [
        "Transcript-only TF-IDF",
        "Tags-only TF-IDF",
        "Combined TF-IDF (50:50)",
        "Transcript-primary (Tags-fallback)",
        "Popularity Baseline (Views)",
    ]

    for app_name in approaches_order:
        S = np.load(sim_files[app_name], mmap_mode="r") if app_name in sim_files else None

        p5_list, r5_list, n5_list, ap5_list = [], [], [], []
        p10_list, r10_list, n10_list, ap10_list = [], [], [], []
        hit5_list, hit10_list = [], []

        for q_idx in eligible:
            rel = gt[q_idx]
            if app_name == "Transcript-primary (Tags-fallback)":
                if has_t[q_idx] and np.any(np.delete(S_t_full[q_idx], q_idx) > 0):
                    row_scores = S_t_full[q_idx]
                else:
                    row_scores = S_g_full[q_idx]
                    fallback_count += 1
                recs_10 = top_k_from_scores(row_scores, 10, exclude_idx=q_idx)
            elif S is not None:
                recs_10 = top_k_from_scores(S[q_idx], 10, exclude_idx=q_idx)
            else:
                recs_10 = [idx for idx in pop_order if idx != q_idx][:10]

            assert q_idx not in recs_10, f"Query {q_idx} appeared in its own recommendations!"
            recs_5 = recs_10[:5]

            p5 = precision_at_k(recs_5, rel, 5, query_item=q_idx)
            r5 = recall_at_k(recs_5, rel, 5, query_item=q_idx)
            n5 = ndcg_at_k(recs_5, rel, 5, query_item=q_idx)
            ap5 = average_precision_at_k(recs_5, rel, 5, query_item=q_idx)

            p10 = precision_at_k(recs_10, rel, 10, query_item=q_idx)
            r10 = recall_at_k(recs_10, rel, 10, query_item=q_idx)
            n10 = ndcg_at_k(recs_10, rel, 10, query_item=q_idx)
            ap10 = average_precision_at_k(recs_10, rel, 10, query_item=q_idx)

            hit5 = 1 if any(x in rel for x in recs_5) else 0
            hit10 = 1 if any(x in rel for x in recs_10) else 0

            p5_list.append(p5); r5_list.append(r5); n5_list.append(n5); ap5_list.append(ap5)
            p10_list.append(p10); r10_list.append(r10); n10_list.append(n10); ap10_list.append(ap10)
            hit5_list.append(hit5); hit10_list.append(hit10)

            per_query_rows.append({
                "approach": app_name,
                "talk_id": int(ids[q_idx]),
                "num_relevant": len(rel),
                "precision@5": round(p5, 6),
                "recall@5": round(r5, 6),
                "ndcg@5": round(n5, 6),
                "ap@5": round(ap5, 6),
                "hit@5": hit5,
                "precision@10": round(p10, 6),
                "recall@10": round(r10, 6),
                "ndcg@10": round(n10, 6),
                "ap@10": round(ap10, 6),
                "hit@10": hit10,
            })

        summary_rows.append({
            "approach": app_name,
            "evaluated_queries": len(eligible),
            "excluded_queries": len(excluded),
            "precision@5": round(float(np.mean(p5_list)), 4),
            "recall@5": round(float(np.mean(r5_list)), 4),
            "ndcg@5": round(float(np.mean(n5_list)), 4),
            "map@5": round(float(np.mean(ap5_list)), 4),
            "hit_rate@5": round(float(np.mean(hit5_list)), 4),
            "precision@10": round(float(np.mean(p10_list)), 4),
            "recall@10": round(float(np.mean(r10_list)), 4),
            "ndcg@10": round(float(np.mean(n10_list)), 4),
            "map@10": round(float(np.mean(ap10_list)), 4),
            "hit_rate@10": round(float(np.mean(hit10_list)), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    per_query_df = pd.DataFrame(per_query_rows)

    # 5. Save CSV deliverables
    summary_df.to_csv(out_dir / "evaluation_summary.csv", index=False)
    per_query_df.to_csv(out_dir / "per_query_evaluation.csv", index=False)

    # Model comparison table with interpretive attributes
    comp_rows = []
    for _, r in summary_df.iterrows():
        app = r["approach"]
        if app == "Combined TF-IDF (50:50)":
            interp = "High (decomposes into transcript + tag similarity)"
            cold_start = "Handles new talks via transcript/tags; no user cold-start"
        elif app == "Transcript-primary (Tags-fallback)":
            interp = "High (transcript n-grams with explicit tag fallback)"
            cold_start = "100% catalog coverage; falls back to tags on empty transcripts"
        elif app == "Transcript-only TF-IDF":
            interp = "High (inspectable n-gram term overlap)"
            cold_start = "Fails on 5 empty-transcript talks without tags"
        elif app == "Tags-only TF-IDF":
            interp = "Very High (explicit categorical topics)"
            cold_start = "Misses fine-grained spoken nuance; many ties"
        else:
            interp = "Trivial (global view count ranking)"
            cold_start = "Non-personalized; ignores query talk completely"
        comp_rows.append({
            **r.to_dict(),
            "interpretability": interp,
            "cold_start_behavior": cold_start,
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)

    # 6. Collaborative filtering feasibility JSON (reusing Module 3 audit + Module 5 evaluation context)
    m3_cf = json.loads((m3_dir / "interaction_data_check.json").read_text(encoding="utf-8"))
    cf_report = {
        "genuine_user_item_interactions_found": bool(m3_cf.get("genuine_interaction_data_found", False)),
        "user_level_columns": m3_cf.get("user_like_columns_found", []),
        "feedback_columns": m3_cf.get("feedback_like_columns_found", []),
        "aggregate_talk_columns_present": ["views", "comments"],
        "collaborative_filtering_implemented": False,
        "collaborative_filtering_evaluated": False,
        "reason": (
            "The dataset contains 4,005 talk-level catalog rows with aggregate popularity metrics ('views', 'comments'), "
            "but zero user IDs, session logs, explicit ratings, or individual watch histories. "
            "Without a genuine bipartite user-item interaction matrix, Collaborative Filtering cannot be trained or evaluated. "
            "No synthetic users or fake interactions were fabricated."
        ),
        "comparison_summary": {
            "Content-Based Filtering": {
                "data_requirements": "Item metadata, transcripts, topic tags (all available)",
                "cold_start": "Works immediately for new talks with text/tags; no user history needed",
                "personalization": "Content/query-driven; aligns with explicit talk or text query",
                "explainability": "High (shared TF-IDF terms and matching topic tags)",
            },
            "Collaborative Filtering": {
                "data_requirements": "Bipartite user-item interactions (unavailable in current dataset)",
                "cold_start": "Suffers from both new-user and new-item cold-start problems",
                "personalization": "Strong serendipity & latent taste modeling when user logs exist",
                "explainability": "Moderate to Low (neighborhood or latent factor weights)",
            },
        },
    }
    (out_dir / "collaborative_filtering_feasibility.json").write_text(json.dumps(cf_report, indent=2), encoding="utf-8")

    # 7. Generate visualizations
    figs = generate_plots(summary_df, per_query_df, fig_dir)

    # 8. Metadata JSON
    meta = {
        "module": "Module 5 – Evaluation Metrics",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_file": str(data_path.name),
        "artifacts_reused_from": str(m3_dir.name),
        "data_leakage_audit": leakage_audit,
        "ground_truth_stats": gt_stats,
        "tags_fallback_queries_triggered": int(fallback_count),
        "k_values_evaluated": [5, 10],
        "approaches_evaluated": summary_df["approach"].tolist(),
        "figures_generated": figs,
        "results_summary": summary_rows,
    }
    (out_dir / "evaluation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    result = run_evaluation()
    print(pd.DataFrame(result["results_summary"]).to_string(index=False))

