
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from preprocess import get_preprocessed_data

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")
DATASET_PATH = os.path.join(
    BASE_DIR,
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


def chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def save_chart(filename):
    path = chart_path(filename)
    plt.tight_layout()
    plt.savefig(path, dpi=115, bbox_inches="tight")
    plt.close("all")
    return "charts/" + filename


def load_employee_dbscan_data(sample_size=2500):
    """Load and standardize employee features for DBSCAN."""

    df_clean = get_preprocessed_data().copy()

    # Attrition is excluded from clustering inputs.
    if "Attrition" in df_clean.columns:
        df_features = df_clean.drop(columns=["Attrition"])
    else:
        df_features = df_clean.copy()

    # DBSCAN requires numerical features.
    df_features = df_features.select_dtypes(include=["number"])

    df_features = df_features.replace(
        [np.inf, -np.inf], np.nan
    )
    df_features = df_features.dropna(axis=1, how="all")
    df_features = df_features.fillna(df_features.median())

    # Remove constant columns.
    varying_columns = [
        col for col in df_features.columns
        if df_features[col].nunique() > 1
    ]
    df_features = df_features[varying_columns]

    if len(df_features) < 3:
        raise ValueError(
            "At least three employee records are required."
        )

    if df_features.shape[1] < 2:
        raise ValueError(
            "At least two varying features are required."
        )

    if len(df_features) > sample_size:
        df_features = df_features.sample(
            n=sample_size,
            random_state=42
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    return df_features, X_scaled, df_features.columns.tolist()


def run_dbscan_clustering(eps=6.0, min_samples=10):
    """Perform DBSCAN clustering and visualize employee groups."""

    os.makedirs(CHARTS_DIR, exist_ok=True)

    try:
        eps = float(eps)
    except (ValueError, TypeError):
        eps = 6.0

    try:
        min_samples = int(min_samples)
    except (ValueError, TypeError):
        min_samples = 10

    # Validate parameters.
    if not np.isfinite(eps) or eps <= 0:
        eps = 6.0

    min_samples = max(2, min(100, min_samples))

    df_sub, X_scaled, features = load_employee_dbscan_data(
        sample_size=2500
    )

    # 1. Fit DBSCAN.
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X_scaled)

    df_clustered = df_sub.copy()
    df_clustered["Cluster"] = labels

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))
    noise_pct = round(
        n_noise / len(labels) * 100, 2
    )

    # 2. PCA visualization.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9.5, 5.5))

    is_clustered = labels != -1

    if np.any(is_clustered):
        plt.scatter(
            X_pca[is_clustered, 0],
            X_pca[is_clustered, 1],
            c=labels[is_clustered],
            cmap="tab10",
            alpha=0.65,
            s=35,
            label="Employee Clusters"
        )

    if n_noise > 0:
        plt.scatter(
            X_pca[~is_clustered, 0],
            X_pca[~is_clustered, 1],
            color="#dc2626",
            marker="x",
            s=45,
            alpha=0.8,
            label=f"Noise / Outliers ({n_noise})"
        )

    plt.title(
        f"SkillMatch DBSCAN Clustering "
        f"(eps={eps}, Min Samples={min_samples}) "
        f"| Clusters: {n_clusters}, Noise: {noise_pct}%",
        fontsize=12,
        fontweight="bold"
    )

    plt.xlabel(
        f"Principal Component 1 "
        f"({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
        fontsize=10
    )
    plt.ylabel(
        f"Principal Component 2 "
        f"({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
        fontsize=10
    )

    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.3)

    chart_file = f"dbscan_eps{eps}_min{min_samples}.png"
    chart_path_rel = save_chart(chart_file)

    # 3. Cluster summaries using original employee values.
    raw_df = pd.read_csv(DATASET_PATH)

    # Align original records to the sampled rows.
    raw_cluster_data = raw_df.loc[df_sub.index].copy()
    raw_cluster_data["Cluster"] = labels

    clusters_summary = []

    for c_id in sorted(unique_labels):
        sub_c = raw_cluster_data[
            raw_cluster_data["Cluster"] == c_id
        ]

        name = (
            "Noise / Outliers"
            if c_id == -1
            else f"Cluster {c_id}"
        )

        count = len(sub_c)

        attrition_pct = (
            sub_c["Attrition"].eq("Yes").mean() * 100
            if count and "Attrition" in sub_c.columns
            else 0.0
        )

        overtime_pct = (
            sub_c["OverTime"].eq("Yes").mean() * 100
            if count and "OverTime" in sub_c.columns
            else 0.0
        )

        clusters_summary.append({
            "cluster_id": int(c_id),
            "name": name,
            "count": count,
            "pct": round(
                count / len(raw_cluster_data) * 100, 1
            ),
            "mean_age": round(
                float(sub_c["Age"].mean()), 1
            ),
            "mean_income": round(
                float(sub_c["MonthlyIncome"].mean()), 1
            ),
            "mean_years_company": round(
                float(sub_c["YearsAtCompany"].mean()), 1
            ),
            "mean_job_satisfaction": round(
                float(sub_c["JobSatisfaction"].mean()), 2
            ),
            "overtime_pct": round(float(overtime_pct), 1),
            "attrition_pct": round(float(attrition_pct), 1)
        })

    return {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "total_samples": len(labels),
        "features": features,
        "chart_path": chart_path_rel,
        "clusters_summary": clusters_summary
    }


if __name__ == "__main__":
    result = run_dbscan_clustering(
        eps=6.0,
        min_samples=10
    )

    print(
        "DBSCAN completed.",
        "Clusters:", result["n_clusters"],
        "Noise:", result["noise_pct"], "%"
    )

    print("\nCluster Summary:")
    for cluster in result["clusters_summary"]:
        print(cluster)
