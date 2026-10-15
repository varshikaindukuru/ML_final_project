
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return "charts/" + filename


def load_employee_clustering_data(sample_size=3500):
    """Load and scale employee features for clustering."""

    df_clean = get_preprocessed_data().copy()

    # Attrition is an outcome, not an input to K-Means.
    if "Attrition" in df_clean.columns:
        df_features = df_clean.drop(columns=["Attrition"])
    else:
        df_features = df_clean.copy()

    # Keep numeric columns only.
    df_features = df_features.select_dtypes(include=["number"])

    # Handle non-finite values and missing data.
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
        raise ValueError("At least three employee records are required.")

    if df_features.shape[1] < 2:
        raise ValueError("At least two varying features are required.")

    # Reproducible sampling for larger datasets.
    if len(df_features) > sample_size:
        df_features = df_features.sample(
            n=sample_size, random_state=42
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    return df_features, X_scaled, df_features.columns.tolist()


def run_elbow_method(X, k_range=range(2, 11)):
    k_values = list(k_range)
    wcss = []

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(X)
        wcss.append(round(float(kmeans.inertia_), 2))

    plt.figure(figsize=(9, 4.8))
    plt.plot(
        k_values, wcss,
        marker="o", linewidth=2.5, markersize=8,
        color="#08b6c9"
    )
    plt.title(
        "SkillMatch: K-Means Elbow Method",
        fontsize=13, fontweight="bold"
    )
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Inertia (WCSS)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    chart_rel = save_chart("elbow_method.png")
    return k_values, wcss, chart_rel


def run_silhouette_method(X, k_range=range(2, 11)):
    k_values = list(k_range)

    if len(X) < 3:
        raise ValueError("Not enough records for silhouette analysis.")

    # Reduce computation time for larger datasets.
    if len(X) > 2500:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X), 2500, replace=False)
        sample_for_sil = X[indices]
    else:
        sample_for_sil = X

    scores = []

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(sample_for_sil)
        score = silhouette_score(sample_for_sil, labels)
        scores.append(round(float(score), 4))

    best_k = k_values[int(np.argmax(scores))]

    plt.figure(figsize=(9, 4.8))
    plt.plot(
        k_values, scores,
        marker="s", linewidth=2.5, markersize=8,
        color="#078fa0"
    )
    plt.axvline(
        best_k,
        color="#dc2626",
        linestyle="--",
        label=f"Best Silhouette K = {best_k}"
    )
    plt.title(
        "SkillMatch: Silhouette Score Analysis",
        fontsize=13, fontweight="bold"
    )
    plt.xlabel("Number of Clusters (K)", fontsize=11)
    plt.ylabel("Silhouette Score", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    chart_rel = save_chart("silhouette_method.png")
    return k_values, scores, best_k, chart_rel


def run_kmeans(method="silhouette", manual_k=None):
    df_features, X_scaled, features = (
        load_employee_clustering_data(sample_size=3500)
    )

    k_list, sil_scores, best_k_sil, sil_chart = (
        run_silhouette_method(X_scaled)
    )

    _, wcss_list, elbow_chart = run_elbow_method(X_scaled)

    if manual_k is not None:
        try:
            chosen_k = int(manual_k)
        except (ValueError, TypeError):
            chosen_k = best_k_sil
    elif method == "silhouette":
        chosen_k = best_k_sil
    else:
        chosen_k = 3

    chosen_k = max(2, min(chosen_k, 10, len(X_scaled) - 1))

    # Fit K-Means with the selected K.
    kmeans = KMeans(
        n_clusters=chosen_k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)

    # 2D PCA visualization.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9.5, 5.5))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.6,
        s=30,
        edgecolor="none"
    )
    plt.colorbar(scatter, label="Assigned Cluster")
    plt.title(
        f"SkillMatch Employee Clusters (K={chosen_k})",
        fontsize=13, fontweight="bold"
    )
    plt.xlabel(
        f"Principal Component 1 "
        f"({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
        fontsize=11
    )
    plt.ylabel(
        f"Principal Component 2 "
        f"({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
        fontsize=11
    )
    plt.grid(True, linestyle="--", alpha=0.3)

    cluster_chart = save_chart("kmeans_clusters.png")

    # Read original values to make cluster summaries understandable.
    raw_df = pd.read_csv(DATASET_PATH)

    # Align the original records to the sampled preprocessed rows.
    raw_cluster_data = raw_df.loc[df_features.index].copy()
    raw_cluster_data["Cluster"] = labels

    summary = []

    for c_id in range(chosen_k):
        sub_c = raw_cluster_data[
            raw_cluster_data["Cluster"] == c_id
        ]
        count = len(sub_c)

        attrition_pct = (
            sub_c["Attrition"].eq("Yes").mean() * 100
            if count and "Attrition" in sub_c.columns else 0.0
        )

        overtime_pct = (
            sub_c["OverTime"].eq("Yes").mean() * 100
            if count and "OverTime" in sub_c.columns else 0.0
        )

        summary.append({
            "cluster": c_id,
            "count": count,
            "pct": round(count / len(raw_cluster_data) * 100, 1),
            "mean_age": round(float(sub_c["Age"].mean()), 1),
            "mean_income": round(
                float(sub_c["MonthlyIncome"].mean()), 1
            ),
            "mean_years_worked": round(
                float(sub_c["TotalWorkingYears"].mean()), 1
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
        "chosen_k": chosen_k,
        "best_k_sil": best_k_sil,
        "method": method,
        "features": features,
        "k_values": k_list,
        "silhouette_scores": sil_scores,
        "wcss_values": wcss_list,
        "chart_method": (
            sil_chart if method == "silhouette" else elbow_chart
        ),
        "cluster_chart": cluster_chart,
        "elbow_chart": elbow_chart,
        "silhouette_chart": sil_chart,
        "cluster_summary": summary
    }


if __name__ == "__main__":
    result = run_kmeans(method="silhouette")
    print("K-Means completed.")
    print("Chosen K:", result["chosen_k"])
    print("Best silhouette K:", result["best_k_sil"])
    print("Number of features:", len(result["features"]))

    print("\nCluster Summary:")
    for cluster in result["cluster_summary"]:
        print(cluster)
