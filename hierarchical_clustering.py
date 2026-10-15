
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
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


def load_employee_hierarchical_data(sample_size=1200):
    """Load and standardize employee features for hierarchical clustering."""

    df_clean = get_preprocessed_data().copy()

    # Attrition is excluded from clustering inputs.
    if "Attrition" in df_clean.columns:
        df_features = df_clean.drop(columns=["Attrition"])
    else:
        df_features = df_clean.copy()

    # Use numerical columns only.
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
        raise ValueError("At least three employee records are required.")

    if df_features.shape[1] < 2:
        raise ValueError("At least two varying features are required.")

    # Sampling makes dendrogram generation more manageable.
    if len(df_features) > sample_size:
        df_features = df_features.sample(
            n=sample_size,
            random_state=42
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    return df_features, X_scaled, df_features.columns.tolist()


def run_hierarchical_clustering(
    n_clusters=3,
    linkage_method="ward"
):
    """Generate a hierarchical dendrogram and employee cluster plot."""

    os.makedirs(CHARTS_DIR, exist_ok=True)

    df_sub, X_scaled, features = load_employee_hierarchical_data(
        sample_size=1200
    )

    try:
        n_clusters = int(n_clusters)
    except (ValueError, TypeError):
        n_clusters = 3

    n_clusters = max(2, min(8, n_clusters, len(df_sub) - 1))

    if linkage_method not in ["ward", "complete", "average"]:
        linkage_method = "ward"

    # 1. Calculate the hierarchical linkage matrix.
    Z = linkage(X_scaled, method=linkage_method)

    # 2. Draw a truncated dendrogram.
    plt.figure(figsize=(10, 5))

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=25,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True
    )

    plt.title(
        f"SkillMatch Hierarchical Clustering Dendrogram "
        f"({linkage_method.capitalize()} Linkage)",
        fontsize=13,
        fontweight="bold"
    )
    plt.xlabel(
        "Cluster Grouping / Sample Index",
        fontsize=11
    )
    plt.ylabel("Euclidean Distance", fontsize=11)

    # This threshold is an illustrative merge-distance reference.
    cutoff_dist = Z[-n_clusters + 1, 2]
    plt.axhline(
        cutoff_dist,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Reference Cutoff (K={n_clusters})"
    )
    plt.legend()

    dendro_chart = save_chart(
        f"dendrogram_{linkage_method}.png"
    )

    # 3. Fit Agglomerative Clustering.
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )

    labels = model.fit_predict(X_scaled)

    # 4. Project the features into two dimensions using PCA.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9.5, 5.5))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.65,
        s=35,
        edgecolor="none"
    )

    plt.colorbar(scatter, label="Assigned Cluster")
    plt.title(
        f"SkillMatch Hierarchical Clusters "
        f"(Linkage: {linkage_method.capitalize()}, K={n_clusters})",
        fontsize=13,
        fontweight="bold"
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

    pca_chart = save_chart(
        f"hierarchical_pca_{linkage_method}_k{n_clusters}.png"
    )

    # 5. Summarize clusters using original employee values.
    raw_df = pd.read_csv(DATASET_PATH)

    # Align raw rows with the selected preprocessed records.
    raw_cluster_data = raw_df.loc[df_sub.index].copy()
    raw_cluster_data["Cluster"] = labels

    summary = []

    for c_id in range(n_clusters):
        sub_c = raw_cluster_data[
            raw_cluster_data["Cluster"] == c_id
        ]

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

        summary.append({
            "cluster": c_id,
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
        "n_clusters": n_clusters,
        "linkage": linkage_method,
        "features": features,
        "dendro_chart": dendro_chart,
        "pca_chart": pca_chart,
        "cluster_summary": summary
    }


if __name__ == "__main__":
    result = run_hierarchical_clustering(
        n_clusters=3,
        linkage_method="ward"
    )

    print(
        "Hierarchical clustering completed:",
        result["n_clusters"]
    )
    print("Linkage:", result["linkage"])
    print("Features used:", len(result["features"]))

    print("\nCluster Summary:")
    for cluster in result["cluster_summary"]:
        print(cluster)
