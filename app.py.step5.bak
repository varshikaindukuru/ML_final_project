
from flask import Flask, render_template, request

from load_data import get_dataset_info

from eda import (
    generate_all_charts,
    get_eda_summary
)

from preprocess import run_preprocessing

from linear_regression import (
    run_linear_regression
)

from logistic_regression import (
    run_logistic_regression
)

from tree_based import (
    run_tree_algorithm
)

from kmeans import (
    run_kmeans
)

from hierarchical_clustering import (
    run_hierarchical_clustering
)

from dbscan import (
    run_dbscan_clustering
)


app = Flask(__name__)


@app.route("/")
def home():

    error = None
    data = None

    try:
        data = get_dataset_info()

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "index.html",
        data=data,
        error=error
    )


@app.route("/eda")
def eda():

    error = None
    summary = None

    try:
        generate_all_charts()
        summary = get_eda_summary()

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "eda.html",
        summary=summary,
        error=error
    )


@app.route("/preprocessing")
def preprocessing():

    error = None
    results = None

    try:
        results = run_preprocessing()

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "preprocessing.html",
        results=results,
        error=error
    )


@app.route("/linear-regression")
def linear_regression_page():

    return render_template(
        "linear_regression.html",
        results=None,
        selected_model=None,
        error=None
    )


@app.route("/linear-regression/without-regularization")
def linear_regression_without():

    error = None
    results = None

    try:
        results = run_linear_regression("none")

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "linear_regression.html",
        results=results,
        selected_model="none",
        error=error
    )


@app.route("/linear-regression/with-regularization")
def linear_regression_with():

    error = None
    results = None

    try:
        results = run_linear_regression("ridge")

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "linear_regression.html",
        results=results,
        selected_model="ridge",
        error=error
    )


@app.route("/linear-regression/lasso")
def linear_regression_lasso():

    error = None
    results = None

    try:
        results = run_linear_regression("lasso")

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "linear_regression.html",
        results=results,
        selected_model="lasso",
        error=error
    )


@app.route("/logistic-regression")
def logistic_regression_page():

    return render_template(
        "logistic_regression.html",
        results=None,
        selected_model=None,
        error=None
    )


@app.route("/logistic-regression/without-regularization")
def logistic_regression_without():

    error = None
    results = None

    try:
        results = run_logistic_regression("none")

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "logistic_regression.html",
        results=results,
        selected_model="none",
        error=error
    )


@app.route("/logistic-regression/with-regularization")
def logistic_regression_with():

    error = None
    results = None

    try:
        results = run_logistic_regression("l2")

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "logistic_regression.html",
        results=results,
        selected_model="l2",
        error=error
    )


@app.route("/logistic-regression/lasso")
def logistic_regression_lasso():

    error = None
    results = None

    try:
        results = run_logistic_regression("lasso")

    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "logistic_regression.html",
        results=results,
        selected_model="lasso",
        error=error
    )


@app.route("/tree-based")
def tree_based_page():

    return render_template(
        "tree_based.html",
        results=None,
        selected_model=None,
        error=None
    )


@app.route("/tree-based-models/id3")
def id3_model():

    try:
        results = run_tree_algorithm("id3")

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="id3",
            error=None
        )

    except Exception as e:
        return render_template(
            "tree_based.html",
            results=None,
            selected_model="id3",
            error=str(e)
        )


@app.route("/tree-based-models/random-forest")
def random_forest_model():

    try:
        results = run_tree_algorithm("random_forest")

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="random_forest",
            error=None
        )

    except Exception as e:
        return render_template(
            "tree_based.html",
            results=None,
            selected_model="random_forest",
            error=str(e)
        )


@app.route("/tree-based-models/adaboost")
def adaboost_model():

    try:
        results = run_tree_algorithm("adaboost")

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="adaboost",
            error=None
        )

    except Exception as e:
        return render_template(
            "tree_based.html",
            results=None,
            selected_model="adaboost",
            error=str(e)
        )


@app.route("/tree-based-models/gradient-boosting")
def gradient_boosting_model():

    try:
        results = run_tree_algorithm("gradient_boosting")

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="gradient_boosting",
            error=None
        )

    except Exception as e:
        return render_template(
            "tree_based.html",
            results=None,
            selected_model="gradient_boosting",
            error=str(e)
        )


@app.route("/tree-based-models/xgboost")
def xgboost_model():

    try:
        results = run_tree_algorithm("xgboost")

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="xgboost",
            error=None
        )

    except Exception as e:
        return render_template(
            "tree_based.html",
            results=None,
            selected_model="xgboost",
            error=str(e)
        )


@app.route("/tree-based-models/lightgbm")
def lightgbm_model():

    try:
        results = run_tree_algorithm("lightgbm")

        return render_template(
            "tree_based.html",
            results=results,
            selected_model="lightgbm",
            error=None
        )

    except Exception as e:
        return render_template(
            "tree_based.html",
            results=None,
            selected_model="lightgbm",
            error=str(e)
        )


@app.route("/tree-based-models")
def tree_based_models_page():

    return render_template(
        "tree_based.html",
        results=None,
        selected_model=None,
        error=None
    )


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

@app.route("/kmeans", methods=["GET", "POST"])
def kmeans_view():

    method = request.values.get("method", "silhouette")
    manual_k = request.values.get("manual_k")

    try:
        result = run_kmeans(
            method=method,
            manual_k=manual_k
        )

        return render_template(
            "kmeans.html",
            result=result,
            method=method,
            error=None
        )

    except Exception as e:
        return render_template(
            "kmeans.html",
            result=None,
            method=method,
            error=str(e)
        )


# =========================================================
# HIERARCHICAL CLUSTERING
# =========================================================

@app.route("/hierarchical")
def hierarchical_view():

    linkage_method = request.args.get("linkage", "ward")

    try:
        n_clusters = int(request.args.get("n_clusters", 3))

        result = run_hierarchical_clustering(
            n_clusters=n_clusters,
            linkage_method=linkage_method
        )

        return render_template(
            "hierarchical.html",
            result=result,
            linkage_method=linkage_method,
            n_clusters=n_clusters,
            error=None
        )

    except Exception as e:
        return render_template(
            "hierarchical.html",
            result=None,
            linkage_method=linkage_method,
            n_clusters=request.args.get("n_clusters", 3),
            error=str(e)
        )


# =========================================================
# DBSCAN CLUSTERING
# =========================================================

@app.route("/dbscan")
def dbscan_view():

    try:
        eps = float(request.args.get("eps", 0.8))
        min_samples = int(request.args.get("min_samples", 10))

        result = run_dbscan_clustering(
            eps=eps,
            min_samples=min_samples
        )

        return render_template(
            "dbscan.html",
            result=result,
            eps=eps,
            min_samples=min_samples,
            error=None
        )

    except Exception as e:
        return render_template(
            "dbscan.html",
            result=None,
            eps=request.args.get("eps", 0.8),
            min_samples=request.args.get("min_samples", 10),
            error=str(e)
        )


if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )
