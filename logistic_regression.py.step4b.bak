import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

from preprocess import get_preprocessed_data


# ==========================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ==========================================================

def train_logistic_model(regularization="none"):

    # ======================================================
    # GET PREPROCESSED DATA
    # ======================================================

    df = get_preprocessed_data(scale=False)


    # ======================================================
    # FEATURES
    # ======================================================

    # Use all preprocessed columns except Attrition.
    # Attrition is the target.

    features = [
        column
        for column in df.columns
        if column != "Attrition"
    ]


    # ======================================================
    # INPUT AND TARGET
    # ======================================================

    X = df[features].copy()

    # Predict employee Attrition
    y = df["Attrition"].copy()


    # ======================================================
    # TRAIN TEST SPLIT
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.30,

        random_state=42,

        stratify=y
    )


    # ======================================================
    # STANDARD SCALING
    # ======================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)


    # ======================================================
    # MODEL SELECTION
    # ======================================================

    if regularization == "l2":

        # L2 Regularization

        model = LogisticRegression(

            penalty="l2",

            C=1.0,

            max_iter=1000
        )

        model_name = "Logistic Regression"

        regularization_name = "L2 Regularization"


    elif regularization == "lasso":

        # L1 / Lasso Regularization

        model = LogisticRegression(

            penalty="l1",

            C=1.0,

            solver="liblinear",

            max_iter=1000
        )

        model_name = "Logistic Regression"

        regularization_name = "L1 (Lasso) Regularization"


    else:

        # No Regularization

        model = LogisticRegression(

            penalty=None,

            max_iter=1000
        )

        model_name = "Logistic Regression"

        regularization_name = "None"


    # ======================================================
    # TRAIN MODEL
    # ======================================================

    model.fit(

        X_train,

        y_train
    )


    # ======================================================
    # PREDICTION
    # ======================================================

    y_pred = model.predict(

        X_test
    )


    # ======================================================
    # PERFORMANCE METRICS
    # ======================================================

    accuracy = accuracy_score(

        y_test,

        y_pred
    )


    precision = precision_score(

        y_test,

        y_pred,

        zero_division=0
    )


    recall = recall_score(

        y_test,

        y_pred,

        zero_division=0
    )


    f1 = f1_score(

        y_test,

        y_pred,

        zero_division=0
    )


    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    matrix = confusion_matrix(

        y_test,

        y_pred
    )


    # ======================================================
    # RETURN RESULTS
    # ======================================================

    return {

        "model": model,

        "scaler": scaler,

        "model_name": model_name,

        "regularization": regularization_name,

        "target_column": "Attrition",

        "total_rows": len(df),

        "total_features": len(features),

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "accuracy": round(

            accuracy * 100,

            2
        ),

        "precision": round(

            precision * 100,

            2
        ),

        "recall": round(

            recall * 100,

            2
        ),

        "f1": round(

            f1 * 100,

            2
        ),

        "confusion_matrix": matrix.tolist()
    }


# ==========================================================
# RUN LOGISTIC REGRESSION
# ==========================================================

def run_logistic_regression(

    regularization="none"

):

    return train_logistic_model(

        regularization
    )