import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import get_preprocessed_data


# ==========================================================
# TRAIN LINEAR REGRESSION MODEL
# ==========================================================

def train_model(
    regularization="none"
):

    # ==========================================================
    # GET PREPROCESSED DATA
    # ==========================================================

    df = get_preprocessed_data()


    # ==========================================================
    # FEATURES
    # ==========================================================

    features = [

        column

        for column in df.columns

        if column not in [

            "Attrition",

            "MonthlyIncome"

        ]

    ]


    # ==========================================================
    # INPUT AND TARGET
    # ==========================================================

    X = df[features].copy()

    # Linear Regression predicts Monthly Income

    y = df["MonthlyIncome"].copy()


    # ==========================================================
    # TRAIN-TEST SPLIT
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.30,

        random_state=42

    )


    # ==========================================================
    # SELECT REGRESSION MODEL
    # ==========================================================

    if regularization == "ridge":

        model = Ridge(
            alpha=1.0
        )

        model_name = "Ridge Regression"

        regularization_name = "L2 Regularization"


    elif regularization == "lasso":

        model = Lasso(
            alpha=0.01,
            max_iter=10000
        )

        model_name = "Lasso Regression"

        regularization_name = "L1 Regularization"


    else:

        model = LinearRegression()

        model_name = "Linear Regression"

        regularization_name = "None"


    # ==========================================================
    # TRAIN MODEL
    # ==========================================================

    model.fit(

        X_train,

        y_train

    )


    # ==========================================================
    # PREDICTION
    # ==========================================================

    y_pred = model.predict(

        X_test

    )


    # ==========================================================
    # EVALUATION
    # ==========================================================

    mae = mean_absolute_error(

        y_test,

        y_pred

    )


    mse = mean_squared_error(

        y_test,

        y_pred

    )


    rmse = mse ** 0.5


    r2 = r2_score(

        y_test,

        y_pred

    )


    # ==========================================================
    # RETURN RESULTS
    # ==========================================================

    return {

        "model_name":
            model_name,

        "regularization":
            regularization_name,

        "target_column":
            "MonthlyIncome",

        "total_rows":
            len(df),

        "total_features":
            len(features),

        "training_rows":
            len(X_train),

        "testing_rows":
            len(X_test),

        "mae":
            round(
                mae,
                4
            ),

        "mse":
            round(
                mse,
                4
            ),

        "rmse":
            round(
                rmse,
                4
            ),

        "r2":
            round(
                r2,
                4
            )
    }


# ==========================================================
# RUN LINEAR REGRESSION
# ==========================================================

def run_linear_regression(

    regularization="none"

):

    return train_model(

        regularization

    )