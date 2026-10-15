import os
import pandas as pd


# Dataset Path
DATA_PATH = r"D:\ML PPTS\Project\WA_Fn-UseC_-HR-Employee-Attrition.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    return df


def get_dataset_info():

    df = load_data()

    return {

        # Dataset information
        "dataset_name": DATA_PATH,

        "rows": df.shape[0],

        "columns": df.shape[1],


        # Reference-style information
        "n_rows": df.shape[0],

        "n_cols": df.shape[1],

        "column_names": list(df.columns),

        "column_info": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum())
            }
            for col in df.columns
        ],

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_counts": {
            col: int(df[col].isnull().sum())
            for col in df.columns
        },


        # First 10 rows
        "preview": df.head(10).to_html(
            classes="table preview-table",
            index=False
        )
    }


def get_data_summary(df: pd.DataFrame) -> dict:

    summary = {

        "n_rows": df.shape[0],

        "n_cols": df.shape[1],

        "columns": list(df.columns),

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_counts": {
            col: int(df[col].isnull().sum())
            for col in df.columns
        },

        "preview": df.head(10).to_dict(
            orient="records"
        ),
    }

    return summary


if __name__ == "__main__":

    data = load_data()

    print(get_data_summary(data))