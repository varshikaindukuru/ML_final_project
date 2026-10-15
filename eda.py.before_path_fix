import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

CHART_FOLDER = "static/charts"


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    return pd.read_csv(DATASET_PATH)


# ==========================================================
# CREATE CHART FOLDER
# ==========================================================

def create_chart_folder():

    os.makedirs(CHART_FOLDER, exist_ok=True)


# ==========================================================
# SAVE CHART
# ==========================================================

def save_chart(filename):

    path = os.path.join(
        CHART_FOLDER,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# 1. ATTRITION DISTRIBUTION
# ==========================================================

def attrition_distribution(df):

    counts = df["Attrition"].value_counts()

    plt.figure(figsize=(6, 5))

    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Attrition Distribution")

    save_chart(
        "attrition_distribution.png"
    )


# ==========================================================
# 2. ATTRITION BY GENDER
# ==========================================================

def attrition_gender(df):

    table = pd.crosstab(
        df["Gender"],
        df["Attrition"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=["Yes", "No"],
        fill_value=0
    )

    ax = table.plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title("Attrition by Gender")

    plt.xlabel("Gender")

    plt.ylabel(
        "Attrition Percentage (%)"
    )

    plt.xticks(rotation=0)

    plt.legend(
        title="Attrition"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.1f%%"
        )

    save_chart(
        "attrition_gender.png"
    )


# ==========================================================
# 3. AGE DISTRIBUTION
# ==========================================================

def age_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="Age",
        bins=15,
        kde=True
    )

    plt.title(
        "Age Distribution"
    )

    plt.xlabel("Age")

    plt.ylabel(
        "Number of Employees"
    )

    save_chart(
        "age_distribution.png"
    )


# ==========================================================
# 4. AGE GROUP VS ATTRITION
# ==========================================================

def age_group_attrition(df):

    df = df.copy()

    bins = [
        18,
        25,
        30,
        35,
        40,
        45,
        100
    ]

    labels = [
        "18-25",
        "26-30",
        "31-35",
        "36-40",
        "41-45",
        "46+"
    ]

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    table = pd.crosstab(
        df["AgeGroup"],
        df["Attrition"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=["Yes", "No"],
        fill_value=0
    )

    ax = table.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title(
        "Age Group vs Attrition"
    )

    plt.xlabel(
        "Age Group"
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Attrition"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.1f%%"
        )

    save_chart(
        "age_group_attrition.png"
    )


# ==========================================================
# 5. JOB ROLE VS ATTRITION
# ==========================================================

def job_role_attrition(df):

    table = pd.crosstab(
        df["JobRole"],
        df["Attrition"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=["Yes", "No"],
        fill_value=0
    )

    table = table.sort_values(
        "Yes",
        ascending=True
    )

    ax = table["Yes"].plot(
        kind="barh",
        figsize=(9, 6)
    )

    plt.title(
        "Attrition Rate by Job Role"
    )

    plt.xlabel(
        "Employees Who Left (%)"
    )

    plt.ylabel(
        "Job Role"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.1f%%"
        )

    save_chart(
        "job_role_attrition.png"
    )


# ==========================================================
# 6. DEPARTMENT VS ATTRITION
# ==========================================================

def department_attrition(df):

    table = pd.crosstab(
        df["Department"],
        df["Attrition"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=["Yes", "No"],
        fill_value=0
    )

    ax = table.plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title(
        "Attrition by Department"
    )

    plt.xlabel(
        "Department"
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.xticks(
        rotation=15
    )

    plt.legend(
        title="Attrition"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.1f%%"
        )

    save_chart(
        "department_attrition.png"
    )


# ==========================================================
# 7. JOB SATISFACTION VS ATTRITION
# ==========================================================

def job_satisfaction_attrition(df):

    table = pd.crosstab(
        df["JobSatisfaction"],
        df["Attrition"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=["Yes", "No"],
        fill_value=0
    )

    ax = table.plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title(
        "Job Satisfaction vs Attrition"
    )

    plt.xlabel(
        "Job Satisfaction Level"
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Attrition"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.1f%%"
        )

    save_chart(
        "job_satisfaction_attrition.png"
    )


# ==========================================================
# 8. WORK-LIFE BALANCE VS ATTRITION
# ==========================================================

def worklife_attrition(df):

    table = pd.crosstab(
        df["WorkLifeBalance"],
        df["Attrition"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=["Yes", "No"],
        fill_value=0
    )

    ax = table.plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title(
        "Work-Life Balance vs Attrition"
    )

    plt.xlabel(
        "Work-Life Balance Level"
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Attrition"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.1f%%"
        )

    save_chart(
        "worklife_attrition.png"
    )


# ==========================================================
# 9. OVERTIME VS ATTRITION
# ==========================================================

def overtime_attrition(df):

    table = pd.crosstab(
        df["OverTime"],
        df["Attrition"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=["Yes", "No"],
        fill_value=0
    )

    ax = table.plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title(
        "Overtime vs Attrition"
    )

    plt.xlabel(
        "Overtime"
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Attrition"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.1f%%"
        )

    save_chart(
        "overtime_attrition.png"
    )


# ==========================================================
# 10. MONTHLY INCOME VS ATTRITION
# ==========================================================

def income_attrition(df):

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="Attrition",
        y="MonthlyIncome"
    )

    plt.title(
        "Monthly Income vs Attrition"
    )

    plt.xlabel(
        "Attrition"
    )

    plt.ylabel(
        "Monthly Income"
    )

    save_chart(
        "income_attrition.png"
    )


# ==========================================================
# 11. YEARS AT COMPANY VS ATTRITION
# ==========================================================

def years_company_attrition(df):

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="Attrition",
        y="YearsAtCompany"
    )

    plt.title(
        "Years at Company vs Attrition"
    )

    plt.xlabel(
        "Attrition"
    )

    plt.ylabel(
        "Years at Company"
    )

    save_chart(
        "years_company_attrition.png"
    )


# ==========================================================
# 12. CORRELATION HEATMAP
# ==========================================================

def correlation_heatmap(df):

    correlation_columns = [

        "Age",

        "DistanceFromHome",

        "JobInvolvement",

        "JobLevel",

        "JobSatisfaction",

        "MonthlyIncome",

        "NumCompaniesWorked",

        "OverTime",

        "PercentSalaryHike",

        "PerformanceRating",

        "RelationshipSatisfaction",

        "TotalWorkingYears",

        "WorkLifeBalance",

        "YearsAtCompany",

        "YearsInCurrentRole",

        "YearsSinceLastPromotion",

        "YearsWithCurrManager"

    ]

    temp = df[
        correlation_columns
    ].copy()

    temp["OverTime"] = temp[
        "OverTime"
    ].map({
        "Yes": 1,
        "No": 0
    })

    correlation = temp.corr()

    plt.figure(
        figsize=(12, 9)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title(
        "Correlation Heatmap"
    )

    save_chart(
        "correlation_heatmap.png"
    )


# ==========================================================
# GENERATE ALL EDA CHARTS
# ==========================================================

def generate_all_charts():

    create_chart_folder()

    df = load_dataset()

    attrition_distribution(df)

    attrition_gender(df)

    age_distribution(df)

    age_group_attrition(df)

    job_role_attrition(df)

    department_attrition(df)

    job_satisfaction_attrition(df)

    worklife_attrition(df)

    overtime_attrition(df)

    income_attrition(df)

    years_company_attrition(df)

    correlation_heatmap(df)


# ==========================================================
# GET EDA SUMMARY
# ==========================================================

def get_eda_summary():

    df = load_dataset()


    # ======================================================
    # DATASET INFORMATION
    # ======================================================

    total_rows = df.shape[0]

    total_columns = df.shape[1]

    duplicate_rows = int(
        df.duplicated().sum()
    )


    # ======================================================
    # SUMMARY CARDS
    # ======================================================

    total_employees = len(df)

    employees_left = (
        df["Attrition"]
        .value_counts()
        .get(
            "Yes",
            0
        )
    )

    attrition_rate = (
        employees_left /
        total_employees
    ) * 100

    average_age = df[
        "Age"
    ].mean()

    average_income = df[
        "MonthlyIncome"
    ].mean()


    # ======================================================
    # RETURN SUMMARY
    # ======================================================

    return {

        # Dataset Information

        "rows":
            total_rows,

        "columns":
            total_columns,

        "duplicates":
            duplicate_rows,


        # Summary Cards

        "total_employees":
            total_employees,

        "employees_left":
            int(
                employees_left
            ),

        "attrition_rate":
            round(
                attrition_rate,
                2
            ),

        "average_age":
            round(
                average_age,
                2
            ),

        "average_income":
            round(
                average_income,
                2
            )

    }