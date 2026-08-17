"""Data quality checks and cleaning for library survey data."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "library_survey.csv"


REQUIRED_COLUMNS = [
    "respondent_id",
    "student_level",
    "academic_area",
    "library_usage_frequency",
    "satisfaction_score",
    "digital_resources_score",
    "study_space_score",
    "research_support_score",
    "open_feedback",
    "validation_theme",
]


def load_and_clean(path=DATA_PATH):

    df = pd.read_csv(path)

    # Check that all expected columns exist.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove duplicate respondents.
    df = df.drop_duplicates(
        subset="respondent_id"
    ).copy()

    score_columns = [
        "satisfaction_score",
        "digital_resources_score",
        "study_space_score",
        "research_support_score",
    ]

    # Convert scores to numeric values.
    for column in score_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove rows with missing required values.
    df = df.dropna(
        subset=score_columns + ["open_feedback"]
    )

    # Keep only valid 1–5 survey scores.
    valid_scores = df[score_columns].apply(
        lambda column: column.between(1, 5)
    )

    df = df[valid_scores.all(axis=1)]

    # Normalize feedback text.
    df["open_feedback"] = (
        df["open_feedback"]
        .astype(str)
        .str.strip()
    )

    # Remove empty feedback.
    df = df[
        df["open_feedback"].str.len() > 0
    ]

    return df


if __name__ == "__main__":

    df = load_and_clean()

    print(
        f"Rows after cleaning: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )