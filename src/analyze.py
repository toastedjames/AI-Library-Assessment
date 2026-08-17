"""Stage 1 descriptive analysis for library assessment data."""

from pathlib import Path

import pandas as pd

from .clean_data import load_and_clean


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = ROOT / "outputs"

OUTPUT_DIRECTORY.mkdir(
    exist_ok=True
)


def main():

    df = load_and_clean()

    summary = pd.DataFrame(
        {
            "metric": [
                "Total responses",
                "Average satisfaction",
                "Average digital resources score",
                "Average study space score",
                "Average research support score",
            ],
            "value": [
                len(df),
                round(
                    df["satisfaction_score"].mean(),
                    2,
                ),
                round(
                    df["digital_resources_score"].mean(),
                    2,
                ),
                round(
                    df["study_space_score"].mean(),
                    2,
                ),
                round(
                    df["research_support_score"].mean(),
                    2,
                ),
            ],
        }
    )

    summary.to_csv(
        OUTPUT_DIRECTORY / "stage1_summary.csv",
        index=False,
    )

    # Compare assessment scores by respondent group.
    by_level = (
        df.groupby("student_level")[
            [
                "satisfaction_score",
                "digital_resources_score",
                "study_space_score",
                "research_support_score",
            ]
        ]
        .mean()
        .round(2)
    )

    by_level.to_csv(
        OUTPUT_DIRECTORY
        / "satisfaction_by_student_level.csv"
    )

    print()
    print(
        "=== Library Assessment — Stage 1 ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "=== Average Scores by Student Level ==="
    )

    print(
        by_level
    )


if __name__ == "__main__":
    main()