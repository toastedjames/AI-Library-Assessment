"""Evaluate Prompt V2 against the same 50-response pilot."""

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = ROOT / "outputs"

INPUT_FILE = (
    OUTPUT_DIRECTORY
    / "ai_analysis_results_v2.csv"
)

THEMES = [
    "Research Support",
    "Digital Resources",
    "Study Space",
    "Collections",
]


def main():

    df = pd.read_csv(INPUT_FILE)

    total = len(df)

    # A response is considered valid only when:
    # 1. valid_json is True
    # 2. theme is exactly one approved theme
    valid = df[
        (df["valid_json"] == True)
        & (df["theme"].isin(THEMES))
    ].copy()

    valid_count = len(valid)

    structured_rate = (
        valid_count / total
    )

    accuracy = accuracy_score(
        valid["validation_theme"],
        valid["theme"],
    )

    macro_f1 = f1_score(
        valid["validation_theme"],
        valid["theme"],
        average="macro",
    )

    report = classification_report(
        valid["validation_theme"],
        valid["theme"],
        output_dict=True,
        zero_division=0,
    )

    print()
    print(
        "=== Prompt V2 Evaluation ==="
    )

    print(
        f"Total pilot responses: {total}"
    )

    print(
        f"Valid outputs: {valid_count}"
    )

    print(
        f"Strict structured-output rate: "
        f"{structured_rate:.1%}"
    )

    print(
        f"Theme accuracy: "
        f"{accuracy:.1%}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.3f}"
    )

    print()
    print(
        "=== Classification Report ==="
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(3)
    )

    print(report_df)

    # -----------------------------------------------------
    # Errors
    # -----------------------------------------------------

    valid["correct"] = (
        valid["theme"]
        == valid["validation_theme"]
    )

    errors = valid[
        valid["correct"] == False
    ].copy()

    errors[
        [
            "respondent_id",
            "validation_theme",
            "theme",
            "confidence",
            "issue",
            "evidence",
            "recommendation",
        ]
    ].to_csv(
        OUTPUT_DIRECTORY
        / "ai_v2_classification_errors.csv",
        index=False,
    )

    print()
    print(
        f"Classification errors: "
        f"{len(errors)}"
    )

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence_summary = (
        valid.groupby("correct")[
            "confidence"
        ]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max",
            ]
        )
        .round(3)
    )

    print()
    print(
        "=== V2 Confidence vs. Correctness ==="
    )

    print(
        confidence_summary
    )

    confidence_summary.to_csv(
        OUTPUT_DIRECTORY
        / "ai_v2_confidence_analysis.csv"
    )

    # -----------------------------------------------------
    # Summary file
    # -----------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "prompt_version": "V2",
                "total_responses": total,
                "valid_outputs": valid_count,
                "structured_output_rate": structured_rate,
                "theme_accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIRECTORY
        / "ai_v2_evaluation_summary.csv",
        index=False,
    )

    print()
    print(
        "V2 evaluation files written to outputs/."
    )


if __name__ == "__main__":
    main()