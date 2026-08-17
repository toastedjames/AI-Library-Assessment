"""Stage 4: Evaluate the AI-assisted library assessment workflow."""

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

AI_RESULTS = (
    OUTPUT_DIRECTORY
    / "ai_analysis_results.csv"
)


def evaluate_ai_results(df):

    # Only evaluate outputs that passed the structured-output
    # validation performed in ai_analysis.py.
    valid = df[
        df["valid_json"] == True
    ].copy()

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

    matrix = confusion_matrix(
        valid["validation_theme"],
        valid["theme"],
    )

    return {
        "total_responses": len(df),
        "valid_responses": len(valid),
        "structured_output_rate": (
            len(valid) / len(df)
        ),
        "accuracy": round(
            accuracy,
            4,
        ),
        "macro_f1": round(
            macro_f1,
            4,
        ),
        "classification_report": report,
        "confusion_matrix": (
            matrix.tolist()
        ),
    }


def main():

    if not AI_RESULTS.exists():

        raise FileNotFoundError(
            "AI results not found. "
            "Run: python -m src.ai_analysis"
        )

    ai_df = pd.read_csv(
        AI_RESULTS
    )

    metrics = evaluate_ai_results(
        ai_df
    )

    print()
    print(
        "=== Generative AI Evaluation ==="
    )

    print(
        f"Total pilot responses: "
        f"{metrics['total_responses']}"
    )

    print(
        f"Valid structured outputs: "
        f"{metrics['valid_responses']}"
    )

    print(
        f"Structured output rate: "
        f"{metrics['structured_output_rate']:.1%}"
    )

    print(
        f"Theme accuracy among valid outputs: "
        f"{metrics['accuracy']:.1%}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.3f}"
    )

    print()
    print(
        "=== AI Classification Report ==="
    )

    report_df = pd.DataFrame(
        metrics[
            "classification_report"
        ]
    ).transpose()

    print(
        report_df.round(3)
    )

    # -----------------------------------------------------
    # Confidence analysis
    # -----------------------------------------------------

    valid_ai = ai_df[
        ai_df["valid_json"] == True
    ].copy()

    valid_ai["correct"] = (
        valid_ai["theme"]
        == valid_ai["validation_theme"]
    )

    confidence_summary = (
        valid_ai.groupby("correct")[
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
        "=== Confidence vs. Correctness ==="
    )

    print(
        confidence_summary
    )

    confidence_summary.to_csv(
        OUTPUT_DIRECTORY
        / "ai_confidence_analysis.csv"
    )

    # -----------------------------------------------------
    # Error analysis
    # -----------------------------------------------------

    errors = valid_ai[
        valid_ai["correct"] == False
    ].copy()

    errors[
        [
            "respondent_id",
            "validation_theme",
            "theme",
            "sentiment",
            "confidence",
            "issue",
            "evidence",
            "recommendation",
        ]
    ].to_csv(
        OUTPUT_DIRECTORY
        / "ai_classification_errors.csv",
        index=False,
    )

    print()
    print(
        f"Classification errors: "
        f"{len(errors)}"
    )

    # -----------------------------------------------------
    # Save evaluation metrics
    # -----------------------------------------------------

    evaluation_summary = pd.DataFrame(
        [
            {
                "metric": "Total pilot responses",
                "value": metrics[
                    "total_responses"
                ],
            },
            {
                "metric": "Valid structured outputs",
                "value": metrics[
                    "valid_responses"
                ],
            },
            {
                "metric": "Structured output rate",
                "value": metrics[
                    "structured_output_rate"
                ],
            },
            {
                "metric": "Theme accuracy",
                "value": metrics[
                    "accuracy"
                ],
            },
            {
                "metric": "Macro F1",
                "value": metrics[
                    "macro_f1"
                ],
            },
        ]
    )

    evaluation_summary.to_csv(
        OUTPUT_DIRECTORY
        / "ai_evaluation_summary.csv",
        index=False,
    )

    print()
    print(
        "Evaluation files written to outputs/."
    )


if __name__ == "__main__":
    main()