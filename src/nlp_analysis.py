"""Stage 2: traditional NLP analysis of library assessment feedback.

This stage establishes a traditional NLP baseline before introducing
generative AI.

Components:

1. TF-IDF text representation
2. Logistic Regression theme classifier
3. Held-out validation set
4. Accuracy and F1 evaluation
5. Interpretable sentiment baseline
6. Row-level predictions and summary outputs
"""

from pathlib import Path
import json
import re

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split

from .clean_data import load_and_clean


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = (
    ROOT / "outputs"
)

OUTPUT_DIRECTORY.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------
# Sentiment vocabulary
# ---------------------------------------------------------

POSITIVE_PHRASES = [
    "very helpful",
    "extremely helpful",
    "really helpful",
    "good support",
    "excellent",
    "very useful",
    "really useful",
    "comfortable",
    "appreciate",
    "convenient",
    "valuable",
    "positive experience",
    "works well",
    "easy to use",
    "save me a lot of time",
]


NEGATIVE_PHRASES = [
    "difficult",
    "difficulty",
    "confusing",
    "frustrating",
    "frustrated",
    "not clear",
    "not enough",
    "not always clear",
    "not very intuitive",
    "hard to find",
    "hard to locate",
    "difficult to locate",
    "difficult to access",
    "difficult to reserve",
    "lack of",
    "limited copies",
    "not available",
    "unavailable",
    "crowded",
    "busy",
    "gaps in",
    "takes too many steps",
]


def count_phrase_matches(
    text,
    phrases,
):
    """Count meaningful sentiment phrases."""

    text = text.lower()

    return sum(
        phrase in text
        for phrase in phrases
    )


def sentiment_score(text):
    """Calculate a simple phrase-based sentiment score."""

    positive = count_phrase_matches(
        text,
        POSITIVE_PHRASES,
    )

    negative = count_phrase_matches(
        text,
        NEGATIVE_PHRASES,
    )

    return positive - negative


def sentiment_label(score):

    if score > 0:
        return "Positive"

    if score < 0:
        return "Negative"

    return "Neutral"


# ---------------------------------------------------------
# Theme classifier
# ---------------------------------------------------------

def train_theme_model(df):

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        df["open_feedback"],
        df["validation_theme"],
        test_size=0.25,
        random_state=42,
        stratify=df["validation_theme"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    x_train_tfidf = (
        vectorizer.fit_transform(
            x_train
        )
    )

    x_test_tfidf = (
        vectorizer.transform(
            x_test
        )
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        x_train_tfidf,
        y_train,
    )

    predictions = model.predict(
        x_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    )

    metrics = {
        "model": (
            "TF-IDF + Logistic Regression"
        ),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "macro_f1": round(
            float(
                report["macro avg"][
                    "f1-score"
                ]
            ),
            4,
        ),
        "weighted_f1": round(
            float(
                report["weighted avg"][
                    "f1-score"
                ]
            ),
            4,
        ),
        "classes": list(
            model.classes_
        ),
        "classification_report": report,
        "confusion_matrix": (
            matrix.tolist()
        ),
    }

    return (
        model,
        vectorizer,
        metrics,
    )


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():

    df = load_and_clean()

    print()
    print(
        "Loading library survey data..."
    )

    print(
        f"Responses available: {len(df)}"
    )

    # Train the traditional NLP classifier.
    (
        model,
        vectorizer,
        metrics,
    ) = train_theme_model(df)

    # Predict themes for every response.
    all_tfidf = (
        vectorizer.transform(
            df["open_feedback"]
        )
    )

    df["predicted_theme"] = (
        model.predict(
            all_tfidf
        )
    )

    # Calculate sentiment.
    df["sentiment_score"] = (
        df["open_feedback"]
        .apply(sentiment_score)
    )

    df["sentiment"] = (
        df["sentiment_score"]
        .apply(sentiment_label)
    )

    # -----------------------------------------------------
    # Row-level results
    # -----------------------------------------------------

    results = df[
        [
            "respondent_id",
            "student_level",
            "academic_area",
            "open_feedback",
            "validation_theme",
            "predicted_theme",
            "sentiment_score",
            "sentiment",
        ]
    ]

    results.to_csv(
        OUTPUT_DIRECTORY
        / "nlp_results.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Theme summary
    # -----------------------------------------------------

    theme_summary = (
        df.groupby(
            "predicted_theme"
        )
        .agg(
            responses=(
                "predicted_theme",
                "size",
            ),
            negative_responses=(
                "sentiment",
                lambda values: (
                    values == "Negative"
                ).sum(),
            ),
            positive_responses=(
                "sentiment",
                lambda values: (
                    values == "Positive"
                ).sum(),
            ),
        )
        .reset_index()
    )

    theme_summary[
        "negative_percent"
    ] = (
        theme_summary[
            "negative_responses"
        ]
        / theme_summary[
            "responses"
        ]
        * 100
    ).round(1)

    theme_summary = (
        theme_summary
        .sort_values(
            "responses",
            ascending=False,
        )
    )

    theme_summary.to_csv(
        OUTPUT_DIRECTORY
        / "theme_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Sentiment summary
    # -----------------------------------------------------

    sentiment_summary = (
        df["sentiment"]
        .value_counts()
        .rename_axis(
            "sentiment"
        )
        .reset_index(
            name="responses"
        )
    )

    sentiment_summary.to_csv(
        OUTPUT_DIRECTORY
        / "sentiment_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Save model evaluation
    # -----------------------------------------------------

    with open(
        OUTPUT_DIRECTORY
        / "nlp_model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print()
    print(
        "=== Library Assessment — Stage 2: NLP ==="
    )

    print(
        f"Model: {metrics['model']}"
    )

    print(
        f"Training responses: "
        f"{metrics['train_size']}"
    )

    print(
        f"Validation responses: "
        f"{metrics['test_size']}"
    )

    print(
        f"Theme accuracy: "
        f"{metrics['accuracy']:.1%}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.3f}"
    )

    print(
        f"Weighted F1: "
        f"{metrics['weighted_f1']:.3f}"
    )

    print()
    print(
        "=== Predicted Theme Summary ==="
    )

    print(
        theme_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "=== Sentiment Summary ==="
    )

    print(
        sentiment_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Outputs written to outputs/."
    )


if __name__ == "__main__":
    main()