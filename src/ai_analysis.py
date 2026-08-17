"""Stage 3/4: Local generative AI analysis of library assessment feedback.

Prompt V2 experiment.

The experiment keeps the model, dataset, sample size, and decoding settings
constant while improving only the prompt design.

This allows us to evaluate whether prompt engineering improves:
1. structured-output reliability
2. theme classification accuracy
3. confidence behavior
"""

from pathlib import Path
import json
import re

import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .clean_data import load_and_clean


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = ROOT / "outputs"

OUTPUT_DIRECTORY.mkdir(
    exist_ok=True
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SAMPLE_SIZE = 50

RANDOM_STATE = 42


THEMES = [
    "Research Support",
    "Digital Resources",
    "Study Space",
    "Collections",
]


# -------------------------------------------------------------------
# PROMPT V2
# -------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an AI assistant supporting university library assessment.

Your task is to classify and analyze one open-ended library survey response.

You MUST follow the assessment taxonomy and decision rules below.

============================================================
APPROVED THEMES
============================================================

1. Research Support

Use this theme when the primary concern involves:
- research guidance
- research consultations
- research workshops
- source evaluation
- research methods
- assistance from librarians
- difficulty knowing how to conduct research

2. Digital Resources

Use this theme when the primary concern involves:
- electronic books
- journal articles
- online databases
- library website access
- digital collections
- online catalog/search tools
- remote access
- difficulty finding electronic material

3. Study Space

Use this theme when the primary concern involves:
- physical study areas
- seating
- quiet spaces
- group study rooms
- noise
- building environment
- study-room availability
- outlets or furniture
- crowding while studying

4. Collections

Use this theme when the primary concern involves:
- books
- physical materials
- course readings
- older materials
- specialized collections
- missing materials
- unavailable editions
- collection gaps

============================================================
DECISION RULES
============================================================

Rule 1:
Choose exactly ONE theme.

Rule 2:
Classify according to the PRIMARY issue expressed by the respondent.

Rule 3:
Do not select a theme simply because one keyword appears in the response.

Rule 4:
If multiple themes are mentioned, determine which issue represents
the main problem or experience.

Rule 5:
Use only information contained in the response.

Rule 6:
Do not invent facts.

Rule 7:
Do not infer demographic information.

Rule 8:
The evidence field must be grounded directly in the response.

Rule 9:
The recommendation must address the primary issue and must not introduce
unsupported claims.

Rule 10:
Confidence must reflect the clarity of the classification.

Suggested confidence interpretation:

0.90-1.00 = very clear classification
0.75-0.89 = reasonably clear classification
0.50-0.74 = ambiguous or overlapping evidence
0.00-0.49 = highly uncertain

============================================================
EXAMPLES
============================================================

Example 1

Response:
"The librarians helped me figure out which sources were appropriate
for my thesis."

Theme:
Research Support

Reason:
The primary issue is research guidance.

Example 2

Response:
"I could access the article from home, but the website made it difficult
to understand which link I should use."

Theme:
Digital Resources

Reason:
The primary issue is remote/digital access.

Example 3

Response:
"The library is comfortable, but I often cannot find an empty desk
during exam periods."

Theme:
Study Space

Reason:
The primary issue is physical seating availability.

Example 4

Response:
"I needed an older book for my project, but the library did not have
a copy available."

Theme:
Collections

Reason:
The primary issue concerns availability of a library collection item.

============================================================
OUTPUT REQUIREMENTS
============================================================

Return ONLY one valid JSON object.

Do not include:
- Markdown
- explanations outside JSON
- headings
- commentary
- code fences

The JSON object MUST contain exactly these fields:

{
    "theme": "Research Support | Digital Resources | Study Space | Collections",
    "sentiment": "Positive | Neutral | Negative",
    "issue": "short description",
    "evidence": "specific evidence grounded in the response",
    "recommendation": "practical recommendation grounded in the response",
    "confidence": 0.00
}

Before returning the JSON, internally check that:
- the theme is one of the four approved themes
- the sentiment is valid
- evidence comes from the response
- the recommendation does not introduce unsupported facts
- confidence is between 0 and 1

Return ONLY the final JSON object.
"""


def load_model():

    print()
    print(
        f"Loading local model: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    print(
        "Model loaded successfully."
    )

    return tokenizer, model


def extract_json(text):

    """
    Extract the first JSON object from model output.
    """

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:

        raise ValueError(
            "No JSON object found."
        )

    return json.loads(
        match.group(0)
    )


def analyze_response(
    tokenizer,
    model,
    feedback,
):

    user_prompt = f"""
Analyze the following university library survey response.

Response:

"{feedback}"

Apply the taxonomy and decision rules from the system instructions.

Return ONLY the required JSON object.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[-1]:
    ]

    output_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    result = extract_json(
        output_text
    )

    return result, output_text


def validate_result(result):

    required_fields = [
        "theme",
        "sentiment",
        "issue",
        "evidence",
        "recommendation",
        "confidence",
    ]

    if not all(
        field in result
        for field in required_fields
    ):
        return False

    if result["theme"] not in THEMES:
        return False

    if result["sentiment"] not in [
        "Positive",
        "Neutral",
        "Negative",
    ]:
        return False

    try:

        confidence = float(
            result["confidence"]
        )

    except (
        TypeError,
        ValueError,
    ):

        return False

    if not 0 <= confidence <= 1:
        return False

    return True


def main():

    df = load_and_clean()

    sample = df.sample(
        n=min(
            SAMPLE_SIZE,
            len(df),
        ),
        random_state=RANDOM_STATE,
    )

    print()
    print(
        "=== Library Assessment — Prompt V2 ==="
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Total survey responses: {len(df)}"
    )

    print(
        f"Pilot sample: {len(sample)}"
    )

    tokenizer, model = load_model()

    results = []

    print()
    print(
        "Beginning Prompt V2 analysis..."
    )

    for counter, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):

        respondent_id = (
            row["respondent_id"]
        )

        feedback = (
            row["open_feedback"]
        )

        print(
            f"[{counter:02d}/{len(sample)}] "
            f"Analyzing {respondent_id}..."
        )

        try:

            result, raw_output = (
                analyze_response(
                    tokenizer,
                    model,
                    feedback,
                )
            )

            valid = validate_result(
                result
            )

            results.append(
                {
                    "respondent_id":
                        respondent_id,

                    "validation_theme":
                        row[
                            "validation_theme"
                        ],

                    "theme":
                        result.get(
                            "theme",
                            "",
                        ),

                    "sentiment":
                        result.get(
                            "sentiment",
                            "",
                        ),

                    "issue":
                        result.get(
                            "issue",
                            "",
                        ),

                    "evidence":
                        result.get(
                            "evidence",
                            "",
                        ),

                    "recommendation":
                        result.get(
                            "recommendation",
                            "",
                        ),

                    "confidence":
                        result.get(
                            "confidence",
                            0,
                        ),

                    "valid_json":
                        valid,

                    "raw_output":
                        raw_output,
                }
            )

        except Exception as error:

            print(
                f"  Error: {error}"
            )

            results.append(
                {
                    "respondent_id":
                        respondent_id,

                    "validation_theme":
                        row[
                            "validation_theme"
                        ],

                    "theme": "ERROR",

                    "sentiment": "ERROR",

                    "issue": "",

                    "evidence": "",

                    "recommendation": "",

                    "confidence": 0,

                    "valid_json": False,

                    "raw_output": "",
                }
            )

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIRECTORY
        / "ai_analysis_results_v2.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    valid_count = (
        results_df[
            "valid_json"
        ].sum()
    )

    print()
    print(
        "=== Prompt V2 Summary ==="
    )

    print(
        f"Responses analyzed: "
        f"{len(results_df)}"
    )

    print(
        f"Valid structured outputs: "
        f"{valid_count}"
    )

    print(
        f"Structured output rate: "
        f"{valid_count / len(results_df):.1%}"
    )

    print()
    print(
        "=== Theme Distribution ==="
    )

    print(
        results_df[
            results_df["theme"] != "ERROR"
        ]["theme"]
        .value_counts()
    )

    print()
    print(
        f"Saved results to: {output_file}"
    )


if __name__ == "__main__":
    main()