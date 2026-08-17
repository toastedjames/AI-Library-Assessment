# AI-Assisted Library Assessment Dashboard

A reproducible Python-based workflow for analyzing university library survey data using statistical analysis, NLP, sentiment analysis, and local generative AI.

The project transforms structured and unstructured library assessment data into interpretable metrics, thematic insights, and an interactive assessment dashboard.

---

## Dashboard

![Library Assessment Dashboard](docs/dashboard.png)

---

## Project Overview

University library surveys often contain both quantitative ratings and open-ended feedback.

This project demonstrates an AI-assisted assessment workflow that combines:

- Structured survey analysis
- Student-level comparisons
- NLP-based theme classification
- Sentiment analysis
- Local generative AI
- Prompt engineering
- Assessment metrics
- Interactive visualization

The goal is to help assessment teams move from raw survey responses to **structured, interpretable insights that can support library planning and decision-making**.

---

## Key Results

The project analyzes **300 library survey responses**.

### Overall Assessment Metrics

| Metric | Result |
|---|---:|
| Total responses | **300** |
| Average satisfaction | **3.70 / 5** |
| Digital resources score | **3.78 / 5** |
| Study space score | **3.69 / 5** |
| Research support score | **3.71 / 5** |

### Student-Level Analysis

The workflow also compares assessment scores across:

- Undergraduate students
- Graduate students
- Faculty/Staff

Example results:

| Student Level | Satisfaction | Digital Resources | Study Space | Research Support |
|---|---:|---:|---:|---:|
| Faculty/Staff | 3.60 | 3.50 | 3.68 | 3.65 |
| Graduate | 3.80 | 3.86 | 3.76 | 3.75 |
| Undergraduate | 3.66 | 3.79 | 3.64 | 3.69 |

---

## NLP Analysis

Open-ended survey responses are analyzed using a reproducible NLP pipeline.

### Theme Classification

The project identifies recurring assessment themes including:

- Research Support
- Digital Resources
- Study Space
- Collections

The initial classification workflow uses:

```text
Open-ended feedback
        |
        v
Text preprocessing
        |
        v
TF-IDF representation
        |
        v
Logistic Regression
        |
        v
Predicted assessment theme
```

The model is evaluated using:

* Accuracy
* Macro F1
* Weighted F1
* Classification reports

---

## Sentiment Analysis

The workflow also analyzes qualitative feedback for sentiment.

Responses are classified as:

* Positive
* Neutral
* Negative

Sentiment results can be combined with thematic classification to identify areas requiring attention.

For example:

```text
Study Space
    |
    +-- Positive feedback
    |
    +-- Neutral feedback
    |
    +-- Negative feedback
```

This allows assessment teams to distinguish between frequently mentioned topics and topics associated with dissatisfaction.

---

## Generative AI Integration

The project incorporates a local generative AI model:

**Qwen2.5-1.5B-Instruct**

The model is used to assist with:

* Qualitative feedback interpretation
* Theme identification
* Sentiment analysis
* Recommendation generation
* Structured assessment outputs

The model runs locally through Hugging Face Transformers, avoiding dependence on a paid external API.

---

## Prompt Engineering

The generative AI workflow uses structured prompts to improve consistency.

The refined workflow requires the model to:

* Identify the main issue
* Assign an approved assessment theme
* Generate a concise recommendation
* Ground the recommendation in the survey response
* Avoid unsupported claims
* Return structured output

This creates a reproducible workflow suitable for experimentation and evaluation.

---

## Dashboard

The project includes an interactive dashboard for communicating assessment results.

The dashboard presents information such as:

* Overall satisfaction
* Digital resource scores
* Study-space scores
* Research-support scores
* Student-level comparisons
* Theme distributions
* Sentiment distributions
* AI-generated assessment insights

The dashboard is designed to make model outputs and survey findings accessible to non-technical stakeholders.

---

## Workflow

```text
                    Library Survey
                          |
                          v
                  Data Preparation
                          |
              +-----------+-----------+
              |                       |
              v                       v
      Structured Data           Open Feedback
              |                       |
              v                       v
      Statistical Analysis       NLP Analysis
              |                       |
              |              +--------+--------+
              |              |                 |
              |              v                 v
              |          Theme Model      Sentiment
              |              |                 |
              +--------------+-----------------+
                             |
                             v
                    Generative AI Layer
                             |
                             v
                    Assessment Insights
                             |
                             v
                       Dashboard
```

---

## Technologies

* **Python**
* **Pandas**
* **NumPy**
* **scikit-learn**
* **PyTorch**
* **Hugging Face Transformers**
* **Qwen2.5-1.5B-Instruct**
* **TF-IDF**
* **Logistic Regression**
* **HTML / CSS / JavaScript**
* **Chart-based data visualization**

---

## Project Structure

```text
AI Library Assessment/
│
├── data/
│   └── library_survey.csv
│
├── src/
│   ├── analyze.py
│   ├── nlp_analysis.py
│   ├── ai_analysis.py
│   ├── evaluate.py
│   └── ...
│
├── outputs/
│   ├── assessment_summary.csv
│   ├── nlp_analysis_results.csv
│   ├── ai_analysis_results_v2.csv
│   └── ...
│
├── dashboard/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── docs/
│   └── dashboard.png
│
├── requirements.txt
└── README.md
```

---

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Analysis

### Stage 1 — Statistical Assessment

Run:

```powershell
python -m src.analyze
```

This produces:

* Total response count
* Average satisfaction
* Digital resource scores
* Study-space scores
* Research-support scores
* Student-level comparisons

---

### Stage 2 — NLP Analysis

Run:

```powershell
python -m src.nlp_analysis
```

This performs:

* TF-IDF feature extraction
* Logistic Regression classification
* Theme prediction
* Sentiment analysis
* Classification evaluation

Results are written to:

```text
outputs/
```

---

### Stage 3 — Local AI Analysis

Run:

```powershell
python -m src.ai_analysis
```

The workflow loads:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

and processes a controlled sample of survey responses.

The model produces structured assessment outputs that can be incorporated into downstream analysis.

---

## Dashboard

After generating the analysis outputs, open:

```text
dashboard/index.html
```

in a browser.

The dashboard reads the generated assessment data and presents the results through an interactive interface.

For a cleaner demonstration, the dashboard screenshot can be stored as:

```text
docs/dashboard.png
```

and displayed in this README.

---

## Responsible AI

The project treats AI as an **assessment-support tool**, rather than an autonomous decision-maker.

Key principles include:

### Human Oversight

AI-generated findings should be reviewed by library assessment professionals before being used for institutional decisions.

### Grounded Analysis

The workflow is designed to keep AI-generated interpretations tied to supplied survey evidence.

### Reproducibility

Analysis scripts and generated outputs are separated into clear stages so the workflow can be rerun and audited.

### Model Evaluation

AI outputs are evaluated quantitatively rather than assuming that generative AI outputs are automatically correct.

### Privacy

Real institutional survey data should be reviewed for personally identifiable or sensitive information before being processed by AI systems.

---

## Limitations

This project is an exploratory assessment prototype.

Limitations include:

* Survey data is experimental rather than production institutional data.
* NLP performance depends on the quality and distribution of labeled examples.
* Small local language models may produce inconsistent interpretations.
* Automated sentiment and theme classifications require validation.
* Dashboard findings should not be treated as institutional decisions without human review.
* Production deployment would require institutional data governance and approved AI tooling.

---

## What This Project Demonstrates

This project demonstrates practical experience with:

* Library assessment analytics
* Structured and unstructured data
* NLP
* Generative AI
* Prompt engineering
* Data visualization
* Python-based analytical workflows
* Reproducible analysis
* AI-assisted reporting
* Responsible AI
* Translating survey data into stakeholder-facing insights

---

## Intended Application

The workflow is designed for potential use cases such as:

* Library user satisfaction analysis
* Assessment survey analysis
* Qualitative feedback categorization
* Strategic planning support
* Identification of recurring service issues
* Student experience analysis
* AI-assisted assessment reporting

The system is intended to **augment assessment professionals**, not replace human interpretation or institutional decision-making.

---

## Author

**Somak Goswami**

M.S. Electrical Engineering
Virginia Tech

Interests: AI/ML, NLP, data analytics, responsible AI, and AI-assisted institutional workflows.
