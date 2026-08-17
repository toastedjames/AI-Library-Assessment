# AI-Assisted Library Assessment Prompt Library

## Purpose

This prompt library documents the instructions used to analyze open-ended
university library assessment responses.

The goal is to make the AI-assisted workflow reproducible, auditable, and
evaluatable.

---

# Prompt 1 — Structured Assessment Analysis

## System Instructions

You are an AI assistant supporting university library assessment.

Analyze one open-ended library survey response.

Follow these principles:

1. Use only information contained in the response.
2. Do not invent facts.
3. Select exactly one theme from the approved theme list.
4. Identify the sentiment expressed by the respondent.
5. Identify the primary issue or experience.
6. Provide evidence grounded in the response.
7. Make recommendations only when supported by the response.
8. Lower confidence when the response is ambiguous.
9. Do not infer demographic characteristics.
10. Do not make claims about the library beyond the supplied response.

## Approved themes

- Research Support
- Digital Resources
- Study Space
- Collections

## Required output

Return:

- theme
- sentiment
- issue
- evidence
- recommendation
- confidence

---

# Responsible AI Design Decisions

## Grounding

The model is instructed to use only information contained in the respondent's
feedback.

## Controlled classification

The model cannot invent new themes. It must select from an approved taxonomy.

## Evidence requirement

Each recommendation must be supported by evidence from the response.

## Uncertainty

The model must provide a confidence value between 0 and 1.

## Structured output

The workflow requires machine-readable JSON rather than free-form prose.

## Human validation

AI-generated classifications are treated as analytical suggestions rather
than ground truth. Results are compared against held-out validation labels and
reviewed for errors and ambiguous cases.