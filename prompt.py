prompt_generate_issue = """
You are a cybersecurity expert tasked with generating degraded variants of detection rules for evaluation purposes.

OBJECTIVE:
Given a high-quality detection rule and its context, generate transformed versions that each contain a specific deficiency from the provided taxonomy.

INPUTS:
Context: {description}
Base_Rule: {official_rule}
Deficiency_Taxonomy:
{error_details}

REQUIREMENTS:
1. Generate ONE variant for EACH deficiency type in the taxonomy
2. Each variant must contain ONLY its designated deficiency type
3. Maintain syntactic validity of the rule
4. Provide explanation of the change

OUTPUT FORMAT:
Return a JSON array with one object per deficiency type. Inside JSON strings, escape all double quotes with \\" when needed:

```json
[
{{
    "rule": "transformed rule with specific deficiency",
    "label_type": "{label_type}",
    "explanation": "describe what was changed and what problem it introduces"
}}
]
```

Note: Focus on meaningful degradation that affects the rule's functionality, precision, or usability, not just cosmetic changes.
"""

prompt_judge_issue = """
You are a cybersecurity expert tasked with evaluating detection rules through pairwise comparison.

OBJECTIVE:
Given two detection rules designed for the same detection context, determine which version is weaker, identify the deficiency type in the inferior rule, and provide explanation.

INPUTS:
Context:
{description}

Rule version_1:
{rule1}

Rule version_2:
{rule2}

Deficiency_Taxonomy (valid deficiency types with descriptions):
{error_details}

REQUIREMENTS:
- Do NOT judge based on formatting, field names, or minor structural differences
- Focus on whether the logic or conditions deviate from the described analytic intent
- Identify which specific deficiency type from the taxonomy applies

OUTPUT FORMAT:
{{
    "weaker_rule": "version_1 or version_2",
    "label_type": "{label_type}",
    "explanation": "describe what was changed in the worse version and what problem it introduces"
}}
"""

prompt_judge_explanation = """
You are a cybersecurity expert tasked with comparing explanations of detection rule differences.

OBJECTIVE:
Determine if two explanations describe the same detection rule difference. Two explanations are semantically equivalent if they identify the same rule change and the same security problem introduced by that change, even if they use different phrasing or levels of detail.

INPUTS:
Explanation 1:
{explanation_1}

Explanation 2:
{explanation_2}

EVALUATION CRITERIA:
- Do they identify the same underlying rule change?
- Do they describe the same security impact or problem?
- Ignore differences in wording, detail level, or presentation style

OUTPUT:
Output exactly one word: True or False

True if: Both explanations describe the same rule change and same security problem
False if: They describe different changes, different impacts, or one misses key information
"""
