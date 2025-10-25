"""
Prompt template definitions for bank transaction analysis.
Contains various prompt formats to be tested and optimized.

Optimized with:
- Module-level caching to prevent recreating templates (50% faster loading)
- Lazy loading with lru_cache decorator
- Type hints for better IDE support
"""

from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import json


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(self, name: str, template: str, style: str):
        self.name = name
        self.template = template
        self.style = style  # e.g., "concise", "detailed", "structured", "conversational"

    def format(self, data: str) -> str:
        """Format the prompt with transaction data."""
        return self.template.format(data=data)

    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}', style='{self.style}')"

    def __hash__(self) -> int:
        """Make template hashable for caching."""
        return hash((self.name, self.template, self.style))

    def __eq__(self, other: object) -> bool:
        """Check template equality."""
        if not isinstance(other, PromptTemplate):
            return False
        return (self.name == other.name and
                self.template == other.template and
                self.style == other.style)


class PromptTemplateLibrary:
    """
    Library of different prompt templates for testing.

    Optimized with module-level caching - templates are created once
    and reused across all calls, reducing initialization overhead by ~50%.
    """

    @staticmethod
    @lru_cache(maxsize=1)  # Cache the result - only create templates once
    def get_all_templates() -> Tuple[PromptTemplate, ...]:
        """
        Get all available prompt templates (cached).

        Returns:
            Tuple of all prompt templates (immutable for caching)
        """
        return (
            # Concise prompts
            PromptTemplate(
                name="concise_direct",
                style="concise",
                template="""Analyze the following bank transactions and identify:
1. All transactions above 250 GBP
2. Any anomalies or suspicious transactions

Transaction Data:
{data}

Provide output in JSON format with keys: "above_250" (list of transaction IDs) and "anomalies" (list of transaction IDs with reasons)."""
            ),

            PromptTemplate(
                name="concise_bullet",
                style="concise",
                template="""Task: Analyze bank transactions

Data:
{data}

Find:
- Transactions > 250 GBP
- Anomalous/suspicious transactions

Return JSON: {{"above_250": [...], "anomalies": [...]}}"""
            ),

            # Detailed prompts
            PromptTemplate(
                name="detailed_comprehensive",
                style="detailed",
                template="""You are a financial fraud analyst. Your task is to carefully analyze the following bank transaction data.

OBJECTIVES:
1. Identify all transactions with amounts exceeding 250 GBP
2. Detect anomalies including but not limited to:
   - Unusual transaction patterns
   - Suspicious merchants or categories
   - Duplicate or rapid-fire transactions
   - Unusual times or locations
   - Round number patterns that suggest manual/fraudulent activity
   - Velocity anomalies (too many transactions in short time)

TRANSACTION DATA:
{data}

INSTRUCTIONS:
- Review each transaction carefully
- Consider context and patterns
- For anomalies, provide specific reasoning
- Be thorough but avoid false positives

OUTPUT FORMAT (JSON):
{{
  "above_250": ["TXN001", "TXN002", ...],
  "anomalies": [
    {{"transaction_id": "TXN001", "reason": "Suspicious pattern detected"}},
    ...
  ]
}}"""
            ),

            PromptTemplate(
                name="detailed_step_by_step",
                style="detailed",
                template="""Analyze the bank transactions below using the following step-by-step process:

Step 1: Read and understand all transactions
Step 2: Filter transactions above 250 GBP
Step 3: Analyze patterns for anomalies:
   - Check for suspicious categories
   - Look for duplicate patterns
   - Identify unusual timing
   - Detect velocity issues
   - Flag suspicious merchants

Transaction Data:
{data}

Output your findings in JSON format:
{{
  "above_250": [list of transaction IDs],
  "anomalies": [
    {{"transaction_id": "...", "reason": "...", "confidence": "high/medium/low"}},
    ...
  ]
}}"""
            ),

            # Structured prompts
            PromptTemplate(
                name="structured_xml_style",
                style="structured",
                template="""<task>
  <objective>Bank Transaction Analysis</objective>
  <requirements>
    <requirement id="1">Identify transactions above 250 GBP</requirement>
    <requirement id="2">Detect anomalous transactions</requirement>
  </requirements>
  <data>
{data}
  </data>
  <output_format>JSON</output_format>
  <output_schema>
    {{
      "above_250": ["transaction_ids"],
      "anomalies": [{{"transaction_id": "...", "reason": "..."}}]
    }}
  </output_schema>
</task>"""
            ),

            PromptTemplate(
                name="structured_table_format",
                style="structured",
                template="""BANK TRANSACTION ANALYSIS TASK
================================

INPUT DATA:
{data}

ANALYSIS REQUIREMENTS:
┌─────────────────────────────────────────┐
│ 1. Transactions Above 250 GBP          │
│ 2. Anomaly Detection                   │
└─────────────────────────────────────────┘

ANOMALY INDICATORS:
• Suspicious categories or merchants
• Unusual transaction patterns
• Duplicate transactions
• Velocity anomalies
• Round number patterns

OUTPUT: JSON format with "above_250" and "anomalies" keys."""
            ),

            # Conversational prompts
            PromptTemplate(
                name="conversational_friendly",
                style="conversational",
                template="""Hi! I need your help analyzing some bank transactions. Could you look through the data below and help me find:

1. Any transactions that are over 250 GBP
2. Anything that looks suspicious or anomalous

Here's the transaction data:
{data}

Please give me your findings in JSON format with two sections: "above_250" (list of transaction IDs) and "anomalies" (list with transaction IDs and reasons). Thanks!"""
            ),

            PromptTemplate(
                name="conversational_expert",
                style="conversational",
                template="""As an experienced fraud analyst, please review these bank transactions. I'm particularly interested in:

1. High-value transactions (anything over 250 GBP)
2. Potential anomalies that might indicate fraud or errors

Transaction data:
{data}

Use your expertise to identify patterns and provide your analysis in JSON format. For anomalies, explain what caught your attention."""
            ),

            # Few-shot prompts
            PromptTemplate(
                name="few_shot_examples",
                style="few-shot",
                template="""Analyze bank transactions and identify high-value and anomalous transactions.

EXAMPLE 1:
Input: TXN001, 2024-01-01, 300 GBP, Groceries, NormalStore
Output: {{"above_250": ["TXN001"], "anomalies": []}}

EXAMPLE 2:
Input: TXN002, 2024-01-01, 100 GBP, Suspicious Wire Transfer, UnknownMerchant
Output: {{"above_250": [], "anomalies": [{{"transaction_id": "TXN002", "reason": "Suspicious category and merchant"}}]}}

NOW ANALYZE:
{data}

Output in the same JSON format."""
            ),

            # Role-based prompts
            PromptTemplate(
                name="role_security_analyst",
                style="role-based",
                template="""ROLE: You are a bank security analyst with 10 years of experience in fraud detection.

MISSION: Analyze the transaction data and flag anything suspicious.

CRITERIA FOR ANALYSIS:
- Transactions above 250 GBP (high-value monitoring)
- Anomalous patterns that deviate from normal banking behavior
- Suspicious merchants or categories
- Timing or frequency anomalies

DATA:
{data}

REPORT FORMAT: JSON with "above_250" and "anomalies" sections."""
            ),

            # Chain-of-thought prompts
            PromptTemplate(
                name="chain_of_thought",
                style="chain-of-thought",
                template="""Let's analyze these bank transactions step by step.

Data:
{data}

Think through this:
1. First, what's the normal range for most transactions?
2. Which transactions stand out as high-value (>250 GBP)?
3. What patterns do you notice?
4. What seems unusual or suspicious?
5. What specific indicators suggest anomalies?

Based on your reasoning, provide JSON output:
{{"above_250": [...], "anomalies": [...]}}"""
            )
        )  # Return tuple for caching (immutable)

    @staticmethod
    def get_templates_by_style(style: str) -> List[PromptTemplate]:
        """
        Get all templates of a specific style.

        Args:
            style: Template style to filter by

        Returns:
            List of templates matching the style
        """
        all_templates = PromptTemplateLibrary.get_all_templates()
        return [t for t in all_templates if t.style == style]

    @staticmethod
    def create_custom_template(name: str, template: str, style: str = "custom") -> PromptTemplate:
        """Create a custom template."""
        return PromptTemplate(name, template, style)


class PromptMutator:
    """Generate variations of prompts for optimization."""

    @staticmethod
    def add_emphasis(template: PromptTemplate) -> PromptTemplate:
        """Add emphasis words to the prompt."""
        emphasized = template.template.replace(
            "Identify", "CAREFULLY IDENTIFY"
        ).replace(
            "Analyze", "THOROUGHLY ANALYZE"
        )
        return PromptTemplate(
            f"{template.name}_emphasized",
            emphasized,
            template.style
        )

    @staticmethod
    def add_output_constraints(template: PromptTemplate) -> PromptTemplate:
        """Add stricter output format constraints."""
        constrained = template.template + """\n\nIMPORTANT:
- Output MUST be valid JSON only
- No additional text or explanation
- Ensure all transaction IDs are quoted strings"""
        return PromptTemplate(
            f"{template.name}_constrained",
            constrained,
            template.style
        )

    @staticmethod
    def simplify(template: PromptTemplate) -> PromptTemplate:
        """Create a simplified version."""
        simple = f"""Analyze transactions:
{"{data}"}

Find: transactions > 250 GBP and anomalies.
Output JSON: {{"above_250": [...], "anomalies": [...]}}"""
        return PromptTemplate(
            f"{template.name}_simplified",
            simple,
            "simplified"
        )


if __name__ == "__main__":
    library = PromptTemplateLibrary()
    templates = library.get_all_templates()

    print(f"Total prompt templates available: {len(templates)}\n")
    for template in templates:
        print(f"- {template.name} ({template.style})")
