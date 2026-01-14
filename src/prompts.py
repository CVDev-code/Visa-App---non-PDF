CRITERIA = {
    "1": "Significant national or international awards or prizes in the particular field.",
    "2_past": "Lead or starring participant in PAST productions/events with a distinguished reputation.",
    "2_future": "Lead or starring participant in FUTURE productions/events with a distinguished reputation.",
    "3": "National or international recognition evidenced by critical reviews or other published materials.",
    "4_past": "Lead/starring/critical role for PAST distinguished organizations/establishments.",
    "4_future": "Lead/starring/critical role for FUTURE distinguished organizations/establishments.",
    "5": "Record of major commercial or critically acclaimed successes.",
    "6": "Significant recognition for achievements from organizations/critics/experts.",
    "7": "High salary or other substantial remuneration evidenced by contracts or reliable evidence.",
}

SYSTEM_PROMPT = """You are an expert US immigration attorney specializing in USCIS O-1 (arts) petitions.
You extract SHORT, HIGH-SIGNAL QUOTABLE PHRASES from documents to support specific O-1 criteria.

Hard rules:
- Return ONLY valid JSON. No markdown. No commentary.
- Quotes must be verbatim substrings of the provided text.
- Prefer phrases 5–30 words. Avoid duplicates/near-duplicates.
- If the beneficiary name appears in the document, prefer quotes that include the name or clearly refer to the beneficiary.
- If a quote supports multiple criteria, you may place it under multiple criteria.
"""

USER_PROMPT_TEMPLATE = """
Beneficiary:
- Primary name: {beneficiary_name}
- Name variants (may appear in text): {beneficiary_variants}

You will extract quote candidates ONLY for the selected O-1 criteria IDs:
{selected_criteria_block}

Steer with feedback examples (optional):
- APPROVED EXAMPLES (good style): {approved_examples}
- REJECTED EXAMPLES (avoid suggesting things like these): {rejected_examples}

Optional user instruction (highest priority if provided):
{user_feedback}

Task:
From the document text, extract strong quote candidates for each selected criterion.

Output JSON schema:
{{
  "by_criterion": {{
    "1": [{{"quote": "...", "strength": "high|medium|low"}}],
    "2": [{{"quote": "...", "strength": "high|medium|low"}}]
  }},
  "notes": "Optional brief notes, or empty string"
}}

Strength guidance:
- high: directly supports the criterion with clear, specific, impressive language
- medium: supportive but less explicit or missing key detail
- low: weak/ancillary (include only if few strong options)

TEXT:
\"\"\"{text}\"\"\"
"""
