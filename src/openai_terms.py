import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, CRITERIA


def _get_secret(name: str):
    # Works on Streamlit Cloud (st.secrets) and locally (.env / env vars)
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def suggest_ovisa_quotes(
    document_text: str,
    beneficiary_name: str,
    beneficiary_variants: List[str],
    selected_criteria_ids: List[str],
    feedback: Optional[Dict] = None,
    user_feedback_text: Optional[str] = None,
) -> Dict:
    """
    Returns:
      {
        "by_criterion": {"1":[{"quote":"...","strength":"high"}], ...},
        "notes": "..."
      }

    feedback optional:
      {"approved_examples":[...], "rejected_examples":[...]}
    user_feedback_text optional:
      a free-text instruction from the user
    """
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = _get_secret("OPENAI_MODEL") or "gpt-4.1-mini"
    client = OpenAI(api_key=api_key)

    # Build criteria block shown to the model (only selected)
    selected_lines = []
    for cid in selected_criteria_ids:
        desc = CRITERIA.get(cid, "")
        selected_lines.append(f"- ({cid}) {desc}")
    selected_criteria_block = "\n".join(selected_lines) if selected_lines else "None"

    approved = (feedback or {}).get("approved_examples", [])
    rejected = (feedback or {}).get("rejected_examples", [])

    uf = (user_feedback_text or "").strip()
    user_feedback_block = f"USER FEEDBACK:\n{uf}" if uf else "None"

    prompt = USER_PROMPT_TEMPLATE.format(
        beneficiary_name=beneficiary_name.strip(),
        beneficiary_variants=", ".join([v.strip() for v in beneficiary_variants if v.strip()]) or "None",
        selected_criteria_block=selected_criteria_block,
        approved_examples="\n".join(approved) if approved else "None",
        rejected_examples="\n".join(rejected) if rejected else "None",
        user_feedback=user_feedback_block,
        text=document_text,
    )

    # New SDKs accept response_format; older ones raise TypeError.
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.output_text
    except TypeError:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw = getattr(resp, "output_text", None)
        if not raw:
            try:
                raw = resp.output[0].content[0].text
            except Exception:
                raw = str(resp)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenAI returned invalid JSON:\n{raw}") from e

    by_criterion = data.get("by_criterion", {})
    if not isinstance(by_criterion, dict):
        by_criterion = {}

    # Normalize
    cleaned = {}
    for cid in selected_criteria_ids:
        items = by_criterion.get(cid, [])
        if not isinstance(items, list):
            items = []
        out_items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            q = it.get("quote")
            s = it.get("strength", "medium")
            if isinstance(q, str) and q.strip():
                out_items.append(
                    {
                        "quote": q.strip(),
                        "strength": s if s in {"high", "medium", "low"} else "medium",
                    }
                )
        cleaned[cid] = out_items

    return {
        "by_criterion": cleaned,
        "notes": data.get("notes", "") if isinstance(data.get("notes", ""), str) else "",
    }
