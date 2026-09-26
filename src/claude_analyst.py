"""
AI investigation report via Anthropic Claude.
Falls back to a structured template if no API key is configured.
Works with: ANTHROPIC_API_KEY in Streamlit secrets or environment variable.
"""
from __future__ import annotations
import os
import textwrap


def _api_key() -> str | None:
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def is_claude_available() -> bool:
    return bool(_api_key())


def _template(amount: float, risk: float, verdict: str,
               proba: float, top5: list) -> str:
    """Structured fallback report when Claude is not configured."""
    factors = "\n".join(
        f"  • {n}: {v:.4f}  (impact: {'+' if s > 0 else ''}{s:.4f} — "
        f"{'increases' if s > 0 else 'decreases'} fraud risk)"
        for n, v, s in top5
    ) or "  • Feature data unavailable (run SHAP analysis first)"

    action_map = {
        "CONFIRMED FRAUD": (
            "IMMEDIATE BLOCK — Decline this transaction and freeze the card. "
            "Escalate to fraud operations. Initiate cardholder verification via "
            "registered phone number."
        ),
        "HIGH RISK": (
            "TEMPORARY HOLD — Place a 4-hour hold. Contact the cardholder via "
            "registered number for out-of-band verification before releasing funds."
        ),
        "NEEDS REVIEW": (
            "MANUAL REVIEW — Flag for fraud analyst review within 4 hours. "
            "Allow transaction provisionally; monitor for follow-up activity."
        ),
        "LEGITIMATE": (
            "NO ACTION REQUIRED — Transaction is within normal behavioral parameters. "
            "Continue standard monitoring."
        ),
    }

    return textwrap.dedent(f"""
    ═══ FRAUD INVESTIGATION REPORT (Demo Mode — Claude not configured) ═══

    EXECUTIVE SUMMARY
    A transaction of ${amount:.2f} has been assessed with a risk score of {risk:.0f}/100,
    yielding a model verdict of {verdict} ({proba:.2%} fraud probability). This report
    was generated using rule-based analysis. Configure ANTHROPIC_API_KEY for
    full AI-generated narrative reports.

    KEY RISK INDICATORS
{factors}

    RECOMMENDED ACTION
    {action_map.get(verdict, 'Review required.')}

    NOTE
    To enable AI-powered reports: add ANTHROPIC_API_KEY to .streamlit/secrets.toml
    ═════════════════════════════════════════════════════════════════════
    """).strip()


def generate_report(
    amount: float,
    risk: float,
    verdict: str,
    proba: float,
    top5: list,
) -> str:
    """
    Generate an investigation report.
    Uses Claude claude-3-5-haiku if API key available; else returns template.
    """
    key = _api_key()
    if not key:
        return _template(amount, risk, verdict, proba, top5)

    try:
        import anthropic

        factors = "\n".join(
            f"  - {n} = {v:.4f}  (SHAP: {s:+.4f} → "
            f"{'increases' if s > 0 else 'decreases'} fraud risk)"
            for n, v, s in top5
        ) or "  - Feature data not available"

        prompt = f"""You are a senior financial fraud analyst.

A real-time monitoring system has flagged this transaction:

Transaction Amount : ${amount:.2f}
Risk Score         : {risk:.0f} / 100
Model Verdict      : {verdict}
Fraud Probability  : {proba:.2%}

Top Contributing Factors (SHAP explainability analysis):
{factors}

Write a concise, professional fraud investigation report with exactly these four sections:

1. EXECUTIVE SUMMARY (2-3 sentences: what happened and how serious it is)
2. KEY RISK INDICATORS (3-5 bullets, each referencing a specific SHAP factor above)
3. RECOMMENDED ACTION (specific and actionable, e.g. block/hold/review)
4. REGULATORY CONSIDERATIONS (mention AML thresholds only if amount > $10,000 or risk > 85)

Rules: Be specific. Reference actual SHAP values. Under 300 words total. No markdown headers."""

        client  = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model      = "claude-3-5-haiku-20241022",
            max_tokens = 600,
            messages   = [{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as exc:
        return _template(amount, risk, verdict, proba, top5) + f"\n\n⚠️ Claude error: {exc}"
