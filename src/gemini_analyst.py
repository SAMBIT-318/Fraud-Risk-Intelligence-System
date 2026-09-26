import streamlit as st
from google import genai
from google.genai import types

def get_client():
    """Retrieve Gemini client using Streamlit secrets or environment variables."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return None

def is_gemini_available() -> bool:
    """Check if the Gemini API key is configured."""
    return bool(st.secrets.get("GEMINI_API_KEY"))

def generate_report(amount: float, risk: float, verdict: str, proba: float, top5: list) -> str:
    """Generate an AI fraud investigation summary using Gemini."""
    client = get_client()
    if not client:
        return "⚠️ Gemini API key is not configured. Please add GEMINI_API_KEY to your Streamlit secrets."

    # Format the top SHAP contributing features
    shap_summary = "\n".join(
        [f"- Feature {fname} (value: {fval:.4f}) {'increased' if impact > 0 else 'decreased'} risk by {abs(impact):.4f}"
         for fname, fval, impact in top5]
    ) if top5 else "No specific SHAP feature attributions available."

    prompt = f"""
You are a senior Fraud Risk Intelligence Analyst. Evaluate the following flagged transaction and provide a concise, structured investigation report.

Transaction Details:
- Amount: ${amount:,.2f}
- Risk Score: {risk}/100
- Model Verdict: {verdict}
- Fraud Probability: {proba:.2%}

Key Contributing Factors (SHAP Analysis):
{shap_summary}

Please provide:
1. **Executive Summary**: A 2-sentence assessment of the transaction risk.
2. **Key Risk Indicators**: Analysis of the primary drivers behind the verdict.
3. **Recommended Actions**: Clear next steps (e.g., immediate freeze, manual analyst review, phone verification, or approve).
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
            )
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"
