# === query_classifier_node.py ===
import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langchain_core.runnables import RunnableLambda
from schemas import KPIQuery

# === Load .env and wrap OpenAI client for LangSmith tracing ===
load_dotenv()
client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# === Mapping KPI → signals ===
SIGNAL_MAPPING = {
    "NET SALES": ["NET SALES", "NUMBER OF BILLS", "AVERAGE BILL VALUE", "DAILY ACHIEVEMENT %", "AVAILABILITY"],
    "SALES": ["NET SALES", "NUMBER OF BILLS", "AVERAGE BILL VALUE", "DAILY ACHIEVEMENT %", "AVAILABILITY"],
    "ABV": ["AVERAGE BILL VALUE", "NUMBER OF BILLS"],
    "NOB": ["NUMBER OF BILLS", "AVERAGE BILL VALUE"]
}

def build_prompt(query: str) -> str:
    today = "2025-02-28"
    return f'''
You are a query classification assistant for a retail KPI chatbot.

Your job is to convert the following user query into structured JSON with:
1. mentioned_kpis
2. start_date
3. end_date
4. days_back
5. important_dates
6. retrieval_strategy
7. store_names
8. mtd_mode → "yes" if the user wants month-to-date metrics, else "no"

⚠️ Rules:
- If user says "MTD", "month to date", or "till date" → mtd_mode = "yes"
- If not mentioned → mtd_mode = "no"
- If user is asking "why was KPI down" → retrieval_strategy = causal_analysis
- If user is asking for trend → retrieval_strategy = trend_analysis

Today is: {today}

User query:
\"\"\"{query}\"\"\"

Convert this into JSON only:
'''


def extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None

def classify_query_node(state: dict) -> dict:
    user_query = state["user_query"]
    prompt = build_prompt(user_query)

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512
    )

    text_response = response.choices[0].message.content
    json_text = extract_json(text_response)
    if not json_text:
        raise ValueError("❌ No JSON found in response.")

    result = json.loads(json_text)

    # Add user query to result
    result["user_query"] = user_query

    # Auto-add required_signals if missing
    if "required_signals" not in result:
        mentioned = [k.upper() for k in result.get("mentioned_kpis", [])]
        signals = set()
        for kpi in mentioned:
            signals.update(SIGNAL_MAPPING.get(kpi, [kpi]))
        result["required_signals"] = list(signals)

    # Debugging output
    print("\n📌 [CLASSIFIER DEBUG]")
    print("User Query:", user_query)
    print("Mentioned KPIs:", result["mentioned_kpis"])
    print("Start Date:", result["start_date"])
    print("End Date:", result["end_date"])
    print("Days Back:", result["days_back"])
    print("Store Names:", result["store_names"])
    print("Strategy:", result["retrieval_strategy"])
    print("Required Signals:", result["required_signals"])

    validated = KPIQuery(**result).dict()
    return {
        **state,
        "structured": result
    }

query_classifier_node = RunnableLambda(classify_query_node)