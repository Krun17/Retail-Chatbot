# === response_agent_node.py ===
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Annotated
import pandas as pd

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langchain.callbacks import tracing_v2_enabled
import os
from dotenv import load_dotenv

load_dotenv()

client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# === 🧠 State Type ===
class ResponseState(TypedDict):
    user_query: str
    structured: dict
    context_df: pd.DataFrame
    final_response: Annotated[str, "final chatbot response"]

# === 🔧 Prompt Builder ===
def build_chat_prompt(user_query: str, structured_query: dict, df: pd.DataFrame) -> list:
    mentioned_kpis = structured_query.get("mentioned_kpis") or []
    if not isinstance(mentioned_kpis, list):
        mentioned_kpis = [mentioned_kpis]
    kpis = ", ".join(mentioned_kpis)
    strategy = structured_query.get("strategy", "")
    dates = ", ".join(structured_query.get("important_dates", [])) or f"{structured_query.get('start_date', '')} to {structured_query.get('end_date', '')}"

    df_cleaned = df.copy()
    table_str = df_cleaned.head(20).to_markdown(index=False)

    system_msg = {
        "role": "system",
        "content": """
You are a retail KPI analytics expert.
- The user is asking for performance, trend, or root cause analysis.
- You are provided data for recent days including: Net Sales, Daily Plan, Daily Actual, Daily Achievement %, Plan, Actual, and other KPI metrics.
- Always analyze the **entire date range** — never just a few rows.
- Your answer must be based **only on the provided table below**. Do not hallucinate values.

📌 THINKING STYLE (Chain-of-Thought):
- Begin by examining the trend of Daily Actual values.
- Then check if performance met expectations using Daily Achievement %.
- Use comparative analysis between days to explain trends.
- Mention if any drop/rise is gradual or sharp.
- Look for consistency and volatility in Daily Achievement %.
- Think: what changed on the day with anomaly? Then explain why.

📊 KPI COLUMN RULES:

1. ✅ **Net Sales**:
    - Default: Use **Daily Actual** and **Daily Achievement %**.
    - ❌ Do NOT use Plan or Actual (MTD) unless user explicitly mentions "MTD", "month to date", or "plan vs actual".
    - Always comment on Daily Achievement % to reflect plan adherence.

2. ✅ **Other KPIs** (Average Bill Value, Number of Bills, Availability, Basket Builder Availability, JioMart SLA Adherence):
    - Use: **Plan** and **Actual** columns.
    - ❌ Ignore: Daily Plan, Daily Actual, Daily Achievement % (not applicable).

3. 🚫 NEVER mix Net Sales rules with other KPIs. Apply each set of rules individually.

📊 MULTI-STORE RULE:
- If data includes multiple stores → analyze each one separately.
- Always name the store in your response.
- If no store name in user query → give insights store-wise clearly.

❓ FEW-SHOT EXAMPLES:

- User Query: *"What is the Net Sales trend of last 7 days?"*
  → Use: Daily Actual + Daily Achievement %.  
  → Example Insight: "Net Sales showed steady growth, rising from X to Y, but Achievement % fell sharply on the last day, suggesting plan hike."

- User Query: *"Give MTD Net Sales trend for GURUGRAM AMBI MALL"*
  → Use: Plan vs Actual columns (MTD).
  → Example Insight: "As of 27th Feb, Net Sales MTD is below plan by X%, driven by shortfall in ABV."

- User Query: *"Why was Net Sales down on 26th?"*
  → Use: Daily Actual and Daily Achievement % of Net Sales.
  → Then examine: ABV, NOB, Availability 7 days before that date.
  → Explain drop using chain-of-thought analysis.

🧠 ROOT CAUSE ANALYSIS:
- Use Daily Achievement % to identify weak performance.
- Then evaluate supporting KPIs (ABV, NOB, Availability) for cause.
- Check 7-day average of those KPIs before the date and compare.
- Use cautious reasoning: “data suggests”, “possibly due to”, “drop from trend”, etc.

🚫 FINAL RESTRICTIONS:
- Never hallucinate or invent data outside table.
- Do not mention "Daily Plan" unless user asks specifically.
- Avoid speculation — focus on trends that can be explained with data.

🧾 Your final output should be:
- Cleanly formatted
- In plain business language
- With a clear summary for decision-making

"""
    }

    user_msg = {
        "role": "user",
        "content": f"""
The user asked: \"{user_query}\"

KPI(s): {kpis}
Strategy: {strategy}
Relevant Dates: {dates}

Here is the relevant data (first 20 rows only):

{table_str}

Please analyze carefully and give full reasoning using trends across all days. Avoid speculation outside the data.
"""
    }

    return [system_msg, user_msg]

# === ✅ LangGraph Node ===
def response_agent_node(state: ResponseState) -> ResponseState:
    print("\n🧠 [DEBUG] Keys in response agent state:")
    print(list(state.keys()))
    user_query = state["user_query"]
    structured = state["structured"]
    df = state["context_df"]

    if df.empty:
        return {**state, "final_response": "❌ No data available to answer this query."}

    with tracing_v2_enabled():
        messages = build_chat_prompt(user_query, structured, df)
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )
            answer = response.choices[0].message.content.strip()
            return {**state, "final_response": answer}
        except Exception as e:
            return {**state, "final_response": f"⚠️ Error generating response: {str(e)}"}

# ✅ Export as LangGraph Runnable
# ✅ Export as LangGraph Runnable
from langchain_core.runnables import RunnableLambda
response_node = RunnableLambda(response_agent_node)


