# === retrieval_agent_node.py ===
import pandas as pd
from fuzzywuzzy import process
from langchain_core.runnables import RunnableLambda

# === KPI Normalization Mapping ===
KPI_MAPPING = {
    "SALES": "NET SALES",
    "NET SALES": "NET SALES",
    "NOB": "NUMBER OF BILLS",
    "NO OF BILLS": "NUMBER OF BILLS",
    "ABV": "AVERAGE BILL VALUE",
    "AVERAGE BILL VALUE": "AVERAGE BILL VALUE",
}

def normalize_kpis(mentioned_kpis):
    normalized = []
    for kpi in mentioned_kpis:
        kpi_clean = kpi.strip().upper()
        norm_kpi = KPI_MAPPING.get(kpi_clean, kpi_clean)
        normalized.append(norm_kpi)
    return normalized

def fuzzy_match_store_names(df, store_names, threshold=80):
    all_stores = df["Store Name"].astype(str).str.strip().str.upper().unique().tolist()
    matched = []
    for s in store_names:
        s_norm = s.strip().upper()
        match, score = process.extractOne(s_norm, all_stores)
        if score >= threshold:
            matched.append(match)
    return matched

def retrieve_context_node(state: dict) -> dict:
    print("\n🔍 [RETRIEVAL DEBUG] Keys in state:", list(state.keys()))
    print("Structured:", state.get("structured"))

    # ✅ Load the base dataframe
    df = state["df"]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Store Name'] = df['Store Name'].astype(str).str.strip().str.upper()
    df['KPI Name'] = df['KPI Name'].astype(str).str.strip().str.upper()

    print("Before filtering → Rows:", len(df))

    # ✅ ✅ ✅ STEP 2: UNPACK STRUCTURED QUERY IF PRESENT
    structured = state.get("structured", {})
    state_store_names = state.get("store_names", [])
    state_strategy = state.get("retrieval_strategy", "")
    state_start_date = state.get("start_date", None)
    state_end_date = state.get("end_date", None)
    state_important_dates = state.get("important_dates", [])
    state_kpis = state.get("mentioned_kpis", [])

    store_names = [s.strip().upper() for s in structured.get("store_names", state_store_names)]
    strategy = structured.get("retrieval_strategy", state_strategy)
    start_date = pd.to_datetime(structured.get("start_date", state_start_date)) if structured.get("start_date", state_start_date) else None
    end_date = pd.to_datetime(structured.get("end_date", state_end_date)) if structured.get("end_date", state_end_date) else None
    important_dates = pd.to_datetime(structured.get("important_dates", state_important_dates))
    mentioned_kpis = structured.get("mentioned_kpis", state_kpis)

    # ✅ Filter logic (no changes here)
    if store_names:
        matched_store_names = fuzzy_match_store_names(df, store_names)
        df = df[df['Store Name'].isin(matched_store_names)]

    if strategy == "single_date_analysis" and not important_dates.empty:
        target_date = important_dates[0]
        df = df[(df['Date'] >= target_date - pd.Timedelta(days=2)) & (df['Date'] <= target_date)]

    elif strategy == "compare_dates" and len(important_dates) >= 2:
        df = df[df['Date'].isin(important_dates)]

    elif strategy in ["trend_analysis", "full_range"] and start_date and end_date:
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    elif strategy == "causal_analysis":
        causal_kpis = ["NET SALES", "NUMBER OF BILLS", "AVERAGE BILL VALUE", "AVAILABILITY"]
        if not important_dates.empty:
            target_date = important_dates[0]
            df = df[(df['Date'] >= target_date - pd.Timedelta(days=7)) & (df['Date'] <= target_date) & (df['KPI Name'].isin(causal_kpis))]
        else:
            df = df[df['KPI Name'].isin(causal_kpis)]

    if strategy == "trend_analysis" and mentioned_kpis:
        normalized_kpis = normalize_kpis(mentioned_kpis)
        df = df[df['KPI Name'].isin(normalized_kpis)]

    df = df.sort_values(by=['Store Name', 'Date'])
    print("After filtering → Rows:", len(df))

    return {**state, "context_df": df}


# ✅ LangGraph-compatible node
# ✅ Export as LangGraph-compatible node
from langchain_core.runnables import RunnableLambda
retrieval_node = RunnableLambda(retrieve_context_node)
