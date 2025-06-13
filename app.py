import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from chatbot_graph import chatbot_graph
from chatbot_graph import run_chat_graph

# === Load environment variables ===
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# === Load precomputed data ===
df_precomputed = pd.read_excel("data/kpi_precomputed.xlsx")

# === Streamlit app config ===
st.set_page_config(page_title="Store KPI Chatbot 💬", layout="wide")
st.title(":bar_chart: Store Manager KPI Chatbot")

# === Run the LangGraph ===
# === Run the LangGraph ===
def run_chat_graph(user_query: str, df: pd.DataFrame):
    inputs = {"user_query": user_query, "df": df}
    print("\n🔍 [DEBUG] Inputs passed to chatbot_graph:")
    for k, v in inputs.items():
        print(f"- {k}: type={type(v)}")
    outputs = chatbot_graph.invoke(inputs)
    print("\n✅ [DEBUG] Outputs returned from chatbot_graph:")
    print(outputs)
    return outputs["final_response"], outputs["context_df"]



# === Streamlit user input ===
user_query = st.text_input("Ask your question:")

if user_query:
    with st.spinner("🧠 Thinking..."):
        try:
            response, context_df = run_chat_graph(user_query, df_precomputed)
            st.success("✅ Response:")
            st.markdown(response)

            with st.expander(":bar_chart: View Filtered Data"):
                st.dataframe(context_df)

        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
