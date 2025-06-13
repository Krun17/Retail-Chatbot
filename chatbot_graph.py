from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import Runnable
from typing import Annotated, TypedDict
from pydantic import BaseModel
import pandas as pd

# ✅ Node imports
from agents.query_classifier_node import query_classifier_node
from agents.retrieval_agent_node import retrieval_node
from agents.response_agent_node import response_node

# ✅ Load data
df_precomputed = pd.read_excel("data/kpi_precomputed.xlsx")

# ✅ Define LangGraph state
class ChatState(TypedDict):
    user_query: str
    df: pd.DataFrame
    structured: dict
    context_df: pd.DataFrame
    final_response: str

# ✅ Build LangGraph
graph = StateGraph(ChatState)
graph.add_node("query_classifier", query_classifier_node)
graph.add_node("retrieve_context", retrieval_node)
graph.add_node("generate_response", response_node)

graph.set_entry_point("query_classifier")
graph.add_edge("query_classifier", "retrieve_context")
graph.add_edge("retrieve_context", "generate_response")
graph.add_edge("generate_response", END)

# ✅ Compile graph
chatbot_graph = graph.compile()

# ✅ Now define this AFTER graph is compiled
def run_chat_graph(user_query: str, df: pd.DataFrame):
    inputs = {"user_query": user_query, "context_df": df}
    outputs = chatbot_graph.invoke(inputs)
    return outputs["final_response"], outputs["context_df"]
