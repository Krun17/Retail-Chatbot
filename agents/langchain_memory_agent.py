from langchain.memory import ConversationBufferMemory

# Initialize memory
chat_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Log a query/response pair
def log_to_memory(user_query: str, bot_response: str):
    chat_memory.chat_memory.add_user_message(user_query)
    chat_memory.chat_memory.add_ai_message(bot_response)

# Retrieve the entire memory history
def get_chat_history():
    memory = chat_memory.load_memory_variables({})
    return memory.get("chat_history", "")

# Clear memory (if needed)
def reset_memory():
    chat_memory.clear()
