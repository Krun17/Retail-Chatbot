from datetime import datetime
import uuid

class MemoryLogger:
    def __init__(self):
        self.logs = []  # Stores all memory objects

    def log_interaction(
        self,
        user_query: str,
        structured_query: dict = None,
        code: str = None,
        result_df_sample: str = None,
        final_response: str = None,
        path_used: str = None  # "precomputed" or "exec"
    ):
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "structured_query": structured_query,
            "path_used": path_used,
            "code": code,
            "result_sample": result_df_sample,
            "final_response": final_response
        }
        self.logs.append(log_entry)

    def get_recent_logs(self, n=5):
        return self.logs[-n:]

    def search_by_query(self, keyword):
        return [log for log in self.logs if keyword.lower() in log["user_query"].lower()]
    
    def clear_logs(self):
        self.logs = []

    def export_logs(self):
        """Return logs as list of dicts (can be written to CSV/JSON if needed)."""
        return self.logs