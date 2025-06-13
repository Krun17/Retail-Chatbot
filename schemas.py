from pydantic import BaseModel, Field
from typing import List, Optional

class KPIQuery(BaseModel):
    user_query: str                                      # ✅ Add this line
    mentioned_kpis: List[str] = Field(default_factory=list)
    mtd_mode: Optional[str] = "no"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    important_dates: List[str] = Field(default_factory=list)
    retrieval_strategy: str
    days_back: Optional[int] = None
    store_names: List[str] = Field(default_factory=list)
    required_signals: List[str] = Field(default_factory=list)

class KPIResponse(BaseModel):
    kpi: str
    plan: float
    actual: float
    achievement_percent: Optional[float] = None
