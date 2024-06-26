from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    request_id: str
    status: str
    results: dict = None
