from fastapi import FastAPI
from .api import analysis
from .core.config import settings

app = FastAPI(title="Visual Content Analysis API")


# Include the router for the analysis endpoints
app.include_router(analysis.router, prefix="/api", tags=["analysis"])

# Create the upload folder if it doesn't exist
import os
if not os.path.exists(settings.UPLOAD_FOLDER):
    os.makedirs(settings.UPLOAD_FOLDER)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Visual Content Analysis API"}
