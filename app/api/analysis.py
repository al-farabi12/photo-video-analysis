from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.utils.analysis_utils import save_upload_file, analyze_content, get_analysis_results, delete_analysis

router = APIRouter()

@router.post("/analysis")
async def create_analysis(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    request_id = await save_upload_file(file)
    background_tasks.add_task(analyze_content, request_id)  # Asynchronous processing
    return {"request_id": request_id}

@router.get("/analysis/{request_id}")
async def get_analysis(request_id: str):
    result = get_analysis_results(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result

@router.delete("/analysis/{request_id}")
async def delete_analysis_data(request_id: str):
    success = delete_analysis(request_id)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"status": "deleted"}
