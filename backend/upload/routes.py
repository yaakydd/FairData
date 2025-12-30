from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
