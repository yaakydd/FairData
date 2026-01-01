# When a file comes in:
#     read it
#     turn it into a table
#     save it in a notebook
#     return an ID and timestamp
# Imagine a physical office:

# User walks in → hands you a document →
# You check if you can read it →
# You put it in a folder →
# You write a label →
# You give the user a receipt
# Imagine a physical office:

# User walks in → hands you a document →
# You check if you can read it →
# You put it in a folder →
# You write a label →
# You give the user a receipt
# =========================
# upload.py
# =========================
# Purpose: Handle dataset uploads (CSV, Excel, JSON, TSV, Parquet)
# Author: You
# Notes:
# - Stores datasets in memory (for simplicity)
# - Returns dataset info and a preview
# - Validates file type and checks for empty files
# - Each dataset gets a unique ID
# =========================

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import pandas as pd
import io
import uuid
from datetime import datetime

# -------------------------
# Create a FastAPI router
# -------------------------
# We use APIRouter so we can separate upload endpoints from other endpoints
router = APIRouter()

# -------------------------
# In-memory dataset storage
# -------------------------
# Key = dataset_id (UUID), Value = info dict
# In production, you would use a database
uploaded_datasets: Dict[str, Any] = {}

# -------------------------
# POST endpoint: Upload a file
# -------------------------
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a dataset file (CSV, Excel, JSON, TSV, Parquet)
    
    Steps:
    1. Check if file type is allowed
    2. Read the file into a pandas DataFrame
    3. Check if DataFrame is empty
    4. Generate a unique ID for this dataset
    5. Store the dataset in memory
    6. Return dataset info + first 5 rows as preview
    """
    
    # -------------------------
    # Allowed file types
    # -------------------------
    allowed_types = [
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/json",
        "application/octet-stream"  # for Parquet
    ]
    
    # If file type is invalid, reject immediately
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "explanation": (
                    "I can only read CSV, Excel, JSON, TSV, and Parquet files. "
                    "Please upload a supported format."
                ),
                "accepted_formats": [".csv", ".xlsx", ".xls", ".json", ".tsv", ".parquet"]
            }
        )

    try:
        # -------------------------
        # Read file contents
        # -------------------------
        # file.read() returns bytes
        contents = await file.read()

        # Convert filename to lowercase to handle CSV vs csv, etc.
        filename = (file.filename or "").lower()

        # -------------------------
        # Parse file into pandas DataFrame
        # -------------------------
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".tsv"):
            df = pd.read_csv(io.BytesIO(contents), sep="\t")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents))
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported file format",
                    "explanation": "Cannot read this file type"
                }
            )

        # -------------------------
        # Check if dataset is empty
        # -------------------------
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Empty file",
                    "explanation": "The uploaded file has no data."
                }
            )

        # -------------------------
        # Generate a unique ID
        # -------------------------
        dataset_id = str(uuid.uuid4())

        # -------------------------
        # Store dataset in memory
        # -------------------------
        uploaded_datasets[dataset_id] = {
            "dataframe": df,
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "size_mb": len(contents) / (1024 * 1024)  # Convert bytes to MB
        }

        # -------------------------
        # Return dataset info + preview
        # -------------------------
        return {
            "success": True,
            "message": "File uploaded successfully!",
            "dataset_id": dataset_id,
            "info": {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "size": f"{len(contents) / (1024 * 1024):.2f} MB",
                "preview": df.head(5).to_dict('records')  # First 5 rows
            },
            "explanation": (
                f"Received '{file.filename}' with {len(df)} rows and {len(df.columns)} columns."
            )
        }

    # -------------------------
    # Handle errors
    # -------------------------
    except pd.errors.EmptyDataError:
        # pandas cannot parse the file
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Cannot parse file",
                "explanation": "The file might be corrupted or in wrong format."
            }
        )
    except Exception as e:
        # Any other unexpected error
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Upload failed",
                "explanation": str(e)
            }
        )
  