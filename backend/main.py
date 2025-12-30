from fastapi import FastAPI
from upload.routes import router as upload_router

app = FastAPI(title="FairData")

app.include_router(upload_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "FairData backend running"}

