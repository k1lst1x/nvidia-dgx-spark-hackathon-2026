import os
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI(title="GuideLens API",version="0.1.0")
app.add_middleware(CORSMiddleware,allow_origins=os.getenv("CORS_ORIGINS","http://localhost:8081").split(","),allow_methods=["*"],allow_headers=["*"])
@app.get("/health")
def health(): return {"status":"ok","mode":"mock" if os.getenv("MOCK_INFERENCE","true").lower()=="true" else "dgx","time":datetime.now(timezone.utc).isoformat()}
@app.post("/v1/describe")
async def describe(image:UploadFile=File(...)):
    _=await image.read(1024)
    return {"summary":"Clear path ahead","speech":"Door 13 feet ahead. Chair to your right.","objects":[{"label":"door","confidence":.97},{"label":"chair","confidence":.93}],"mock":True}
