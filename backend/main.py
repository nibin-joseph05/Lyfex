from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

frontend_ip = os.getenv("FRONTEND_IP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_ip],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is working"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Placeholder logic — replace with actual analysis
    return {
        "heartRate": "78 bpm",
        "respiratoryRate": "17 breaths/min",
        "stressLevel": "Moderate",
        "emotion": "Happy",
        "facialAsymmetry": "Low",
        "tremor": "None",
        "skinAnalysis": "Healthy",
        "fatigue": "Low"
    }
