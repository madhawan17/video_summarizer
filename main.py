from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv
import mimetypes

# Load .env
load_dotenv()

# Get API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise Exception("❌ GEMINI_API_KEY not found in .env or environment variables.")

# Configure Gemini client (v1 API)
genai.configure(api_key=API_KEY)

# FastAPI app
app = FastAPI()

# ----------------------------------
# FIX 1: Enable CORS
# ----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------
# FIX 2: Allow large uploads
# ----------------------------------
class LargeUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request._receive = request.receive
        return await call_next(request)

app.add_middleware(LargeUploadMiddleware)

# ----------------------------------
# HOME ROUTE (Serve index.html)
# ----------------------------------
@app.get("/")
def home():
    return HTMLResponse(open("index.html", encoding="utf-8").read())

# ----------------------------------
# TRANSCRIPTION (MP4 / MP3 / WAV)
# ----------------------------------
import time

def transcribe_audio(file_path):
    print("📤 Uploading file to Gemini...")

    # Upload file to Gemini (required for video/audio)
    uploaded_file = genai.upload_file(file_path)
    
    print(f"✔ File uploaded to Gemini: {uploaded_file.name}")
    
    # Wait for file processing to complete
    while uploaded_file.state.name == "PROCESSING":
        print("⏳ Waiting for file processing...")
        time.sleep(5)
        uploaded_file = genai.get_file(uploaded_file.name)

    if uploaded_file.state.name != "ACTIVE":
        raise Exception(f"❌ File processing failed: {uploaded_file.state.name}")

    print("✔ File is ACTIVE and ready for processing")

    # Initialize model
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    # Request transcription
    response = model.generate_content(
        [
            "Transcribe this lecture into clean readable text.",
            uploaded_file
        ]
    )

    print("✔ Transcription complete")
    return response.text

# ----------------------------------
# SUMMARIZATION
# ----------------------------------
def summarize_text(text):
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    prompt = f"""
    Summarize this lecture clearly and in depth.

    Include:
    - Detailed explanation
    - Key points
    - Important concepts
    - Definitions
    - Chapter-wise breakdown
    - Final overview like a teacher

    LECTURE TEXT:
    {text}
    """

    response = model.generate_content(prompt)
    return response.text

# ----------------------------------
# MAIN UPLOAD ROUTE
# ----------------------------------
import shutil

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    print(f"📁 Received file: {file.filename}")

    # Save file to temp folder (Streaming to avoid RAM issues)
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Step 1: Transcription
        transcript = transcribe_audio(file_path)

        # Step 2: Summary
        summary = summarize_text(transcript)

        return {
            "transcript": transcript,
            "summary": summary
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"summary": f"Error processing file: {str(e)}"}
    finally:
        # Optional: Cleanup local file
        if os.path.exists(file_path):
            os.remove(file_path)
