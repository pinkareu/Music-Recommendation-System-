from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil
import os
import uvicorn
from backend.recommend import recommendation  # ML code
from fastapi.middleware.cors import CORSMiddleware
import imageio_ffmpeg as ffmpeg
import subprocess
import os

app = FastAPI()

# Enable CORS for frontend running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recommend")
async def recommendations(file: UploadFile = File(...)):
    # Create directory if it doesn't exist
    save_dir = f"frontend/build/static/audio"
    os.makedirs(save_dir, exist_ok=True)

    # Delete all existing files in the directory
    for f in os.listdir(save_dir):
        file_path = os.path.join(save_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save uploaded file to a temporary path
    temp_path = os.path.join(save_dir, "temp_audio")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Path for final MP3
    mp3_path = os.path.join(save_dir, "input_song.mp3")

    # Use imageio-ffmpeg to convert to mp3
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [ffmpeg_path, "-y", "-i", temp_path, mp3_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": "FFmpeg conversion failed", "details": e.stderr.decode()}, status_code=500)

    # Run your recommendation function
    result = recommendation()
    return JSONResponse(result)

# Serve React build at root
app.mount("/", StaticFiles(directory=f"frontend/build/static", html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default 8000 for local dev
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
