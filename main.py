import logging
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from model_loader import load_model

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model, processor, and pipeline
model, processor, pipe = load_model()

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        logger.debug(f"Saving file to: {temp_audio_path}")

        with open(temp_audio_path, "wb") as f:  # Open the file in write binary mode
            f.write(file.file.read())

        # Transcribe the audio file
        transcription = pipe(temp_audio_path)

        # Check if transcription is valid
        if not transcription or "text" not in transcription:
            raise ValueError("Transcription result is invalid")

        text = transcription["text"]
        logger.info(f"Transcription: {text}")
        return {"text": text}

    except Exception as e:
        logger.exception(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")

    finally:
        # Ensure the temporary file is deleted
        os.unlink(temp_audio_path)
