import logging
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

app = FastAPI()


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            f.write(file.file.read())
            logger.info(f"Received audio file: {file.filename}")

        # Transcribe the audio file
        text = pipe(temp_audio_path)["text"]

        logger.info(f"Received audio file: {file.filename}")
        return {"text": text}

    except Exception as e:
        logger.exception(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")
    
    finally:
        # Ensure the temporary file is deleted
        import os
        os.unlink(temp_audio_path) 