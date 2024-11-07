import tempfile

from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")

with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
    temp_audio_path = temp_audio.name
    with open("audio.mp3", "rb") as f:  # Replace "audio.mp3" 
        temp_audio.write(f.read())

result = pipe(temp_audio_path)
print(result)