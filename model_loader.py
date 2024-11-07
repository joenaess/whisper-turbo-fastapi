import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def load_model(model_id="openai/whisper-large-v3-turbo"):
    """
    Downloads and loads the Whisper model and processor.

    Args:
        model_id (str): The Hugging Face model ID for the Whisper model.

    Returns:
        tuple: A tuple containing the loaded model, processor, and pipeline.
    """
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Download and load the model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        # Download and load the processor
        processor = AutoProcessor.from_pretrained(model_id)

        # Create the pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True
        )

        return model, processor, pipe

    except Exception as e:
        print(f"Error loading model: {e}")
        raise
