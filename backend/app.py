from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile
import os
import uuid
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and processor
logger.info("Loading MusicGen model...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
logger.info("Model loaded successfully.")

# Constants
STATIC_DIR = os.path.join(os.getcwd(), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Request schema
class PromptRequest(BaseModel):
    prompt: str
    duration: int = 10  # seconds

@app.post("/generate")
async def generate_audio(request_data: PromptRequest):
    prompt = request_data.prompt.strip()
    duration = request_data.duration

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if duration < 1 or duration > 30:
        raise HTTPException(status_code=400, detail="Duration must be between 1 and 30 seconds.")

    logger.info(f"Generating audio for prompt: '{prompt}', duration: {duration}s")

    # Estimate tokens (approx. 50 tokens per second)
    max_tokens = duration * 50

    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    audio_values = model.generate(**inputs, max_new_tokens=max_tokens)

    # Get sampling rate from model config
    sampling_rate = model.config.audio_encoder.sampling_rate

    # Convert to numpy and save using scipy
    audio_np = audio_values[0, 0].cpu().numpy()
    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(STATIC_DIR, filename)
    scipy.io.wavfile.write(filepath, rate=sampling_rate, data=audio_np)
    logger.info(f"Audio saved: {filepath}")

    return {
        "message": "Audio generated successfully.",
        "audio_url": f"/static/{filename}"
    }
