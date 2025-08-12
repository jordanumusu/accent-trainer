FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# cache model weights on build (optional)
RUN python - <<'PY'
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
mid = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
Wav2Vec2Processor.from_pretrained(mid)
Wav2Vec2ForCTC.from_pretrained(mid)
PY

COPY main.py .
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000","--log-level","info"]
