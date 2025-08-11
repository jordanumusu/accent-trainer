import io, subprocess, json
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from phonemizer import phonemize

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()

# ---- utils ----

def ffmpeg_to_wav16k_mono(data: bytes) -> bytes:
    # decode any (webm/opus etc.) -> 16kHz mono wav
    p = subprocess.run(
        ["ffmpeg","-hide_banner","-loglevel","error","-i","pipe:0",
         "-f","wav","-ac","1","-ar","16000","pipe:1"],
        input=data, stdout=subprocess.PIPE, check=True
    )
    return p.stdout

def espeak_lang(lang: str) -> str:
    # map ISO to espeak voice
    return {"fr":"fr", "en":"en-us"}.get(lang, lang)

def phonemize_target(text: str, lang: str) -> str:
    # espeak backend phoneme string (no stress marks)
    return phonemize(text, language=espeak_lang(lang), backend="espeak",
                     strip=True, preserve_punctuation=False, with_stress=False)

def argmax_decode(logits: torch.Tensor, vocab) -> str:
    ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    id2tok = {v:k for k,v in vocab.items()}
    out = []
    prev = None
    blank = processor.tokenizer.pad_token_id
    for i in ids:
        if i == blank or i == prev:
            prev = i
            continue
        tok = id2tok.get(i, "")
        out.append(tok)
        prev = i
    s = "".join(out)
    return s.replace("|","").strip()

def align(a: str, b: str):
    # character-level DP alignment (quick & dirty for demo)
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    bt = [[None]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i]==b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
                bt[i+1][j+1] = "D"
            else:
                if dp[i][j+1] >= dp[i+1][j]:
                    dp[i+1][j+1] = dp[i][j+1]
                    bt[i+1][j+1] = "U"   # delete from a
                else:
                    dp[i+1][j+1] = dp[i+1][j]
                    bt[i+1][j+1] = "L"   # insert into a
    i, j = m, n
    pairs = []
    while i>0 or j>0:
        if i>0 and j>0 and bt[i][j]=="D":
            pairs.append((a[i-1], b[j-1]))
            i-=1; j-=1
        elif j>0 and (i==0 or bt[i][j]=="L"):
            pairs.append(("-", b[j-1])); j-=1
        else:
            pairs.append((a[i-1], "-")); i-=1
    pairs.reverse()
    return pairs

def per_phone_scores(target: str, hyp: str):
    pairs = align(target, hyp)
    # score 1 for exact matches, 0 otherwise (toy). Extend later.
    scores = []
    for t, h in pairs:
        if t == "-":  # insertion
            s = 0.0
        elif h == "-":  # deletion
            s = 0.0
        else:
            s = 1.0 if t==h else 0.0
        scores.append({"target": t, "hyp": h, "score": s})
    # aggregate average over real target phones
    real = [p for p in scores if p["target"]!="-"]
    avg = sum(p["score"] for p in real) / max(1,len(real))
    return avg, scores

# minimal confusion tips (extend in web if you want)
TIPS = {
    ("ʁ","r"): "French /ʁ/ is uvular; relax tongue tip, constrict near the uvula.",
    ("y","u"): "French /y/ = rounded /i/; say /i/ then round lips tightly.",
    ("θ","s"): "For /θ/, place tongue lightly between teeth and blow.",
    ("ð","z"): "For /ð/, tongue between teeth, keep voicing.",
    ("ɪ","i"): "Shorter, laxer /ɪ/; don’t tense like /i/.",
    ("æ","e"): "Open jaw more for /æ/; tongue front and low.",
}

def pick_tip(target: str, hyp: str) -> str:
    for (t,h), tip in TIPS.items():
        if t in target and h in hyp:
            return tip
    return "Slow down. Hold the vowel steady, then shape the consonant cleanly."

# ---- api ----

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/eval")
async def eval(audio: UploadFile, text: str = Form(...), lang: str = Form(...)):
    raw = await audio.read()
    wav = ffmpeg_to_wav16k_mono(raw)
    y, sr = sf.read(io.BytesIO(wav), dtype="float32")
    if y.ndim > 1:
        y = y[:,0]
    with torch.no_grad():
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        logits = model(**inputs).logits
    hyp = argmax_decode(logits, processor.tokenizer.get_vocab())
    tgt = phonemize_target(text, lang)
    avg, phones = per_phone_scores(tgt, hyp)
    tip = pick_tip(tgt, hyp)
    return {
        "avg": avg,                 # 0..1
        "targetIPA": tgt,           # actually espeak phones; close enough for demo
        "hypothesisIPA": hyp,
        "perPhone": phones,         # [{target,hyp,score}]
        "tip": tip
    }
