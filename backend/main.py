"""
AccentIQ - FastAPI Backend
Main entry point. Wires all 3 analyzers + scoring engine.
"""

import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from models.accent_detector import load_accent_model, predict_accent
from models.phoneme_analyzer import load_phoneme_model, load_phoneme_reference, analyze_phonemes
from models.prosody_analyzer import load_prosody_fingerprints, analyze_prosody
from scoring.engine import calculate_combined_score

# ── App ──────────────────────────────────────────────────
app = FastAPI(title="AccentIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load all models once at startup ─────────────────────
print("Loading models...")

accent_model      = load_accent_model("weights/classifier.pt")
phoneme_model     = load_phoneme_model()
phoneme_reference = load_phoneme_reference("data/phase6_summary.json")
prosody_prints    = load_prosody_fingerprints("data/prosody_fingerprints.json")

print("✅ All models loaded")

# ── Main endpoint ────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    target_accent: str = Form(...)
):
    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        # Phase 1: Accent Classification
        accent_result = predict_accent(tmp_path, accent_model)

        # Phase 2: Phoneme Analysis
        phoneme_result = analyze_phonemes(
            tmp_path, target_accent, phoneme_model, phoneme_reference
        )

        # Phase 3: Prosody Analysis
        prosody_result = analyze_prosody(
            tmp_path, target_accent, prosody_prints
        )

        # Combined Score
        final = calculate_combined_score(
            phoneme_score     = phoneme_result.get('phoneme_score', 0),
            prosody_score     = prosody_result.get('prosody_score', 0),
            accent_confidence = accent_result.get('confidence', 0),
            target_accent     = target_accent,
            predicted_accent  = accent_result.get('accent', 'unknown'),
            substitutions     = [s['type'] for s in phoneme_result.get('substitutions', [])]
        )

        return {
            'status':          'success',
            'target_accent':   target_accent,
            'accent_result':   accent_result,
            'phoneme_result':  phoneme_result,
            'prosody_result':  prosody_result,
            'final_score':     final
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e)}

    finally:
        os.remove(tmp_path)

# ── Health check ─────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "AccentIQ API is running"}