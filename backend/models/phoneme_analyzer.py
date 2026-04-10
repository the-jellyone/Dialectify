import os
import json
import librosa
import soundfile as sf
import tempfile
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
from allosaurus.app import read_recognizer

# ── Load reference data ─────────────────────────────────
def load_phoneme_reference(summary_path: str):
    with open(summary_path, 'r') as f:
        return json.load(f)

# ── Load allosaurus once ────────────────────────────────
def load_phoneme_model():
    print("Loading allosaurus...")
    model = read_recognizer()
    print("✅ Allosaurus loaded")
    return model

# ── Extract phonemes from audio ─────────────────────────
def extract_phonemes(audio_path: str, model, lang: str = 'eng'):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, y, 16000, subtype='PCM_16')
            tmp_path = tmp.name

        result = model.recognize(tmp_path, lang_id=lang)
        os.remove(tmp_path)

        phonemes = result.strip().split()
        return phonemes

    except Exception as e:
        print(f"⚠️ Phoneme extraction error: {e}")
        return []

# ── Score user phonemes vs target accent ────────────────
def analyze_phonemes(audio_path: str, target_accent: str, model, reference: dict):
    """
    Takes user audio, extracts phonemes, scores against target accent.
    Returns score + substitution flags.
    """
    # Step 1: Extract phonemes
    phonemes = extract_phonemes(audio_path, model)

    if len(phonemes) < 10:
        return {
            'error': 'Audio too short or no speech detected',
            'phoneme_score': 0,
            'substitutions': [],
            'raw_phonemes': []
        }

    # Step 2: Build user frequency vector
    total = len(phonemes)
    counter = Counter(phonemes)
    user_freq = {p: (c / total) * 1000 for p, c in counter.items()}

    # Step 3: Load target accent reference
    target_ref = reference[target_accent]
    american_ref = reference['american']

    # Step 4: JS divergence vs target accent
    # Build vocab from reference
    target_freq = {
        'θ': target_ref['th_rate'],
        'ð': target_ref['voiced_th_rate'],
        'ɹ': target_ref['r_rate'],
        'w': target_ref['w_rate'],
        'v': target_ref['v_rate'],
        'd': target_ref['d_rate'],
    }

    key_phones = list(target_freq.keys())

    user_vec   = np.array([user_freq.get(p, 0) for p in key_phones], dtype=float)
    target_vec = np.array([target_freq.get(p, 0) for p in key_phones], dtype=float)

    # Normalize to probability
    user_vec   = user_vec / (user_vec.sum() + 1e-9)
    target_vec = target_vec / (target_vec.sum() + 1e-9)

    js_div = jensenshannon(user_vec, target_vec)
    phoneme_score = round(max(0, 1 - js_div) * 100, 1)

    # Step 5: Detect substitutions vs TARGET accent
    substitutions = []

    # TH → D/T
    if (target_ref['th_rate'] > 0 and 
    user_freq.get('θ', 0) < target_ref['th_rate'] * 0.5):
        substitutions.append({
        'type': 'TH→D/T',
        'message': 'You are replacing the TH sound with D or T. Practice words like "think", "three", "that".'
    })

    # R dropping
    if (target_ref['r_rate'] > 0 and 
    user_freq.get('ɹ', 0) < target_ref['r_rate'] * 0.5):
        substitutions.append({
        'type': 'R-dropping',
        'message': 'You are dropping the R sound. Practice words like "car", "bird", "here".'
    })

    # W → V
    if (target_ref['w_rate'] > 0 and 
    user_freq.get('w', 0) < target_ref['w_rate'] * 0.5):
        substitutions.append({
        'type': 'W→V',
        'message': 'You are replacing W with V. Practice words like "wine", "water", "we".'
    })

    return {
        'phoneme_score':  phoneme_score,
        'js_divergence':  round(float(js_div), 4),
        'substitutions':  substitutions,
        'raw_phonemes':   phonemes[:30]  # first 30 for debugging
    }