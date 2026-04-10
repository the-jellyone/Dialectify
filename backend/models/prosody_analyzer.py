import numpy as np
import json
import parselmouth
from scipy.signal import find_peaks
from scipy.spatial.distance import mahalanobis

FEATURES = ['mean_f0', 'f0_std', 'speech_rate', 'pause_freq', 'mean_pause_duration', 'pvi']

FEEDBACK_MESSAGES = {
    'mean_f0': {
        'high': "Your pitch is higher than typical for this accent. Try speaking in a slightly lower tone.",
        'low':  "Your pitch is lower than typical for this accent. Try raising your tone slightly."
    },
    'f0_std': {
        'high': "Your pitch varies too much. Try to speak more evenly.",
        'low':  "Your pitch is too flat. Try adding more natural rise and fall to your speech."
    },
    'speech_rate': {
        'high': "You are speaking too fast. Slow down slightly.",
        'low':  "You are speaking too slowly. Try to pick up your pace slightly."
    },
    'pause_freq': {
        'high': "You are pausing too often. Try to speak in longer continuous phrases.",
        'low':  "You are not pausing enough. Add more natural breaks between phrases."
    },
    'mean_pause_duration': {
        'high': "Your pauses are too long. Keep pauses short and natural.",
        'low':  "Your pauses are too short. Let your pauses breathe a little more."
    },
    'pvi': {
        'high': "Your rhythm is too varied. Try to even out your stress patterns.",
        'low':  "Your rhythm is too uniform. Stress important words more and reduce others."
    }
}

# ── Load fingerprints ────────────────────────────────────
def load_prosody_fingerprints(fingerprints_path: str):
    with open(fingerprints_path, 'r') as f:
        return json.load(f)

# ── Extract features ─────────────────────────────────────
def extract_all_prosody_features(audio_path: str):
    try:
        snd = parselmouth.Sound(audio_path)
        duration = snd.duration

        if duration < 0.5:
            return None

        pitch = snd.to_pitch()
        f0_values = pitch.selected_array['frequency']
        voiced_flag = f0_values > 0
        f0_voiced = f0_values[voiced_flag]

        if len(f0_voiced) < 10:
            return None

        mean_f0 = round(float(np.mean(f0_voiced)), 3)
        f0_std  = round(float(np.std(f0_voiced)), 3)

        intensity = snd.to_intensity()
        intensity_values = intensity.values[0]
        time_step = intensity.time_step
        mean_intensity = np.mean(intensity_values)

        peaks, _ = find_peaks(
            intensity_values,
            height=mean_intensity - 10,
            distance=10
        )
        speech_rate = round(len(peaks) / duration, 3)

        silence_threshold = mean_intensity - 25
        is_silent = intensity_values < silence_threshold
        transitions = np.diff(is_silent.astype(int))
        pause_count = np.sum(transitions == 1)
        pause_freq  = round(float(pause_count / duration), 3)

        pause_durations = []
        in_pause = False
        pause_start = 0
        for i, silent in enumerate(is_silent):
            if silent and not in_pause:
                pause_start = i
                in_pause = True
            elif not silent and in_pause:
                pause_durations.append((i - pause_start) * time_step)
                in_pause = False

        mean_pause_duration = round(
            float(np.mean(pause_durations)) if pause_durations else 0.0, 3
        )

        voiced_segments = []
        in_voiced = False
        start = 0
        for i, v in enumerate(voiced_flag):
            if v and not in_voiced:
                start = i
                in_voiced = True
            elif not v and in_voiced:
                voiced_segments.append(i - start)
                in_voiced = False

        if len(voiced_segments) < 2:
            return None

        durations = np.array(voiced_segments, dtype=float)
        pvi = 0.0
        for k in range(len(durations) - 1):
            denom = (durations[k] + durations[k+1]) / 2
            if denom > 0:
                pvi += abs(durations[k] - durations[k+1]) / denom
        pvi = round((pvi / (len(durations) - 1)) * 100, 3)

        return {
            'mean_f0':             mean_f0,
            'f0_std':              f0_std,
            'speech_rate':         speech_rate,
            'pause_freq':          pause_freq,
            'mean_pause_duration': mean_pause_duration,
            'pvi':                 pvi
        }

    except Exception:
        return None

# ── Score ────────────────────────────────────────────────
def analyze_prosody(audio_path: str, target_accent: str, fingerprints: dict):
    """
    Takes audio path + target accent.
    Returns prosody score + actionable feedback.
    """
    user_features = extract_all_prosody_features(audio_path)

    if user_features is None:
        return {
            'error': 'Audio too short or no speech detected',
            'prosody_score': 0,
            'feedback': []
        }

    fp = fingerprints[target_accent]
    mean_vec = np.array(fp['mean'])
    cov_inv  = np.array(fp['cov_inv'])
    user_vec = np.array([user_features[f] for f in FEATURES])

    distance = mahalanobis(user_vec, mean_vec, cov_inv)
    score    = round(100 * np.exp(-distance / 10), 2)

    # Per feature feedback
    deviations = []
    for i, feature in enumerate(FEATURES):
        diff = user_vec[i] - mean_vec[i]
        std  = np.sqrt(fp['cov'][i][i])
        z    = diff / std

        deviations.append({
            'feature':   feature,
            'abs_z':     abs(z),
            'direction': 'high' if z > 0 else 'low',
            'severity':  'major' if abs(z) > 2.5 else 'minor' if abs(z) > 1.5 else 'tip'
        })

    deviations.sort(key=lambda x: x['abs_z'], reverse=True)
    flagged = [d for d in deviations if d['severity'] in ['major', 'minor']]
    if len(flagged) < 2:
        flagged = deviations[:2]

    feedback = []
    for d in flagged:
        feedback.append({
            'feature':  d['feature'],
            'severity': d['severity'],
            'message':  FEEDBACK_MESSAGES[d['feature']][d['direction']]
        })

    return {
        'prosody_score': score,
        'distance':      round(distance, 4),
        'feedback':      feedback
    }