"""
AccentIQ Scoring Engine with Feedback
3-component scoring + detailed feedback from Phase 6/7 research
"""

# Research-based substitution patterns from Phase 6
ACCENT_PATTERNS = {
    'indian': {
        'substitutions': ['TH→D/T', 'W→V', 'Retroflex-R'],
        'common_issues': 'TH sounds, W/V distinction, retroflex consonants'
    },
    'british': {
        'substitutions': ['R-dropping', 'T-glottalization', 'BATH-vowel'],
        'common_issues': 'Non-rhotic R, glottal stops, vowel shifts'
    },
    'australian': {
        'substitutions': ['AI→OI', 'EE→I', 'Syllable-final-L'],
        'common_issues': 'Diphthong shifts, vowel raising, L-vocalization'
    },
    'canadian': {
        'substitutions': ['OU-raising', 'Cot-caught-merger', 'EH-raising'],
        'common_issues': 'Canadian raising, vowel mergers, eh particle'
    },
    'american': {
        'substitutions': [],
        'common_issues': 'Target accent - focus on regional consistency'
    }
}

# Practice phrases for common issues
PRACTICE_PHRASES = {
    'TH→D/T': ['Think, thank, thought, through', 'The three thieves threw things with them'],
    'W→V': ['We will walk westward', 'Wine is not vine - wet is not vet'],
    'R-dropping': ['Her car is far from here', 'The butter on the counter'],
    'T-glottalization': ['Better butter, water bottle', 'Little bit of butter'],
    'Prosody': ['I REALLY want to go (stress REALLY)', "That's AMAZING (pitch jump on AMAZING)"],
    'Stress': ['PHOtograph, phoTOgraphy, photoGRAPHic', 'comPUter, PROgram, inforMAtion']
}

def generate_feedback(target_accent, phoneme_score, prosody_score, accent_confidence, 
                     predicted_accent, substitutions=None):
    """Generate detailed feedback based on scores and accent patterns."""
    phoneme_score     = max(0, min(100, phoneme_score))
    prosody_score     = max(0, min(100, prosody_score))
    accent_confidence = max(0, min(100, accent_confidence))

    feedback = {'issues': [], 'quick_wins': [], 'exercises': []}
    target = target_accent.lower()
    predicted = predicted_accent.lower()
    
    # Get accent-specific patterns
    patterns = ACCENT_PATTERNS.get(target, {})
    
    # Phoneme issues
    if phoneme_score < 75:
        subs = substitutions or patterns.get('substitutions', [])
        for pattern in subs:
            if pattern in PRACTICE_PHRASES:
                feedback['issues'].append({
                    'severity': 'high' if phoneme_score < 60 else 'medium',
                    'category': 'phoneme',
                    'pattern': pattern,
                    'description': f'{pattern} substitution detected',
                    'practice_phrases': PRACTICE_PHRASES[pattern]
                })
    
    # Prosody issues
    if prosody_score < 70:
        feedback['issues'].append({
            'severity': 'medium',
            'category': 'prosody',
            'description': 'Narrow pitch range - speech sounds monotone',
            'detail': f'{target.capitalize()} English uses wider pitch variation. Exaggerate word stress.',
            'practice_phrases': PRACTICE_PHRASES['Prosody']
        })
    
    # Accent mismatch guidance
    if predicted != target:
        feedback['issues'].append({
            'severity': 'high',
            'category': 'accent',
            'description': f'Currently sounds {predicted.capitalize()}, targeting {target.capitalize()}',
            'detail': f"Focus on {target.capitalize()} patterns: {ACCENT_PATTERNS.get(target, {}).get('common_issues', 'consult accent guide')}"
        })
    
    # Quick wins (always included)
    feedback['quick_wins'] = [
        'Record yourself reading the same sentence daily and compare to a native speaker',
        'Use shadowing: listen to 10 seconds of speech, then immediately repeat exactly',
        'Focus on ONE substitution pattern at a time - master it before moving on'
    ]
    
    return feedback


def calculate_combined_score(phoneme_score, prosody_score, accent_confidence, 
                             target_accent, predicted_accent, substitutions=None):
    """
    Calculate weighted combined score for accent correction app.
    
    Args:
        phoneme_score: 0-100, from Phoneme Analyzer (Phase 2)
        prosody_score: 0-100, from Prosody Analyzer (Phase 3)
        accent_confidence: 0-100, from Accent Classifier (Phase 1)
        target_accent: string, one of: american, british, indian, australian, canadian
        predicted_accent: string, one of: american, british, indian, australian, canadian
        substitutions: optional list of detected substitution patterns (e.g., ['TH→D/T', 'W→V'])
    
    Returns:
        dict with combined_score, grade, breakdown, summary, feedback
    """
    # Weights: phoneme 40%, prosody 30%, accent confidence 30%
    combined_score = round(
        0.40 * phoneme_score + 
        0.30 * prosody_score + 
        0.30 * accent_confidence,
        2
    )
    
    # Grade based on score
    if combined_score >= 90:
        grade = 'A'
    elif combined_score >= 80:
        grade = 'B'
    elif combined_score >= 70:
        grade = 'C'
    elif combined_score >= 60:
        grade = 'D'
    else:
        grade = 'F'
    
    # Check if predicted matches target
    accent_match = predicted_accent.lower() == target_accent.lower()
    
    # Generate summary message
    if accent_match and combined_score >= 85:
        summary = f"Excellent! Your {target_accent.capitalize()} accent is highly accurate."
    elif accent_match and combined_score >= 70:
        summary = f"Good progress on {target_accent.capitalize()} accent. Focus on fine-tuning."
    elif accent_match:
        summary = f"Detected {target_accent.capitalize()} accent, but needs improvement."
    else:
        summary = f"Currently sounds {predicted_accent.capitalize()}, working toward {target_accent.capitalize()}."
    
    # Generate detailed feedback
    feedback = generate_feedback(
        target_accent, phoneme_score, prosody_score, 
        accent_confidence, predicted_accent, substitutions
    )
    
    return {
        'combined_score': combined_score,
        'grade': grade,
        'breakdown': {
            'phoneme_score': phoneme_score,
            'prosody_score': prosody_score,
            'accent_confidence': accent_confidence
        },
        'summary': summary,
        'feedback': feedback
    }