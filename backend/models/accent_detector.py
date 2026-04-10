import torch
import torch.nn as nn
import numpy as np
import librosa

# ── Model Definition ────────────────────────────────────
class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * freq)
        x, (hidden, _) = self.lstm(x)
        x = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = self.classifier(x)
        return x


ACCENT_LABELS = {
    0: 'australian',
    1: 'american',
    2: 'indian',
    3: 'british',
    4: 'canadian'
}

# ── Load Model ──────────────────────────────────────────
def load_accent_model(model_path: str, device: str = 'cpu'):
    model = CNNBiLSTM(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ── Audio → Mel Spectrogram ─────────────────────────────
def audio_to_mel(audio_path: str):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)

    # Fix to exactly 4 seconds
    target_samples = 16000 * 4
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=128,
        n_fft=1024,  # fixed
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db  # shape will be (128, 126) naturally now


# ── Inference ───────────────────────────────────────────
def predict_accent(audio_path: str, model, device: str = 'cpu'):
    mel = audio_to_mel(audio_path)

    # (1, 1, 128, 126)
    tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_idx = int(np.argmax(probs))
    
    return {
        'accent':     ACCENT_LABELS[predicted_idx],
        'confidence': round(float(probs[predicted_idx]) * 100, 2),
        'all_scores': {ACCENT_LABELS[i]: round(float(p) * 100, 2) 
                      for i, p in enumerate(probs)}
    }