import { useState, useEffect, useCallback, useRef } from "react";
import {
  Upload, Mic, RefreshCw, ChevronRight, Activity, Brain,
  AlertCircle, Clock, Volume2, BookOpen, Zap, CheckCircle,
  TrendingUp, BarChart2, Target, Award
} from "lucide-react";
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  Cell, Tooltip
} from "recharts";

// ── Configuration ──────────────────────────────────────────────────
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const GROQ_API_KEY = import.meta.env.VITE_GROQ_KEY || "";

const FLAGS = { 
  american:"🇺🇸", british:"🇬🇧", indian:"🇮🇳", 
  australian:"🇦🇺", canadian:"🇨🇦" 
};

const ACCENTS = ["american", "british", "indian", "australian", "canadian"];

const SEV = { high:"#f43f5e", medium:"#f59e0b", low:"#10b981" };

const PIPELINE_STEPS = [
  {label:"Audio Feature Extraction", sub:"MFCC, spectral features"},
  {label:"Accent Detection",         sub:"Phase 1: Classifier"},
  {label:"Phoneme Analysis",         sub:"Phase 2: Phoneme scoring"},
  {label:"Prosody Analysis",         sub:"Phase 3: Prosody scoring"},
];

// ── Helpers ────────────────────────────────────────────────────────
function scoreColor(s) {
  return s >= 85 ? "#10b981" : s >= 70 ? "#f59e0b" : "#f43f5e";
}

function ScoreRing({ score, size=96 }) {
  const r=36, circ=2*Math.PI*r;
  const dash = circ - (Math.min(score,100)/100)*circ;
  const col = scoreColor(score);
  return (
    <svg width={size} height={size} viewBox="0 0 96 96">
      <circle cx="48" cy="48" r={r} fill="none" stroke="#1e293b" strokeWidth="9"/>
      <circle cx="48" cy="48" r={r} fill="none" stroke={col} strokeWidth="9"
        strokeDasharray={circ} strokeDashoffset={dash} strokeLinecap="round"
        transform="rotate(-90 48 48)"
        style={{transition:"stroke-dashoffset 1.4s cubic-bezier(.4,0,.2,1)"}}/>
      <text x="48" y="44" textAnchor="middle" fill="white" fontSize="20" fontWeight="bold"
        fontFamily="monospace">{score}</text>
      <text x="48" y="61" textAnchor="middle" fill="#64748b" fontSize="10">/100</text>
    </svg>
  );
}

// ── Main Component ─────────────────────────────────────────────────
export default function AccentIQ() {
  const [view, setView] = useState("upload");
  const [file, setFile] = useState(null);
  const [targetAccent, setTargetAccent] = useState("american");
  const [processing, setProcessing] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [results, setResults] = useState(null);
  const [feedback, setFeedback] = useState("");
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  // ── File upload ────────────────────────────────────────────────
  const handleFileSelect = (e) => {
    const f = e.target.files?.[0];
    if (f && f.type.startsWith("audio/")) {
      setFile(f);
      setError(null);
    } else {
      setError("Please select a valid audio file");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("audio/")) {
      setFile(f);
      setError(null);
    }
  };

  // ── Microphone recording ───────────────────────────────────────
  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        chunksRef.current = [];

        mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
        mediaRecorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const recordedFile = new File([blob], "recording.webm", { type: "audio/webm" });
          setFile(recordedFile);
          stream.getTracks().forEach(t => t.stop());
        };

        mediaRecorder.start();
        setIsRecording(true);
      } catch (err) {
        setError("Microphone access denied");
      }
    }
  };

  // ── Audio Preprocessing ────────────────────────────────
const preprocessAudio = async (inputFile) => {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: 16000
  });

  const arrayBuffer = await inputFile.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Resample to 16kHz mono
  const offlineCtx = new OfflineAudioContext(
    1,                          // mono
    audioBuffer.duration * 16000, // samples at 16kHz
    16000                       // sample rate
  );

  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start();

  const resampled = await offlineCtx.startRendering();

  // Convert to WAV
  const wavBuffer = audioBufferToWav(resampled);
  const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
  return new File([wavBlob], 'processed.wav', { type: 'audio/wav' });
};

// ── AudioBuffer → WAV ──────────────────────────────────
function audioBufferToWav(buffer) {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  const samples = buffer.getChannelData(0);
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const bufferSize = 44 + dataSize;
  const arrayBuffer = new ArrayBuffer(bufferSize);
  const view = new DataView(arrayBuffer);

  // WAV header
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++)
      view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, bufferSize - 8, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  // PCM samples
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

    return arrayBuffer;
  }
  // ── Backend API call ───────────────────────────────────────────
  const analyzeAudio = async () => {
    if (!file) return;

    setError(null);
    setProcessing(true);
    setView("processing");
    setStepIndex(0);

    const formData = new FormData();
    const processedFile = await preprocessAudio(file);
    formData.append("audio", processedFile);
    formData.append("target_accent", targetAccent);

    try {
      // Simulate step progression
      const stepInterval = setInterval(() => {
        setStepIndex(prev => Math.min(prev + 1, PIPELINE_STEPS.length));
      }, 1200);

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      clearInterval(stepInterval);

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      // Transform backend response to UI format
      const transformedResults = {
        accent: data.accent_result?.accent || targetAccent,
        confidence: data.accent_result?.confidence || 0,
        overall: data.final_score?.combined_score || 0,
        grade: data.final_score?.grade || 'F',
        phoneme: data.final_score?.breakdown?.phoneme_score || 0,
        prosody: data.final_score?.breakdown?.prosody_score || 0,
        summary: data.final_score?.summary || '',
        substitutions: data.phoneme_result?.substitutions?.map(s => ({
        pattern: s.type,
        severity: 'medium',
        example: s.message,
        phrases: []
        })) || [],
        exercises: data.prosody_result?.feedback?.map(f => ({
        title: f.message,
        focus: f.feature,
        phrases: []
        })) || [],

        // Prosody metrics (if available from backend)
        pitchMean: data.prosody_metrics?.pitch_mean || 0,
        pitchRange: data.prosody_metrics?.pitch_range || 0,
        wpm: data.prosody_metrics?.speech_rate || 0,
      };

      setResults(transformedResults);
      setStepIndex(PIPELINE_STEPS.length);
      
      setTimeout(() => {
        setProcessing(false);
        setView("results");
        
        // Generate AI coaching feedback
        if (GROQ_API_KEY) {
          generateAIFeedback(transformedResults);
        }
      }, 800);

    } catch (err) {
      clearInterval(stepInterval);
      setError(err.message || "Analysis failed. Check if backend is running.");
      setProcessing(false);
      setView("upload");
    }
  };

const generateAIFeedback = async (resultData) => {
  if (!GROQ_API_KEY) {
    setFeedback("Add VITE_GROQ_KEY to .env for AI coaching feedback");
    return;
  }

  setFeedbackLoading(true);

  const prompt = `You are an accent coach. A speaker is working on their ${targetAccent} accent.

  Current analysis:
  - Predicted accent: ${resultData.accent}
  - Overall score: ${resultData.overall}/100 (Grade ${resultData.grade})
  - Phoneme score: ${resultData.phoneme}/100
  - Prosody score: ${resultData.prosody}/100
  - Key issues: ${resultData.substitutions.map(s => s.pattern).join(", ") || "None detected"}

  Write a brief, encouraging 2-3 sentence coaching message focusing on their next steps. Be specific and actionable.`;

  try {
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${GROQ_API_KEY}`
      },
      body: JSON.stringify({
        model: "llama-3.1-8b-instant",
        max_tokens: 200,
        messages: [{ role: "user", content: prompt }]
      })
    });

    const data = await response.json();
    const text = data.choices?.[0]?.message?.content || "Keep practicing!";
    setFeedback(text);
    } catch (err) {
    setFeedback("AI feedback unavailable. Continue with the practice exercises below!");
    } finally {
    setFeedbackLoading(false);
    }
  };

  // ── Reset ──────────────────────────────────────────────────────
  const reset = () => {
    setView("upload");
    setFile(null);
    setResults(null);
    setFeedback("");
    setError(null);
    setStepIndex(0);
  };

  // ═══════════════════════════════════════════════════════════════
  // VIEWS
  // ═══════════════════════════════════════════════════════════════

  // ── Upload View ────────────────────────────────────────────────
  const UploadView = (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold mb-2">AccentIQ Analysis</h1>
        <p className="text-slate-400 text-sm">
          Upload audio or record your voice for accent analysis
        </p>
      </div>

      {/* Target accent selector */}
      <div>
        <label className="block text-sm text-slate-400 mb-3">Target Accent</label>
        <div className="flex gap-3">
          {ACCENTS.map(acc => (
            <button key={acc}
              onClick={() => setTargetAccent(acc)}
              className={`flex-1 rounded-xl px-4 py-3 border transition-all ${
                targetAccent === acc
                  ? "border-sky-500 bg-sky-500/10"
                  : "border-slate-700 hover:border-slate-600"
              }`}>
              <div className="text-2xl mb-1">{FLAGS[acc]}</div>
              <div className="text-xs capitalize">{acc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* File upload */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className="border-2 border-dashed border-slate-700 rounded-2xl p-12 text-center hover:border-slate-600 transition-colors cursor-pointer"
        onClick={() => fileInputRef.current?.click()}>
        <Upload size={32} className="mx-auto mb-3 text-slate-500"/>
        <p className="text-sm text-slate-400 mb-1">
          Drop audio file or click to browse
        </p>
        <p className="text-xs text-slate-600">
          Supports WAV, MP3, M4A, FLAC
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      {/* Microphone recording */}
      <div className="text-center">
        <button
          onClick={toggleRecording}
          className={`inline-flex items-center gap-2 px-6 py-3 rounded-xl border transition-all ${
            isRecording
              ? "border-red-500 bg-red-500/10 text-red-400"
              : "border-slate-700 hover:border-slate-600 text-slate-400"
          }`}>
          <Mic size={16}/>
          {isRecording ? "Stop Recording" : "Record Audio"}
        </button>
      </div>

      {/* Selected file */}
      {file && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-sky-500/10 flex items-center justify-center">
                <Volume2 size={16} className="text-sky-400"/>
              </div>
              <div>
                <p className="text-sm font-medium">{file.name}</p>
                <p className="text-xs text-slate-500">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
            <button
              onClick={() => setFile(null)}
              className="text-slate-500 hover:text-red-400 transition-colors">
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Analyze button */}
      <button
        onClick={analyzeAudio}
        disabled={!file || processing}
        className="w-full bg-gradient-to-r from-sky-600 to-blue-600 hover:from-sky-500 hover:to-blue-500 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-xl px-6 py-3.5 font-medium transition-all disabled:cursor-not-allowed flex items-center justify-center gap-2">
        <Brain size={16}/>
        Analyze Accent
        <ChevronRight size={16}/>
      </button>
    </div>
  );

  // ── Processing View ────────────────────────────────────────────
  const ProcessingView = (
    <div className="space-y-8 py-12">
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-sky-500/10 mb-4">
          <Brain size={32} className="text-sky-400 animate-pulse"/>
        </div>
        <h2 className="text-xl font-bold mb-2">Analyzing Audio</h2>
        <p className="text-slate-400 text-sm">Processing through 4-phase pipeline</p>
      </div>

      {/* Pipeline steps */}
      <div className="space-y-4">
        {PIPELINE_STEPS.map((step, i) => {
          const status = i < stepIndex ? "done" : i === stepIndex ? "active" : "pending";
          return (
            <div key={i} className="flex items-center gap-4">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                status === "done" ? "bg-green-500/20 text-green-400" :
                status === "active" ? "bg-sky-500/20 text-sky-400 animate-pulse" :
                "bg-slate-800 text-slate-600"
              }`}>
                {status === "done" ? "✓" : i + 1}
              </div>
              <div className="flex-1">
                <div className={`text-sm font-medium transition-colors ${
                  status === "pending" ? "text-slate-600" : "text-white"
                }`}>
                  {step.label}
                </div>
                <div className="text-xs text-slate-500">{step.sub}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );

  // ── Results View ───────────────────────────────────────────────
  const ResultsView = !results ? null : (
    <div className="space-y-5">
      {/* Row 1: Detected accent */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-5xl">{FLAGS[results.accent]}</div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Detected Accent</p>
              <p className="text-xl font-bold capitalize">{results.accent}</p>
            </div>
          </div>
          <div>
            <p className="text-xs text-slate-500 mb-1 text-right">Confidence</p>
            <div className="flex items-center gap-2">
              <div className="h-2 w-32 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full transition-all duration-1000" 
                  style={{
                    width: `${results.confidence}%`,
                    background: scoreColor(results.confidence)
                  }}/>
              </div>
              <span className="text-sm font-mono font-bold" 
                style={{color: scoreColor(results.confidence)}}>
                {results.confidence}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Overall score */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-500 mb-2">Overall Score</p>
            <p className="text-3xl font-bold mb-1">
              Grade <span style={{color: scoreColor(results.overall)}}>{results.grade}</span>
            </p>
            <p className="text-sm text-slate-400">{results.summary}</p>
          </div>
          <ScoreRing score={results.overall} size={110}/>
        </div>
      </div>

      {/* Row 3: Breakdown */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <p className="text-xs text-slate-500 mb-4">Score Breakdown</p>
        <div className="space-y-3">
          {[
            {label: "Phoneme Accuracy", score: results.phoneme, weight: "40%"},
            {label: "Prosody Match", score: results.prosody, weight: "30%"},
            {label: "Accent Confidence", score: results.confidence, weight: "30%"},
          ].map(({label, score, weight}) => (
            <div key={label}>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="text-slate-300">{label}</span>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">{weight}</span>
                  <span className="font-mono font-bold" style={{color: scoreColor(score)}}>
                    {score}
                  </span>
                </div>
              </div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full transition-all duration-1000"
                  style={{width: `${score}%`, background: scoreColor(score)}}/>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Row 4: Substitution patterns */}
      {results.substitutions?.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <p className="text-xs text-slate-500 mb-4">Detected Issues</p>
          <div className="space-y-3">
            {results.substitutions.map((s, i) => (
              <div key={i} className="rounded-xl border border-slate-700 p-3"
                style={{background: "rgba(15,23,42,0.6)"}}>
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full" style={{background: SEV[s.severity]}}/>
                  <span className="text-sm font-medium">{s.pattern}</span>
                  <span className="text-xs text-slate-500 ml-auto">{s.severity}</span>
                </div>
                {s.example && (
                  <p className="text-xs text-slate-500 italic ml-4">{s.example}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Row 5: AI Feedback */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs text-slate-500 flex items-center gap-1.5">
            <Zap size={11} color="#f59e0b"/> AI Coaching
          </p>
          {feedbackLoading && (
            <span className="text-xs text-slate-500 flex items-center gap-1">
              <RefreshCw size={11} className="animate-spin"/> Generating...
            </span>
          )}
        </div>
        {feedbackLoading ? (
          <div className="space-y-2">
            {[90, 100, 75].map((w, i) => (
              <div key={i} className="h-3 bg-slate-800 rounded-full animate-pulse"
                style={{width: `${w}%`, animationDelay: `${i * 120}ms`}}/>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-300 leading-relaxed">{feedback}</p>
        )}
      </div>

      {/* Row 6: Practice exercises */}
      {results.exercises?.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <p className="text-xs text-slate-500 mb-4 flex items-center gap-1.5">
            <BookOpen size={11} color="#10b981"/> Practice Exercises
          </p>
          <div className="space-y-3">
            {results.exercises.map((ex, i) => (
              <div key={i} className="rounded-xl border border-slate-700 p-4"
                style={{background: "rgba(15,23,42,0.6)"}}>
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                    style={{background: "rgba(16,185,129,0.15)", color: "#10b981"}}>
                    {i + 1}
                  </div>
                  <span className="text-sm font-medium">{ex.title}</span>
                </div>
                <div className="ml-8 space-y-1">
                  {ex.phrases.map((p, j) => (
                    <p key={j} className="text-sm text-slate-400 italic">{p}</p>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Analyze another */}
      <div className="flex justify-center pt-2">
        <button onClick={reset}
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-xl px-5 py-2.5 transition-colors">
          <RefreshCw size={13}/> Analyze Another
        </button>
      </div>
    </div>
  );

  // ── Main Render ────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-950 text-white"
      style={{fontFamily: "system-ui,-apple-system,sans-serif"}}>
      
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-slate-800 px-6 py-3.5 flex items-center justify-between"
        style={{background: "rgba(2,6,23,0.92)", backdropFilter: "blur(12px)"}}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{background: "linear-gradient(135deg,#0369a1,#0ea5e9)"}}>
            <Volume2 size={15} color="white"/>
          </div>
          <div>
            <span className="font-semibold text-sm">AccentIQ</span>
            <span className="text-slate-600 text-xs ml-2">Accent Analysis</span>
          </div>
        </div>
        {view !== "upload" && (
          <button onClick={reset}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 rounded-lg px-3 py-1.5 transition-colors">
            <RefreshCw size={11}/> Reset
          </button>
        )}
      </div>

      {/* Body */}
      <div className="max-w-3xl mx-auto px-4 py-8">
        {view === "upload" && UploadView}
        {view === "processing" && ProcessingView}
        {view === "results" && ResultsView}
      </div>
    </div>
  );
}