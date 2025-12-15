#!/usr/bin/env python3
"""
voice_e2e_test.py
Linux/Raspberry Pi: Push-to-talk (Enter) -> record 4s -> STT (faster-whisper) -> Ollama chat -> print.

Requirements (Python):
  pip install pyaudio requests faster-whisper

System:
  sudo apt install -y portaudio19-dev ffmpeg
"""

from __future__ import annotations

import json
import sys
import time
import wave
from dataclasses import dataclass
from typing import Optional

import requests

try:
    import pyaudio
except ImportError as e:
    print("[ERR] PyAudio not installed. pip install pyaudio")
    raise

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    print("[ERR] faster-whisper not installed. pip install faster-whisper")
    raise


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    wav_path: str = "tmp.wav"
    seconds: int = 4
    rate: int = 16000
    channels: int = 1
    chunk: int = 1024
    pyaudio_format: int = pyaudio.paInt16  # int16
    whisper_model: str = "small"  # or "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"  # good on CPU
    language: str = "es"

    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "tinyllama:latest"
    ollama_timeout_s: int = 30


CFG = Config()


# -----------------------------
# Audio recording
# -----------------------------
def record_wav(path: str, seconds: int, rate: int, channels: int, chunk: int, fmt: int) -> None:
    """Record audio from default input device and write WAV file."""
    pa = pyaudio.PyAudio()

    try:
        # Basic microphone availability check
        default_in = pa.get_default_input_device_info()
        if default_in.get("maxInputChannels", 0) <= 0:
            raise RuntimeError("Default input device has no input channels.")
    except Exception as e:
        pa.terminate()
        raise RuntimeError(f"No usable microphone input device found: {e}") from e

    stream = None
    frames: list[bytes] = []
    total_frames = int(rate / chunk * seconds)

    try:
        stream = pa.open(
            format=fmt,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

        print(f"[REC] Recording {seconds}s @ {rate} Hz, mono... (speak now)")
        t0 = time.time()
        for i in range(total_frames):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        dt = time.time() - t0
        print(f"[REC] Done ({dt:.2f}s). Writing WAV -> {path}")

        # Write WAV
        sampwidth = pa.get_sample_size(fmt)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        pa.terminate()


# -----------------------------
# STT (faster-whisper)
# -----------------------------
def transcribe_whisper(wav_path: str, model_name: str, device: str, compute_type: str, language: str) -> str:
    """Transcribe WAV with faster-whisper and return normalized text."""
    print(f"[STT] Loading faster-whisper model='{model_name}' device='{device}' compute='{compute_type}'...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print("[STT] Transcribing...")
    segments, info = model.transcribe(
        wav_path,
        language=language,
        vad_filter=True,          # helps with silence
        vad_parameters={"min_silence_duration_ms": 300},
    )

    text_parts = []
    for seg in segments:
        if seg.text:
            text_parts.append(seg.text.strip())

    text = " ".join([t for t in text_parts if t]).strip()
    print(f"[STT] Detected language: {getattr(info, 'language', 'unknown')}, prob: {getattr(info, 'language_probability', 0):.2f}")
    return text


# -----------------------------
# Ollama chat
# -----------------------------
def ask_ollama(text: str, url: str, model: str, timeout_s: int) -> str:
    """Send a chat message to Ollama and return assistant response."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Responde en español, breve y claro."},
            {"role": "user", "content": text},
        ],
    }

    print(f"[LLM] POST {url} model={model} timeout={timeout_s}s")
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out.")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Cannot connect to Ollama at {url}. Is it running? ({e})")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e} | Body: {resp.text[:300]}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama returned invalid JSON: {e} | Body: {resp.text[:300]}")

    # Expected shape: {"message":{"role":"assistant","content":"..."} ...}
    msg = data.get("message", {})
    content = (msg.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Ollama response missing content. Raw: {str(data)[:300]}")
    return content


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    print("=== Voice E2E Test (Enter -> Record -> STT -> Ollama) ===")
    print(f"Config: {CFG.seconds}s, {CFG.rate}Hz mono, whisper='{CFG.whisper_model}', ollama='{CFG.ollama_model}'")
    print("Press Enter to record. Type 'q' + Enter to quit.\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            print("[OK] Exit.")
            return 0

        try:
            record_wav(CFG.wav_path, CFG.seconds, CFG.rate, CFG.channels, CFG.chunk, CFG.pyaudio_format)
        except Exception as e:
            print(f"[ERR] Recording failed: {e}")
            continue

        try:
            text = transcribe_whisper(CFG.wav_path, CFG.whisper_model, CFG.whisper_device, CFG.whisper_compute_type, CFG.language)
        except Exception as e:
            print(f"[ERR] STT failed: {e}")
            continue

        if not text:
            print("[STT] Empty transcription (silence or too noisy). Try again.")
            continue

        print(f"\n[STT] Transcribed: {text}\n")

        try:
            answer = ask_ollama(text, CFG.ollama_url, CFG.ollama_model, CFG.ollama_timeout_s)
        except Exception as e:
            print(f"[ERR] LLM failed: {e}")
            continue

        print("[LLM] Answer:")
        print(answer)
        print("\n---\n")


if __name__ == "__main__":
    raise SystemExit(main())
