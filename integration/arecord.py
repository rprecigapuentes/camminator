import os
import sys
import json
import time
import math
import threading
import tempfile
import subprocess
import wave
import requests

import numpy as np
import cv2

from gpiozero import Button
from ultralytics import YOLO
from picamera2 import Picamera2


# --------------------------
# CONFIG
# --------------------------
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERROR: El archivo 'config.json' no fue encontrado.")
    sys.exit(1)
except json.JSONDecodeError:
    print("ERROR: El archivo 'config.json' tiene un formato inválido.")
    sys.exit(1)


# --------------------------
# SHARED STATE
# --------------------------
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.yolo_labels = set()
        self.lidar_min_mm = None
        self.lidar_last_good_mm = None
        self.last_alert_ts = 0.0
        self.audio_busy = False

        # Para hablar respuestas largas del botón (Ollama)
        self.pending_tts_text = None

    def set_audio_busy(self, v: bool):
        with self._lock:
            self.audio_busy = bool(v)

    def is_audio_busy(self) -> bool:
        with self._lock:
            return bool(self.audio_busy)

    def set_yolo_labels(self, labels_set):
        with self._lock:
            self.yolo_labels = set(labels_set)

    def get_yolo_labels(self):
        with self._lock:
            return set(self.yolo_labels)

    def set_lidar_min(self, v):
        with self._lock:
            self.lidar_min_mm = v
            if v is not None:
                self.lidar_last_good_mm = v

    def get_lidar_min(self):
        with self._lock:
            return self.lidar_min_mm

    def get_lidar_last_good(self):
        with self._lock:
            return self.lidar_last_good_mm

    def should_alert(self, cooldown_s: float) -> bool:
        with self._lock:
            now = time.time()
            if (now - self.last_alert_ts) >= cooldown_s:
                self.last_alert_ts = now
                return True
            return False

    def set_pending_tts(self, text: str):
        with self._lock:
            self.pending_tts_text = text

    def pop_pending_tts(self):
        with self._lock:
            t = self.pending_tts_text
            self.pending_tts_text = None
            return t


# --------------------------
# TTS (pyttsx3)
# --------------------------
class Speaker:
    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        # intenta poner voz en inglés si existe (no siempre hay)
        try:
            voices = self.engine.getProperty("voices")
            for v in voices:
                name = (getattr(v, "name", "") or "").lower()
                lang = str(getattr(v, "languages", "")).lower()
                if "en" in name or "english" in name or "en" in lang:
                    self.engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass

    def say(self, text: str):
        # no metas strings vacíos
        if not text or not text.strip():
            return
        self.engine.say(text)
        self.engine.runAndWait()


# --------------------------
# AUDIO: arecord -> wav -> numpy
# --------------------------
def record_audio_arecord() -> str:
    alsa_dev = CONFIG["asistente"].get("alsa_capture_device", "plughw:2,0")
    rate = int(CONFIG["asistente"].get("audio_rate", 48000))
    channels = int(CONFIG["asistente"].get("audio_channels", 1))
    seconds = float(CONFIG["asistente"].get("record_seconds", 4))

    fd, wav_path = tempfile.mkstemp(prefix="mic_", suffix=".wav")
    os.close(fd)

    cmd = [
        "arecord",
        "-D", alsa_dev,
        "-f", "S16_LE",
        "-r", str(rate),
        "-c", str(channels),
        "-d", str(int(seconds)),
        wav_path,
    ]

    # Importante: si arecord se cuelga, te cuelga el hilo del botón.
    # Le metemos timeout.
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=int(seconds) + 3,
    )
    return wav_path


def load_wav_float32(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise RuntimeError(f"WAV no es 16-bit PCM (sampwidth={sampwidth})")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio


def whisper_transcribe(audio_np: np.ndarray) -> str:
    import whisper
    model_name = CONFIG["asistente"].get("whisper_model", "base")
    model = whisper.load_model(model_name)
    res = model.transcribe(audio_np, fp16=False, language="en")
    return (res.get("text") or "").strip()


# --------------------------
# OLLAMA (EN inglés)
# --------------------------
def ask_ollama_english(question_text: str, yolo_labels, lidar_mm) -> str:
    url = CONFIG["asistente"]["ollama_url"]
    model = CONFIG["asistente"]["ollama_model"]

    labels_str = ", ".join(sorted(list(yolo_labels))) if yolo_labels else "nothing"
    dist_str = f"{int(lidar_mm)} mm" if lidar_mm is not None else "unknown distance"

    system_prompt = (
        "You are an assistant for a navigation aid.\n"
        "Always reply in English. Be concise.\n"
        "If asked about the scene, use the provided detections and distance.\n"
    )

    user_context = (
        f"YOLO sees: {labels_str}\n"
        f"Closest LiDAR distance (front sector): {dist_str}\n"
        f"User question: {question_text}\n"
    )

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ],
    }

    try:
        r = requests.post(url, json=payload, timeout=40)
        r.raise_for_status()
        data = r.json()
        return (data["message"]["content"] or "").strip()
    except Exception as e:
        return f"I could not reach the language model. Error: {e}"


# --------------------------
# LIDAR THREAD (mínimo: distancia mínima en sector frontal)
# --------------------------
def lidar_thread(shared: SharedState):
    print("[LIDAR] Thread started.")

    from pyrplidar import PyRPlidar

    port = CONFIG["lidar"]["port"]
    baud = int(CONFIG["lidar"]["baudrate"])
    pwm = int(CONFIG["lidar"]["motor_pwm"])

    a1_min = float(CONFIG["lidar"].get("front_angle_min1", 0))
    a1_max = float(CONFIG["lidar"].get("front_angle_max1", 60))
    a2_min = float(CONFIG["lidar"].get("front_angle_min2", 300))
    a2_max = float(CONFIG["lidar"].get("front_angle_max2", 360))

    max_mm = float(CONFIG["lidar"].get("alert_distance", 1000)) * 1.5  # margen

    publish_every = float(CONFIG["lidar"].get("publish_every_s", 0.15))

    # Filtros para evitar basura (ajústalos si tu lidar real da valores raros)
    min_valid_mm = int(CONFIG["lidar"].get("min_valid_mm", 80))
    max_valid_mm = int(CONFIG["lidar"].get("max_valid_mm", 6000))

    while True:
        lidar = PyRPlidar()
        try:
            lidar.connect(port=port, baudrate=baud, timeout=3)
            lidar.set_motor_pwm(pwm)
            time.sleep(1.2)
            gen = lidar.force_scan()
            print("[LIDAR] Connected. Scanning...")

            window_start = time.time()
            window_min = None

            for s in gen():
                ang = float(s.angle) % 360.0
                dist = float(s.distance)

                # rango frontal
                in_front = ((a1_min <= ang <= a1_max) or (a2_min <= ang <= a2_max))
                if not in_front:
                    continue

                # filtros duros contra basura/corrupción
                if dist <= 0:
                    continue
                if dist < min_valid_mm:
                    continue
                if dist > max_valid_mm:
                    continue
                if dist > max_mm:
                    continue

                if window_min is None or dist < window_min:
                    window_min = dist

                now = time.time()
                if (now - window_start) >= publish_every:
                    if window_min is not None:
                        shared.set_lidar_min(float(window_min))
                    else:
                        # Ventana vacía: NO pongas 0. Mantén último valor bueno.
                        shared.set_lidar_min(shared.get_lidar_last_good())

                    window_min = None
                    window_start = now

        except Exception as e:
            print(f"[LIDAR] ERROR: {e}")
            time.sleep(0.8)

        finally:
            try:
                lidar.stop()
            except Exception:
                pass
            try:
                lidar.set_motor_pwm(0)
            except Exception:
                pass
            try:
                lidar.disconnect()
            except Exception:
                pass

        time.sleep(0.6)


# --------------------------
# BUTTON THREAD: arecord -> whisper -> ollama -> TTS
# --------------------------
def button_thread(shared: SharedState, speaker: Speaker, button_pin: int):
    button = Button(button_pin)
    print("[BUTTON] Thread started. Waiting for press...")

    while True:
        button.wait_for_press()

        # Marca ocupado (para debug; no “pausa” lidar)
        shared.set_audio_busy(True)
        try:
            print("[BUTTON] Pressed. Recording with arecord...")

            wav_path = None
            try:
                wav_path = record_audio_arecord()
            except Exception as e:
                print(f"[BUTTON] arecord failed: {e}")
                speaker.say("Recording failed.")
                continue

            try:
                audio = load_wav_float32(wav_path)
                text = whisper_transcribe(audio)
            except Exception as e:
                print(f"[BUTTON] Whisper failed: {e}")
                speaker.say("Transcription failed.")
                continue
            finally:
                if wav_path:
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

            print(f"[BUTTON] Transcribed (EN): {text}")

            # Contexto actual
            labels = shared.get_yolo_labels()
            lidar_mm = shared.get_lidar_min()

            # Ollama en inglés
            reply = ask_ollama_english(text, labels, lidar_mm)
            print(f"[BUTTON] Ollama reply: {reply}")

            # Habla respuesta
            speaker.say(reply)

        finally:
            shared.set_audio_busy(False)

        time.sleep(0.2)


# --------------------------
# MAIN: YOLO loop + ALERTA TTS (YOLO labels + LiDAR min)
# --------------------------
def main():
    print("[MAIN] Starting...")

    shared = SharedState()
    speaker = Speaker()

    # YOLO
    model = YOLO(CONFIG["yolo"]["model_path"], task="detect")
    labels_map = model.names

    # CameraPi0 (Picamera2). config.json camera_source no se usa aquí.
    picam2 = Picamera2()
    if CONFIG["yolo"].get("resolution"):
        resW, resH = map(int, CONFIG["yolo"]["resolution"].split("x"))
        cam_config = picam2.create_video_configuration(main={"size": (resW, resH), "format": "RGB888"})
    else:
        cam_config = picam2.create_video_configuration(main={"format": "RGB888"})
    picam2.configure(cam_config)
    picam2.start()
    time.sleep(0.7)

    # Threads
    threading.Thread(target=lidar_thread, args=(shared,), daemon=True).start()
    threading.Thread(
        target=button_thread,
        args=(shared, speaker, int(CONFIG["gpio"]["button_pin"])),
        daemon=True
    ).start()

    # Alert settings
    alert_mm = float(CONFIG["lidar"].get("alert_distance", 1000))  # 1m default
    alert_cooldown_s = float(CONFIG["sistema"].get("alert_cooldown_s", 2.0))

    print("[MAIN] Running YOLO loop. Press 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # YOLO inference
            results = model(frame, verbose=False)
            boxes = results[0].boxes

            seen = set()
            thr = float(CONFIG["yolo"].get("threshold", 0.5))

            for i in range(len(boxes)):
                cls = int(boxes[i].cls.item())
                conf = float(boxes[i].conf.item())
                if conf >= thr:
                    seen.add(str(labels_map[cls]))

            shared.set_yolo_labels(seen)

            # LiDAR min
            dmm = shared.get_lidar_min()

            # LOG en consola (tu pedido)
            # 1) salida yolo
            # 2) salida lidar
            # 3) “asociación” básica: texto con ambos
            yolo_str = ", ".join(sorted(seen)) if seen else "none"
            lidar_str = f"{int(dmm)} mm" if dmm is not None else "none"
            assoc = f"YOLO:[{yolo_str}] | LIDAR_MIN_FRONT:{lidar_str}"
            print(f"[STATUS] {assoc}")

            # ALERTA: solo si lidar < 1m y hay labels
            if dmm is not None and dmm <= alert_mm and shared.should_alert(alert_cooldown_s):
                phrase = f"Attention. I see {yolo_str}. Closest distance {int(dmm)} millimeters."
                print(f"[ALERT] {phrase}")
                speaker.say(phrase)

            # Display opcional (sin cajas)
            cv2.imshow("Sistema Integrado (no boxes)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")
        sys.exit(0)
