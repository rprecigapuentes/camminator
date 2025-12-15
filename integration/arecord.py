import threading
import time
import math
import json
import sys
import os
import subprocess
import tempfile
import wave

import cv2
import numpy as np
from gpiozero import Button
from ultralytics import YOLO
from picamera2 import Picamera2
from pyrplidar import PyRPlidar

# --------------------------
# CONFIG
# --------------------------
with open("config.json", "r") as f:
    CONFIG = json.load(f)

# --------------------------
# SHARED STATE
# --------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.min_lidar_distance_mm = None
        self.yolo_labels = set()
        self.pause_lidar = threading.Event()   # 🔑 CLAVE OPCIÓN A

shared = SharedState()

# --------------------------
# LIDAR THREAD (MINIMO)
# --------------------------
def lidar_thread():
    print("[LIDAR] Thread iniciado")

    while True:
        # Espera si el botón pausó el LiDAR
        shared.pause_lidar.wait()

        lidar = PyRPlidar()
        try:
            lidar.connect(
                port=CONFIG["lidar"]["port"],
                baudrate=CONFIG["lidar"]["baudrate"],
                timeout=3
            )
            lidar.set_motor_pwm(CONFIG["lidar"]["motor_pwm"])
            time.sleep(2)

            scan_gen = lidar.force_scan()

            for scan in scan_gen():
                if shared.pause_lidar.is_set() is False:
                    break

                angle = scan.angle % 360
                dist = scan.distance

                # Solo frente (±60°)
                if (angle <= 60 or angle >= 300) and dist > 0:
                    with shared.lock:
                        shared.min_lidar_distance_mm = dist

        except Exception as e:
            print(f"[LIDAR] Error: {e}")

        finally:
            try:
                lidar.stop()
                lidar.set_motor_pwm(0)
                lidar.disconnect()
            except Exception:
                pass

        time.sleep(1)

# --------------------------
# YOLO THREAD (SIN DIBUJAR)
# --------------------------
def yolo_thread():
    print("[YOLO] Thread iniciado")

    model = YOLO(CONFIG["yolo"]["model_path"])
    labels = model.names

    cam = Picamera2()
    cam.configure(
        cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
    )
    cam.start()
    time.sleep(1)

    while True:
        frame = cam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame, verbose=False)[0]
        current_labels = set()

        for box in results.boxes:
            if float(box.conf) > CONFIG["yolo"]["threshold"]:
                current_labels.add(labels[int(box.cls)])

        with shared.lock:
            shared.yolo_labels = current_labels

# --------------------------
# AUDIO UTILS (ARECORD + WHISPER)
# --------------------------
def record_wav():
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "arecord",
        "-D", CONFIG["asistente"]["alsa_capture_device"],
        "-f", "S16_LE",
        "-r", str(CONFIG["asistente"]["audio_rate"]),
        "-c", "1",
        "-d", str(CONFIG["asistente"]["record_seconds"]),
        path
    ]
    subprocess.run(cmd, check=True)
    return path

def load_wav(path):
    with wave.open(path, "rb") as wf:
        data = wf.readframes(wf.getnframes())
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio

# --------------------------
# BUTTON THREAD (PAUSA LIDAR)
# --------------------------
def button_thread():
    print("[BUTTON] Thread iniciado")
    import whisper
    import requests
    import pyttsx3

    whisper_model = whisper.load_model(CONFIG["asistente"]["whisper_model"])
    tts = pyttsx3.init()

    button = Button(CONFIG["gpio"]["button_pin"])

    while True:
        button.wait_for_press()

        print("[BUTTON] Presionado → pausando LiDAR")
        shared.pause_lidar.clear()   # 🔴 PAUSA LIDAR

        try:
            wav = record_wav()
            audio = load_wav(wav)

            text = whisper_model.transcribe(
                audio,
                language="en",
                fp16=False
            )["text"].strip()

            if text:
                print(f"[WHISPER] {text}")

                payload = {
                    "model": CONFIG["asistente"]["ollama_model"],
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "Answer briefly in English."},
                        {"role": "user", "content": text}
                    ]
                }
                r = requests.post(CONFIG["asistente"]["ollama_url"], json=payload, timeout=30)
                reply = r.json()["message"]["content"]

                print(f"[OLLAMA] {reply}")
                tts.say(reply)
                tts.runAndWait()

        except Exception as e:
            print(f"[BUTTON] Error: {e}")

        finally:
            try:
                os.remove(wav)
            except Exception:
                pass

            print("[BUTTON] Reanudando LiDAR")
            shared.pause_lidar.set()   # 🟢 REANUDA LIDAR

# --------------------------
# ALERT LOOP (YOLO + LIDAR)
# --------------------------
def alert_loop():
    print("[ALERT] Loop iniciado")
    import pyttsx3
    tts = pyttsx3.init()

    while True:
        time.sleep(0.5)

        with shared.lock:
            d = shared.min_lidar_distance_mm
            labels = list(shared.yolo_labels)

        if d and d < 1000 and labels:
            msg = f"There is {', '.join(labels)} at {d/1000:.1f} meters. Attention."
            print(f"[ALERT] {msg}")
            tts.say(msg)
            tts.runAndWait()

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    print("[MAIN] Sistema mínimo iniciando")

    shared.pause_lidar.set()  # LiDAR activo por defecto

    threading.Thread(target=lidar_thread, daemon=True).start()
    threading.Thread(target=yolo_thread, daemon=True).start()
    threading.Thread(target=button_thread, daemon=True).start()
    threading.Thread(target=alert_loop, daemon=True).start()

    while True:
        time.sleep(1)
