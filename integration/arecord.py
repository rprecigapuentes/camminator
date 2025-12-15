# sistema_minimo_v2.py
import os
import sys
import json
import time
import math
import threading
import subprocess
import tempfile
import wave
import requests
import numpy as np

from gpiozero import Button
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2


# --------------------------
# CONFIG
# --------------------------
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERROR: falta config.json")
    sys.exit(1)
except json.JSONDecodeError:
    print("ERROR: config.json inválido")
    sys.exit(1)


# --------------------------
# SHARED STATE
# --------------------------
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.lidar_min_mm = None
        self.yolo_labels = set()
        self.last_alert_ts = 0.0

    def set_lidar_min(self, v):
        with self._lock:
            self.lidar_min_mm = v

    def get_lidar_min(self):
        with self._lock:
            return self.lidar_min_mm

    def set_yolo_labels(self, labels):
        with self._lock:
            self.yolo_labels = set(labels)

    def get_yolo_labels(self):
        with self._lock:
            return sorted(self.yolo_labels)

    def can_alert(self, min_interval_s):
        now = time.time()
        with self._lock:
            if (now - self.last_alert_ts) >= min_interval_s:
                self.last_alert_ts = now
                return True
            return False


# --------------------------
# TTS (speaker)
# --------------------------
class Speaker:
    def __init__(self):
        self._lock = threading.Lock()
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            # Opcional: ajustar velocidad/volumen
            rate = CONFIG.get("tts", {}).get("rate", None)
            volume = CONFIG.get("tts", {}).get("volume", None)
            if rate is not None:
                self._engine.setProperty("rate", int(rate))
            if volume is not None:
                self._engine.setProperty("volume", float(volume))
        except Exception as e:
            print(f"[TTS] ERROR init: {e}")
            self._engine = None

    def say(self, text: str):
        if not text:
            return
        if self._engine is None:
            print("[TTS] No disponible. Texto:", text)
            return
        # Evitar que se pisen mensajes
        def _run():
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f"[TTS] ERROR hablando: {e}")
        threading.Thread(target=_run, daemon=True).start()


# --------------------------
# OLLAMA
# --------------------------
def ask_ollama(text: str) -> str:
    payload = {
        "model": CONFIG["asistente"]["ollama_model"],
        "stream": False,
        "messages": [
            {"role": "system", "content": "Responde en español, breve y claro."},
            {"role": "user", "content": text},
        ],
    }
    try:
        r = requests.post(CONFIG["asistente"]["ollama_url"], json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}).get("content", "") or "").strip()
    except Exception as e:
        return f"[Ollama ERROR] {e}"


# --------------------------
# AUDIO: arecord -> whisper
# --------------------------
def record_wav_arecord() -> str:
    alsa_dev = CONFIG["asistente"].get("alsa_capture_device", "plughw:2,0")
    rate = int(CONFIG["asistente"]["audio_rate"])
    channels = int(CONFIG["asistente"]["audio_channels"])
    seconds = float(CONFIG["asistente"]["record_seconds"])

    fd, wav_path = tempfile.mkstemp(prefix="mic_", suffix=".wav")
    os.close(fd)

    cmd = [
        "arecord",
        "-D", alsa_dev,
        "-f", "S16_LE",
        "-r", str(rate),
        "-c", str(channels),
        "-d", str(int(seconds)),
        wav_path
    ]
    # Deja stderr visible si algo falla
    subprocess.run(cmd, check=True)
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


# --------------------------
# LIDAR THREAD: min distance in frontal angles only
# --------------------------
def lidar_thread(shared: SharedState):
    print("[LIDAR] Iniciando (mínimo por ángulo frontal)...")
    from pyrplidar import PyRPlidar

    port = CONFIG["lidar"]["port"]
    baud = int(CONFIG["lidar"]["baudrate"])
    pwm = int(CONFIG["lidar"]["motor_pwm"])

    a1_min = float(CONFIG["lidar"].get("front_angle_min1", 0))
    a1_max = float(CONFIG["lidar"].get("front_angle_max1", 60))
    a2_min = float(CONFIG["lidar"].get("front_angle_min2", 300))
    a2_max = float(CONFIG["lidar"].get("front_angle_max2", 360))

    max_mm = float(CONFIG["lidar"].get("max_considered_mm", 2000))
    publish_every = float(CONFIG["lidar"].get("publish_every_s", 0.10))

    def norm(a):
        a = a % 360.0
        return a if a >= 0 else a + 360.0

    while True:
        lidar = PyRPlidar()
        try:
            lidar.connect(port=port, baudrate=baud, timeout=3)
            lidar.set_motor_pwm(pwm)
            time.sleep(1.5)
            gen = lidar.force_scan()
            print("[LIDAR] Conectado.")

            last_pub = 0.0
            current_min = None

            for s in gen():
                ang = norm(float(s.angle))
                dist = float(s.distance)

                frontal = (a1_min <= ang <= a1_max) or (a2_min <= ang <= a2_max)
                if not frontal:
                    continue
                if dist <= 0 or dist > max_mm:
                    continue

                if (current_min is None) or (dist < current_min):
                    current_min = dist

                now = time.time()
                if (now - last_pub) >= publish_every:
                    shared.set_lidar_min(current_min)
                    last_pub = now

        except Exception as e:
            print(f"[LIDAR] ERROR: {e}")
            shared.set_lidar_min(None)
            time.sleep(0.5)

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

        time.sleep(0.5)


# --------------------------
# MAIN: YOLO loop + ALERT (TTS) + BUTTON (Whisper+Ollama+TTS)
# --------------------------
def main():
    print("[Main] Iniciando sistema v2...")

    shared = SharedState()
    speaker = Speaker()

    # LIDAR thread
    threading.Thread(target=lidar_thread, args=(shared,), daemon=True).start()

    # Whisper init (solo botón)
    try:
        import whisper
        whisper_model = whisper.load_model(CONFIG["asistente"]["whisper_model"])
    except Exception as e:
        print(f"[Audio] WARNING: Whisper no disponible: {e}")
        whisper_model = None

    # YOLO + CameraPi0
    model = YOLO(CONFIG["yolo"]["model_path"], task="detect")
    labels_map = model.names

    picam2 = Picamera2()
    resW, resH = map(int, CONFIG["yolo"]["resolution"].split("x"))
    cam_config = picam2.create_video_configuration(main={"size": (resW, resH), "format": "RGB888"})
    picam2.configure(cam_config)
    picam2.start()
    time.sleep(1.0)

    button = Button(int(CONFIG["gpio"]["button_pin"]))

    # params
    alert_mm = float(CONFIG["sistema"].get("alert_distance_mm", 1000))
    alert_min_interval = float(CONFIG["sistema"].get("min_alert_interval_s", 2.0))
    yolo_threshold = float(CONFIG["yolo"]["threshold"])
    console_every = float(CONFIG["sistema"].get("console_every_s", 0.5))

    # ------------ BOTÓN: arecord -> whisper -> ollama -> TTS
    def on_button():
        print("[BOTON] Presionado. Grabando con arecord...")
        if whisper_model is None:
            print("[BOTON] Whisper no está disponible.")
            return

        try:
            wav_path = record_wav_arecord()
            audio = load_wav_float32(wav_path)
            os.remove(wav_path)
        except Exception as e:
            print(f"[BOTON] ERROR grabando/leyendo WAV: {e}")
            return

        try:
            print("[BOTON] Transcribiendo (Whisper)...")
            r = whisper_model.transcribe(audio, fp16=False, language="es")
            text = (r.get("text") or "").strip()
            print("[BOTON] Texto:", text if text else "(vacío)")
        except Exception as e:
            print(f"[BOTON] ERROR Whisper: {e}")
            return

        if not text:
            speaker.say("No entendí el audio.")
            return

        # Contexto mínimo: etiquetas YOLO + lidar actual
        yolo_seen = shared.get_yolo_labels()
        dmm = shared.get_lidar_min()
        ctx = (
            f"Etiquetas YOLO actuales: {', '.join(yolo_seen) if yolo_seen else '(ninguna)'}.\n"
            f"Distancia mínima LiDAR frontal: {int(dmm)} mm.\n"
        )

        prompt = (
            "Contexto del sistema:\n"
            f"{ctx}\n"
            f"Pregunta del usuario: {text}\n"
            "Responde breve y útil."
        )

        resp = ask_ollama(prompt)
        print("[Ollama] Respuesta:", resp)
        speaker.say(resp)

    button.when_pressed = lambda: threading.Thread(target=on_button, daemon=True).start()

    # ------------ LOOP YOLO + ALERTA TTS
    last_console = 0.0
    last_labels = []
    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = model(frame, verbose=False)
            boxes = results[0].boxes

            seen = set()
            # dibujar boxes
            for i in range(len(boxes)):
                conf = float(boxes[i].conf.item())
                if conf < yolo_threshold:
                    continue

                cls = int(boxes[i].cls.item())
                name = str(labels_map.get(cls, cls))
                seen.add(name)

                xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            shared.set_yolo_labels(seen)

            dmm = shared.get_lidar_min()
            now = time.time()
            if (now - last_console) >= console_every:
                last_labels = sorted(seen)
                print(f"[Estado] LiDAR_min_mm={int(dmm) if dmm is not None else None} | YOLO={last_labels}")
                last_console = now

            # ALERTA sin Ollama: solo TTS con etiquetas + distancia
            if (dmm is not None) and (dmm < alert_mm) and shared.can_alert(alert_min_interval):
                labels_txt = ", ".join(shared.get_yolo_labels()) if shared.get_yolo_labels() else "algo"
                meters = dmm / 1000.0
                msg = f"Atención. Hay {labels_txt} a {meters:.2f} metros."
                print("[ALERTA]", msg)
                speaker.say(msg)

            if CONFIG["sistema"].get("show_window", True):
                cv2.imshow("Sistema v2", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                time.sleep(0.01)

    finally:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSistema detenido por el usuario.")
        sys.exit(0)
