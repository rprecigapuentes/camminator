# sistema_integrado.py
import threading
import time
import cv2
import numpy as np
import math
import os
import sys
import json
import requests
import subprocess
import tempfile
import wave

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
# Helpers: regiones + debug
# --------------------------
def region_from_x(cx: int, frame_w: int) -> str:
    z = frame_w // 3
    if cx < z:
        return "LEFT"
    if cx < 2 * z:
        return "CENTER"
    return "RIGHT"


def region_from_lidar_angle_front(angle_deg: float) -> str:
    """
    Tu LiDAR frontal está en 0..60 (derecha) y 300..360 (izquierda).
    Centro aproximado: 330..360 y 0..30.
    """
    a = angle_deg % 360.0
    if (330.0 <= a <= 360.0) or (0.0 <= a <= 30.0):
        return "CENTER"
    if 300.0 <= a <= 360.0:
        return "LEFT"
    if 0.0 <= a <= 60.0:
        return "RIGHT"
    return "UNKNOWN"


def fmt_mm(x):
    return "None" if x is None else f"{x:.0f} mm"


# --------------------------
# SHARED STATE
# --------------------------
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "detections": {},
            "lidar_objects": {},
            "proximity_alert": None,
            "llm_response": None,
            "current_speech_action": None,
        }

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def update_detections(self, new_detections):
        with self._lock:
            self._data["detections"] = new_detections

    def update_lidar_objects(self, new_lidar_objects):
        with self._lock:
            self._data["lidar_objects"] = new_lidar_objects

    def set_proximity_alert(self, alert_data):
        with self._lock:
            self._data["proximity_alert"] = alert_data
            self._data["current_speech_action"] = "proximity_alert"

    def set_llm_response(self, response_text):
        with self._lock:
            if self._data.get("current_speech_action") != "proximity_alert":
                self._data["llm_response"] = response_text
                self._data["current_speech_action"] = "llm_response"

    def clear_speech_action(self):
        with self._lock:
            self._data["current_speech_action"] = None


# --------------------------
# LIDAR THREAD (SMART + AUTO-RECOVERY)
# --------------------------
def lidar_process(shared_state):
    print("[Hilo - LIDAR] Iniciando LIDAR (modo smart + auto-recovery)...")

    from pyrplidar import PyRPlidar

    def angle_difference(a1, a2):
        diff = abs(a1 - a2)
        return min(diff, 360 - diff)

    def normalize_angle(a):
        a = a % 360
        if a < 0:
            a += 360
        return a

    def analyze_object(group):
        if not group:
            return None
        angles = [p[0] for p in group]
        dists = [p[1] for p in group]

        x_sum = 0.0
        y_sum = 0.0
        for ang in angles:
            rad = math.radians(ang)
            x_sum += math.cos(rad)
            y_sum += math.sin(rad)
        x_avg = x_sum / len(angles)
        y_avg = y_sum / len(angles)
        avg_angle = normalize_angle(math.degrees(math.atan2(y_avg, x_avg)))

        min_d = min(dists)
        avg_d = sum(dists) / len(dists)

        if len(angles) > 1:
            adjusted = []
            for a in angles:
                if abs(angle_difference(a, avg_angle)) > 180:
                    a = a + 360 if a < avg_angle else a - 360
                adjusted.append(a)
            angle_span = max(adjusted) - min(adjusted)
        else:
            angle_span = 0.0

        width_est = 0.0
        if angle_span > 0.1 and avg_d > 0:
            width_est = 2 * avg_d * math.tan(math.radians(angle_span / 2))

        if 0 <= avg_angle <= 60:
            pos = "FRONTAL DERECHO"
        elif 300 <= avg_angle <= 360:
            pos = "FRONTAL IZQUIERDO"
        elif 0 <= avg_angle <= 30 or 330 <= avg_angle <= 360:
            pos = "CENTRO FRONTAL"
        else:
            pos = "FRONTAL"

        return {
            "avg_angle": avg_angle,
            "min_distance": min_d,
            "avg_distance": avg_d,
            "angle_span": angle_span,
            "width_est": width_est,
            "point_count": len(group),
            "position": pos,
            "timestamp": time.time(),
        }

    def are_objects_same(obj1, obj2):
        dist_diff = abs(obj1["min_distance"] - obj2["min_distance"])
        ang_diff = angle_difference(obj1["avg_angle"], obj2["avg_angle"])
        return (dist_diff <= OBJECT_DISTANCE_THRESHOLD and ang_diff <= OBJECT_ANGLE_THRESHOLD)

    def merge_object_groups(groups):
        merged = []
        for group in groups:
            info = analyze_object(group)
            if info is None:
                continue
            done = False
            for i, existing in enumerate(merged):
                if are_objects_same(info, existing["info"]):
                    merged[i]["points"].extend(group)
                    merged[i]["info"] = analyze_object(merged[i]["points"])
                    done = True
                    break
            if not done:
                merged.append({"points": list(group), "info": info})
        return merged

    def group_points_by_distance_and_angle(points):
        if len(points) < MIN_POINTS_PER_OBJECT:
            return []
        pts = sorted(points, key=lambda p: p[0])
        groups, cur = [], []
        for ang, dist in pts:
            if not cur:
                cur.append((ang, dist))
                continue
            last_ang, last_dist = cur[-1]
            a_diff = angle_difference(ang, last_ang)
            d_diff = abs(dist - last_dist)
            if a_diff <= ANGLE_THRESHOLD and d_diff <= DISTANCE_THRESHOLD:
                cur.append((ang, dist))
            else:
                if len(cur) >= MIN_POINTS_PER_OBJECT:
                    groups.append(cur)
                cur = [(ang, dist)]
        if len(cur) >= MIN_POINTS_PER_OBJECT:
            groups.append(cur)
        return groups

    def should_alert_for_object(obj_info, history):
        now = time.time()
        for t, old in history:
            if (now - t) < MIN_TIME_BETWEEN_ALERTS and are_objects_same(obj_info, old):
                return False
        return True

    FRONT_ANGLE_MIN1 = float(CONFIG["lidar"].get("front_angle_min1", 0))
    FRONT_ANGLE_MAX1 = float(CONFIG["lidar"].get("front_angle_max1", 60))
    FRONT_ANGLE_MIN2 = float(CONFIG["lidar"].get("front_angle_min2", 300))
    FRONT_ANGLE_MAX2 = float(CONFIG["lidar"].get("front_angle_max2", 360))

    ALERT_DISTANCE = float(CONFIG["lidar"].get("alert_distance", 1000))  # mm
    DISTANCE_THRESHOLD = float(CONFIG["lidar"].get("distance_threshold", 50))  # mm
    ANGLE_THRESHOLD = float(CONFIG["lidar"].get("angle_gap_threshold", 5))  # deg

    MIN_POINTS_PER_OBJECT = int(CONFIG["lidar"].get("min_points_per_object", 3))
    BUFFER_SIZE = int(CONFIG["lidar"].get("buffer_size", 150))
    OBJECT_DISTANCE_THRESHOLD = float(CONFIG["lidar"].get("object_distance_threshold", 100))  # mm
    OBJECT_ANGLE_THRESHOLD = float(CONFIG["lidar"].get("object_angle_threshold", 15))  # deg
    MIN_TIME_BETWEEN_ALERTS = float(CONFIG["lidar"].get("min_time_between_alerts", 2.0))  # s
    PROCESS_EVERY_N_POINTS = int(CONFIG["lidar"].get("process_every_n_points", 40))

    port = CONFIG["lidar"]["port"]
    baud = int(CONFIG["lidar"]["baudrate"])
    pwm = int(CONFIG["lidar"]["motor_pwm"])

    while True:
        lidar = PyRPlidar()
        try:
            lidar.connect(port=port, baudrate=baud, timeout=3)
            lidar.set_motor_pwm(pwm)
            time.sleep(2)
            print("[Hilo - LIDAR] Conectado y escaneando.")

            scan_generator = lidar.force_scan()
            point_buffer = []
            alert_history = []
            total_points = 0

            for scan in scan_generator():
                total_points += 1
                angle = normalize_angle(scan.angle)
                dist = scan.distance

                in_frontal = (
                    (FRONT_ANGLE_MIN1 <= angle <= FRONT_ANGLE_MAX1)
                    or (FRONT_ANGLE_MIN2 <= angle <= FRONT_ANGLE_MAX2)
                )

                if in_frontal and dist > 0 and dist <= ALERT_DISTANCE * 1.2:
                    point_buffer.append((angle, dist))
                    if len(point_buffer) > BUFFER_SIZE:
                        point_buffer = point_buffer[-BUFFER_SIZE:]

                if total_points % PROCESS_EVERY_N_POINTS == 0:
                    now = time.time()
                    critical = [(a, d) for a, d in point_buffer if 0 < d <= ALERT_DISTANCE]

                    out = {}
                    if len(critical) >= MIN_POINTS_PER_OBJECT * 2:
                        groups = group_points_by_distance_and_angle(critical)
                        if groups:
                            merged = merge_object_groups(groups)
                            idx = 0
                            for obj in merged:
                                info = obj["info"]
                                if info and should_alert_for_object(info, alert_history):
                                    ang = float(info["avg_angle"])
                                    out[f"objeto_{idx}"] = {
                                        "angle": ang,
                                        "distance": float(info["min_distance"]),  # mm
                                        "region": region_from_lidar_angle_front(ang),
                                        "position": info["position"],
                                        "timestamp": float(info["timestamp"]),
                                    }
                                    idx += 1
                                    alert_history.append((now, info))

                    shared_state.update_lidar_objects(out)

                    alert_history = [(t, obj) for t, obj in alert_history if (now - t) < 10.0]
                    if len(point_buffer) > int(BUFFER_SIZE * 0.8):
                        point_buffer = point_buffer[-int(BUFFER_SIZE * 0.7):]

        except Exception as e:
            print(f"[Hilo - LIDAR] ERROR critico: {e}")
            time.sleep(1.0)

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

        time.sleep(0.8)


# --------------------------
# AUDIO/LLM THREAD (ARECORD + WHISPER) - FIXED
# --------------------------
def audio_llm_process(shared_state, button):
    print("[Hilo - Audio/LLM] Iniciando. Esperando activación por botón...")

    try:
        import whisper
        model = whisper.load_model(CONFIG["asistente"]["whisper_model"])
    except Exception as e:
        print(f"[Hilo - Audio/LLM] ERROR: Whisper no disponible: {e}")
        return

    def record_audio_arecord() -> str:
        """
        FIXES:
        - Usa -q (silencioso) pero si falla, imprime error.
        - Timeout duro para que nunca se quede colgado.
        - Duración con ceil.
        """
        alsa_dev = CONFIG["asistente"].get("alsa_capture_device", "plughw:2,0")
        rate = int(CONFIG["asistente"].get("audio_rate", 48000))
        channels = int(CONFIG["asistente"].get("audio_channels", 1))
        seconds = float(CONFIG["asistente"].get("record_seconds", 5))

        fd, wav_path = tempfile.mkstemp(prefix="mic_", suffix=".wav")
        os.close(fd)

        cmd = [
            "arecord",
            "-q",
            "-D", alsa_dev,
            "-f", "S16_LE",
            "-r", str(rate),
            "-c", str(channels),
            "-d", str(int(math.ceil(seconds))),
            wav_path,
        ]

        # timeout: duración + margen
        try:
            subprocess.run(
                cmd,
                check=True,
                timeout=seconds + 3.0,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("arecord timeout: se colgó el capture device.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"arecord falló: {e.stderr.strip() or e.stdout.strip() or str(e)}")

        # sanity: WAV válido
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 44:
            raise RuntimeError("WAV inválido: archivo muy pequeño o no se escribió.")

        return wav_path

    def load_wav_float32(path: str) -> tuple[np.ndarray, int]:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth != 2:
            raise RuntimeError(f"WAV no es 16-bit PCM (sampwidth={sampwidth})")

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        return audio, framerate

    def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
        # tu caso: 48000 -> 16000 (decimación x3)
        if sr == 48000:
            return audio[::3].astype(np.float32)
        if sr == 16000:
            return audio.astype(np.float32)

        # fallback interpolación lineal
        target_sr = 16000
        duration = len(audio) / float(sr)
        t_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        n_new = max(1, int(duration * target_sr))
        t_new = np.linspace(0.0, duration, num=n_new, endpoint=False)
        return np.interp(t_new, t_old, audio).astype(np.float32)

    def preguntar_llm(texto_usuario: str) -> str:
        payload = {
            "model": CONFIG["asistente"]["ollama_model"],
            "stream": False,
            "messages": [
                {"role": "system", "content": "Responde siempre en español y de forma breve y clara."},
                {"role": "user", "content": texto_usuario},
            ],
        }
        try:
            resp = requests.post(CONFIG["asistente"]["ollama_url"], json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except Exception as e:
            return f"No pude contactar el LLM: {e}"

    while True:
        button.wait_for_press()
        print("[Hilo - Audio/LLM] Botón presionado. Grabando audio...")

        try:
            wav_path = record_audio_arecord()
            print(f"[Hilo - Audio/LLM] WAV listo: {wav_path}")
        except Exception as e:
            print(f"[Hilo - Audio/LLM] ERROR grabando: {e}")
            time.sleep(0.3)
            continue

        try:
            audio, sr = load_wav_float32(wav_path)
            audio16 = resample_to_16k(audio, sr)

            print(f"[Hilo - Audio/LLM] Transcribiendo (sr={sr} -> 16000, samples={len(audio16)})...")
            result = model.transcribe(audio16, fp16=False, language="es")
            texto = (result.get("text") or "").strip()
        except Exception as e:
            print(f"[Hilo - Audio/LLM] ERROR whisper/transcribe: {e}")
            texto = ""
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

        if not texto:
            print("[Hilo - Audio/LLM] Transcripción vacía.")
            continue

        print(f"[Hilo - Audio/LLM] Usuario: {texto}")
        respuesta = preguntar_llm(texto)
        print(f"[Hilo - Audio/LLM] LLM: {respuesta}")
        if respuesta:
            shared_state.set_llm_response(respuesta)

        time.sleep(0.2)


# --------------------------
# TTS THREAD
# --------------------------
def tts_process(shared_state):
    print("[Hilo - TTS] Iniciando gestor de audio...")
    try:
        import pyttsx3
        engine = pyttsx3.init()
    except Exception as e:
        print(f"[Hilo - TTS] ERROR pyttsx3: {e}")
        return

    while True:
        action = shared_state.get("current_speech_action")

        if action == "proximity_alert":
            alert = shared_state.get("proximity_alert")
            if alert:
                text_to_speak = f"Alerta, {alert['name']} cerca a {alert['distance']:.2f} metros."
                print(f"[Hilo - TTS] {text_to_speak}")
                engine.say(text_to_speak)
                engine.runAndWait()
                shared_state.clear_speech_action()

        elif action == "llm_response":
            response = shared_state.get("llm_response")
            if response:
                print(f"[Hilo - TTS] {response}")
                engine.say(response)
                engine.runAndWait()
                shared_state.clear_speech_action()

        time.sleep(0.2)


# --------------------------
# MAIN (YOLO LOOP + CONSOLE DEBUG)
# --------------------------
def main():
    print("[Main] Iniciando sistema principal...")
    shared_state = SharedState()

    # YOLO + CSI camera (Picamera2)
    model = YOLO(CONFIG["yolo"]["model_path"], task="detect")
    labels = model.names

    picam2 = Picamera2()
    if CONFIG["yolo"].get("resolution"):
        resW, resH = map(int, CONFIG["yolo"]["resolution"].split("x"))
        cam_config = picam2.create_video_configuration(main={"size": (resW, resH), "format": "RGB888"})
    else:
        cam_config = picam2.create_video_configuration(main={"format": "RGB888"})
    picam2.configure(cam_config)
    picam2.start()
    time.sleep(1)

    button = Button(CONFIG["gpio"]["button_pin"])

    # threads
    threading.Thread(target=lidar_process, args=(shared_state,), daemon=True).start()
    threading.Thread(target=audio_llm_process, args=(shared_state, button), daemon=True).start()
    threading.Thread(target=tts_process, args=(shared_state,), daemon=True).start()

    print("[Main] Hilos secundarios iniciados. Iniciando bucle YOLO.")

    zone_colors = {"LEFT": (0, 0, 255), "CENTER": (0, 255, 0), "RIGHT": (255, 0, 0)}
    frame_rate_buffer = []
    fps_avg_len = 200

    debug_period = float(CONFIG.get("debug", {}).get("print_period_s", 0.5))
    last_debug = 0.0

    try:
        while True:
            t_start = time.perf_counter()

            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_h, frame_w, _ = frame.shape
            z = frame_w // 3

            results = model(frame, verbose=False)
            detections = results[0].boxes

            lidar_objects = shared_state.get("lidar_objects") or {}

            # LiDAR min por región (para consola)
            lidar_min_by_region = {"LEFT": None, "CENTER": None, "RIGHT": None}
            for obj in lidar_objects.values():
                reg = obj.get("region")
                dmm = obj.get("distance")
                if reg in lidar_min_by_region and dmm is not None:
                    prev = lidar_min_by_region[reg]
                    lidar_min_by_region[reg] = dmm if (prev is None or dmm < prev) else prev

            detected_objects = {}

            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                conf = float(detections[i].conf.item())

                if conf <= float(CONFIG["yolo"]["threshold"]):
                    continue

                cx = xmin + (xmax - xmin) // 2
                reg = region_from_x(cx, frame_w)

                # ángulo relativo a centro (tu fórmula)
                angle_rad = math.atan2(cx - (frame_w / 2), (frame_h / 2))
                angle_deg = float(math.degrees(angle_rad))

                # asociación LiDAR por ángulo
                min_distance_mm = float("inf")
                for obj_data in lidar_objects.values():
                    a = float(obj_data.get("angle", 0.0))
                    d = float(obj_data.get("distance", 0.0))
                    if d > 0 and abs(angle_deg - a) < 20:
                        if d < min_distance_mm:
                            min_distance_mm = d

                dist_m = (min_distance_mm / 1000.0) if min_distance_mm != float("inf") else 0.0

                key = f"{classname}_{i}"
                detected_objects[key] = {
                    "name": classname,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "region": reg,
                    "angle": angle_deg,
                    "distance_m": dist_m,
                    "conf": conf,
                }

                color = zone_colors[reg]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f"{classname} {int(conf*100)}%", (xmin, max(20, ymin - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            shared_state.update_detections(detected_objects)

            # proximidad
            for _, data in detected_objects.items():
                d = float(data["distance_m"])
                if 0 < d < float(CONFIG["sistema"]["min_proximity_distance"]):
                    print(f"[Main] Proximidad: {data['name']} ({d:.2f} m)")
                    shared_state.set_proximity_alert({"name": data["name"], "distance": d})
                    break

            # UI regiones
            cv2.line(frame, (z, 0), (z, frame_h), (255, 255, 0), 2)
            cv2.line(frame, (2 * z, 0), (2 * z, frame_h), (255, 255, 0), 2)

            # FPS
            t_stop = time.perf_counter()
            fps = float(1.0 / (t_stop - t_start))
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(fps)
            fps_avg = float(np.mean(frame_rate_buffer))
            cv2.putText(frame, f"FPS: {fps_avg:.2f}", (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Sistema Integrado", frame)
            if cv2.waitKey(1) == ord("q"):
                break

            # --------------------------
            # CONSOLE DEBUG (3 cosas)
            # --------------------------
            now = time.time()
            if (now - last_debug) >= debug_period:
                last_debug = now

                # 1) YOLO salida por región (etiquetas)
                yolo_by_region = {"LEFT": [], "CENTER": [], "RIGHT": []}
                for d in detected_objects.values():
                    yolo_by_region[d["region"]].append(d["name"])

                # 2) LiDAR por región (dist min)
                print("\n=== DEBUG ===")
                print(f"YOLO: LEFT={yolo_by_region['LEFT']} | CENTER={yolo_by_region['CENTER']} | RIGHT={yolo_by_region['RIGHT']}")
                print(f"LIDAR(min): LEFT={fmt_mm(lidar_min_by_region['LEFT'])} | CENTER={fmt_mm(lidar_min_by_region['CENTER'])} | RIGHT={fmt_mm(lidar_min_by_region['RIGHT'])}")

                # 3) Asociación YOLO<->LiDAR
                for k, d in detected_objects.items():
                    reg = d["region"]
                    print(
                        f"ASSOC: {k} -> region={reg} | yolo_angle={d['angle']:.1f}° | "
                        f"lidar_by_angle={d['distance_m']:.2f} m | lidar_min_region={fmt_mm(lidar_min_by_region[reg])}"
                    )

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
        print("\nSistema detenido por el usuario.")
        sys.exit(0)
