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
    sys.exit(0)
except json.JSONDecodeError:
    print("ERROR: El archivo 'config.json' tiene un formato inválido.")
    sys.exit(0)


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
            "current_speech_action": None,  # 'proximity_alert', 'llm_response', None
            "stop_speech_flag": threading.Event(),
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
            self._data["stop_speech_flag"].set()

    def set_llm_response(self, response_text):
        with self._lock:
            if self._data.get("current_speech_action") != "proximity_alert":
                self._data["llm_response"] = response_text
                self._data["current_speech_action"] = "llm_response"

    def clear_speech_action(self):
        with self._lock:
            self._data["current_speech_action"] = None
            self._data["stop_speech_flag"].clear()


# --------------------------
# LIDAR THREAD (SMART + AUTO-RECOVERY)
# --------------------------
def lidar_process(shared_state):
    """
    Hilo LIDAR: copia la lógica smart del standalone (buffer+agrupación+fusión),
    publica objetos en shared_state.update_lidar_objects().

    Incluye auto-recovery si el LIDAR se desincroniza (sync bytes mismatched).
    """
    print("[Hilo - LIDAR] Iniciando LIDAR (modo smart + auto-recovery)...")

    from pyrplidar import PyRPlidar

    # Helpers
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

        # promedio circular
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

        if angle_span > 0.1 and avg_d > 0:
            width_est = 2 * avg_d * math.tan(math.radians(angle_span / 2))
        else:
            width_est = 0.0

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

    # Params
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

    # Auto-recovery outer loop
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
                                    out[f"objeto_{idx}"] = {
                                        "angle": float(info["avg_angle"]),
                                        "distance": float(info["min_distance"]),  # mm
                                        "position": info["position"],
                                        "width_est": float(info["width_est"]),
                                        "angle_span": float(info["angle_span"]),
                                        "point_count": int(info["point_count"]),
                                        "timestamp": float(info["timestamp"]),
                                    }
                                    idx += 1
                                    alert_history.append((now, info))

                    shared_state.update_lidar_objects(out)

                    # housekeeping
                    alert_history = [(t, obj) for t, obj in alert_history if (now - t) < 10.0]
                    if len(point_buffer) > int(BUFFER_SIZE * 0.8):
                        point_buffer = point_buffer[-int(BUFFER_SIZE * 0.7):]

        except Exception as e:
            print(f"[Hilo - LIDAR] ERROR critico: {e}")
            # recovery wait
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

        # reintenta
        time.sleep(0.8)


# --------------------------
# AUDIO/LLM THREAD (ARECORD + WHISPER)
# --------------------------
def audio_llm_process(shared_state, button):
    print("[Hilo - Audio/LLM] Iniciando. Esperando activación por botón...")

    try:
        import whisper
        model = whisper.load_model(CONFIG["asistente"]["whisper_model"])
    except ImportError:
        print("[Hilo - Audio/LLM] ERROR: falta whisper. pip install openai-whisper")
        return

    def record_audio_arecord() -> str:
        """
        Graba WAV con arecord usando el device ALSA explícito (plughw:2,0 / hw:2,0).
        """
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

        # Silenciar salida para que no ensucie logs
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path

    def load_wav_float32(path: str) -> np.ndarray:
        """
        Lee WAV 16-bit PCM y devuelve float32 mono en [-1,1].
        (sin dependencias externas)
        """
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

    def preguntar_llm(texto_usuario):
        print("[+] Enviando texto a Ollama...")
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
        except requests.exceptions.RequestException as e:
            print(f"[!] Error al contactar con Ollama: {e}")
            return "Lo siento, no pude contactar al modelo de lenguaje."

    while True:
        button.wait_for_press()
        print("[Hilo - Audio/LLM] Botón presionado. Grabando con arecord...")

        try:
            wav_path = record_audio_arecord()
        except Exception as e:
            print(f"[Hilo - Audio/LLM] ERROR grabando con arecord: {e}")
            time.sleep(0.5)
            continue

        try:
            audio_np = load_wav_float32(wav_path)
            result = model.transcribe(audio_np, fp16=False, language="es")
            texto = result.get("text", "").strip()
        except Exception as e:
            print(f"[Hilo - Audio/LLM] ERROR en Whisper/transcripción: {e}")
            texto = ""
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

        if texto:
            print(f"Usuario pregunta: {texto}")
            llm_response = preguntar_llm(texto)
            if llm_response:
                shared_state.set_llm_response(llm_response)

        time.sleep(0.2)


# --------------------------
# TTS THREAD
# --------------------------
def tts_process(shared_state):
    print("[Hilo - TTS] Iniciando gestor de audio...")
    try:
        import pyttsx3
        engine = pyttsx3.init()
    except ImportError:
        print("[Hilo - TTS] ERROR: falta pyttsx3. pip install pyttsx3")
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
# MAIN (YOLO LOOP)
# --------------------------
def main():
    print("[Main] Iniciando sistema principal...")

    shared_state = SharedState()

    # YOLO + camera
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

    # GPIO
    button = Button(CONFIG["gpio"]["button_pin"])

    # threads
    threading.Thread(target=lidar_process, args=(shared_state,), daemon=True).start()
    threading.Thread(target=audio_llm_process, args=(shared_state, button), daemon=True).start()
    threading.Thread(target=tts_process, args=(shared_state,), daemon=True).start()

    print("[Main] Hilos secundarios iniciados. Iniciando bucle YOLO.")

    zone_colors = {"left": (0, 0, 255), "center": (0, 255, 0), "right": (255, 0, 0)}
    frame_rate_buffer = []
    fps_avg_len = 200

    try:
        while True:
            t_start = time.perf_counter()

            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_h, frame_w, _ = frame.shape
            zone_width = frame_w // 3
            x_left_max = zone_width
            x_center_min = zone_width
            x_center_max = 2 * zone_width
            x_right_min = 2 * zone_width

            results = model(frame, verbose=False)
            detections = results[0].boxes
            detected_objects = {}

            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                conf = float(detections[i].conf.item())

                if conf > float(CONFIG["yolo"]["threshold"]):
                    bbox_center_x = xmin + (xmax - xmin) // 2

                    angle_rad = math.atan2(bbox_center_x - (frame_w / 2), (frame_h / 2))
                    angle_deg = math.degrees(angle_rad)

                    lidar_objects = shared_state.get("lidar_objects") or {}
                    min_distance = float("inf")

                    for obj_data in lidar_objects.values():
                        if abs(angle_deg - obj_data["angle"]) < 20:
                            if obj_data["distance"] < min_distance:
                                min_distance = obj_data["distance"]

                    detected_objects[classname] = {
                        "bbox": (xmin, ymin, xmax, ymax),
                        "angle": angle_deg,
                        "distance": (min_distance / 1000.0) if min_distance != float("inf") else 0.0,
                    }

                    if bbox_center_x < x_left_max:
                        color = zone_colors["left"]
                    elif bbox_center_x < x_center_max:
                        color = zone_colors["center"]
                    else:
                        color = zone_colors["right"]

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    label = f"{classname}: {int(conf*100)}%"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(
                        frame,
                        (xmin, label_ymin - labelSize[1] - 10),
                        (xmin + labelSize[0], label_ymin + baseLine - 10),
                        color,
                        cv2.FILLED,
                    )
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            shared_state.update_detections(detected_objects)

            for obj_name, data in detected_objects.items():
                if 0 < data["distance"] < float(CONFIG["sistema"]["min_proximity_distance"]):
                    print(f"[Main] Objeto {obj_name} en proximidad. Disparando alerta.")
                    shared_state.set_proximity_alert({"name": obj_name, "distance": data["distance"]})
                    break

            cv2.line(frame, (x_left_max, 0), (x_left_max, frame_h), (255, 255, 0), 2)
            cv2.line(frame, (x_center_max, 0), (x_center_max, frame_h), (255, 255, 0), 2)
            cv2.putText(frame, "IZQUIERDA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "CENTRO", (x_center_min + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "DERECHA", (x_right_min + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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
