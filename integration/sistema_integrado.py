import threading
import time
import cv2
import numpy as np
import math
import os
import sys
import json
import base64
import argparse
import glob
import pyaudio
import requests
from prettytable import PrettyTable
from collections import Counter
from gpiozero import Button, LED
from ultralytics import YOLO
from picamera2 import Picamera2

# --- CONFIGURACIÓN GLOBAL ---
# Se asume que el archivo config.json está en el mismo directorio
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERROR: El archivo 'config.json' no fue encontrado. Asegúrate de que existe.")
    sys.exit(0)
except json.JSONDecodeError:
    print("ERROR: El archivo 'config.json' tiene un formato inválido.")
    sys.exit(0)


# --- ESTADO COMPARTIDO (Thread-Safe) ---
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            # Detecciones de YOLO: {'objeto1': {'bbox': (x,y,w,h), 'angle': a, 'distance': d}, ...}
            "detections": {},
            # Objeto detectados por el LIDAR: {'objeto_id': {'angle': a, 'distance': d}}
            "lidar_objects": {},
            # Última alerta de proximidad: {'name': 'objeto', 'distance': d}
            "proximity_alert": None,
            # Última respuesta del LLM para ser leída: "texto de la respuesta"
            "llm_response": None,
            # Control de prioridad de audio
            "current_speech_action": None,  # 'proximity_alert', 'llm_response', o None
            "stop_speech_flag": threading.Event(),  # Bandera para detener el audio inmediatamente
        }

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

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
            # Solo actualiza si no hay una alerta de proximidad activa
            if self._data.get("current_speech_action") != "proximity_alert":
                self._data["llm_response"] = response_text
                self._data["current_speech_action"] = "llm_response"

    def clear_speech_action(self):
        with self._lock:
            self._data["current_speech_action"] = None
            self._data["stop_speech_flag"].clear()


# --- DEFINICIÓN DE HILOS (Procesos Paralelos) ---
def lidar_process(shared_state):
    """
    Hilo 1: Ejecuta el escaneo del LIDAR de forma continua.
    Lógica integrada de 'DistanciaAgrupada.py'.
    """
    print("[Hilo - LIDAR] Iniciando...")
    try:
        from pyrplidar import PyRPLidar
    except ImportError:
        print("[Hilo - LIDAR] ERROR: La librería 'pyrplidar' no está instalada. pip install pyrplidar")
        return

    lidar = PyRPLidar()
    try:
        lidar.connect(
            port=CONFIG["lidar"]["port"],
            baudrate=CONFIG["lidar"]["baudrate"],
            timeout=3,
        )
        lidar.set_motor_pwm(CONFIG["lidar"]["motor_pwm"])
        time.sleep(2)
        scan_generator = lidar.force_scan()
    except Exception as e:
        print(f"[Hilo - LIDAR] Error al conectar o iniciar LIDAR: {e}")
        return

    try:
        while True:
            try:
                current_object = None
                object_points = []
                lidar_detections = {}

                for scan in scan_generator:
                    angle = scan.angle
                    distance = scan.distance
                    quality = scan.quality

                    if quality < 10 or distance == 0:
                        continue

                    angle_norm = angle % 360

                    in_front_sector = (
                        (CONFIG["lidar"]["front_angle_min1"] <= angle_norm <= CONFIG["lidar"]["front_angle_max1"])
                        or (CONFIG["lidar"]["front_angle_min2"] <= angle_norm <= CONFIG["lidar"]["front_angle_max2"])
                    )

                    if in_front_sector and distance <= CONFIG["lidar"]["alert_distance"]:
                        if not current_object:
                            current_object = {
                                "min_angle": angle_norm,
                                "max_angle": angle_norm,
                                "min_distance": distance,
                                "max_distance": distance,
                                "avg_distance": distance,
                                "count": 1,
                            }
                            object_points = [(angle_norm, distance)]
                        else:
                            last_angle = object_points[-1][0]
                            last_distance = object_points[-1][1]

                            if last_angle > 350 and angle_norm < 10:
                                angle_for_check = angle_norm + 360
                                last_angle_for_check = last_angle
                            else:
                                angle_for_check = angle_norm
                                last_angle_for_check = last_angle

                            angle_diff = abs(angle_for_check - last_angle_for_check)
                            distance_diff = abs(distance - last_distance)

                            if (
                                angle_diff <= CONFIG["lidar"]["angle_gap_threshold"]
                                and distance_diff <= CONFIG["lidar"]["distance_threshold"]
                            ):
                                object_points.append((angle_norm, distance))
                                current_object["min_angle"] = min(current_object["min_angle"], angle_norm)
                                current_object["max_angle"] = max(current_object["max_angle"], angle_norm)
                                current_object["min_distance"] = min(current_object["min_distance"], distance)
                                current_object["max_distance"] = max(current_object["max_distance"], distance)
                                total_distance = current_object["avg_distance"] * current_object["count"]
                                current_object["count"] += 1
                                current_object["avg_distance"] = (total_distance + distance) / current_object["count"]
                            else:
                                if current_object["count"] > 0:
                                    lidar_detections[f"objeto_{len(lidar_detections)}"] = {
                                        "angle": (current_object["min_angle"] + current_object["max_angle"]) / 2,
                                        "distance": current_object["min_distance"],
                                    }
                                current_object = {
                                    "min_angle": angle_norm,
                                    "max_angle": angle_norm,
                                    "min_distance": distance,
                                    "max_distance": distance,
                                    "avg_distance": distance,
                                    "count": 1,
                                }
                                object_points = [(angle_norm, distance)]

                if current_object and current_object["count"] > 0:
                    lidar_detections[f"objeto_{len(lidar_detections)}"] = {
                        "angle": (current_object["min_angle"] + current_object["max_angle"]) / 2,
                        "distance": current_object["min_distance"],
                    }

                shared_state.update_lidar_objects(lidar_detections)
                time.sleep(0.1)  # ~10 Hz

            except Exception as e:
                print(f"[Hilo - LIDAR] Error durante el escaneo: {e}")
                time.sleep(5)

    finally:
        print("[Hilo - LIDAR] Deteniendo LIDAR...")
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
        print("[Hilo - LIDAR] LIDAR detenido correctamente.")


def audio_llm_process(shared_state, button):
    """
    Hilo 2: Espera el disparador (botón), graba audio, lo transcribe y pregunta al LLM.
    Lógica integrada de 'asistente_vozv2.py'.
    """
    print("[Hilo - Audio/LLM] Iniciando. Esperando activación por botón...")
    try:
        import whisper

        model = whisper.load_model(CONFIG["asistente"]["whisper_model"])
    except ImportError:
        print("[Hilo - Audio/LLM] ERROR: La librería 'whisper' no está instalada. pip install openai-whisper")
        return

    def record_audio():
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CONFIG["asistente"]["audio_channels"],
            rate=CONFIG["asistente"]["audio_rate"],
            input=True,
            frames_per_buffer=CONFIG["asistente"]["audio_chunk"],
        )

        frames = []
        for _ in range(
            0,
            int(
                CONFIG["asistente"]["audio_rate"]
                / CONFIG["asistente"]["audio_chunk"]
                * CONFIG["asistente"]["record_seconds"]
            ),
        ):
            data = stream.read(CONFIG["asistente"]["audio_chunk"], exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        return b"".join(frames)

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
            respuesta = data["message"]["content"].strip()
            return respuesta
        except requests.exceptions.RequestException as e:
            print(f"[!] Error al contactar con Ollama: {e}")
            return "Lo siento, no pude contactar al modelo de lenguaje."

    while True:
        button.wait_for_press()
        print("[Hilo - Audio/LLM] Botón presionado. Iniciando proceso de voz...")

        audio_data = record_audio()

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = model.transcribe(audio_np, fp16=False, language="es")
        texto_transcrito = result["text"].strip()

        if texto_transcrito:
            print(f"Usuario pregunta: {texto_transcrito}")
            llm_response = preguntar_llm(texto_transcrito)
            if llm_response:
                shared_state.set_llm_response(llm_response)

        time.sleep(1)


def tts_process(shared_state):
    """
    Hilo 3: Gestiona la reproducción de audio (TTS) con prioridades.
    """
    print("[Hilo - TTS] Iniciando gestor de audio...")
    try:
        import pyttsx3

        engine = pyttsx3.init()
    except ImportError:
        print("[Hilo - TTS] ERROR: La librería 'pyttsx3' no está instalada. pip install pyttsx3")
        return

    while True:
        action = shared_state.get("current_speech_action")

        if action == "proximity_alert":
            alert = shared_state.get("proximity_alert")
            if alert:
                text_to_speak = f"Alerta, {alert['name']} cerca a {alert['distance']:.2f} metros."
                print(f"[Hilo - TTS] Hablando alerta: {text_to_speak}")
                engine.say(text_to_speak)
                engine.runAndWait()
                shared_state.clear_speech_action()

        elif action == "llm_response":
            response = shared_state.get("llm_response")
            if response:
                print(f"[Hilo - TTS] Hablando respuesta LLM: {response}")
                engine.say(response)
                engine.runAndWait()
                shared_state.clear_speech_action()

        time.sleep(0.5)


# --- FUNCIÓN PRINCIPAL (Hilo Principal) ---
def main():
    """
    Función principal que orquesta todos los hilos y ejecuta el bucle de YOLO.
    Lógica integrada de 'yolo_detect_seccionado.py'.
    """
    print("[Main] Iniciando sistema principal...")

    shared_state = SharedState()

    # Inicialización de YOLO y Cámara
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

    # Inicialización de GPIO
    button = Button(CONFIG["gpio"]["button_pin"])

    # Iniciar hilos secundarios
    threading.Thread(target=lidar_process, args=(shared_state,), daemon=True).start()
    threading.Thread(target=audio_llm_process, args=(shared_state, button), daemon=True).start()
    threading.Thread(target=tts_process, args=(shared_state,), daemon=True).start()

    print("[Main] Hilos secundarios iniciados. Iniciando bucle de detección YOLO.")

    # Colores para las zonas
    zone_colors = {"left": (0, 0, 255), "center": (0, 255, 0), "right": (255, 0, 0)}

    frame_rate_buffer = []
    fps_avg_len = 200
    avg_frame_rate = 0.0

    try:
        while True:
            t_start = time.perf_counter()

            frame = picam2.capture_array()
            # Picamera2 entrega RGB, OpenCV/YOLO esperan BGR
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
                conf = detections[i].conf.item()

                if conf > CONFIG["yolo"]["threshold"]:
                    bbox_center_x = xmin + (xmax - xmin) // 2

                    # Calcular ángulo relativo al centro de la imagen
                    angle_rad = math.atan2(bbox_center_x - (frame_w / 2), (frame_h / 2))
                    angle_deg = math.degrees(angle_rad)

                    # Asociar con el objeto más cercano del LIDAR
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

                    # Dibujar en la zona correspondiente
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

            # Actualizar detecciones en el estado compartido
            shared_state.update_detections(detected_objects)

            # Lógica de proximidad
            for obj_name, data in detected_objects.items():
                if 0 < data["distance"] < CONFIG["sistema"]["min_proximity_distance"]:
                    print(f"[Main] ¡Objeto {obj_name} en zona de proximidad! Disparando alerta.")
                    shared_state.set_proximity_alert({"name": obj_name, "distance": data["distance"]})
                    break

            # Dibujar zonas en el frame
            cv2.line(frame, (x_left_max, 0), (x_left_max, frame_h), (255, 255, 0), 2)
            cv2.line(frame, (x_center_max, 0), (x_center_max, frame_h), (255, 255, 0), 2)
            cv2.putText(frame, "IZQUIERDA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "CENTRO", (x_center_min + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "DERECHA", (x_right_min + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # FPS
            t_stop = time.perf_counter()
            frame_rate_calc = float(1 / (t_stop - t_start))
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            avg_frame_rate = float(np.mean(frame_rate_buffer))
            cv2.putText(frame, f"FPS: {avg_frame_rate:.2f}", (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
