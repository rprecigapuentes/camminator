import subprocess
import requests
import os
import time
from gtts import gTTS
import pygame

# ==== CONFIGURACION ====

# Rutas
WHISPER_DIR = "/home/grupo5/asistente_voz/whisper.cpp"
WHISPER_MODEL = os.path.join(WHISPER_DIR, "models", "ggml-base.bin")

INPUT_WAV = "/home/grupo5/asistente_voz/audio_entrada.wav"
OUTPUT_TXT = "/home/grupo5/asistente_voz/transcripcion.txt"
OUTPUT_MP3 = "/home/grupo5/asistente_voz/respuesta.mp3"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "tinyllama"

# Variable global para controlar el proceso de grabacin
arecord_process = None

# ==== FUNCIONES ====

def start_recording():
    global arecord_process
    if arecord_process is None:
        print("\n[+] INICIANDO GRABACIN... (Pulsa 'ENTER' para detener)")
        cmd = [
            "arecord",
            "-D", "hw:2,0",  # Asegrate de que este sea tu dispositivo correcto
            "-f", "cd",
            "-t", "wav",
            "-r", "16000",
            "-c", "1",
            INPUT_WAV,
        ]
        try:
            arecord_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[!] Error: 'arecord' no encontrado. Asegrate de que alsa-utils est instalado.")
            arecord_process = None

def stop_recording():
    global arecord_process
    if arecord_process is not None:
        print("[+] Deteniendo grabacin...")
        arecord_process.terminate()
        arecord_process.wait()
        arecord_process = None
        print("[+] Grabacin guardada en:", INPUT_WAV)
        return True
    else:
        print("[!] No hay ninguna grabaci en curso para detener.")
        return False


def transcribir():
    if os.path.exists(OUTPUT_TXT):
        os.remove(OUTPUT_TXT)

    print("[+] Transcribiendo con Whisper...")
    cmd = [
        os.path.join(WHISPER_DIR, "build", "bin", "whisper-cli"),
        "-m", WHISPER_MODEL,
        "-f", INPUT_WAV,
        "--language", "es", 
        "-otxt",
        "-of", OUTPUT_TXT.replace(".txt", "")
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        print("[!] Error: No se encontr el ejecutable de whisper-cli.")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"[!] Error al ejecutar Whisper: {e}")
        return ""

    if os.path.exists(OUTPUT_TXT):
        with open(OUTPUT_TXT, "r", encoding="utf-8", errors="ignore") as f:
            texto = f.read().strip()
            if texto:
                print("[+] Texto reconocido:", texto)
                return texto
            else:
                print("[!] No se escribitexto en el archivo.")
                return ""
    else:
        print("[!] El archivo de transcripcin no fue encontrado.")
        return ""
        

def preguntar_llm(texto_usuario: str) -> str:
    print("[+] Enviando texto a Ollama...")
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Responde siempre en espaol y de forma breve y clara."},
            {"role": "user", "content": texto_usuario}
        ]
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        respuesta = data["message"]["content"].strip()
        print("\n[+] Respuesta de la IA:")
        print(respuesta)
        return respuesta
    except requests.exceptions.RequestException as e:
        print(f"[!] Error al contactar con Ollama: {e}")
        return "Lo siento, no pude contactar al modelo de lenguaje."

def hablar(texto: str):
    print("\n[+] Generando respuesta de voz con gTTS...")
    try:
        tts = gTTS(text=texto, lang='es', slow=False)
        tts.save(OUTPUT_MP3)

        pygame.mixer.init()
        pygame.mixer.music.load(OUTPUT_MP3)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"[!] Error al generar o reproducir el audio: {e}")
        print("Mostrando el texto en su lugar:")
        print(texto)

# Bucle principal mejorado
def ciclo():
    print("\n--- Asistente de Voz Activo ---")
    print("Pulsa ENTER para iniciar grabacin y nuevamente para detener.")
    print("Escribe 'exit' para salir.")
    print("---------------------------------")

    while True:
        input("Pulsa ENTER para comenzar...")  # Comienza la grabacin al presionar ENTER
        start_recording()

        input("Presiona ENTER nuevamente para detener la grabacin...")  # Detn la grabacin al presionar ENTER otra vez
        if stop_recording():
            texto_usuario = transcribir()

            if not texto_usuario:
                print("[!] No se reconoci nada, intenta de nuevo.")
                continue

            if "salir" in texto_usuario.lower() or "adis" in texto_usuario.lower():
                print("[+] Comando de salida detectado.")
                hablar("Hasta luego!")
                break

            respuesta = preguntar_llm(texto_usuario)
            if respuesta:
                hablar(respuesta)

# ==== PROGRAMA PRINCIPAL ====

if __name__ == "__main__":
    ciclo()
