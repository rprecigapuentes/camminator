import pyaudio
import whisper
import requests
import numpy as np
import threading  # Para manejar la grabacin y transcripcin en hilos separados

# Configuracin de PyAudio
rate = 48000  # Tasa de muestreo
chunk_size = 256  # Tamao del bloque de audio
channels = 1  # Mono
format = pyaudio.paInt16  # Formato de audio

# Cargar el modelo de Whisper
model = whisper.load_model("base")  # Puedes usar "small", "medium", "large" segn tu necesidad

# Configuracin de Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "tinyllama"  # Modelo a utilizar en Ollama

# Variables de control
is_recording = False
audio_stream = None
p = None
# Funcin para enviar el texto a Ollama y obtener una respuesta
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

# Funcin para capturar audio y transcribir en tiempo real
def transcribir_en_tiempo_real():
    global is_recording, audio_stream, p
    # Inicializamos PyAudio para grabar
    p = pyaudio.PyAudio()
    audio_stream = p.open(format=format,
                          channels=channels,
                          rate=rate,
                          input=True,
                          frames_per_buffer=chunk_size)
    
    print("[+] Grabando y transcribiendo en tiempo real...")

    while is_recording:
        # Captura de un bloque de audio
        audio_data = audio_stream.read(chunk_size)
        
        # Convertir el audio a formato numpy
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / 32768.0  # Normalizar el audio para que est entre -1 y 1

        # Transcripcin de audio
        result = model.transcribe(audio_np, language="es")  # Especificamos el idioma espaol
        texto_transcrito = result['text'].strip()
        if texto_transcrito:
            print(f"Texto transcrito: {texto_transcrito}")

            # Si el texto transcrito no est vaco, lo enviamos a Ollama
            if texto_transcrito.lower() == "salir" or texto_transcrito.lower() == "adi":
                print("[+] Comando de salida detectado.")
                stop_recording()  # Detener la grabacin si se detecta el comando de salida

            # Enviar el texto a Ollama
            respuesta = preguntar_llm(texto_transcrito)
            if respuesta:
                print(f"[+] Respuesta de Ollama: {respuesta}")

# Funcin para iniciar la grabacin
def start_recording():
    global is_recording
    if not is_recording:
        is_recording = True
        threading.Thread(target=transcribir_en_tiempo_real).start()  # Ejecutar la transcripcin en tiempo real en un hilo separado
        print("[+] Grabacin iniciada...")

# Funcin para detener la grabacin
def stop_recording():
    global is_recording
    if is_recording:
        is_recording = False
        if audio_stream:
            audio_stream.stop_stream()
            audio_stream.close()
        if p:
            p.terminate()
        print("[+] Grabacin detenida.")
    else:
        print("[!] No hay grabacin en curso.")
# Funcin para gestionar el control del usuario
def ciclo():
    print("\n--- Asistente de Voz Activo ---")
    print("Pulsa 'r' para iniciar la grabacin y 's' para detenerla.")
    print("Escribe 'exit' para salir.")
    print("---------------------------------")

    while True:
        comando = input("Escribe tu comando: ")
        if comando.lower() == "r":
            start_recording()  # Iniciar la grabacin
        elif comando.lower() == "s":
            stop_recording()  # Detener la grabacin
        elif comando.lower() == "exit":
            print("[+] Salida del asistente.")
            stop_recording()  # Asegurar que la grabacin se detenga antes de salir
            break
        else:
            print("[!] Comando no reconocido. Usa 'r' para grabar, 's' para detener o 'exit' para salir.")

# Ejecutar el programa
if __name__ == "__main__":
    ciclo()
