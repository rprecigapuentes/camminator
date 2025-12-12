import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "tinyllama",  # o el modelo que tengas: p.ej. "phi3" o "llama3"
    "stream": False,
    "messages": [
        {
            "role": "user",
            "content": "Hola,¿quien eres?"
	}
     ]
}
resp = requests.post(url, json=payload)

print("Status:", resp.status_code)
print("Respuesta cruda:")
print(resp.text)
try:
    data = resp.json()
    print("\nSolo el contenido del mensaje:")
    print(data["message"]["content"])
except Exception as e:
    print("No se pudo parsear JSON:", e)
