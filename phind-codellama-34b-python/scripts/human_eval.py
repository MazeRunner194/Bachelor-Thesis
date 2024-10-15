
import json
import requests
import time
from datasets import load_dataset

# Funktion zum Schreiben von JSONL-Dateien
def write_jsonl(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

# Laden des Datasets von Huggingface
dataset = load_dataset("openai_humaneval")

# Extrahieren der Probleme und Prompts
problems = {
    f"HumanEval/{idx}": {"prompt": example["prompt"]}
    for idx, example in enumerate(dataset["test"])
}

model_name = "phind-codellama:34b-python"
server_url = 'http://192.168.144.243:11434/api/generate'
num_samples_per_task = 1  # Anzahl der Samples pro Aufgabe

# Funktion zur Generierung von Vorhersagen
def generate_predictions(prompt, server_url):
    data = {
        "model": model_name,
        "prompt": f"{prompt}",
        "stream": False
    }

    completion = ""
    start_time = time.time()  # Startzeit messen
    try:
        with requests.post(server_url, json=data, stream=True) as response:
            response.raise_for_status()  # Check for HTTP errors

            for line in response.iter_lines():
                if line:
                    response_json = json.loads(line.decode('utf-8'))
                    completion += response_json.get('response', '')

        end_time = time.time()  # Endzeit messen
        print(f"API call duration: {end_time - start_time} seconds")  # Dauer ausgeben
        return [completion]
    except requests.exceptions.RequestException as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return []
    except json.JSONDecodeError:
        print("Fehler beim Decodieren der JSON-Antwort. Rohantwort war:")
        print(response.text)
        return []

# Generierung der Samples
samples = [
    dict(task_id=task_id, completion=generate_predictions(problems[task_id]["prompt"], server_url)[0])
    for task_id in problems
    for _ in range(num_samples_per_task)
]

# Schreiben der Samples in eine JSONL-Datei
write_jsonl("samples-new.jsonl", samples)

print("Samples.jsonl wurde erfolgreich erstellt.")