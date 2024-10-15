import json
import requests
import os
import re

model_name = "codegemma:2b"
problems_file_path = '/mnt/c/Users/egantemirov/IdeaProjects/Exxcellent-AI-Model-Eval/data/mxeval/python/English.jsonl'
num_samples_per_task = 1
samples_file_name= "mxeval-python.jsonl"
# Function to read problems from a JSONL file
def read_problems(file_path):
    problems = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            problem = json.loads(line.strip())
            task_id = problem.pop('task_id')
            problems[task_id] = problem
    return problems

# Function to generate predictions directly from the Ollama API on the server
def generate_predictions(prompt, server_url):
    data = {
        "model": model_name,
        "prompt": f"{prompt}",
        "stream": True
    }

    completion = ""
    try:
        with requests.post(server_url, json=data, stream=True) as response:
            response.raise_for_status()  # Check for HTTP errors

            for line in response.iter_lines():
                if line:
                    response_json = json.loads(line.decode('utf-8'))
                    completion += response_json.get('response', '')

        return [completion]
    except requests.exceptions.RequestException as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return []
    except json.JSONDecodeError:
        print("Fehler beim Decodieren der JSON-Antwort. Rohantwort war:")
        print(response.text)
        return []

def extract_code(response):
    """
    Extrahiert den Python-Codeblock aus der Antwort.
    Geht davon aus, dass der Code in dreifachen Backticks (```) eingeschlossen ist.
    """
    code_match = re.search(r'```(?:python|cpp|javascript|typescript|java)?(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        return code
    else:
        return response.strip()

def main():
    # URL der Ollama API auf dem Server
    server_url = 'http://192.168.144.243:11434/api/generate'

    # SSH-Verbindungsdetails
    hostname = 'ki-server.corp.exxcellent.de'  # Serveradresse
    username = 'egantemirov'  # Benutzername
    password = os.getenv('SSH_PASSWORD')  # Passwort aus der Umgebungsvariable abrufen

    if not password:
        print("Fehler: Das Passwort ist nicht als Umgebungsvariable 'SSH_PASSWORD' gesetzt.")
        return

    # Pfad zur Probleme-Datei
    problems = read_problems(problems_file_path)

    samples = []

    # Pfad zum Samples-Verzeichnis bestimmen
    script_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(script_dir, '../samples/human-eval-xl')
    os.makedirs(samples_dir, exist_ok=True)
    samples_path = os.path.join(samples_dir, samples_file_name)

    try:
        # Vorhersagen für jede Aufgabe generieren
        for task_id in problems:
            prompt = problems[task_id]["prompt"]
            language = problems[task_id].get("language", "unknown")
            for _ in range(num_samples_per_task):
                predictions = generate_predictions(prompt, server_url)
                for prediction in predictions:
                    code = extract_code(prediction)
                    samples.append({
                        "task_id": task_id,
                        "language": language,
                        "completion": code
                    })

    finally:
        # Vorhersagen in eine JSONL-Datei speichern
        with open(samples_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

    print(f"Files samples.jsonl generated successfully at {samples_path}.")

if __name__ == "__main__":
    main()
