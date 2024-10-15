
import json
import requests
import os

model_name = "codegemma:2b"
prompt = "\ndef generate_integers(a, b):\n    \"\"\"\n    Given two positive integers a and b, return the even digits between a\n    and b, in ascending order.\n\n    For example:\n    generate_integers(2, 8) => [2, 4, 6, 8]\n    generate_integers(8, 2) => [2, 4, 6, 8]\n    generate_integers(10, 14) => []\n    \"\"\"\n"
server_url = 'http://192.168.144.243:11434/api/generate'
timeout_seconds = 15  # Timeout duration in seconds

# Function to generate a prediction directly from the Ollama API on the server
def generate_prediction(prompt, server_url):
    data = {
        "model": model_name,
        "prompt": f"{prompt}",
        "stream": False  # Turn off streaming
    }

    try:
        with requests.post(server_url, json=data, timeout=timeout_seconds) as response:
            response.raise_for_status()  # Check for HTTP errors

            raw_output = response.text  # Get the raw text of the response

            print("Raw API response:")
            print(raw_output)  # Print the raw response to the terminal

            response_json = response.json()
            text_output = response_json.get('response', '')

            print("\nExtracted text output:")
            print(text_output)  # Print the text output to the terminal

            return text_output
    except requests.exceptions.Timeout:
        print(f"Timeout occurred after {timeout_seconds} seconds for prompt: {prompt}")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return ""
    except json.JSONDecodeError:
        print("Fehler beim Decodieren der JSON-Antwort. Rohantwort war:")
        print(raw_output)
        return ""

# Generate a prediction and print the raw output
completion = generate_prediction(prompt, server_url)
