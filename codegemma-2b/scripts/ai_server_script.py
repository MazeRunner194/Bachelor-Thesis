import json
import requests
import os
import time
import paramiko
import threading
import re

# Variables for model and file path
model_name = "codegemma:2b"
problems_file_path = '/mnt/c/Users/egantemirov/IdeaProjects/Exxcellent-AI-Model-Eval/data/human_eval/Part-1.jsonl'
num_samples_per_task = 1
samples_file_name= "human-eval-part1.jsonl"
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
        "stream": True,
        "verbose": True  # Include the verbose flag to get detailed metrics | Not relevant for the Bachelor's thesis
    }
    #Start the timer to measure the first token time
    start_time = time.time()
    first_token_time = None
    completion = ""
    metrics = {}
    try:
        with requests.post(server_url, json=data, stream=True) as response:
            response.raise_for_status()  # Check for HTTP errors

            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    response_json = json.loads(line.decode('utf-8'))
                    completion += response_json.get('response', '')

                    # Capture detailed metrics from Ollama response
                    if 'total_duration' in response_json:
                        metrics = {
                            "total_duration": response_json.get("total_duration"),
                            "load_duration": response_json.get("load_duration"),
                            "prompt_eval_count": response_json.get("prompt_eval_count"),
                            "prompt_eval_duration": response_json.get("prompt_eval_duration"),
                            "eval_count": response_json.get("eval_count"),
                            "eval_duration": response_json.get("eval_duration")
                        }

        latency = time.time() - start_time
        tokens = len(completion.split())
        return [completion], latency, tokens, first_token_time, metrics
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return [], 0, 0, None, {}
    except json.JSONDecodeError:
        print("Error decoding JSON response. Raw response was:")
        print(response.text)
        return [], 0, 0, None, {}

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

# Function to collect GPU metrics from the remote server
def get_remote_gpu_metrics(hostname, username, password, stop_event, gpu_metrics_list):
    command = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv,nounits,noheader"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password)

    while not stop_event.is_set():
        stdin, stdout, stderr = client.exec_command(command)
        response = stdout.read().decode()
        error = stderr.read().decode()

        if error:
            print(f"Error output from command: {error}")

        gpu_metrics = response.strip().split('\n')
        gpu_metrics = [list(map(int, metric.split(','))) for metric in gpu_metrics]
        if gpu_metrics:
            gpu_metrics_list.append(gpu_metrics[0])

    client.close()

def main():
    # URL of the Ollama API on the server
    server_url = 'http://192.168.144.243:11434/api/generate'

    # SSH connection details
    hostname = 'ki-server.corp.exxcellent.de'  # Server address
    username = 'egantemirov'  # Username
    password = os.getenv('SSH_PASSWORD')  # Retrieve password from environment variable

    if not password:
        print("Error: The password is not set as an environment variable 'SSH_PASSWORD'.")
        return

    # Path to the problems file
    problems = read_problems(problems_file_path)

    samples = []
    latencies = []
    first_token_times = []
    total_tokens = 0
    gpu_metrics_list = []
    ollama_metrics_list = []

    # Determine path to samples directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(script_dir, '../samples/human-eval')
    os.makedirs(samples_dir, exist_ok=True)
    samples_path = os.path.join(samples_dir, samples_file_name)

    # Start a separate thread to collect GPU metrics
    stop_event = threading.Event()
    gpu_thread = threading.Thread(target=get_remote_gpu_metrics, args=(hostname, username, password, stop_event, gpu_metrics_list))
    gpu_thread.start()

    try:
        # Generate predictions for each task
        for task_id in problems:
            prompt = problems[task_id]["prompt"]
            for _ in range(num_samples_per_task):
                predictions, latency, tokens, first_token_time, ollama_metrics = generate_predictions(prompt, server_url)
                for prediction in predictions:
                    code = extract_code(prediction)
                    samples.append({
                        "task_id": task_id,
                        "completion": code
                    })
                if latency > 0:  # Only count successful requests
                    latencies.append(latency)
                    total_tokens += tokens
                if first_token_time is not None:
                    first_token_times.append(first_token_time)

                # Save Ollama metrics
                if ollama_metrics:
                    ollama_metrics["task_id"] = task_id
                    ollama_metrics_list.append(ollama_metrics)

    finally:
        # Stop the GPU metrics thread
        stop_event.set()
        gpu_thread.join()

    # Save predictions to a JSONL file
    with open(samples_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    # Calculate performance metrics
    total_time = sum(latencies)
    average_latency = total_time / len(latencies) if latencies else 0  # TTLT
    average_first_token_time = sum(first_token_times) / len(first_token_times) if first_token_times else 0  # TTFT
    throughput = total_tokens / total_time if total_time > 0 else 0

    # Filter out 0 values for GPU utilization and calculate average
    valid_gpu_utils = [metric[0] for metric in gpu_metrics_list if metric[0] != 0]
    avg_gpu_util = sum(valid_gpu_utils) / len(valid_gpu_utils) if valid_gpu_utils else 0

    avg_mem_util = sum(metric[1] for metric in gpu_metrics_list) / len(gpu_metrics_list) if gpu_metrics_list else 0
    total_mem = gpu_metrics_list[0][2] if gpu_metrics_list else 0
    avg_mem_used = sum(metric[3] for metric in gpu_metrics_list) / len(gpu_metrics_list) if gpu_metrics_list else 0

    metrics = {
        "Total Processing Time": f"{total_time:.2f} seconds",
        "Average Latency per Request": f"{average_latency:.2f} seconds",
        "Average Time to First Token": f"{average_first_token_time:.2f} seconds",
        "Throughput": f"{throughput:.2f} tokens per second",
        "Average GPU Utilization": f"{avg_gpu_util:.2f}%",
        "Average Memory Utilization": f"{avg_mem_util:.2f}%",
        "Total Memory": f"{total_mem} MiB",
        "Average Memory Used": f"{avg_mem_used} MiB",
        "Ollama Metrics": ollama_metrics_list  # Include Ollama metrics in the final output
    }

    print(json.dumps(metrics, indent=4))

    # Determine path to GPU metrics directory
    gpu_metrics_dir = os.path.join(script_dir, '../gpu-metrics')
    os.makedirs(gpu_metrics_dir, exist_ok=True)
    gpu_metrics_path = os.path.join(gpu_metrics_dir, "gpu-metrics.json")

    # Save metrics to a JSON file
    with open(gpu_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f"Files {samples_file_name} and gpu-metrics.json generated successfully at {samples_path} and {gpu_metrics_path}.")

if __name__ == "__main__":
    main()
