import json
import requests
import os
import time
import paramiko
import threading
import re

# Variables for model and file path
model_name = "codeqwen:7b-chat"
problems_file_path = '/mnt/c/Users/egantemirov/IdeaProjects/Exxcellent-AI-Model-Eval/data/human_eval/problems.jsonl'
num_samples_per_task = 1
samples_file_name = "human-eval.jsonl"
timeout_seconds = 50  # Timeout duration in seconds

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
def generate_predictions(prompt, server_url, timeout_seconds):
    data = {
        "model": model_name,
        "prompt": f"{prompt}",
        "stream": True
    }
    start_time = time.time()
    first_token_time = None
    completion = ""

    def fetch_response():
        nonlocal completion, first_token_time
        try:
            with requests.post(server_url, json=data, stream=True) as response:
                response.raise_for_status()  # Check for HTTP errors
                for line in response.iter_lines():
                    if line:
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                        response_json = json.loads(line.decode('utf-8'))
                        completion += response_json.get('response', '')
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON response. Raw response was:")
            print(response.text)

    thread = threading.Thread(target=fetch_response)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print(f"Timeout occurred after {timeout_seconds} seconds for prompt: {prompt}")
        return [], 0, 0, None
    else:
        latency = time.time() - start_time
        tokens = len(completion.split())
        print(f"API call duration: {latency:.2f} seconds")
        return [completion], latency, tokens, first_token_time

def extract_code(response):
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
    server_url = 'http://192.168.144.243:11434/api/generate'
    hostname = 'ki-server.corp.exxcellent.de'  # Server address
    username = 'egantemirov'  # Username
    password = os.getenv('SSH_PASSWORD')  # Retrieve password from environment variable

    if not password:
        print("Error: The password is not set as an environment variable 'SSH_PASSWORD'.")
        return

    problems = read_problems(problems_file_path)
    samples = []
    latencies = []
    first_token_times = []
    total_tokens = 0
    gpu_metrics_list = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(script_dir, '../samples/human-eval')
    os.makedirs(samples_dir, exist_ok=True)
    samples_path = os.path.join(samples_dir, samples_file_name)

    stop_event = threading.Event()
    gpu_thread = threading.Thread(target=get_remote_gpu_metrics, args=(hostname, username, password, stop_event, gpu_metrics_list))
    gpu_thread.start()

    try:
        for task_id in problems:
            prompt = problems[task_id]["prompt"]
            for _ in range(num_samples_per_task):
                time.sleep(6)
                predictions, latency, tokens, first_token_time = generate_predictions(prompt, server_url, timeout_seconds)
                for prediction in predictions:
                    code = extract_code(prediction)
                    samples.append({
                        "task_id": task_id,
                        "completion": code
                    })
                if latency > 0:
                    latencies.append(latency)
                    total_tokens += tokens
                if first_token_time is not None:
                    first_token_times.append(first_token_time)
    finally:
        stop_event.set()
        gpu_thread.join()

    with open(samples_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    total_time = sum(latencies)
    average_latency = total_time / len(latencies) if latencies else 0
    average_first_token_time = sum(first_token_times) / len(first_token_times) if first_token_times else 0
    throughput = total_tokens / total_time if total_time > 0 else 0

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
        "Average Memory Used": f"{avg_mem_used} MiB"
    }

    print(json.dumps(metrics, indent=4))

    gpu_metrics_dir = os.path.join(script_dir, '../gpu-metrics')
    os.makedirs(gpu_metrics_dir, exist_ok=True)
    gpu_metrics_path = os.path.join(gpu_metrics_dir, "gpu-metrics.json")

    with open(gpu_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f"Files {samples_file_name} and gpu-metrics.json generated successfully at {samples_path} and {gpu_metrics_path}.")

if __name__ == "__main__":
    main()

