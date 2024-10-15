import os
import json
from openai import OpenAI
import re

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Function to read JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [json.loads(line) for line in lines]

# Function to write JSONL file
def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

# Function to extract Python code block from the response
def extract_code(response):
    """
    Extracts the code block from the response.
    Handles cases where the code block starts with a language identifier.
    """
    code_match = re.search(r'```(?:\w+\n)?(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        return code
    else:
        # If no code block is found, return the original response trimmed
        return response.strip()

# Read the example_problem.jsonl file
problems = read_jsonl('/mnt/c/Users/egantemirov/IdeaProjects/Exxcellent-AI-Model-Eval/data/mxeval/HumanEval.jsonl')

# Process each problem to create a prompt and generate a response
samples = []
for problem in problems:
    # Create a prompt based on the problem
    prompt = problem['prompt']

    # Generate a response using the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    # Extract the completion text from the response
    completion = response.choices[0].message.content

    # Extract the code from the completion text
    code = extract_code(completion)

    # Create a new entry for example_samples.jsonl
    samples.append({
        "task_id": problem['task_id'],
        "language": problem['language'],  # Adding the language field
        "completion": code
    })

# Save the generated completions to example_samples.jsonl
write_jsonl('../samples/mxeval/python-mxeval.jsonl', samples)

print("Done!")
