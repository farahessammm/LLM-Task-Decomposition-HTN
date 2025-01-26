import os
import time
import datetime
from groq import Groq
from ratelimit import limits, sleep_and_retry
from datetime import datetime


# Define the rate limit (10 calls per minute)
CALLS = 10
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_groq_api(prompt, max_tokens=None, temperature=0.7, strip=False):
    start_time = time.time()
    retries = 3
    delay = 5

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    while retries > 0:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                max_tokens=max_tokens,
                temperature=temperature
            )
            end_time = time.time()  # End timing the API call
            elapsed_time = end_time - start_time
            log_api_timing(prompt, elapsed_time) 
            return chat_completion.choices[0].message.content.strip() if strip else chat_completion
        except Exception as e:
            print(f"Error encountered: {e}. Retrying in {delay} seconds...")
            retries -= 1
            time.sleep(delay)

    raise Exception("Failed to get a response from Groq API after multiple retries.")

updated_log_files = {}


def log_response(function_name, response):
    global updated_log_files

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/{function_name}.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        if function_name not in updated_log_files:
            log_file.write("\n--- Application run start ---\n")
            updated_log_files[function_name] = True
        log_file.write(f"{timestamp}:\n{response}\n")

def log_api_timing(prompt, elapsed_time):
    """
    Log the time taken for each API call.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/api_timing.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp} - API Call Duration: {elapsed_time:.2f} seconds\n")
        log_file.write(f"Prompt: {prompt}\n\n")