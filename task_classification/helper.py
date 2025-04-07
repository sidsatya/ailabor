import os
import time
import pandas as pd
from openai import OpenAI
from typing import List, Tuple
import concurrent.futures
from tqdm import tqdm
import tenacity
from datetime import datetime
from dotenv import load_dotenv  
import tiktoken
from time import sleep

load_dotenv()  # Load environment variables from .env file
# Ensure the OpenAI API key is set in the environment

if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

def read_system_prompt(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()

class TokenRateLimiter:
    def __init__(self, tokens_per_minute=10000):
        self.tokens_per_minute = tokens_per_minute
        self.tokens_used = 0
        self.last_reset = time.time()
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    def estimate_tokens(self, messages):
        return sum(len(self.encoder.encode(msg["content"])) for msg in messages)
    
    def check_and_wait(self, new_tokens):
        current_time = time.time()
        # Reset counter if a minute has passed
        if current_time - self.last_reset >= 60:
            self.tokens_used = 0
            self.last_reset = current_time
        
        # If adding new tokens would exceed limit, wait
        if self.tokens_used + new_tokens > self.tokens_per_minute:
            sleep_time = 60 - (current_time - self.last_reset)
            if sleep_time > 0:
                print(f"\nRate limit approached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            self.tokens_used = 0
            self.last_reset = time.time()
        
        self.tokens_used += new_tokens

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=60, max=3600),
    stop=tenacity.stop_after_attempt(100),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: print(f"Retrying after {retry_state.next_action.sleep} seconds...")
)
def classify_task(client: OpenAI, system_prompt: str, task: str, iteration: int, rate_limiter: TokenRateLimiter) -> Tuple[str, str, int]:
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        # Check rate limit before making the API call
        estimated_tokens = rate_limiter.estimate_tokens(messages)
        rate_limiter.check_and_wait(estimated_tokens)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        return task, response.choices[0].message.content.strip(), iteration
    except Exception as e:
        print(f"Error classifying task: {str(e)}")
        raise

def append_to_results(df: pd.DataFrame, filepath: str, first_write: bool = False):
    mode = 'w' if first_write else 'a'
    header = first_write
    df.to_csv(filepath, mode=mode, header=header, index=False)

def process_batch(batch: pd.DataFrame, system_prompt: str, client: OpenAI, num_samples: int) -> pd.DataFrame:
    results = []
    rate_limiter = TokenRateLimiter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(classify_task, client, system_prompt, task, i, rate_limiter): (task, i)
            for task in batch['Task']
            for i in range(num_samples)
        }
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                task, label, iteration = future.result()
                results.append({
                    'Task': task, 
                    'gpt_label': label, 
                    'sample_num': iteration
                })
            except Exception as e:
                print(f"Task failed: {str(e)}")
    
    return pd.DataFrame(results)

def process_dataframe(df: pd.DataFrame
                      , system_prompt: str
                      , batch_size: int = 50
                      , num_samples: int = 1
                      , final_savepath = "classified_tasks_final.csv") -> pd.DataFrame:
    os.chdir(os.path.dirname(__file__))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create output directory
    output_dir = "data/intermediate_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create single intermediate file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_path = os.path.join(output_dir, f"classified_tasks_intermediate_{timestamp}.csv")
    
    # Process in batches
    all_results = []
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        
        # Process batch
        batch_results = process_batch(batch, system_prompt, client, num_samples)
        all_results.append(batch_results)
        
        # Append results to intermediate file
        append_to_results(batch_results, intermediate_path, first_write=(i==0))
        
        # Small delay between batches
        time.sleep(1)
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save final results (copy intermediate to final location)
    final_path = os.path.join("data", final_savepath)
    final_df.to_csv(final_path, index=False)
    print(f"Saved final results to {final_path}")
    
    return final_df

def main():
    # Read system prompt
    system_prompt = read_system_prompt('prompts/system_level_prompt.txt')
    
    # Read data
    onet_data = pd.read_csv("data/task_statements.csv")
    
    # Process the data
    classified_data = process_dataframe(onet_data, system_prompt, batch_size=50, num_samples=3)
    print("Classification complete!")
    return classified_data