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
load_dotenv()  # Load environment variables from .env file
# Ensure the OpenAI API key is set in the environment


if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

def read_system_prompt(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: print(f"Retrying after {retry_state.next_action.sleep} seconds...")
)
def classify_task(client: OpenAI, system_prompt: str, task: str) -> Tuple[str, str]:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task}
            ],
            temperature=0
        )
        return task, response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error classifying task: {str(e)}")
        raise

def save_batch(df: pd.DataFrame, batch_num: int, output_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"classified_tasks_batch_{batch_num}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved batch {batch_num} to {filepath}")

def process_batch(batch: pd.DataFrame, system_prompt: str, client: OpenAI) -> pd.DataFrame:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(classify_task, client, system_prompt, task): task 
            for task in batch['Task']
        }
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                task, label = future.result()
                results.append({'Task': task, 'gpt_label': label})
            except Exception as e:
                print(f"Task failed: {str(e)}")
    
    return pd.DataFrame(results)

def process_dataframe(df: pd.DataFrame, system_prompt: str, batch_size: int = 50) -> pd.DataFrame:
    os.chdir(os.path.dirname(__file__))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create output directory for batches
    output_dir = "data/intermediate_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in batches
    all_results = []
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        
        # Process batch
        batch_results = process_batch(batch, system_prompt, client)
        all_results.append(batch_results)
        
        # Save intermediate results
        current_results = pd.concat(all_results, ignore_index=True)
        save_batch(current_results, i + 1, output_dir)
        
        # Small delay between batches
        time.sleep(1)
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join("data", f"classified_tasks_final_{timestamp}.csv")
    final_df.to_csv(final_path, index=False)
    print(f"Saved final results to {final_path}")
    
    return final_df

def main():
    # Read system prompt
    system_prompt = read_system_prompt('prompts/system_level_prompt.txt')
    
    # Read data
    onet_data = pd.read_csv("data/task_statements.csv")
    
    # Process the data
    classified_data = process_dataframe(onet_data, system_prompt, batch_size=50)
    print("Classification complete!")
    return classified_data

if __name__ == "__main__":
    main()