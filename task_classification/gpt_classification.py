import pandas as pd
import os
from helper import read_system_prompt, process_dataframe

os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

# Healthcare occupations only
onet_data_healthcare_occs = pd.read_csv("../data/onet/onet_task_statements_healthcare_industries.csv")

if not os.path.exists("data/classified_tasks_healthcare_industry.csv"):
    print("No final classification file found. Proceeding with classification...")

    # Read system prompt
    system_prompt = read_system_prompt('prompts/system_level_prompt_2.txt')

    # Process the DataFrame
    classified_data = process_dataframe(onet_data_healthcare_occs
                                        , system_prompt
                                        , batch_size=2
                                        , num_samples=3
                                        , final_savepath="classified_tasks_healthcare_industry.csv")

    # Display results
    classified_data.head()

written_data = pd.read_csv("data/classified_tasks_healthcare_industry.csv") 

written_data_grped = written_data.groupby('Task').agg({'gpt_label': lambda x: x.mode()[0],  # Get the most common classification
                                                                'Task': 'first'  # Keep the first task statement
                                                                }).reset_index(drop=True) 

# merge with healthcare onet data joiningo n task_statement
onet_healthcare_merged = pd.merge(onet_data_healthcare_occs, written_data_grped, on='Task', how='inner')

# save the final grouped data
onet_healthcare_merged.to_csv("../data/onet/onet_task_statements_classified.csv", index=False)