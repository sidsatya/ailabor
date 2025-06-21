import pandas as pd
import os
from helper import read_system_prompt, process_dataframe

os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

### BEFORE RUNNING THIS SCRIPT, MAKE SURE TO SET UP THE ENVIRONMENT VARIABLES
num_samples = 3
prompt_path = "prompts/system_level_prompt_felten.txt"
intermediate_savepath = "intermediate_results/classified_tasks_felten_intermediate.csv"
final_savepath = "classified_tasks_felten.csv"

onet_output = "../data/onet/onet_task_statements_classified_felten.csv"

# Healthcare occupations only
datapath = "/Users/sidsatya/dev/ailabor/onet_transformations/intermediate_data/unique_task_statements_healthcare.csv"
onet_data_healthcare_occs = pd.read_csv(datapath)


if not os.path.exists(os.path.join("data", final_savepath)):
    print("No final classification file found. Proceeding with classification...")

    # Filter out already classified tasks in the intermediate file. Enforce that there must be at least num_samples samples
    if os.path.exists(os.path.join("data", intermediate_savepath)):
        classified_tasks = pd.read_csv(os.path.join("data", intermediate_savepath))
        classified_tasks = classified_tasks[classified_tasks['gpt_label'].notna()]
        classified_tasks = classified_tasks.groupby('Task').filter(lambda x: len(x) >= num_samples)
        unclassified_tasks = onet_data_healthcare_occs[~onet_data_healthcare_occs['Task'].isin(classified_tasks['Task'])]
        print(f"Unclassified tasks yet to be classified: {len(unclassified_tasks)}")
    else: 
        unclassified_tasks = onet_data_healthcare_occs
        print(f"All tasks yet to be classified: {len(unclassified_tasks)}")

    # Read system prompt
    system_prompt = read_system_prompt(prompt_path)
   
    # Run the classification
    classified_data = process_dataframe(unclassified_tasks
                                        , system_prompt
                                        , batch_size=2
                                        , num_samples=3
                                        , intermediate_savepath=intermediate_savepath
                                        , final_savepath=final_savepath)

    # Display results
    classified_data.head()

written_data = pd.read_csv(os.path.join("data", intermediate_savepath)) 

written_data_grped = written_data.groupby('Task').agg({'gpt_label': lambda x: x.mode()[0],  # Get the most common classification
                                                                'Task': 'first'  # Keep the first task statement
                                                                }).reset_index(drop=True) 

# merge with healthcare onet data joining on task_statements
onet_healthcare_merged = pd.merge(onet_data_healthcare_occs, written_data_grped, on='Task', how='inner')

# save the final grouped data
onet_healthcare_merged.to_csv(onet_output, index=False)