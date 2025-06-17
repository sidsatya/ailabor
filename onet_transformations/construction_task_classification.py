import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

datapath = '/Users/sidsatya/dev/ailabor/task_classification/data/intermediate_results/classified_tasks_16_dim_intermediate.csv'
data = pd.read_csv(datapath)

def read_gpt_label(label):
    # label is formatted as a JSON string with 10 key-value pairs
    label = label.replace("'", '"')  # Replace single quotes with double quotes
    label = label.replace('nan', 'null')  # Replace 'nan' with 'null' for JSON compatibility
    try:
        label_dict = eval(label)  # Use eval to convert the string to a dictionary
        return label_dict
    except Exception as e:
        print(f"Error parsing label: {e}")
        return {}

data['read_label'] = data['gpt_label'].apply(read_gpt_label)
data['interpersonal'] = data['read_label'].apply(lambda x: 1 if x.get('Interpersonal', 'No') == 'Yes' else 0)
data['routine'] = data['read_label'].apply(lambda x: 1 if x.get('Routine', 'No') == 'Yes' else 0)
data['manual'] = data['read_label'].apply(lambda x: 1 if x.get('Manual', 'No') == 'Yes' else 0)
data['high_codifiable'] = data['read_label'].apply(lambda x: 1 if x.get('High Cod.', 'No') == 'Yes' else 0)

# Group by Task and compute mode and mean for each category
grouped_data = data.groupby('Task').agg({
    'interpersonal': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
    'routine': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
    'manual': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
    'high_codifiable': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0]
}).reset_index()

# collapse the multi-level columns into dimension_mean and dimension_mode
grouped_data.columns = ['Task', 'interpersonal_mean', 'interpersonal_mode',
                        'routine_mean', 'routine_mode',
                        'manual_mean', 'manual_mode',
                        'high_codifiable_mean', 'high_codifiable_mode']



# Merge with the overall task dataset 
all_task_datapath = '/Users/sidsatya/dev/ailabor/onet_transformations/intermediate_data/task_data_merged_attributes.csv'
all_task_data = pd.read_csv(all_task_datapath)

# Merge all_data with grouped data on 'Task'
merged_data = pd.merge(all_task_data, grouped_data, on='Task', how='left')
print("Shape of data once merging with all task data: ", merged_data.shape)

# Routine-Cognitive: any task with routine = 1 and manual = 0
# Routine-Manual: any task with routine = 1 and manual = 1
# Non-Routine Manual: any task with routine = 0 and manual = 1
# Non-Routine Interpersonal: any task with routine = 0 and interpersonal = 1
# Non Routine Analytical: any task with routine = 0 and manual = 0 and interpersonal = 0
# Highly Codifiable: any task with high_codifiable = 1
# Non Highly Codifiable: any task with high_codifiable = 0
# Label each task based on the above criteria and using the 'mode' labels and assign them to new columns
def label_task_alm(row):
    first_letter = 'X'
    second_letter = 'X'
    third_letter ='X'
    if row['routine_mode'] == 1: 
        first_letter = 'R'
    else: 
        first_letter = 'NR'
    
    if row['manual_mode'] == 1:
        second_letter = 'M'
    else:
        second_letter = 'NM'
    
    if row['interpersonal_mode'] == 1:
        third_letter = 'I'
    else:
        third_letter = 'P'
    
    return '-'.join([first_letter, second_letter, third_letter])
    
def label_task_codifiable(row):
    first_letter = 'X'
    second_letter = 'X'
    third_letter ='X'
    if row['high_codifiable_mode'] == 1: 
        first_letter = 'HC'
    else: 
        first_letter = 'LC'
    
    if row['manual_mode'] == 1:
        second_letter = 'M'
    else:
        second_letter = 'NM'
    
    if row['interpersonal_mode'] == 1:
        third_letter = 'I'
    else:
        third_letter = 'P'
    
    return '-'.join([first_letter, second_letter, third_letter])
    
merged_data['task_label_alm'] = merged_data.apply(label_task_alm, axis=1)
merged_data['task_label_codifiable'] = merged_data.apply(label_task_codifiable, axis=1)

obs_with_mode_and_intensity = merged_data[merged_data['task_intensity'].notna()]

##### SAVE ALL TASKS ##### 
# Get all Routine Tasks, sorted descending by task intensity
routine_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['routine_mode'] == 1].sort_values(by='task_intensity', ascending=False)
routine_tasks['Task'].to_csv('/Users/sidsatya/dev/ailabor/results/alm_classification_results/routine_tasks.txt', index=False)
# Get all Non-Routine Tasks, sorted descending by task intensity
non_routine_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['routine_mode'] == 0].sort_values(by='task_intensity', ascending=False)
non_routine_tasks['Task'].to_csv('/Users/sidsatya/dev/ailabor/results/alm_classification_results/non_routine_tasks.txt', index=False)
# Get all Manual Tasks, sorted descending by task intensity
manual_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['manual_mode'] == 1].sort_values(by='task_intensity', ascending=False)
manual_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['manual_mode'] == 1].sort_values(by='task_intensity', ascending=False)
manual_tasks['Task'].to_csv('/Users/sidsatya/dev/ailabor/results/alm_classification_results/manual_tasks.txt', index=False)
# Get all Non-Manual Tasks, sorted descending by task intensity
non_manual_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['manual_mode'] == 0].sort_values(by='task_intensity', ascending=False)
non_manual_tasks['Task'].to_csv('/Users/sidsatya/dev/ailabor/results/alm_classification_results/non_manual_tasks.txt', index=False)
# Get all Interpersonal Tasks, sorted descending by task intensity
interpersonal_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['interpersonal_mode'] == 1].sort_values(by='task_intensity', ascending=False)
interpersonal_tasks['Task'].to_csv('/Users/sidsatya/dev/ailabor/results/alm_classification_results/interpersonal_tasks.txt', index=False)
# Get all Non-Interpersonal Tasks, sorted descending by task intensity
non_interpersonal_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['interpersonal_mode'] == 0].sort_values(by='task_intensity', ascending=False)
non_interpersonal_tasks['Task'].to_csv('/Users/sidsatya/dev/ailabor/results/alm_classification_results/non_interpersonal_tasks.txt', index=False)
# Get all Highly Codifiable Tasks, sorted descending by task intensity
highly_codifiable_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['high_codifiable_mode'] == 1].sort_values(by='task_intensity', ascending=False)
highly_codifiable_tasks['Task'].to_csv('/Users/sidsatya/dev/ailabor/results/alm_classification_results/highly_codifiable_tasks.txt', index=False, header=False)
# Get all Non-Highly Codifiable Tasks, sorted descending by task intensity
non_highly_codifiable_tasks = obs_with_mode_and_intensity[obs_with_mode_and_intensity['high_codifiable_mode'] == 0].sort_values(by='task_intensity', ascending=False)
non_highly_codifiable_tasks['Task'].to_csv('/Users/sidsatya/dev/ailabor/results/alm_classification_results/low_codifiable_tasks.txt', index=False)

# Group by O*NET SOC Code and ONET_release_year, aggregating the task labels and task intensity
# Sum task_intensity for each task_label within each group
alm_task_intensity_by_label = obs_with_mode_and_intensity.pivot_table(
    index=['O*NET-SOC Code', 'O*NET 2010 SOC Code', 'O*NET 2018 SOC Code', 'ONET_release_year'],
    columns='task_label_alm',
    values='task_intensity',
    aggfunc='sum',
    fill_value=0
)
cod_task_intensity_by_label = obs_with_mode_and_intensity.pivot_table(
    index=['O*NET-SOC Code', 'O*NET 2010 SOC Code', 'O*NET 2018 SOC Code', 'ONET_release_year'],
    columns='task_label_codifiable',
    values='task_intensity',
    aggfunc='sum',
    fill_value=0
)

# Reset index to flatten the DataFrame
alm_task_intensity_by_label = alm_task_intensity_by_label.reset_index()
alm_task_intensity_by_label.columns = ['O*NET-SOC Code', 'O*NET 2010 SOC Code', 'O*NET 2018 SOC Code', 'ONET_release_year'] + list(alm_task_intensity_by_label.columns[4:])

cod_task_intensity_by_label = cod_task_intensity_by_label.reset_index()
cod_task_intensity_by_label.columns = ['O*NET-SOC Code', 'O*NET 2010 SOC Code', 'O*NET 2018 SOC Code', 'ONET_release_year'] + list(cod_task_intensity_by_label.columns[4:])


# Merge with Healthcare and Overall BLS data
bls_full_datapath = '/Users/sidsatya/dev/ailabor/bls_transformations/output_data/oes_data_filtered_soc_2018.csv'
bls_healthcare_datapath = '/Users/sidsatya/dev/ailabor/bls_transformations/output_data/oes_data_filtered_healthcare_soc_2018.csv'
bls_full_emp_data = pd.read_csv(bls_full_datapath)
bls_healthcare_emp_data = pd.read_csv(bls_healthcare_datapath)

# Merge BLS data with the full aggregated task intensity data
merged_bls_data_alm = pd.merge(alm_task_intensity_by_label, bls_full_emp_data, left_on=['O*NET 2018 SOC Code', 'ONET_release_year'], right_on = ['soc_2018', 'bls_release_year'], how='inner')
merged_bls_data_cod = pd.merge(cod_task_intensity_by_label, bls_full_emp_data, left_on=['O*NET 2018 SOC Code', 'ONET_release_year'], right_on = ['soc_2018', 'bls_release_year'], how='inner')

# Merge BLS healthcare data with the full aggregated task intensity data
merged_bls_data_alm_healthcare = pd.merge(alm_task_intensity_by_label, bls_healthcare_emp_data, left_on=['O*NET 2018 SOC Code', 'ONET_release_year'], right_on = ['soc_2018', 'bls_release_year'], how='inner')
merged_bls_data_cod_healthcare = pd.merge(cod_task_intensity_by_label, bls_healthcare_emp_data, left_on=['O*NET 2018 SOC Code', 'ONET_release_year'], right_on = ['soc_2018', 'bls_release_year'], how='inner')

######################
###### FIGS ##########
######################

## region: Full ALM Graph
# Routine combinations
merged_bls_data_alm['R-M-P_weighted'] = merged_bls_data_alm['R-M-P'] * merged_bls_data_alm['pct_year_tot_emp']
merged_bls_data_alm['R-M-I_weighted'] = merged_bls_data_alm['R-M-I'] * merged_bls_data_alm['pct_year_tot_emp']
merged_bls_data_alm['R-NM-P_weighted'] = merged_bls_data_alm['R-NM-P'] * merged_bls_data_alm['pct_year_tot_emp']
merged_bls_data_alm['R-NM-I_weighted'] = merged_bls_data_alm['R-NM-I'] * merged_bls_data_alm['pct_year_tot_emp']

# Non-routine combinations
merged_bls_data_alm['NR-M-P_weighted'] = merged_bls_data_alm['NR-M-P'] * merged_bls_data_alm['pct_year_tot_emp']
merged_bls_data_alm['NR-M-I_weighted'] = merged_bls_data_alm['NR-M-I'] * merged_bls_data_alm['pct_year_tot_emp']
merged_bls_data_alm['NR-NM-P_weighted'] = merged_bls_data_alm['NR-NM-P'] * merged_bls_data_alm['pct_year_tot_emp']
merged_bls_data_alm['NR-NM-I_weighted'] = merged_bls_data_alm['NR-NM-I'] * merged_bls_data_alm['pct_year_tot_emp']


# Sum the total weighted task intensities for each dimension in each year. Normalize wrt to sum across all dimension in each year
total_weighted_intensity = merged_bls_data_alm.groupby('ONET_release_year').agg({
    'NR-M-P_weighted': 'sum',
    'NR-M-I_weighted': 'sum',
    'NR-NM-P_weighted': 'sum',
    'NR-NM-I_weighted': 'sum',
    'R-M-P_weighted': 'sum',
    'R-M-I_weighted': 'sum',
    'R-NM-P_weighted': 'sum',
    'R-NM-I_weighted': 'sum'
}).reset_index()
total_weighted_intensity['total_weighted_intensity'] = total_weighted_intensity[['NR-M-P_weighted', 
                                                                                'NR-M-I_weighted', 
                                                                                'NR-NM-P_weighted', 
                                                                                'NR-NM-I_weighted', 
                                                                                'R-M-P_weighted', 
                                                                                'R-M-I_weighted', 
                                                                                'R-NM-P_weighted', 
                                                                                'R-NM-I_weighted']].sum(axis=1)
total_weighted_intensity = total_weighted_intensity.set_index('ONET_release_year')
total_weighted_intensity = total_weighted_intensity.div(total_weighted_intensity['total_weighted_intensity'], axis=0)  

# Plot the the share corresponding to each dimension over the years, line plot
plt.figure(figsize=(12, 6))
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-M-P_weighted'], label='Non Routine-Manual-Personal', marker='o', color='red')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-M-I_weighted'], label='Non Routine-Manual-Interpersonal', marker='o', color='blue')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-NM-P_weighted'], label='Non Routine-Manual', marker='o', color='black')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['R-M-P_weighted'], label='Routine-Manual-Personal', marker='o', color='green')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['R-M-I_weighted'], label='Routine-Manual-Interpersonal', marker='o', color='orange')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['R-NM-P_weighted'], label='Routine-Non Manual-Personal', marker='o', color='purple')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['R-NM-I_weighted'], label='Routine-Non Manual-Interpersonal', marker='o', color='pink')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-NM-P_weighted'], label='Non Routine-Non Manual-Personal', marker='o', color='brown')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-NM-I_weighted'], label='Non Routine-Non Manual-Interpersonal', marker='o', color='cyan')
plt.title('Share of Task Intensity by Dimension Over the Years')
plt.xlabel('Year')
plt.ylabel('Share of Total Task Intensity')
plt.xticks(total_weighted_intensity.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/sidsatya/dev/ailabor/results/alm_classification_results/full_alm_task_intensity_share_over_years.png')

## region: Full Codifiable Graph
# HC combinations
merged_bls_data_cod['HC-M-P_weighted'] = merged_bls_data_cod['HC-M-P'] * merged_bls_data_cod['pct_year_tot_emp']
merged_bls_data_cod['HC-M-I_weighted'] = merged_bls_data_cod['HC-M-I'] * merged_bls_data_cod['pct_year_tot_emp']
merged_bls_data_cod['HC-NM-P_weighted'] = merged_bls_data_cod['HC-NM-P'] * merged_bls_data_cod['pct_year_tot_emp']
merged_bls_data_cod['HC-NM-I_weighted'] = merged_bls_data_cod['HC-NM-I'] * merged_bls_data_cod['pct_year_tot_emp']

# LC combinations
merged_bls_data_cod['LC-M-P_weighted'] = merged_bls_data_cod['LC-M-P'] * merged_bls_data_cod['pct_year_tot_emp']
merged_bls_data_cod['LC-M-I_weighted'] = merged_bls_data_cod['LC-M-I'] * merged_bls_data_cod['pct_year_tot_emp']
merged_bls_data_cod['LC-NM-P_weighted'] = merged_bls_data_cod['LC-NM-P'] * merged_bls_data_cod['pct_year_tot_emp']
merged_bls_data_cod['LC-NM-I_weighted'] = merged_bls_data_cod['LC-NM-I'] * merged_bls_data_cod['pct_year_tot_emp']


# Sum the total weighted task intensities for each dimension in each year. Normalize wrt to sum across all dimension in each year
total_weighted_intensity = merged_bls_data_cod.groupby('ONET_release_year').agg({
    'HC-M-P_weighted': 'sum',
    'HC-M-I_weighted': 'sum',
    'HC-NM-P_weighted': 'sum',
    'HC-NM-I_weighted': 'sum',
    'LC-M-P_weighted': 'sum',
    'LC-M-I_weighted': 'sum',
    'LC-NM-P_weighted': 'sum',
    'LC-NM-I_weighted': 'sum'
}).reset_index()
total_weighted_intensity['total_weighted_intensity'] = total_weighted_intensity[['HC-M-P_weighted', 
                                                                                'HC-M-I_weighted', 
                                                                                'HC-NM-P_weighted', 
                                                                                'HC-NM-I_weighted', 
                                                                                'LC-M-P_weighted', 
                                                                                'LC-M-I_weighted', 
                                                                                'LC-NM-P_weighted', 
                                                                                'LC-NM-I_weighted']].sum(axis=1)
total_weighted_intensity = total_weighted_intensity.set_index('ONET_release_year')
total_weighted_intensity = total_weighted_intensity.div(total_weighted_intensity['total_weighted_intensity'], axis=0)  

# Plot the the share corresponding to each dimension over the years, line plot
plt.figure(figsize=(12, 6))
plt.plot(total_weighted_intensity.index, total_weighted_intensity['HC-M-P_weighted'], label='High Codifiable-Manual-Personal', marker='o', color='red')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['HC-M-I_weighted'], label='High Codifiable-Manual-Interpersonal', marker='o', color='blue')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['HC-NM-P_weighted'], label='High Codifiable-Non Manual-Personal', marker='o', color='black')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['LC-M-P_weighted'], label='Low Codifiable-Manual-Personal', marker='o', color='green')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['LC-M-I_weighted'], label='Low Codifiable-Manual-Interpersonal', marker='o', color='orange')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['LC-NM-P_weighted'], label='Low Codifiable-Non Manual-Personal', marker='o', color='purple')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['LC-NM-I_weighted'], label='Low Codifiable-Non Manual-Interpersonal', marker='o', color='pink')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['HC-NM-I_weighted'], label='High Codifiable-Non Manual-Interpersonal', marker='o', color='cyan')
plt.title('Share of Task Intensity by Dimension Over the Years')
plt.xlabel('Year')
plt.ylabel('Share of Total Task Intensity')
plt.xticks(total_weighted_intensity.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/sidsatya/dev/ailabor/results/alm_classification_results/full_codifiable_task_intensity_share_over_years.png')


## region: Full Healthcare Graph
# Routine combinations
merged_bls_data_alm_healthcare['R-M-P_weighted'] = merged_bls_data_alm_healthcare['R-M-P'] * merged_bls_data_alm_healthcare['pct_year_tot_emp']
merged_bls_data_alm_healthcare['R-M-I_weighted'] = merged_bls_data_alm_healthcare['R-M-I'] * merged_bls_data_alm_healthcare['pct_year_tot_emp']
merged_bls_data_alm_healthcare['R-NM-P_weighted'] = merged_bls_data_alm_healthcare['R-NM-P'] * merged_bls_data_alm_healthcare['pct_year_tot_emp']
merged_bls_data_alm_healthcare['R-NM-I_weighted'] = merged_bls_data_alm_healthcare['R-NM-I'] * merged_bls_data_alm_healthcare['pct_year_tot_emp']

# Non-routine combinations
merged_bls_data_alm_healthcare['NR-M-P_weighted'] = merged_bls_data_alm_healthcare['NR-M-P'] * merged_bls_data_alm_healthcare['pct_year_tot_emp']
merged_bls_data_alm_healthcare['NR-M-I_weighted'] = merged_bls_data_alm_healthcare['NR-M-I'] * merged_bls_data_alm_healthcare['pct_year_tot_emp']
merged_bls_data_alm_healthcare['NR-NM-P_weighted'] = merged_bls_data_alm_healthcare['NR-NM-P'] * merged_bls_data_alm_healthcare['pct_year_tot_emp']
merged_bls_data_alm_healthcare['NR-NM-I_weighted'] = merged_bls_data_alm_healthcare['NR-NM-I'] * merged_bls_data_alm_healthcare['pct_year_tot_emp']


# Sum the total weighted task intensities for each dimension in each year. Normalize wrt to sum across all dimension in each year
total_weighted_intensity = merged_bls_data_alm_healthcare.groupby('ONET_release_year').agg({
    'NR-M-P_weighted': 'sum',
    'NR-M-I_weighted': 'sum',
    'NR-NM-P_weighted': 'sum',
    'NR-NM-I_weighted': 'sum',
    'R-M-P_weighted': 'sum',
    'R-M-I_weighted': 'sum',
    'R-NM-P_weighted': 'sum',
    'R-NM-I_weighted': 'sum'
}).reset_index()
total_weighted_intensity['total_weighted_intensity'] = total_weighted_intensity[['NR-M-P_weighted', 
                                                                                'NR-M-I_weighted', 
                                                                                'NR-NM-P_weighted', 
                                                                                'NR-NM-I_weighted', 
                                                                                'R-M-P_weighted', 
                                                                                'R-M-I_weighted', 
                                                                                'R-NM-P_weighted', 
                                                                                'R-NM-I_weighted']].sum(axis=1)
total_weighted_intensity = total_weighted_intensity.set_index('ONET_release_year')
total_weighted_intensity = total_weighted_intensity.div(total_weighted_intensity['total_weighted_intensity'], axis=0)  

# Plot the the share corresponding to each dimension over the years, line plot
plt.figure(figsize=(12, 6))
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-M-P_weighted'], label='Non Routine-Manual-Personal', marker='o', color='red')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-M-I_weighted'], label='Non Routine-Manual-Interpersonal', marker='o', color='blue')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-NM-P_weighted'], label='Non Routine-Manual', marker='o', color='black')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['R-M-P_weighted'], label='Routine-Manual-Personal', marker='o', color='green')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['R-M-I_weighted'], label='Routine-Manual-Interpersonal', marker='o', color='orange')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['R-NM-P_weighted'], label='Routine-Non Manual-Personal', marker='o', color='purple')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['R-NM-I_weighted'], label='Routine-Non Manual-Interpersonal', marker='o', color='pink')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-NM-P_weighted'], label='Non Routine-Non Manual-Personal', marker='o', color='brown')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['NR-NM-I_weighted'], label='Non Routine-Non Manual-Interpersonal', marker='o', color='cyan')
plt.title('Share of Task Intensity by Dimension Over the Years')
plt.xlabel('Year')
plt.ylabel('Share of Total Task Intensity')
plt.xticks(total_weighted_intensity.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/sidsatya/dev/ailabor/results/alm_classification_results/healthcare_alm_task_intensity_share_over_years.png')

## region: Healthcare Codifiable Graph
# HC combinations
merged_bls_data_cod_healthcare['HC-M-P_weighted'] = merged_bls_data_cod_healthcare['HC-M-P'] * merged_bls_data_cod_healthcare['pct_year_tot_emp']
merged_bls_data_cod_healthcare['HC-M-I_weighted'] = merged_bls_data_cod_healthcare['HC-M-I'] * merged_bls_data_cod_healthcare['pct_year_tot_emp']
merged_bls_data_cod_healthcare['HC-NM-P_weighted'] = merged_bls_data_cod_healthcare['HC-NM-P'] * merged_bls_data_cod_healthcare['pct_year_tot_emp']
merged_bls_data_cod_healthcare['HC-NM-I_weighted'] = merged_bls_data_cod_healthcare['HC-NM-I'] * merged_bls_data_cod_healthcare['pct_year_tot_emp']

# LC combinations
merged_bls_data_cod_healthcare['LC-M-P_weighted'] = merged_bls_data_cod_healthcare['LC-M-P'] * merged_bls_data_cod_healthcare['pct_year_tot_emp']
merged_bls_data_cod_healthcare['LC-M-I_weighted'] = merged_bls_data_cod_healthcare['LC-M-I'] * merged_bls_data_cod_healthcare['pct_year_tot_emp']
merged_bls_data_cod_healthcare['LC-NM-P_weighted'] = merged_bls_data_cod_healthcare['LC-NM-P'] * merged_bls_data_cod_healthcare['pct_year_tot_emp']
merged_bls_data_cod_healthcare['LC-NM-I_weighted'] = merged_bls_data_cod_healthcare['LC-NM-I'] * merged_bls_data_cod_healthcare['pct_year_tot_emp']


# Sum the total weighted task intensities for each dimension in each year. Normalize wrt to sum across all dimension in each year
total_weighted_intensity = merged_bls_data_cod_healthcare.groupby('ONET_release_year').agg({
    'HC-M-P_weighted': 'sum',
    'HC-M-I_weighted': 'sum',
    'HC-NM-P_weighted': 'sum',
    'HC-NM-I_weighted': 'sum',
    'LC-M-P_weighted': 'sum',
    'LC-M-I_weighted': 'sum',
    'LC-NM-P_weighted': 'sum',
    'LC-NM-I_weighted': 'sum'
}).reset_index()
total_weighted_intensity['total_weighted_intensity'] = total_weighted_intensity[['HC-M-P_weighted', 
                                                                                'HC-M-I_weighted', 
                                                                                'HC-NM-P_weighted', 
                                                                                'HC-NM-I_weighted', 
                                                                                'LC-M-P_weighted', 
                                                                                'LC-M-I_weighted', 
                                                                                'LC-NM-P_weighted', 
                                                                                'LC-NM-I_weighted']].sum(axis=1)
total_weighted_intensity = total_weighted_intensity.set_index('ONET_release_year')
total_weighted_intensity = total_weighted_intensity.div(total_weighted_intensity['total_weighted_intensity'], axis=0)  

# Plot the the share corresponding to each dimension over the years, line plot
plt.figure(figsize=(12, 6))
plt.plot(total_weighted_intensity.index, total_weighted_intensity['HC-M-P_weighted'], label='High Codifiable-Manual-Personal', marker='o', color='red')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['HC-M-I_weighted'], label='High Codifiable-Manual-Interpersonal', marker='o', color='blue')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['HC-NM-P_weighted'], label='High Codifiable-Non Manual-Personal', marker='o', color='black')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['LC-M-P_weighted'], label='Low Codifiable-Manual-Personal', marker='o', color='green')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['LC-M-I_weighted'], label='Low Codifiable-Manual-Interpersonal', marker='o', color='orange')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['LC-NM-P_weighted'], label='Low Codifiable-Non Manual-Personal', marker='o', color='purple')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['LC-NM-I_weighted'], label='Low Codifiable-Non Manual-Interpersonal', marker='o', color='pink')
plt.plot(total_weighted_intensity.index, total_weighted_intensity['HC-NM-I_weighted'], label='High Codifiable-Non Manual-Interpersonal', marker='o', color='cyan')
plt.title('Share of Task Intensity by Dimension Over the Years')
plt.xlabel('Year')
plt.ylabel('Share of Total Task Intensity')
plt.xticks(total_weighted_intensity.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/sidsatya/dev/ailabor/results/alm_classification_results/healthcare_codifiable_task_intensity_share_over_years.png')


######## USING 2003 EMPLOYMENT AS THE BASELINE ####### 
# 1) Identify the occupations that exist in 2003 and their 2003 shares
occs_2003 = merged_bls_data_alm.loc[
    merged_bls_data_alm.ONET_release_year == 2003, ["O*NET 2018 SOC Code", "pct_year_tot_emp"]
].rename(columns={"pct_year_tot_emp": "pct_2003_emp"})

# Helper to prepare any merged_bls_data_* DataFrame
def prepare(df, save_subdir):
    # keep only occs in 2003
    df = df.merge(occs_2003, on="O*NET 2018 SOC Code", how="inner")
    
    # use pct_2003_emp as the weight instead of the year-specific share
    df["weight"] = df["pct_2003_emp"]
    
    # compute weighted intensities for each of the 8 ALM cells
    cells = ['R-M-P','R-M-I','R-NM-P','R-NM-I','NR-M-P','NR-M-I','NR-NM-P','NR-NM-I']
    for cell in cells:
        df[f"{cell}_w"] = df[cell] * df["weight"]
    
    # aggregate and normalize
    agg = df.groupby("ONET_release_year")[
        [f"{c}_w" for c in cells]
    ].sum().reset_index()
    agg["total"] = agg[[f"{c}_w" for c in cells]].sum(axis=1)
    agg = agg.set_index("ONET_release_year")
    shares = agg.div(agg["total"], axis=0)
    
    # make sure output directory exists
    out_dir = f"/Users/sidsatya/dev/ailabor/results/alm_classification_results/alm_2003base/{save_subdir}"
    os.makedirs(out_dir, exist_ok=True)
    return shares, out_dir

# --- Full ALM Graph ---
shares_alm, out_alm = prepare(merged_bls_data_alm, "full_alm")
plt.figure(figsize=(12,6))
for cell,color in zip(
    ['R-M-P','R-M-I','R-NM-P','R-NM-I','NR-M-P','NR-M-I','NR-NM-P','NR-NM-I'],
    ['green','orange','purple','pink','red','blue','black','brown']
):
    plt.plot(shares_alm.index, shares_alm[f"{cell}_w"], label=cell, marker='o', color=color)
plt.title("ALM Task Shares Indexed to 2003 Employment Weights")
plt.xlabel("Year")
plt.ylabel("Share of Task Intensity")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"{out_alm}/full_alm_2003base.png")

# --- Full Codifiable Graph ---
shares_cod, out_cod = prepare(merged_bls_data_cod, "full_codifiable")
# if your codifiable cells are named HC-M-P etc, replace cells list accordingly
cod_cells = ['HC-M-P','HC-M-I','HC-NM-P','HC-NM-I','LC-M-P','LC-M-I','LC-NM-P','LC-NM-I']
cod_colors = ['red','blue','black','cyan','green','orange','purple','pink']
plt.figure(figsize=(12,6))
for cell,color in zip(cod_cells, cod_colors):
    plt.plot(shares_cod.index, shares_cod[f"{cell}_w"], label=cell, marker='o', color=color)
plt.title("Codifiable Task Shares Indexed to 2003 Employment Weights")
plt.xlabel("Year")
plt.ylabel("Share of Task Intensity")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"{out_cod}/full_cod_2003base.png")

# --- Full Healthcare ALM Graph ---
shares_hc_alm, out_hc_alm = prepare(merged_bls_data_alm_healthcare, "healthcare_alm")
plt.figure(figsize=(12,6))
for cell,color in zip(
    ['R-M-P','R-M-I','R-NM-P','R-NM-I','NR-M-P','NR-M-I','NR-NM-P','NR-NM-I'],
    ['green','orange','purple','pink','red','blue','black','brown']
):
    plt.plot(shares_hc_alm.index, shares_hc_alm[f"{cell}_w"], label=cell, marker='o', color=color)
plt.title("Healthcare ALM Task Shares Indexed to 2003 Employment Weights")
plt.xlabel("Year")
plt.ylabel("Share of Task Intensity")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"{out_hc_alm}/hc_alm_2003base.png")

# --- Healthcare Codifiable Graph ---
shares_hc_cod, out_hc_cod = prepare(merged_bls_data_cod_healthcare, "healthcare_codifiable")
plt.figure(figsize=(12,6))
for cell,color in zip(cod_cells, cod_colors):
    plt.plot(shares_hc_cod.index, shares_hc_cod[f"{cell}_w"], label=cell, marker='o', color=color)
plt.title("Healthcare Codifiable Task Shares Indexed to 2003 Employment Weights")
plt.xlabel("Year")
plt.ylabel("Share of Task Intensity")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"{out_hc_cod}/hc_cod_2003base.png")
