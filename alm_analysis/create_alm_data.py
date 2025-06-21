import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


ailabor_root = '/Users/sidsatya/dev/ailabor/'

# IMPORT the Classified Tasks data
datapath = os.path.join(ailabor_root, 'task_classification/data/classified_tasks_16_dim.csv')
classified_data = pd.read_csv(datapath)

def clean_text(text: str) -> str:
    # 2) remove any occurrences of "x92"
    text = re.sub(r'\x92', "'", text)
    # 1) replace any punctuation (i.e. non-word, non-space) with a space
    text = re.sub(r'[^\w\s]', '', text)
    # 3) lowercase everything
    text = text.lower()
    # 4) collapse multiple whitespace into one, and strip ends
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def read_gpt_label(label):
    # label is formatted as a JSON string with 10 key-value pairs
    label = label.replace("'", '"')  # Replace single quotes with double quotes
    label = label.replace('nan', 'null')  # Replace 'nan' with 'null' for JSON compatibility
    if label == 'gpt_label': 
        return {} 
    try:
        label_dict = eval(label)  # Use eval to convert the string to a dictionary
        return label_dict
    except Exception as e:
        print(f"Error parsing label: {e}")
        return {}

classified_data['read_label'] = classified_data['gpt_label'].apply(read_gpt_label)
classified_data['interpersonal'] = classified_data['read_label'].apply(lambda x: 1 if x.get('Interpersonal', 'No') == 'Yes' else 0)
classified_data['routine'] = classified_data['read_label'].apply(lambda x: 1 if x.get('Routine', 'No') == 'Yes' else 0)
classified_data['manual'] = classified_data['read_label'].apply(lambda x: 1 if x.get('Manual', 'No') == 'Yes' else 0)
classified_data['high_codifiable'] = classified_data['read_label'].apply(lambda x: 1 if x.get('High Cod.', 'No') == 'Yes' else 0)

classified_data['task_clean'] = classified_data['Task'].apply(clean_text)

# Group by Task and compute mode and mean for each category
classified_data_grouped = classified_data.groupby('task_clean').agg({
    'interpersonal': ['count', 'mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
    'routine': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
    'manual': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
    'high_codifiable': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0]
    }).reset_index()

# collapse the multi-level columns into dimension_mean and dimension_mode
classified_data_grouped.columns = ['task_clean', 'count', 'interpersonal_mean', 'interpersonal_mode',
                        'routine_mean', 'routine_mode',
                        'manual_mean', 'manual_mode',
                        'high_codifiable_mean', 'high_codifiable_mode']

print("Shape of classified data after grouping by Task: ", classified_data_grouped.shape)
# save classified data grouped by task
print("*"*50)


# IMPORT the BLS/OES data
oes_data = pd.read_csv(os.path.join(ailabor_root, 'bls_transformations/output_data/oes_data_filtered_soc_2018.csv'))
oes_data_occs = oes_data['soc_2018'].unique()
oes_data_healthcare_occs = oes_data[oes_data['is_healthcare_obs']]['soc_2018'].unique()

print("There are {} unique occupations in OES data.".format(len(oes_data_occs)))
print("There are {} unique occupations in OES healthcare data.".format(len(oes_data_healthcare_occs)))

# IMPORT the task data
all_task_datapath = os.path.join(ailabor_root, 'onet_transformations/intermediate_data/task_data_merged_attributes.csv')
all_task_data = pd.read_csv(all_task_datapath)
all_task_data['task_clean'] = all_task_data['Task'].apply(clean_text)

# Check how many occs in OES data are present in the task data
unique_occs_in_task_data = all_task_data['O*NET 2018 SOC Code'].unique()
oes_in_task_data = np.isin(oes_data_occs, unique_occs_in_task_data).sum()
print("There are {} occupations in OES data that are also present in the task data.".format(oes_in_task_data))

# Check how many occs in OES healthcare data are present in the task data
oes_healthcare_in_task_data = np.isin(oes_data_healthcare_occs, unique_occs_in_task_data).sum()
print("There are {} occupations in OES healthcare data that are also present in the task data.".format(oes_healthcare_in_task_data))

# Save the healthcare occupations in the OES data that are not in the task data to a CSV file
oes_healthcare_not_in_task_data = oes_data[oes_data['is_healthcare_obs'] & ~oes_data['soc_2018'].isin(unique_occs_in_task_data)]
oes_healthcare_not_in_task_data.to_csv(os.path.join(ailabor_root, 'bls_transformations/output_data/oes_healthcare_not_in_task_data.csv'), index=False)

print("*"*50)


## SECTION: Construction of ALM Dataset
# datasets 
# oes_data: the OES data with occupations
# oes_data_healthcare: the OES data with healthcare occupations
# all_task_data: the task data with attributes
# classified_data_grouped: the classified task data
oes_data_aggregated_occ = oes_data.groupby(['soc_2018', 'bls_release_year']).agg({'tot_emp': 'sum', 'pct_year_tot_emp': 'sum'}).reset_index()
oes_healthcare_data_aggregated_occ = oes_data[oes_data['is_healthcare_obs']].copy().groupby(['soc_2018', 'bls_release_year']).agg({'tot_emp': 'sum', 'pct_healthcare_tot_emp': 'sum'}).reset_index()

# quick check
chk = (oes_healthcare_data_aggregated_occ
       .groupby('bls_release_year')['pct_healthcare_tot_emp']
       .sum().round(6))
print(chk.head())   # should print a column of 1.0s

combined_data =  pd.merge(all_task_data, oes_data_aggregated_occ, left_on=['O*NET 2018 SOC Code', 'ONET_release_year'], right_on=['soc_2018', 'bls_release_year'], how='inner')
combined_data =  pd.merge(combined_data, classified_data_grouped, on='task_clean', how = 'left')
combined_data_core = combined_data[combined_data['Task Type'] == 'Core'].copy()

combined_data_healthcare = (
    all_task_data
        .merge(oes_healthcare_data_aggregated_occ,
               left_on=['O*NET 2018 SOC Code','ONET_release_year'],
               right_on=['soc_2018','bls_release_year'],
               how='inner')
        .merge(classified_data_grouped, on='task_clean', how='inner')  # INNER keeps only labelled tasks
)
combined_data_healthcare_core = combined_data_healthcare[combined_data_healthcare['Task Type'] == 'Core'].copy()

# another check 
print("#"*25, " For healthcare occupations ", "#"*25)
comb1 = combined_data_healthcare.groupby(['ONET_release_year', 'O*NET 2018 SOC Code']).agg({'pct_healthcare_tot_emp': 'mean'}).reset_index()
cov = (comb1
          .groupby('ONET_release_year')['pct_healthcare_tot_emp']
          .sum()
          .rename('covered_share')
          .reset_index())

print(cov)        # or plot(cov)

# another check 
print("#"*25, " For full occupations ", "#"*25)
comb2 = combined_data.groupby(['ONET_release_year', 'O*NET 2018 SOC Code']).agg({'pct_year_tot_emp': 'mean'}).reset_index()
cov = (comb2
          .groupby('ONET_release_year')['pct_year_tot_emp']
          .sum()
          .rename('covered_share')
          .reset_index())

print(cov)        # or plot(cov)

print("*"*50)

print("Combined (Full) data shape:", combined_data.shape)
print("Combined data limited to observations with occupations in healthcare industry shape:", combined_data_healthcare.shape)
print("Combined data limited to Core tasks shape:", combined_data_core.shape)
print("Combined data limited to Core tasks in healthcare industry shape:", combined_data_healthcare_core.shape)

print("*"*50)

print("There are {} unique occupations in the combined data for 2003.".format(len(combined_data[combined_data['ONET_release_year'] == 2003]['O*NET 2018 SOC Code'].unique())))
print("Date range for combined data:", combined_data['ONET_release_year'].min(), "to", combined_data['ONET_release_year'].max())

print("There are {} unique occupations in the combined data limited to core tasks for 2003.".format(len(combined_data_core[combined_data_core['ONET_release_year'] == 2003]['O*NET 2018 SOC Code'].unique())))
print("Date range for combined core data:", combined_data_core['ONET_release_year'].min(), "to", combined_data_core['ONET_release_year'].max())

print("There are {} unique occupations in the combined healthcare data for 2003.".format(len(combined_data_healthcare[combined_data_healthcare['ONET_release_year'] == 2003]['O*NET 2018 SOC Code'].unique())))
print("Date range for combined healthcare data:", combined_data_healthcare['ONET_release_year'].min(), "to", combined_data_healthcare['ONET_release_year'].max())

print("There are {} unique occupations in the combined healthcare data limited to core tasks for 2003.".format(len(combined_data_healthcare_core[combined_data_healthcare_core['ONET_release_year'] == 2003]['O*NET 2018 SOC Code'].unique())))
print("Date range for combined healthcare core data:", combined_data_healthcare_core['ONET_release_year'].min(), "to", combined_data_healthcare_core['ONET_release_year'].max())

print("*"*50)

# Combine the 'mode' columns into a single 'mode' column for each task type
def combine_modes_full(row):
    s1 = 'X'
    s2 = 'X'
    s3 = 'X'
    s4 = 'X'

    if pd.isna(row['routine_mode']) or pd.isna(row['interpersonal_mode']) or pd.isna(row['manual_mode']) or pd.isna(row['high_codifiable_mode']): 
        return np.nan

    if row['routine_mode'] == 1:
        s1 = 'R'
    else:
        s1 = 'NR'

    if row['interpersonal_mode'] == 1:
        s2 = 'I'
    else: 
        s2 = 'P'

    if row['manual_mode'] == 1:
        s3 = 'M'
    else:
        s3 = 'NM'  
    
    if row['high_codifiable_mode'] == 1:
        s4 = 'HC'
    else:   
        s4 = 'LC'

    return '-'.join([s1, s2, s3, s4])

def combine_modes_alm(row):
    s1 = 'X'
    s2 = 'X'
    s3 = 'X'

    if pd.isna(row['routine_mode']) or pd.isna(row['interpersonal_mode']) or pd.isna(row['manual_mode']):
        return np.nan
    if row['routine_mode'] == 1:
        s1 = 'R'
    else: 
        s1 = 'NR'
    if row['interpersonal_mode'] == 1:
        s2 = 'I'
    else:
        s2 = 'P'
    if row['manual_mode'] == 1:
        s3 = 'M'
    else:
        s3 = 'NM'  

    return '-'.join([s1, s2, s3])


def combine_modes_code(row):
    s1 = 'X'
    s2 = 'X'
    s3 = 'X'
    if pd.isna(row['interpersonal_mode']) or pd.isna(row['manual_mode']) or pd.isna(row['high_codifiable_mode']): 
        return np.nan

    if row['high_codifiable_mode'] == 1:
        s1 = 'HC'
    else:
        s1 = 'LC'

    if row['interpersonal_mode'] == 1:
        s2 = 'I'
    else:
        s2 = 'P'
    if row['manual_mode'] == 1:
        s3 = 'M'
    else:
        s3 = 'NM'  

    return '-'.join([s1, s2, s3])

dfs = {'combined_full_all': combined_data, 
       'combined_healthcare_all': combined_data_healthcare, 
       'combined_full_core': combined_data_core, 
       'combined_healthcare_core': combined_data_healthcare_core}


for name, df in dfs.items():
    df['full_classification'] = df.apply(combine_modes_full, axis=1)
    df['alm_classification'] = df.apply(combine_modes_alm, axis=1)
    df['code_classification'] = df.apply(combine_modes_code, axis=1)
    print("Final shape of the {} dataframe with classifications:".format(name), df.shape)

# Save the final dataframes, use a loop
for name, df in dfs.items():
    df.to_csv(os.path.join(ailabor_root, 'alm_analysis/data/{}.csv'.format(name)), index=False)