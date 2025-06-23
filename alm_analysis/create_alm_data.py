import pandas as pd
import numpy as np
import os
import re
import json

# --- CONFIGURATION ---
AILABOR_ROOT = '/Users/sidsatya/dev/ailabor/'
CLASSIFIED_TASKS_PATH = os.path.join(AILABOR_ROOT, 'task_classification/data/classified_tasks_16_dim.csv')
OES_DATA_PATH = os.path.join(AILABOR_ROOT, 'bls_transformations/output_data1/oes_data_filtered_soc_2018.csv')
ALL_TASK_DATA_PATH = os.path.join(AILABOR_ROOT, 'onet_transformations/output_data/task_statements_harmonized_with_attributes.csv')
OUTPUT_DIR = os.path.join(AILABOR_ROOT, 'alm_analysis/data1/')
HEALTHCARE_NOT_IN_TASK_DATA_PATH = os.path.join(AILABOR_ROOT, 'bls_transformations/output_data1/oes_healthcare_not_in_task_data.csv')

# --- HELPER FUNCTIONS ---

def clean_text(text: str) -> str:
    """
    Standardizes text by removing special characters, converting to lowercase,
    and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\x92', "'", text)  # Replace non-standard apostrophe
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    return text

def read_gpt_label_safe(label: str):
    """
    Safely parses a string that resembles a JSON object into a dictionary.
    Handles single quotes and 'nan' values for robust parsing.
    """
    if not isinstance(label, str) or label == 'gpt_label':
        return {}
    try:
        # Prepare string for safe JSON parsing
        label_str = label.replace("'", '"').replace('nan', 'null')
        return json.loads(label_str)
    except json.JSONDecodeError:
        # Fallback for malformed strings that json.loads can't handle
        try:
            return eval(label)
        except Exception as e:
            print(f"Error parsing label string: '{label}'. Error: {e}")
            return {}

def create_classification_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized function to create classification strings from mode columns.
    This is significantly faster than applying a function row-by-row.
    """
    df = df.copy()
    
    # Define required columns for each classification type
    mode_cols_full = ['routine_mode', 'interpersonal_mode', 'manual_mode', 'high_codifiable_mode']
    mode_cols_alm = ['routine_mode', 'interpersonal_mode', 'manual_mode']
    mode_cols_code = ['high_codifiable_mode', 'interpersonal_mode', 'manual_mode']

    # Ensure mode columns are numeric, coercing errors to NaN
    for col in mode_cols_full:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create components for classification strings using np.where for efficiency
    s1_alm = np.where(df['routine_mode'] == 1, 'R', 'NR')
    s2_alm = np.where(df['interpersonal_mode'] == 1, 'I', 'P')
    s3_alm = np.where(df['manual_mode'] == 1, 'M', 'NM')
    s1_code = np.where(df['high_codifiable_mode'] == 1, 'HC', 'LC')

    # Build classification strings and set to NaN where components are missing
    df['alm_classification'] = pd.Series(s1_alm + '-' + s2_alm + '-' + s3_alm, index=df.index)
    df.loc[df[mode_cols_alm].isna().any(axis=1), 'alm_classification'] = np.nan

    df['code_classification'] = pd.Series(s1_code + '-' + s2_alm + '-' + s3_alm, index=df.index)
    df.loc[df[mode_cols_code].isna().any(axis=1), 'code_classification'] = np.nan

    df['full_classification'] = df['alm_classification'] + '-' + s1_code
    df.loc[df[mode_cols_full].isna().any(axis=1), 'full_classification'] = np.nan
    
    return df

# --- MAIN PROCESSING SCRIPT ---

def main():
    """
    Main function to orchestrate the creation of the ALM dataset.
    """
    print("--- Starting ALM Dataset Creation ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Prepare Classified Tasks Data ---
    print("\n[1/5] Processing classified task data...")
    classified_data = pd.read_csv(CLASSIFIED_TASKS_PATH)
    classified_data['read_label'] = classified_data['gpt_label'].apply(read_gpt_label_safe)

    # Vectorized creation of classification flags using json_normalize
    labels_df = pd.json_normalize(classified_data['read_label'])
    classification_dims = {'interpersonal': 'Interpersonal', 'routine': 'Routine', 'manual': 'Manual', 'high_codifiable': 'High Cod.'}
    for new_col, old_col in classification_dims.items():
        if old_col in labels_df.columns:
            classified_data[new_col] = (labels_df[old_col].fillna('No') == 'Yes').astype(int)
        else:
            classified_data[new_col] = 0
            
    classified_data['task_clean'] = classified_data['Task'].apply(clean_text)

    # Group by task to get a single classification per unique task
    classified_data_grouped = classified_data.groupby('task_clean').agg({
        'interpersonal': ['count', 'mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
        'routine': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
        'manual': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0],
        'high_codifiable': ['mean', lambda x: x.mode()[0] if not x.mode().empty else 0]
    }).reset_index()
    classified_data_grouped.columns = [
        'task_clean', 'count', 'interpersonal_mean', 'interpersonal_mode',
        'routine_mean', 'routine_mode', 'manual_mean', 'manual_mode',
        'high_codifiable_mean', 'high_codifiable_mode'
    ]
    print(f"Shape of classified data after grouping by task: {classified_data_grouped.shape}")
    print("*"*50)

    # --- 2. Load OES and O*NET Task Data ---
    print("\n[2/5] Loading OES and O*NET data...")
    oes_data = pd.read_csv(OES_DATA_PATH)
    all_task_data = pd.read_csv(ALL_TASK_DATA_PATH)
    all_task_data['task_clean'] = all_task_data['Task'].apply(clean_text)

    # --- 3. Analyze Occupation Coverage ---
    print("\n[3/5] Analyzing occupation coverage...")
    oes_data_occs = oes_data['soc_2018'].unique()
    oes_data_healthcare_occs = oes_data[oes_data['is_healthcare_obs']]['soc_2018'].unique()
    unique_occs_in_task_data = all_task_data['O*NET 2018 SOC Code'].unique()

    print(f"There are {len(oes_data_occs)} unique occupations in OES data.")
    print(f"There are {len(oes_data_healthcare_occs)} unique occupations in OES healthcare data.")
    
    oes_in_task_data = np.isin(oes_data_occs, unique_occs_in_task_data).sum()
    print(f"Coverage: {oes_in_task_data}/{len(oes_data_occs)} OES occupations are present in the task data.")

    oes_healthcare_in_task_data = np.isin(oes_data_healthcare_occs, unique_occs_in_task_data).sum()
    print(f"Healthcare Coverage: {oes_healthcare_in_task_data}/{len(oes_data_healthcare_occs)} OES healthcare occupations are present in the task data.")

    # Save healthcare occupations not covered in task data for review
    oes_healthcare_not_in_task_data = oes_data[oes_data['is_healthcare_obs'] & ~oes_data['soc_2018'].isin(unique_occs_in_task_data)]
    oes_healthcare_not_in_task_data.to_csv(HEALTHCARE_NOT_IN_TASK_DATA_PATH, index=False)
    print(f"Saved {len(oes_healthcare_not_in_task_data)} uncovered healthcare occupations to {os.path.basename(HEALTHCARE_NOT_IN_TASK_DATA_PATH)}")
    print("*"*50)

    # --- 4. Construct ALM Datasets by Merging ---
    print("\n[4/5] Constructing ALM datasets...")
    # Aggregate OES employment data by occupation and year
    oes_data_aggregated_occ = oes_data.groupby(['soc_2018', 'bls_release_year']).agg({'tot_emp': 'sum', 'pct_year_tot_emp': 'sum'}).reset_index()
    oes_healthcare_data_aggregated_occ = oes_data[oes_data['is_healthcare_obs']].groupby(['soc_2018', 'bls_release_year']).agg({'tot_emp': 'sum', 'pct_healthcare_tot_emp': 'sum'}).reset_index()

    # --- Merge for the full dataset ---
    combined_data = pd.merge(all_task_data, oes_data_aggregated_occ, left_on=['O*NET 2018 SOC Code', 'ONET_release_year'], right_on=['soc_2018', 'bls_release_year'], how='inner')
    combined_data = pd.merge(combined_data, classified_data_grouped, on='task_clean', how='left') # Left merge to keep all tasks
    
    # --- Merge for the healthcare dataset ---
    combined_data_healthcare = pd.merge(all_task_data, oes_healthcare_data_aggregated_occ, left_on=['O*NET 2018 SOC Code', 'ONET_release_year'], right_on=['soc_2018', 'bls_release_year'], how='inner')
    combined_data_healthcare = pd.merge(combined_data_healthcare, classified_data_grouped, on='task_clean', how='inner') # Inner merge to keep only classified tasks

    # Create 'Core' task subsets
    combined_data_core = combined_data[combined_data['Task Type'] == 'Core'].copy()
    combined_data_healthcare_core = combined_data_healthcare[combined_data_healthcare['Task Type'] == 'Core'].copy()

    dfs = {
        'combined_full_all': combined_data, 
        'combined_healthcare_all': combined_data_healthcare, 
        'combined_full_core': combined_data_core, 
        'combined_healthcare_core': combined_data_healthcare_core
    }

    print("Initial merged data shapes:")
    for name, df in dfs.items():
        print(f"  - {name}: {df.shape}")
    print("*"*50)

    # --- 5. Add Classifications and Save Results ---
    print("\n[5/5] Adding classification strings and saving final datasets...")
    for name, df in dfs.items():
        # Add classification strings using the vectorized function
        df_classified = create_classification_strings(df)
        
        # Save the final dataframe
        output_path = os.path.join(OUTPUT_DIR, f'{name}.csv')
        df_classified.to_csv(output_path, index=False)
        print(f"Saved '{name}' dataframe with shape {df_classified.shape} to {os.path.basename(output_path)}")

    print("\n--- ALM Dataset Creation Complete ---")

if __name__ == "__main__":
    main()