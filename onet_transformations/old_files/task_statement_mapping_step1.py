"""
This script harmonizes historical O*NET task statement data by mapping SOC codes
from various years to the 2010 and 2018 SOC standards.

The script performs the following steps:
1.  **Configuration**: Sets up file paths and defines the years of data to be processed.
2.  **Load Crosswalks**: Reads multiple SOC crosswalk files (e.g., 2000-to-2006, 2006-to-2009) and loads them into dictionaries for efficient mapping.
3.  **Define SOC Conversion Logic**: Creates functions that chain lookups through the crosswalk dictionaries to convert any given SOC code from its original year to the target 2010 or 2018 standard.
4.  **Process Data in a Loop**:
    - Iterates through each year of O*NET task data (2003-2025).
    - Reads the corresponding CSV file into a pandas DataFrame.
    - Applies the conversion functions to create new columns for the 2010 and 2018 SOC codes.
    - Adds a column to track the release year of the O*NET data.
5.  **Combine and Clean Data**:
    - Concatenates all the yearly DataFrames into a single master DataFrame.
    - Selects a final set of relevant columns.
6.  **Analyze and Export**:
    - Calculates the frequency of each unique task statement across all years.
    - Saves the unique task counts to a CSV file.
    - Saves the final, harmonized dataset of all task statements with mapped SOC codes to another CSV file.
"""

import pandas as pd
import os
from collections import defaultdict

# --- Configuration ---
# Define the root directory for the project.
# Using a relative path or environment variable is recommended for portability.
ROOT_DIR = "/Users/sidsatya/dev/ailabor"
print(f"Using ROOT_DIR: {ROOT_DIR}")

# Define directories for input data and output results.
ONET_DATA_DIR = os.path.join(ROOT_DIR, "data/onet")
TASK_STATEMENTS_DIR = os.path.join(ONET_DATA_DIR, "historical_onet_task_statements")
CROSSWALK_DIR = os.path.join(ONET_DATA_DIR, "onet_occsoc_crosswalks")
OUTPUT_DIR = os.path.join(ROOT_DIR, "onet_transformations/task_statement_harmonization/intermediate_data")
# OUTPUT_DIR = os.path.join(ROOT_DIR, "onet_transformations/output_data")

# --- Helper Functions ---

def load_crosswalk(filepath, key_col, val_col1, val_col2):
    """
    Reads a crosswalk CSV file and loads it into a dictionary.

    Args:
        filepath (str): The full path to the CSV file.
        key_col (str): The name of the column to use as the dictionary key.
        val_col1 (str): The name of the first column for the dictionary value.
        val_col2 (str): The name of the second column for the dictionary value.

    Returns:
        dict: A dictionary mapping keys to a tuple of (value1, value2).
              Returns an empty dictionary if the file is not found.
    """
    try:
        df = pd.read_csv(filepath, encoding='latin1', dtype=str)
        # Clean whitespace from columns to ensure accurate lookups
        df[key_col] = df[key_col].str.strip()
        df[val_col1] = df[val_col1].str.strip()
        df[val_col2] = df[val_col2].str.strip()
        return pd.Series(list(zip(df[val_col1], df[val_col2])), index=df[key_col]).to_dict()
    except FileNotFoundError:
        print(f"Warning: Crosswalk file not found at {filepath}")
        return {}
    except Exception as e:
        print(f"An error occurred while loading {filepath}: {e}")
        return {}

def get_soc_from_dict(soc_code, mapping_dict):
    """Safely retrieves a value from a crosswalk dictionary."""
    return mapping_dict.get(soc_code)

# --- Load All Crosswalks ---

print("Loading SOC crosswalks...")
soc_2000_to_2006 = load_crosswalk(
    os.path.join(CROSSWALK_DIR, "onet_2000_to_2006_crosswalk.csv"),
    "O*NET-SOC 2000 Code", "O*NET-SOC 2006 Code", "O*NET-SOC 2006 Title"
)
soc_2006_to_2009 = load_crosswalk(
    os.path.join(CROSSWALK_DIR, "onet_2006_to_2009_crosswalk.csv"),
    "O*NET-SOC 2006 Code", "O*NET-SOC 2009 Code", "O*NET-SOC 2009 Title"
)
soc_2009_to_2010 = load_crosswalk(
    os.path.join(CROSSWALK_DIR, "onet_2009_to_2010_crosswalk.csv"),
    "O*NET-SOC 2009 Code", "O*NET-SOC 2010 Code", "O*NET-SOC 2010 Title"
)

# Safer reverse mapping for 2019 -> 2010
soc_2010_to_2019_df = pd.read_csv(os.path.join(CROSSWALK_DIR, "onet_2010_to_2019_crosswalk.csv"), dtype=str)
# Clean whitespace to ensure accurate lookups
soc_2010_to_2019_df['O*NET-SOC 2019 Code'] = soc_2010_to_2019_df['O*NET-SOC 2019 Code'].str.strip()
soc_2010_to_2019_df['O*NET-SOC 2010 Code'] = soc_2010_to_2019_df['O*NET-SOC 2010 Code'].str.strip()
soc_2010_to_2019_df['O*NET-SOC 2019 Title'] = soc_2010_to_2019_df['O*NET-SOC 2019 Title'].str.strip()
soc_2010_to_2019_df['O*NET-SOC 2010 Title'] = soc_2010_to_2019_df['O*NET-SOC 2010 Title'].str.strip()

# Create dictionaries with tuples (code, title) as values
soc_2010_to_2019 = dict(zip(
    soc_2010_to_2019_df['O*NET-SOC 2010 Code'], 
    zip(soc_2010_to_2019_df['O*NET-SOC 2019 Code'], soc_2010_to_2019_df['O*NET-SOC 2019 Title'])
))
soc_2019_to_2010 = dict(zip(
    soc_2010_to_2019_df['O*NET-SOC 2019 Code'], 
    zip(soc_2010_to_2019_df['O*NET-SOC 2010 Code'], soc_2010_to_2019_df['O*NET-SOC 2010 Title'])
))


soc_2019_to_2018 = load_crosswalk(
    os.path.join(CROSSWALK_DIR, "onet_2019_to_2018_crosswalk.csv"),
    "O*NET-SOC 2019 Code", "2018 SOC Code", "2018 SOC Title"
)
print("Crosswalks loaded.")

# --- SOC Conversion Functions ---

def convert_to_2010_soc(soc_code, year):
    """
    Converts a SOC code from a given year to its 2010 equivalent by chaining lookups.
    """
    try:
        if 2003 <= year <= 2005:
            soc_2006 = get_soc_from_dict(soc_code, soc_2000_to_2006)
            soc_2009 = get_soc_from_dict(soc_2006[0], soc_2006_to_2009) if soc_2006 else None
            soc_2010 = get_soc_from_dict(soc_2009[0], soc_2009_to_2010) if soc_2009 else None
            return soc_2010[0] if soc_2010 else None
        if 2006 <= year <= 2008:
            soc_2009 = get_soc_from_dict(soc_code, soc_2006_to_2009)
            soc_2010 = get_soc_from_dict(soc_2009[0], soc_2009_to_2010) if soc_2009 else None
            return soc_2010[0] if soc_2010 else None
        if 2009 <= year <= 2010:
            soc_2010 = get_soc_from_dict(soc_code, soc_2009_to_2010)
            return soc_2010[0] if soc_2010 else None
        if 2011 <= year <= 2019:
            # For 2012-2019, we can directly use the 2010 SOC code
            return soc_code 
        if 2020 <= year <= 2025:
            soc_2010 = get_soc_from_dict(soc_code, soc_2019_to_2010)
            return soc_2010[0] if soc_2010 else None
    except (TypeError, IndexError):
        # This catches errors if a lookup returns None
        return None
    return None

def convert_to_2018_soc(soc_code, year):
    """
    Converts a SOC code from a given year to its 2018 equivalent by chaining lookups.
    ## Conversion to 2018 SOC codes
    # 2003 - 2005 need to be mapped first to 2006 SOC, then 2009 SOC, then 2010 SOC, then 2019 SOC, then 2018 SOC
    # 2006 - 2009 need to be mapped first to 2009 SOC, then 2010 SOC, then 2019 SOC, then 2018 SOC
    # 2010 - 2011 need to be mapped to 2010 SOC, then 2019 SOC, then 2018 SOC
    # 2012 - 2019 need to be mapped to 2019 SOC, then 2018 SOC
    # 2020 - 2025 need to be mapped to 2018 SOC
    """
    try:
        if 2003 <= year <= 2019: 
            soc_2010 = convert_to_2010_soc(soc_code, year)
            if soc_2010:
                soc_2019 = get_soc_from_dict(soc_2010, soc_2010_to_2019)
                if soc_2019:
                    soc_2018 = get_soc_from_dict(soc_2019[0], soc_2019_to_2018)
                    return soc_2018[0] if soc_2018 else None
        elif 2020 <= year <= 2025:
            soc_2018 = get_soc_from_dict(soc_code, soc_2019_to_2018)
            return soc_2018[0] if soc_2018 else None
    except (TypeError, IndexError):
        return None
    return None

# --- Main Processing Logic ---

def main():
    """
    Main function to orchestrate the data loading, processing, and saving.
    """
    # Define the list of ONET files to process
    onet_files_info = {
        2003: "task_statements_2003_nov.csv", 2004: "task_statements_2004_dec.csv",
        2005: "task_statements_2005_dec.csv", 2006: "task_statements_2006_dec.csv",
        2007: "task_statements_2007_jun.csv", 2008: "task_statements_2008_jun.csv",
        2009: "task_statements_2009_jun.csv", 2010: "task_statements_2010_jul.csv",
        2011: "task_statements_2011_jul.csv", 2012: "task_statements_2012_jul.csv",
        2013: "task_statements_2013_jul.csv", 2014: "task_statements_2014_jul.csv",
        2015: "task_statements_2015_oct.csv", 2016: "task_statements_2016_nov.csv",
        2017: "task_statements_2017_oct.csv", 2018: "task_statements_2018_nov.csv",
        2019: "task_statements_2019_nov.csv", 2020: "task_statements_2020_nov.csv",
        2021: "task_statements_2021_nov.csv", 2022: "task_statements_2022_nov.csv",
        2023: "task_statements_2023_nov.csv", 2024: "task_statements_2024_nov.csv",
        2025: "task_statements_2025_feb.csv"
    }

    all_dataframes = []
    print("\n--- Starting Data Processing ---")
    for year, filename in onet_files_info.items():
        filepath = os.path.join(TASK_STATEMENTS_DIR, filename)
        print(f"Processing {year}: {filename}...")

        try:
            df = pd.read_csv(filepath, encoding='latin1', dtype=str)
        except FileNotFoundError:
            print(f"  -> Warning: File not found. Skipping.")
            continue
        
        # Add a column for the release year
        df['ONET_release_year'] = year

        # Apply the SOC conversion functions
        # Note: .apply is used for row-wise operations. It can be slow on very large
        # datasets, but is clear and suitable for this complex conversion logic.
        if "O*NET-SOC Code" in df.columns:
            df["O*NET-SOC Code"] = df["O*NET-SOC Code"].str.strip()
            df['O*NET 2010 SOC Code'] = df['O*NET-SOC Code'].apply(lambda x: convert_to_2010_soc(x, year))
            df['O*NET 2018 SOC Code'] = df['O*NET-SOC Code'].apply(lambda x: convert_to_2018_soc(x, year))

            #df.to_csv(os.path.join(OUTPUT_DIR, f"task_statements_{year}_mapped_soc_codes.csv"), index=False, encoding="utf-8")

            # Assert that the number of unmapped codes is tolerably low
            missing_2010 = df['O*NET 2010 SOC Code'].isna().mean()
            assert missing_2010 < 0.01, f"{missing_2010:.1%} of codes in {filename} unmapped to 2010 SOC - investigate"

            missing_2018 = df['O*NET 2018 SOC Code'].isna().mean()
            assert missing_2018 < 0.01, f"{missing_2018:.1%} of codes in {filename} unmapped to 2018 SOC - investigate"


            all_dataframes.append(df)
        else:
            print(f"  -> Warning: 'O*NET-SOC Code' column not found in {filename}.")

    # --- Combine, Clean, and Save ---
    if not all_dataframes:
        print("\nNo data was processed. Exiting.")
        return

    print("\n--- Combining and Finalizing Data ---")
    # Concatenate all yearly dataframes into one
    all_onet_data_harmonized = pd.concat(all_dataframes, ignore_index=True, sort=False)

    # Define the final set of columns to keep
    columns_of_interest = [
        "O*NET-SOC Code", "O*NET 2010 SOC Code", "O*NET 2018 SOC Code",
        "ONET_release_year", "Task ID", "Task", "Task Type",
        "Incumbents Responding", "Date", "Domain Source"
    ]
    
    # Filter for existing columns to avoid errors
    final_columns = [col for col in columns_of_interest if col in all_onet_data_harmonized.columns]
    all_onet_data_final = all_onet_data_harmonized[final_columns]

    # --- Analyze and Export ---
    print("Analyzing unique task statements...")
    # Group by 'Task' and count occurrences
    task_counts = all_onet_data_final['Task'].value_counts().reset_index()
    task_counts.columns = ['Task', 'Count']
    print(f"Found {len(task_counts)} unique task statements.")

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the unique task counts to a CSV file
    output_file_tasks = os.path.join(OUTPUT_DIR, "unique_task_statements.csv")
    task_counts.to_csv(output_file_tasks, index=False, encoding="utf-8")
    print(f"Saved unique task counts to: {output_file_tasks}")

    # Save only task IDs and task statements to a separate CSV file
    output_file_tasks_only = os.path.join(OUTPUT_DIR, "task_statements_and_ids.csv")
    tasks_only = all_onet_data_final[['Task ID', 'Task']].drop_duplicates().dropna()
    tasks_only.to_csv(output_file_tasks_only, index=False, encoding="utf-8")
    print(f"Saved task statements to: {output_file_tasks_only}")

    # Save the final harmonized dataset to a CSV file
    output_file_all_data = os.path.join(OUTPUT_DIR, "all_onet_data_mapped_soc_codes.csv")
    all_onet_data_final.to_csv(output_file_all_data, index=False, encoding="utf-8")
    print(f"Saved final harmonized data to: {output_file_all_data}")
    print(f"Final dataset contains {len(all_onet_data_final)} rows.")
    print("\nProcessing complete.")

if __name__ == '__main__':
    main()
