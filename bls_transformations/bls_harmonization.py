import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# Define file paths for input data and output results.
# Using absolute paths is recommended for clarity and portability.
OES_DATAPATH = '/Users/sidsatya/dev/ailabor/data/bls/naics_4digit/'
CROSSWALK_DIR = '/Users/sidsatya/dev/ailabor/data/bls/crosswalks/'
OUTPUT_DIR = '/Users/sidsatya/dev/ailabor/bls_transformations/output_data1/'

# List of columns that should be treated as numeric.
NUMERIC_COLS = [
    'tot_emp', 'emp_prse', 'h_mean', 'a_mean', 'mean_prse',
    'h_pct10', 'h_pct25', 'h_median', 'h_pct75', 'h_pct90',
    'a_pct10', 'a_pct25', 'a_median', 'a_pct75', 'a_pct90'
]
# List of columns to keep for the final analysis.
COLUMNS_OF_INTEREST = ['naics', 'naics_title', 'occ_code', 'occ_title', 'bls_release_year', 'tot_emp',
                       'emp_prse', 'pct_total', 'h_mean', 'a_mean', 'mean_prse', 'h_pct10', 'h_pct25', 'h_median', 'h_pct75',
                       'h_pct90', 'a_pct10', 'a_pct25', 'a_median', 'a_pct75', 'a_pct90']

# --- DATA PROCESSING MODULES ---

def load_and_prepare_data(oes_datapath):
    """
    Loads all OES CSV files from a directory, standardizes columns, and concatenates them.

    - Reads all CSV files from the specified directory.
    - Converts all column names to lowercase for consistency.
    - Renames occupation group columns ('group' or 'o_group') to a standard 'occ_group'.
    - Extracts the BLS release year from the filename (e.g., 'oes_2019_...csv' -> 2019).
    - Concatenates all individual dataframes into a single large dataframe.

    Args:
        oes_datapath (str): The path to the directory containing OES data files.

    Returns:
        pd.DataFrame: A single dataframe with combined data from all CSV files.
    """
    print("1. Loading and preparing data...")
    oes_files = [f for f in os.listdir(oes_datapath) if f.endswith('.csv')]
    all_df = []
    for file in oes_files:
        file_path = os.path.join(oes_datapath, file)
        df = pd.read_csv(file_path, dtype=str)
        # Standardize column names to lowercase for easier access.
        df.columns = [col.lower() for col in df.columns]
        # Harmonize column names for occupational groups across different file versions.
        if 'group' in df.columns:
            df.rename(columns={'group': 'occ_group'}, inplace=True)
        if 'o_group' in df.columns:
            df.rename(columns={'o_group': 'occ_group'}, inplace=True)
        # Extract the release year from the filename. Assumes format like '..._YYYY_...'.
        df['bls_release_year'] = file.split('_')[1]
        all_df.append(df)
    return pd.concat(all_df, ignore_index=True)

def filter_oes_data(df):
    """
    Filters out summary rows and high-level occupational groups.

    - Removes rows representing totals for all occupations (occ_code '00-0000').
    - Removes rows for broad occupational categories ('major', 'minor', 'broad')
      to keep only detailed occupation data.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: A filtered dataframe containing only detailed occupation data.
    """
    print("2. Filtering unnecessary data...")
    return df[(df['occ_code'] != '00-0000') &
              ~(df['occ_group'].isin(['major', 'minor', 'broad']))].copy()

def handle_suppression(df):
    """
    Identifies and removes data for highly suppressed industry-occupation pairs.

    Suppressed data is marked as '**' or NaN by BLS for confidentiality reasons.
    This function improves data reliability by:
    - Counting suppressed and total rows for each industry-occupation pair over the years.
    - Calculating the percentage of years a pair's data is suppressed.
    - Keeping only pairs with suppression rates of 25% or less.
    - Removing any remaining individual rows that have suppressed employment data.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: A dataframe with unreliable and suppressed data removed.
    """
    print("3. Handling data suppression...")
    # For each industry-occupation pair, count suppressed vs. total entries.
    suppression_counts = df.groupby(['naics', 'occ_code'])['tot_emp'].agg(
        num_suppressed=lambda x: ((x == '**') | (x.isna())).sum(),
        num_rows='count'
    ).reset_index()
    suppression_counts['pct_years_suppressed'] = suppression_counts['num_suppressed'] / suppression_counts['num_rows']

    print(f"Found {suppression_counts[suppression_counts['pct_years_suppressed'] > 0.25].shape[0]} naics-occ pairs with suppression > 25%.")

    # Identify pairs that are reliable (low suppression rate).
    reliable_pairs = suppression_counts[suppression_counts['pct_years_suppressed'] <= 0.25][['naics', 'occ_code']]
    # Keep only the data for these reliable pairs.
    df_less_suppressed = df.merge(reliable_pairs, on=['naics', 'occ_code'], how='inner')

    # Remove any individual rows that still contain suppression markers.
    df_no_suppression = df_less_suppressed[(df_less_suppressed['tot_emp'] != '**') &
                                           (df_less_suppressed['tot_emp'].notna())].copy()

    print(f"Original shape after filtering: {df.shape}")
    print(f"New shape after suppression handling: {df_no_suppression.shape}")
    return df_no_suppression

def convert_to_float(x):
    """Helper function to convert string values to numeric floats, handling special characters."""
    if isinstance(x, str):
        # Remove common non-numeric characters from BLS data.
        x = x.replace(' ', '').replace('*', '').replace('#', '').replace('%', '').replace(',', '')
        if x == '':
            return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def convert_data_types(df, numeric_cols):
    """
    Converts specified columns to their appropriate data types.

    - Converts 'bls_release_year' to a numeric type for calculations and sorting.
    - For other specified numeric columns, it removes special characters ('*', '#', ',')
      and converts them to float, handling potential conversion errors gracefully.

    Args:
        df (pd.DataFrame): The input dataframe.
        numeric_cols (list): A list of column names to convert to float.

    Returns:
        pd.DataFrame: The dataframe with converted data types.
    """
    print("4. Converting data types...")
    df['bls_release_year'] = pd.to_numeric(df['bls_release_year'])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_to_float)
    return df

def calculate_employment_shares(df):
    """
    Calculates and adds employment share columns to the dataframe.

    - Calculates total employment for each year across all industries.
    - Calculates total employment for each industry (NAICS) for each year.
    - Computes each occupation's employment share of the yearly total.
    - Computes each occupation's employment share of its industry-year total.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with new employment share columns.
    """
    print("5. Calculating employment shares...")
    df_final = df[COLUMNS_OF_INTEREST].copy()

    # Calculate total employment per year.
    oes_year_totals = df_final.groupby(['bls_release_year']).agg({'tot_emp': 'sum'}).reset_index()
    oes_year_totals.rename(columns={'tot_emp': 'year_tot_emp'}, inplace=True)

    # Calculate total employment per industry per year.
    oes_naics_year_totals = df_final.groupby(['naics', 'bls_release_year']).agg({'tot_emp': 'sum'}).reset_index()
    oes_naics_year_totals.rename(columns={'tot_emp': 'naics_year_tot_emp'}, inplace=True)

    # Merge these totals back into the main dataframe.
    df_merged = df_final.merge(oes_year_totals, on='bls_release_year', how='left')
    df_merged = df_merged.merge(oes_naics_year_totals, on=['naics', 'bls_release_year'], how='left')

    # Calculate employment shares.
    df_merged['pct_year_tot_emp'] = df_merged['tot_emp'] / df_merged['year_tot_emp']
    df_merged['pct_naics_year_tot_emp'] = df_merged['tot_emp'] / df_merged['naics_year_tot_emp']

    return df_merged

def harmonize_soc_codes(df: pd.DataFrame, crosswalk_dir: str) -> pd.DataFrame:
    """
    Attach 2010-SOC and 2018-SOC codes to every OES row.

    ─────────  SOC timeline in OES  ─────────
      2003–2009 : 2000 SOC
      2010–2011 : hybrid (2000 + 2010 SOC)         ← needs 2000→10 *and* hybrid map
      2012–2018 : pure 2010 SOC
      2019–2020 : hybrid (2010 + 2018 SOC)         ← needs 10→18 *and* hybrid map
      2021–     : pure 2018 SOC
    ──────────────────────────────────────────
    """

    print("Step 6 – Harmonising SOC codes …")

    # ── 1. LOAD CROSSWALKS ────────────────────────────────────────────────
    cw_00_10   = pd.read_csv(os.path.join(crosswalk_dir, "crosswalk_2000_to_2010.csv"), dtype=str)
    cw_10_18   = pd.read_csv(os.path.join(crosswalk_dir, "crosswalk_2010_to_2018.csv"), dtype=str)
    cw_mix_19  = pd.read_csv(os.path.join(crosswalk_dir, "crosswalk_hybrid_2010_to_2018.csv"), dtype=str)
    cw_mix_00  = pd.read_csv(os.path.join(crosswalk_dir, "crosswalk_hybrid_2000_to_2010.csv"), dtype=str)

    CROSSWALK = {
        "00_to_10": cw_00_10.set_index("2000 SOC Code")["2010 SOC Code"].to_dict(),
        "10_to_18": cw_10_18.set_index("2010 SOC Code")["2018 SOC Code"].to_dict(),
        "18_to_10": cw_10_18.set_index("2018 SOC Code")["2010 SOC Code"].to_dict(),
        # hybrid placeholders
        "mix19_to_10": cw_mix_19.set_index("OES 2019 Estimates Code")["2010 SOC Code"].to_dict(),
        "mix19_to_18": cw_mix_19.set_index("OES 2019 Estimates Code")["2018 SOC Code"].to_dict(),
        "mix00_to_10": cw_mix_00.set_index("2000 SOC code")["OES 2010 code"].to_dict(),
    }

    SET_2010 = set(CROSSWALK["10_to_18"].keys())
    SET_2018 = set(CROSSWALK["18_to_10"].keys())

    # ── 2. ROW-WISE CONVERTERS ────────────────────────────────────────────
    def map_to_2010(code: str, yr: int) -> str | None:
        """Return 2010 SOC code, or original / None if unmapped."""
        if 2003 <= yr <= 2009:                # pure 2000 SOC
            return CROSSWALK["00_to_10"].get(code, code)

        if 2010 <= yr <= 2011:                # mix 2000 + 2010
            if code in SET_2010:
                return code
            return (CROSSWALK["00_to_10"].get(code) or
                    CROSSWALK["mix00_to_10"].get(code) or code)

        if 2012 <= yr <= 2018:                # pure 2010 SOC
            return code

        if 2019 <= yr <= 2020:                # mix 2010 + 2018
            return (CROSSWALK["mix19_to_10"].get(code) or
                    (code if code in SET_2010 else None) or
                    CROSSWALK["18_to_10"].get(code) or code)

        if yr >= 2021:                        # pure 2018 SOC
            return CROSSWALK["18_to_10"].get(code, code)

        return None  # should never hit

    def map_to_2018(code: str, yr: int) -> str | None:
        """Return 2018 SOC code, or original / None if unmapped."""
        if 2003 <= yr <= 2009:                # 2000 → 10 → 18
            code10 = CROSSWALK["00_to_10"].get(code, code)
            return CROSSWALK["10_to_18"].get(code10, code10)

        if 2010 <= yr <= 2011:                # mix 2000 + 2010
            if code in SET_2010:
                return CROSSWALK["10_to_18"].get(code, code)
            code10 = (CROSSWALK["00_to_10"].get(code) or
                      CROSSWALK["mix00_to_10"].get(code))
            return CROSSWALK["10_to_18"].get(code10, code10)

        if 2012 <= yr <= 2018:                # pure 2010
            return CROSSWALK["10_to_18"].get(code, code)

        if 2019 <= yr <= 2020:                # mix 2010 + 2018
            return (CROSSWALK["mix19_to_18"].get(code) or
                    (CROSSWALK["10_to_18"].get(code) if code in SET_2010 else None) or
                    code)  # already 2018

        if yr >= 2021:                        # pure 2018
            return code

        return None

    # ── 3. APPLY CONVERTERS ──────────────────────────────────────────────
    df["occ_code"] = df["occ_code"].astype(str).str.strip()

    df["soc_2010"] = df.apply(lambda r: map_to_2010(r["occ_code"],
                                                    int(r["bls_release_year"])), axis=1)
    df["soc_2018"] = df.apply(lambda r: map_to_2018(r["occ_code"],
                                                    int(r["bls_release_year"])), axis=1)

    # ── 4. VERIFICATION PRINT-OUTS ───────────────────────────────────────
    def _report(col):
        miss = df[col].isna().mean()
        print(f"  • {col}: {100*(1-miss):.2f}% mapped "
              f"({df[col].notna().sum():,}/{len(df):,})")
        if miss:
            print("    examples of unmapped codes:",
                  df.loc[df[col].isna(), "occ_code"].unique()[:5])

    print("Mapping success rates")
    _report("soc_2010")
    _report("soc_2018")
    print("-" * 50)

    return df






def aggregate_by_soc(df):
    """
    Aggregates data by the harmonized 2018 SOC code, industry, and year.

    This creates a consistent time series for each occupation within each industry.
    - Sums employment figures ('tot_emp', 'pct_year_tot_emp', etc.).
    - Averages wage and percentile data ('h_mean', 'a_pct10', etc.).

    Args:
        df (pd.DataFrame): The dataframe with harmonized SOC codes.

    Returns:
        pd.DataFrame: An aggregated dataframe.
    """
    print("7. Aggregating data to harmonized SOC codes...")
    agg_spec = {
        'tot_emp': 'sum', 'pct_year_tot_emp': 'sum', 'pct_naics_year_tot_emp': 'sum',
        'h_mean': 'mean', 'a_mean': 'mean', 'h_pct10': 'mean', 'h_pct25': 'mean',
        'h_median': 'mean', 'h_pct75': 'mean', 'h_pct90': 'mean', 'a_pct10': 'mean',
        'a_pct25': 'mean', 'a_median': 'mean', 'a_pct75': 'mean', 'a_pct90': 'mean',
        # These totals are constant within each group, so 'first' is used to avoid re-calculating.
        'naics_year_tot_emp': 'first', 'year_tot_emp': 'first'
    }
    return df.groupby(['naics', 'soc_2018', 'bls_release_year']).agg(agg_spec).reset_index()

def analyze_healthcare_sector(df_agg):
    """
    Performs a specific analysis on the healthcare sector (NAICS codes starting with '62').

    - Filters for healthcare data.
    - Calculates total employment in healthcare for each year.
    - Computes each occupation's share of total healthcare employment for that year.
    - Merges these healthcare-specific metrics back into the main aggregated dataframe.

    Args:
        df_agg (pd.DataFrame): The aggregated dataframe.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The main aggregated dataframe with added healthcare metrics.
            - pd.DataFrame: A dataframe containing only the healthcare sector data.
    """
    print("8. Analyzing healthcare sector...")
    # Filter for healthcare industries (NAICS starts with '62').
    df_healthcare = df_agg[df_agg['naics'].astype(str).str.startswith('62')].copy()

    # Calculate total employment in healthcare for each year.
    healthcare_totals = df_healthcare.groupby(['bls_release_year']).agg({'tot_emp': 'sum'}).reset_index()
    healthcare_totals.rename(columns={'tot_emp': 'healthcare_tot_emp'}, inplace=True)

    # Merge totals back and calculate the percentage of healthcare employment.
    df_healthcare = df_healthcare.merge(healthcare_totals, on='bls_release_year', how='left')
    df_healthcare['pct_healthcare_tot_emp'] = df_healthcare['tot_emp'] / df_healthcare['healthcare_tot_emp']

    # Merge the new healthcare percentage column back to the main aggregated dataframe.
    df_agg_merged = pd.merge(df_agg, df_healthcare[['naics', 'soc_2018', 'bls_release_year', 'pct_healthcare_tot_emp']],
                             on=['naics', 'soc_2018', 'bls_release_year'], how='left')
    
    # Add a boolean flag for easy filtering of healthcare observations.
    df_agg_merged['is_healthcare_obs'] = df_agg_merged['pct_healthcare_tot_emp'].notna()

    return df_agg_merged, df_healthcare

def save_results(df_full, df_agg_merged, df_healthcare, output_dir):
    """
    Saves the final dataframes to CSV files in the specified output directory.

    Args:
        df_full (pd.DataFrame): The full, harmonized (but not aggregated) dataframe.
        df_agg_merged (pd.DataFrame): The final aggregated dataframe with healthcare metrics.
        df_healthcare (pd.DataFrame): The dataframe containing only healthcare data.
        output_dir (str): The path to the directory where files will be saved.
    """
    print(f"9. Saving results to {output_dir}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_full.to_csv(os.path.join(output_dir, 'oes_data_filtered_full.csv'), index=False)
    df_agg_merged.to_csv(os.path.join(output_dir, 'oes_data_filtered_soc_2018.csv'), index=False)
    df_healthcare.to_csv(os.path.join(output_dir, 'oes_data_filtered_healthcare_soc_2018.csv'), index=False)
    print("Done.")

def main():
    """Main function to run the entire data processing pipeline."""
    # Step 1: Load and perform initial preparation of the raw data from multiple CSVs.
    oes_data = load_and_prepare_data(OES_DATAPATH)

    # Step 2: Filter out high-level summary data to focus on detailed occupations.
    oes_data_filtered = filter_oes_data(oes_data)

    # Step 3: Handle data suppression to ensure data reliability for analysis.
    oes_data_no_suppression = handle_suppression(oes_data_filtered)

    # Step 4: Convert data columns to their appropriate types (e.g., string to numeric).
    oes_data_typed = convert_data_types(oes_data_no_suppression, NUMERIC_COLS)

    # Step 5: Calculate employment shares relative to various totals (yearly, industry-yearly).
    oes_data_shares = calculate_employment_shares(oes_data_typed)

    # Step 6: Harmonize Standard Occupational Classification (SOC) codes across different years.
    oes_data_harmonized = harmonize_soc_codes(oes_data_shares, CROSSWALK_DIR)

    # Step 7: Aggregate data by the harmonized 2018 SOC codes for consistent time-series analysis.
    oes_data_2018_agg = aggregate_by_soc(oes_data_harmonized)

    # Step 8: Perform a specific analysis on the healthcare sector and merge results.
    oes_data_2018_final, oes_data_healthcare = analyze_healthcare_sector(oes_data_2018_agg)

    # Step 9: Save the final processed dataframes to CSV files for future use.
    save_results(oes_data_harmonized, oes_data_2018_final, oes_data_healthcare, OUTPUT_DIR)

if __name__ == "__main__":
    main()