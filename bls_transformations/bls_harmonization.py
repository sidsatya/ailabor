import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple


# --- CONFIGURATION ---
# Define file paths for input data and output results.
# Using absolute paths is recommended for clarity and portability.
OES_DATAPATH = '/Users/sidsatya/dev/ailabor/data/bls/naics_4digit/'
CROSSWALK_DIR = '/Users/sidsatya/dev/ailabor/data/bls/crosswalks/'
OUTPUT_DIR = '/Users/sidsatya/dev/ailabor/bls_transformations/output_data1/'
CURR_DIR = '/Users/sidsatya/dev/ailabor/bls_transformations/'

# List of columns that should be treated as numeric.
NUMERIC_COLS = [
    'tot_emp', 'emp_prse', 'h_mean', 'a_mean', 'mean_prse',
    'h_pct10', 'h_pct25', 'h_median', 'h_pct75', 'h_pct90',
    'a_pct10', 'a_pct25', 'a_median', 'a_pct75', 'a_pct90'
]
# List of columns to keep for the final analysis.
COLUMNS_OF_INTEREST = ['naics', 'naics_title', 'occ_code', 'occ_title', 'bls_release_year', 'tot_emp',
                       'emp_prse', 'pct_total', 'h_mean', 'a_mean', 'mean_prse', 'h_pct10', 'h_pct25', 'h_median', 'h_pct75',
                       'h_pct90', 'a_pct10', 'a_pct25', 'a_median', 'a_pct75', 'a_pct90', 'soc_2018']

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

        # Convert naics column to int for consistency. flag any non-numeric values.
        # Safely convert 'naics' to a numeric type, reporting any errors.
        df['naics'] = df['naics'].astype(str).str.replace(r'\.0*$', '', regex=True)

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
    # Filter out rows with NaN occ_code and handle NaNs in occ_group to prevent errors.
    df = df[df['occ_code'].notna()]
    return df[(df['occ_code'] != '00-0000') &
              ~(df['occ_code'].str.endswith('0')) &
              ~(df['occ_group'].fillna('').isin(['major', 'minor', 'broad']))].copy()

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

    print(f"Found {suppression_counts[suppression_counts['pct_years_suppressed'] > 0.25].groupby(['naics', 'occ_code']).agg('count').shape[0]} unique naics-occ pairs with suppression > 25%.")
    # Print distribution of suppression counts by year
    naics_occs_suppressed = suppression_counts[suppression_counts['pct_years_suppressed'] > 0.25].copy()
    naics_occs_suppressed['is_suppressed'] = True
    df_sup = df.merge(naics_occs_suppressed[['naics', 'occ_code', 'is_suppressed']], on=['naics', 'occ_code'], how='left')
    df_sup = df_sup[df_sup['is_suppressed'].notna()].copy()
    df_sup['tot_emp'] = pd.to_numeric(
        df_sup['tot_emp'].astype(str).str.replace('**', '0', regex=False).str.replace(',', '', regex=False),
        errors='coerce'
    ).fillna(0)
    df_sup_grp = df_sup.groupby(['bls_release_year']).agg({'naics': 'count', 'tot_emp': 'sum'}).reset_index()
    print("Some data on suppressions: ")
    print(df_sup_grp.sort_values('bls_release_year'))

    # Save the suppression counts for reference.
    suppression_counts.to_csv(os.path.join(CURR_DIR, 'intermediate_data/suppression_counts.csv'), index=False)
    print("Suppression counts saved to intermediate_data/suppression_counts.csv")

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

def _create_and_save_mapping_matrix(
    oes_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    parent_col: str,
    child_col: str,
    parent_title_col: str,
    crosswalk_merge_col: str,
    output_mapping_path: str,
    output_children_path: str,
    final_parent_col_name: str,
    final_parent_title_col_name: str
):
    """
    Generates and saves a parent-child SOC mapping matrix based on employment shares.

    This helper function encapsulates the logic for processing one crosswalk file. It:
    1. Identifies parent SOC codes that map to multiple child SOC codes.
    2. Merges OES data from a "pure" SOC year with the crosswalk.
    3. Calculates total employment for each parent code within each industry (NAICS).
    4. Computes the employment share of each child code relative to its parent's total.
    5. Saves the resulting mapping table to a CSV file.
    6. Saves a list of parent codes with multiple children to another CSV file.

    Args:
        oes_df (pd.DataFrame): OES data for a "pure" SOC year (e.g., 2012 for 2010 SOC).
        crosswalk_df (pd.DataFrame): The crosswalk dataframe to process.
        parent_col (str): The column name for the parent SOC code in the crosswalk.
        child_col (str): The column name for the child SOC code in the crosswalk.
        parent_title_col (str): The column name for the parent SOC title in the crosswalk.
        crosswalk_merge_col (str): The column in the crosswalk to merge with OES `occ_code`.
        output_mapping_path (str): File path to save the final mapping matrix.
        output_children_path (str): File path to save the parent codes with multiple children.
        final_parent_col_name (str): Standardized name for the parent SOC code column in the output.
        final_parent_title_col_name (str): Standardized name for the parent SOC title column in the output.
    """
    # 1. Find parent codes that map to more than one child code.
    parent_codes_with_children = crosswalk_df.groupby(parent_col).agg(
        {child_col: ['unique', 'nunique']}
    ).reset_index()
    parent_codes_with_children.columns = [parent_col, f'Unique {child_col}s', f'Num Unique {child_col}s']
    parent_codes_with_children = parent_codes_with_children[parent_codes_with_children[f'Num Unique {child_col}s'] > 1]

    # 2. Merge OES data with the crosswalk to link child codes to parent codes.
    oes_merged = pd.merge(
        oes_df,
        crosswalk_df,
        how='left',
        left_on='occ_code',
        right_on=crosswalk_merge_col
    ).dropna(subset=[parent_col])

    # 3. For each industry-year, calculate the total employment for each parent SOC code.
    parent_sum = oes_merged.groupby(['bls_release_year', 'naics', parent_col])['tot_emp'].sum().reset_index()
    parent_sum.rename(columns={'tot_emp': 'tot_emp_parent_sum'}, inplace=True)

    # 4. Merge parent totals back and calculate each child's employment share.
    oes_merged = pd.merge(
        oes_merged,
        parent_sum,
        how='left',
        on=['bls_release_year', 'naics', parent_col]
    )
    oes_merged['naics_year_parent_emp_share'] = np.where(
        oes_merged['tot_emp_parent_sum'] == 0,
        np.nan,
        oes_merged['tot_emp'] / oes_merged['tot_emp_parent_sum']
    )

    # 5. Prepare the final mapping dataframe with selected and renamed columns.
    mapping_cols = [
        'bls_release_year', 'naics', 'naics_title', 'occ_code', 'occ_title',
        parent_col, parent_title_col, 'tot_emp', 'tot_emp_parent_sum',
        'naics_year_parent_emp_share'
    ]
    output_df = oes_merged[mapping_cols].copy()
    output_df.rename(columns={
        parent_col: final_parent_col_name,
        parent_title_col: final_parent_title_col_name
    }, inplace=True)

    # 6. Save the mapping matrix and the list of parents with children.
    output_df.to_csv(output_mapping_path, index=False)
    parent_codes_with_children.to_csv(output_children_path, index=False)
    print(f"  - Saved mapping to {os.path.basename(output_mapping_path)}")
    print(f"  - Saved parent codes with children to {os.path.basename(output_children_path)}")

def create_mapping_matrices(df, crosswalk_dir, curr_dir):
    """
    Creates and saves SOC code mapping matrices using employment data for weighting.

    This function processes four different SOC crosswalks to create mapping files.
    These files are used to harmonize occupation codes across different SOC vintages.
    It uses employment data from "pure" SOC years (2012 for 2010 SOC, 2021 for 2018 SOC)
    to calculate employment-weighted shares for parent-child code relationships. This is
    particularly important when one occupation code splits into multiple codes in a
    newer SOC version.

    The generated files are saved to an 'intermediate_data' directory.

    Args:
        df (pd.DataFrame): The main OES dataframe after initial processing.
        crosswalk_dir (str): Directory containing the raw crosswalk CSV files.
        curr_dir (str): The current working directory, used to create the output folder.
    """
    print("5. Creating mapping matrices...")

    # --- Setup ---
    # Use data from "pure" SOC years to create the mappings.
    oes_2012 = df[df['bls_release_year'] == 2012].copy()
    oes_2021 = df[df['bls_release_year'] == 2021].copy()

    # Load all necessary crosswalk files.
    crosswalks = {
        '2000_to_2010': pd.read_csv(os.path.join(crosswalk_dir, 'crosswalk_2000_to_2010.csv')),
        '2010_to_2018': pd.read_csv(os.path.join(crosswalk_dir, 'crosswalk_2010_to_2018.csv')),
        'hybrid_2000_to_2010': pd.read_csv(os.path.join(crosswalk_dir, 'crosswalk_hybrid_2000_to_2010.csv')),
        'hybrid_2010_to_2018': pd.read_csv(os.path.join(crosswalk_dir, 'crosswalk_hybrid_2010_to_2018.csv'))
    }

    # Create a directory for intermediate output files.
    intermediate_dir = os.path.join(curr_dir, 'intermediate_data')
    os.makedirs(intermediate_dir, exist_ok=True)

    # --- Process Each Crosswalk ---

    print("\nProcessing 2000-to-2010 crosswalks...")
    # 1. Standard 2000 -> 2010 crosswalk, using 2012 OES data (pure 2010 SOC).
    _create_and_save_mapping_matrix(
        oes_df=oes_2012,
        crosswalk_df=crosswalks['2000_to_2010'],
        parent_col='2000 SOC Code',
        child_col='2010 SOC Code',
        parent_title_col='2000 SOC Title',
        crosswalk_merge_col='2010 SOC Code',
        output_mapping_path=os.path.join(intermediate_dir, 'oes_2012_mapping_cw_2000_to_2010.csv'),
        output_children_path=os.path.join(intermediate_dir, 'soc_2000_codes_with_children.csv'),
        final_parent_col_name='2000 SOC Code',
        final_parent_title_col_name='2000 SOC Title'
    )

    # 2. Hybrid 2000 -> 2010 crosswalk (for 2010-2011 OES data).
    # Maps from the hybrid 'OES 2010 code' to the detailed '2010 SOC code'.
    _create_and_save_mapping_matrix(
        oes_df=oes_2012,
        crosswalk_df=crosswalks['hybrid_2000_to_2010'],
        parent_col='OES 2010 code',
        child_col='2010 SOC code',
        parent_title_col='OES 2010 Title',
        crosswalk_merge_col='2010 SOC code',
        output_mapping_path=os.path.join(intermediate_dir, 'oes_2012_mapping_cw_hybrid_to_2010.csv'),
        output_children_path=os.path.join(intermediate_dir, 'soc_hybrid_2010_codes_with_children.csv'),
        final_parent_col_name='OES Hybrid Code',
        final_parent_title_col_name='OES Hybrid Title'
    )

    print("\nProcessing 2010-to-2018 crosswalks...")
    # 3. Standard 2010 -> 2018 crosswalk, using 2021 OES data (pure 2018 SOC).
    _create_and_save_mapping_matrix(
        oes_df=oes_2021,
        crosswalk_df=crosswalks['2010_to_2018'],
        parent_col='2010 SOC Code',
        child_col='2018 SOC Code',
        parent_title_col='2010 SOC Title',
        crosswalk_merge_col='2018 SOC Code',
        output_mapping_path=os.path.join(intermediate_dir, 'oes_2021_mapping_cw_2010_to_2018.csv'),
        output_children_path=os.path.join(intermediate_dir, 'soc_2010_codes_with_children.csv'),
        final_parent_col_name='2010 SOC Code',
        final_parent_title_col_name='2010 SOC Title'
    )

    # 4. Hybrid 2010 -> 2018 crosswalk (for 2019 OES data).
    # Maps from the hybrid 'OES 2018 Estimates Code' to the detailed '2018 SOC code'.
    _create_and_save_mapping_matrix(
        oes_df=oes_2021,
        crosswalk_df=crosswalks['hybrid_2010_to_2018'],
        parent_col='OES 2018 Estimates Code',
        child_col='2018 SOC Code',
        parent_title_col='OES 2018 Estimates Title',
        crosswalk_merge_col='2018 SOC Code',
        output_mapping_path=os.path.join(intermediate_dir, 'oes_2021_mapping_cw_hybrid_2019_to_2018.csv'),
        output_children_path=os.path.join(intermediate_dir, 'soc_hybrid_2019_to_2018_codes_with_children.csv'),
        final_parent_col_name='OES Hybrid Code',
        final_parent_title_col_name='OES Hybrid Title'
    )

    # 4. Hybrid 2010 -> 2018 crosswalk (for 2020 OES data).
    # Maps from the hybrid 'OES 2018 Estimates Code' to the detailed '2018 SOC code'.
    _create_and_save_mapping_matrix(
        oes_df=oes_2021,
        crosswalk_df=crosswalks['hybrid_2010_to_2018'],
        parent_col='OES 2019 Estimates Code',
        child_col='2018 SOC Code',
        parent_title_col='OES 2019 Estimates Title',
        crosswalk_merge_col='2018 SOC Code',
        output_mapping_path=os.path.join(intermediate_dir, 'oes_2021_mapping_cw_hybrid_2020_to_2018.csv'),
        output_children_path=os.path.join(intermediate_dir, 'soc_hybrid_2020_to_2018_codes_with_children.csv'),
        final_parent_col_name='OES Hybrid Code',
        final_parent_title_col_name='OES Hybrid Title'
    )

def _build_split_weights(matrix: pd.DataFrame,
                         parent_col: str,
                         child_col: str,
                         weight_col: str = 'naics_year_parent_emp_share') -> pd.DataFrame:
    """Return a tidy DF with columns  ['naics','parent','child','share']  ."""
    out = (matrix[[
                'naics', parent_col, child_col, weight_col
            ]]
            .rename(columns={parent_col: 'parent', child_col: 'child', weight_col: 'share'})
            .dropna())
    return out

def _apply_harmonization(rows: pd.DataFrame,
                         mapping_matrix: pd.DataFrame,
                         parents_with_children: list,
                         one2one_map: Dict[str, str],
                         target_soc_col: str,
                         source_soc_col: str = 'occ_code') -> pd.DataFrame:
    """
    Harmonizes SOC codes for a given set of rows by splitting or mapping them.

    This function takes rows with older SOC codes and converts them to a target SOC vintage.
    It handles two cases:
    1.  One-to-one mapping: The old code maps directly to a new code.
    2.  One-to-many split: The old code splits into multiple new codes. In this case,
        the function uses pre-calculated employment shares to distribute `tot_emp`
        across the new codes.

    Args:
        rows: DataFrame subset containing the rows to be harmonized.
        mapping_matrix: A DataFrame with ['naics', 'parent', 'child', 'share'] columns,
                        used for splitting parent codes.
        parents_with_children: A list of parent codes that split.
        one2one_map: A dict for simple one-to-one code mappings.
        target_soc_col: The name of the new column for the harmonized SOC codes.
        source_soc_col: The column containing the codes to be harmonized.

    Returns:
        A DataFrame with the harmonized rows, including the new SOC code column and
        adjusted employment figures.
    """
    rows = rows.copy()
    rows['parent_has_split'] = rows[source_soc_col].isin(parents_with_children)

    # --- Case 1: Rows that DO NOT split (simple 1-to-1 mapping) ---
    simple = rows[~rows['parent_has_split']].copy()
    simple[target_soc_col] = simple[source_soc_col].map(one2one_map)

    # Warn about and drop any codes that couldn't be mapped.
    missing = simple[target_soc_col].isna()
    if missing.any():
        unmapped = simple.loc[missing, [source_soc_col, 'tot_emp', 'bls_release_year']]
        print(f"[WARN] Dropping {len(unmapped)} unmapped codes from '{source_soc_col}':\n{unmapped.groupby([source_soc_col, 'bls_release_year']).agg('sum').head(25)}")
        simple = simple[~missing]
    simple['naics_year_parent_emp_share'] = 1.0

    # --- Case 2: Rows that DO split (1-to-many mapping) ---
    split_parents = rows[rows['parent_has_split']]
    if split_parents.empty:
        exploded = pd.DataFrame(columns=simple.columns)
    else:
        # Merge with the mapping matrix to get child codes and employment shares.
        # This will "explode" the rows, creating a new row for each child code.
        
        # first, convert mapping_matrix naics to str
        mapping_matrix['naics'] = mapping_matrix['naics'].astype(str)

        exploded = (split_parents
                    .merge(mapping_matrix,
                           left_on=['naics', source_soc_col],
                           right_on=['naics', 'parent'],
                           how='left'))

        # Fallback for rare cases where a share is missing: distribute employment equally.
        mask_no_share = exploded['share'].isna()
        if mask_no_share.any():
            exploded.loc[mask_no_share, 'share'] = (
                1.0 / exploded.groupby(['naics', source_soc_col])['child'].transform('count')
            )

        exploded[target_soc_col] = exploded['child']
        exploded['naics_year_parent_emp_share'] = exploded['share']
        # Distribute the total employment based on the share.
        # tot_emp is assumed to be numeric at this stage from `convert_data_types`.
        exploded['tot_emp'] = exploded['tot_emp'] * exploded['share']
        exploded['tot_emp'].replace([np.inf, -np.inf], np.nan, inplace=True)
        exploded = exploded.drop(columns=['parent', 'child', 'share'])

    # --- Combine and return ---
    return pd.concat([simple, exploded], ignore_index=True)

def _harmonize_to_2010(df: pd.DataFrame, crosswalk_dir: str, intermediate_dir: str) -> pd.DataFrame:
    """Harmonizes SOC codes from 2000-vintage to 2010-vintage."""
    print("6.1. Harmonizing SOC codes from 2000-vintage to 2010...")
    df = df.copy()

    # --- Load crosswalks and mapping matrices ---
    cw00 = pd.read_csv(os.path.join(crosswalk_dir, 'crosswalk_2000_to_2010.csv'), dtype=str)
    one2one_00_10 = dict(zip(cw00['2000 SOC Code'].str.strip(), cw00['2010 SOC Code'].str.strip()))

    cw00h = pd.read_csv(os.path.join(crosswalk_dir, 'crosswalk_hybrid_2000_to_2010.csv'), dtype=str)
    one2one_00h_10 = dict(zip(cw00h['OES 2010 code'].str.strip(), cw00h['2010 SOC code'].str.strip()))

    mm_split = _build_split_weights(
        pd.read_csv(os.path.join(intermediate_dir, 'oes_2012_mapping_cw_2000_to_2010.csv')),
        '2000 SOC Code', 'occ_code')
    mm_split_h = _build_split_weights(
        pd.read_csv(os.path.join(intermediate_dir, 'oes_2012_mapping_cw_hybrid_to_2010.csv')),
        'OES Hybrid Code', 'occ_code')

    parents_children = mm_split.groupby('parent')['child'].nunique()
    parents_children = parents_children[parents_children > 1].index.to_list()

    parents_children_h = mm_split_h.groupby('parent')['child'].nunique()
    parents_children_h = parents_children_h[parents_children_h > 1].index.to_list()

    # --- Slice DataFrame by year and apply harmonization ---
    df_to_harmonize = df[df['bls_release_year'] < 2012]
    df_post_2012 = df[df['bls_release_year'] >= 2012].copy()
    df_post_2012['soc_2010'] = df_post_2012['occ_code']

    print("Applying harmonization to pre-2009 OES data...")
    df_pre09 = df_to_harmonize[df_to_harmonize['bls_release_year'] <= 2009]
    mapped_pre09 = _apply_harmonization(df_pre09, mm_split, parents_children, one2one_00_10, 'soc_2010')

    print("Applying harmonization to 2010-2011 OES data...")
    df_1011 = df_to_harmonize[df_to_harmonize['bls_release_year'].isin([2010, 2011])]
    mapped_1011 = _apply_harmonization(df_1011, mm_split_h, parents_children_h, one2one_00h_10, 'soc_2010')

    return pd.concat([df_post_2012, mapped_pre09, mapped_1011], ignore_index=True)

def _harmonize_to_2018(df: pd.DataFrame, crosswalk_dir: str, intermediate_dir: str) -> pd.DataFrame:
    """Harmonizes SOC codes from 2010-vintage to 2018-vintage."""
    print("6.2. Harmonizing SOC codes from 2010-vintage to 2018...")
    df = df.copy()

    # --- Load crosswalks and mapping matrices ---
    cw10 = pd.read_csv(os.path.join(crosswalk_dir, 'crosswalk_2010_to_2018.csv'), dtype=str)
    one2one_10_18 = dict(zip(cw10['2010 SOC Code'].str.strip(), cw10['2018 SOC Code'].str.strip()))
    mm_split_10 = _build_split_weights(
        pd.read_csv(os.path.join(intermediate_dir, 'oes_2021_mapping_cw_2010_to_2018.csv')),
        '2010 SOC Code', 'occ_code')
    parents_children_10 = mm_split_10.groupby('parent')['child'].nunique()
    parents_children_10 = parents_children_10[parents_children_10 > 1].index.to_list()

    cw10h = pd.read_csv(os.path.join(crosswalk_dir, 'crosswalk_hybrid_2010_to_2018.csv'), dtype=str)
    
    one2one_10h_18_2019 = dict(zip(cw10h['OES 2018 Estimates Code'].str.strip(), cw10h['2018 SOC Code'].str.strip()))
    mm_split_h2019 = _build_split_weights(
        pd.read_csv(os.path.join(intermediate_dir, 'oes_2021_mapping_cw_hybrid_2019_to_2018.csv')),
        'OES Hybrid Code', 'occ_code')
    parents_children_h2019 = mm_split_h2019.groupby('parent')['child'].nunique()
    parents_children_h2019 = parents_children_h2019[parents_children_h2019 > 1].index.to_list()

    one2one_10h_18_2020 = dict(zip(cw10h['OES 2019 Estimates Code'].str.strip(), cw10h['2018 SOC Code'].str.strip()))
    mm_split_h2020 = _build_split_weights(
        pd.read_csv(os.path.join(intermediate_dir, 'oes_2021_mapping_cw_hybrid_2020_to_2018.csv')),
        'OES Hybrid Code', 'occ_code')
    parents_children_h2020 = mm_split_h2020.groupby('parent')['child'].nunique()
    parents_children_h2020 = parents_children_h2020[parents_children_h2020 > 1].index.to_list()

    # --- Slice DataFrame by year and apply harmonization ---
    df_post_2021 = df[df['bls_release_year'] >= 2021].copy()
    df_post_2021['soc_2018'] = df_post_2021['occ_code']
    print("Apply harmonization to 2017 OES data...")
    df_17 = df[df['bls_release_year'] == 2017].copy()
    mapped_17 = _apply_harmonization(df_17, mm_split_h2019, parents_children_h2019, one2one_10h_18_2019, 'soc_2018')

    print("Applying harmonization to 2018 OES data...")
    df_18 = df[df['bls_release_year'] == 2018].copy()
    mapped_18 = _apply_harmonization(df_18, mm_split_h2019, parents_children_h2019, one2one_10h_18_2019, 'soc_2018')

    print("Applying harmonization to 2019 OES data...")
    df_19 = df[df['bls_release_year']==2019].copy()
    mapped_19 = _apply_harmonization(df_19, mm_split_h2020, parents_children_h2020, one2one_10h_18_2020, 'soc_2018')

    print("Applying harmonization to 2020 OES data...")
    df_20 = df[df['bls_release_year']==2020].copy()
    mapped_20 = _apply_harmonization(df_20, mm_split_h2020, parents_children_h2020, one2one_10h_18_2020, 'soc_2018')

    print("Applying harmonization to pre-2019 OES data...")
    df_pre19 = df[df['bls_release_year'] < 2017].copy()
    mapped_pre19 = _apply_harmonization(df_pre19, mm_split_10, parents_children_10, one2one_10_18, 'soc_2018', source_soc_col='soc_2010')

    return pd.concat([df_post_2021, mapped_17, mapped_18, mapped_19, mapped_20, mapped_pre19], ignore_index=True)

def harmonize_soc_codes_outer(df, crosswalk_dir):
    """
    Harmonizes SOC codes in the dataframe using crosswalks and employment-weighted mappings.

    This function orchestrates a two-step harmonization process:
    1.  Harmonizes all SOC codes from pre-2012 data (2000 and hybrid vintages) to the 2010 SOC standard.
    2.  Harmonizes all SOC codes from pre-2021 data (2010 and hybrid vintages) to the 2018 SOC standard.

    This process handles cases where occupation codes split into multiple new codes by distributing
    employment figures based on weights calculated from "pure" SOC years.

    Args:
        df (pd.DataFrame): The input dataframe with `occ_code` and `bls_release_year`.
        crosswalk_dir (str): The directory containing the crosswalk files.

    Returns:
        pd.DataFrame: The dataframe with new 'soc_2010' and 'soc_2018' columns containing
                      the fully harmonized codes.
    """
    print(f"6. Harmonizing SOC codes, starting with {len(df)} rows...")
    intermediate_dir = os.path.join(CURR_DIR, 'intermediate_data')

    # Step 1: Harmonize everything to 2010 SOC codes
    df_h2010 = _harmonize_to_2010(df, crosswalk_dir, intermediate_dir)
    print(f"  - After harmonizing to 2010 SOC, dataframe has {len(df_h2010)} rows.")

    # Step 2: Harmonize everything to 2018 SOC codes
    df_h2018 = _harmonize_to_2018(df_h2010, crosswalk_dir, intermediate_dir)
    print(f"  - After harmonizing to 2018 SOC, dataframe has {len(df_h2018)} rows.")

    return df_h2018

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
    print("7. Calculating employment shares...")
    df_final = df[COLUMNS_OF_INTEREST].copy()
    print(len(df_final), df_final.columns)

    print(df_final[df_final['tot_emp'] == np.inf])

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

    print(df_merged)

    # --- Verification Step ---
    # Verify that the calculated shares sum to approximately 1 within their respective groups.
    # This is a sanity check to ensure the calculations are correct.
    
    # Check yearly totals
    year_share_sums = df_merged.groupby('bls_release_year')['pct_year_tot_emp'].sum()
    if not np.allclose(year_share_sums, 1.0):
        print("[WARN] Yearly employment shares do not sum to 1 for all years:")
        print(year_share_sums[~np.isclose(year_share_sums, 1.0)])
    else:
        print("  - Verified: Yearly employment shares sum to ~1.0 for all years.")

    # Check industry-year totals
    naics_year_share_sums = df_merged.groupby(['naics', 'bls_release_year'])['pct_naics_year_tot_emp'].sum()
    if not np.allclose(naics_year_share_sums, 1.0):
        print("[WARN] NAICS-level yearly employment shares do not sum to 1 for all groups.")
        # Optionally, print the groups that do not sum to 1 for debugging.
        # print(naics_year_share_sums[~np.isclose(naics_year_share_sums, 1.0)])
    else:
        print("  - Verified: NAICS-level yearly employment shares sum to ~1.0 for all groups.")
    

    return df_merged

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

    # Step 5: Compute and save mapping matrices 
    create_mapping_matrices(oes_data_typed, CROSSWALK_DIR, CURR_DIR)

    # Step 6: Harmonize Standard Occupational Classification (SOC) codes across different years.
    oes_data_harmonized = harmonize_soc_codes_outer(oes_data_typed, CROSSWALK_DIR)

    # Step 7: Calculate employment shares relative to various totals (yearly, industry-yearly).
    oes_data_shares = calculate_employment_shares(oes_data_harmonized)

    # Step 8: Aggregate data by the harmonized 2018 SOC codes for consistent time-series analysis.
    oes_data_2018_agg = aggregate_by_soc(oes_data_shares)

    # Step 9: Perform a specific analysis on the healthcare sector and merge results.
    oes_data_2018_final, oes_data_healthcare = analyze_healthcare_sector(oes_data_2018_agg)

    # Step 10: Save the final processed dataframes to CSV files for future use.
    save_results(oes_data_harmonized, oes_data_2018_final, oes_data_healthcare, OUTPUT_DIR)

if __name__ == "__main__":
    main()