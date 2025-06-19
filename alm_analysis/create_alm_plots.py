import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import numpy as np
import itertools

def filter_data(data, emp_share_col): 
    '''
    1. Drop observations that do not have a task_intensity defined
    2. Drop observations that do not have a relevant employment_share_defined
    3. Drop any occ-year combination that has any unclassified tasks. 
    '''
    # Drop all observations that do not have a task intensity defined
    data_drop_intensities = data.dropna(subset=['task_intensity']).copy()

    # Drop all observations that do not have a pct_year_tot_emp defined
    data_drop_emp = data_drop_intensities.dropna(subset=[emp_share_col]).copy()

    # Drop occ-year combinations that have any unclassified tasks
    data_drop_emp['task_classified'] = data_drop_emp[['interpersonal_mean', 'routine_mean', 'manual_mean', 'high_codifiable_mean']].notna().all(axis=1)
    grp_occ_year = data_drop_emp.groupby(['O*NET 2018 SOC Code', 'ONET_release_year']).agg({'task_classified': ['count', 'sum']}).reset_index()
    grp_occ_year.columns = ['O*NET 2018 SOC Code', 'ONET_release_year', 'task_classified_count', 'task_classified_sum']
    grp_occ_year_valid = grp_occ_year[grp_occ_year['task_classified_sum'] == grp_occ_year['task_classified_count']]
    data_final = pd.merge(data_drop_emp, grp_occ_year_valid[['O*NET 2018 SOC Code', 'ONET_release_year']], on=['O*NET 2018 SOC Code', 'ONET_release_year'], how='inner')

    print("There are {} occupations in the filtered data that we can use for further analysis.".format(len(data_final.groupby('O*NET 2018 SOC Code'))))

    return data_final

def get_all_possible_classifications(choices_per_dimension): 
    '''
    Given a list of choices for each dimension, generate all possible classifications.
    For example, if choices_per_dimension = [['P', 'NP'], ['NR', 'R'], ['M', 'NM']], then the output will be:
    ['P-NR-M', 'P-NR-NM', 'P-R-M', 'P-R-NM', 'NP-NR-M', 'NP-NR-NM', 'NP-R-M', 'NP-R-NM']
    '''
    all_classifications = []

    # Generate all combinations
    combinations = list(itertools.product(*choices_per_dimension))

    # Join each combination into a string like 'P-NR-M'
    all_classifications = ['-'.join(combo) for combo in combinations]
    return all_classifications

def weighted_ecdf(values, weights):
    """
    Compute the weighted empirical CDF.
    
    Parameters:
    - values: 1D array of observations (e.g., occupation indices in baseline year).
    - weights: 1D array of non-negative weights (e.g., employment shares), same length as values.
    
    Returns:
    - x_ecdf: sorted values
    - cdf_vals: corresponding cumulative probabilities (sums to 1)
    """
    # Sort values and associated weights
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]
    
    # Build cumulative distribution
    cum_weights = np.cumsum(weights_sorted)
    cum_weights /= cum_weights[-1]  # normalize to 1
    
    return values_sorted, cum_weights

def percentile_of(x_new, x_ecdf, cdf_vals):
    """
    Interpolate a new value onto the ECDF to get its percentile.
    
    Parameters:
    - x_new: scalar or array of new observations you want percentiles for.
    - x_ecdf: sorted baseline values from weighted_ecdf.
    - cdf_vals: ECDF probabilities corresponding to x_ecdf.
    
    Returns:
    - p: percentile(s) between 0 and 1 (or array of same shape as x_new).
    """
    return np.interp(x_new, x_ecdf, cdf_vals, left=0.0, right=1.0)

def percentile_of_midpoint(x_new, x_ecdf, cdf_vals, weights_norm):
    """
    Percentile based on the *mid-point* of the ECDF step that x_new belongs to.
    """
    idx = np.searchsorted(x_ecdf, x_new, side='right') - 1   # index of step
    if idx < 0:                      # x_new below all data
        return 0.0
    cdf_lower = 0.0 if idx == 0 else cdf_vals[idx-1]
    return cdf_lower + weights_norm[idx]/2.0                 # mid-point

def create_empirical_distributions(data_min_year, weight_col, all_classifications): 
    '''
    For every occupation in the minimum year, compute the percentiles of task intensity, weighted by employment share
    '''

    # Group by occupation-classification and sum task intensity and average the employment share
    grp_occ_class = data_min_year.groupby(['O*NET 2018 SOC Code', 'classification']).agg({
        'task_intensity': 'sum',
        weight_col: 'mean'
    }).reset_index()

    # Check to make sure each occ-class exists otherwise add a row with 0 task intensity and the mean employment share
    new_rows = []
    for occ_code in grp_occ_class['O*NET 2018 SOC Code'].unique():
        for classification in all_classifications:
            if not ((grp_occ_class['O*NET 2018 SOC Code'] == occ_code) & (grp_occ_class['classification'] == classification)).any():
                new_row = {
                    'O*NET 2018 SOC Code': occ_code,
                    'classification': classification,
                    'task_intensity': 0,
                    weight_col: grp_occ_class[grp_occ_class['O*NET 2018 SOC Code'] == occ_code][weight_col].mean()
                }
                new_rows.append(new_row)

    if new_rows:
        grp_occ_class = pd.concat(
            [grp_occ_class, pd.DataFrame(new_rows)],
            ignore_index=True
        )

    # Compute ECDF for each classification dimension
    ecdf_results = {}
    for classification in all_classifications:
        # Filter for the current classification
        filtered_data = grp_occ_class[grp_occ_class['classification'] == classification]

        # Get values and weights
        values = filtered_data['task_intensity'].values
        weights = filtered_data[weight_col].values

        # Compute ECDF
        x_ecdf, cdf_vals = weighted_ecdf(values, weights)
        weights_norm = weights[np.argsort(values)] / weights.sum()

        # Store results
        ecdf_results[classification] = (x_ecdf, cdf_vals, weights_norm)
    return ecdf_results

def compute_percentiles_for_classification(val, ecdf_results, classification): 
    """
    Compute percentiles for each task intensity based on the empirical distributions.
    
    Parameters:
    - data: DataFrame with task intensity and employment share.
    - ecdf_results: Dictionary of ECDF results from create_empirical_distributions.

    """

    x_ecdf, cdf_vals, w_norm = ecdf_results[classification]
    return percentile_of_midpoint(val, x_ecdf, cdf_vals, w_norm)
    
def create_filtered_data_grp(data, emp_share_col, all_classifications):
    # 1. collapse to occupation-year-bucket
    occ_year_cls = (
        data.groupby(['O*NET 2018 SOC Code', 'ONET_release_year', 'classification'],
                     as_index=False)
            .agg(task_intensity=('task_intensity', 'sum'))
    )

    # 2. total intensity per occupation-year  (denominator for shares)
    total_int = (occ_year_cls.groupby(['O*NET 2018 SOC Code', 'ONET_release_year'],
                                      as_index=False)['task_intensity']
                               .sum()
                               .rename(columns={'task_intensity':'total_intensity'}))

    # 3. bring in employment share
    occ_year_emp = (
        data.groupby(['O*NET 2018 SOC Code', 'ONET_release_year'], as_index=False)[emp_share_col]
            .mean()                  # one number per occ-year
    )

    # 4. full grid  (occ-year × all buckets)
    base = occ_year_emp.merge(total_int, on=['O*NET 2018 SOC Code', 'ONET_release_year'])
    full_grid = (
        base.assign(key=1)
            .merge(pd.DataFrame({'classification': all_classifications, 'key': 1}),
                   on='key')
            .drop('key', axis=1)
    )

    # 5. merge in observed intensities
    df = full_grid.merge(
            occ_year_cls,
            on=['O*NET 2018 SOC Code', 'ONET_release_year', 'classification'],
            how='left'
        )
    df['task_intensity'] = df['task_intensity'].fillna(0.0)

    # 6. compute bucket weight  w = emp_share * intensity / total_intensity
    df['bucket_weight'] = np.where(
        df['total_intensity'] > 0,
        df[emp_share_col] * df['task_intensity'] / df['total_intensity'],
        0.0
    )

    return df



def create_alm_plots(data, all_classifications, y_col, save_file, output_dir='output_plots'):
    """
    Create ALM plots for task intensity distributions across different classifications.
    
    Parameters:
    - data: DataFrame with task intensity and employment share.
    - y_col: Column name for plotting.
    - output_dir: Directory to save the plots.
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # For each year, compute the mean percentile for each classification
    data_year_class_grp = (
        filtered_data_grp
        .groupby(['ONET_release_year', 'classification'])
        .apply(lambda df: np.average(df['percentile'],
                                    weights=df['bucket_weight']))
        .reset_index(name='percentile')
    )
    # Plot the percentiles for each classification over the years
    plt.figure(figsize=(12, 8))

    num_classes = len(all_classifications)
    colors = cm.get_cmap('tab20', num_classes)

    for idx, classification in enumerate(all_classifications):
        subset = data_year_class_grp[data_year_class_grp['classification'] == classification]
        # Vary alpha (opacity) between 0.5 and 1.0 for visual distinction
        alpha = 0.5 + 0.5 * (idx / max(1, num_classes - 1))
        plt.plot(
            subset['ONET_release_year'],
            subset[y_col],
            marker='o',
            label=classification,
            color=colors(idx),
            alpha=alpha
        )
    plt.title('Percentiles of Mean Task Contribution by Classification Over Years')
    plt.xlabel('Year')
    plt.ylabel('Percentile')
    plt.xticks(subset['ONET_release_year'].unique())
    plt.legend(title='Classification')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_file))







##########################################################
##########################################################
##########################################################
ailabor_root = '/Users/sidsatya/dev/ailabor'

classification_types_and_cats = {'alm_classification': [['R', 'NR'], ['I', 'P'], ['M', 'NM']], 
                                 'code_classification': [['HC', 'LC'], ['I', 'P'], ['M', 'NM']], 
                                 'full_classification': [['R', 'NR'], ['I', 'P'], ['M', 'NM'], ['HC', 'LC']]} 



classification_type = 'alm_classification'
all_classifications = get_all_possible_classifications([['R', 'NR'], ['I', 'P'], ['M', 'NM']])
in_file = 'alm_analysis/data/combined_healthcare.csv'
out_file = 'alm_plot_healthcare_dataset.png'

# Generate ALM plot for full dataset
full_data = pd.read_csv(os.path.join(ailabor_root, in_file))
full_data['classification'] = full_data[classification_type]

# Filter the data
filtered_data = filter_data(full_data, 'pct_healthcare_tot_emp')
filtered_data_grp = create_filtered_data_grp(filtered_data, 'pct_healthcare_tot_emp', all_classifications)

# Create empirical distributions
filtered_data_2003 = filtered_data_grp[filtered_data_grp['ONET_release_year'] == 2003].copy()
ecdf_results = create_empirical_distributions(filtered_data_2003, 'bucket_weight', all_classifications)

# Compute percentiles for each observation for each classification
for c in all_classifications:
    mask = filtered_data_grp['classification'] == c
    # use task_intensity
    filtered_data_grp.loc[mask, 'percentile'] = (
        filtered_data_grp.loc[mask, 'task_intensity']
            .apply(lambda x: compute_percentiles_for_classification(x,
                                                                    ecdf_results,
                                                                    c))
    )

# Create ALM plots
create_alm_plots(filtered_data_grp, all_classifications, 'percentile', out_file, output_dir=os.path.join(ailabor_root, 'results/alm_classification_results/'))
