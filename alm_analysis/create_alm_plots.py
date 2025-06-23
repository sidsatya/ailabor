import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import numpy as np
import itertools

def filter_data(data, emp_share_col, intensity_col): 
    '''
    1. Drop observations that do not have a task_intensity defined
    2. Drop observations that do not have a relevant employment_share_defined
    3. Drop any occ-year combination that has any unclassified tasks. 
    '''
    # Drop all observations that do not have a task intensity defined
    data_drop_intensities = data.dropna(subset=[intensity_col]).copy()

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

def diagnostics_missing(df, emp_col, out_prefix):
    # mark per-row classification success
    df = df.copy()
    df['row_classified'] = df[
        ['interpersonal_mean','routine_mean','manual_mean','high_codifiable_mean']
    ].notna().all(axis=1)

    

    # occ-year summary
    occyr = (df.groupby(['O*NET 2018 SOC Code','ONET_release_year'], as_index=False)
               .agg(total_rows      =('row_classified','size'),
                    classified_rows =('row_classified','sum'),
                    emp_share       =(emp_col,'first')))

    occyr['kept'] = occyr['total_rows'] == occyr['classified_rows']
    print(occyr)

    # ---------- console summary ----------
    print("\n===== missing-task diagnostics =====")
    print(f"total occ-years: {len(occyr)}")
    print(f"kept occ-years : {occyr['kept'].sum()}")
    print(f"dropped occ-yrs: {(~occyr['kept']).sum()}")

    yearly = (occyr.groupby(['ONET_release_year','kept'])
                    .agg(emp =('emp_share','sum'))
                    .reset_index()
                    .pivot(index='ONET_release_year', columns='kept', values='emp')
                    .rename(columns={True:'emp_kept', False:'emp_dropped'}))
    try: 
        yearly['pct_dropped'] = yearly['emp_dropped'] / (yearly['emp_kept']+yearly['emp_dropped'])
    except KeyError: 
        yearly['pct_dropped'] = np.nan
    print("\nshare of HC employment dropped by year:")
    print(yearly.head(25))

    # ---------- write to disk ----------
    os.makedirs('diagnostics', exist_ok=True)
    yearly.to_csv(f'diagnostics/{out_prefix}_yearly_coverage.csv')

    dropped_top = (occyr[~occyr['kept']]
                   .groupby('O*NET 2018 SOC Code', as_index=False)
                   .agg(avg_emp=('emp_share','mean'),
                        years   =('ONET_release_year','nunique'))
                   .sort_values('avg_emp', ascending=False)
                   .head(30))
    dropped_top.to_csv(f'diagnostics/{out_prefix}_top_dropped_occ.csv', index=False)
    print(f"\n(top 30 dropped occupations written to diagnostics/{out_prefix}_top_dropped_occ.csv)")

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

def create_empirical_distributions(data_min_year, weight_col, all_classifications, intensity_col='task_intensity'): 
    '''
    For every occupation in the minimum year, compute the percentiles of task intensity, weighted by employment share
    '''

    # Group by occupation-classification and sum task intensity and average the employment share
    grp = (data_min_year
           .groupby(['O*NET 2018 SOC Code','classification'], as_index=False)
           .agg(intensity_col=(intensity_col,'sum'),
                weight=('bucket_weight','first')))

    # Check if the intensity_col is in the columns
    if 'intensity_col' in grp.columns:
        # rename to the value of intensity_col
        grp = grp.rename(columns={'intensity_col': intensity_col})

    # Check to make sure each occ-class exists otherwise add a row with 0 task intensity and the mean employment share
    # add missing buckets with zero intensity *and zero weight*
    missing = []
    for occ in grp['O*NET 2018 SOC Code'].unique():
        for cls in all_classifications:
            if not ((grp['O*NET 2018 SOC Code']==occ) & (grp['classification']==cls)).any():
                missing.append({'O*NET 2018 SOC Code':occ,
                                'classification':cls,
                                intensity_col:0.0,
                                'weight':0.0})        # ← weight 0, not mean
    if missing:
        grp = pd.concat([grp, pd.DataFrame(missing)], ignore_index=True)

    ecdf = {}
    for cls in all_classifications:
        sub = grp[grp['classification']==cls]
        x, cdf = weighted_ecdf(sub[intensity_col].values,
                               sub['weight'].values)
        w_norm = sub['weight'].values[np.argsort(sub[intensity_col].values)]
        w_norm = w_norm / w_norm.sum()                # safety
        ecdf[cls] = (x, cdf, w_norm)
    return ecdf

def compute_percentiles_for_classification(val, ecdf_results, classification, intensity_col='task_intensity'): 
    """
    Compute percentiles for each task intensity based on the empirical distributions.
    
    Parameters:
    - data: DataFrame with task intensity and employment share.
    - ecdf_results: Dictionary of ECDF results from create_empirical_distributions.

    """

    x_ecdf, cdf_vals, w_norm = ecdf_results[classification]
    return percentile_of_midpoint(val, x_ecdf, cdf_vals, w_norm)
    
def create_filtered_data_grp(data, emp_share_col, all_classifications, intensity_col='task_intensity'):
    # 1. collapse to occupation-year-bucket
    occ_year_cls = (
        data.groupby(['O*NET 2018 SOC Code', 'ONET_release_year', 'classification'],
                     as_index=False)
            .agg(intensity_col=(intensity_col, 'sum'))
    )

    # Check if the intensity_col is in the columns
    if 'intensity_col' in occ_year_cls.columns:
        # rename to the value of intensity_col
        occ_year_cls = occ_year_cls.rename(columns={'intensity_col': intensity_col})

    # 2. total intensity per occupation-year  (denominator for shares)
    total_int = (occ_year_cls.groupby(['O*NET 2018 SOC Code', 'ONET_release_year'],
                                      as_index=False)[intensity_col]
                               .sum()
                               .rename(columns={intensity_col:'total_intensity'}))

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
    df[intensity_col] = df[intensity_col].fillna(0.0)

    # 6. compute bucket weight  w = emp_share * intensity / total_intensity
    df['bucket_weight'] = np.where(
        df['total_intensity'] > 0,
        df[emp_share_col] * df[intensity_col] / df['total_intensity'],
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
        data
        .groupby(['ONET_release_year', 'classification'])
        .apply(lambda df: np.average(df['percentile'],
                                    weights=df['bucket_weight']), include_groups=False)
        .reset_index(name='percentile')
    )
    # Plot the percentiles for each classification over the years
    plt.figure(figsize=(12, 8))

    # Define color mapping based on classification labels
    red_classifications = sorted([c for c in all_classifications if 'R' in c or 'HC' in c])
    blue_classifications = sorted([c for c in all_classifications if 'NR' in c or 'LC' in c])
    green_classifications = blue_classifications # Keep red vs green logic

    num_red = len(red_classifications)
    num_green = len(green_classifications)

    # Get colormaps for reddish/orangish and greenish colors
    red_cmap = plt.get_cmap('YlOrRd')
    green_cmap = plt.get_cmap('YlGn') # Changed to a green colormap

    # Generate colors from the colormaps, avoiding the lightest shades
    red_colors = red_cmap(np.linspace(0.3, 1.0, num_red)) if num_red > 0 else []
    green_colors = green_cmap(np.linspace(0.3, 1.0, num_green)) if num_green > 0 else []

    color_map = {}
    for i, classification in enumerate(red_classifications):
        color_map[classification] = red_colors[i]
    for i, classification in enumerate(green_classifications):
        color_map[classification] = green_colors[i]

    for classification in sorted(all_classifications):
        subset = data_year_class_grp[data_year_class_grp['classification'] == classification]
        if subset.empty:
            continue
        
        color = color_map.get(classification, 'gray') # Default for un-matched

        plt.plot(
            subset['ONET_release_year'],
            subset[y_col],
            marker='o',
            label=classification,
            color=color,
            alpha=0.5 # Use a constant alpha for all lines
        )
    plt.title('Percentiles of Mean Task Contribution by Classification Over Years')
    plt.xlabel('Year')
    plt.ylabel('Percentile')
    plt.xticks(data_year_class_grp['ONET_release_year'].unique())
    plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_file), bbox_inches='tight')

def create_between_occ_composition_plots(data, all_classifications, y_col, emp_share_col, intensity_col, save_file, output_dir='output_plots'):
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
        data
        .groupby(['ONET_release_year', 'classification'])
        .apply(lambda df: (df[emp_share_col] * df[intensity_col]).sum())
        .reset_index(name='percentile')
    )
    
    # Plot the percentiles for each classification over the years
    plt.figure(figsize=(12, 8))

    # Define color mapping based on classification labels
    red_classifications = sorted([c for c in all_classifications if 'R' in c or 'HC' in c])
    blue_classifications = sorted([c for c in all_classifications if 'NR' in c or 'LC' in c])
    green_classifications = blue_classifications # Keep red vs green logic

    num_red = len(red_classifications)
    num_green = len(green_classifications)

    # Get colormaps for reddish/orangish and greenish colors
    red_cmap = plt.get_cmap('YlOrRd')
    green_cmap = plt.get_cmap('YlGn') # Changed to a green colormap

    # Generate colors from the colormaps, avoiding the lightest shades
    red_colors = red_cmap(np.linspace(0.3, 1.0, num_red)) if num_red > 0 else []
    green_colors = green_cmap(np.linspace(0.3, 1.0, num_green)) if num_green > 0 else []

    color_map = {}
    for i, classification in enumerate(red_classifications):
        color_map[classification] = red_colors[i]
    for i, classification in enumerate(green_classifications):
        color_map[classification] = green_colors[i]

    for classification in sorted(all_classifications):
        subset = data_year_class_grp[data_year_class_grp['classification'] == classification]
        if subset.empty:
            continue
        
        color = color_map.get(classification, 'gray') # Default for un-matched
        plt.plot(
            subset['ONET_release_year'],
            subset[y_col],
            marker='o',
            label=classification,
            color=color,
            alpha=0.5 # Use a constant alpha for all lines
        )
    plt.title('Mean Task Contribution by Classification Over Years')
    plt.xlabel('Year')
    plt.ylabel('Mean Task Contribution')
    plt.xticks(data_year_class_grp['ONET_release_year'].unique())
    plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_file), bbox_inches='tight')



##########################################################
##########################################################
##########################################################
ailabor_root = '/Users/sidsatya/dev/ailabor'

classification_types_and_cats = {'alm_classification': [['R', 'NR'], ['I', 'P'], ['M', 'NM']], 
                                 'code_classification': [['HC', 'LC'], ['I', 'P'], ['M', 'NM']]}
                                 # 'full_classification': [['R', 'NR'], ['I', 'P'], ['M', 'NM'], ['HC', 'LC']]} 

for scope in ['full', 'healthcare']: 
    for task_scope in ['all', 'core']: 
        for classification_type, choices_per_dimension in classification_types_and_cats.items():
            print("#" * 50)
            print(f"Processing {classification_type} for scope {scope} and task scope {task_scope}")
    
            # Generate all possible classifications
            all_classifications = get_all_possible_classifications(choices_per_dimension)

            # Define input and output file paths
            in_file = f'alm_analysis/data1/combined_{scope}_{task_scope}.csv'
            out_file = f'{classification_type}_plot_{scope}_{task_scope}_tasks.png'

            emp_share_col = 'pct_healthcare_tot_emp' if scope == 'healthcare' else 'pct_year_tot_emp'
            intensity_col = 'task_intensity' if task_scope == 'all' else 'task_intensity_core'

            # Generate ALM plot for full dataset
            full_data = pd.read_csv(os.path.join(ailabor_root, in_file))
            full_data['classification'] = full_data[classification_type]

            # Limit data to 2005 and later
            baseline_year = 2006
            full_data = full_data[full_data['ONET_release_year'] >= 2006].copy()

            # Filter the data
            diagnostics_missing(full_data, emp_share_col,
                    out_prefix=f'{scope}_{task_scope}_{classification_type}')

            filtered_data = filter_data(full_data, emp_share_col, intensity_col)
            filtered_data_grp = create_filtered_data_grp(filtered_data, emp_share_col, all_classifications, intensity_col)

            # Create empirical distributions
            filtered_data_min_year = filtered_data_grp[filtered_data_grp['ONET_release_year'] == baseline_year].copy()
            ecdf_results = create_empirical_distributions(filtered_data_min_year, 'bucket_weight', all_classifications, intensity_col)

            # Compute percentiles for each observation for each classification
            for c in all_classifications:
                mask = filtered_data_grp['classification'] == c
                # use task_intensity
                filtered_data_grp.loc[mask, 'percentile'] = (
                    filtered_data_grp.loc[mask, intensity_col]
                        .apply(lambda x: compute_percentiles_for_classification(x,
                                                                                ecdf_results,
                                                                                c))
                )

            baseline = (filtered_data_grp.query('ONET_release_year == @baseline_year')
                                        .groupby('classification')
                                        .apply(lambda d: np.average(d['percentile'],
                                                                    weights=d['bucket_weight']), include_groups=False))
            print("Sanity check: ", baseline.round(4))

            # Create ALM plots
            output_dir = os.path.join(ailabor_root, 'results/alm_classification_results_new/')
            create_alm_plots(filtered_data_grp, all_classifications, 'percentile', out_file, output_dir=output_dir)

            # Create between-occupation composition plots
            create_between_occ_composition_plots(filtered_data_grp, all_classifications, 'percentile', emp_share_col, intensity_col, out_file.replace('.png', '_between_occ.png'), output_dir=output_dir)

            # Save examples of each classification group if scope is healthcare
            if scope == 'healthcare':
                for classification in all_classifications:
                    subset = filtered_data[filtered_data['classification'] == classification]
                    if not subset.empty:
                        subset.to_csv(f'{output_dir}classification_examples/{classification_type}_{scope}_{task_scope}_{classification}_examples.csv', index=False)
            print(f"Completed processing for {classification_type} for scope {scope} and task scope {task_scope}")
            print("#" * 50)

            # save the filtered data for further analysis
            filtered_data_grp.to_csv(os.path.join(ailabor_root, f'alm_analysis/data1/plot_data/{classification_type}_{scope}_{task_scope}_filtered_data.csv'), index=False)