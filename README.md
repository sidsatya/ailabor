# AI Labor Project - Healthcare Occupation Task Analysis

This repository contains code and data for analyzing task dimensions in healthcare occupations and their relationship to wage and employment outcomes over time. The project uses O*NET task data combined with IPUMS employment data to classify healthcare occupations along different task dimensions and analyze economic trends.

## Project Overview

This project investigates how different task dimensions in healthcare occupations correlate with labor market outcomes. I classify occupational tasks along two frameworks:

### 1. 4-Dimension Framework
- Interpersonal Non-Routine (INR)
- Interpersonal Routine (IR)
- Personal Non-Routine (PNR)
- Personal Routine (PR)

### 2. 8-Dimension Framework
- Interpersonal Routine Manual (IP_R_M)
- Interpersonal Routine Non-Manual (IP_R_NM)
- Interpersonal Non-Routine Manual (IP_NR_M)
- Interpersonal Non-Routine Non-Manual (IP_NR_NM)
- Personal Routine Manual (P_R_M)
- Personal Routine Non-Manual (P_R_NM)
- Personal Non-Routine Manual (P_NR_M)
- Personal Non-Routine Non-Manual (P_NR_NM)

The analysis examines how these task dimensions relate to wages, employment growth, and other labor market indicators in the healthcare sector.

## Repository Structure
```
/ailabor/
├── analysis/               # Analysis scripts for running EDA and regressions
│   ├── shares_eda_4_dim.R
│   └── shares_eda_8_dim.R
├── cleaning/               # Data preparation and cleaning scripts
│   ├── compute_onet_task_shares.r
│   ├── ipums_cleaning.r
│   └── ipums_soc_cleaning.r
├── data/                   # Data directory (not included in repo)
│   ├── ipums/
│   ├── onet/
│   └── occsoc_crosswalks/
└── results/                # Output files (created by scripts)
│   ├── dim_4_results/
│   └── dim_8_results/
└── task_classification/    # Files for GPT classification of O*NET tasks
    ├── data/
    └── prompts/
    └── gpt_classification.py
    └── helper.py

```

## Key Files and Their Functions

### Cleaning Scripts

- `ipums_cleaning.r`
  - Filter IPUMS data for healthcare-related industries
  - Harmonize occupation codes (SOC) across 2008-2023
  - Apply crosswalks to ensure consistent occupation coding over time
  - Output: `ipums_healthcare_data.csv`

- `compute_onet_task_shares.r`
  - Processes O*NET task statements classified into task dimensions
  - Computes the share of tasks in each dimension for each occupation
  - Works with both 4-dimension and 8-dimension frameworks
  - Outputs: `onet_data_with_shares_4_dim.csv`, `onet_data_with_shares_8_dim.csv`

### Analysis Scripts

- `shares_eda_4_dim.R`
  - Analyzes the 4-dimension framework
  - Merges task shares with IPUMS employment/wage data
  - Produces visualizations and regression analysis of wage/employment growth

- `shares_eda_8_dim.R`
  - Same as above, but with the 8-dimension task structure
  - Includes PCA analysis to reduce dimensionality

## Data Requirements

For replication:

### O*NET
- `onet_task_statements_classified_4_dim.csv`
- `onet_task_statements_classified_8_dim.csv`

### IPUMS USA
- Extract the ACS from 2008 to 2023 with the following fields:
  - `YEAR`
  - `IND1990`
  - `OCCSOC`
  - `INCWAGE`
  - `PERWT`

### Crosswalks
- SOC 2000 to 2010
- SOC 2010 to 2018

## How to Run the Code

### Step 1: Setup
```r
install.packages(c("data.table", "dplyr", "tidyr", "ggplot2", "knitr", 
                 "scales", "stringr", "gridExtra", "broom", "ipumsr",
                 "stringdist", "readr"))
```

### Step 2: Data Preparation
1. Place O*NET classification files in `data/onet/`
2. Run:
```r
source("cleaning/compute_onet_task_shares.r")
```
3. If working with raw IPUMS data, place the extract in `data/ipums/` and run:
```r
source("cleaning/ipums_cleaning.r")
```

### Step 3: Analysis
- For 4-dimension results:
```r
source("analysis/shares_eda_4_dim.R")
```
- For 8-dimension results:
```r
source("analysis/shares_eda_8_dim.R")
```

## Output Files
Output files will be saved in `results/dim_4_results/` and `results/dim_8_results/`
- **Balance Tables**: Summary stats by task dimension
- **Growth Plots**: Wage and employment growth visualizations
- **Regression Tables**: Output from statistical models
- **Bubble Plots**: Joint wage/employment/size plots

## Task Classification 

### 4-Dimension
- INR: Interpersonal, Non-Routine
- IR: Interpersonal, Routine
- PNR: Personal, Non-Routine
- PR: Personal, Routine

### 8-Dimension
Adds a manual/non-manual distinction to the above. For example:
- IP_R_M: Interpersonal, Routine, Manual
- IP_R_NM: Interpersonal, Routine, Non-Manual
- etc.

Task classification was done using OpenAI's GPT-4o-mini model. 

## Replication Instructions

To fully replicate results:
1. Place the required data in appropriate directories
2. Run `gpt_classification.py`

Pre-processed files are also available for analysis without running the task.

## Output Files
Intermediate data files will be saved to `task_classification/data`. Final output files with O*NET data merged to classified tasks will be saved in `data/onet/` as: 
- `data/onet/onet_task_statements_classified_4_dim.csv`
- `data/onet/onet_task_statements_classified_8_dim.csv`

## TODO
- Analyze changes in task shares over time. Think about core vs. supplementary tasks and date as well.
- Double check and understand differences in wage growth (check relative vs. absolute growth)
- Check that replication can be run top to bottom and include full instructions.

---

For questions or additional information, please contact me (Sid Satya).
