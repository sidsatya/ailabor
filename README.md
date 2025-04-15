# The Impact of AI on Healthcare (Still in Progress)

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

- `ipums_extract_and_clean.r`
  - Extract IPUMS data for ACS from 2008-2023
  - Filter ACS data for healthcare-related industries
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

For replication, all data files should already be included in the repository or be part of the extraction in the code files. However, if you are curious, some key files are:

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
  - `COUNTYFIP`
  - `STATEFIP`

### SOC Crosswalks
- SOC 2000 to 2010
- SOC 2010 to 2018

## How to Run the Code

### Step 1: Setup

This project uses `renv` for dependency management. To set up the environment and install the required packages, follow these steps:

1. Ensure that the `renv` package is installed. If not, install it using:
  ```r
  install.packages("renv")
  ```

2. Initialize the `renv` environment using the lockfile in this directory. Run:
  ```r
  renv::restore()
  ```
  This will install all the required packages specified in the `renv.lock` file.

3. If the `renv` environment does not boot up automatically, you can activate it manually by running:
  ```r
  renv::activate()
  ```

4. Request an IPUMS API Key
To run the data extraction, you must follow the steps at [https://developer.ipums.org/docs/v2/get-started/](https://developer.ipums.org/docs/v2/get-started/) to create an IPUMS account and get an API key. Then, you must create a `.env` file in the project's root directory and set `IPUMS_API_KEY=YOUR_API_KEY`.

Once the environment is set up, you can proceed with the data preparation and analysis steps.

### Step 2: Running Code 

Edit line 4 of `run_main_R_files.R` and set it to your local path to this project's root directory. Then, you can just run the code in `run_main_files.R`! 

*Important Note!!*: When I run the code on my machine (MacBook Pro 2019, 8GB), I run out of memory after extracting the ACS data when running `ipums_extract_and_clean.r`. If the extraction fails, you may find it helpful to restart your machine and then run the code again. If the extraction succeeds, it should save the extracted files in `data/ipums`. However, if the code then fails or segfaults when reading this extracted data (also happens to me at times), you might find it helpful to restart your machine again and then re-run the code. If data is already existing in the `data/ipums` folder, the code will not rerun the extraction and only rely on already extracted data. 

## Output Files
Output files will be saved in `results/dim_4_results/` and `results/dim_8_results/`
- **Balance Tables**: Summary stats by task dimension
- **Growth Plots**: Wage and employment growth visualizations
- **Regression Tables**: Output from statistical models
- **Bubble Plots**: Joint wage/employment/size plots

## Task Classification using GPT

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

## Replication Instructions for Task Classification (Optional, you don't have to do this)

To fully replicate results:
1. Create an [OpenAI API](https://platform.openai.com/) account, and add about $5 to your account. 
2. Add `OPENAI_API_KEY=YOUR_API_KEY` to your `.env` file. 
3. Run `gpt_classification.py`

Pre-processed files are also available for analysis without running the task.

## Output Files
Intermediate data files will be saved to `task_classification/data`. Final output files with O*NET data merged to classified tasks will be saved in `data/onet/` as: 
- `data/onet/onet_task_statements_classified_4_dim.csv`
- `data/onet/onet_task_statements_classified_8_dim.csv`

## TODO
- Analyze changes in task shares over time. Think about core vs. supplementary tasks and date as well. Finish `construct_historical_onet_task_statements.R`.
- Write extraction of US PTO Patent Data and arXiv papers. 
- Write matching of historical O*NET task data to patent data and arXiv papers. 
- Compute AI-exposure score. Consider expanding analysis to general software and automation technologies like in Webb (2020)
- Shift-share analysis for state-county-occupation-industry obs. 

---

For questions or additional information, please contact me (Sid Satya).
