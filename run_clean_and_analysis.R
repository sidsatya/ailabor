# run_all.R

# 1. Set root directory explicitly
root_dir <- "/Users/sidsatya/dev/ailabor"  # TODO: You must change this to your own path to the project root! 
setwd(root_dir)

# 2. Optionally pass this to all scripts via a global variable
assign("ROOT_DIR", root_dir, envir = .GlobalEnv)

# 3. Run scripts in order
source("cleaning/ipums_extract_and_cleaning.R")
source("cleaning_compute_onet_task_shares.R")
source("analysis/shares_eda_4_dim.R")
source("analysis/shares_eda_8_dim.R")