library(readxl)
library(data.table)

# Check if ROOT_DIR exists as an R variable in the global environment
if (exists("ROOT_DIR", envir = .GlobalEnv)) {
  ROOT_DIR <- get("ROOT_DIR", envir = .GlobalEnv)
  message("Using ROOT_DIR from global R environment: ", ROOT_DIR)
} else {
  # Fallback: set manually if not found (though it should be if run via run_main_R_files.R)
  ROOT_DIR <- "/Users/sidsatya/dev/ailabor"  # Change to your actual project path
  warning("ROOT_DIR not found in R's global environment. Using fallback path: ", ROOT_DIR)
}

task_dir <- file.path(ROOT_DIR, "data/bls/national_oes")

# List all .xlsx and .xls files
xlsx_files <- list.files(task_dir, pattern = "\\.xlsx$", full.names = TRUE)
xls_files <- list.files(task_dir, pattern = "\\.xls$", full.names = TRUE)

# Function to convert .xlsx to .csv
convert_xlsx_to_csv <- function(file_path) {
  data <- read_excel(file_path)  # Read the Excel file
  csv_path <- sub("\\.xlsx$", ".csv", file_path)  # Replace .xlsx with .csv
  fwrite(data, csv_path)  # Write to CSV
  message("Converted ", file_path, " to ", csv_path)
}

# Function to convert .xls to .csv
convert_xls_to_csv <- function(file_path) {
  data <- read_excel(file_path)  # Read the xls file
  csv_path <- sub("\\.xls$", ".csv", file_path)  # Replace .xls with .csv
  fwrite(data, csv_path)  # Write to CSV
  message("Converted ", file_path, " to ", csv_path)
}

# Convert all .xlsx files to .csv
lapply(xlsx_files, convert_xlsx_to_csv)

# Convert all .xls files to .csv
lapply(xls_files, convert_xls_to_csv)