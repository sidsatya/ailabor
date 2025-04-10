# =============================================================================
# O*NET Healthcare Task Statement Cleaning Script
# 
# Purpose: Extract healthcare-related task statements from O*NET database
# using two different filtering methods:
# 1. Healthcare occupations in healthcare industries (based on IPUMS codes)
# 2. Healthcare occupations only (based on SOC code prefixes)
# =============================================================================

# Load required libraries
library(data.table)

# 1. DATA LOADING
# Load the main O*NET task statements dataset
onet_data <- fread("data/onet/task_statements.csv")

# Load healthcare occupation codes from IPUMS
healthcare_occ_ind_codes <- fread("data/ipums/ipums_unique_healthcare_occsoc_codes.csv")
healthcare_occ_ind_codes <- healthcare_occ_ind_codes$OCCSOC_2018

# Function to convert O*NET SOC codes to IPUMS SOC code format
# Example: Converts "29-1141.00" to "291141"
onet_to_ipums_occsoc <- function(onet_codes) {
  # Remove hyphens and periods, then coerce to numeric
  ipums_codes <- gsub("-", "", onet_codes)    # Remove hyphens
  ipums_codes <- gsub("\\..*", "", ipums_codes)  # Remove decimal and anything after
  return(ipums_codes)
}

# 2. FILTER METHOD 1: HEALTHCARE OCCUPATIONS IN HEALTHCARE INDUSTRIES
# Convert O*NET codes to IPUMS format for matching
onet_data$ONETOCCSOC_2018 <- onet_to_ipums_occsoc(onet_data$`O*NET-SOC Code`)

# Filter the O*NET data to include only occupations that appear in healthcare industries
# (Based on IPUMS occupation codes in healthcare sectors)
onet_occs_ind_data <- onet_data[ONETOCCSOC_2018 %in% healthcare_occ_ind_codes]

# Set output file path
onet_occs_ind_save_file <- "data/onet/onet_task_statements_occs_in_healthcare_industries.csv"

# Save the filtered data
fwrite(onet_occs_ind_data, onet_occs_ind_save_file, row.names = FALSE)

# Log confirmation message
cat("Filtered O*NET data for occupations in healthcare-related industries saved to: ", 
    onet_occs_ind_save_file, "\n")

# 4. FILTER METHOD 2: HEALTHCARE OCCUPATIONS BY SOC CODE PREFIX
# Define SOC code prefixes for healthcare occupations:
# 29-: Healthcare practitioners (doctors, nurses, etc.)
# 31-: Healthcare support occupations
# 21-1022: Healthcare social workers
# 21-1014: Mental health counselors
# 11-9111: Medical and health services managers
healthcare_occ_prefixes <- c("29-", "31-", "21-1022", "21-1014", "11-9111")

# Create a regex pattern to match any of these prefixes
pattern <- paste(healthcare_occ_prefixes, collapse = "|")

# Filter O*NET data based on SOC code prefixes
onet_healthcare_occs_only <- onet_data[grepl(pattern, `O*NET-SOC Code`)]

# Set output file path
onet_healthcare_occs_only_save_file <- "data/onet/onet_task_statements_healthcare_occs_only.csv"

# Save the filtered data
fwrite(onet_healthcare_occs_only, onet_healthcare_occs_only_save_file, row.names = FALSE)

# Log confirmation message
cat("Filtered O*NET data for healthcare-related occupations only saved to: ", 
    onet_healthcare_occs_only_save_file, "\n")
