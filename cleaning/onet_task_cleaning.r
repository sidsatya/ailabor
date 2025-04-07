library(data.table)

onet_data <- fread("data/onet/task_statements.csv")

# Filter for healthcare-related occupations using healthcare codes 
healthcare_occ_ind_codes <- fread("data/healthcare_occ_codes.csv")
healthcare_occ_ind_codes <- healthcare_occ_ind_codes$OCCSOC

# Convert onet soc codes to ipums soc codes
onet_to_ipums_occsoc <- function(onet_codes) {
  # Remove hyphens and periods, then coerce to numeric
  ipums_codes <- gsub("-", "", onet_codes)    # Remove hyphens
  ipums_codes <- gsub("\\..*", "", ipums_codes)  # Remove decimal and anything after
  return(ipums_codes)
}

# Convert the onet codes to ipums codes
onet_data$IPUMSOCCSOC <- onet_to_ipums_occsoc(onet_data$`O*NET-SOC Code`)

# Filter the onet data for healthcare-related occupations
healthcare_onet_industries_data <- onet_data[IPUMSOCCSOC %in% healthcare_occ_ind_codes]

# Save the filtered data to a CSV file
fwrite(healthcare_onet_data, "data/onet/onet_task_statements_healthcare_industries.csv", row.names = FALSE)

# Confirm the file was saved
cat("Filtered O*NET data for healthcare-related occupations saved to onet_task_statements_healthcare_only.csv\n")

# Now filter for only healthcare-related occupations
healthcare_occ_prefixes <- c("29-", "31-", "21-1022", "21-1014", "11-9111")
pattern <- paste(healthcare_occ_prefixes, collapse = "|")

# Filter the O*NET data for healthcare-related occupations by SOC code only
healthcare_onet_occ_data <- onet_data[grepl(pattern, `O*NET-SOC Code`)]

# Save the filtered data to a CSV file
fwrite(healthcare_onet_occ_data, "data/onet/onet_task_statements_healthcare_occupations.csv", row.names = FALSE)

# Confirm the file was saved
cat("Filtered O*NET data for healthcare-related occupations saved to onet_task_statements_healthcare_occupartions.csv\n")
