# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).
library(data.table)
if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

ddi <- read_ipums_ddi("data/ipums/usa_00011.xml")
data <- read_ipums_micro(ddi)
setDT(data)

# Filter for healthcare-related industries using INDNAICS
healthcare_codes <- c(621, 622, 623, 624)

# Filter efficiently
healthcare_data <- data[INDNAICS %in% healthcare_codes]

# View the filtered data
head(healthcare_data)

# Save as a compressed CSV file
fwrite(healthcare_data, "data/ipums/ipums_healthcare_data.csv", row.names = FALSE)

# Extract unique OCCSOC codes
unique_occ_codes <- unique(data$OCCSOC)

# Convert to a data frame for saving
unique_occ_codes_df <- data.frame(OCCSOC = unique_occ_codes)

# Save to a CSV file
write.csv(unique_occ_codes_df, "data/healthcare_occ_codes.csv", row.names = FALSE)

# Confirm the file was saved
cat("Unique OCCSOC codes saved to healthcare_occ_codes.csv\n")

