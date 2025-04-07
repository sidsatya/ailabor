# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).
library(dplyr)
library(readr)
if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

ddi <- read_ipums_ddi("data/ipums/usa_00012.xml")
data <- read_ipums_micro(ddi)


# Define healthcare-related IND1990 codes
healthcare_industries_1990 <- c(812, 820, 821, 822, 830, 831, 832, 840)

# Filter the data for healthcare-related industries
filtered_data <- data %>%
  filter(!is.na(IND1990) & as.integer(IND1990) %in% healthcare_industries_1990)

head(filtered_data)

# Get the min year and max year
min_year <- min(filtered_data$YEAR)
max_year <- max(filtered_data$YEAR)
cat("The data contains records from", min_year, "to", max_year, "\n")

# Save as a compressed CSV file
filtered_data %>%
  write_csv("data/ipums/ipums_healthcare_data.csv")
