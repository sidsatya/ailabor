library(ipumsr)
library(dplyr)
library(data.table)
library(readr)
library(stringdist)  # Add this library for string similarity
library(dotenv)

# Check if ROOT_DIR exists as an R variable in the global environment
if (exists("ROOT_DIR", envir = .GlobalEnv)) {
  ROOT_DIR <- get("ROOT_DIR", envir = .GlobalEnv)
  message("Using ROOT_DIR from global R environment: ", ROOT_DIR)
} else {
  # Fallback: set manually if not found (though it should be if run via run_main_R_files.R)
  ROOT_DIR <- "/Users/sidsatya/dev/ailabor"  # Change to your actual project path
  warning("ROOT_DIR not found in R's global environment. Using fallback path: ", ROOT_DIR)
}

# Define the download directory
download_dir <- file.path(ROOT_DIR, "data", "ipums")
dir.create(download_dir, recursive = TRUE, showWarnings = FALSE) # Ensure directory exists

# Check for existing IPUMS data files (usa_XXXXX.xml and usa_XXXXX.csv.gz)
existing_ddi_files <- list.files(download_dir, pattern = "^usa_\\d+\\.xml$", full.names = TRUE)
existing_data_files <- list.files(download_dir, pattern = "^usa_\\d+\\.csv\\.gz$", full.names = TRUE)
# Initialize ddi_file_path
ddi_file_path <- NULL

# Check for existing IPUMS DDI file (usa_XXXXX.xml)
existing_ddi_files <- list.files(download_dir, pattern = "^usa_\\d+\\.xml$", full.names = TRUE)

if (length(existing_ddi_files) > 0) {
  # Use the first existing DDI file found (consider sorting by name/date if multiple exist)
  ddi_file_path <- existing_ddi_files[1]
  message("Found existing IPUMS DDI file: ", ddi_file_path)
} else {
  # No existing DDI found, proceed with API download
  message("No existing IPUMS DDI file found. Submitting extract request.")

  # Load .env file and access your IPUMS API key
  dotenv::load_dot_env(file = ".env")
  ipums_api_key <- Sys.getenv("IPUMS_API_KEY")
  if (is.null(ipums_api_key) || ipums_api_key == "") {
      stop("IPUMS_API_KEY not found in environment or .env file.")
  }
  set_ipums_api_key(ipums_api_key, overwrite = TRUE)

  # Define the extract
  extract_def <- define_extract_micro(
    collection = "usa",
    description = "ACS 2008–2023 extract via API",
    samples = c(
      "us2008a", "us2009a", "us2010a", "us2011a", "us2012a", "us2013a",
      "us2014a", "us2015a", "us2016a", "us2017a", "us2018a",
      "us2019a", "us2020a", "us2021a", "us2022a", "us2023a"
    ),
    variables = c("YEAR", "PERWT", "INCWAGE", "OCCSOC", "IND1990", "COUNTYFIP", "STATEFIP"),
    data_format= "csv",
    data_structure="rectangular",
    case_select_who="individuals",
    data_quality_flags=TRUE
  )

  # Submit, wait, download
  submitted <- submit_extract(extract_def)
  message("Waiting for extract to complete...")
  downloaded <- wait_for_extract(submitted)
  message("Downloading extract files...")
  # download_extract returns paths to both .xml and .csv.gz
  downloaded_files <- download_extract(downloaded, download_dir = download_dir)

  # Find the downloaded DDI (.xml) file path
  downloaded_ddi_file <- downloaded_files[grepl("\\.xml$", downloaded_files)]

  if (length(downloaded_ddi_file) == 1) {
      ddi_file_path <- downloaded_ddi_file
      message("Downloaded new IPUMS DDI file: ", ddi_file_path)
  } else {
      stop("Could not find the downloaded DDI file (.xml). Download might have failed or returned unexpected results.")
  }
}

# Ensure a DDI file path was determined
if (is.null(ddi_file_path) || !file.exists(ddi_file_path)) {
    stop("Failed to find or download a valid IPUMS DDI file.")
}

# Step 1: Read the DDI metadata
message("Reading DDI metadata from: ", ddi_file_path)

# Step 2: Read the microdata using the DDI object
# This automatically finds the associated .csv.gz file based on the DDI info
message("Reading IPUMS microdata using the DDI object...")
data <- read_ipums_micro(ddi_file_path)

# Ensure data is loaded
if (!exists("data")) {
  stop("Failed to load IPUMS data.")
} else {
  message("IPUMS data loaded successfully.")
}

# Define healthcare-related IND1990 codes
healthcare_industries_1990 <- c(812, 820, 821, 822, 830, 831, 832, 840)

# Filter the data for healthcare-related industries
filtered_data <- data %>%
  filter(!is.na(IND1990) & as.integer(IND1990) %in% healthcare_industries_1990)

cat("Successfully filtered data for healthcare industries only\n")

# Get the min year and max year
min_year <- min(filtered_data$YEAR)
max_year <- max(filtered_data$YEAR)
cat("The data contains records from", min_year, "to", max_year, "\n")

# section: Harmonizing OCCSOC codes
# crosswalks taken from: 
# 1. https://www.bls.gov/soc/soc_2000_to_2010_crosswalk.xls
# 2. https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx
# 3. https://usa.ipums.org/usa/resources/volii/occ_occsoc_crosswalk_2000_onward_without_code_descriptions.csv
master_crosswalk_data <- fread(file.path(ROOT_DIR,"data/occsoc_crosswalks/occsoc_crosswalk_2000_onward_without_code_descriptions.csv"))
crosswalk_2000_to_2010 <- fread(file.path(ROOT_DIR, "data/occsoc_crosswalks/soc_2000_to_2010_crosswalk.csv"))
crosswalk_2010_to_2018 <- fread(file.path(ROOT_DIR, "data/occsoc_crosswalks/soc_2010_to_2018_crosswalk.csv"))

# Step 1: Ensure YEAR is numeric
filtered_data <- filtered_data %>%
  mutate(YEAR = as.numeric(YEAR))

# Step 2: Convert SOC codes in crosswalks and handle duplicates
# For 2000 to 2010 crosswalk
crosswalk_2000_to_2010 <- crosswalk_2000_to_2010 %>%
  mutate(
    SOC_2000 = gsub("-", "", `2000 SOC code`),
    SOC_2010 = gsub("-", "", `2010 SOC code`),
    # Calculate similarity between titles
    title_similarity = 1 - stringdist(`2000 SOC title`, `2010 SOC title`, method = "cosine") / 
                      (nchar(`2000 SOC title`) + nchar(`2010 SOC title`))
  ) %>%
  # For each SOC_2000, keep the row with highest title similarity
  group_by(SOC_2000) %>%
  slice_max(order_by = title_similarity, n = 1, with_ties = FALSE) %>%
  ungroup()

# For 2010 to 2018 crosswalk
crosswalk_2010_to_2018 <- crosswalk_2010_to_2018 %>%
  mutate(
    SOC_2010 = gsub("-", "", `2010 SOC Code`),
    SOC_2018 = gsub("-", "", `2018 SOC Code`),
    # Calculate similarity between titles
    title_similarity = 1 - stringdist(`2010 SOC Title`, `2018 SOC Title`, method = "cosine") / 
                       (nchar(`2010 SOC Title`) + nchar(`2018 SOC Title`))
  ) %>%
  # For each SOC_2010, keep the row with highest title similarity
  group_by(SOC_2010) %>%
  slice_max(order_by = title_similarity, n = 1, with_ties = FALSE) %>%
  ungroup()

# Step 3: Harmonize OCCSOC to OCCSOC_2018
# For 2008–2009 Data
data_2008_2009 <- filtered_data %>%
  filter(YEAR %in% 2008:2009) %>%
  left_join(crosswalk_2000_to_2010, by = c("OCCSOC" = "SOC_2000")) %>%
  left_join(crosswalk_2010_to_2018, by = "SOC_2010") %>%
  rename(OCCSOC_2018 = SOC_2018) %>%
  # Keep only original columns plus the new OCCSOC_2018 column
  select(names(filtered_data), OCCSOC_2018)

# For 2010–2017 Data
data_2010_2017 <- filtered_data %>%
  filter(YEAR %in% 2010:2017) %>%
  left_join(crosswalk_2010_to_2018, by = c("OCCSOC" = "SOC_2010")) %>%
  rename(OCCSOC_2018 = SOC_2018) %>%
  # Keep only original columns plus the new OCCSOC_2018 column
  select(names(filtered_data), OCCSOC_2018)

# For 2018–2023 Data
data_2018_2023 <- filtered_data %>%
  filter(YEAR %in% 2018:2023) %>%
  mutate(OCCSOC_2018 = OCCSOC) %>%
  # Keep only original columns plus the new OCCSOC_2018 column
  select(names(filtered_data), OCCSOC_2018)

# ----- Combining and finalizing the harmonized dataset -----
# Merge all time periods into a single dataset with consistent occupation coding
filtered_data_harmonized <- bind_rows(data_2008_2009, data_2010_2017, data_2018_2023)

# Add occupation titles to make the data more interpretable
# Extract just what we need from the crosswalk to avoid duplicate columns
crosswalk_2010_to_2018_titles <- crosswalk_2010_to_2018 %>%
  select(SOC_2018, `2018 SOC Title`) %>%
  distinct()

# Join the titles to our harmonized dataset based on the 2018 SOC codes
filtered_data_harmonized <- filtered_data_harmonized %>%
  left_join(
    crosswalk_2010_to_2018_titles, 
    by = c("OCCSOC_2018" = "SOC_2018")
  ) %>%
  rename(OCCSOC_title_2018 = `2018 SOC Title`)

# Remove records where occupation couldn't be properly harmonized to 2018 classification
filtered_data_harmonized <- filtered_data_harmonized %>%
  filter(!is.na(OCCSOC_2018))

# Save the cleaned and harmonized dataset for further analysis
fwrite(filtered_data_harmonized, file.path(ROOT_DIR, "/data/ipums/ipums_healthcare_data.csv"))

# ----- Extract unique occupation codes for reference -----
# This creates a simple lookup of all healthcare occupations in our dataset
unique_occsoc_codes <- unique(filtered_data_harmonized$OCCSOC_2018)

unique_occsoc_codes %>% 
  as.data.frame(columns=) %>%
  setNames("OCCSOC_2018") %>%
  write_csv(file.path(ROOT_DIR, "data/ipums/ipums_unique_healthcare_occsoc_codes.csv"))
