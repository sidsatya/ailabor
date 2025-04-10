library(data.table)
library(dplyr)
library(stringdist)  # Add this library for string similarity

ipums_data <- fread("/Users/sidsatya/dev/ailabor/data/ipums/ipums_healthcare_data.csv")
master_crosswalk_data <- fread("/Users/sidsatya/dev/ailabor/data/occsoc_crosswalks/occsoc_crosswalk_2000_onward_without_code_descriptions.csv")
crosswalk_2000_to_2010 <- fread("/Users/sidsatya/dev/ailabor/data/occsoc_crosswalks/soc_2000_to_2010_crosswalk.csv")
crosswalk_2010_to_2018 <- fread("/Users/sidsatya/dev/ailabor/data/occsoc_crosswalks/soc_2010_to_2018_crosswalk.csv")

# Step 1: Ensure YEAR is numeric
ipums_data <- ipums_data %>%
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
data_2008_2009 <- ipums_data %>%
  filter(YEAR %in% 2008:2009) %>%
  left_join(crosswalk_2000_to_2010, by = c("OCCSOC" = "SOC_2000")) %>%
  left_join(crosswalk_2010_to_2018, by = "SOC_2010") %>%
  rename(OCCSOC_2018 = SOC_2018) %>%
  # Keep only original columns plus the new OCCSOC_2018 column
  select(names(ipums_data), OCCSOC_2018)

# For 2010–2017 Data
data_2010_2017 <- ipums_data %>%
  filter(YEAR %in% 2010:2017) %>%
  left_join(crosswalk_2010_to_2018, by = c("OCCSOC" = "SOC_2010")) %>%
  rename(OCCSOC_2018 = SOC_2018) %>%
  # Keep only original columns plus the new OCCSOC_2018 column
  select(names(ipums_data), OCCSOC_2018)

# For 2018–2023 Data
data_2018_2023 <- ipums_data %>%
  filter(YEAR %in% 2018:2023) %>%
  mutate(OCCSOC_2018 = OCCSOC) %>%
  # Keep only original columns plus the new OCCSOC_2018 column
  select(names(ipums_data), OCCSOC_2018)

# Combine all data
ipums_data_harmonized <- bind_rows(data_2008_2009, data_2010_2017, data_2018_2023)

# Step 4: Add OCCSOC_title_2018 using the correct column from crosswalk
# Convert the 2018 SOC code in the crosswalk for joining
crosswalk_2010_to_2018_titles <- crosswalk_2010_to_2018 %>%
  select(SOC_2018, `2018 SOC Title`) %>%
  distinct()

ipums_data_harmonized <- ipums_data_harmonized %>%
  left_join(
    crosswalk_2010_to_2018_titles, 
    by = c("OCCSOC_2018" = "SOC_2018")
  ) %>%
  rename(OCCSOC_title_2018 = `2018 SOC Title`)

# drop any observations with a null OCCSOC_2018
ipums_data_harmonized <- ipums_data_harmonized %>%
  filter(!is.na(OCCSOC_2018))

# Save the harmonized data
fwrite(ipums_data_harmonized, "/Users/sidsatya/dev/ailabor/data/ipums/ipums_healthcare_data.csv")

# Get all unique OCCSOC codes
unique_occsoc_codes <- unique(ipums_data_harmonized$OCCSOC_2018)

unique_occsoc_codes %>% 
  as.data.frame(columns=) %>%
  setNames("OCCSOC_2018") %>%
  write_csv("data/ipums/unique_healthcare_occsoc_codes.csv")
