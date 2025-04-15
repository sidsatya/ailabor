library(data.table)
library(dplyr)
library(tidyr)
library(broom)

# Check if ROOT_DIR is defined in the environment
root_dir_env <- Sys.getenv("ROOT_DIR", unset = NA)

if (is.na(root_dir_env) || root_dir_env == "") {
  # Fallback: set manually
  ROOT_DIR <- "/Users/sidsatya/dev/ailabor"  # Change to your actual project path
  message("ROOT_DIR not found in environment. Using fallback path: ", ROOT_DIR)
} else {
  ROOT_DIR <- root_dir_env
  message("Using ROOT_DIR from environment: ", ROOT_DIR)
}


merged_data = fread(file.path(ROOT_DIR, "data/merged_data_with_dimensions.csv"))


exposure_weights <- data.frame(
  Task_Type = c("IP_R_M", "IP_R_NM", "IP_NR_M", "IP_NR_NM", "P_R_M", "P_R_NM", "P_NR_M", "P_NR_NM"),
  Exposure_Weight = c(
    0.1,  # IP_R_M
    1.0,  # IP_R_NM
    0.0,  # IP_NR_M
    0.3,  # IP_NR_NM
    0.1,  # P_R_M
    1.0,  # P_R_NM
    0.0,  # P_NR_M
    0.6   # P_NR_NM
  )
)

# multiply and sum across columns in merged_data to construct 
# exposure score
exposure_score <- merged_data %>%
  select("IP_R_M", "IP_R_NM", "IP_NR_M", "IP_NR_NM", "P_R_M", "P_R_NM", "P_NR_M", "P_NR_NM") %>%
  as.matrix() %*% exposure_weights$Exposure_Weight

# Add the exposure score to the merged_data 
merged_data <- merged_data %>%
  mutate(exposure_score = exposure_score)

# Get all unique years in the dataset
all_years <- unique(merged_data$YEAR)

# Find occupation codes that appear in all years
common_occsoc_codes <- merged_data %>%
  group_by(OCCSOC_2018) %>%
  summarize(
    num_years = n_distinct(YEAR),
    .groups = "drop"
  ) %>%
  filter(num_years == length(all_years)) %>%
  pull(OCCSOC_2018)

# Create balanced dataset with only occupations present in all years
balanced_data <- merged_data %>%
  filter(OCCSOC_2018 %in% common_occsoc_codes)

# Report dataset balancing statistics
cat("Original number of unique occupation codes:", n_distinct(merged_data$OCCSOC_2018), "\n")
cat("Number of occupation codes present in all years:", length(common_occsoc_codes), "\n")
cat("Number of observations before filtering:", nrow(merged_data), "\n")
cat("Number of observations after filtering:", nrow(balanced_data), "\n")

first_year <- min(balanced_data$YEAR)
last_year <- max(balanced_data$YEAR)
growth_data <- balanced_data %>%
  filter(YEAR %in% c(first_year, last_year) & INCWAGE > 5000 & PERWT >= 1) %>%
  group_by(Title, OCCSOC_2018, YEAR) %>%
  summarize(
    total_wage = sum(INCWAGE * PERWT, na.rm = TRUE),
    total_employment = sum(PERWT, na.rm = TRUE),
    n_obs = n(),
    min_wage = min(INCWAGE, na.rm = TRUE),
    max_wage = max(INCWAGE, na.rm = TRUE),
    zero_weights = sum(PERWT == 0, na.rm = TRUE),
    avg_incwage = if(total_employment > 0) total_wage / total_employment else NA_real_,
    dominant_dimension_label = first(dominant_dimension_label),
    primary_dimension_label = first(primary_dimension_label),
    exposure_score = mean(exposure_score),
    .groups = "drop"
  ) %>%
  # Pivot to wide format to have first_year and last_year as separate columns
  pivot_wider(
    names_from = YEAR,
    values_from = c(avg_incwage, total_employment, total_wage, n_obs, min_wage, max_wage, zero_weights)
  ) %>%
  # Calculate growth rates
  mutate(
    wage_growth = ((get(paste0("avg_incwage_", last_year)) / get(paste0("avg_incwage_", first_year))) - 1) * 100,
    employment_growth = ((get(paste0("total_employment_", last_year)) / get(paste0("total_employment_", first_year))) - 1) * 100
  ) %>% 
  filter(!is.na(wage_growth) & is.finite(wage_growth) & 
         !is.na(employment_growth) & is.finite(employment_growth)) %>%
  mutate(
    wage_growth_log = get(paste0("avg_incwage_", last_year)) / get(paste0("avg_incwage_", first_year)),
    emp_growth_log = get(paste0("total_employment_", last_year)) / get(paste0("total_employment_", first_year))
  )

yearly_wage_data <- balanced_data %>%
  filter(INCWAGE > 5000 & PERWT >= 1) %>% # Apply similar filters as growth analysis
  group_by(Title, OCCSOC_2018, YEAR) %>%
  summarize(
    total_wage = sum(INCWAGE * PERWT, na.rm = TRUE),
    total_employment = sum(PERWT, na.rm = TRUE),
    avg_incwage = if(total_employment > 0) total_wage / total_employment else NA_real_,
    exposure_score = mean(exposure_score),
    .groups = "drop"
  ) %>%
  filter(!is.na(avg_incwage)) # Remove rows where avg_incwage couldn't be calculated

# Fit regression model using principal components
inc_growth_model <- lm(log(avg_incwage) ~ exposure_score + c(YEAR), 
                      data = yearly_wage_data)
inc_growth_summary <- summary(inc_growth_model)
inc_growth_tidy <- tidy(inc_growth_model, conf.int = TRUE)
# Display and save regression table
knitr::kable(inc_growth_tidy, digits = 3, 
             caption = "Regression: Income Growth vs. Task Dimensions")
