# =============================================================================
# HEALTHCARE OCCUPATION TASK DIMENSION ANALYSIS (4-DIMENSION VERSION)
# =============================================================================
# This script analyzes task dimensions in healthcare occupations using ONET and IPUMS data.
# It classifies occupations by their task dimensions (4-dimension model), analyzes wage and 
# employment trends, and visualizes relationships between task composition and labor market outcomes.

# -----------------------------------------------------------------------------
# 1. SETUP AND LIBRARY IMPORTS
# -----------------------------------------------------------------------------
library(data.table)
library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)
library(scales)
library(stringr)
library(gridExtra)
library(broom)


# Check if ROOT_DIR exists as an R variable in the global environment
if (exists("ROOT_DIR", envir = .GlobalEnv)) {
  ROOT_DIR <- get("ROOT_DIR", envir = .GlobalEnv)
  message("Using ROOT_DIR from global R environment: ", ROOT_DIR)
} else {
  # Fallback: set manually if not found (though it should be if run via run_main_R_files.R)
  ROOT_DIR <- "/Users/sidsatya/dev/ailabor"  # Change to your actual project path
  warning("ROOT_DIR not found in R's global environment. Using fallback path: ", ROOT_DIR)
}

# -----------------------------------------------------------------------------
# 2. DATA IMPORT AND PREPARATION
# -----------------------------------------------------------------------------
# Read source datasets: ONET task dimensions (4-dimension model) and IPUMS labor market data
onet_shares_path = file.path(ROOT_DIR, "data", "onet", "onet_data_with_shares_4_dim.csv")
ipums_data_path = file.path(ROOT_DIR, "data", "ipums", "ipums_healthcare_data.csv")

onet_shares <- fread(onet_shares_path)
ipums_data <- fread(ipums_data_path)

# Set directory for results and directory if it doesn't exist
RESULTS_DIR = file.path(ROOT_DIR, "results", "dim_4_results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Merge datasets based on occupation codes (ONETOCCSOC_2018 matches OCCSOC_2018)
merged_data <- ipums_data %>%
  left_join(onet_shares %>%
              mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>%
              select(ONETOCCSOC_2018, Title, INR, IR, PNR, PR),
              by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# Create total wage and total employment columns using person weights
merged_data <- merged_data %>%
  mutate(
    total_wage = INCWAGE * PERWT,
    total_employment = PERWT
  )

# Print matching statistics
cat("Matched", nrow(merged_data[!is.na(merged_data$INR),]), "of", nrow(ipums_data), "rows\n")

# -----------------------------------------------------------------------------
# 3. TASK DIMENSION CLASSIFICATION
# -----------------------------------------------------------------------------
# Define task dimensions (4 dimensions):
# - Interpersonal Non-Routine (INR)
# - Interpersonal Routine (IR)
# - Personal Non-Routine (PNR)
# - Personal Routine (PR)

# METHOD 1: Primary dimension - classify by the dimension with the highest share
merged_data <- merged_data %>%
  mutate(
    primary_dimension = case_when(
      INR >= IR & INR >= PNR & INR >= PR ~ "INR",
      IR >= INR & IR >= PNR & IR >= PR ~ "IR",
      PNR >= INR & PNR >= IR & PNR >= PR ~ "PNR",
      TRUE ~ "PR"
    ),
    # Create human-readable labels for the primary dimensions
    primary_dimension_label = case_when(
      primary_dimension == "INR" ~ "Interpersonal Non-Routine",
      primary_dimension == "IR" ~ "Interpersonal Routine",
      primary_dimension == "PNR" ~ "Personal Non-Routine",
      primary_dimension == "PR" ~ "Personal Routine"
    )
  )

# METHOD 2: Dominant dimension - classify by dimensions exceeding 50% threshold
merged_data <- merged_data %>%
  mutate(
    dominant_dimension = case_when(
      INR > 0.5 ~ "INR",
      IR > 0.5 ~ "IR",
      PNR > 0.5 ~ "PNR",
      PR > 0.5 ~ "PR",
      TRUE ~ "Mixed"  # No single dimension exceeds threshold
    ),
    # Create human-readable labels for dominant dimensions
    dominant_dimension_label = case_when(
      dominant_dimension == "INR" ~ "Primarily Interpersonal Non-Routine",
      dominant_dimension == "IR" ~ "Primarily Interpersonal Routine",
      dominant_dimension == "PNR" ~ "Primarily Personal Non-Routine",
      dominant_dimension == "PR" ~ "Primarily Personal Routine",
      TRUE ~ "Mixed Tasks"
    )
  )

# -----------------------------------------------------------------------------
# 4. CREATE BALANCED PANEL DATASET
# -----------------------------------------------------------------------------
# To ensure consistent analysis over time, identify occupations present in all years

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

# -----------------------------------------------------------------------------
# 5. SUMMARY STATISTICS BY DIMENSION
# -----------------------------------------------------------------------------
# Create summary statistics for occupations by primary dimension

primary_dim_summary <- balanced_data %>%
  group_by(Title, primary_dimension_label) %>%
  summarize(
    count = n(),
    # Calculate average shares for each dimension
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    avg_PNR = mean(PNR, na.rm = TRUE),
    avg_PR = mean(PR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(primary_dimension_label, desc(count))

# Create summary statistics for occupations by dominant dimension
dominant_dim_summary <- balanced_data %>%
  group_by(Title, dominant_dimension_label) %>%
  summarize(
    count = n(),
    # Calculate average shares for each dimension
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    avg_PNR = mean(PNR, na.rm = TRUE),
    avg_PR = mean(PR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(dominant_dimension_label, desc(count))

# -----------------------------------------------------------------------------
# 6. WAGE AND EMPLOYMENT GROWTH ANALYSIS
# -----------------------------------------------------------------------------
# Define first and last year for growth analysis
first_year <- min(balanced_data$YEAR, na.rm = TRUE)
last_year <- max(balanced_data$YEAR, na.rm = TRUE)
cat("Analyzing growth of variables from", first_year, "to", last_year, "\n")

# Compute weighted average income and total employment by occupation and year
# Filter out unrealistically low incomes and zero weights
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
  )

# Join growth data to occupation dimension data
occupation_analysis <- primary_dim_summary %>%
  left_join(growth_data %>% select(Title, wage_growth, employment_growth), by = "Title")

# -----------------------------------------------------------------------------
# 7. BALANCE TABLES AND DATA EXPORTS
# -----------------------------------------------------------------------------
# Create balance table by primary dimension
balance_table <- balanced_data %>%
  group_by(primary_dimension_label) %>%
  summarize(
    n_occupations = n_distinct(Title),
    n_workers = n(),
    avg_incwage = mean(INCWAGE, na.rm = TRUE),
    median_incwage = median(INCWAGE, na.rm = TRUE),
    # Include average shares of each dimension
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    avg_PNR = mean(PNR, na.rm = TRUE),
    avg_PR = mean(PR, na.rm = TRUE)
  )

# Display and save primary dimension balance table
knitr::kable(balance_table, digits = 2, caption = "Balance Table by Primary Task Dimension")
write.csv(balance_table, file.path(RESULTS_DIR, "balance_table_primary_dimension.csv"), row.names = FALSE)

# Create and save balance table by dominant dimension
balance_table_dominant <- balanced_data %>%
  group_by(dominant_dimension_label) %>%
  summarize(
    n_occupations = n_distinct(Title),
    n_workers = n(),
    avg_incwage = mean(INCWAGE, na.rm = TRUE),
    median_incwage = median(INCWAGE, na.rm = TRUE),
    # Include average shares of each dimension
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    avg_PNR = mean(PNR, na.rm = TRUE),
    avg_PR = mean(PR, na.rm = TRUE)
  )

# Display and save dominant dimension balance table
knitr::kable(balance_table_dominant, digits = 2, caption = "Balance Table by Dominant Task Dimension")
write.csv(balance_table_dominant, file.path(RESULTS_DIR, "balance_table_dominant_dimension.csv"), row.names = FALSE)

# Save titles pertaining to each dimension to a txt file using a for loop
for (dim in c("INR", "IR", "PNR", "PR")) {
  write.table(
    balanced_data %>% 
      filter(primary_dimension == dim) %>% 
      select(Title) %>% 
      distinct(),
    file = paste0(RESULTS_DIR, "/primary_", dim, "_titles.txt"),
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
}

# =============================================================================
# VISUALIZATIONS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. OCCUPATION COUNT BY PRIMARY DIMENSION
# -----------------------------------------------------------------------------
# Create bar chart showing worker counts by primary task dimension

ggplot(primary_dim_summary, aes(x = primary_dimension_label, fill = primary_dimension_label)) +
  geom_bar(aes(weight = count)) +
  labs(title = "Number of Workers by Primary Task Dimension",
       x = NULL,
       y = "Count",
       fill = "Primary Dimension") +
  theme_minimal() +
  theme(axis.text.x = element_blank())

ggsave(file.path(RESULTS_DIR, "occupation_count_by_primary_dimension.png"), 
       width = 12, height = 8)

# -----------------------------------------------------------------------------
# 2. DIMENSION SHARE DISTRIBUTION ANALYSIS
# -----------------------------------------------------------------------------
# Create boxplot showing distribution of each dimension's share across occupations

dimension_shares <- balanced_data %>%
  select(Title, INR, IR, PNR, PR) %>%
  distinct() %>%
  pivot_longer(cols = c(INR, IR, PNR, PR), 
               names_to = "Dimension", values_to = "Share")

ggplot(dimension_shares, aes(x = Dimension, y = Share, fill = Dimension)) +
  geom_boxplot() +
  labs(title = "Distribution of Task Dimension Shares Across Healthcare Occupations",
       x = "Dimension",
       y = "Share",
       fill = "Dimension") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(RESULTS_DIR, "dimension_shares_distribution.png"), 
       width = 12, height = 8)

# -----------------------------------------------------------------------------
# 3. DIMENSION HEATMAP FOR TOP OCCUPATIONS
# -----------------------------------------------------------------------------
# Create heatmap showing task dimension composition for top 20 occupations

# Identify top 20 occupations by number of observations
top_occupations <- balanced_data %>%
  filter(!is.na(Title)) %>%
  group_by(Title) %>%
  summarize(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  head(20) %>%
  pull(Title)

# Prepare data for heatmap
heatmap_data <- balanced_data %>%
  filter(Title %in% top_occupations) %>%
  select(Title, INR, IR, PNR, PR) %>%
  distinct() %>%
  pivot_longer(cols = c(INR, IR, PNR, PR), 
               names_to = "Dimension", values_to = "Share")

# Create heatmap
ggplot(heatmap_data, aes(x = Dimension, y = reorder(Title, Share), fill = Share)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Task Dimension Shares for Top 20 Healthcare Occupations",
       x = "Dimension",
       y = "Occupation",
       fill = "Share") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(RESULTS_DIR, "heatmap_top_occupations.png"), 
       width = 12, height = 8)

# -----------------------------------------------------------------------------
# 4. TIME SERIES ANALYSIS BY DIMENSION
# -----------------------------------------------------------------------------
# Create time series plots of income and employment trends by dimension

# Prepare time series data aggregated by dimension and year
time_series <- balanced_data %>%
  group_by(YEAR, primary_dimension_label) %>%
  summarize(
    avg_incwage = mean(INCWAGE, na.rm = TRUE),
    median_incwage = median(INCWAGE, na.rm = TRUE),
    total_employment = sum(total_employment, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

# Plot average income trends by dimension
ggplot(time_series, aes(x = YEAR, y = avg_incwage, color = primary_dimension_label)) +
  geom_line(size = 1) +
  geom_point() +
  labs(title = "Average Income by Task Dimension Over Time",
       x = "Year",
       y = "Average Income (USD)",
       color = "Primary Dimension") +
  theme_minimal() +
  scale_y_continuous(labels = dollar_format()) +
  scale_color_brewer(palette = "Set2") +
  theme(legend.position = "bottom", legend.box = "horizontal")

ggsave(file.path(RESULTS_DIR, "average_income_by_dimension_over_time.png"), 
       width = 12, height = 8)

# Plot median income trends by dimension
ggplot(time_series, aes(x = YEAR, y = median_incwage, color = primary_dimension_label)) +
  geom_line(size = 1) +
  geom_point() +
  labs(title = "Median Income by Task Dimension Over Time",
       x = "Year",
       y = "Median Income (USD)",
       color = "Primary Dimension") +
  theme_minimal() +
  scale_y_continuous(labels = dollar_format()) +
  scale_color_brewer(palette = "Set2") +
  theme(legend.position = "bottom", legend.box = "horizontal")

ggsave(file.path(RESULTS_DIR, "median_income_by_dimension_over_time.png"), 
       width = 12, height = 8)

# Plot total employment trends by dimension
ggplot(time_series, aes(x = YEAR, y = total_employment, color = primary_dimension_label)) +
  geom_line(size = 1) +
  geom_point() +
  labs(title = "Total Employment by Task Dimension Over Time",
       x = "Year",
       y = "Total Employment",
       color = "Primary Dimension") +
  theme_minimal() +
  scale_y_continuous() +
  scale_color_brewer(palette = "Set2") +
  theme(legend.position = "bottom", legend.box = "horizontal")

ggsave(file.path(RESULTS_DIR, "total_employment_by_dimension_over_time.png"), 
       width = 12, height = 8)

# -----------------------------------------------------------------------------
# 5. GROWTH VS. INITIAL DIMENSION SHARE ANALYSIS
# -----------------------------------------------------------------------------
# Examine relationship between initial task dimension shares and growth rates

# Prepare datasets for growth analysis
growth_vs_share <- growth_data %>% 
  filter(!is.na(wage_growth) & is.finite(wage_growth) & 
         !is.na(employment_growth) & is.finite(employment_growth)) %>%
  left_join(onet_shares %>% 
          mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>% 
          select(ONETOCCSOC_2018, Title, INR, IR, PNR, PR), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# Prepare dataset for wage growth analysis
growth_vs_share_inc <- growth_data %>% 
  filter(!is.na(wage_growth) & is.finite(wage_growth)) %>%
  left_join(onet_shares %>% 
          mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>% 
          select(ONETOCCSOC_2018, Title, INR, IR, PNR, PR), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# 5.1 Create plots for each dimension vs wage growth
INR_plot <- ggplot(growth_vs_share_inc, aes(x = INR, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "green") +
  labs(title = "Income Growth vs. INR Share",
       x = "INR Share",
       y = "Income Growth (%)") +
  theme_minimal()

IR_plot <- ggplot(growth_vs_share_inc, aes(x = IR, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "purple") +
  labs(title = "Income Growth vs. IR Share",
       x = "IR Share",
       y = "Income Growth (%)") +
  theme_minimal()

PNR_plot <- ggplot(growth_vs_share_inc, aes(x = PNR, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Income Growth vs. PNR Share",
       x = "PNR Share",
       y = "Income Growth (%)") +
  theme_minimal()

PR_plot <- ggplot(growth_vs_share_inc, aes(x = PR, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Income Growth vs. PR Share",
       x = "PR Share",
       y = "Income Growth (%)") +
  theme_minimal()

# Combine all wage growth plots into a grid
income_grid <- grid.arrange(INR_plot, IR_plot, PNR_plot, PR_plot, ncol = 2)

# Save combined plot
ggsave(file.path(RESULTS_DIR, "income_growth_vs_dimension_share.png"), 
       plot = income_grid, width = 12, height = 8)

# Save individual plots for detailed analysis
ggsave(file.path(RESULTS_DIR, "income_growth_vs_INR.png"), 
       plot = INR_plot, width = 6, height = 4)  
ggsave(file.path(RESULTS_DIR, "income_growth_vs_IR.png"), 
       plot = IR_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_PNR.png"), 
       plot = PNR_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_PR.png"), 
       plot = PR_plot, width = 6, height = 4)

# 5.2 Employment growth analysis
# Prepare dataset for employment growth analysis
growth_vs_share_emp <- growth_data %>% 
  filter(!is.na(employment_growth) & is.finite(employment_growth)) %>%
  left_join(onet_shares %>% 
          mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>% 
          select(ONETOCCSOC_2018, Title, INR, IR, PNR, PR), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# Create plots for each dimension vs employment growth
INR_emp_plot <- ggplot(growth_vs_share_emp, aes(x = INR, y = employment_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "green") +
  labs(title = "Employment Growth vs. INR Share",
       x = "INR Share",
       y = "Employment Growth (%)") +
  theme_minimal()

IR_emp_plot <- ggplot(growth_vs_share_emp, aes(x = IR, y = employment_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "purple") +
  labs(title = "Employment Growth vs. IR Share",
       x = "IR Share",
       y = "Employment Growth (%)") +
  theme_minimal()

PNR_emp_plot <- ggplot(growth_vs_share_emp, aes(x = PNR, y = employment_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Employment Growth vs. PNR Share",
       x = "PNR Share",
       y = "Employment Growth (%)") +
  theme_minimal()

PR_emp_plot <- ggplot(growth_vs_share_emp, aes(x = PR, y = employment_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Employment Growth vs. PR Share",
       x = "PR Share",
       y = "Employment Growth (%)") +
  theme_minimal()

# Combine all employment growth plots into a grid
emp_grid <- grid.arrange(INR_emp_plot, IR_emp_plot, PNR_emp_plot, PR_emp_plot, ncol = 2)

# Save combined plot
ggsave(file.path(RESULTS_DIR, "employment_growth_vs_dimension_share.png"), 
       plot = emp_grid, width = 12, height = 8)

# Save individual plots
ggsave(file.path(RESULTS_DIR, "employment_growth_vs_INR.png"), 
       plot = INR_emp_plot, width = 6, height = 4)  
ggsave(file.path(RESULTS_DIR, "employment_growth_vs_IR.png"), 
       plot = IR_emp_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "employment_growth_vs_PNR.png"), 
       plot = PNR_emp_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "employment_growth_vs_PR.png"), 
       plot = PR_emp_plot, width = 6, height = 4)

# -----------------------------------------------------------------------------
# 7. REGRESSION ANALYSIS FOR WAGE GROWTH
# -----------------------------------------------------------------------------
# Run direct regression of growth on task dimensions (no PCA needed for 4 dimensions)

# Fit regression model using task dimensions
inc_growth_model <- lm(wage_growth ~ INR + IR + PNR + PR, data = growth_vs_share_inc)
inc_growth_summary <- summary(inc_growth_model)
inc_growth_tidy <- tidy(inc_growth_model, conf.int = TRUE)

# Display and save regression table
knitr::kable(inc_growth_tidy, digits = 3, 
             caption = "Regression: Income Growth vs. Task Dimensions")
write.csv(inc_growth_tidy, file.path(RESULTS_DIR, "income_growth_regression.csv"), 
          row.names = FALSE)

# -----------------------------------------------------------------------------
# 8. REGRESSION ANALYSIS FOR EMPLOYMENT GROWTH
# -----------------------------------------------------------------------------
# Run direct regression of employment growth on task dimensions

# Fit regression model using task dimensions
emp_growth_model <- lm(employment_growth ~ INR + IR + PNR + PR, data = growth_vs_share_emp)
emp_growth_summary <- summary(emp_growth_model)
emp_growth_tidy <- tidy(emp_growth_model, conf.int = TRUE)

# Display and save regression table
knitr::kable(emp_growth_tidy, digits = 3, 
             caption = "Regression: Employment Growth vs. Task Dimensions")
write.csv(emp_growth_tidy, file.path(RESULTS_DIR, "employment_growth_regression.csv"), 
          row.names = FALSE)

# -----------------------------------------------------------------------------
# 9. BUBBLE PLOT ANALYSIS: WAGE & EMPLOYMENT GROWTH BY TASK DIMENSION
# -----------------------------------------------------------------------------
# Create bubble plots showing relationship between wage growth and employment growth
# across occupations, with point size indicating occupation size and color showing
# the dominant task dimension

# Define key plot parameters
plot_title_base <- paste("Income & Employment Growth by Task Dimension (", first_year, "-", last_year, ")", sep = "")
size_column <- paste0("n_obs_", last_year)  # Column to use for bubble size

# Create filtered dataset without outliers (5th to 95th percentile range)
growth_vs_share_no_outliers <- growth_vs_share %>%
  filter(between(wage_growth, 
                quantile(wage_growth, 0.05, na.rm = TRUE),
                quantile(wage_growth, 0.95, na.rm = TRUE)) &
         between(employment_growth,
                quantile(employment_growth, 0.05, na.rm = TRUE),
                quantile(employment_growth, 0.95, na.rm = TRUE)))

# 9.1 Complete dataset bubble plot
bubble_plot <- ggplot(growth_vs_share, 
                     aes(x = wage_growth, 
                         y = employment_growth, 
                         size = get(size_column), 
                         color = dominant_dimension_label)) +
  geom_point(alpha = 0.7) +
  labs(title = plot_title_base,
       x = "Wage Growth (%)",
       y = "Employment Growth (%)",
       color = "Dominant Dimension",
       size = "Number of Workers") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "bottom", legend.box = "vertical")

# 9.2 Filtered dataset bubble plot (without outliers)
bubble_plot_no_outliers <- ggplot(growth_vs_share_no_outliers, 
                     aes(x = wage_growth, 
                         y = employment_growth, 
                         size = get(size_column), 
                         color = dominant_dimension_label)) +
  geom_point(alpha = 0.7) +
  labs(title = paste(plot_title_base, "- Without Outliers"),
       x = "Wage Growth (%)",
       y = "Employment Growth (%)",
       color = "Dominant Dimension",
       size = "Number of Workers") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "bottom", legend.box = "vertical")

# Save both visualizations to files
ggsave(file.path(RESULTS_DIR, "growth_bubble_plot.png"), 
       plot = bubble_plot, width = 12, height = 8)
ggsave(file.path(RESULTS_DIR, "growth_bubble_plot_no_outliers.png"), 
       plot = bubble_plot_no_outliers, width = 12, height = 8)

# ---- END OF SCRIPT ----