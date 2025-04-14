# =============================================================================
# HEALTHCARE OCCUPATION TASK DIMENSION ANALYSIS
# =============================================================================
# This script analyzes task dimensions in healthcare occupations using ONET and IPUMS data.
# It classifies occupations by their task dimensions, analyzes wage and employment trends,
# and visualizes relationships between task composition and labor market outcomes.

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

# -----------------------------------------------------------------------------
# 2. DATA IMPORT AND PREPARATION
# -----------------------------------------------------------------------------
# Read source datasets: ONET task dimensions and IPUMS labor market data
onet_shares_path = file.path(ROOT_DIR, "data", "onet", "onet_data_with_shares_8_dim.csv")
ipums_data_path = file.path(ROOT_DIR, "data", "ipums", "ipums_healthcare_data.csv")

onet_shares <- fread(onet_shares_path)
ipums_data <- fread(ipums_data_path)

# Set directory for results and directory if it doesn't exist
RESULTS_DIR = file.path(ROOT_DIR, "results", "dim_8_results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Merge datasets based on occupation codes (ONETOCCSOC_2018 matches OCCSOC_2018)
merged_data <- ipums_data %>%
  left_join(onet_shares %>%
              mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>%
              select(ONETOCCSOC_2018, Title, IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, 
                     P_R_M, P_R_NM, P_NR_M, P_NR_NM),
              by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# Create total wage and total employment columns using person weights
merged_data <- merged_data %>%
  mutate(
    total_wage = INCWAGE * PERWT,
    total_employment = PERWT
  )

# Print matching statistics
cat("Matched", nrow(merged_data[!is.na(merged_data$IP_R_M),]), "of", nrow(ipums_data), "rows\n")

# -----------------------------------------------------------------------------
# 3. TASK DIMENSION CLASSIFICATION
# -----------------------------------------------------------------------------
# Define task dimensions (8 dimensions based on combinations of):
# - Interpersonal (IP) vs. Personal (P)
# - Routine (R) vs. Non-Routine (NR)
# - Manual (M) vs. Non-Manual (NM)

# METHOD 1: Primary dimension - classify by the dimension with the highest share
merged_data <- merged_data %>%
  mutate(
    primary_dimension = case_when(
      IP_R_M >= IP_R_NM & IP_R_M >= IP_NR_M & IP_R_M >= IP_NR_NM & 
      IP_R_M >= P_R_M & IP_R_M >= P_R_NM & IP_R_M >= P_NR_M & IP_R_M >= P_NR_NM ~ "IP_R_M",
      IP_R_NM >= IP_R_M & IP_R_NM >= IP_NR_M & IP_R_NM >= IP_NR_NM & 
      IP_R_NM >= P_R_M & IP_R_NM >= P_R_NM & IP_R_NM >= P_NR_M & IP_R_NM >= P_NR_NM ~ "IP_R_NM",
      IP_NR_M >= IP_R_M & IP_NR_M >= IP_R_NM & IP_NR_M >= IP_NR_NM & 
      IP_NR_M >= P_R_M & IP_NR_M >= P_R_NM & IP_NR_M >= P_NR_M & IP_NR_M >= P_NR_NM ~ "IP_NR_M",
      IP_NR_NM >= IP_R_M & IP_NR_NM >= IP_R_NM & IP_NR_NM >= IP_NR_M & 
      IP_NR_NM >= P_R_M & IP_NR_NM >= P_R_NM & IP_NR_NM >= P_NR_M & IP_NR_NM >= P_NR_NM ~ "IP_NR_NM",
      P_R_M >= IP_R_M & P_R_M >= IP_R_NM & P_R_M >= IP_NR_M & P_R_M >= IP_NR_NM & 
      P_R_M >= P_R_NM & P_R_M >= P_NR_M & P_R_M >= P_NR_NM ~ "P_R_M",
      P_R_NM >= IP_R_M & P_R_NM >= IP_R_NM & P_R_NM >= IP_NR_M & P_R_NM >= IP_NR_NM & 
      P_R_NM >= P_R_M & P_R_NM >= P_NR_M & P_R_NM >= P_NR_NM ~ "P_R_NM",
      P_NR_M >= IP_R_M & P_NR_M >= IP_R_NM & P_NR_M >= IP_NR_M & P_NR_M >= IP_NR_NM & 
      P_NR_M >= P_R_M & P_NR_M >= P_R_NM & P_NR_M >= P_NR_NM ~ "P_NR_M",
      TRUE ~ "P_NR_NM"
    ),
    # Create human-readable labels for the primary dimensions
    primary_dimension_label = case_when(
      primary_dimension == "IP_R_M" ~ "Interpersonal Routine Manual",
      primary_dimension == "IP_R_NM" ~ "Interpersonal Routine Non-Manual",
      primary_dimension == "IP_NR_M" ~ "Interpersonal Non-Routine Manual",
      primary_dimension == "IP_NR_NM" ~ "Interpersonal Non-Routine Non-Manual",
      primary_dimension == "P_R_M" ~ "Personal Routine Manual",
      primary_dimension == "P_R_NM" ~ "Personal Routine Non-Manual",
      primary_dimension == "P_NR_M" ~ "Personal Non-Routine Manual",
      primary_dimension == "P_NR_NM" ~ "Personal Non-Routine Non-Manual"
    )
  )

# METHOD 2: Dominant dimension - classify by dimensions exceeding 50% threshold
merged_data <- merged_data %>%
  mutate(
    dominant_dimension = case_when(
      IP_R_M > 0.5 ~ "IP_R_M",
      IP_R_NM > 0.5 ~ "IP_R_NM",
      IP_NR_M > 0.5 ~ "IP_NR_M",
      IP_NR_NM > 0.5 ~ "IP_NR_NM",
      P_R_M > 0.5 ~ "P_R_M",
      P_R_NM > 0.5 ~ "P_R_NM",
      P_NR_M > 0.5 ~ "P_NR_M",
      P_NR_NM > 0.5 ~ "P_NR_NM",
      TRUE ~ "Mixed"  # No single dimension exceeds threshold
    ),
    # Create human-readable labels for dominant dimensions
    dominant_dimension_label = case_when(
      dominant_dimension == "IP_R_M" ~ "Primarily Interpersonal Routine Manual",
      dominant_dimension == "IP_R_NM" ~ "Primarily Interpersonal Routine Non-Manual",
      dominant_dimension == "IP_NR_M" ~ "Primarily Interpersonal Non-Routine Manual",
      dominant_dimension == "IP_NR_NM" ~ "Primarily Interpersonal Non-Routine Non-Manual",
      dominant_dimension == "P_R_M" ~ "Primarily Personal Routine Manual",
      dominant_dimension == "P_R_NM" ~ "Primarily Personal Routine Non-Manual",
      dominant_dimension == "P_NR_M" ~ "Primarily Personal Non-Routine Manual",
      dominant_dimension == "P_NR_NM" ~ "Primarily Personal Non-Routine Non-Manual",
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
    avg_IP_R_M = mean(IP_R_M, na.rm = TRUE),
    avg_IP_R_NM = mean(IP_R_NM, na.rm = TRUE),
    avg_IP_NR_M = mean(IP_NR_M, na.rm = TRUE),
    avg_IP_NR_NM = mean(IP_NR_NM, na.rm = TRUE),
    avg_P_R_M = mean(P_R_M, na.rm = TRUE),
    avg_P_R_NM = mean(P_R_NM, na.rm = TRUE),
    avg_P_NR_M = mean(P_NR_M, na.rm = TRUE),
    avg_P_NR_NM = mean(P_NR_NM, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(primary_dimension_label, desc(count))

# Create summary statistics for occupations by dominant dimension
dominant_dim_summary <- balanced_data %>%
  group_by(Title, dominant_dimension_label) %>%
  summarize(
    count = n(),
    # Calculate average shares for each dimension
    avg_IP_R_M = mean(IP_R_M, na.rm = TRUE),
    avg_IP_R_NM = mean(IP_R_NM, na.rm = TRUE),
    avg_IP_NR_M = mean(IP_NR_M, na.rm = TRUE),
    avg_IP_NR_NM = mean(IP_NR_NM, na.rm = TRUE),
    avg_P_R_M = mean(P_R_M, na.rm = TRUE),
    avg_P_R_NM = mean(P_R_NM, na.rm = TRUE),
    avg_P_NR_M = mean(P_NR_M, na.rm = TRUE),
    avg_P_NR_NM = mean(P_NR_NM, na.rm = TRUE),
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
    avg_IP_R_M = mean(IP_R_M, na.rm = TRUE),
    avg_IP_R_NM = mean(IP_R_NM, na.rm = TRUE),
    avg_IP_NR_M = mean(IP_NR_M, na.rm = TRUE),
    avg_IP_NR_NM = mean(IP_NR_NM, na.rm = TRUE),
    avg_P_R_M = mean(P_R_M, na.rm = TRUE),
    avg_P_R_NM = mean(P_R_NM, na.rm = TRUE),
    avg_P_NR_M = mean(P_NR_M, na.rm = TRUE),
    avg_P_NR_NM = mean(P_NR_NM, na.rm = TRUE)
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
    avg_IP_R_M = mean(IP_R_M, na.rm = TRUE),
    avg_IP_R_NM = mean(IP_R_NM, na.rm = TRUE),
    avg_IP_NR_M = mean(IP_NR_M, na.rm = TRUE),
    avg_IP_NR_NM = mean(IP_NR_NM, na.rm = TRUE),
    avg_P_R_M = mean(P_R_M, na.rm = TRUE),
    avg_P_R_NM = mean(P_R_NM, na.rm = TRUE),
    avg_P_NR_M = mean(P_NR_M, na.rm = TRUE),
    avg_P_NR_NM = mean(P_NR_NM, na.rm = TRUE)
  )

# Display and save dominant dimension balance table
knitr::kable(balance_table_dominant, digits = 2, caption = "Balance Table by Dominant Task Dimension")
write.csv(balance_table_dominant, file.path(RESULTS_DIR, "balance_table_dominant_dimension.csv"), row.names = FALSE)

# Save occupation titles for each primary dimension to separate text files
for (dim in c("IP_R_M", "IP_R_NM", "IP_NR_M", "IP_NR_NM", "P_R_M", "P_R_NM", "P_NR_M", "P_NR_NM")) {
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
  select(Title, IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, P_R_M, P_R_NM, P_NR_M, P_NR_NM) %>%
  distinct() %>%
  pivot_longer(cols = c(IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, P_R_M, P_R_NM, P_NR_M, P_NR_NM), 
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
  select(Title, IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, P_R_M, P_R_NM, P_NR_M, P_NR_NM) %>%
  distinct() %>%
  pivot_longer(cols = c(IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, P_R_M, P_R_NM, P_NR_M, P_NR_NM), 
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
          select(ONETOCCSOC_2018, Title, IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, 
                 P_R_M, P_R_NM, P_NR_M, P_NR_NM), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# Prepare dataset for wage growth analysis
growth_vs_share_inc <- growth_data %>% 
  filter(!is.na(wage_growth) & is.finite(wage_growth)) %>%
  left_join(onet_shares %>% 
          mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>% 
          select(ONETOCCSOC_2018, Title, IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, 
                 P_R_M, P_R_NM, P_NR_M, P_NR_NM), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# 5.1 Income growth vs. Interpersonal dimensions
# Create plots for each interpersonal dimension
IP_R_M_plot <- ggplot(growth_vs_share_inc, aes(x = IP_R_M, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "green") +
  labs(title = "Income Growth vs. IP_R_M Share",
       x = "IP_R_M Share",
       y = "Income Growth (%)") +
  theme_minimal()

IP_R_NM_plot <- ggplot(growth_vs_share_inc, aes(x = IP_R_NM, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "purple") +
  labs(title = "Income Growth vs. IP_R_NM Share",
       x = "IP_R_NM Share",
       y = "Income Growth (%)") +
  theme_minimal()

IP_NR_M_plot <- ggplot(growth_vs_share_inc, aes(x = IP_NR_M, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Income Growth vs. IP_NR_M Share",
       x = "IP_NR_M Share",
       y = "Income Growth (%)") +
  theme_minimal()

IP_NR_NM_plot <- ggplot(growth_vs_share_inc, aes(x = IP_NR_NM, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Income Growth vs. IP_NR_NM Share",
       x = "IP_NR_NM Share",
       y = "Income Growth (%)") +
  theme_minimal()

# Combine interpersonal dimension plots
IP_grid <- grid.arrange(IP_R_M_plot, IP_R_NM_plot, IP_NR_M_plot, IP_NR_NM_plot, ncol = 2)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_IP_dimensions.png"), 
       plot = IP_grid, width = 12, height = 8)

# 5.2 Income growth vs. Personal dimensions
# Create plots for each personal dimension
P_R_M_plot <- ggplot(growth_vs_share_inc, aes(x = P_R_M, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "orange") +
  labs(title = "Income Growth vs. P_R_M Share",
       x = "P_R_M Share",
       y = "Income Growth (%)") +
  theme_minimal()

P_R_NM_plot <- ggplot(growth_vs_share_inc, aes(x = P_R_NM, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "brown") +
  labs(title = "Income Growth vs. P_R_NM Share",
       x = "P_R_NM Share",
       y = "Income Growth (%)") +
  theme_minimal()

P_NR_M_plot <- ggplot(growth_vs_share_inc, aes(x = P_NR_M, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "cyan") +
  labs(title = "Income Growth vs. P_NR_M Share",
       x = "P_NR_M Share",
       y = "Income Growth (%)") +
  theme_minimal()

P_NR_NM_plot <- ggplot(growth_vs_share_inc, aes(x = P_NR_NM, y = wage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "magenta") +
  labs(title = "Income Growth vs. P_NR_NM Share",
       x = "P_NR_NM Share",
       y = "Income Growth (%)") +
  theme_minimal()

# Combine personal dimension plots
P_grid <- grid.arrange(P_R_M_plot, P_R_NM_plot, P_NR_M_plot, P_NR_NM_plot, ncol = 2)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_P_dimensions.png"), 
       plot = P_grid, width = 12, height = 8)

# Save individual plots for detailed analysis
ggsave(file.path(RESULTS_DIR, "income_growth_vs_IP_R_M.png"), 
       plot = IP_R_M_plot, width = 6, height = 4)  
ggsave(file.path(RESULTS_DIR, "income_growth_vs_IP_R_NM.png"), 
       plot = IP_R_NM_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_IP_NR_M.png"), 
       plot = IP_NR_M_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_IP_NR_NM.png"), 
       plot = IP_NR_NM_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_P_R_M.png"), 
       plot = P_R_M_plot, width = 6, height = 4)  
ggsave(file.path(RESULTS_DIR, "income_growth_vs_P_R_NM.png"), 
       plot = P_R_NM_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_P_NR_M.png"), 
       plot = P_NR_M_plot, width = 6, height = 4)
ggsave(file.path(RESULTS_DIR, "income_growth_vs_P_NR_NM.png"), 
       plot = P_NR_NM_plot, width = 6, height = 4)

# 5.3 Employment growth analysis
# Prepare dataset for employment growth analysis
growth_vs_share_emp <- growth_data %>% 
  filter(!is.na(employment_growth) & is.finite(employment_growth)) %>%
  left_join(onet_shares %>% 
          mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>% 
          select(ONETOCCSOC_2018, Title, IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, 
                 P_R_M, P_R_NM, P_NR_M, P_NR_NM), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# Create employment growth plots for interpersonal dimensions
IP_emp_grid <- grid.arrange(
  ggplot(growth_vs_share_emp, aes(x = IP_R_M, y = employment_growth)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", color = "green") +
    labs(title = "Employment Growth vs. IP_R_M Share",
         x = "IP_R_M Share",
         y = "Employment Growth (%)") +
    theme_minimal(),
  
  ggplot(growth_vs_share_emp, aes(x = IP_R_NM, y = employment_growth)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", color = "purple") +
    labs(title = "Employment Growth vs. IP_R_NM Share",
         x = "IP_R_NM Share",
         y = "Employment Growth (%)") +
    theme_minimal(),
    
  ggplot(growth_vs_share_emp, aes(x = IP_NR_M, y = employment_growth)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", color = "blue") +
    labs(title = "Employment Growth vs. IP_NR_M Share",
         x = "IP_NR_M Share",
         y = "Employment Growth (%)") +
    theme_minimal(),
    
  ggplot(growth_vs_share_emp, aes(x = IP_NR_NM, y = employment_growth)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", color = "red") +
    labs(title = "Employment Growth vs. IP_NR_NM Share",
         x = "IP_NR_NM Share",
         y = "Employment Growth (%)") +
    theme_minimal(),
  
  ncol = 2
)

# Create employment growth plots for personal dimensions
P_emp_grid <- grid.arrange(
  ggplot(growth_vs_share_emp, aes(x = P_R_M, y = employment_growth)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", color = "orange") +
    labs(title = "Employment Growth vs. P_R_M Share",
         x = "P_R_M Share",
         y = "Employment Growth (%)") +
    theme_minimal(),
  
  ggplot(growth_vs_share_emp, aes(x = P_R_NM, y = employment_growth)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", color = "brown") +
    labs(title = "Employment Growth vs. P_R_NM Share",
         x = "P_R_NM Share",
         y = "Employment Growth (%)") +
    theme_minimal(),
    
  ggplot(growth_vs_share_emp, aes(x = P_NR_M, y = employment_growth)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", color = "cyan") +
    labs(title = "Employment Growth vs. P_NR_M Share",
         x = "P_NR_M Share",
         y = "Employment Growth (%)") +
    theme_minimal(),
    
  ggplot(growth_vs_share_emp, aes(x = P_NR_NM, y = employment_growth)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", color = "magenta") +
    labs(title = "Employment Growth vs. P_NR_NM Share",
         x = "P_NR_NM Share",
         y = "Employment Growth (%)") +
    theme_minimal(),
  
  ncol = 2
)

# Save employment growth plots
ggsave(file.path(RESULTS_DIR, "employment_growth_vs_IP_dimensions.png"), 
       plot = IP_emp_grid, width = 12, height = 8)
ggsave(file.path(RESULTS_DIR, "employment_growth_vs_P_dimensions.png"), 
       plot = P_emp_grid, width = 12, height = 8)

# -----------------------------------------------------------------------------
# 7. REGRESSION ANALYSIS WITH PCA FOR WAGE GROWTH
# -----------------------------------------------------------------------------
# Use principal component analysis to reduce dimension and identify key factors

# Prepare data for PCA analysis of wage growth
growth_vs_share_clean <- growth_vs_share_inc %>%
  select(wage_growth, avg_incwage_2008, avg_incwage_2023, 
         IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, P_R_M, P_R_NM, P_NR_M, P_NR_NM) %>%
  mutate(wage_growth_log = log(avg_incwage_2023/avg_incwage_2008)) %>%
  na.omit()

# Extract task dimension columns for PCA
task_shares <- growth_vs_share_clean %>%
  select(-wage_growth_log, -avg_incwage_2008, -avg_incwage_2023, -wage_growth)

# Run PCA on task dimensions
pca_task_shares_inc <- prcomp(task_shares, center=TRUE, scale. = TRUE)
pc_inc_df <- as.data.frame(pca_task_shares_inc$x)   # Extract PC scores

# Fit regression model using principal components
inc_growth_model <- lm(wage_growth_log ~ PC1 + PC2 + PC3 + PC4, 
                      data = cbind(growth_vs_share_clean[,"wage_growth_log"], pc_inc_df))
inc_growth_summary <- summary(inc_growth_model)
inc_growth_tidy <- tidy(inc_growth_model, conf.int = TRUE)

# Display and save regression table
knitr::kable(inc_growth_tidy, digits = 3, 
             caption = "Regression: Income Growth vs. Task Dimensions")
write.csv(inc_growth_tidy, file.path(RESULTS_DIR, "income_growth_regression.csv"), 
          row.names = FALSE)

# Visualize PCA loadings
# Extract PC loadings
loadings <- pca_task_shares_inc$rotation[, 1:4]
loadings_df <- as.data.frame(loadings)
loadings_df$Variable <- rownames(loadings_df)

# Convert to long format for plotting
loadings_long <- loadings_df %>%
  pivot_longer(cols = c("PC1", "PC2", "PC3", "PC4"),
               names_to = "Component",
               values_to = "Loading")

# Create individual loading plots for each PC
pc1_plot <- ggplot(loadings_long %>% filter(Component == "PC1"), 
                   aes(x = reorder(Variable, Loading), y = Loading, fill = Loading > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "PC1 Loadings", x = "", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

pc2_plot <- ggplot(loadings_long %>% filter(Component == "PC2"), 
                   aes(x = reorder(Variable, Loading), y = Loading, fill = Loading > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "PC2 Loadings", x = "", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

pc3_plot <- ggplot(loadings_long %>% filter(Component == "PC3"), 
                   aes(x = reorder(Variable, Loading), y = Loading, fill = Loading > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "PC3 Loadings", x = "", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

pc4_plot <- ggplot(loadings_long %>% filter(Component == "PC4"), 
                   aes(x = reorder(Variable, Loading), y = Loading, fill = Loading > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "PC4 Loadings", x = "", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

# Combine loading plots
loadings_grid <- grid.arrange(pc1_plot, pc2_plot, pc3_plot, pc4_plot, ncol = 2)

# Save the loadings visualization
ggsave(file.path(RESULTS_DIR, "inc_pca_loadings_plot.png"), 
       plot = loadings_grid, width = 12, height = 10)

# Display the loadings visualization
print(loadings_grid)

# -----------------------------------------------------------------------------
# 8. REGRESSION ANALYSIS WITH PCA FOR EMPLOYMENT GROWTH
# -----------------------------------------------------------------------------
# Similar PCA approach for analyzing employment growth relationships

# Prepare data for PCA analysis of employment growth
growth_vs_share_clean <- growth_vs_share_emp %>%
  select(employment_growth, total_employment_2008, total_employment_2023, 
         IP_R_M, IP_R_NM, IP_NR_M, IP_NR_NM, P_R_M, P_R_NM, P_NR_M, P_NR_NM) %>%
  mutate(emp_growth_log = log(total_employment_2023/total_employment_2008)) %>%
  na.omit()

# Extract task dimension columns for PCA
task_shares <- growth_vs_share_clean %>%
  select(-emp_growth_log, -total_employment_2008, -total_employment_2023, -employment_growth)

# Run PCA on task dimensions
pca_task_shares_emp <- prcomp(task_shares, center=TRUE, scale. = TRUE)
pc_emp_df <- as.data.frame(pca_task_shares_emp$x)   # Extract PC scores

# Fit regression model using principal components
emp_growth_model <- lm(emp_growth_log ~ PC1 + PC2 + PC3 + PC4, 
                      data = cbind(growth_vs_share_clean[,"emp_growth_log"], pc_emp_df))
emp_growth_summary <- summary(emp_growth_model)
emp_growth_tidy <- tidy(emp_growth_model, conf.int = TRUE)

# Display and save regression table
knitr::kable(emp_growth_tidy, digits = 3, 
             caption = "Regression: Employment Growth vs. Task Dimensions")
write.csv(emp_growth_tidy, file.path(RESULTS_DIR, "employment_growth_regression.csv"), 
          row.names = FALSE)

# Visualize PCA loadings for employment analysis
# Extract PC loadings
loadings <- pca_task_shares_emp$rotation[, 1:4]
loadings_df <- as.data.frame(loadings)
loadings_df$Variable <- rownames(loadings_df)

# Convert to long format for plotting
loadings_long <- loadings_df %>%
  pivot_longer(cols = c("PC1", "PC2", "PC3", "PC4"),
               names_to = "Component",
               values_to = "Loading")

# Create individual loading plots for each PC
pc1_plot <- ggplot(loadings_long %>% filter(Component == "PC1"), 
                   aes(x = reorder(Variable, Loading), y = Loading, fill = Loading > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "PC1 Loadings", x = "", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

pc2_plot <- ggplot(loadings_long %>% filter(Component == "PC2"), 
                   aes(x = reorder(Variable, Loading), y = Loading, fill = Loading > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "PC2 Loadings", x = "", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

pc3_plot <- ggplot(loadings_long %>% filter(Component == "PC3"), 
                   aes(x = reorder(Variable, Loading), y = Loading, fill = Loading > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "PC3 Loadings", x = "", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

pc4_plot <- ggplot(loadings_long %>% filter(Component == "PC4"), 
                   aes(x = reorder(Variable, Loading), y = Loading, fill = Loading > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "PC4 Loadings", x = "", y = "Loading") +
  theme_minimal() +
  theme(legend.position = "none")

# Combine loading plots
loadings_grid <- grid.arrange(pc1_plot, pc2_plot, pc3_plot, pc4_plot, ncol = 2)

# Save the loadings visualization
ggsave(file.path(RESULTS_DIR, "emp_pca_loadings_plot.png"), 
       plot = loadings_grid, width = 12, height = 10)

# Display the loadings visualization
print(loadings_grid)

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

# Display the plots
print(bubble_plot)
print(bubble_plot_no_outliers)

# ---- END OF SCRIPT ----