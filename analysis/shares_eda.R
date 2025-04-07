library(data.table)
library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)
library(scales)
library(stringr)
library(gridExtra)
library(broom)

# Read datasets
onet_shares <- fread("/Users/sidsatya/dev/ailabor/data/onet/onet_healthcare_occupations_with_shares.csv")
ipums_data <- fread("/Users/sidsatya/dev/ailabor/data/ipums/ipums_healthcare_data.csv")

# Merge datasets based on IPUMSOCCSOC
merged_data <- ipums_data %>%
  left_join(onet_shares %>%
              mutate(IPUMSOCCSOC = as.character(IPUMSOCCSOC)) %>%
              select(IPUMSOCCSOC, Title, DMNR, DMR, INR, IR),
              by = c("OCCSOC" = "IPUMSOCCSOC"))

# Print missing match information
cat("Matched", nrow(merged_data[!is.na(merged_data$DMNR),]), "of", nrow(ipums_data), "rows\n")

# Define primary dimension for each occupation
# Method 1: Dimension with highest share
merged_data <- merged_data %>%
  mutate(
    primary_dimension = case_when(
      DMNR >= DMR & DMNR >= INR & DMNR >= IR ~ "DMNR",
      DMR >= DMNR & DMR >= INR & DMR >= IR ~ "DMR",
      INR >= DMNR & INR >= DMR & INR >= IR ~ "INR",
      TRUE ~ "IR"
    ),
    primary_dimension_label = case_when(
      primary_dimension == "DMNR" ~ "Decision Making Non-Routine",
      primary_dimension == "DMR" ~ "Decision Making Routine",
      primary_dimension == "INR" ~ "Interpersonal Non-Routine",
      primary_dimension == "IR" ~ "Interpersonal Routine"
    )
  )

# Method 2: Using threshold cutoffs
# Let's define a job as primarily in a dimension if that dimension's share > 0.5
merged_data <- merged_data %>%
  mutate(
    dominant_dimension = case_when(
      DMNR > 0.5 ~ "DMNR",
      DMR > 0.5 ~ "DMR",
      INR > 0.5 ~ "INR",
      IR > 0.5 ~ "IR",
      TRUE ~ "Mixed"
    ),
    dominant_dimension_label = case_when(
      dominant_dimension == "DMNR" ~ "Primarily Dexterous Manual Non-Routine",
      dominant_dimension == "DMR" ~ "Primarily Dexterous Manual Routine",
      dominant_dimension == "INR" ~ "Primarily Information Non-Routine",
      dominant_dimension == "IR" ~ "Primarily Information Routine",
      TRUE ~ "Mixed Tasks"
    )
  )

# Summary of occupations by primary dimension
primary_dim_summary <- merged_data %>%
  group_by(Title, primary_dimension_label) %>%
  summarize(
    count = n(),
    avg_DMNR = mean(DMNR, na.rm = TRUE),
    avg_DMR = mean(DMR, na.rm = TRUE),
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(primary_dimension_label, desc(count))

# Summary of occupations by dominant dimension
dominant_dim_summary <- merged_data %>%
  group_by(Title, dominant_dimension_label) %>%
  summarize(
    count = n(),
    avg_DMNR = mean(DMNR, na.rm = TRUE),
    avg_DMR = mean(DMR, na.rm = TRUE),
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(dominant_dimension_label, desc(count))

# Define first and last year for wage growth analysis
first_year <- min(merged_data$YEAR, na.rm = TRUE)
last_year <- max(merged_data$YEAR, na.rm = TRUE)
cat("Analyzing wage growth from", first_year, "to", last_year, "\n")

# Calculate income growth by occupation between first and last year
income_growth <- merged_data %>%
  filter(YEAR %in% c(first_year, last_year)) %>%
  group_by(Title, OCCSOC, YEAR) %>%
  summarize(
    n_workers = n(),
    avg_incwage = mean(INCWAGE, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = YEAR,
    values_from = c(n_workers, avg_incwage)
  ) %>%
  mutate(
    incwage_growth = ((get(paste0("avg_incwage_", last_year)) / get(paste0("avg_incwage_", first_year))) - 1) * 100
  )

# Join income growth data back to occupation dimensions
occupation_analysis <- primary_dim_summary %>%
  left_join(income_growth %>% select(Title, incwage_growth), by = "Title")

# Create balance table by primary dimension
balance_table <- merged_data %>%
  group_by(primary_dimension_label) %>%
  summarize(
    n_occupations = n_distinct(Title),
    n_workers = n(),
    avg_incwage = mean(INCWAGE, na.rm = TRUE),
    median_incwage = median(INCWAGE, na.rm = TRUE),
    avg_DMNR = mean(DMNR, na.rm = TRUE),
    avg_DMR = mean(DMR, na.rm = TRUE),
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE)
  )

# Print and save primary dimension balance table
knitr::kable(balance_table, digits = 2, caption = "Balance Table by Primary Task Dimension")
write.csv(balance_table, "/Users/sidsatya/dev/ailabor/results/balance_table_primary_dimension.csv", row.names = FALSE)

# Create balance table by dominant dimension
balance_table_dominant <- merged_data %>%
  group_by(dominant_dimension_label) %>%
  summarize(
    n_occupations = n_distinct(Title),
    n_workers = n(),
    avg_incwage = mean(INCWAGE, na.rm = TRUE),
    median_incwage = median(INCWAGE, na.rm = TRUE),
    avg_DMNR = mean(DMNR, na.rm = TRUE),
    avg_DMR = mean(DMR, na.rm = TRUE),
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE)
  )

# Print and save dominant dimension balance table
knitr::kable(balance_table_dominant, digits = 2, caption = "Balance Table by Dominant Task Dimension")
write.csv(balance_table_dominant, "/Users/sidsatya/dev/ailabor/results/balance_table_dominant_dimension.csv", row.names = FALSE)

# ---- VISUALIZATIONS ----

# 1. Occupation count by primary dimension
ggplot(primary_dim_summary, aes(x = primary_dimension_label, fill = primary_dimension_label)) +
  geom_bar(aes(weight = count)) +
  labs(title = "Number of Workers by Primary Task Dimension",
       x = "Primary Dimension",
       y = "Count",
       fill = "Primary Dimension") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("/Users/sidsatya/dev/ailabor/results/occupation_count_by_primary_dimension.png", width = 10, height = 6)

# 2. Distribution of dimension shares across occupations
dimension_shares <- merged_data %>%
  select(Title, DMNR, DMR, INR, IR) %>%
  distinct() %>%
  pivot_longer(cols = c(DMNR, DMR, INR, IR), names_to = "Dimension", values_to = "Share")

ggplot(dimension_shares, aes(x = Dimension, y = Share, fill = Dimension)) +
  geom_boxplot() +
  labs(title = "Distribution of Task Dimension Shares Across Healthcare Occupations",
       x = "Dimension",
       y = "Share",
       fill = "Dimension") +
  theme_minimal()

ggsave("/Users/sidsatya/dev/ailabor/results/dimension_shares_distribution.png", width = 10, height = 6)

# 3. Heatmap of dimension shares by top occupations
top_occupations <- merged_data %>%
  group_by(Title) %>%
  summarize(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  head(20) %>%
  pull(Title)

heatmap_data <- merged_data %>%
  filter(Title %in% top_occupations) %>%
  select(Title, DMNR, DMR, INR, IR) %>%
  distinct() %>%
  pivot_longer(cols = c(DMNR, DMR, INR, IR), names_to = "Dimension", values_to = "Share")

ggplot(heatmap_data, aes(x = Dimension, y = reorder(Title, Share), fill = Share)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Task Dimension Shares for Top 20 Healthcare Occupations",
       x = "Dimension",
       y = "Occupation",
       fill = "Share") +
  theme_minimal()

ggsave("/Users/sidsatya/dev/ailabor/results/heatmap_top_occupations.png", width = 10, height = 6)

# 4. Income by dimension over time
time_series <- merged_data %>%
  group_by(YEAR, primary_dimension_label) %>%
  summarize(
    avg_incwage = mean(INCWAGE, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

ggplot(time_series, aes(x = YEAR, y = avg_incwage, color = primary_dimension_label)) +
  geom_line(size = 1) +
  geom_point() +
  labs(title = "Average Income by Task Dimension Over Time",
       x = "Year",
       y = "Average Income (USD)",
       color = "Primary Dimension") +
  theme_minimal() +
  scale_y_continuous(labels = dollar_format()) +
  scale_color_brewer(palette = "Set1")

ggsave("/Users/sidsatya/dev/ailabor/results/income_by_dimension_over_time.png", width = 10, height = 6)

# 5. Income growth vs. initial dimension share
growth_vs_share <- income_growth %>%
  left_join(onet_shares %>% 
          mutate(IPUMSOCCSOC = as.character(IPUMSOCCSOC)) %>% 
          select(IPUMSOCCSOC, Title, DMNR, DMR, INR, IR), 
          by = c("OCCSOC" = "IPUMSOCCSOC"))

# Create scatter plots for each dimension
DMNR_plot <- ggplot(growth_vs_share, aes(x = DMNR, y = incwage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Income Growth vs. DMNR Share",
       x = "DMNR Share",
       y = "Income Growth (%)") +
  theme_minimal()

DMR_plot <- ggplot(growth_vs_share, aes(x = DMR, y = incwage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Income Growth vs. DMR Share",
       x = "DMR Share",
       y = "Income Growth (%)") +
  theme_minimal()

INR_plot <- ggplot(growth_vs_share, aes(x = INR, y = incwage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "green") +
  labs(title = "Income Growth vs. INR Share",
       x = "INR Share",
       y = "Income Growth (%)") +
  theme_minimal()

IR_plot <- ggplot(growth_vs_share, aes(x = IR, y = incwage_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "purple") +
  labs(title = "Income Growth vs. IR Share",
       x = "IR Share",
       y = "Income Growth (%)") +
  theme_minimal()

grid.arrange(DMNR_plot, DMR_plot, INR_plot, IR_plot, ncol = 2)

# Save plots
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_dimension_share.png", width = 12, height = 8)
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_DMNR.png", plot = DMNR_plot, width = 6, height = 4)  
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_DMR.png", plot = DMR_plot, width = 6, height = 4)
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_INR.png", plot = INR_plot, width = 6, height = 4)
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_IR.png", plot = IR_plot, width = 6, height = 4)

# 6. Regression analysis of wage growth and task dimensions
growth_model <- lm(incwage_growth ~ DMNR + DMR + INR + IR, data = growth_vs_share)
growth_summary <- summary(growth_model)
growth_tidy <- tidy(growth_model, conf.int = TRUE)

# Create regression table
knitr::kable(growth_tidy, digits = 3, 
             caption = "Regression: Income Growth vs. Task Dimensions")

# 7. Bubble chart of wage growth by dimension and occupation size
bubble_plot <- ggplot(growth_vs_share, 
                     aes(x = incwage_growth, 
                         y = earnings_growth, 
                         size = get(paste0("n_workers_", last_year)), 
                         color = dominant_dimension)) +
  geom_point(alpha = 0.7) +
  labs(title = paste("Income & Earnings Growth by Task Dimension (", first_year, "-", last_year, ")", sep = ""),
       x = "Wage Growth (%)",
       y = "Earnings Growth (%)",
       color = "Dominant Dimension",
       size = "Number of Workers") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

print(bubble_plot)

# 8. Task composition change over time (if available in multiple years)
if(length(unique(onet_shares$Date)) > 1) {
  # Convert Date column to year
  onet_shares$year <- as.numeric(str_sub(onet_shares$Date, -4))
  
  # Analyze change in task composition
  task_change <- onet_shares %>%
    group_by(year) %>%
    summarize(
      avg_DMNR = mean(DMNR, na.rm = TRUE),
      avg_DMR = mean(DMR, na.rm = TRUE),
      avg_INR = mean(INR, na.rm = TRUE),
      avg_IR = mean(IR, na.rm = TRUE)
    ) %>%
    pivot_longer(cols = starts_with("avg_"), 
                 names_to = "dimension", 
                 values_to = "share") %>%
    mutate(dimension = str_remove(dimension, "avg_"))
  
  ggplot(task_change, aes(x = year, y = share, color = dimension, group = dimension)) +
    geom_line(size = 1) +
    geom_point() +
    labs(title = "Change in Task Composition Over Time",
         x = "Year",
         y = "Average Share",
         color = "Dimension") +
    theme_minimal()
}

# Export results to CSV
write.csv(balance_table, "healthcare_occupation_balance_table.csv", row.names = FALSE)
write.csv(occupation_analysis, "healthcare_occupation_analysis.csv", row.names = FALSE)
write.csv(growth_tidy, "income_growth_regression.csv", row.names = FALSE)

# Print summary and conclusions
cat("\nSUMMARY OF FINDINGS:\n")
cat("1. Distribution of occupations by primary dimension:\n")
print(table(merged_data$primary_dimension_label))
cat("\n2. Distribution of occupations by dominant dimension (>50% share):\n")
print(table(merged_data$dominant_dimension_label))
cat("\n3. Regression analysis of income growth vs. task dimensions:\n")
print(growth_summary)
cat("\n4. Average income growth by primary dimension:\n")
merged_data %>%
  group_by(primary_dimension_label) %>%
  inner_join(income_growth, by = "Title") %>%
  summarize(avg_growth = mean(incwage_growth, na.rm = TRUE)) %>%
  print()

