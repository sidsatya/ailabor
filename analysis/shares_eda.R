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
onet_shares <- fread("/Users/sidsatya/dev/ailabor/data/onet/onet_data_with_shares.csv")
ipums_data <- fread("/Users/sidsatya/dev/ailabor/data/ipums/ipums_healthcare_data.csv")

# Merge datasets based on ONETOCCSOC_2018
merged_data <- ipums_data %>%
  left_join(onet_shares %>%
              mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>%
              select(ONETOCCSOC_2018, Title, INR, IR, PNR, PR),
              by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# create total wage and total employment columns
merged_data <- merged_data %>%
  mutate(
    total_wage = INCWAGE * PERWT,
    total_employment = PERWT
  )

# Print missing match information
cat("Matched", nrow(merged_data[!is.na(merged_data$INR),]), "of", nrow(ipums_data), "rows\n")

# Define primary dimension for each occupation
# Method 1: Dimension with highest share
merged_data <- merged_data %>%
  mutate(
    primary_dimension = case_when(
      INR >= IR & INR >= PNR & INR >= PR ~ "INR",
      IR >= INR & IR >= PNR & IR >= PR ~ "IR",
      PNR >= INR & PNR >= IR & PNR >= PR ~ "PNR",
      TRUE ~ "PR"
    ),
    primary_dimension_label = case_when(
      primary_dimension == "INR" ~ "Interpersonal Non-Routine",
      primary_dimension == "IR" ~ "Interpersonal Routine",
      primary_dimension == "PNR" ~ "Personal Non-Routine",
      primary_dimension == "PR" ~ "Personal Routine"
    )
  )

# Method 2: Using threshold cutoffs
# Let's define a job as primarily in a dimension if that dimension's share > 0.5
merged_data <- merged_data %>%
  mutate(
    dominant_dimension = case_when(
      INR > 0.5 ~ "INR",
      IR > 0.5 ~ "IR",
      PNR > 0.5 ~ "PNR",
      PR > 0.5 ~ "PR",
      TRUE ~ "Mixed"
    ),
    dominant_dimension_label = case_when(
      dominant_dimension == "INR" ~ "Primarily Interpersonal Non-Routine",
      dominant_dimension == "IR" ~ "Primarily Interpersonal Routine",
      dominant_dimension == "PNR" ~ "Primarily Personal Non-Routine",
      dominant_dimension == "PR" ~ "Primarily Personal Routine",
      TRUE ~ "Mixed Tasks"
    )
  )

# Summary of occupations by primary dimension
primary_dim_summary <- merged_data %>%
  group_by(Title, primary_dimension_label) %>%
  summarize(
    count = n(),
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    avg_PNR = mean(PNR, na.rm = TRUE),
    avg_PR = mean(PR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(primary_dimension_label, desc(count))

# Summary of occupations by dominant dimension
dominant_dim_summary <- merged_data %>%
  group_by(Title, dominant_dimension_label) %>%
  summarize(
    count = n(),
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    avg_PNR = mean(PNR, na.rm = TRUE),
    avg_PR = mean(PR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(dominant_dimension_label, desc(count))

# Define first and last year for wage growth analysis
first_year <- min(merged_data$YEAR, na.rm = TRUE)
last_year <- max(merged_data$YEAR, na.rm = TRUE)
cat("Analyzing wage growth from", first_year, "to", last_year, "\n")

# Compute weighted average income and total employment by occupation and year
growth_data <- merged_data %>%
  filter(YEAR %in% c(first_year, last_year)) %>%
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
  pivot_wider(
    names_from = YEAR,
    values_from = c(avg_incwage, total_employment, total_wage, n_obs, min_wage, max_wage, zero_weights)
  ) %>%
  mutate(
    wage_growth = ((get(paste0("avg_incwage_", last_year)) / get(paste0("avg_incwage_", first_year))) - 1) * 100,
    employment_growth = ((get(paste0("total_employment_", last_year)) / get(paste0("total_employment_", first_year))) - 1) * 100
  )

# Join income growth data back to occupation dimensions
occupation_analysis <- primary_dim_summary %>%
  left_join(growth_data %>% select(Title, wage_growth, employment_growth), by = "Title")

# Create balance table by primary dimension
balance_table <- merged_data %>%
  group_by(primary_dimension_label) %>%
  summarize(
    n_occupations = n_distinct(Title),
    n_workers = n(),
    avg_incwage = mean(INCWAGE, na.rm = TRUE),
    median_incwage = median(INCWAGE, na.rm = TRUE),
    
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    avg_PNR = mean(PNR, na.rm = TRUE),
    avg_PR = mean(PR, na.rm = TRUE)
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
    avg_INR = mean(INR, na.rm = TRUE),
    avg_IR = mean(IR, na.rm = TRUE),
    avg_PNR = mean(PNR, na.rm = TRUE),
    avg_PR = mean(PR, na.rm = TRUE)
  )

# Print and save dominant dimension balance table
knitr::kable(balance_table_dominant, digits = 2, caption = "Balance Table by Dominant Task Dimension")
write.csv(balance_table_dominant, "/Users/sidsatya/dev/ailabor/results/balance_table_dominant_dimension.csv", row.names = FALSE)

# Save titles pertaining to each dimension to a txt file using a for loop
for (dim in c("INR", "IR", "PNR", "PR")) {
  write.table(
    merged_data %>% 
      filter(primary_dimension == dim) %>% 
      select(Title) %>% 
      distinct(),
    file = paste0("/Users/sidsatya/dev/ailabor/results/primary_", dim, "_titles.txt"),
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
}

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
  select(Title, INR, IR, PNR, PR) %>%
  distinct() %>%
  pivot_longer(cols = c(INR, IR, PNR, PR), names_to = "Dimension", values_to = "Share")

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
  filter(!is.na(Title)) %>%
  group_by(Title) %>%
  summarize(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  head(20) %>%
  pull(Title)

heatmap_data <- merged_data %>%
  filter(Title %in% top_occupations) %>%
  select(Title, INR, IR, PNR, PR) %>%
  distinct() %>%
  pivot_longer(cols = c(INR, IR, PNR, PR), names_to = "Dimension", values_to = "Share")

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

growth_vs_share <- growth_data %>% 
  filter(!is.na(wage_growth) & is.finite(wage_growth) & !is.na(wage_growth) & is.finite(wage_growth)) %>%
  left_join(onet_shares %>% 
          mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>% 
          select(ONETOCCSOC_2018, Title, INR, IR, PNR, PR), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

growth_vs_share_inc <- growth_data %>% 
  filter(!is.na(wage_growth) & is.finite(wage_growth)) %>%
  left_join(onet_shares %>% 
          mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>% 
          select(ONETOCCSOC_2018, Title, INR, IR, PNR, PR), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# Create scatter plots for each dimension
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

grid.arrange(INR_plot, IR_plot, PNR_plot, PR_plot, ncol = 2)

# Save plots
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_dimension_share.png", width = 12, height = 8)
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_INR.png", plot = INR_plot, width = 6, height = 4)  
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_IR.png", plot = IR_plot, width = 6, height = 4)
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_PNR.png", plot = PNR_plot, width = 6, height = 4)
ggsave("/Users/sidsatya/dev/ailabor/results/income_growth_vs_PR.png", plot = PR_plot, width = 6, height = 4)

# 6. Employment growth vs. initial dimension share
growth_vs_share_emp <- growth_data %>% 
  filter(!is.na(employment_growth) & is.finite(employment_growth)) %>%
  left_join(onet_shares %>% 
          mutate(ONETOCCSOC_2018 = as.character(ONETOCCSOC_2018)) %>% 
          select(ONETOCCSOC_2018, Title, INR, IR, PNR, PR), 
          by = c("OCCSOC_2018" = "ONETOCCSOC_2018"))

# Create scatter plots for each dimension
INR_plot <- ggplot(growth_vs_share_emp, aes(x = INR, y = employment_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "green") +
  labs(title = "Employment Growth vs. INR Share",
       x = "INR Share",
       y = "Employment Growth (%)") +
  theme_minimal()

IR_plot <- ggplot(growth_vs_share_emp, aes(x = IR, y = employment_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "purple") +
  labs(title = "Employment Growth vs. IR Share",
       x = "IR Share",
       y = "Employment Growth (%)") +
  theme_minimal()

PNR_plot <- ggplot(growth_vs_share_emp, aes(x = PNR, y = employment_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Employment Growth vs. PNR Share",
       x = "PNR Share",
       y = "Employment Growth (%)") +
  theme_minimal()

PR_plot <- ggplot(growth_vs_share_emp, aes(x = PR, y = employment_growth)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Employment Growth vs. PR Share",
       x = "PR Share",
       y = "Employment Growth (%)") +
  theme_minimal()

grid.arrange(INR_plot, IR_plot, PNR_plot, PR_plot, ncol = 2)

# Save plots
ggsave("/Users/sidsatya/dev/ailabor/results/employment_growth_vs_dimension_share.png", width = 12, height = 8)
ggsave("/Users/sidsatya/dev/ailabor/results/employment_growth_vs_INR.png", plot = INR_plot, width = 6, height = 4)  
ggsave("/Users/sidsatya/dev/ailabor/results/employment_growth_vs_IR.png", plot = IR_plot, width = 6, height = 4)
ggsave("/Users/sidsatya/dev/ailabor/results/employment_growth_vs_PNR.png", plot = PNR_plot, width = 6, height = 4)
ggsave("/Users/sidsatya/dev/ailabor/results/employment_growth_vs_PR.png", plot = PR_plot, width = 6, height = 4)


# 7. Regression analysis of wage growth and task dimensions
inc_growth_model <- lm(wage_growth ~ INR + IR + PNR + PR, data = growth_vs_share_inc)
inc_growth_summary <- summary(inc_growth_model)
inc_growth_tidy <- tidy(inc_growth_model, conf.int = TRUE)

# Create regression table
knitr::kable(inc_growth_tidy, digits = 3, 
             caption = "Regression: Income Growth vs. Task Dimensions")

write.csv(inc_growth_tidy, "results/income_growth_regression.csv", row.names = FALSE)

# 8. Regression analysis of employment growth and task dimensions
emp_growth_model <- lm(employment_growth ~ INR + IR + PNR + PR, data = growth_vs_share_inc)
emp_growth_summary <- summary(emp_growth_model)
emp_growth_tidy <- tidy(emp_growth_model, conf.int = TRUE)

# Create regression table
knitr::kable(emp_growth_tidy, digits = 3, 
             caption = "Regression: Income Growth vs. Task Dimensions")

write.csv(growth_tidy, "results/employment_growth_regression.csv", row.names = FALSE)

# 7. Bubble chart of wage growth by dimension and occupation size
# Create a version of the data without outliers
growth_vs_share_no_outliers <- growth_vs_share %>%
  filter(between(wage_growth, 
                quantile(wage_growth, 0.05, na.rm = TRUE),
                quantile(wage_growth, 0.95, na.rm = TRUE)) &
         between(employment_growth,
                quantile(employment_growth, 0.05, na.rm = TRUE),
                quantile(employment_growth, 0.95, na.rm = TRUE)))

# Create both plots
bubble_plot <- ggplot(growth_vs_share, 
                     aes(x = wage_growth, 
                         y = employment_growth, 
                         size = get(paste0("n_obs_", last_year)), 
                         color = dominant_dimension_label)) +
  geom_point(alpha = 0.7) +
  labs(title = paste("Income & Employment Growth by Task Dimension (", first_year, "-", last_year, ")", sep = ""),
       x = "Wage Growth (%)",
       y = "Employment Growth (%)",
       color = "Dominant Dimension",
       size = "Number of Workers") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

bubble_plot_no_outliers <- ggplot(growth_vs_share_no_outliers, 
                     aes(x = wage_growth, 
                         y = employment_growth, 
                         size = get(paste0("n_obs_", last_year)), 
                         color = dominant_dimension_label)) +
  geom_point(alpha = 0.7) +
  labs(title = paste("Income & Employment Growth by Task Dimension (", first_year, "-", last_year, ") - Without Outliers", sep = ""),
       x = "Wage Growth (%)",
       y = "Employment Growth (%)",
       color = "Dominant Dimension",
       size = "Number of Workers") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

# Save both plots
ggsave("/Users/sidsatya/dev/ailabor/results/growth_bubble_plot.png", plot = bubble_plot, width = 10, height = 6)
ggsave("/Users/sidsatya/dev/ailabor/results/growth_bubble_plot_no_outliers.png", plot = bubble_plot_no_outliers, width = 10, height = 6)

print(bubble_plot)
print(bubble_plot_no_outliers)


# Export results to CSV
write.csv(balance_table, "healthcare_occupation_balance_table.csv", row.names = FALSE)
write.csv(occupation_analysis, "healthcare_occupation_analysis.csv", row.names = FALSE)