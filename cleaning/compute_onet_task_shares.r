library(data.table)

# Load the data
data <- fread("data/onet/onet_task_statements_healthcare_occupations_classified.csv")

# Compute the share of tasks for each gpt_label grouped by O*NET-SOC Code, Title, Date, and IPUMSOCCSOC
task_shares <- data[, .N, by = .(`O*NET-SOC Code`, Title, Date, IPUMSOCCSOC, gpt_label)][
  , share := N / sum(N), by = .(`O*NET-SOC Code`, Title, Date, IPUMSOCCSOC)][
  , dcast(.SD, `O*NET-SOC Code` + Title + Date + IPUMSOCCSOC ~ gpt_label, value.var = "share", fill = 0)]


# Save the result to a CSV file
fwrite(task_shares, "data/onet/onet_healthcare_occupations_with_shares.csv", row.names = FALSE)
