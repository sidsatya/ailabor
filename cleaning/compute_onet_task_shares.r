library(data.table)

# Load the data
data <- fread("data/onet/onet_task_statements_classified.csv")

# Compute the share of tasks for each gpt_label grouped by O*NET-SOC Code, Title, Date, and IPUMSOCCSOC
task_shares <- data[, .N, by = .(`O*NET-SOC Code`, Title, Date, ONETOCCSOC_2018, gpt_label)][
  , share := N / sum(N), by = .(`O*NET-SOC Code`, Title, Date, ONETOCCSOC_2018)][
  , dcast(.SD, `O*NET-SOC Code` + Title + Date + ONETOCCSOC_2018 ~ gpt_label, value.var = "share", fill = 0)]

# Rename the columns to match the original data
setnames(task_shares, old = names(task_shares)[-c(1:4)], new = paste0("share_", names(task_shares)[-c(1:4)]))

# For each ONETOCCSOC_2018, keep info from latest date and average the shares
task_shares <- task_shares[order(-Date), .(
  `O*NET-SOC Code` = first(`O*NET-SOC Code`),
  Title = first(Title),
  Date = first(Date),
  INR = mean(share_INR),
  IR = mean(share_IR),
  PNR = mean(share_PNR),
  PR = mean(share_PR)
), by = ONETOCCSOC_2018]

# Save the result to a CSV file
fwrite(task_shares, "data/onet/onet_data_with_shares.csv", row.names = FALSE)
