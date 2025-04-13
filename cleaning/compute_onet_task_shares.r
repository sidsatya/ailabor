library(data.table)

# Load the data
dim_data = 4
if (dim_data == 8) {
  # Load the data for 8 dimensions
  data <- fread("data/onet/onet_task_statements_classified_8_dim.csv")
} else if (dim_data == 4) {
  # Load the data for 6 dimensions
  data <- fread("data/onet/onet_task_statements_classified_4_dim.csv")
} else {
  stop("Invalid dimension data specified.")
}

# Compute the share of tasks for each gpt_label grouped by O*NET-SOC Code, Title, Date, and IPUMSOCCSOC
task_shares <- data[, .N, by = .(`O*NET-SOC Code`, Title, Date, ONETOCCSOC_2018, gpt_label)][
  , share := N / sum(N), by = .(`O*NET-SOC Code`, Title, Date, ONETOCCSOC_2018)][
  , dcast(.SD, `O*NET-SOC Code` + Title + Date + ONETOCCSOC_2018 ~ gpt_label, value.var = "share", fill = 0)]

# Rename the columns to match the original data
setnames(task_shares, old = names(task_shares)[-c(1:4)], new = paste0("share_", names(task_shares)[-c(1:4)]))

# For each ONETOCCSOC_2018, keep info from latest date and average the shares
if (dim_data == 8) { 
  task_shares <- task_shares[order(-Date), .(
    `O*NET-SOC Code` = first(`O*NET-SOC Code`),
    Title = first(Title),
    Date = first(Date),
    IP_R_M = mean(`share_IP-R-M`),
    IP_R_NM = mean(`share_IP-R-NM`),
    IP_NR_M = mean(`share_IP-NR-M`),
    IP_NR_NM = mean(`share_IP-NR-NM`),
    P_R_M = mean(`share_P-R-M`),
    P_R_NM = mean(`share_P-R-NM`),
    P_NR_M = mean(`share_P-NR-M`),
    P_NR_NM = mean(`share_P-NR-NM`)
  ), by = ONETOCCSOC_2018]
} else if (dim_data == 4) {
  task_shares <- task_shares[order(-Date), .(
    `O*NET-SOC Code` = first(`O*NET-SOC Code`),
    Title = first(Title),
    Date = first(Date),
    INR = mean(share_INR),
    IR = mean(share_IR),
    PNR = mean(share_PNR),
    PR = mean(share_PR)
  ), by = ONETOCCSOC_2018]
} else {
  stop("Invalid dimension data specified.") 
}

# Save the result to a CSV file
if (dim_data == 8) { 
  fwrite(task_shares, "data/onet/onet_data_with_shares_8_dim.csv", row.names = FALSE)

} else if (dim_data == 4) { 
  fwrite(task_shares, "data/onet/onet_data_with_shares_4_dim.csv", row.names = FALSE)
} else {
  stop("Invalid dimension data specified.")
}

