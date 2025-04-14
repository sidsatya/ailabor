library(data.table)

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

# --- Configuration ---
# Define the dimension data types to process
dim_data_values <- c(4, 8)

# --- Loop through Dimension Types ---
for (dim_data in dim_data_values) {

  message(sprintf("\n--- Processing for dim_data = %d ---", dim_data))

  # Validate dim_data (redundant within loop but good practice)
  if (!dim_data %in% c(4, 8)) {
    warning(sprintf("Skipping invalid dimension data: %d", dim_data))
    next # Skip to the next iteration
  }

  # --- Load Data ---
  # Construct the input file path based on the current dim_data in the loop
  input_file <- sprintf(file.path(ROOT_DIR, "data/onet/onet_task_statements_classified_%d_dim.csv"), dim_data)
  message("Loading data from: ", input_file)
  # Load the classified task statements data
  # Use tryCatch for robustness in case a file is missing
  data <- tryCatch({
    fread(input_file)
  }, error = function(e) {
    warning(sprintf("Could not load file %s: %s", input_file, e$message))
    return(NULL) # Return NULL if file loading fails
  })

  # Skip processing if data loading failed
  if (is.null(data)) {
    next
  }

  # --- Compute Task Shares ---
  # Calculate the count of tasks for each gpt_label within each ONETOCCSOC_2018 group
  task_counts_agg <- data[, .N, by = .(ONETOCCSOC_2018, gpt_label)]

  # Calculate share within ONETOCCSOC_2018 and reshape to wide format
  # This calculates the share based on the total tasks for that ONETOCCSOC_2018
  task_shares_wide_agg <- task_counts_agg[, share := N / sum(N), by = ONETOCCSOC_2018][
    , dcast(.SD, ONETOCCSOC_2018 ~ gpt_label, value.var = "share", fill = 0)]

  # --- Rename Share Columns ---
  # Identify the newly created share columns (all columns except the first grouping column)
  share_cols_original <- names(task_shares_wide_agg)[-1] # Exclude ONETOCCSOC_2018
  # Create new names by prefixing "share_"
  share_cols_new <- paste0("share_", share_cols_original)
  # Rename the columns
  setnames(task_shares_wide_agg, old = share_cols_original, new = share_cols_new)

  # --- Add Metadata ---
  # Get the latest metadata for each ONETOCCSOC_2018
  metadata <- data[order(-Date), .SD[1], by = ONETOCCSOC_2018, .SDcols = c("O*NET-SOC Code", "Title", "Date")]

  # Join metadata with shares
  final_data <- metadata[task_shares_wide_agg, on = "ONETOCCSOC_2018"]

  # Reorder columns for clarity
  setcolorder(final_data, c("ONETOCCSOC_2018", "O*NET-SOC Code", "Title", "Date", share_cols_new))

  # --- Save Results ---
  # Construct the output file path based on the current dim_data in the loop
  output_file <- sprintf(file.path(ROOT_DIR, "data/onet/onet_data_with_shares_%d_dim_2.csv"), dim_data)
  message("Saving aggregated shares to: ", output_file)
  # Save the aggregated task shares to a CSV file
  fwrite(final_data, output_file, row.names = FALSE)

  message(sprintf("--- Finished processing for dim_data = %d ---", dim_data))

} # End of loop for dim_data_values

message("\nScript finished processing all specified dimension types.")

