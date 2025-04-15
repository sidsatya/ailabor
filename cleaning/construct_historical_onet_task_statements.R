library(data.table)
library(dplyr)
library(tidyr)
library(readxl) # Added for reading Excel files
library(lubridate) # Added for date parsing
library(stringr) # Added for string manipulation

# Check if ROOT_DIR exists as an R variable in the global environment
if (exists("ROOT_DIR", envir = .GlobalEnv)) {
  ROOT_DIR <- get("ROOT_DIR", envir = .GlobalEnv)
  message("Using ROOT_DIR from global R environment: ", ROOT_DIR)
} else {
  # Fallback: set manually if not found (though it should be if run via run_main_R_files.R)
  ROOT_DIR <- "/Users/sidsatya/dev/ailabor"  # Change to your actual project path
  warning("ROOT_DIR not found in R's global environment. Using fallback path: ", ROOT_DIR)
}

# Define the directory containing the ONET task files
onet_task_dir <- file.path(ROOT_DIR, "data/onet/historical_onet_task_statements")

# List all files in the directory
all_files <- list.files(onet_task_dir, full.names = TRUE)

# List to store individual dataframes
all_tasks_list <- list()

# Month mapping for date parsing
month_map <- c(jan = 1, feb = 2, mar = 3, apr = 4, may = 5, jun = 6,
               jul = 7, aug = 8, sep = 9, oct = 10, nov = 11, dec = 12)

# Loop through each file
for (file_path in all_files) {
  file_name <- basename(file_path)
  message("Processing file: ", file_name)

  # Extract year and month from filename (adjust regex if naming pattern varies)
  # Example pattern: task_statements_YYYY_MMM.ext or Tasks_YYYY.ext
  year_month_match <- str_match(file_name, "(\\d{4})_?([a-zA-Z]{3})?")
  year <- year_month_match[, 2]
  month_abbr <- tolower(year_month_match[, 3])

  if (is.na(year)) {
      message("Could not extract year from filename: ", file_name, ". Skipping.")
      next
  }

  # Default to January if month is missing or not matched
  month_num <- if (!is.na(month_abbr) && month_abbr %in% names(month_map)) month_map[[month_abbr]] else 1

  # Create date (using 1st of the month)
  # Use dmy format for parsing "01/month/year"
  onet_release_date_val <- dmy(paste("01", month_num, year, sep = "/"))

  if (is.na(onet_release_date_val)) {
      message("Could not parse date for file: ", file_name, ". Skipping.")
      next
  }

  # Read file based on extension
  df <- NULL
  tryCatch({
    if (endsWith(tolower(file_name), ".txt")) {
      # Assuming tab-separated text files, adjust sep if needed
      df <- fread(file_path, sep = "\t", header = TRUE)
    } else if (endsWith(tolower(file_name), ".xlsx")) {
      # Assuming data is in the first sheet, adjust sheet if needed
      df <- read_excel(file_path, sheet = 1)
      setDT(df) # Convert to data.table
    } else {
      message("Unsupported file type: ", file_name, ". Skipping.")
      next
    }

    # Add the onet_release_date column
    df[, onet_release_date := onet_release_date_val]

    # Add the processed dataframe to the list
    all_tasks_list[[file_name]] <- df

  }, error = function(e) {
    message("Error processing file ", file_name, ": ", e$message)
  })
}

# Combine all dataframes into one
if (length(all_tasks_list) > 0) {
  historical_tasks <- rbindlist(all_tasks_list, use.names = TRUE, fill = TRUE)
  message("Successfully combined data from ", length(all_tasks_list), " files.")
} else {
  message("No valid task files were processed.")
  historical_tasks <- data.table() # Create empty data.table if no files processed
}

# --- OCCSOC Normalization to 2018 ---

if (nrow(historical_tasks) > 0) {
  message("Starting OCCSOC normalization to 2018 version...")

  # Define crosswalk directory
  crosswalk_dir <- file.path(ROOT_DIR, "data/onet/onet_occsoc_crosswalks")

  # Helper function to load and standardize crosswalks
  load_crosswalk <- function(from_year, to_year, dir_path) {
    # Handle the 2010 -> 2019 case which maps to SOC 2018
    to_year_file <- ifelse(to_year == 2018, 2019, to_year)
    file_name <- sprintf("onet_%d_to_%d_crosswalk.csv", from_year, to_year_file)
    file_path <- file.path(dir_path, file_name)

    if (!file.exists(file_path)) {
      stop("Crosswalk file not found: ", file_path)
    }

    message("Loading crosswalk: ", file_name)
    dt <- fread(file_path)

    # Standardize column names (adjust if actual names differ)
    from_col_name <- paste0("O*NET-SOC ", from_year, " Code")
    to_col_name <- paste0("O*NET-SOC ", to_year_file, " Code") # Use to_year_file for filename consistency

    # Check if expected columns exist
    if (!from_col_name %in% names(dt)) stop("Column '", from_col_name, "' not found in ", file_name)
    if (!to_col_name %in% names(dt)) stop("Column '", to_col_name, "' not found in ", file_name)

    # Select and rename relevant columns
    dt <- dt[, .(occsoc_from = get(from_col_name), occsoc_to = get(to_col_name))]
    dt[, occsoc_from := as.character(occsoc_from)]
    dt[, occsoc_to := as.character(occsoc_to)]
    setkey(dt, occsoc_from) # Set key for faster joins
    return(dt)
  }

  # Load required crosswalks
  xwalk_00_06 <- load_crosswalk(2000, 2006, crosswalk_dir)
  xwalk_06_09 <- load_crosswalk(2006, 2009, crosswalk_dir)
  xwalk_09_10 <- load_crosswalk(2009, 2010, crosswalk_dir)
  xwalk_10_19 <- load_crosswalk(2010, 2018, crosswalk_dir) # Maps 2010 to 2018 SOC (via 2019 file)

  # Prepare historical_tasks data
  # ** Assuming the column name is consistently 'O*NET-SOC Code' after rbindlist **
  original_occsoc_col <- "O*NET-SOC Code"
  if (!original_occsoc_col %in% names(historical_tasks)) {
      message("Available columns: ", paste(names(historical_tasks), collapse=", "))
      stop("Expected OCCSOC column '", original_occsoc_col, "' not found in historical_tasks data table after combining files.")
  }
  message("Using OCCSOC column: '", original_occsoc_col, "'")

  # Rename the identified column to 'occsoc'
  setnames(historical_tasks, old = original_occsoc_col, new = "occsoc")
  historical_tasks[, occsoc := as.character(occsoc)]

  # Initialize the target column
  historical_tasks[, occsoc_2018 := occsoc]

  # Define date thresholds
  onet_2006_cutoff <- as.Date("2006-01-01")
  onet_2010_cutoff <- as.Date("2010-01-01")
  onet_2019_cutoff <- as.Date("2019-01-01") # Start date for 2018 SOC codes

  # Apply crosswalks sequentially based on onet_release_date

  # --- Step 1: Apply 2000 -> 2006 crosswalk ---
  idx1 <- historical_tasks$onet_release_date <= onet_2006_cutoff
  if (any(idx1)) {
    message("Processing Step 1 (<= ", onet_2006_cutoff, "): Applying 00->06 to relevant rows within ", sum(idx1), " candidates.")
    historical_tasks[idx1 == TRUE][xwalk_00_06, on = .(occsoc_2018 = occsoc_from), occsoc_2018 := i.occsoc_to]
  }

  # --- Step 2: Apply 2006 -> 2009 crosswalk ---
  idx2 <- historical_tasks$onet_release_date < onet_2010_cutoff # Apply to all before 2010
   if (any(idx2)) {
    message("Processing Step 2 (< ", onet_2010_cutoff, "): Applying 06->09 to relevant rows within ", sum(idx2), " candidates.")
    historical_tasks[idx2 == TRUE][xwalk_06_09, on = .(occsoc_2018 = occsoc_from), occsoc_2018 := i.occsoc_to]
  }

  # --- Step 3: Apply 2009 -> 2010 crosswalk ---
  idx3 <- historical_tasks$onet_release_date < onet_2010_cutoff # Apply to all before 2010
   if (any(idx3)) {
    message("Processing Step 3 (< ", onet_2010_cutoff, "): Applying 09->10 to relevant rows within ", sum(idx3), " candidates.")
    historical_tasks[idx3 == TRUE][xwalk_09_10, on = .(occsoc_2018 = occsoc_from), occsoc_2018 := i.occsoc_to]
  }

  # --- Step 4: Apply 2010 -> 2019 (2018 SOC) crosswalk ---
  idx4 <- historical_tasks$onet_release_date < onet_2019_cutoff # Apply to all before 2019
   if (any(idx4)) {
    message("Processing Step 4 (< ", onet_2019_cutoff, "): Applying 10->19 to relevant rows within ", sum(idx4), " candidates.")
    historical_tasks[idx4 == TRUE][xwalk_10_19, on = .(occsoc_2018 = occsoc_from), occsoc_2018 := i.occsoc_to]
  }

  # Final check (optional): Verify codes >= 2019 were not changed
  idx5 <- historical_tasks$onet_release_date >= onet_2019_cutoff
  if (any(idx5) && !all(historical_tasks[idx5, occsoc == occsoc_2018])) {
      warning("Some OCCSOC codes for releases >= 2019-01-01 were unexpectedly modified.")
      print(historical_tasks[idx5][occsoc != occsoc_2018])
  }

  message("Finished OCCSOC normalization. Final column: 'occsoc_2018'.")

} else {
    message("Skipping OCCSOC normalization as historical_tasks data table is empty.")
}

# --- Harmonize Task Importance ---
if (nrow(historical_tasks) > 0) {
    message("Starting task importance harmonization...")

    # Initialize the new column with NA
    historical_tasks[, importance_harmonized := NA_character_]

    # --- Logic for older releases based on Relevance and Importance ---
    # ** IMPORTANT: Adjust these column names if they differ in your data **
    relevance_col <- "Percent Relevant"
    importance_col <- "Mean Importance"

    # Check if both relevance and importance columns exist
    if (relevance_col %in% names(historical_tasks) && importance_col %in% names(historical_tasks)) {
        message("Applying importance logic based on '", relevance_col, "' and '", importance_col, "'.")

        # Ensure columns are numeric, coercing if necessary
        cols_to_numeric <- c(relevance_col, importance_col)
        for (col in cols_to_numeric) {
            if (!is.numeric(historical_tasks[[col]])) {
                message("Coercing '", col, "' to numeric.")
                # Use suppressWarnings to handle potential NAs introduced by coercion
                historical_tasks[, (col) := suppressWarnings(as.numeric(get(col)))]
            }
        }

        # Apply the new logic where both relevance and importance are not NA
        # Using chained conditions for clarity
        historical_tasks[!is.na(get(relevance_col)) & !is.na(get(importance_col)),
                         importance_harmonized := fcase(
                             # Core condition
                             get(relevance_col) >= 67 & get(importance_col) >= 3.0, "Core",
                             # Supplemental conditions (combined)
                             (get(relevance_col) >= 67 & get(importance_col) < 3.0) | (get(relevance_col) < 67), "Supplemental",
                             # Default case (shouldn't be reached if logic covers all non-NA cases, but good practice)
                             default = NA_character_
                         )]

    } else {
        message("Columns '", relevance_col, "' and/or '", importance_col, "' not found. Skipping relevance/importance logic.")
    }

    # --- Logic for newer releases based on 'Task Type' (overwrites previous logic if applicable) ---
    task_type_col <- "Task Type" # Adjust if actual name differs
    if (task_type_col %in% names(historical_tasks)) {
         # Ensure task_type is character
         if (!is.character(historical_tasks[[task_type_col]])) {
             historical_tasks[, (task_type_col) := as.character(get(task_type_col))]
         }
         # Apply logic where task_type is not NA and is either "Core" or "Supplemental"
         valid_task_types <- c("Core", "Supplemental")
         # Only update importance_harmonized where task_type is valid
         historical_tasks[!is.na(get(task_type_col)) & get(task_type_col) %in% valid_task_types,
                          importance_harmonized := get(task_type_col)]
         message("Applied importance logic based on '", task_type_col, "'.")
    } else {
         message("Column '", task_type_col, "' not found. Skipping task_type logic.")
    }

    # Report counts
    importance_counts <- historical_tasks[, .N, by = importance_harmonized]
    message("Finished importance harmonization. Counts for 'importance_harmonized':")
    print(importance_counts)

    # --- Investigate NA importance_harmonized ---
    na_importance_rows <- historical_tasks[is.na(importance_harmonized)]
    num_na_rows <- nrow(na_importance_rows)

    if (num_na_rows > 0) {
        message("Investigating ", num_na_rows, " rows where 'importance_harmonized' is NA:")

        # Identify relevant columns that exist in the data
        # Add relevance/importance cols to the check if they exist
        cols_to_check <- c("occsoc", "occsoc_2018", "onet_release_date", "importance_harmonized")
        if (relevance_col %in% names(historical_tasks)) cols_to_check <- c(cols_to_check, relevance_col)
        if (importance_col %in% names(historical_tasks)) cols_to_check <- c(cols_to_check, importance_col)
        if (task_type_col %in% names(historical_tasks)) cols_to_check <- c(cols_to_check, task_type_col)

        # Print summary or head of the NA rows
        message("First few rows with NA importance_harmonized:")
        # Ensure columns exist before trying to select them
        cols_to_check_existing <- intersect(cols_to_check, names(na_importance_rows))
        print(head(na_importance_rows[, ..cols_to_check_existing]))

        # Optional: Summarize by release year or other factors
        message("Summary of NA importance_harmonized by release year:")
        print(na_importance_rows[, .N, by = .(year = year(onet_release_date))][order(year)])

        # Optional: Check unique values in original columns for these NA rows
        if (relevance_col %in% names(historical_tasks)) {
             message("Unique values of '", relevance_col, "' for NA rows:")
             print(unique(na_importance_rows[[relevance_col]]))
        }
        if (importance_col %in% names(historical_tasks)) {
             message("Unique values of '", importance_col, "' for NA rows:")
             print(unique(na_importance_rows[[importance_col]]))
        }
         if (task_type_col %in% names(historical_tasks)) {
             message("Unique values of '", task_type_col, "' for NA rows:")
             print(unique(na_importance_rows[[task_type_col]]))
        }

    } else {
        message("No rows found with NA 'importance_harmonized'.")
    }

} else {
    message("Skipping task importance harmonization as historical_tasks data table is empty.")
}

# Further processing can be done on historical_tasks dataframe
# ... (e.g., save the result)
# fwrite(historical_tasks, file.path(ROOT_DIR, "output/historical_tasks_normalized.csv"))

# Save the final processed data
output_file_path <- file.path(ROOT_DIR, "data/onet/historical_tasks_normalized.csv")
fwrite(historical_tasks, output_file_path)
message("Final processed data saved to: ", output_file_path)




