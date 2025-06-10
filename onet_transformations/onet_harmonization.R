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

# Define task statements files used for each year
onet_2003_release <- file.path(onet_task_dir, "task_statements_2003_nov.csv") # Follows SOC 2000
onet_2004_release <- file.path(onet_task_dir, "task_statements_2004_dec.csv") # Follows SOC 2000
onet_2005_release <- file.path(onet_task_dir, "task_statements_2005_dec.csv") # Follows SOC 2000
onet_2006_release <- file.path(onet_task_dir, "task_statements_2006_dec.csv") # Follows SOC 2006
onet_2007_release <- file.path(onet_task_dir, "task_statements_2007_jun.csv") # Follows SOC 2006
onet_2008_release <- file.path(onet_task_dir, "task_statements_2008_jun.csv") # Follows SOC 2006
onet_2009_release <- file.path(onet_task_dir, "task_statements_2009_jun.csv") # Follows SOC 2006
onet_2010_release <- file.path(onet_task_dir, "task_statements_2010_jul.csv") # Follows SOC 2009
onet_2011_release <- file.path(onet_task_dir, "task_statements_2011_jul.csv") # Follows SOC 2009
onet_2012_release <- file.path(onet_task_dir, "task_statements_2012_jul.csv") # Follows SOC 2010
onet_2013_release <- file.path(onet_task_dir, "task_statements_2013_jul.csv") # Follows SOC 2010
onet_2014_release <- file.path(onet_task_dir, "task_statements_2014_jul.csv") # Follows SOC 2010
onet_2015_release <- file.path(onet_task_dir, "task_statements_2015_oct.csv") # Follows SOC 2010
onet_2016_release <- file.path(onet_task_dir, "task_statements_2016_nov.csv") # Follows SOC 2010
onet_2017_release <- file.path(onet_task_dir, "task_statements_2017_oct.csv") # Follows SOC 2010
onet_2018_release <- file.path(onet_task_dir, "task_statements_2018_nov.csv") # Follows SOC 2010
onet_2019_release <- file.path(onet_task_dir, "task_statements_2019_nov.csv") # Follows SOC 2010
onet_2020_release <- file.path(onet_task_dir, "task_statements_2020_nov.csv") # Follows SOC 2019
onet_2021_release <- file.path(onet_task_dir, "task_statements_2021_nov.csv") # Follows SOC 2019
onet_2022_release <- file.path(onet_task_dir, "task_statements_2022_nov.csv") # Follows SOC 2019
onet_2023_release <- file.path(onet_task_dir, "task_statements_2023_nov.csv") # Follows SOC 2019
onet_2024_release <- file.path(onet_task_dir, "task_statements_2024_nov.csv") # Follows SOC 2019
onet_2025_release <- file.path(onet_task_dir, "task_statements_2025_feb.csv") # Follows SOC 2019


ONET_2015_aug <- file.path(onet_task_dir, "task_statements_2015_aug.csv")
ONET_2015_aug_data <- fread(ONET_2015_aug)

# Convert 2000 to 2006 SOC crosswalk to dictionary 
csv_path <- file.path(ROOT_DIR, "data/onet/onet_occsoc_crosswalks/onet_2000_to_2006_crosswalk.csv")
cw_data <- read.csv(csv_path, stringsAsFactors = FALSE)

soc_2000_to_soc_2006_dict <- new.env()
for (i in seq_len(nrow(cw_data))) {
  key <- cw_data$O.NET.SOC.2000.Code[i]
  value <- list(cw_data$O.NET.SOC.2006.Code[i], cw_data$O.NET.SOC.2006.Title[i])
  print(key)
  print(value)
  soc_2000_to_soc_2006_dict[[key]] <- value
}

# Note, to access the second value using a key, run the following:
# soc_2000_to_soc_2006_dict[["11-1011.00"]][[2]])

# Convert 2006 to 2009 SOC crosswalk to dictionary 
csv_path <- file.path(ROOT_DIR, "data/onet/onet_occsoc_crosswalks/onet_2006_to_2009_crosswalk.csv")
cw_data <- read.csv(csv_path, stringsAsFactors = FALSE)

soc_2006_to_soc_2009_dict <- new.env()
for (i in seq_len(nrow(cw_data))) {
  key <- cw_data$O.NET.SOC.2006.Code[i]
  value <- list(cw_data$O.NET.SOC.2009.Code[i], cw_data$O.NET.SOC.2009.Title[i])
  print(key)
  print(value)
  soc_2006_to_soc_2009_dict[[key]] <- value
}

# Convert 2009 to 2010 SOC crosswalk to dictionary 
csv_path <- file.path(ROOT_DIR, "data/onet/onet_occsoc_crosswalks/onet_2009_to_2010_crosswalk.csv")
cw_data <- read.csv(csv_path, stringsAsFactors = FALSE)

soc_2009_to_soc_2010_dict <- new.env()
for (i in seq_len(nrow(cw_data))) {
  key <- cw_data$O.NET.SOC.2009.Code[i]
  value <- list(cw_data$O.NET.SOC.2010.Code[i], cw_data$O.NET.SOC.2010.Title[i])
  print(key)
  print(value)
  soc_2009_to_soc_2010_dict[[key]] <- value
}

# Convert 2010 to 2019 SOC crosswalk to dictionary mapping 2019 -> 2010 and 2010 -> 2019
csv_path <- file.path(ROOT_DIR, "data/onet/onet_occsoc_crosswalks/onet_2010_to_2019_crosswalk.csv")
cw_data <- read.csv(csv_path, stringsAsFactors = FALSE)

soc_2010_to_soc_2019_dict <- new.env()
soc_2019_to_soc_2010_dict <- new.env()
for (i in seq_len(nrow(cw_data))) {
  key <- cw_data$O.NET.SOC.2010.Code[i]
  value <- list(cw_data$O.NET.SOC.2019.Code[i], cw_data$O.NET.SOC.2019.Title[i])
  soc_2010_to_soc_2019_dict[[key]] <- value

  key <- cw_data$O.NET.SOC.2019.Code[i]
  value <- list(cw_data$O.NET.SOC.2010.Code[i], cw_data$O.NET.SOC.2010.Title[i])
  soc_2019_to_soc_2010_dict[[key]] <- value
}

# Convert 2019 to 2018 SOC crosswalk to dictionary mapping 2018 -> 2019 and one for 2019 -> 2018
csv_path <- file.path(ROOT_DIR, "data/onet/onet_occsoc_crosswalks/onet_2019_to_2018_crosswalk.csv")
cw_data <- read.csv(csv_path, stringsAsFactors = FALSE)


soc_2018_to_soc_2019_dict <- new.env()
soc_2019_to_soc_2018_dict <- new.env()
for (i in seq_len(nrow(cw_data))) {
  key <- cw_data$X2018.SOC.Code[i]
  value <- list(cw_data$O.NET.SOC.2019.Code[i], cw_data$O.NET.SOC.2019.Title[i])
  soc_2018_to_soc_2019_dict[[key]] <- value

  key <- cw_data$O.NET.SOC.2019.Code[i]
  value <- list(cw_data$X2018.SOC.Code[i], cw_data$X2018.SOC.Title[i])
  soc_2019_to_soc_2018_dict[[key]] <- value
}

## Conversion to 2010 SOC codes 
# 2003 - 2005 need to be mapped first to 2006 SOC, then 2009 SOC, then 2010 SOC 
# 2006 - 2009 need to be mapped first to 2009 SOC, then 2010 SOC
# 2010 - 2011 need to be mapped to 2010 SOC 
# 2010 - 2019 can be unchanged  
# 2020 - 2025 need to be mapped to 2010 SOC
convert_to_2010_soc <- function(soc_code, year) {
  if (year >= 2003 && year <= 2005) {
    # Map from 2000 SOC to 2006 SOC
    soc_2006 <- soc_2000_to_soc_2006_dict[[soc_code]]
    if (!is.null(soc_2006)) {
      # Map from 2006 SOC to 2009 SOC
      soc_2009 <- soc_2006_to_soc_2009_dict[[soc_2006[[1]]]]
      if (!is.null(soc_2009)) {
        # Map from 2009 SOC to 2010 SOC
        soc_2010 <- soc_2009_to_soc_2010_dict[[soc_2009[[1]]]]
        return(soc_2010[[1]])
      }
    }
  } else if (year >= 2006 && year <= 2008) {
    # Map from 2006 SOC to 2009 SOC
    soc_2009 <- soc_2006_to_soc_2009_dict[[soc_code]]
    if (!is.null(soc_2009)) {
      # Map from 2009 SOC to 2010 SOC
      soc_2010 <- soc_2009_to_soc_2010_dict[[soc_2009[[1]]]]
      return(soc_2010[[1]])
    }
  } else if (year >= 2009 && year <= 2010) {
    # Directly map from 2010 SOC
    soc_2010 <- soc_2009_to_soc_2010_dict[[soc_code[[1]]]]
    return(soc_2010[[1]])
  } else if (year >= 2011 && year <= 2019) {
    # Directly map from 2010 SOC
    return(soc_code)
  } else if (year >= 2020 && year <= 2025) {
    soc_2010 <- soc_2019_to_soc_2010_dict[[soc_code[[1]]]]
    return(soc_2010[[1]])
  }

  print(paste("Unsupported year or SOC code:", year, soc_code))
  return(NA) # Return NA for unsupported years or codes
}

# Example usage:
# soc_code_2024 <- "17-3026.00" # Example SOC code from 2004
# year_2024 <- 2004 # Example year
# converted_soc_2004 <- convert_to_2010_soc(soc_code_2004, year_2004)
# print(paste("Converted SOC code for 2004:", converted_soc_2004))

## Conversion to 2018 SOC codes
# 2003 - 2005 need to be mapped first to 2006 SOC, then 2009 SOC, then 2010 SOC, then 2019 SOC, then 2018 SOC
# 2006 - 2009 need to be mapped first to 2009 SOC, then 2010 SOC, then 2019 SOC, then 2018 SOC
# 2010 - 2011 need to be mapped to 2010 SOC, then 2019 SOC, then 2018 SOC
# 2010 - 2019 need to be mapped to 2019 SOC, then 2018 SOC
# 2020 - 2025 need to be mapped to 2018 SOC
convert_to_2018_soc <- function(soc_code, year) {
  if (year >= 2003 && year <= 2005) {
    # Map from 2000 SOC to 2006 SOC
    soc_2006 <- soc_2000_to_soc_2006_dict[[soc_code]]
    if (!is.null(soc_2006)) {
      # Map from 2006 SOC to 2009 SOC
      soc_2009 <- soc_2006_to_soc_2009_dict[[soc_2006[[1]]]]
      if (!is.null(soc_2009)) {
        # Map from 2009 SOC to 2010 SOC
        soc_2010 <- soc_2009_to_soc_2010_dict[[soc_2009[[1]]]]
        if (!is.null(soc_2010)) {
          # Map from 2010 SOC to 2019 SOC
          soc_2019 <- soc_2010_to_soc_2019_dict[[soc_2010[[1]]]]
          if (!is.null(soc_2019)) {
            # Map from 2019 SOC to 2018 SOC
            soc_2018 <- soc_2019_to_soc_2018_dict[[soc_2019[[1]]]]
            return(soc_2018[[1]])
          }
        }
      }
    }
  } else if (year >= 2006 && year <= 2008) {
    # Map from 2006 SOC to 2009 SOC
    soc_2009 <- soc_2006_to_soc_2009_dict[[soc_code]]
    if (!is.null(soc_2009)) {
      # Map from 2009 SOC to 2010 SOC
      soc_2010 <- soc_2009_to_soc_2010_dict[[soc_2009[[1]]]]
      if (!is.null(soc_2010)) {
        # Map from 2010 SOC to 2019 SOC
        soc_2019 <- soc_2010_to_soc_2019_dict[[soc_2010[[1]]]]
        if (!is.null(soc_2019)) {
          # Map from 2019 SOC to 2018 SOC
          soc_2018 <- soc_2019_to_soc_2018_dict[[soc_2019[[1]]]]
          return(soc_2018)
        }
      }
    }
  } else if (year >= 2009 && year <= 2010) { 
    # Directly map from 2010 SOC to 2019 SOC, then to 2018 SOC
    soc_2010 <- soc_2009_to_soc_2010_dict[[soc_code[[1]]]]
    if (!is.null(soc_2010)) {
      soc_2019 <- soc_2010_to_soc_2019_dict[[soc_2010[[1]]]]
      if (!is.null(soc_2019)) {
        soc_2018 <- soc_2019_to_soc_2018_dict[[soc_2019[[1]]]]
        return(soc_2018)
      }
    }
  } else if (year >= 2011 && year <= 2019) {
    # Directly map from 2010 SOC to 2019 SOC, then to 2018 SOC
    soc_2019 <- soc_2010_to_soc_2019_dict[[soc_code[[1]]]]
    if (!is.null(soc_2019)) {
      soc_2018 <- soc_2019_to_soc_2018_dict[[soc_2019[[1]]]]
      return(soc_2018)
    }
  } else if (year >= 2020 && year <= 2025) {
    # Directly map from 2020 SOC to 2018 SOC
    soc_2018 <- soc_2019_to_soc_2018_dict[[soc_code[[1]]]]
    return(soc_2018)
  }
  print(paste("Unsupported year or SOC code:", year, soc_code))
  return(NA)
}

# read onet_2003_release
onet_2003_data <- fread(onet_2003_release, stringsAsFactors = FALSE)
onet_2004_data <- fread(onet_2004_release, stringsAsFactors = FALSE)
onet_2005_data <- fread(onet_2005_release, stringsAsFactors = FALSE)
onet_2006_data <- fread(onet_2006_release, stringsAsFactors = FALSE)
onet_2007_data <- fread(onet_2007_release, stringsAsFactors = FALSE)
onet_2008_data <- fread(onet_2008_release, stringsAsFactors = FALSE)
onet_2009_data <- fread(onet_2009_release, stringsAsFactors = FALSE)
onet_2010_data <- fread(onet_2010_release, stringsAsFactors = FALSE)
onet_2011_data <- fread(onet_2011_release, stringsAsFactors = FALSE)
onet_2012_data <- fread(onet_2012_release, stringsAsFactors = FALSE)
onet_2013_data <- fread(onet_2013_release, stringsAsFactors = FALSE)
onet_2014_data <- fread(onet_2014_release, stringsAsFactors = FALSE)
onet_2015_data <- fread(onet_2015_release, stringsAsFactors = FALSE)
onet_2016_data <- fread(onet_2016_release, stringsAsFactors = FALSE)
onet_2017_data <- fread(onet_2017_release, stringsAsFactors = FALSE)
onet_2018_data <- fread(onet_2018_release, stringsAsFactors = FALSE)
onet_2019_data <- fread(onet_2019_release, stringsAsFactors = FALSE)
onet_2020_data <- fread(onet_2020_release, stringsAsFactors = FALSE)
onet_2021_data <- fread(onet_2021_release, stringsAsFactors = FALSE)
onet_2022_data <- fread(onet_2022_release, stringsAsFactors = FALSE)   
onet_2023_data <- fread(onet_2023_release, stringsAsFactors = FALSE)
onet_2024_data <- fread(onet_2024_release, stringsAsFactors = FALSE)
onet_2025_data <- fread(onet_2025_release, stringsAsFactors = FALSE)

# convert SOC codes to 2010 SOC codes
onet_2003_data$`O*NET 2010 SOC Code` <- sapply(onet_2003_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2003)
onet_2004_data$`O*NET 2010 SOC Code` <- sapply(onet_2004_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2004)
onet_2005_data$`O*NET 2010 SOC Code` <- sapply(onet_2005_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2005)
onet_2006_data$`O*NET 2010 SOC Code` <- sapply(onet_2006_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2006)
onet_2007_data$`O*NET 2010 SOC Code` <- sapply(onet_2007_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2007)
onet_2008_data$`O*NET 2010 SOC Code` <- sapply(onet_2008_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2008)
onet_2009_data$`O*NET 2010 SOC Code` <- sapply(onet_2009_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2009)
onet_2010_data$`O*NET 2010 SOC Code` <- sapply(onet_2010_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2010)
onet_2011_data$`O*NET 2010 SOC Code` <- sapply(onet_2011_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2011)
onet_2012_data$`O*NET 2010 SOC Code` <- sapply(onet_2012_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2012)
onet_2013_data$`O*NET 2010 SOC Code` <- sapply(onet_2013_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2013)
onet_2014_data$`O*NET 2010 SOC Code` <- sapply(onet_2014_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2014)
onet_2015_data$`O*NET 2010 SOC Code` <- sapply(onet_2015_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2015)
onet_2016_data$`O*NET 2010 SOC Code` <- sapply(onet_2016_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2016)
onet_2017_data$`O*NET 2010 SOC Code` <- sapply(onet_2017_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2017)
onet_2018_data$`O*NET 2010 SOC Code` <- sapply(onet_2018_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2018)
onet_2019_data$`O*NET 2010 SOC Code` <- sapply(onet_2019_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2019)
onet_2020_data$`O*NET 2010 SOC Code` <- sapply(onet_2020_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2020)
onet_2021_data$`O*NET 2010 SOC Code` <- sapply(onet_2021_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2021)
onet_2022_data$`O*NET 2010 SOC Code` <- sapply(onet_2022_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2022)
onet_2023_data$`O*NET 2010 SOC Code` <- sapply(onet_2023_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2023)
onet_2024_data$`O*NET 2010 SOC Code` <- sapply(onet_2024_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2024)
onet_2025_data$`O*NET 2010 SOC Code` <- sapply(onet_2025_data$`O*NET-SOC Code`, convert_to_2010_soc, year = 2025)

# print the NA rows over the O*NET 2010 SOC Code column for each year
print("NA rows in onet_2003_data:")
print(onet_2003_data[is.na(onet_2003_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2004_data:")
print(onet_2004_data[is.na(onet_2004_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2005_data:")
print(onet_2005_data[is.na(onet_2005_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2006_data:")
print(onet_2006_data[is.na(onet_2006_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2007_data:")
print(onet_2007_data[is.na(onet_2007_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2008_data:")
print(onet_2008_data[is.na(onet_2008_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2009_data:")                                     
print(onet_2009_data[is.na(onet_2009_data$`O*NET 2010 SOC Code`), ]) ### problems with 3000 obs
print("NA rows in onet_2010_data:")
print(onet_2010_data[is.na(onet_2010_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2011_data:")
print(onet_2011_data[is.na(onet_2011_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2012_data:")
print(onet_2012_data[is.na(onet_2012_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2013_data:")
print(onet_2013_data[is.na(onet_2013_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2014_data:")
print(onet_2014_data[is.na(onet_2014_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2015_data:")
print(onet_2015_data[is.na(onet_2015_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2016_data:")
print(onet_2016_data[is.na(onet_2016_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2017_data:")
print(onet_2017_data[is.na(onet_2017_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2018_data:")
print(onet_2018_data[is.na(onet_2018_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2019_data:")
print(onet_2019_data[is.na(onet_2019_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2020_data:")
print(onet_2020_data[is.na(onet_2020_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2021_data:")
print(onet_2021_data[is.na(onet_2021_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2022_data:")
print(onet_2022_data[is.na(onet_2022_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2023_data:")
print(onet_2023_data[is.na(onet_2023_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2024_data:")
print(onet_2024_data[is.na(onet_2024_data$`O*NET 2010 SOC Code`), ])
print("NA rows in onet_2025_data:")
print(onet_2025_data[is.na(onet_2025_data$`O*NET 2010 SOC Code`), ])
