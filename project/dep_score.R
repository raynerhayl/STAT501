
rm(list = ls())

library(lubridate)

data <- read.csv("data.csv", header = TRUE, stringsAsFactors = FALSE)

#date format 23/10/2012 DD/MM/YYYY
#create dates variable for your column that contains dates 
dates <- data$AdmitDate

#get the month of the date, create new column called month
data$month <- (month(dates, label=TRUE))

#new column called season
season <- data$season

#divide seasons by month
winter <- c("Jun","Jul", "Aug")
spring <- c("Sep", "Oct","Nov")
summer <- c("Dec", "Jan", "Feb")
autumn <- c("Mar","Apr", "May")


#reformat to work with grep
winter <- paste(winter,collapse="|")
spring <- paste(spring,collapse="|")
summer<- paste(summer,collapse="|")
autumn <- paste(autumn,collapse="|")

#populate the season field based on matching seasons in the month field
season[grep(winter, data$month)] <- "winter"
season[grep(spring, data$month)] <- "spring"
season[grep(summer, data$month)] <- "summer"
season[grep(autumn, data$month)] <- "autumn"

# Add seasons
data <- cbind(data, season)

# Match the domicile description provided to a dep score
# domicile_desc - column to match codes to
# multiple_match - function to handle multiple location prefixes eg. median, mean default is to take the first code
# dep_fn - filemane for the deprivation index csv
# domicile_codes_fn - filename for the domicile codes csv
#
# returns coloumn of deprivation scores

get.dep <- function(domicile_desc, multiple_match = function(x) x[1], dep_fn='dep_au.csv', domicile_codes_fn='domicile_code.csv'){
  dep <- read.csv2(dep_fn, header=TRUE, sep=',', stringsAsFactors = F)
  domicile_codes <- read.csv2(domicile_codes_fn, header=TRUE, sep=',', stringsAsFactors = F)
  
  dep_score <- domicile_desc
  # Get post code
  names <- sapply(dep_score, function(x)trimws(strsplit(x, "-")[[1]][2]))
  dep_score <- sapply(dep_score, function(x) strsplit(x, "-")[[1]][1])
  # Get AU-13 code
  dep_score <- sapply(dep_score, function(x) domicile_codes$AU.13[which(x == domicile_codes$dom)])
  # Get DEP score
  dep_score <- sapply(dep_score, function(x) dep$CAU_average_NZDep2013[which(x == dep$CAU_2013)])
  
  empty <- which(sapply(dep_score, function(x) length(x) == 0 || is.na(x)))
  # Name of entries with no AU-13 Code
  empty.codes <- unique(names[empty])
  
  # Take score based on prefix of domicile name
  dep_score[empty] <- sapply(empty, function(x) multiple_match(unlist(dep$CAU_average_NZDep2013[which(startsWith(dep$CAU_name_2013, names[x]))])))
  # Name of entries with no matching domicile name or AU-13
  empty <- which(sapply(dep_score, function(x) length(x) == 0 || is.na(x)))
  empty.names <- unique(names[empty])
  
  dep_score[empty] <- NA
  
  return(unlist(dep_score))
}

# Add deprivation score
dep_score <- get.dep(data$DomicileDesc)
data <- cbind(data, dep_score)

# Remove those with domicile codes which are not matched
data <- data[which(!is.na(data$dep_score)), ]
