library(tidyverse)
library(arrow)
library(FactoMineR)
library(factoextra)
library(zoo)
library(mice)
library(reticulate)
library(caret)

## Relative path
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Load dataset
Master.dat <- read_parquet("../Data/trots_2013-2022.parquet") %>%
  as_tibble() %>%
  mutate(RaceDate = as.Date(RaceStartTime)) %>%
  filter(RaceDate > "2020-06-01")

## Clean data 
clean.fun <- function(.dat){
  
  Full.dat <- .dat %>% as_tibble() %>%
    mutate(across(c(everything(), -FoalingDate, -RaceDate, -RaceStartTime), ~case_when(. == "" ~ NA,
                                                                            . == " " ~ NA,
                                                                            TRUE ~ .))) %>%
    mutate(RaceDate = as.Date(RaceStartTime),
           RaceTime = strftime(RaceStartTime, format="%H:%M"),
           FoalingDate = as.Date(FoalingDate),
           HorseAge = as.double(difftime(RaceDate, FoalingDate, units = c("weeks")))) %>%
    mutate(BeatenMargin=na_if(BeatenMargin,999)) %>%
    group_by(RaceID) %>%
    mutate(min_age=min(HorseAge, na.rm=T), 
           max_age=max(HorseAge, na.rm=T),
           BeatenMargin = ifelse(is.na(BeatenMargin), max(BeatenMargin, na.rm = TRUE), BeatenMargin)) %>%
    ungroup() %>%
    mutate(StartingLine=as.factor(StartingLine),
           NoFrontCover=as.factor(NoFrontCover))
  
  # obtain the prefix (e.g., 'NW$101'):
  splits = str_split(Full.dat$ClassRestriction, " ")
  splits_prefix=sapply(splits, function(y) str_split(y[1], "\\$"))
  
  # define function to extract prefix only (e.g., NW)
  extract_prefix <- function(x) {
    prefix <- ifelse(length(x)>=2, x[1], "")
    return(prefix)
  }
  
  # define function to extract number code only (e.g. 101)
  extract_code <- function(x) {
    code <- ifelse(length(x)>=2, x[2], "")
    return(code)
  }
  
  # divide ClassRestriction into prefix, code, suffix, and binary for contains 'CND'
  Full.dat$prefix <- sapply(splits_prefix, extract_prefix)
  Full.dat$code   <- sapply(splits_prefix, extract_code)
  Full.dat$code   <- ifelse(Full.dat$code=="",0,Full.dat$code)
  Full.dat$cnd    <- ifelse(grepl("CND",Full.dat$ClassRestriction),1,0)
  Full.dat$suffix <- gsub(".*\\s(\\w{2})$", "\\1", Full.dat$ClassRestriction)
  Full.dat$suffix[!grepl(".*\\s(\\w{2})$", Full.dat$ClassRestriction)]<-""
  
  return(Full.dat)
}

Master.data <- clean.fun(Master.dat)

## Create features - May take several minutes
FeatureCreation.fun <- function(Train.dat){
  Train.dat$HandicapType[is.na(Train.dat$HandicapType)]<-"None"
  
  # clean data further
  non_finish_codes <- c("BS ", "DQ ", "FL ", "NP ", "PU ", "UN ", "UR ", "WC ")
  Train.dat$FinishPosition[Train.dat$FinishPosition %in% non_finish_codes]<-NA
  Train.dat.2 <- Train.dat %>%
    mutate(DNF=is.na(FinishPosition),
           FinishPosition=as.numeric(as.character(FinishPosition)),
           FinishPosition=ifelse(is.na(FinishPosition), max(FinishPosition, na.rm = TRUE)+1, FinishPosition),
           WideOffDummy=ifelse(WideOffRail==-9,1,0),
           WideOffRail=ifelse(WideOffRail==-9,0,WideOffRail),
           FrontShoes=as.factor(FrontShoes),
           HindShoes=as.factor(HindShoes),
    )
  
  ###
  pca_dat <- Train.dat.2 %>%
    select(Barrier, BeatenMargin,
           Disqualified, Distance,
           FinishPosition, DNF,
           HandicapDistance, HandicapType,
           PIRPosition, PriceSP,
           RaceOverallTime, RacingSubType,
           Saddlecloth,StartType,
           StartingLine, Surface,
           NoFrontCover, WideOffRail,
           WideOffDummy, WeightCarried,
           Prizemoney, WetnessScale)
  result <- FAMD(pca_dat, graph = FALSE)
  
  factors <- result$ind$coord
  
  Train.dat.2 <- cbind(Train.dat.2, factors) 
  
  ###--- create rolling averages of factors for horse, jockey, trainer, dam
  # Define a function to compute the rolling average
  rolling_avg <- function(x, n = 5) {
    x_avg = rollapplyr(x, width = n, FUN = function(i) mean(i[-length(i)], na.rm = TRUE), fill = NA, partial=TRUE)
    return(x_avg)
  }
  
  # Define a function to compute the rolling sd
  rolling_sd <- function(x, n = 5) {
    x_sd = rollapplyr(x, width = n, FUN = function(i) sd(i[-length(i)], na.rm = TRUE), fill = NA, partial=TRUE)
    return(x_sd)
  }
  
  Train.dat.3 <- Train.dat.2 %>%
    group_by(HorseID) %>%
    arrange(HorseID, RaceStartTime) %>%
    mutate(h_dim1_Avg = rolling_avg(Dim.1, n = 5),
           h_dim1_SD = rolling_sd(Dim.1, n = 5),
           h_dim2_Avg = rolling_avg(Dim.2, n = 5),
           h_dim2_SD = rolling_sd(Dim.2, n = 5),
           h_dim3_Avg = rolling_avg(Dim.3, n = 5),
           h_dim3_SD = rolling_sd(Dim.3, n = 5),
           h_dim4_Avg = rolling_avg(Dim.4, n = 5),
           h_dim4_SD = rolling_sd(Dim.4, n = 5),
           h_dim5_Avg = rolling_avg(Dim.5, n = 5),
           h_dim5_SD = rolling_sd(Dim.5, n = 5)) %>%
    ungroup() %>%
    arrange(RaceStartTime) %>%
    group_by(DamID) %>%
    arrange(DamID, RaceStartTime) %>%
    mutate(d_dim1_Avg = rolling_avg(Dim.1, n = 5),
           d_dim1_SD = rolling_sd(Dim.1, n = 5),
           d_dim2_Avg = rolling_avg(Dim.2, n = 5),
           d_dim2_SD = rolling_sd(Dim.2, n = 5),
           d_dim3_Avg = rolling_avg(Dim.3, n = 5),
           d_dim3_SD = rolling_sd(Dim.3, n = 5),
           d_dim4_Avg = rolling_avg(Dim.4, n = 5),
           d_dim4_SD = rolling_sd(Dim.4, n = 5),
           d_dim5_Avg = rolling_avg(Dim.5, n = 5),
           d_dim5_SD = rolling_sd(Dim.5, n = 5)) %>%
    ungroup() %>%
    arrange(RaceStartTime) %>%
    group_by(SireID) %>%
    arrange(SireID, RaceStartTime) %>%
    mutate(s_dim1_Avg = rolling_avg(Dim.1, n = 5),
           s_dim1_SD = rolling_sd(Dim.1, n = 5),
           s_dim2_Avg = rolling_avg(Dim.2, n = 5),
           s_dim2_SD = rolling_sd(Dim.2, n = 5),
           s_dim3_Avg = rolling_avg(Dim.3, n = 5),
           s_dim3_SD = rolling_sd(Dim.3, n = 5),
           s_dim4_Avg = rolling_avg(Dim.4, n = 5),
           s_dim4_SD = rolling_sd(Dim.4, n = 5),
           s_dim5_Avg = rolling_avg(Dim.5, n = 5),
           s_dim5_SD = rolling_sd(Dim.5, n = 5)) %>%
    ungroup() %>%
    arrange(RaceStartTime) %>%
    group_by(JockeyID) %>%
    arrange(JockeyID, RaceStartTime) %>%
    mutate(j_dim1_Avg = rolling_avg(Dim.1, n = 5),
           j_dim1_SD = rolling_sd(Dim.1, n = 5),
           j_dim2_Avg = rolling_avg(Dim.2, n = 5),
           j_dim2_SD = rolling_sd(Dim.2, n = 5),
           j_dim3_Avg = rolling_avg(Dim.3, n = 5),
           j_dim3_SD = rolling_sd(Dim.3, n = 5),
           j_dim4_Avg = rolling_avg(Dim.4, n = 5),
           j_dim4_SD = rolling_sd(Dim.4, n = 5),
           j_dim5_Avg = rolling_avg(Dim.5, n = 5),
           j_dim5_SD = rolling_sd(Dim.5, n = 5)) %>%
    ungroup() %>%
    arrange(RaceStartTime) %>%
    group_by(TrainerID) %>%
    arrange(TrainerID, RaceStartTime) %>%
    mutate(t_dim1_Avg = rolling_avg(Dim.1, n = 5),
           t_dim1_SD = rolling_sd(Dim.1, n = 5),
           t_dim2_Avg = rolling_avg(Dim.2, n = 5),
           t_dim2_SD = rolling_sd(Dim.2, n = 5),
           t_dim3_Avg = rolling_avg(Dim.3, n = 5),
           t_dim3_SD = rolling_sd(Dim.3, n = 5),
           t_dim4_Avg = rolling_avg(Dim.4, n = 5),
           t_dim4_SD = rolling_sd(Dim.4, n = 5),
           t_dim5_Avg = rolling_avg(Dim.5, n = 5),
           t_dim5_SD = rolling_sd(Dim.5, n = 5)) %>%
    ungroup() %>%
    arrange(RaceStartTime)

  return(Train.dat.3)
}

MasterFeatures <- FeatureCreation.fun(Master.data)


## Now split into training and testing sets?
Master.onehot <- MasterFeatures %>%
  mutate(code = as.numeric(code),
         across(where(is.character), ~as.factor(.x)))

CleanFeatures.train.onehot <- Master.onehot %>%
  filter(RaceDate >= "2021-11-01") %>%
  select(-c(Barrier, CourseIndicator, SexRestriction, NoFrontCover, Saddlecloth, Disqualified, PIRPosition, Prizemoney, ClassRestriction, AgeRestriction, FoalingCountry, FrontShoes, HindShoes, GoingAbbrev, GoingID, RaceGroup, FinishPosition, PriceSP, PositionInRunning, WideOffRail),
         -contains('Date'),
         -contains('Time')) 

CleanFeatures.test.onehot <- Master.onehot %>%
  filter(RaceDate < "2021-11-01",
         RaceDate > "2021-01-01") %>% # Reduced training set to help with training times.
  select(-c(Barrier, CourseIndicator, SexRestriction, NoFrontCover, Saddlecloth, Disqualified, PIRPosition, Prizemoney, ClassRestriction, AgeRestriction, FoalingCountry, FrontShoes, HindShoes, GoingAbbrev, GoingID, RaceGroup, FinishPosition, PriceSP, PositionInRunning, WideOffRail),
         -contains('Date'),
         -contains('Time')) 

## Onehot encode the data
dmy.test <- dummyVars(" ~ .", data = CleanFeatures.test.onehot)
CleanFeatures.test.onehot <- data.frame(predict(dmy.test, newdata = CleanFeatures.test.onehot))

dmy.train <- dummyVars(" ~ .", data = CleanFeatures.train.onehot)
CleanFeatures.train.onehot <- data.frame(predict(dmy.train, newdata = CleanFeatures.train.onehot))

## Finalize datasets
CleanFeatures.test.onehot <- CleanFeatures.test.onehot %>%
  mutate(across(everything(), ~case_when(.x == "NAN" ~ NA,
                                         TRUE ~ .x))) %>%
  select(-c(Dim.1:Dim.5)) %>% # remove the PCs used to make the rolling averages
  mutate(across(h_dim1_Avg:t_dim5_SD, ~case_when(is.na(.x) ~ mean(.x, na.rm = T),
                                                 TRUE ~ .x)))

CleanFeatures.train.onehot <- CleanFeatures.train.onehot %>%
  mutate(across(everything(), ~case_when(.x == "NAN" ~ NA,
                                         TRUE ~ .x))) %>%
  select(-c(Dim.1:Dim.5)) %>% # remove the PCs used to make the rolling averages
  mutate(across(h_dim1_Avg:t_dim5_SD, ~case_when(is.na(.x) ~ mean(.x, na.rm = T), ## impute missing valus with column mean
                                                 TRUE ~ .x)))


write_parquet(CleanFeatures.test.onehot, "../Data/Testing_Set.parquet")
write_parquet(CleanFeatures.train.onehot, "../Data/Training_Set.parquet")



