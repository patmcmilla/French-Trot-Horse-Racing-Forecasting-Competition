library(tidyverse)
library(arrow)
library(reticulate)
library(caret)

## Relative path
 # Windows
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
Training.data <- read_parquet("../Data/Training_Set.parquet")
Testing.data <- read_parquet("../Data/Testing_Set.parquet")

# Source python functions
source_python('SiameseFNN.py')
pd <- import("pandas")

## Fit model 
race_data_py <- Training.data %>%
  mutate(across(everything(), ~as.numeric(.x))) %>%
  pd$DataFrame()

## Train batch model 
BatchModel <- Batch_siamese_FNN(race_data_py)



## Assess model 
Test_data_py <- Testing.data %>%
  mutate(across(everything(), ~as.numeric(.x))) %>%
  pd$DataFrame()

## Predict on Testing set
Predictions <- validate_siamese_FNN(Test_data_py)

Results <- Testing.data %>%
  select(RaceID, HorseID, BeatenMargin) %>%
  mutate(Pred_BeatenMargin = as.vector(Predictions),
         winprobability = 1/(1 + exp(Pred_BeatenMargin))) %>%
  group_by(RaceID) %>%
  mutate(total_prob = sum(winprobability)) %>%
  ungroup() %>%
  mutate(winprobability = winprobability/total_prob) %>%
  select(RaceID, HorseID, BeatenMargin, winprobability) 
  
write.csv(Results, 'Forecasts.csv')
