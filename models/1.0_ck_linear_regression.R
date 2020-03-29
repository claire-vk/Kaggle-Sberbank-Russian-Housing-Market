library(data.table)
library(tidyverse)
library(lubridate)
library(scales)
library(corrplot)
library(DT)
library(car)
library(ggplot2)


setwd('/Users/clairevignon/DataScience/NYC_DSA/project3_ML_kaggle')
train = read.csv("train_cleaned.csv", stringsAsFactors = FALSE)
# macro = read.csv("macro.csv", stringsAsFactors = FALSE)
test = read.csv("test_cleaned.csv", stringsAsFactors = FALSE)

# using log
train$price_doc = log(train$price_doc+1)

##### Top Variables from Jack's lasso #####
top_35 = c("full_sq", "num_room", "kitch_sq", "state", "sub_area", "indust_part",
           "build_count_frame", "green_zone_km", "sadovoe_km", "railroad_km", "office_km", "theater_km",
           "catering_km", "trc_sqm_500", "church_count_500", "leisure_count_500", "trc_sqm_1000",
           "cafe_count_1000_price_high",  "mosque_count_1500", "prom_part_2000", "cafe_sum_2000_min_price_avg",
           "prom_part_3000", "office_sqm_5000", "cafe_sum_5000_min_price_avg", "cafe_count_5000_price_high",
           "mosque_count_5000", "sport_count_5000", 'price_doc')

# splitting top 35 to make it easier to look at corplot
top_35_1 = c("full_sq", "num_room", "kitch_sq", "build_count_frame", 'price_doc') # only numeric
top_35_2 = c("green_zone_km", "sadovoe_km", "railroad_km", "office_km", "theater_km",
             "catering_km", "trc_sqm_500", 'price_doc')
top_35_3 = c("trc_sqm_500", "church_count_500", "leisure_count_500", "trc_sqm_1000",
             "cafe_count_1000_price_high",  "mosque_count_1500", "prom_part_2000", "cafe_sum_2000_min_price_avg",
             "prom_part_3000", "office_sqm_5000", "cafe_sum_5000_min_price_avg", "cafe_count_5000_price_high",
             "mosque_count_5000", "sport_count_5000", 'price_doc')

# look at multi-collinearity between the variables
corrplot(cor(train[, top_35_1], use="complete.obs"))
# full_sq & num_room are correlated. We will drop num_room and keep full_sq (which is
# highly correlated with price)

corrplot(cor(train[, top_35_2], use="complete.obs"))
# office_km, theater_km seem somewhat correlated but there is no real good reason
# to think so so will keep both

# catering_km and office km are correlated which can make sense so will keep
# office_km only. None seem highly correlated with price

corrplot(cor(train[, top_35_3], use="complete.obs"))
# variables of the same family (e.g. trc_sqm_500 and trc_sqm_1000) are correlated 
# which make sense. We won't consider them in our linear model for now

top_35_reduced = c("id", "full_sq", "kitch_sq", "state", "sub_area", 'price_doc', "green_zone_km", 
                   "sadovoe_km", "railroad_km", "office_km", "theater_km")
top_35_reduced_noprice = c("id", "full_sq", "kitch_sq", "state", "sub_area", "green_zone_km", 
                           "sadovoe_km", "railroad_km", "office_km", "theater_km")

train_top35_reduced = subset(train, select= top_35_reduced)
test_top35_reduced = subset(test, select= top_35_reduced_noprice)

str(train_top35_reduced)
str(test_top35_reduced)

# convert state and sub_area variables to factor
train_top35_reduced$state = factor(train_top35_reduced$state)
train_top35_reduced$sub_area = factor(train_top35_reduced$sub_area)

test_top35_reduced$state = factor(test_top35_reduced$state)
test_top35_reduced$sub_area = factor(test_top35_reduced$sub_area)


# Impute NA with random for factor variables
sum(is.na(train_top35_reduced$state)) #13559 NAs which is more than 30% of the
# dataset so we will impute with random method

library(Hmisc)
train_top35_reduced$state = impute(train_top35_reduced$state, "random")
test_top35_reduced$state = impute(test_top35_reduced$state, "random")

sum(is.na(train_top35_reduced)) # no more NAs
sum(is.na(test_top35_reduced))

# # check relationship between variables (matrix scatter plot)
# plot(train_top35_reduced, pch=16, col="blue", main="Matrix Scatterplot")

# 1- Is there a relationship between price and house characterictics?
# fitting a multiple regression model of price onto top variables 
model = lm(price_doc ~ . - id, data = train_top35_reduced)

# # 2- Is the relationship linear
# # checking if we violate any assumptions
# plot(model) # none of the assumptions are violated

# 3- How strong is the relationship?
summary(model)
vif(model) # sadovoe_km  VIF is higher than 10 so removing it from model

model_1 = update(model, ~.-sadovoe_km)
summary(model_1)
vif(model_1)

# 4- How accurately can we predict the relationship?
prediction = predict(model_1,test_top35_reduced) 
prediction = exp(prediction)-1
prediction = data.frame(id=test$id, price_doc=prediction)
# prediction = apply(prediction,2,function(x){x[x<0] = min(x[x>0]);x})
write.csv(prediction, "prediction.csv", row.names=F)


##### METHOD 2 - Linear Regression, different features, median imputation grouped by sub_area #####
train_reduced = read.csv("train_reduced.csv", stringsAsFactors = FALSE)
test_reduced = read.csv("test_reduced.csv", stringsAsFactors = FALSE)

train_reduced = subset(train_reduced, select = -c(X))
test_reduced = subset(test_reduced, select = -c(X))

train_reduced$price_doc = log(train_reduced$price_doc+1)

model_2 = lm(price_doc ~ . - id - usdrub - cpi - mortgage_rate - micex - timestamp, data = train_reduced)

summary(model_2)
vif(model_2)

# VIF is higher than 5 for preschool_km, public_healthcare_km, swim_pool_km, big_church_km ,
# public_transport_station_km, kindergarten_km, hospice_morgue_km so removing them
model_2a = update(model_2, ~ . -preschool_km - public_healthcare_km - swim_pool_km - public_transport_station_km - kindergarten_km - hospice_morgue_km - big_church_km )
summary(model_2a)
vif(model_2a)

prediction_2 = predict(model_2a,test_reduced) 
prediction_2 = exp(prediction_2)-1
prediction_2 = data.frame(id=test$id, price_doc=prediction_2)
write.csv(prediction_2, "prediction_2.csv", row.names=F)

##Improved score by 78 from method 2##

##### METHOD 3 - Linear Regression, same as METHOD 2 + take macro into account in model #####
train_reduced_2 = read.csv("train_reduced.csv", stringsAsFactors = FALSE)
test_reduced_2 = read.csv("test_reduced.csv", stringsAsFactors = FALSE)

train_reduced_2 = subset(train_reduced_2, select = -c(X))
test_reduced_2 = subset(test_reduced_2, select = -c(X))

train_reduced_2$price_doc = log(train_reduced_2$price_doc+1)

model_3 = lm(price_doc ~ . - id - timestamp, data = train_reduced_2)

summary(model_3)
vif(model_3)

# VIF is higher than 5 for preschool_km, public_healthcare_km, swim_pool_km, big_church_km,
# public_transport_station_km, kindergarten_km, hospice_morgue_km so removing them
model_3a = update(model_3, ~ . -preschool_km - public_healthcare_km - swim_pool_km - public_transport_station_km - kindergarten_km - hospice_morgue_km - big_church_km)
summary(model_3a)
vif(model_3a)

prediction_3 = predict(model_3a,test_reduced_2) 
prediction_3 = exp(prediction_3)-1
prediction_3 = data.frame(id=test$id, price_doc=prediction_3)
write.csv(prediction_3, "prediction_3.csv", row.names=F)

## adding macro did not help with the score...###

##### METHOD 4 - same as model 3 with less macro data #####
train_reduced_3 = read.csv("train_reduced.csv", stringsAsFactors = FALSE)
test_reduced_3 = read.csv("test_reduced.csv", stringsAsFactors = FALSE)

train_reduced_3 = subset(train_reduced_3, select = -c(X))
test_reduced_3 = subset(test_reduced_3, select = -c(X))

train_reduced_3$price_doc = log(train_reduced_3$price_doc+1)

model_4 = lm(price_doc ~ . - id - timestamp - usdrub - micex, data = train_reduced_3)

summary(model_4)
vif(model_4)

# VIF is higher than 5 for preschool_km, public_healthcare_km, swim_pool_km, big_church_km,
# public_transport_station_km, kindergarten_km, hospice_morgue_km so removing them
model_4a = update(model_4, ~ . -preschool_km - public_healthcare_km - swim_pool_km - public_transport_station_km - kindergarten_km - hospice_morgue_km - big_church_km)
summary(model_4a)
vif(model_4a)

prediction_4 = predict(model_4a,test_reduced_3) 
prediction_4 = exp(prediction_4)-1
prediction_4 = data.frame(id=test$id, price_doc=prediction_4)
write.csv(prediction_4, "prediction_4.csv", row.names=F)

##### METHOD 5 #####
## tweaking the macro does not help. Doing some feature engineering (in python)
# Feature engineering revealed new important variables. We'll use them.

train_2 = read.csv("train_cleaned_2.csv", stringsAsFactors = FALSE)
test_2 = read.csv("test_cleaned_2.csv", stringsAsFactors = FALSE)

train_2$price_doc = log(train_2$price_doc+1)

top_feat = c("full_sq", "life_sq", "build_year", "year_month", "floor", "additional_education_km", "catering_km", 
            "big_church_km", "public_healthcare_km", "railroad_km", "preschool_km", "industrial_km", 
            "radiation_km", "big_road1_km", "public_transport_station_km", "cemetery_km",
            "swim_pool_km", "green_zone_km", "big_road2_km", "hospice_morgue_km", "fitness_km",
            "kindergarten_km", "max_floor", "extra_sq", "green_part_1000", "ratio_life_sq_full_sq",
            "age_house", "ratio_kitch_sq_full_sq", "ratio_preschool", "ratio_floor_max_floor",
            "sales_per_month", "ratio_kitch_sq_life_sq", "kitch_sq",
            "preschool_quota", "month", "sub_area", "area_m", "state", "pop_density_raion",
            "full_all", "ratio_school", "ekder_all", "num_room", "school_quota",
            "retire_proportion", "raion_popul", "year", 'price_doc')

# turn categorical variables into factor
cat_feat = c("build_year", "year_month", "floor", "max_floor", 
             "month", "sub_area", "state", "year")

# scale and center numeric vectors
num_feat = c("full_sq", "life_sq", "additional_education_km", "catering_km",
             "big_church_km", "public_healthcare_km", "railroad_km", "preschool_km", "industrial_km", 
             "radiation_km", "big_road1_km", "public_transport_station_km", "cemetery_km",
             "swim_pool_km", "green_zone_km", "big_road2_km", "hospice_morgue_km", "fitness_km",
             "kindergarten_km","extra_sq", "green_part_1000", "ratio_life_sq_full_sq",
             "age_house", "ratio_kitch_sq_full_sq", "ratio_preschool", "ratio_floor_max_floor",
             "sales_per_month", "ratio_kitch_sq_life_sq", "kitch_sq", "preschool_quota",
             "area_m", "pop_density_raion",
             "full_all", "ratio_school", "ekder_all", "num_room", "school_quota",
             "retire_proportion", "raion_popul")


train_lnr_cat = subset(train_2, select= cat_feat)
test_lnr_cat = subset(test_2, select= cat_feat)
ifelse(n <- sapply(train_lnr_cat, function(x) length(levels(x))) == 1, "DROP", "NODROP") # check if factors have more than 1 level

train_lnr_num = subset(train_2, select= num_feat)
test_lnr_num = subset(test_2, select= num_feat)

f_factor=function(x){
  x = as.factor(x) #turn into factor type
}

f_numeric=function(x){
  x = as.numeric(x) #turn into numeric type
}

train_lnr_cat=data.frame(apply(train_lnr_cat,2,f_factor))
test_lnr_cat=data.frame(apply(test_lnr_cat,2,f_factor))
sapply(train_lnr_cat,class) # checking that it worked

train_lnr_num=data.frame(apply(train_lnr_num,2,f_numeric))
test_lnr_num=data.frame(apply(test_lnr_num,2,f_numeric))
sapply(train_lnr_num,class) # checking that it worked


# scaling the numeric values
train_lnr_num = scale(train_lnr_num)
test_lnr_num = scale(test_lnr_num)

train_lnr = cbind(train_lnr_num, train_lnr_cat)
test_lnr = cbind(test_lnr_num, test_lnr_cat)

train_lnr = cbind(train_lnr, train_2$price_doc) # adding the price col
train_lnr = as.data.frame(train_lnr)  # turning into dataframe

names(train_lnr)[names(train_lnr) == 'train_2$price_doc'] = 'price_doc' # changing col name

test_lnr = as.data.frame(test_lnr) # turning test into df

which(sapply(train_lnr, function(x) length(unique(x))<2)) # check if factors have less than 2 levels
# ratio preschool is problematic so we will remove it


# restart with less variables

train_2 = read.csv("train_cleaned_2.csv", stringsAsFactors = FALSE)
test_2 = read.csv("test_cleaned_2.csv", stringsAsFactors = FALSE)

train_reduced_2 = subset(train_2, select = -c(X))
test_reduced_2 = subset(test_2, select = -c(X))

train_2$price_doc = log(train_2$price_doc+1)

top_feat = c("full_sq", "life_sq", "additional_education_km", "catering_km",
             "big_church_km", "public_healthcare_km", "railroad_km", "preschool_km", "industrial_km", 
             "radiation_km", "big_road1_km", "public_transport_station_km", "cemetery_km",
             "swim_pool_km", "green_zone_km", "big_road2_km", "hospice_morgue_km", "fitness_km",
             "kindergarten_km", "green_part_1000", "ratio_life_sq_full_sq",
             "age_house", "ratio_kitch_sq_full_sq", "ratio_floor_max_floor",
             "sales_per_month", "ratio_kitch_sq_life_sq", "kitch_sq", "preschool_quota",
             "area_m", "pop_density_raion",
             "full_all", "ratio_school", "ekder_all", "num_room", "school_quota",
             "retire_proportion", "raion_popul", "price_doc")

top_feat_test = c("full_sq", "life_sq", "additional_education_km", "catering_km",
             "big_church_km", "public_healthcare_km", "railroad_km", "preschool_km", "industrial_km", 
             "radiation_km", "big_road1_km", "public_transport_station_km", "cemetery_km",
             "swim_pool_km", "green_zone_km", "big_road2_km", "hospice_morgue_km", "fitness_km",
             "kindergarten_km","extra_sq", "green_part_1000", "ratio_life_sq_full_sq",
             "age_house", "ratio_kitch_sq_full_sq", "ratio_floor_max_floor",
             "sales_per_month", "ratio_kitch_sq_life_sq", "kitch_sq", "preschool_quota",
             "area_m", "pop_density_raion",
             "full_all", "ratio_school", "ekder_all", "num_room", "school_quota",
             "retire_proportion", "raion_popul")

train_lnr = subset(train_2, select= top_feat)
test_lnr = subset(test_2, select= top_feat_test)

# train_lnr$state = factor(train_lnr$state)
# train_lnr$sub_area = factor(train_lnr$sub_area)
# 
# test_lnr$state = factor(test_lnr$state)
# test_lnr$sub_area = factor(test_lnr$sub_area)


# scaling the numeric values
numeric_col = train_lnr[, names(train_lnr) != c("price_doc")]   
train_lnr[, names(train_lnr) != c("price_doc")]  = scale(numeric_col)

numeric_col = test_lnr[, names(test_lnr) != c("price_doc")]   
test_lnr[, names(test_lnr) != c("price_doc")]  = scale(numeric_col)

model_5 = lm(price_doc ~ . , data = train_lnr)

summary(model_5)
prediction_5 = predict(model_5, test_lnr) 
prediction_5 = exp(prediction_5)-1
prediction_5 = data.frame(id=test_2$id, price_doc=prediction_5)
write.csv(prediction_5, "prediction_5.csv", row.names=F)


##### METHOD 6 - same as method 2 with feature engineering #####
train_reduced_2 = read.csv("train_cleaned_2.csv", stringsAsFactors = FALSE)
test_reduced_2 = read.csv("test_cleaned_2.csv", stringsAsFactors = FALSE)

train_reduced_2 = subset(train_reduced_2, select = -c(X))
test_reduced_2 = subset(test_reduced_2, select = -c(X))

top_feat = c("state", "catering_km",
             "big_church_km", "public_healthcare_km", "railroad_km", "preschool_km", "industrial_km", 
             "radiation_km", "big_road1_km", "public_transport_station_km", "cemetery_km",
             "swim_pool_km", "green_zone_km", "big_road2_km", "hospice_morgue_km", "fitness_km",
             "kindergarten_km", "green_part_1000", "ratio_life_sq_full_sq",
             "age_house", "ratio_kitch_sq_full_sq", "ratio_floor_max_floor",
             "sales_per_month", "ratio_kitch_sq_life_sq", "num_room", "price_doc")
top_feat_test = c("state", "catering_km",
                  "big_church_km", "public_healthcare_km", "railroad_km", "preschool_km", "industrial_km", 
                  "radiation_km", "big_road1_km", "public_transport_station_km", "cemetery_km",
                  "swim_pool_km", "green_zone_km", "big_road2_km", "hospice_morgue_km", "fitness_km",
                  "kindergarten_km", "green_part_1000", "ratio_life_sq_full_sq",
                  "age_house", "ratio_kitch_sq_full_sq", "ratio_floor_max_floor",
                  "sales_per_month", "ratio_kitch_sq_life_sq", "num_room")



train_reduced_2 = subset(train_reduced_2, select = top_feat)
test_reduced_2 = subset(test_reduced_2, select = top_feat_test)

train_reduced_2$price_doc = log(train_reduced_2$price_doc+1)

train_reduced_2$state = factor(train_reduced_2$state)
train_reduced_2$sub_area = factor(train_reduced_2$sub_area)

test_reduced_2$state = factor(test_reduced_2$state)
test_reduced_2$sub_area = factor(test_reduced_2$sub_area)

model_2 = lm(price_doc ~ . , data = train_reduced_2)

library(car)
summary(model_2)
vif(model_2)

prediction_6 = predict(model_2,test_reduced_2) 
prediction_6 = exp(prediction_6)-1
prediction_6 = data.frame(id=test$id, price_doc=prediction_6)
write.csv(prediction_6, "prediction_6.csv", row.names=F)


##### Elastic Net #####


