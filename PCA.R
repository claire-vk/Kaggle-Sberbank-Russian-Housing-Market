setwd('/Users/clairevignon/DataScience/NYC_DSA/project3_ML_kaggle')
train = read.csv("train_cleaned.csv", stringsAsFactors = FALSE)
macro = read.csv("macro.csv", stringsAsFactors = FALSE)
test = read.csv("test_cleaned.csv", stringsAsFactors = FALSE)

# using log
train$price_doc = log(train$price_doc+1)

# grab only a few numeric variables from train dataset to test PCA
sapply(train, class)
pca_features = c("full_sq", "life_sq", "kitch_sq",
             "num_room", "big_church_km", "preschool_km", 
             "cafe_avg_price_500", "big_road2_km", "green_zone_km",
             "kindergarten_km", "catering_km", "big_road1_km", 
             "public_healthcare_km", "hospice_morgue_km", "swim_pool_km", 
             "green_part_1000", "railroad_km", "industrial_km", 
             "cemetery_km", "fitness_km", "theater_km", "radiation_km",
             'area_m', 'children_preschool', 'preschool_quota', 
             'preschool_education_centers_raion', 'children_school', 
             'school_quota', 'school_education_centers_raion', 
             'school_education_centers_top_20_raion', 
             'university_top_20_raion', 'additional_education_raion', 
             'additional_education_km', 'university_km',
             'nuclear_reactor_km', 'thermal_power_plant_km', 
             'power_transmission_line_km', 'incineration_km',
             'water_treatment_km', 
             'railroad_station_walk_km', 'railroad_station_walk_min',
             'railroad_station_avto_km', 'railroad_station_avto_min', 
             'public_transport_station_km', 'public_transport_station_min_walk', 
             'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km','bulvar_ring_km',
             "price_doc", "id")
# not all the numeric data is here, e.g. cafes, etc.

train_pca = subset(train, select = pca_features)

# not working because there are items that are highly correlated. 
# checking it with correlation matrix
cm = cor(train_pca, method = "pearson")

dim(cm) # 52 x 52 matrix

for(i in which(cm==1)){ # looking for highly correlated items (could change to 0.99)
  if(i%%52 != (i%/%52+1)){
    print(i)
  }
}

# look for rows and columns of highly correlated items 
# i.e. 2281, 2332, 2704
2281 %/% 52 # find row
2281 %% 52 # find column
cm[43:44, 45:46] # correlation between public_transport_station_min_walk and public_transport_station_km 

2332 %/% 52 # find row
2332 %% 52 # find column
cm[44:45, 44:45] # correlation between public_transport_station_min_walk and public_transport_station_km 

## update the features by removing one of the highly correlated features
# remove 'public_transport_station_min_walk' and 'railroad_station_walk_min'

pca_features = c("full_sq", "life_sq", "kitch_sq",
                 "num_room", "big_church_km", "preschool_km", 
                 "cafe_avg_price_500", "big_road2_km", "green_zone_km",
                 "kindergarten_km", "catering_km", "big_road1_km", 
                 "public_healthcare_km", "hospice_morgue_km", "swim_pool_km", 
                 "green_part_1000", "railroad_km", "industrial_km", 
                 "cemetery_km", "fitness_km", "theater_km", "radiation_km",
                 'area_m', 'children_preschool', 'preschool_quota', 
                 'preschool_education_centers_raion', 'children_school', 
                 'school_quota', 'school_education_centers_raion', 
                 'school_education_centers_top_20_raion', 
                 'university_top_20_raion', 'additional_education_raion', 
                 'additional_education_km', 'university_km',
                 'nuclear_reactor_km', 'thermal_power_plant_km', 
                 'power_transmission_line_km', 'incineration_km',
                 'water_treatment_km', 
                 'railroad_station_walk_km',
                 'railroad_station_avto_km', 'railroad_station_avto_min', 
                 'public_transport_station_km', 
                 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km','bulvar_ring_km',
                 "price_doc", "id")
pca_features_test = c("full_sq", "life_sq", "kitch_sq",
                 "num_room", "big_church_km", "preschool_km", 
                 "cafe_avg_price_500", "big_road2_km", "green_zone_km",
                 "kindergarten_km", "catering_km", "big_road1_km", 
                 "public_healthcare_km", "hospice_morgue_km", "swim_pool_km", 
                 "green_part_1000", "railroad_km", "industrial_km", 
                 "cemetery_km", "fitness_km", "theater_km", "radiation_km",
                 'area_m', 'children_preschool', 'preschool_quota', 
                 'preschool_education_centers_raion', 'children_school', 
                 'school_quota', 'school_education_centers_raion', 
                 'school_education_centers_top_20_raion', 
                 'university_top_20_raion', 'additional_education_raion', 
                 'additional_education_km', 'university_km',
                 'nuclear_reactor_km', 'thermal_power_plant_km', 
                 'power_transmission_line_km', 'incineration_km',
                 'water_treatment_km', 
                 'railroad_station_walk_km',
                 'railroad_station_avto_km', 'railroad_station_avto_min', 
                 'public_transport_station_km', 
                 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km','bulvar_ring_km', "id")
train_pca = subset(train, select = pca_features)
test_pca = subset(test, select = pca_features_test)


library(psych)

# Impute NAs?
sum(is.na(train_pca)) # 52734 missing values

# impute NAs by median 
f=function(x){
  x = as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
train_pca=data.frame(apply(train_pca,2,f))
test_pca=data.frame(apply(test_pca,2,f))

# remove price and ID from dataframe
# train_pca_noprice = train_pca[ , names(train_pca) != "price_doc"]
train_pca_noprice = train_pca[ , !names(train_pca) %in% c("price_doc", "id")]

fa.parallel(train_pca_noprice, 
            fa = "pc", 
            n.iter = 100)
abline(h = 1)
# graph shows that 10 PCs is about the right number

# apply to test dataset
library(caret)

ctrl_1 <- trainControl(preProcOptions = list(pcaComp = 10)) # how many components we 


md_1 = train(price_doc ~ . - id , data = train_pca,
           method = 'lm', # change if plug it in another model
           preProc = 'pca',
           trControl = ctrl_1)

# md = train(price_doc ~ ., data = train_pca_2,
#            method = 'lm',
#            preProc = c('medianImpute', 'pca'), # can do computation automatically here
#            trControl = ctrl) # specify the kind of CV we want to use


prediction_pca_1 = predict.train(md_1, newdata=test_pca)
prediction_pca_1 = exp(prediction_pca_1)-1
prediction_pca_1 = data.frame(id=test$id, price_doc=prediction_pca_1)
write.csv(prediction_pca_1, "prediction_pca_2.csv", row.names=F)


##### TRY WITH LESS FEATURES #####

pca_features_2 = c("full_sq", "life_sq", "kitch_sq",
                 "num_room", "big_church_km", "preschool_km", 
                 "cafe_avg_price_500", "big_road2_km", "green_zone_km",
                 "kindergarten_km", "catering_km", "big_road1_km", 
                 "public_healthcare_km", "hospice_morgue_km", "swim_pool_km", 
                 "green_part_1000", "railroad_km", "industrial_km", 
                 "cemetery_km", "fitness_km", "theater_km", "radiation_km",
                 'area_m', 'children_preschool', 'preschool_quota', 
                 'preschool_education_centers_raion', 'children_school', 
                 'school_quota', 'school_education_centers_raion', 
                 'school_education_centers_top_20_raion', 
                 'university_top_20_raion', 'price_doc','id') 

pca_features_2_test = c("full_sq", "life_sq", "kitch_sq",
                   "num_room", "big_church_km", "preschool_km", 
                   "cafe_avg_price_500", "big_road2_km", "green_zone_km",
                   "kindergarten_km", "catering_km", "big_road1_km", 
                   "public_healthcare_km", "hospice_morgue_km", "swim_pool_km", 
                   "green_part_1000", "railroad_km", "industrial_km", 
                   "cemetery_km", "fitness_km", "theater_km", "radiation_km",
                   'area_m', 'children_preschool', 'preschool_quota', 
                   'preschool_education_centers_raion', 'children_school', 
                   'school_quota', 'school_education_centers_raion', 
                   'school_education_centers_top_20_raion', 
                   'university_top_20_raion','id') 

# not all the numeric data is here, e.g. cafes, etc.

library(psych)

train_pca_2 = subset(train, select = pca_features_2)
test_pca_2 = subset(test, select = pca_features_2_test)

##### Need to impute NAs?
sum(is.na(train_pca_2)) # 52734 missing values

# impute NAs by median 
f=function(x){
  x = as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
train_pca_2=data.frame(apply(train_pca_2,2,f))
test_pca_2=data.frame(apply(test_pca_2,2,f))

# remove price from dataframe
# train_pca_noprice_2 = train_pca_2[ , names(train_pca_2) != "price_doc"]
train_pca_noprice_2 = train_pca_2[ , !names(train_pca_2) %in% c("price_doc", "id")]
# test_pca_2 = test_pca_2[ , names(test_pca_2) != "id"]

fa.parallel(train_pca_noprice_2, 
            fa = "pc", 
            n.iter = 100)
abline(h = 1)

# graph shows that 6 PCs is about the right number

### NOT NEEDED BECAUSE WE ARE USING CARET ###
# performing PCA

# pc_train_pca_noprice_2 = principal(train_pca_noprice_2, 
#                                  nfactors = 6, 
#                                  rotate = "none")
# pc_train_pca_noprice_2
# 
# # transform the PCs back by multiply SS loading scores with PC values
# a = as.data.frame(pc_train_pca_noprice_2$scores)
# b = as.data.frame(t(pc_train_pca_noprice_2$values[1:6])) # [1:6 because we have 6 PCs]
# 
# colnames(b) = colnames(a)
# 
# c  = sapply(1:ncol(a), function(i){a[,i]*b[,i]})
# 
# data = data.frame(c, train$price_doc)



# apply to test dataset
library(caret)

ctrl <- trainControl(preProcOptions = list(pcaComp = 6))


md = train(price_doc ~ . - id, data = train_pca_2,
           method = 'lm',
           preProc = 'pca',
           trControl = ctrl)

# md = train(price_doc ~ ., data = train_pca_2,
#            method = 'lm',
#            preProc = c('medianImpute', 'pca'), # can do computation automatically here
#            trControl = ctrl) # specify the kind of CV we want to use


prediction_pca = predict.train(md, newdata=test_pca_2)
prediction_pca = exp(prediction_pca)-1
prediction_pca = data.frame(id=test$id, price_doc=prediction_pca)
write.csv(prediction_pca, "prediction_pca_1.csv", row.names=F)


