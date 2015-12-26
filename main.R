# PROJECT LOAD
print('Start Project')

## set seed
set.seed(514514)

## load library
library(caret)
library(dplyr)
library(randomForest)
library(gbm)
library(e1071)

## read data
tn = read.csv('pml-training.csv')
tt = read.csv('pml-testing.csv')



# PREPROCESSING 
print('Start Preprocessing')

summary(tn) 
## find out there are many NAs: 19216 / 19622
## find out different columns share the same number of NAs

## randomly check: find out all NAs appear at same instance
all(is.na(subset(tn, is.na(stddev_yaw_forearm), select=var_pitch_forearm)))

## remove those columns
na_idx_tn = sapply(tn, function(x) any(is.na(x)))
na_idx_tt = sapply(tt, function(x) any(is.na(x)))
na_idx = na_idx_tn | na_idx_tt
tn = tn[, !na_idx]
tt = tt[, !na_idx]

## because of ordered label, we should remove time-related feature
tn = select(tn, -c(X, cvtd_timestamp, raw_timestamp_part_1, raw_timestamp_part_2))
tt = select(tt, -c(X, cvtd_timestamp, raw_timestamp_part_1, raw_timestamp_part_2))

## find out the relationship between user_name and classe
tmp = group_by(tn, user_name, classe)
tmp = summarize(tmp, sum = n())
pic = ggplot(tmp, aes=c(user_name, sum, color=classe))
## find out there is no obvious relationship

## divide x, y
tn_x = select(tn, -classe)
tn_y = tn$classe
tt_x = select(tt, -problem_id)

## expand user_name to dummy variable
## since tt_x has only one value in new_window, must combine first then run dummyVars
x = rbind(tn_x, tt_x)
dmy = dummyVars(~ ., data=x)
x = data.frame(predict(dmy, newdata=x))
tn_x = head(x, dim(tn_x)[1])
tt_x = tail(x, 20)

## create sub-train and validation: 0.6 * 19622 = 11773
num_tn = 19622
num_sn = 11773
idx = sample(1:num_tn, num_tn)
tn_x = tn_x[idx, ]
tn_y = tn_y[idx]
sn_x = tn_x[1:num_sn, ]
sn_y = tn_y[1:num_sn]
va_x = tn_x[(num_sn+1):num_tn, ]
va_y = tn_y[(num_sn+1):num_tn]



# MACHINE LEARNING - default randomForest (tried gbm, e1071, etc.)
print('Start Machine Learning')

## results for subtrain and validation
rf_model = randomForest(x=sn_x, y=sn_y)
sn_p = predict(rf_model, sn_x)
acc = sum(sn_p == sn_y) / length(sn_y)
print(c('sn predict sn: ', acc))
va_p = predict(rf_model, va_x)
acc = sum(va_p == va_y) / length(va_y)
print(c('sn predict va: ', acc))

## results for train and test
rf_model = randomForest(x=tn_x, y=tn_y)
tn_p = predict(rf_model, tn_x)
acc = sum(tn_p == tn_y) / length(tn_y)
print(c('tn predict tn: ', acc))
tt_p = predict(rf_model, tt_x)

## Note: I omit cross-validation process in this project.
## Based on my past contest experience, 0.99 random validation 
## accuracy is stable, and I am certain C.V. is unneccessay here.

## write out results to file
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(tt_p)
