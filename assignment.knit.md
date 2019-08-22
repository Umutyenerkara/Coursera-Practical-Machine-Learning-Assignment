---
title: "Untitled"
author: "Umut Yener Kara"
date: "8/18/2019"
output: html_document
---



## Summary

Human Activity Recognition is a new research area driven by recent developments such as e-health, quantified self movement and ubuquity of consumer devices with accelerometers. Classifying and modeling different physical activity types using machine learning on accelerometer data is a common technique in the field. In this project we attempt to develop a machine learning classifier on "Weight Lifting Exercises Dataset" which contains data from accelerometers on different body parts and five different classes (one for correct execution, four for common erronous executions). After preprocessing and wrangling steps, we built a Random Forests model which gave highly accurate results with 99.2 accuracy and 0.08 out-of-sample error. Our model scored %100 on quiz dataset. 

The dataset comes from this paper: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
 
## Data Preprocessing and Cleaning

Our dataset comes in two parts, pml_testing and pml_training. Despite what its name suggest pml_testing is not really for testing models but designed for quiz with only 20 observations and "classe" variable missing. So we will work on pml_training dataset. The argument "na.strings = c("", "NA")" inside the read.csv function is important in that we tell R to read blank values as "NAs" which will help us to clean dataset from blank and NA columns. 


```r
setwd("c:/Users/umuty/OneDrive/R/Coursera/Course 8/Assignment")

data <- read.csv("pml-training.csv", na.strings = c("", "NA"))
quiz <- read.csv("pml-testing.csv", na.strings = c("", "NA"))
```
A quick look inside the dataset show us that there are columns with many NA values. We can check actual NA numbers with "colsums" function.  

```r
str(data[1:20])
```

```
## 'data.frame':	19622 obs. of  20 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt  : Factor w/ 396 levels "-0.016850","-0.021024",..: NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt : Factor w/ 316 levels "-0.021887","-0.060755",..: NA NA NA NA NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt   : Factor w/ 1 level "#DIV/0!": NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt  : Factor w/ 394 levels "-0.003095","-0.010002",..: NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1: Factor w/ 337 levels "-0.005928","-0.005960",..: NA NA NA NA NA NA NA NA NA NA ...
##  $ skewness_yaw_belt   : Factor w/ 1 level "#DIV/0!": NA NA NA NA NA NA NA NA NA NA ...
##  $ max_roll_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt      : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt        : Factor w/ 67 levels "-0.1","-0.2",..: NA NA NA NA NA NA NA NA NA NA ...
```

```r
colSums(is.na(data))
```

```
##                        X                user_name     raw_timestamp_part_1 
##                        0                        0                        0 
##     raw_timestamp_part_2           cvtd_timestamp               new_window 
##                        0                        0                        0 
##               num_window                roll_belt               pitch_belt 
##                        0                        0                        0 
##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
##                        0                        0                    19216 
##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
##                    19216                    19216                    19216 
##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
##                    19216                    19216                    19216 
##           max_picth_belt             max_yaw_belt            min_roll_belt 
##                    19216                    19216                    19216 
##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
##                    19216                    19216                    19216 
##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
##                    19216                    19216                    19216 
##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
##                    19216                    19216                    19216 
##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
##                    19216                    19216                    19216 
##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
##                    19216                    19216                    19216 
##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
##                        0                        0                        0 
##             accel_belt_x             accel_belt_y             accel_belt_z 
##                        0                        0                        0 
##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
##                        0                        0                        0 
##                 roll_arm                pitch_arm                  yaw_arm 
##                        0                        0                        0 
##          total_accel_arm            var_accel_arm             avg_roll_arm 
##                        0                    19216                    19216 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                    19216                    19216                    19216 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                    19216                    19216                    19216 
##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
##                    19216                    19216                        0 
##              gyros_arm_y              gyros_arm_z              accel_arm_x 
##                        0                        0                        0 
##              accel_arm_y              accel_arm_z             magnet_arm_x 
##                        0                        0                        0 
##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
##                        0                        0                    19216 
##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
##                    19216                    19216                    19216 
##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
##                    19216                    19216                    19216 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                    19216                    19216                    19216 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                    19216                    19216                    19216 
##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
##                    19216                    19216                        0 
##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
##                        0                        0                    19216 
##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
##                    19216                    19216                    19216 
##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
##                    19216                    19216                    19216 
##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
##                    19216                    19216                    19216 
##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
##                    19216                    19216                    19216 
## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
##                    19216                    19216                        0 
##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
##                    19216                    19216                    19216 
##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
##                    19216                    19216                    19216 
##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
##                    19216                    19216                    19216 
##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
##                    19216                        0                        0 
##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
##                        0                        0                        0 
##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
##                        0                        0                        0 
##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
##                        0                        0                        0 
##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
##                        0                    19216                    19216 
##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
##                    19216                    19216                    19216 
##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
##                    19216                    19216                    19216 
##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
##                    19216                    19216                    19216 
##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                    19216                    19216                    19216 
##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
##                    19216                        0                    19216 
##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
##                    19216                    19216                    19216 
##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
##                    19216                    19216                    19216 
##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
##                    19216                    19216                    19216 
##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
##                        0                        0                        0 
##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
##                        0                        0                        0 
##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
##                        0                        0                        0 
##                   classe 
##                        0
```

Almost all rows of some columns are NAs. We can remove them with dplyr "select_if" function on the condition that they have higher NA percentages than %90. In this way we are left with 60 columns. Since we are interested in only accelerometer data for model building, first seven columns are also irrevelant as they contain subject aliases, timestamps etc. In this way final form of the dataset will include 53 columns and 19622 rows. 


```r
data <- data %>% select_if(~!sum(is.na(.)) /nrow(data) > 0.90)
data  <- data[,-c(1:7)]
dim(data)
```

```
## [1] 19622    53
```
## Checking for low or zero variance

As variables with zero variance and near zero variance has limited information value and can negatively impact some machine learning models, it's a good practice to detect and remove them.


```r
NZV<- nearZeroVar(data, saveMetrics = TRUE)
sum(NZV$zeroVar == T)
```

```
## [1] 0
```

```r
sum(NZV$nzv == T)
```

```
## [1] 0
```

There are no variables with zero variation or near zero variation so we can go ahead and split the dataset for prediction.

## Data Splitting

We follow the 60/40 rule for splitting our dataset to training and testing partitions. 


```r
set.seed(1234)
inTrain <- createDataPartition(data$classe, p = 0.60)[[1]]
training <- data[inTrain,]
testing <- data[-inTrain,]
```

## Prediction with Random Forests

Random Forest is a really popular algorithm for machine building tasks due to its high accuracy so it's a good starting point. We will first set the seed for reproducibility as random forests use random sampling. Then we set parameters of our model with 3 Folds of Cross Validation and parallel processing to increase speed of computations. After this we train our model with train function from caret package.  


```r
set.seed(1234)
trcontrol <- trainControl( method = "cv",
                           number = 3,
                           allowParallel = TRUE)

rfmod <- train(classe~., method = "rf", data = training, trControl = trcontrol, importance = TRUE)
rfmod
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 7851, 7851, 7850 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9833562  0.9789433
##   27    0.9835260  0.9791597
##   52    0.9752043  0.9686305
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```
Our model shows very high accuracy on training set. We also see that it automatically selected the 27 variable model as it had the highest accuracy among others. We can also visualize how accuracy values change when number of variables change as well. 


```r
plot(rfmod)
```

<img src="assignment_files/figure-html/random forests plot-1.png" width="672" />
## Prediction 

Now that we trained our model, we can test it on our testing set and see how accurate it is with a confusion matrix.


```r
predictrf <- predict(rfmod, testing)

confusionMatrix(testing$classe, predictrf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    0    1    0    0
##          B    9 1506    2    1    0
##          C    0   11 1347   10    0
##          D    0    0    9 1275    2
##          E    0    1    4    6 1431
## 
## Overall Statistics
##                                                
##                Accuracy : 0.9929               
##                  95% CI : (0.9907, 0.9946)     
##     No Information Rate : 0.2855               
##     P-Value [Acc > NIR] : < 0.00000000000000022
##                                                
##                   Kappa : 0.991                
##                                                
##  Mcnemar's Test P-Value : NA                   
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9960   0.9921   0.9883   0.9868   0.9986
## Specificity            0.9998   0.9981   0.9968   0.9983   0.9983
## Pos Pred Value         0.9996   0.9921   0.9846   0.9914   0.9924
## Neg Pred Value         0.9984   0.9981   0.9975   0.9974   0.9997
## Prevalence             0.2855   0.1935   0.1737   0.1647   0.1826
## Detection Rate         0.2843   0.1919   0.1717   0.1625   0.1824
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9979   0.9951   0.9925   0.9926   0.9984
```
As we can see, our random forests model is very accurate(99.3%) with only 0.7% out-of-sample error or generalization error.

## Predicting the Quiz dataset

Applying our model to quiz dataset resulted in a perfect score (the results are suppressed as recommended). 



## Interpreting the Model 

Even our model is very accurate it's not really interpretable, we don't know what criteria and weights it uses for classification. A simple way to do this is looking at importance metrics. 


```r
varImp(rfmod, scale=TRUE)
```

```
## rf variable importance
## 
##   variables are sorted by maximum importance across the classes
##   only 20 most important variables shown (out of 52)
## 
##                        A     B     C     D      E
## roll_belt         73.178 80.09 80.12 68.56 100.00
## pitch_forearm     51.196 62.04 85.53 49.14  49.76
## pitch_belt        21.899 82.35 57.58 38.25  31.03
## magnet_dumbbell_y 60.814 62.37 78.76 56.06  50.30
## magnet_dumbbell_z 71.025 52.77 65.69 48.55  46.31
## yaw_belt          61.909 50.28 54.26 58.30  42.48
## roll_forearm      46.129 35.66 41.37 28.47  31.96
## accel_forearm_x   17.023 32.97 29.19 43.41  32.34
## accel_dumbbell_y  28.044 27.43 37.72 25.34  28.15
## gyros_dumbbell_y  30.867 21.48 36.83 19.45  17.26
## yaw_arm           35.041 22.85 21.20 28.21  14.75
## gyros_belt_z      19.328 25.82 23.97 17.29  30.99
## accel_dumbbell_z  20.133 25.58 18.48 23.65  29.18
## magnet_belt_z     16.423 29.07 20.30 28.35  24.01
## roll_dumbbell     21.266 27.81 15.94 19.84  27.24
## gyros_arm_y       21.301 26.81 19.21 23.64  14.64
## magnet_forearm_z  26.410 25.93 20.48 18.96  23.30
## yaw_dumbbell      10.014 25.27 14.20 15.04  18.61
## magnet_arm_z      14.515 23.80 19.35 13.68  13.25
## gyros_forearm_y    7.053 23.25 19.41 16.70  13.21
```
In addition to giving us an idea about how our model works, importance metrics also shows what types of motions are associated with which activity classes (e.g. roll_belt variable was most important for classifying class E which is "throwing the hips to the front").

## Conclusion 

In this project we built a Random Forests classifier with 99.3% accuracy and 0.07% out-of-sample error which scored on 100% on quiz dataset. We also tried to interpret the model with importance metrics. 





