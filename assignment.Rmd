---
title: "Practical Machine Learning Course Project"
author: "Umut Yener Kara"
date: "8/18/2019"
output: html_document
  
---



## Summary

Human Activity Recognition is a young research field driven by recent developments such as e-health, quantified self movement and ubuquity of consumer devices with accelerometers. Classifying and modeling different physical activity types using machine learning on accelerometer data is a common technique in the field. In this project we attempt to develop a machine learning classifier on "Weight Lifting Exercises Dataset" which contains data from accelerometers on different body parts and five different classes (one for correct execution, four for common erronous executions). After preprocessing and wrangling steps, we built a Random Forests model which gave highly accurate results with 99.3% accuracy and 0.07% out-of-sample error. Our model scored 100% on quiz dataset. 

The dataset comes from this paper: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

## Packages and Options

We set options(scipen) to 999 to prevent R from using scientific notation for very large or very small numbers. We utilize parallel processing since we use Random Forests and it's a computationally intensive algorithm.

```{r setup,include=TRUE, results="hide", message=FALSE, warning=FALSE}

library(caret)
library(tidyverse)
library(parallel)
library(doParallel)
options(scipen = 999)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
```
 
## Data Preprocessing and Cleaning

Our dataset comes in two parts, pml_testing and pml_training. Despite what its name suggest pml_testing is not really for testing models but designed for quiz with only 20 observations and outcome variable "classe" missing. So we will work on pml_training dataset. The argument "na.strings = c("", "NA")" inside the read.csv function is important in that we tell R to read blank values as "NAs" which will help us to clean dataset from blank and NA columns. 

```{r loading}
setwd("c:/Users/umuty/OneDrive/R/Coursera/Course 8/Assignment")

data <- read.csv("pml-training.csv", na.strings = c("", "NA"))
quiz <- read.csv("pml-testing.csv", na.strings = c("", "NA"))

```
A quick look inside the dataset show us that there are columns with many NA values. We can check actual NA numbers with "colsums" function.  
```{r str and colsums}

str(data[1:20])
colSums(is.na(data))

```

Almost all rows of some columns are NAs. We can remove them with dplyr "select_if" function on the condition that they have higher NA percentages than %90. In this way we are left with 60 columns. Since we are interested in only accelerometer data for model building, first seven columns are also irrevelant as they contain subject aliases, timestamps etc. In this way final form of the dataset will include 53 columns and 19622 rows. 

```{r select_if}

data <- data %>% select_if(~!sum(is.na(.)) /nrow(data) > 0.90)
data  <- data[,-c(1:7)]
dim(data)

```
## Checking for low or zero variance

As variables with zero variance and near zero variance has limited information value and can negatively impact some machine learning models, it's a good practice to detect and remove them.

```{r variance, warning=FALSE}

NZV<- nearZeroVar(data, saveMetrics = TRUE)
sum(NZV$zeroVar == T)
sum(NZV$nzv == T)

```

There are no variables with zero variation or near zero variation so we can go ahead and split the dataset for prediction.

## Data Splitting

We follow the 60/40 rule for splitting our dataset to training and testing partitions. 

```{r splitting}
set.seed(1234)
inTrain <- createDataPartition(data$classe, p = 0.60)[[1]]
training <- data[inTrain,]
testing <- data[-inTrain,]

```

## Random Forest Model Training

Random Forest is a really popular algorithm for machine building tasks due to its high accuracy so we will use it for this project. We will first set the seed for reproducibility as random forests use random sampling. Then we set parameters of our model with 3 Folds of Cross Validation and parallel processing to increase speed of computations. After this we train our model with train function from caret package. Later, we will look at importance metrics to interpret the model so it's set to TRUE. 

```{r random forests, cache=TRUE}
set.seed(1234)
trcontrol <- trainControl( method = "cv",
                           number = 3,
                           allowParallel = TRUE)

rfmod <- train(classe~., method = "rf", data = training, trControl = trcontrol, importance = TRUE)
rfmod

```
Our model shows very high accuracy on training set. We also see that it automatically selected the 27 variable model as it had the highest accuracy among others. We can also visualize how accuracy values change when number of variables change as well. 

```{r random forests plot}

plot(rfmod)

```
## Prediction on Testing Dataset

Now that we trained our model, we can test it on our testing set and see how accurate it is with a confusion matrix.

```{r random forests prediction}

predictrf <- predict(rfmod, testing)

confusionMatrix(testing$classe, predictrf)



```
As we can see, our random forests model is very accurate(99.3%) with only 0.7% out-of-sample error or generalization error.

## Prediction on Quiz dataset

Applying our model to quiz dataset resulted in a perfect score (the results are suppressed as recommended). 

```{r random forests quiz prediction, include = TRUE, results="hide"}

predictquiz <- predict(rfmod, quiz)


```

## Interpreting the Model 

Even our model is very accurate it's not really interpretable, we don't know what criteria and weights it uses for classification. A simple way to do this is looking at importance metrics. 

```{r random forests importance}

varImp(rfmod, scale=TRUE)

```
In addition to giving us an idea about how our model works, importance metrics also shows what types of motions are associated with activity classes (e.g. roll_belt variable was most important for classifying class E which is "throwing the hips to the front").

## Conclusion 

In this project we built a Random Forest classifier with 99.3% accuracy and 0.07% out-of-sample error which scored on 100% on quiz dataset. We also tried to interpret the model with importance metrics. 

