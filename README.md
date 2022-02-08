# STUDENT DROPOUT PREDICTION

## Introduction

Teachers and school administrators have striven to reduce dropout for quite some time, but it continues to persist in schools as a problem through the present day. Dropping out of colleges is considered not just a serious educational problem but also a severe social problem, especially in recent decades when technology and societal developments have rendered more and more people without at least a college degree less likely to find a job. It is critical to understand the causes and recognize the signs, in this project I will aim to accurately predict the probability of a student dropping out of a college. I will measure prediction accuracy and analyze aspects of the students’ data to recognize the most important factors leading to high dropout rates. Machine learning techniques can effectively facilitate the determination of at-risk students and timely planning for interventions. I will implement several classification algorithms to find the best prediction model.

## Data set and features

The data was gathered from New Jersey City University undergraduate students from 2012 to 2017. The data set contains three types of data:
Student Static Data: Static data include demographic and educational background information about each student in the cohort; these data do not change over time. These data are collected through a CSV file, uploaded once for each student. This file contains one record per student, and each student appears in only one static file, corresponding to the year in which he/she first enrolled.
Student Progress Data: Progress/General data reflect your students’ academic progression and outcomes over time. These data are CSV files to be uploaded, reflecting each student’s activity for each term in each academic year. This file contains one record per student. Multiple cohorts are included in each term file.
Student Financial Aid Data: Financial Aid Data was collected for each student for each academic year, and it is stored in different columns for different years. It contains Financial Aid and other related information such as scholarships, loans, gross income.

## Exploratory Data Analysis - EDA

Basic descriptive statistics of the variables in the Student Static Data

```
summary(StudentStaticData)
##    StudentID         Cohort            CohortTerm     Campus       
##  Min.   : 20932   Length:13261       Min.   :1.000   Mode:logical  
##  1st Qu.:305254   Class :character   1st Qu.:1.000   NA's:13261    
##  Median :321478   Mode  :character   Median :1.000                 
##  Mean   :316151                      Mean   :1.391                 
##  3rd Qu.:343511                      3rd Qu.:1.000                 
##  Max.   :359783                      Max.   :3.000                                                                                 
##    Address1           Address2             City              State          
##  Length:13261       Length:13261       Length:13261       Length:13261      
##  Class :character   Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character   Mode  :character                                                                           
##       Zip        RegistrationDate       Gender        BirthYear   
##  Min.   :  747   Min.   :20110111   Min.   :1.000   Min.   :1945  
##  1st Qu.: 7060   1st Qu.:20120710   1st Qu.:1.000   1st Qu.:1986  
##  Median : 7304   Median :20140121   Median :2.000   Median :1992  
##  Mean   : 7790   Mean   :20136109   Mean   :1.596   Mean   :1989  
##  3rd Qu.: 7307   3rd Qu.:20150624   3rd Qu.:2.000   3rd Qu.:1995  
##  Max.   :98118   Max.   :20160912   Max.   :2.000   Max.   :2000  
##  NA's   :134                                                      
##    BirthMonth        Hispanic       AmericanIndian         Asian         
##  Min.   : 1.000   Min.   :-1.0000   Min.   :-1.00000   Min.   :-1.00000  
##  1st Qu.: 4.000   1st Qu.: 0.0000   1st Qu.: 0.00000   1st Qu.: 0.00000  
##  Median : 7.000   Median : 0.0000   Median : 0.00000   Median : 0.00000  
##  Mean   : 6.581   Mean   : 0.2568   Mean   :-0.06742   Mean   : 0.01848  
##  3rd Qu.:10.000   3rd Qu.: 1.0000   3rd Qu.: 0.00000   3rd Qu.: 0.00000  
##  Max.   :12.000   Max.   : 1.0000   Max.   : 1.00000   Max.   : 1.00000                                                                    
##      Black         NativeHawaiian         White        TwoOrMoreRace     
##  Min.   :-1.0000   Min.   :-1.00000   Min.   :-1.000   Min.   :-1.00000  
##  1st Qu.: 0.0000   1st Qu.: 0.00000   1st Qu.: 0.000   1st Qu.: 0.00000  
##  Median : 0.0000   Median : 0.00000   Median : 0.000   Median : 0.00000  
##  Mean   : 0.1447   Mean   :-0.06757   Mean   : 0.183   Mean   :-0.05181  
##  3rd Qu.: 0.0000   3rd Qu.: 0.00000   3rd Qu.: 1.000   3rd Qu.: 0.00000  
##  Max.   : 1.0000   Max.   : 1.00000   Max.   : 1.000   Max.   : 1.00000                                                                       
##      HSDip            HSDipYr         HSGPAUnwtd         HSGPAWtd     FirstGen 
##  Min.   :-1.0000   Min.   :  -1.0   Min.   :-1.0000   Min.   :-1   Min.   :-1  
##  1st Qu.: 1.0000   1st Qu.:  -1.0   1st Qu.:-1.0000   1st Qu.:-1   1st Qu.:-1  
##  Median : 1.0000   Median :  -1.0   Median :-1.0000   Median :-1   Median :-1  
##  Mean   : 0.9643   Mean   : 557.8   Mean   : 0.1624   Mean   :-1   Mean   :-1  
##  3rd Qu.: 1.0000   3rd Qu.:2010.0   3rd Qu.: 2.4000   3rd Qu.:-1   3rd Qu.:-1  
##  Max.   : 4.0000   Max.   :2016.0   Max.   : 4.0000   Max.   :-1   Max.   :-1                                                                             
##  DualHSSummerEnroll EnrollmentStatus NumColCredAttemptTransfer
##  Min.   :0          Min.   :1.000    Min.   : -2.00           
##  1st Qu.:0          1st Qu.:1.000    1st Qu.: -2.00           
##  Median :0          Median :2.000    Median : 14.00           
##  Mean   :0          Mean   :1.589    Mean   : 36.97           
##  3rd Qu.:0          3rd Qu.:2.000    3rd Qu.: 73.00           
##  Max.   :0          Max.   :2.000    Max.   :150.00           
##                                                               
##  NumColCredAcceptTransfer CumLoanAtEntry      HighDeg       MathPlacement    
##  Min.   :-2.00            Min.   :-2.000   Min.   :0.0000   Min.   :-1.0000  
##  1st Qu.:-2.00            1st Qu.:-2.000   1st Qu.:0.0000   1st Qu.: 0.0000  
##  Median :22.00            Median :-1.000   Median :0.0000   Median : 0.0000  
##  Mean   :31.77            Mean   :-1.411   Mean   :0.5849   Mean   : 0.2793  
##  3rd Qu.:66.00            3rd Qu.:-1.000   3rd Qu.:2.0000   3rd Qu.: 1.0000  
##  Max.   :96.00            Max.   :-1.000   Max.   :4.0000   Max.   : 1.0000                                                                          
##   EngPlacement     GatewayMathStatus GatewayEnglishStatus
##  Min.   :-1.0000   Min.   :0.0000    Min.   :0.0000      
##  1st Qu.: 0.0000   1st Qu.:0.0000    1st Qu.:0.0000      
##  Median : 0.0000   Median :0.0000    Median :0.0000      
##  Mean   : 0.1869   Mean   :0.1197    Mean   :0.1902      
##  3rd Qu.: 0.0000   3rd Qu.:0.0000    3rd Qu.:0.0000      
##  Max.   : 1.0000   Max.   :1.0000    Max.   :1.0000      
```
Distribution of Cohort

`bar1 <- ggplot(data=StudentStaticData, aes(x=Cohort)) + geom_bar(color="red", fill=rgb(0,0,1,0.7))`

![Cohort bar chart](/img/cohort.png "Cohort bar chart")

Distribution of Academic Year, most students were in the year 2016-2017

`bar3 <- ggplot(data=ProgressData, aes(x=AcademicYear)) + geom_bar(color="red", fill=rgb(0,0,1,0.7))`

![academic bar chart](/img/academic.png "Academic Year bar chart")

Distribution of Gender

`bar2 <- ggplot(data=StudentStaticData, aes(x=Gender)) + geom_bar(color="red", fill=rgb(0,0,1,0.7))`

![gender bar chart](/img/gender.png "Gender bar chart")

Distribution of Housing, most students were living out of campus

`bar5 <- ggplot(data=FinancialAid, aes(x=Housing)) + geom_bar(color="red", fill=rgb(0,0,1,0.7))`

![housing bar chart](/img/housing.png "Housing bar chart")

Distribution of Mairital Status, most students were single

`bar6 <- ggplot(data=FinancialAid, aes(x=MaritalStatus)) + geom_bar(color="red", fill=rgb(0,0,1,0.7))`

![maritalstatus bar chart](/img/maritalstatus.png "Marital Status bar chart")

## Data Cleaning 

### Data cleaning for Student Static Data
```
#Remove address columns because we won't use them for training and testing data: Address1, Address2,    City,   State,  Zip,    RegistrationDate
#Remove columns because of missing most values: Campus, HSDipYr HSGPAWtd, FirstGen, DualHSSummerEnroll, CumLoanAtEntry,   
StudentStatic <- StudentStaticData[-c(4, 5, 6, 7, 8, 9, 10, 22, 24, 25, 26, 30)]
#Replace value = -1 in Hispanic,    AmericanIndian, Asian,  Black,  NativeHawaiian, White,  TwoOrMoreRace = 0
StudentStatic["Hispanic"][StudentStatic["Hispanic"] == -1] <- 0
StudentStatic["AmericanIndian"][StudentStatic["AmericanIndian"] == -1] <- 0
StudentStatic["Asian"][StudentStatic["Asian"] == -1] <- 0
StudentStatic["Black"][StudentStatic["Black"] == -1] <- 0
StudentStatic["NativeHawaiian"][StudentStatic["NativeHawaiian"] == -1] <- 0
StudentStatic["White"][StudentStatic["White"] == -1] <- 0
StudentStatic["TwoOrMoreRace"][StudentStatic["TwoOrMoreRace"] == -1] <- 0
#Replace value = -1 in HSDip = 1 because all students completed high school before applying for college
StudentStatic["HSDip"][StudentStatic["HSDip"] == -1] <- 0
#Replace values = -1 in HSGPAUnwtd = mean
StudentStatic["HSGPAUnwtd"][StudentStatic["HSGPAUnwtd"] == -1] <- mean(StudentStatic$HSGPAUnwtd>0)
#Replace missing values = -1, -2 in NumColCredAttemptTransfer = 0
StudentStatic["NumColCredAttemptTransfer"][StudentStatic["NumColCredAttemptTransfer"] == -1] <- 0
StudentStatic["NumColCredAttemptTransfer"][StudentStatic["NumColCredAttemptTransfer"] == -2] <- 0
#Replace missing values = -1, -2 in NumColCredAcceptTransfer = 0
StudentStatic["NumColCredAcceptTransfer"][StudentStatic["NumColCredAcceptTransfer"] == -1] <- 0
StudentStatic["NumColCredAcceptTransfer"][StudentStatic["NumColCredAcceptTransfer"] == -2] <- 0
#Replace missing values = -1 in MathPlacement column by majority value = 0 
StudentStatic["MathPlacement"][StudentStatic["MathPlacement"] == -1] <- 0
#Replace missing values = -1 in EngPlacement column by majority value = 0 
StudentStatic["EngPlacement"][StudentStatic["EngPlacement"] == -1] <- 0
```
### Data cleaning for Student Progress Data

```
# Data cleaning for Student Progress Data
#Remove columns because missing data: Complete2,CompleteCIP2,TransferIntent,DegreeTypeSought,AcademicYearID
Progress <- ProgressData[-c(11, 13, 14, 15, 18)]
#Replace missing values = -1, -2 in CompleteDevMath = 0
Progress["CompleteDevMath"][Progress["CompleteDevMath"] == -1] <- 0
Progress["CompleteDevMath"][Progress["CompleteDevMath"] == -2] <- 0
#Replace missing values = -1, -2 in CompleteDevEnglish = 0
Progress["CompleteDevEnglish"][Progress["CompleteDevEnglish"] == -1] <- 0
Progress["CompleteDevEnglish"][Progress["CompleteDevEnglish"] == -2] <- 0
#Replace missing values = -1 in Major1 = 0
Progress["Major1"][Progress["Major1"] == -1] <- 0
#Replace missing values = -1 in Major2 = 0
Progress["Major2"][Progress["Major2"] == -1] <- 0
#Replace missing values = -2 in CompleteCIP1 = 0
Progress["CompleteCIP1"][Progress["CompleteCIP1"] == -2] <- 0

```
### Data cleaning for Financial Aid Data

```
# Data cleaning for Financial Aid Data
# Most of students are single, so fill the empty values of Marital Status column with Single.
FinancialAid["MaritalStatus"][FinancialAid["MaritalStatus"] == ""] <- "Single"
# Most of students live Off campus, so fill the empty values of Housing column with Off Campus.
FinancialAid["Housing"][FinancialAid["Housing"] == ""] <- "Off Campus"
# Fill the empty values of parent's Highest Grade level with 'Unknown'.
FinancialAid["FathersHighestGradeLevel"][FinancialAid["FathersHighestGradeLevel"] == ""] <- "Unknown"
FinancialAid["MotherHighestGradeLevel"][FinancialAid["MotherHighestGradeLevel"] == ""] <- "Unknown"
# Replace all other missing values by 0
FinancialAid <- na_replace(FinancialAid, 0)
```
### Merge Static Data, Progress Data, Fiancial Data

```
StaticProgressData <- merge(x=StudentStatic,y=Progress,by="StudentID")
FinancialStaticProgressData <- merge(x=StaticProgressData,y=FinancialAid, by="StudentID")
# Merge FinancailStaticProgressData with TrainLabels Data
StaticProgressData_Train <- merge(x=FinancialStaticProgressData,y=TrainLabels,by="StudentID")
DataTrain <- StaticProgressData_Train[-c(2, 3, 4, 24, 25)]
DataTrain$Dropout <- as.factor(DataTrain$Dropout)
```
## Methodology and Results

The data set was split in 75% train and 25% test, training the models using grid search and cross-validation on the training set and evaluating them on the test set.

```
library(caret)
intrain <- createDataPartition(DataTrain$Dropout,p=0.75,list = FALSE)
test1 <- DataTrain[-intrain,]
```
### Fit the classification tree model

```
model1 <- train(Dropout ~., data = train1, method = "rpart", trControl=trctrl)
predictions1 <- predict(model1, newdata = test1)
confusionMatrix(predictions1,test1$Dropout)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    0    1
##          0 1845   93
##          1   36 1090
##                                           
##                Accuracy : 0.9579          
##                  95% CI : (0.9502, 0.9647)
##     No Information Rate : 0.6139          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9104          
##                                           
##  Mcnemar's Test P-Value : 8.201e-07       
##                                           
##             Sensitivity : 0.9809          
##             Specificity : 0.9214          
##          Pos Pred Value : 0.9520          
##          Neg Pred Value : 0.9680          
##              Prevalence : 0.6139          
##          Detection Rate : 0.6022          
##    Detection Prevalence : 0.6325          
##       Balanced Accuracy : 0.9511                                
##        'Positive' Class : 0               
bagImp1 <- varImp(model1, scale=TRUE)
bagImp1
## rpart variable importance
##   only 20 most important variables shown (out of 80)                                        Overall
## CompleteCIP1                            100.0000
## Complete1                                99.9018
## CumGPA                                   38.4670
## AcademicYear2016-17                      36.8324
## X2017Grant                               21.0350
## TermGPA                                  20.0160
## EnrollmentStatus                         11.6191
## BirthYear                                 3.9111
## StudentID                                 2.9725
## cohort2016-17                             2.6334
## X2012Grant                                2.3654
## cohort2015-16                             2.0686
## X2016Loan                                 1.7867
## X2016Scholarship                          1.7374
## X2013Grant                                1.6631
## ParentAdjustedGrossIncome                 1.1870
## X2016Grant                                1.1677
## X2012Loan                                 0.5082
## X2017Scholarship                          0.4335
## `FathersHighestGradeLevelMiddle School`   0.0000
```
### Fit the Logistic Regression Model

```
model2 <- train(Dropout ~., data = train1, method = "glm", trControl=trctrl)
predictions2 <- predict(model2, newdata = test1)
confusionMatrix(predictions2,test1$Dropout)
## Confusion Matrix and Statistics
##           Reference
## Prediction    0    1
##          0 1832   90
##          1   49 1093                                         
##                Accuracy : 0.9546          
##                  95% CI : (0.9467, 0.9617)
##     No Information Rate : 0.6139          
##     P-Value [Acc > NIR] : < 2.2e-16                                              
##                   Kappa : 0.9037                                                
##  Mcnemar's Test P-Value : 0.0006919                                            
##             Sensitivity : 0.9740          
##             Specificity : 0.9239          
##          Pos Pred Value : 0.9532          
##          Neg Pred Value : 0.9571          
##              Prevalence : 0.6139          
##          Detection Rate : 0.5979          
##    Detection Prevalence : 0.6273          
##       Balanced Accuracy : 0.9489                                               
##        'Positive' Class : 0               
bagImp2 <- varImp(model2, scale=TRUE)
bagImp2
## glm variable importance
##   only 20 most important variables shown (out of 77)
##                            Overall
## Complete1                   100.00
## `cohort2015-16`              87.61
## X2016Grant                   61.81
## `cohort2014-15`              47.16
## `AcademicYear2016-17`        39.74
## CumGPA                       37.36
## ParentAdjustedGrossIncome    35.13
## X2016Loan                    34.63
## X2016Scholarship             27.97
## EnrollmentStatus             27.67
## X2017Grant                   27.12
## CompleteDevMath              26.51
## `AcademicYear2015-16`        25.48
## `cohort2013-14`              24.26
## HSGPAUnwtd                   23.29
## X2015Loan                    22.48
## `HousingOn Campus Housing`   21.19
## MathPlacement                21.10
## Term                         20.87
## X2012Work_Study              19.57
```
### Fit the Bagging Model

```
model3 <- train(Dropout ~., data = train1, method = "treebag", trControl=trctrl)
predictions3 <- predict(model3, newdata = test1)
confusionMatrix(predictions3,test1$Dropout)
## Confusion Matrix and Statistics
##           Reference
## Prediction    0    1
##          0 1846   78
##          1   35 1105                                       
##                Accuracy : 0.9631          
##                  95% CI : (0.9558, 0.9695)
##     No Information Rate : 0.6139          
##     P-Value [Acc > NIR] : < 2.2e-16                                               
##                   Kappa : 0.9217                                                   
##  Mcnemar's Test P-Value : 7.782e-05                                              
##             Sensitivity : 0.9814          
##             Specificity : 0.9341          
##          Pos Pred Value : 0.9595          
##          Neg Pred Value : 0.9693          
##              Prevalence : 0.6139          
##          Detection Rate : 0.6025          
##    Detection Prevalence : 0.6279          
##       Balanced Accuracy : 0.9577                                                  
##        'Positive' Class : 0               
bagImp3 <- varImp(model3, scale=TRUE)
bagImp3
## treebag variable importance
##   only 20 most important variables shown (out of 93)
##                           Overall
## CompleteCIP1              100.000
## Complete1                  99.579
## CumGPA                     39.351
## AcademicYear2016-17        35.413
## TermGPA                    28.644
## X2017Grant                 22.798
## EnrollmentStatus           12.036
## StudentID                   8.478
## BirthYear                   7.320
## BirthMonth                  3.808
## X2016Grant                  3.672
## X2016Loan                   3.619
## Major1                      3.504
## NumColCredAttemptTransfer   3.416
## ParentAdjustedGrossIncome   3.283
## cohort2016-17               3.148
## cohort2015-16               3.024
## X2013Grant                  2.991
## X2012Grant                  2.965
## NumColCredAcceptTransfer    2.923
```
### Fit the SVM Radial Model

```
model4 <- train(Dropout ~., data = train1, method = "svmRadial", trControl=trctrl)
predictions4 <- predict(model4, newdata = test1)
confusionMatrix(predictions4,test1$Dropout)
## Confusion Matrix and Statistics
##           Reference
## Prediction    0    1
##          0 1860  120
##          1   21 1063                                       
##                Accuracy : 0.954          
##                  95% CI : (0.946, 0.9611)
##     No Information Rate : 0.6139         
##     P-Value [Acc > NIR] : < 2.2e-16                                             
##                   Kappa : 0.9014                                             
##  Mcnemar's Test P-Value : < 2.2e-16                                             
##             Sensitivity : 0.9888         
##             Specificity : 0.8986         
##          Pos Pred Value : 0.9394         
##          Neg Pred Value : 0.9806         
##              Prevalence : 0.6139         
##          Detection Rate : 0.6070         
##    Detection Prevalence : 0.6462         
##       Balanced Accuracy : 0.9437                                                 
##        'Positive' Class : 0
```
### Stacking using Random Forest

```
# Construct data frame with predictions
library(caret)
predDF <- data.frame(predictions1, predictions2, predictions3, predictions4, class = test1$Dropout)
predDF$class <- as.factor(predDF$class)
#Combine models using random forest
combModFit.rf <- train(class ~ ., method = "rf", data = predDF, distribution = 'multinomial')
combPred.rf <- predict(combModFit.rf, predDF)
confusionMatrix(combPred.rf, predDF$class)$overall[1]
##  Accuracy 
## 0.9631201
```
### Compare the accuracy of each model

The performance of the classifiers is assessed using the standard measure of accuracy.

Model                                                 Accuracy Score 

Classification Tree:                                     95.79%

Logistic Regression:                                     95.46% 

Bagging:                                                 96.31% 

SVM Radial:                                              95.4 % 

Stacking with Random Forest:                             96.31%

Bagging and Stacking model have the higher accuracy score than the others. 

### ROC Curve

```
library(pROC)
# ROC Curve
roccurve1 <- roc(test1$Dropout ~ as.numeric(predictions1))
roccurve2 <- roc(test1$Dropout ~ as.numeric(predictions2))
roccurve3 <- roc(test1$Dropout ~ as.numeric(predictions3))
roccurve4 <- roc(test1$Dropout ~ as.numeric(predictions4))
roccurve <- roc(predDF$class ~ as.numeric(combPred.rf))
roccurve$auc
## Area under the curve: 0.9577
roccurve$sensitivities
## [1] 1.0000000 0.9340659 0.0000000
roccurve$specificities
## [1] 0.0000000 0.9813929 1.0000000
plot(roccurve1, print.auc = TRUE, col = "red",print.auc.y = .4, lwd =4,legacy.axes=TRUE,main="ROC Curves")
plot(roccurve2, print.auc = TRUE, col = "black", print.auc.y = .5, add = TRUE, lwd =4,legacy.axes=TRUE,main="ROC Curves")
plot(roccurve3, print.auc = TRUE, col = "blue", print.auc.y = .6, add = TRUE, lwd =4,legacy.axes=TRUE,main="ROC Curves")
plot(roccurve4, print.auc = TRUE, col = "green", print.auc.y = .7, add = TRUE, lwd =4,legacy.axes=TRUE,main="ROC Curves")
plot(roccurve, print.auc = TRUE, col = "orange", print.auc.y = .8, add = TRUE,lwd =4,legacy.axes=TRUE,main="ROC Curves")
```

![ROC Curves plot](/img/ROCcurves.png "ROC Curves plot")

The plot presents the ROC curves for the fine binary classifiers used in this study. The bagging model and Stacking model using Random Forest performed the same AUC and better than the Classification Tree, Logistic Regression and SVM.

## Conclusion and Future Works 

### Conclusion 

By this project, I have presented many machine learning models to predict New Jersey City student dropout. We see that these models achieve high predictive power, combining values of AUC ROC for decision-making with capable of achieving with accuracy score of over 96% in its predictions. The result was that the Bagging and Stacking with Random Forest performed the best, followed by the Classification Tree, Logistic Regression and SVM. 

### Limitation

Some improvements that can be made to the experiment include a more advanced solution dealing with missing values rather than replacing missing values to 0 or the majority value. For great quality to be achieved, this means there should be no missing or wrong data points in the dataset, as well as consistent and useable formatting of the data. 
Developing such a model demands analytics and coding skills. These two skills, even if required, are not enough: having subject-matter experts providing input on the industry practices and interpreting results and data is crucial to success. 

### Future works 

In this study, I limited our scope to New Jersey City students, but the same models developed for this purpose could be used for colleges, given that the models are trained and supplied with the appropriate data. Consequently, the relevant factors we have identified as impact for predicting dropout students in these models are relevant for any other college’s students.










