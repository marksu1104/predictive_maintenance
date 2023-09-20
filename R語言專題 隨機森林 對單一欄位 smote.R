###################################################
#
# 隨機森林
# 
# 從訓練集(母體N)抽樣取出n筆資料(採抽樣可放回方式)來建立決策樹
# 每棵決策樹採用隨機挑選K個特徵建立
# 重複M次建立M棵決策樹
# 分類(預測)過程中，若為線性問題採平均數，非線性問題採眾數(或稱投票機制)
#
###################################################


library(rpart) #install.packages("rpart")
library(rpart.plot) #install.packages("rpart.plot")
library("randomForest") #install.packages("randomForest")
library("performanceEstimation") 

#清空暫存資料
rm(list=ls())

#設定檔案路徑及資料編碼
setwd("C:/R語言/隨機森林")
Sys.setlocale(category = "LC_ALL", locale = "zh_TW.UTF-8") # 避免中文亂碼

#讀取檔案資料
#讀取檔案，header = TRUE表示第一筆資料為標題，stringsAsFactors = FALSE表示不自動設定分類
dp=read.csv("predictive_maintenance.csv", header=TRUE, stringsAsFactors = TRUE, fileEncoding = 'utf-8')
#建立引數(就可以直接使用欄位名稱)
attach(dp)

#分割資料為訓練資料及測試資料
n=dim(dp)[1]
dp_idx=sample(1:n,size=round(0.7*n))
#訓練資料
dp_data=dp[dp_idx,]
#測試資料
test_data=dp[-dp_idx,]

#應變數欄位名稱~. 表示除了應變數欄位其餘全是自變數
#應變數欄位名稱~自變數名稱+自變數名稱...
#randomforestM=randomForest(Failure~feature+Life, data=dp_data, importance=T,
#                           proximity=T, do.trace=50,ntree=200)
#設定亂數
set.seed(123)


randomforestM=randomForest(FailureType~AirTemperature_K+ProcessTemperature_K+RotationalSpeed_rpm+Torque_Nm+ToolWear_min
                           , data=dp_data, importance=T, proximity=T, do.trace=100,ntree=500)

randomforestM
plot(randomforestM)

round(importance(randomforestM,2))
result=predict(randomforestM,newdata=test_data)
CM=table(result,test_data$Failure)
sum(diag(CM))/sum(CM)

######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)


######################################################################################
########################################### 樣本數調整 ###############################
######################################################################################
table(dp_data$FailureType)
newData <- smote(FailureType ~ ., dp_data, perc.over = 13,perc.under=43)
table(newData$FailureType)


#分割資料為訓練資料及測試資料
n=dim(newData)[1]
dp_idx=sample(1:n,size=round(0.7*n))
#訓練資料
dp_data=newData[dp_idx,]

#應變數欄位名稱~. 表示除了應變數欄位其餘全是自變數
#應變數欄位名稱~自變數名稱+自變數名稱...
#randomforestM=randomForest(Failure~feature+Life, data=dp_data, importance=T,
#                           proximity=T, do.trace=50,ntree=200)
#設定亂數
set.seed(123)


randomforestM=randomForest(FailureType~AirTemperature_K+ProcessTemperature_K+RotationalSpeed_rpm+Torque_Nm+ToolWear_min
                           , data=dp_data, importance=T, proximity=T, do.trace=100,ntree=200)

randomforestM
plot(randomforestM)

round(importance(randomforestM,2))
result=predict(randomforestM,newdata=test_data)
CM=table(result,test_data$Failure)
sum(diag(CM))/sum(CM)
CM
######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)


#########################################################################
###
###            剔除class.error為1的
###
#########################################################################
#dp = cbind(dp,nft=factor(ifelse(dp$FailureType == "Heat Dissipation Failure ","Heat Dissipation Failure ",
#   ifelse(dp$FailureType == "Overstrain Failure","Overstrain Failure",
#      ifelse(dp$FailureType == "Power Failure","Power Failure","No Failure")                                        
#   )                           
#)))

dp = cbind(dp,nft=factor(ifelse(dp$FailureType == "Heat Dissipation Failure","Heat Dissipation Failure",
                                ifelse(dp$FailureType == "Overstrain Failure","Overstrain Failure",
                                       ifelse(dp$FailureType == "Power Failure","Power Failure","No Failure")                     
)
)))

#分割資料為訓練資料及測試資料
n=dim(dp)[1]
dp_idx=sample(1:n,size=round(0.7*n))
#訓練資料
dp_data=dp[dp_idx,]
#測試資料
test_data=dp[-dp_idx,]

randomforestM=randomForest(nft~AirTemperature_K+ProcessTemperature_K+RotationalSpeed_rpm+Torque_Nm+ToolWear_min
                           , data=dp_data, importance=T, proximity=T, do.trace=100,ntree=200)

randomforestM
plot(randomforestM)

round(importance(randomforestM,2))
result=predict(randomforestM,newdata=test_data)
CM=table(result,test_data$Failure)
sum(diag(CM))/sum(CM)

######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)

#找出最適當的變量個數
ndp = cbind(dp[,4],dp[,5],dp[,6],dp[,7],dp[,8])
tuneRF(x=ndp, y=dp$nft, mtryStart = 1, ntreeTry = 200)


randomforestM=randomForest(nft~AirTemperature_K+ProcessTemperature_K+RotationalSpeed_rpm+Torque_Nm+ToolWear_min
                           , data=dp_data, importance=T, proximity=T, do.trace=100,ntree=200, mtry=4)

randomforestM
plot(randomforestM)

round(importance(randomforestM,2))
result=predict(randomforestM,newdata=test_data)
CM=table(result,test_data$Failure)
sum(diag(CM))/sum(CM)

######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)

#########################################################################
###
###            每種Failure獨立處理 看預測率是否更加
###
#########################################################################


######################## 重新分類FailureType裡的值 #######################
#取出Heat Dissipation Failure
dp = cbind(dp,hdf=factor(ifelse(dp$FailureType == "Heat Dissipation Failure","Failure","NotFailure")))

#取出Overstrain Failure
dp = cbind(dp,of=factor(ifelse(dp$FailureType == "Overstrain Failure","Failure","NotFailure")))

#取出Power Failure
dp = cbind(dp,pf=factor(ifelse(dp$FailureType == "Power Failure","Failure","NotFailure")))

#取出Random Failures
dp = cbind(dp,rf=factor(ifelse(dp$FailureType == "Random Failures","Failure","NotFailure")))

#取出Tool Wear Failure
dp = cbind(dp,twf=factor(ifelse(dp$FailureType == "Tool Wear Failure","Failure","NotFailure")))


####################################### 針對hdf欄位做學習及預測 ####################################

table(dp$of)
newData <- smote(of ~ ., dp, perc.over = 13,perc.under=43)
table(newData$of)

#分割資料為訓練資料及測試資料
n=dim(newData)[1]
dp_idx=sample(1:n,size=round(0.7*n))
#訓練資料
dp_newdata=dp[dp_idx,]
#測試資料
test_data=dp[-dp_idx,]

#應變數欄位名稱~. 表示除了應變數欄位其餘全是自變數
#應變數欄位名稱~自變數名稱+自變數名稱...
#randomforestM=randomForest(Failure~feature+Life, data=dp_data, importance=T,
#                           proximity=T, do.trace=50,ntree=200)
#設定亂數
set.seed(123)

randomforestM=randomForest(of~AirTemperature_K+ProcessTemperature_K+RotationalSpeed_rpm+Torque_Nm+ToolWear_min
                           , data=dp_data, importance=T, proximity=T, do.trace=100,ntree=500)

randomforestM
plot(randomforestM)

round(importance(randomforestM,2))
result=predict(randomforestM,newdata=test_data)
CM=table(result,test_data$of)
sum(diag(CM))/sum(CM)
CM

######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)

#找出最適當的變量個數
ndp = cbind(dp[,4],dp[,5],dp[,6],dp[,7],dp[,8])
tuneRF(x=ndp, y=dp$of, mtryStart = 1, ntreeTry = 200)

randomforestM=randomForest(of~AirTemperature_K+ProcessTemperature_K+RotationalSpeed_rpm+Torque_Nm+ToolWear_min
                           , data=dp_data, importance=T, proximity=T, do.trace=100,ntree=200, mtry = 5)

randomforestM
plot(randomforestM)

round(importance(randomforestM,2))
result=predict(randomforestM,newdata=test_data)
CM=table(result,test_data$of)
sum(diag(CM))/sum(CM)
CM
######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)


