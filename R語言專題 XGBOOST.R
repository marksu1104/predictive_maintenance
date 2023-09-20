###################################################
#
# XGboost
#
###################################################
#
#  data：             應變數
#  label：            自變數(想預測的結果)
#  nrounds：          樹的數量
#  eta：              學習率
#  max_depth：        樹分幾層
#  min_child_weight： 最底下節點最小本數,分到樣本數小於該數時決策樹停止生長
#  subsample：        每次隨機抽樣比例
#  colsample_bytree： 如mtry每次產生一顆樹挑多少變數進來
#  nrounds：          樹的數量
#  objective：        預測方法
#  params：           可以將上述的參數，用list方式帶入
#
#####################################################
#install.packages("xgboost")
library("xgboost")
library("performanceEstimation") 

#清空記憶體
rm(list=ls())

#設定檔案路徑及資料編碼
setwd("C:/R語言/隨機森林")
Sys.setlocale(category = "LC_ALL", locale = "zh_TW.UTF-8") # 避免中文亂碼

#讀取檔案資料
#讀取檔案，header = TRUE表示第一筆資料為標題，stringsAsFactors = FALSE表示不自動設定分類
dp=read.csv("predictive_maintenance.csv", header=TRUE, stringsAsFactors = TRUE, fileEncoding = 'utf-8')
#建立引數(就可以直接使用欄位名稱)
attach(dp)


#設定圖表每Page顯示4個
par(mfrow=c(1,5))
#繪製箱型圖：$out輸出離群值結果
boxplot(dp$AirTemperature_K)$out
boxplot(dp$ProcessTemperature_K)$out
boxplot(dp$RotationalSpeed_rpm)$out
boxplot(dp$Torque_Nm)$out
boxplot(dp$ToolWear_min)$out


######################## 重新分類FailureType裡的值 #######################


#取出Heat Dissipation Failure
dp = cbind(dp,hdf=factor(ifelse(dp$FailureType == "Heat Dissipation Failure",1,0)))

#取出Overstrain Failure
dp = cbind(dp,of=factor(ifelse(dp$FailureType == "Overstrain Failure",1,0)))

#取出Power Failure
dp = cbind(dp,pf=factor(ifelse(dp$FailureType == "Power Failure",1,0)))

#取出Random Failures
dp = cbind(dp,rf=factor(ifelse(dp$FailureType == "Random Failures",1,0)))

#取出Tool Wear Failure
dp = cbind(dp,twf=factor(ifelse(dp$FailureType == "Tool Wear Failure",1,0)))


######################從隨機森林的預測模型中知道有不平衡問題###########################
# 針對每種錯誤訊息進行smote處理，來增加少數樣本

table(dp$hdf)
newData = smote(hdf ~ ., dp, perc.over = 30,perc.under=3)
table(newData$hdf)


#取得總比數，並隨機成學習集及測試集
n=dim(newData)[1]
dp_idx=sample(1:n,size=round(0.7*n))
dp_data=newData[dp_idx,]
test_data=newData[-dp_idx,]

################## 分割自變數和應變數 #################
# 自變數：矩陣格式
# 應變數：數值格式

dp_x = as.matrix(cbind(dp_data[,4],dp_data[,5],dp_data[,6],dp_data[,7],dp_data[,8]))
dp_y = as.integer(as.matrix(dp_data$hdf))

test_x = as.matrix(cbind(test_data[,4],test_data[,5],test_data[,6],test_data[,7],test_data[,8]))
test_y = as.integer(as.matrix(test_data$hdf))

################# 進行xGB預測 ##################

set.seed(123)
model = xgboost(data=dp_x, label =dp_y, nrounds= 500)


########################## 驗證預測 ##############################
result = predict(model,test_x)
CM=table(result,test_data$hdf)
sum(diag(CM))/sum(CM)

######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)

######################### 模式改成分類 ##########################
model = xgboost(data=dp_x, label =dp_y, nrounds= 500, 
                objective = "binary:hinge")

########################## 驗證預測 ##############################
result = predict(model,test_x)
CM=table(result,test_data$hdf)
sum(diag(CM))/sum(CM)

######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)



################ 進行自變數調整 ##################
#使用xgb內建的功能
##################################################
## xgb.plot.importance
## 觀察那些變數比較有用
## top_n:前n重要
## measure: 繪製重要性度量的名稱，預設NULL, Gain:用於tree，Weight：用於gb線性

importance = xgb.importance(model = model)
par(mfrow=c(1,1))
xgb.plot.importance(importance, top_n = 5, measure = "Gain")

######################### 找出最佳參數 ####################

#v = expand.grid(
#  eta = c(.01, .05, .1, .3),            #學習率
#  max_depth = c(1, 3, 5, 7),            #樹分幾層
#  min_child_weight = c(1, 3, 5, 7),     #最底下節點最小本數,分到樣本數小於該數時決策樹停止生長
#  subsample = c(.65, .8, 1),            #每次隨機抽樣比例
#  colsample_bytree = c(.8, .9, 1),      #如mtry每次產生一顆樹挑多少變數進來
#  nrounds = 0,                          #用來儲存樹的最佳數量
#  loss = 0                              #用來儲存預測成功機率
#)


##### v的組合太多，需要花很多的時間去處理因此，教學時先縮減

v = expand.grid(
  eta = c(.05),                     #學習率
  max_depth = c(5),                  #樹分幾層
  min_child_weight = c(3),        #最底下節點最小本數,分到樣本數小於該數時決策樹停止生長
  subsample = c(.65, .8),               #每次隨機抽樣比例
  colsample_bytree = c(.8, 1),      #如mtry每次產生一顆樹挑多少變數進來
  nrounds = 0,                          #用來儲存樹的最佳數量
  error = 0                              #用來儲存預測成功機率
)


nrow(v)

for(i in 1:nrow(v)){
  params = list(
    eta = v$eta[i],
    max_depth = v$max_depth[i],
    min_child_weight = v$min_child_weight[i],
    subsample = v$subsample[i],
    colsample_bytree = v$colsample_bytree[i]
  )
  
  set.seed(123)
  
  ############################# 交叉驗證效能 ##########################
  ## xgboost函式庫中的xgb.cv
  ## xgb.cv產出的結果中的值中，其中evaluation_log用來儲存評估記錄
  
  cr = xgb.cv(
    params = params,
    data = dp_x[,-5],    #剔除參考價值最少的欄位,
    label =dp_y ,
    nrounds = 500, 
    nfold = 5,
    objective = "binary:hinge",
    verbose = 0,               
    early_stopping_rounds = 10 
  )
  
  v$nrounds[i] = which.min(cr$evaluation_log$test_error_mean)
  v$error[i] = min(cr$evaluation_log$test_error_mean)
  
}

########################### 根據結果重新建立模型 ###############################
set.seed(123)
model = xgboost(data=dp_x[,-5], label=dp_y,
                eta=0.1, max_depth=5, min_child_weight=3, subsample = 0.8, colsample_bytree= 0.8, nrounds= 69,
                objective = "binary:hinge")

########################## 驗證預測 ##############################
result = predict(model,test_x[,-c(4:5)])
CM=table(result,test_data$hdf)
sum(diag(CM))/sum(CM)

######################### 資料合併看預測結果與實際結果 ###########################
mm = as.data.frame(result)
f = cbind(test_data,mm)
