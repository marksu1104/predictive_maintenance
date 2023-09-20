#清空記憶體
rm(list=ls())

#設定檔案路徑及資料編碼
Sys.setlocale(category = "LC_ALL", locale = "zh_TW.UTF-8") # 避免中文亂碼

#讀取檔案資料
#讀取檔案，header = TRUE表示第一筆資料為標題，stringsAsFactors = FALSE表示不自動設定分類
da=read.csv("predictive_maintenance.csv", header=TRUE, stringsAsFactors = FALSE, fileEncoding = 'utf-8')
#建立引數(就可以直接使用欄位名稱)
attach(da)

da=subset(da,select=c(-UDI,-Product.ID,-Type,-Target))

#########################################################################
###
###            每種Failure獨立處理
###
#########################################################################


######################## 重新分類FailureType裡的值 #######################
#取出Heat Dissipation Failure
da = cbind(da,hdf=factor(ifelse(da$Failure.Type == "Heat Dissipation Failure","Failure","NotFailure")))

#取出Overstrain Failure
da = cbind(da,of=factor(ifelse(da$Failure.Type == "Overstrain Failure","Failure","NotFailure")))

#取出Power Failure
da = cbind(da,pf=factor(ifelse(da$Failure.Type == "Power Failure","Failure","NotFailure")))

#取出Random Failures
da = cbind(da,rf=factor(ifelse(da$Failure.Type == "Random Failures","Failure","NotFailure")))

#取出Tool Wear Failure
da = cbind(da,twf=factor(ifelse(da$Failure.Type == "Tool Wear Failure","Failure","NotFailure")))



#取得總比數，並隨機成學習集及測試集
n=dim(da)[1]
da_idx=sample(1:n,size=round(0.7*n))
da_data=da[da_idx,]
test_data=da[-da_idx,]

########################################################################################
#
#   KNN (K最近鄰居分類法)
#   使用點間的距離為分類標準，當新的觀測值準備預測時，演算法會計算出和它接近最多的類目標點
#
#   train： 訓練集(應變數)
#   test：  測試集(應變數)
#   cl：    真實分類(自變數)
#   k：     參考鄰居數量
#
########################################################################################
library(class)

################## 分割自變數和應變數 #################

dp_x = cbind(da_data[,1],da_data[,2],da_data[,3],da_data[,4],da_data[,5])
dp_y = da_data$hdf

test_x = cbind(test_data[,1],test_data[,2],test_data[,3],test_data[,4],test_data[,5])
test_y = test_data$hdf

#計算k值(幾個鄰居)通常可以用資料數的平方根
kv = round(sqrt(n))
kv

model = knn(train = dp_x, test = test_x, cl = dp_y, k = kv)

cm = table(x = test_y, y = model, dnn = c("實際", "預測"))
cm

knnaccuracy = sum(diag(cm)) / sum(cm)
knnaccuracy

#########################進行資料不平衡處理########################
library("performanceEstimation") 
table(da$hdf)
newData = smote(hdf ~ ., da, perc.over = 30,perc.under=3)
table(newData$hdf)


#取得總比數，並隨機成學習集及測試集
n=dim(newData)[1]
da_idx=sample(1:n,size=round(0.7*n))
da_data=newData[da_idx,]
test_data=newData[-da_idx,]


##重複KNN動作
dp_x = cbind(da_data[,1],da_data[,2],da_data[,3],da_data[,4],da_data[,5])
dp_y = da_data$hdf

test_x = cbind(test_data[,1],test_data[,2],test_data[,3],test_data[,4],test_data[,5])
test_y = test_data$hdf

#計算k值(幾個鄰居)通常可以用資料數的平方根
kv = round(sqrt(n))
kv

model = knn(train = dp_x, test = test_x, cl = dp_y, k = kv)

cm = table(x = test_y, y = model, dnn = c("實際", "預測"))
cm

knnaccuracy = sum(diag(cm)) / sum(cm)
knnaccuracy


t = data.frame(kv=c(1:kv),p=0)
t

for(i in 1:kv){
  model = knn(train = dp_x, test = test_x, cl = dp_y, k = i)
  
  cm = table(x = test_y, y = model, dnn = c("實際", "預測"))
  
  t$p[i] = sum(diag(cm)) / sum(cm)
}  

model = knn(train = dp_x, test = test_x, cl = dp_y, k = 1)

cm = table(x = test_y, y = model, dnn = c("實際", "預測"))
cm

knnaccuracy = sum(diag(cm)) / sum(cm)
knnaccuracy


########################################################################################
#
#   SVM 
#   kernel:核心函數
#   cost: 容忍值, 預設值1(數值越大容錯越小)
#   gamma: 影響力
#   type: 模型類型(分類機、回歸機或新穎性檢測)
#   epsilon: (僅存在數值型預測，主要影響殘差值)
#
########################################################################################

#install.packages("e1071")
library("e1071")

dp_data=subset(da,select = c(-hdf,-of,-pf,-rf,-twf))

#取得總比數，並隨機成學習集及測試集
n=dim(dp_data)[1]
da_idx=sample(1:n,size=round(0.7*n))
da_data=dp_data[da_idx,]
test_data=dp_data[-da_idx,]

# 建立模型
model = svm(FailureType ~ ., data = da_data, type = "C-classification", kernel = "radial")

# 預測
results = predict(model, test_data)

# 評估
cm = table(x = test_data$FailureType, y = results, dnn = c("實際", "預測"))
cm
SVMaccuracy = sum(diag(cm)) / sum(cm)
SVMaccuracy

##############################針對Tool Wear Failure#########################
library("performanceEstimation") 
table(da$rf)
newData = smote(twf ~ ., da, perc.over = 30,perc.under=5)
table(newData$twf)

newData=subset(newData,select = c(-FailureType, -hdf,-of,-pf,-rf))

#取得總比數，並隨機成學習集及測試集
n=dim(newData)[1]
da_idx=sample(1:n,size=round(0.7*n))
da_data=newData[da_idx,]
test_data=newData[-da_idx,]


# 建立模型
model = svm(twf ~ ., data = da_data, type = "C-classification", kernel = "radial")

# 預測
results = predict(model, test_data)

# 評估
cm = table(x = test_data$twf, y = results, dnn = c("實際", "預測"))
cm
SVMaccuracy = sum(diag(cm)) / sum(cm)
SVMaccuracy

#################################找出最佳參數#################################
m = tune.svm(twf~., data=da_data, type="C-classification", kernel="radial",
             range=list(cost = 2^c(-4,-2,0,2,4)))

m$best.model

