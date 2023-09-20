#清空記憶體
rm(list=ls())

# 避免中文亂碼
Sys.setlocale(category = "LC_ALL", locale = "zh_TW.UTF-8")

#讀取檔案資料
data = read.csv("predictive_maintenance.csv", header=TRUE, 
                stringsAsFactors = TRUE, fileEncoding = 'utf-8')

data = subset(data,select=c(-UDI,-Product.ID))


#取出Heat Dissipation Failure
data = cbind(data,hdf=factor(ifelse(data$Failure.Type == "Heat Dissipation Failure",1,0)))

#取出Overstrain Failure
data = cbind(data,of=factor(ifelse(data$Failure.Type == "Overstrain Failure",1,0)))

#取出Power Failure
data = cbind(data,pf=factor(ifelse(data$Failure.Type == "Power Failure",1,0)))

#取出Random Failures
data = cbind(data,rf=factor(ifelse(data$Failure.Type == "Random Failures",1,0)))

#取出Tool Wear Failure
data = cbind(data,twf=factor(ifelse(data$Failure.Type == "Tool Wear Failure",1,0)))