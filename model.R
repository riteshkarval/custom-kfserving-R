library(caret)
library(mlbench)
library(randomForest)
library(doMC)
function(IN_DIR){
    registerDoMC(cores=8)
    data(Sonar)
    set.seed(7)
    validation_index <- createDataPartition(Sonar$Class, p=0.80, list=FALSE)
    validation <- Sonar[-validation_index,]
    model <- readRDS("model.rds")
    final_predictions <- predict(model, validation[,1:60])
    final_predictions
}
