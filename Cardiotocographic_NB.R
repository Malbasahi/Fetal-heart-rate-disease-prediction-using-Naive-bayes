#libraries
library(dplyr)
library(ggplot2)
library(psych)
library(GGally)
library(caret)
library(naivebayes)
library(Amelia)
library(tidyr)
library(heatmap3)
library(mlbench)
library(e1071)
library(MLmetrics)

# Read Data
data <- read.csv("C:\\Users\\malba\\Downloads\\Cardiotocographic.csv")
View(data)
str(data)
data

# ------------------- Data cleaning --------------------- 
# Remove missing values
data<- na.omit(data)
# Remove duplicates
data<- unique(data)
#showing the data after cleaning 
data 
#change NSP in integer to factor format
data$NSPF <- factor(data$NSP)
xtabs(~NSPF,data=data)
# Display statistics of the Data set
describe(data)
# Count output elements number
table(data$NSPF)
# Identify the numerical features
num_features <- sapply(data, is.numeric)
# Normalize the numerical features
data[, num_features] <- scale(data[, num_features])
#to see IV in scaled value or ggplot
summary(data[,2:5])
summary(data[,1:4])
# Calculate the summary statistics for each variable
summary(data[, sapply(data, is.numeric)])
# Print the lower and upper bounds for each variable
sapply(data[, sapply(data, is.numeric)], function(x) c(min(x), max(x)))
num_features

#-------------perform Correlation-based feature selection------------------
colnames(data);

# Compute correlation matrix
correlation_matrix <- cor(data[-23])
# Set correlation threshold
threshold <- 0.5
# Identify highly correlated or redundant features
highly_correlated <- findCorrelation(correlation_matrix, cutoff = threshold)
# Select relevant features
selected_features <- data[, -highly_correlated]
# Print selected features
print(colnames(selected_features))
# Print the lower and upper bounds for each selected variable
sapply(selected_features[, sapply(selected_features, is.numeric)], function(x) c(min(x), max(x)))

correlation_matrix 

# Extract feature variables
X <- selected_features[-15]  # Exclude the target variable
# Standardize the feature variables
X_scaled <- scale(X)

# ------------------- Data visualizations ------------------- 

# Creating a scatterplot matrix using Scatterplot Matrix
pairs.panels(data[, c("LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV",
                      "ALTV", "MLTV")], # Subset the data to include only the specified features
             gap = 0, # Set the gap between panels to 0
             lm = TRUE, # Include linear regression lines
             stars = TRUE) # Include star plots for each variable
# Scatterplot Matrix
pairs.panels(data[, c("Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
                      "Median", "Variance", "Tendency", "NSPF")], 
             gap = 0, 
             lm = TRUE, 
             stars = TRUE) 

# Scatterplot Matrix for the selected features
pairs.panels(selected_features, # Subset the data to include only the the selected features
             gap = 0, 
             lm = TRUE, 
             stars = TRUE) 

#scatter plot
scatter_plot <- ggplot(selected_features, aes(x = Tendency, y = LB, color = NSPF)) +
  geom_point() +
  labs(title = "Scatter Plot of Tendency vs. LB by NSPF",
       x = "Tendency",
       y = "LB",
       color = "NSPF")
scatter_plot

#Scatter plot with significance
pairs.panels(selected_features[1:4],bg=c("red","yellow","blue")[selected_features$NSPF],
             pch=21+as.numeric(selected_features$NSPF),main="Pairs panels by NSPF",hist.col="red") 

#box plot
selected_features %>% ggplot(aes(x=NSPF, y=AC, fill= NSPF)) +
  geom_boxplot() +
  ggtitle("BoxPlot")


#box plot
selected_features %>% ggplot(aes(x=NSPF, y=LB, fill= NSPF)) +
  geom_boxplot() +
  ggtitle("BoxPlot")

#box plot
selected_features %>% ggplot(aes(x=NSPF, y=Tendency, fill= NSPF)) +
  geom_boxplot() +
  ggtitle("BoxPlot")

# Frequency histogram | Display the distribution of the the "NSPF" by "UC"
selected_features %>% ggplot(aes(x=UC, fill=NSPF, color=NSPF)) +
  geom_histogram(binwidth = 1) +
  labs(title="UC Distribution by NSPF")

# Frequency histogram | Display the distribution of the "NSPF" by "ASTV"
selected_features %>% ggplot(aes(x = ASTV, fill=NSPF, colour = NSPF)) +
  geom_histogram(binwidth = 1) +
  labs(title = "ASTV distribution by NSPF")

# Frequency polygon histogram | Display the distribution of the "NSPF" by "ALTV"
selected_features %>% ggplot(aes(ALTV, colour = NSPF)) +
  geom_freqpoly(binwidth = 1) +
  labs(title = "ALTV distribution by NSPF")

# Frequency polygon histogram | Display the distribution of the "NSPF" by "Nzeros"
selected_features %>% ggplot(aes(Nzeros, colour = NSPF)) +
  geom_freqpoly(binwidth = 1) +
  labs(title = "Nzeros distribution by NSPF")

#density plot
selected_features %>%
  ggplot(aes(x=MLTV,fill=NSPF))+
  geom_density(alpha=0.8,color='black')+
  ggtitle("density Plot")

#density plot
selected_features %>%
  ggplot(aes(x=Max,fill=NSPF))+
  geom_density(alpha=0.8,color='black')+
  ggtitle("density Plot")

#density plot
selected_features %>%
  ggplot(aes(x=DL,fill=NSPF))+
  geom_density(alpha=1,color='black')+
  ggtitle("density Plot")

#-----------using cross-validation and grid search to identify the optimal hyper parameter value--------------

# Standardize the feature variables
X
y <- selected_features$NSPF  # Target variable
y
# Convert factor levels to valid variable names
y <- make.names(as.character(y))

# Set up the training control for cross-validation
ctrl <- trainControl(method = "cv",  # Cross-validation method
                     number = 10,  # Number of folds
                     verboseIter = TRUE,  # Print progress
                     classProbs = TRUE,  # Enable class probabilities
                     summaryFunction = multiClassSummary)  # Summary function for multi-class classification

# Set up the grid of hyper parameter values to search
grid <- expand.grid(laplace = seq(0, 1, by = 0.1),  # Hyper parameter values for Laplace smoothing
                    usekernel = c(TRUE, FALSE),  # Hyper parameter values for use kernel
                    adjust = c(TRUE, FALSE))  # Hyper parameter values for adjust

# Train a Naive Bayes model with cross-validation and grid search
nb_model <- train(x = X_scaled, 
                  y = y, 
                  method = "naive_bayes",  # Specify the Naive Bayes method
                  trControl = ctrl,  # Specify the training control
                  tuneGrid = grid,  # Specify the grid of hyper parameter values
                  metric = "ROC")  # Specify the evaluation metric which is Receiver Opening Characteristic 

# Print the optimal hyper parameter values
print(nb_model$bestTune)
plot(nb_model)

# using time-based splitting to avoid data leakage
# Set up time-based train/test split
set.seed(123)  # Set a random seed for reproducibility
train_indices <- sample(1:nrow(X), nrow(X) * 0.7)  # 70% of the data for training
X_train <- X_scaled[train_indices, ]
y_train <- y[train_indices]
X_test <- X_scaled[-train_indices, ]
y_test <- y[-train_indices]

# Make predictions on test and train set
y_pred1 <- predict(nb_model, newdata = X_test)
y_pred2 <- predict(nb_model, newdata = X_train)

# Ensure predicted values and true values have the same factor levels
combined_levels <- union(levels(y_pred1), levels(y_test))
combined_levels <- union(levels(y_pred2), levels(y_train))
y_pred1 <- factor(y_pred1, levels = combined_levels)
y_pred2 <- factor(y_pred2, levels = combined_levels)
y_test <- factor(y_test, levels = combined_levels)
y_train <- factor(y_train, levels = combined_levels)

# Evaluate the model
confusion1 <- confusionMatrix(y_pred1, y_test)
confusion1
confusion2 <- confusionMatrix(y_pred2, y_train)
confusion2

plot(y_test)
plot(y_train)

# Calculate misclassification rate of testing 
misclassification_rate <- 1 - confusion1$overall["Accuracy"]
print(paste("Misclassification Rate:", misclassification_rate))
# Calculate misclassification rate of training 
misclassification_rate <- 1 - confusion2$overall["Accuracy"]
print(paste("Misclassification Rate:", misclassification_rate))



