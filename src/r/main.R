# Titanic survival prediction with logistic regression (R version)
 
suppressPackageStartupMessages(library(dplyr))
 
# In Docker the data folder is mounted at /data; locally it is ../data
DATA_DIR <- if (dir.exists("/data")) "/data" else file.path("..", "data")
 
# FamilySize is computed but not used as a feature: it equals SibSp + Parch + 1,
# so including all three would make glm's coefficients non-identifiable.
FEATURES <- c(
  "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
  "IsAlone", "Embarked", "Title"
)
 
section <- function(title) {
  cat("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60), "\n", sep = "")
}
 
extract_title <- function(name) {
  # "Braund, Mr. Owen Harris" -> "Mr"
  title <- trimws(sub("^[^,]*,\\s*([^.]*)\\..*$", "\\1", name))
  title[title %in% c("Mlle", "Ms")] <- "Miss"
  title[title == "Mme"] <- "Mrs"
  title[!title %in% c("Mr", "Mrs", "Miss", "Master")] <- "Rare"
  title
}
 
prepare <- function(df, fill_values) {
  df$Title <- extract_title(df$Name)
  cat("Extracted Title from Name:\n")
  print(table(df$Title))
 
  # read.csv reads empty strings as "" rather than NA for text columns
  df$Embarked[df$Embarked == ""] <- NA
 
  for (col in c("Age", "Fare", "Embarked")) {
    n_missing <- sum(is.na(df[[col]]))
    df[[col]][is.na(df[[col]])] <- fill_values[[col]]
    cat(sprintf("Filled %d missing values in %s with %s\n",
                n_missing, col, fill_values[[col]]))
  }
 
  df <- df %>% select(-Cabin, -Ticket, -Name)
  cat("Dropped columns: Cabin (mostly missing), Ticket, Name\n")
 
  df <- df %>% mutate(Sex = ifelse(Sex == "female", 1, 0))
  cat("Encoded Sex: male -> 0, female -> 1\n")
 
  df <- df %>% mutate(
    FamilySize = SibSp + Parch + 1,
    IsAlone = as.integer(FamilySize == 1)
  )
  cat("Added FamilySize (SibSp + Parch + 1) and IsAlone (FamilySize == 1)\n")
 
  # Fixed factor levels so train and test are encoded identically
  df <- df %>% mutate(
    Embarked = factor(Embarked, levels = c("C", "Q", "S")),
    Title = factor(Title, levels = c("Master", "Miss", "Mr", "Mrs", "Rare"))
  )
  cat("Converted Embarked and Title to factors (glm creates dummy variables)\n")
  df
}
 
# ------------------------------------------------------------------
section("Step 1: Load training data")
train <- read.csv(file.path(DATA_DIR, "train.csv"), stringsAsFactors = FALSE)
cat(sprintf("Loaded train.csv: %d rows, %d columns\n", nrow(train), ncol(train)))
cat("\nFirst 5 rows:\n")
print(head(train[, c("PassengerId", "Survived", "Pclass", "Sex", "Age", "Fare", "Embarked")], 5))
 
# ------------------------------------------------------------------
section("Step 2: Explore training data")
cat("Missing values per column:\n")
print(sapply(train, function(x) sum(is.na(x) | x == "")))
cat(sprintf("\nOverall survival rate: %.3f\n", mean(train$Survived)))
cat("\nSurvival rate by Sex:\n")
print(train %>% group_by(Sex) %>% summarise(survival_rate = round(mean(Survived), 3)))
cat("\nSurvival rate by Pclass:\n")
print(train %>% group_by(Pclass) %>% summarise(survival_rate = round(mean(Survived), 3)))
 
# ------------------------------------------------------------------
section("Step 3: Clean and engineer features (training data)")
# Fill values come from the TRAINING set only, then reused on the test set
embarked_values <- train$Embarked[train$Embarked != ""]
fill_values <- list(
  Age = median(train$Age, na.rm = TRUE),
  Fare = round(median(train$Fare, na.rm = TRUE), 2),
  Embarked = names(sort(table(embarked_values), decreasing = TRUE))[1]
)
cat(sprintf("Fill values computed from training data: Age = %s, Fare = %s, Embarked = %s\n",
            fill_values$Age, fill_values$Fare, fill_values$Embarked))
train_clean <- prepare(train, fill_values)
cat("\nFeatures used:", paste(FEATURES, collapse = ", "), "\n")
cat("Remaining missing values:", sum(is.na(train_clean[, FEATURES])), "\n")
 
# ------------------------------------------------------------------
section("Step 4: Train logistic regression model")
formula <- as.formula(paste("Survived ~", paste(FEATURES, collapse = " + ")))
cat("Model formula:", deparse(formula, width.cutoff = 200), "\n")
model <- glm(formula, data = train_clean, family = binomial)
cat("Model trained.\n")
cat("\nCoefficients (positive = higher chance of survival):\n")
coefs <- coef(model)
print(round(sort(coefs, decreasing = TRUE), 3))
 
# ------------------------------------------------------------------
section("Step 5: Accuracy on training set")
train_pred <- ifelse(predict(model, train_clean, type = "response") > 0.5, 1, 0)
train_acc <- mean(train_pred == train_clean$Survived)
cat(sprintf("Training accuracy: %.4f\n", train_acc))
 
# ------------------------------------------------------------------
section("Step 6: Load test data and predict")
test <- read.csv(file.path(DATA_DIR, "test.csv"), stringsAsFactors = FALSE)
cat(sprintf("Loaded test.csv: %d rows, %d columns\n", nrow(test), ncol(test)))
cat("Missing values per column:\n")
print(sapply(test, function(x) sum(is.na(x) | x == "")))
cat("\n")
test_clean <- prepare(test, fill_values)
test_clean$Predicted <- ifelse(predict(model, test_clean, type = "response") > 0.5, 1, 0)
cat(sprintf("\nPredicted survivors: %d of %d passengers\n",
            sum(test_clean$Predicted), nrow(test_clean)))
cat("\nFirst 10 predictions:\n")
print(head(test_clean[, c("PassengerId", "Predicted")], 10), row.names = FALSE)
 
# ------------------------------------------------------------------
section("Step 7: Accuracy on test set")
# test.csv has no Survived column. Kaggle's gender_submission.csv is used as
# the reference label (note: it is a baseline that predicts all women survive).
labels <- read.csv(file.path(DATA_DIR, "gender_submission.csv"))
merged <- inner_join(test_clean, labels, by = "PassengerId")
test_acc <- mean(merged$Predicted == merged$Survived)
cat("Compared predictions with gender_submission.csv\n")
cat(sprintf("Test accuracy: %.4f\n", test_acc))
 
section("Done")