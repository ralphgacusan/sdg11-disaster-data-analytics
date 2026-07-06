# ============================================================
# SDG 11 DISASTER RESILIENCE DASHBOARD
# Interactive R Shiny Application
#
# This dashboard is the front-end for the full analytics
# pipeline (data prep, feature engineering, EDA, classification,
# regression, clustering) developed in
# sdg11_disaster_data_preprocessing.R
#
# It expects "sdg11_disaster_dataset.csv" (the FINAL MODELING
# DATASET produced at the end of that script) in the same
# working directory, together with "merged_full.csv" (an
# optional richer export containing risk_class, vulnerability
# scores, municipality names etc.). If the richer file is not
# found, the app will reconstruct the needed engineered columns
# from sdg11_disaster_dataset.csv alone.
# ============================================================

# ------------------------------------------------------------
# 0. PACKAGES
# ------------------------------------------------------------
required_packages <- c(
  "shiny", "shinydashboard", "shinyWidgets", "shinycssloaders",
  "DT", "plotly", "dplyr", "tidyr", "ggplot2", "corrplot",
  "e1071", "rpart", "rpart.plot", "class", "caret",
  "cluster", "factoextra", "readr", "reshape2"
)

new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) {
  install.packages(new_packages, repos = "https://cran.rstudio.com")
}

library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(shinycssloaders)
library(DT)
library(plotly)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(e1071)
library(rpart)
library(rpart.plot)
library(class)
library(caret)
library(cluster)
library(factoextra)
library(readr)
library(reshape2)

# ------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ------------------------------------------------------------
# Helper to safely compute mode
get_mode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Try to load the model dataset produced by the preprocessing script
model_data_path <- "sdg11_disaster_dataset.csv"

if (file.exists(model_data_path)) {
  raw_data <- read_csv(model_data_path, show_col_types = FALSE)
} else {
  # Fallback synthetic dataset so the app is always runnable,
  # structured to match the columns expected by the models.
  set.seed(123)
  n <- 500
  raw_data <- data.frame(
    housing_units = sample(1000:15000, n, replace = TRUE),
    strong_ratio = runif(n, 0.05, 0.95),
    light_ratio = runif(n, 0.05, 0.9),
    salvage_ratio = runif(n, 0, 0.05),
    total_population = sample(5000:80000, n, replace = TRUE),
    hospital = sample(0:3, n, replace = TRUE),
    rural_health_unit = sample(0:5, n, replace = TRUE),
    barangay_health_station = sample(0:30, n, replace = TRUE),
    municipal_health_office = sample(0:2, n, replace = TRUE),
    birthing_home = sample(0:5, n, replace = TRUE),
    number_of_evacuation_center = sample(0:10, n, replace = TRUE),
    drr_adoption_pct = runif(n, 0, 100),
    impact_score = rgamma(n, shape = 2, scale = 2),
    risk_class = sample(c("Low", "High"), n, replace = TRUE)
  )
  # Normalize ratios so strong+light+salvage ~ 1
  total_ratio <- raw_data$strong_ratio + raw_data$light_ratio + raw_data$salvage_ratio
  raw_data$strong_ratio <- raw_data$strong_ratio / total_ratio
  raw_data$light_ratio <- raw_data$light_ratio / total_ratio
  raw_data$salvage_ratio <- raw_data$salvage_ratio / total_ratio
}

# Ensure risk_class is a factor with consistent levels
raw_data$risk_class <- factor(raw_data$risk_class, levels = c("Low", "High"))

# ------------------------------------------------------------
# 1b. VALIDATE EXPECTED COLUMNS ARE PRESENT
# ------------------------------------------------------------
# The corrected preprocessing script now exports region, province,
# municipality_city, and all real engineered features directly, so
# no reconstruction or approximation is needed here. We only check
# that the expected columns exist and fail loudly if they don't,
# rather than silently generating placeholder/synthetic values that
# could mask a real data problem.

required_cols <- c(
  "region", "province", "municipality_city",
  "housing_resilience_score", "health_access_index",
  "evacuation_capacity", "population_pressure",
  "disaster_intensity_municipal", "vulnerability_index",
  "impact_score", "risk_class"
)

missing_required <- setdiff(required_cols, names(raw_data))

if (length(missing_required) > 0 && file.exists(model_data_path)) {
  stop(
    "sdg11_disaster_dataset.csv is missing required columns: ",
    paste(missing_required, collapse = ", "),
    ". Please re-run the preprocessing script with the updated ",
    "model_df select() that includes these columns."
  )
}

# For the synthetic fallback dataset (no CSV found), generate the
# minimal placeholder columns so the app still runs for demo purposes.
if (!file.exists(model_data_path)) {
  if (!"municipality_city" %in% names(raw_data)) {
    raw_data$municipality_city <- paste0("MUNICIPALITY_", seq_len(nrow(raw_data)))
  }
  if (!"region" %in% names(raw_data)) {
    raw_data$region <- "DEMO REGION"
  }
  if (!"province" %in% names(raw_data)) {
    raw_data$province <- "DEMO PROVINCE"
  }
  
  health_cols <- c("hospital", "rural_health_unit", "barangay_health_station",
                   "municipal_health_office", "birthing_home")
  missing_health_cols <- setdiff(health_cols, names(raw_data))
  for (mc in missing_health_cols) raw_data[[mc]] <- 0
  
  raw_data <- raw_data |>
    mutate(
      health_access_index = rowSums(across(all_of(health_cols)), na.rm = TRUE),
      evacuation_capacity = ifelse(total_population > 0,
                                   (number_of_evacuation_center / total_population) * 1000, 0),
      population_pressure = ifelse(housing_units > 0,
                                   total_population / housing_units, NA),
      housing_resilience_score = strong_ratio * 1.0 + light_ratio * 0.5 + salvage_ratio * 0.0
    )
  
  if (!"disaster_intensity_municipal" %in% names(raw_data)) {
    set.seed(42)
    raw_data$disaster_intensity_municipal <- abs(rnorm(nrow(raw_data), mean = 6, sd = 2))
  }
  
  raw_data <- raw_data |>
    mutate(
      vulnerability_index = (1 - housing_resilience_score) +
        (disaster_intensity_municipal / 10000) +
        population_pressure -
        (health_access_index / (max(health_access_index, na.rm = TRUE) + 1)) -
        (evacuation_capacity / (max(evacuation_capacity, na.rm = TRUE) + 1))
    )
}

# ------------------------------------------------------------
# 2. GLOBAL CONSTANTS
# ------------------------------------------------------------
numeric_features <- c(
  "housing_resilience_score", "health_access_index", "evacuation_capacity",
  "disaster_intensity_municipal", "population_pressure",
  "vulnerability_index", "impact_score"
)
numeric_features <- numeric_features[numeric_features %in% names(raw_data)]

knn_features <- c("strong_ratio", "light_ratio", "salvage_ratio",
                  "population_pressure", "health_access_index",
                  "evacuation_capacity", "disaster_intensity_municipal",
                  "drr_adoption_pct")
knn_features <- knn_features[knn_features %in% names(raw_data)]

predictor_features <- knn_features  # same set used for classification + regression predictors

# ------------------------------------------------------------
# 3. TRAIN MODELS AT APP START
# ------------------------------------------------------------

# --- Classification data prep ---
classification_data <- raw_data |>
  select(all_of(c(knn_features, "risk_class"))) |>
  na.omit()
classification_data$risk_class <- factor(classification_data$risk_class, levels = c("Low", "High"))

set.seed(123)
train_index <- createDataPartition(classification_data$risk_class, p = 0.7, list = FALSE)
train_data <- classification_data[train_index, ]
test_data  <- classification_data[-train_index, ]

# Naive Bayes
nb_model <- naiveBayes(risk_class ~ ., data = train_data)
nb_pred  <- predict(nb_model, test_data)
nb_cm    <- confusionMatrix(nb_pred, test_data$risk_class)

# Decision Tree
dt_model <- rpart(risk_class ~ ., data = train_data, method = "class")
dt_pred  <- predict(dt_model, test_data, type = "class")
dt_cm    <- confusionMatrix(dt_pred, test_data$risk_class)

# KNN scaling
train_matrix <- as.matrix(train_data[, knn_features])
test_matrix  <- as.matrix(test_data[, knn_features])

train_center <- colMeans(train_matrix, na.rm = TRUE)
train_scale_attr <- apply(train_matrix, 2, sd, na.rm = TRUE)
train_scale_attr[train_scale_attr == 0] <- 1

train_scaled <- sweep(sweep(train_matrix, 2, train_center, "-"), 2, train_scale_attr, "/")
test_scaled  <- sweep(sweep(test_matrix, 2, train_center, "-"), 2, train_scale_attr, "/")

knn_k_default <- 5
knn_pred <- knn(train = train_scaled, test = test_scaled,
                cl = train_data$risk_class, k = knn_k_default)
knn_cm   <- confusionMatrix(knn_pred, test_data$risk_class)

# KNN accuracy across k = 1..15
k_values <- 1:15
k_accuracy <- sapply(k_values, function(k) {
  pred <- knn(train = train_scaled, test = test_scaled,
              cl = train_data$risk_class, k = k)
  mean(pred == test_data$risk_class)
})
k_results <- data.frame(k = k_values, accuracy = round(k_accuracy, 4))
best_k <- k_results$k[which.max(k_results$accuracy)]

# --- Regression data prep ---
regression_data <- raw_data |>
  select(all_of(c("impact_score", knn_features))) |>
  na.omit()

simple_lm <- lm(impact_score ~ population_pressure, data = regression_data)

mlr_formula <- as.formula(
  paste("impact_score ~", paste(knn_features, collapse = " + "))
)
multiple_lm <- lm(mlr_formula, data = regression_data)

regression_data$predicted_impact <- predict(multiple_lm, regression_data)
mlr_rmse <- sqrt(mean((regression_data$impact_score - regression_data$predicted_impact)^2))
mlr_r2 <- summary(multiple_lm)$r.squared
mlr_adj_r2 <- summary(multiple_lm)$adj.r.squared

# --- Clustering data prep ---
kmeans_features_cols <- c("housing_resilience_score", "health_access_index",
                          "evacuation_capacity", "disaster_intensity_municipal",
                          "population_pressure")
kmeans_features_cols <- kmeans_features_cols[kmeans_features_cols %in% names(raw_data)]

kmeans_data_raw <- raw_data |> select(all_of(kmeans_features_cols)) |> na.omit()
kmeans_scaled_global <- scale(kmeans_data_raw)

# ============================================================
# UI
# ============================================================
ui <- dashboardPage(
  skin = "blue",
  
  # ---------------- HEADER ----------------
  dashboardHeader(
    title = "SDG 11 Disaster Resilience Dashboard",
    titleWidth = 320
  ),
  
  # ---------------- SIDEBAR ----------------
  dashboardSidebar(
    width = 260,
    sidebarMenu(
      id = "sidebar_tabs",
      menuItem("Overview", tabName = "overview", icon = icon("gauge-high")),
      menuItem("Descriptive Stats", tabName = "descriptive", icon = icon("chart-simple")),
      menuItem("Exploratory Analysis", tabName = "eda", icon = icon("magnifying-glass-chart"),
               menuSubItem("Histograms", tabName = "eda_hist"),
               menuSubItem("Boxplots", tabName = "eda_box"),
               menuSubItem("Scatterplots", tabName = "eda_scatter"),
               menuSubItem("Correlation", tabName = "eda_corr"),
               menuSubItem("Vulnerability", tabName = "eda_vuln")),
      menuItem("Classification", tabName = "classification", icon = icon("sitemap"),
               menuSubItem("Model Evaluation", tabName = "class_eval"),
               menuSubItem("Predict Risk Class", tabName = "class_predict")),
      menuItem("Regression", tabName = "regression", icon = icon("chart-line"),
               menuSubItem("Model Evaluation", tabName = "reg_eval"),
               menuSubItem("Predict Impact Score", tabName = "reg_predict")),
      menuItem("Clustering", tabName = "clustering", icon = icon("circle-nodes"),
               menuSubItem("K-Means", tabName = "kmeans_tab"),
               menuSubItem("Hierarchical", tabName = "hclust_tab")),
      menuItem("Data Upload / Batch", tabName = "upload", icon = icon("upload"))
    )
  ),
  
  # ---------------- BODY ----------------
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side { background-color: #f4f6f9; }
        .small-box h3 { font-weight: 700; }
        .box.box-solid.box-primary>.box-header { background: #3c8dbc; }
        .skin-blue .main-header .logo { font-weight: 700; }
      "))
    ),
    
    tabItems(
      
      # ================= OVERVIEW =================
      tabItem(tabName = "overview",
              fluidRow(
                valueBoxOutput("vbox_municipalities", width = 3),
                valueBoxOutput("vbox_vuln", width = 3),
                valueBoxOutput("vbox_impact", width = 3),
                valueBoxOutput("vbox_population", width = 3)
              ),
              fluidRow(
                valueBoxOutput("vbox_resilience", width = 3),
                valueBoxOutput("vbox_health", width = 3),
                valueBoxOutput("vbox_evac", width = 3),
                valueBoxOutput("vbox_riskhigh", width = 3)
              ),
              fluidRow(
                box(title = "Vulnerability Index Distribution", status = "primary",
                    solidHeader = TRUE, width = 6,
                    withSpinner(plotlyOutput("overview_vuln_hist", height = 320))),
                box(title = "Risk Class Distribution", status = "primary",
                    solidHeader = TRUE, width = 6,
                    withSpinner(plotlyOutput("overview_risk_pie", height = 320)))
              ),
              fluidRow(
                box(title = "Dataset Overview", status = "primary", solidHeader = TRUE,
                    width = 12,
                    withSpinner(DTOutput("overview_table")))
              )
      ),
      
      # ================= DESCRIPTIVE STATS =================
      tabItem(tabName = "descriptive",
              fluidRow(
                box(title = "Select Feature", status = "primary", solidHeader = TRUE,
                    width = 12,
                    pickerInput("desc_feature", "Feature:",
                                choices = numeric_features,
                                selected = numeric_features[1],
                                options = list(`live-search` = TRUE)))
              ),
              fluidRow(
                valueBoxOutput("desc_mean", width = 3),
                valueBoxOutput("desc_median", width = 3),
                valueBoxOutput("desc_mode", width = 3),
                valueBoxOutput("desc_sd", width = 3)
              ),
              fluidRow(
                valueBoxOutput("desc_min", width = 3),
                valueBoxOutput("desc_max", width = 3),
                valueBoxOutput("desc_range", width = 3),
                valueBoxOutput("desc_variance", width = 3)
              ),
              fluidRow(
                valueBoxOutput("desc_q1", width = 4),
                valueBoxOutput("desc_q2", width = 4),
                valueBoxOutput("desc_q3", width = 4)
              ),
              fluidRow(
                box(title = "Histogram", status = "primary", solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("desc_hist", height = 300))),
                box(title = "Density Plot", status = "primary", solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("desc_density", height = 300))),
                box(title = "Boxplot", status = "primary", solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("desc_box", height = 300)))
              ),
              fluidRow(
                box(title = "Summary Table", status = "primary", solidHeader = TRUE, width = 12,
                    withSpinner(DTOutput("desc_summary_table")))
              )
      ),
      
      # ================= EDA: HISTOGRAMS =================
      tabItem(tabName = "eda_hist",
              fluidRow(
                box(title = "Histogram Controls", status = "primary", solidHeader = TRUE,
                    width = 3,
                    pickerInput("hist_var", "Variable:",
                                choices = numeric_features, selected = numeric_features[1]),
                    sliderInput("hist_bins", "Number of Bins:", min = 5, max = 60, value = 30)),
                box(title = "Histogram", status = "primary", solidHeader = TRUE, width = 9,
                    withSpinner(plotlyOutput("eda_histogram", height = 420)))
              )
      ),
      
      # ================= EDA: BOXPLOTS =================
      tabItem(tabName = "eda_box",
              fluidRow(
                box(title = "Boxplot Controls", status = "primary", solidHeader = TRUE,
                    width = 3,
                    pickerInput("box_var", "Variable (Y-axis):",
                                choices = numeric_features, selected = numeric_features[1]),
                    pickerInput("box_group", "Group By:",
                                choices = c("risk_class", "region"), selected = "risk_class")),
                box(title = "Boxplot", status = "primary", solidHeader = TRUE, width = 9,
                    withSpinner(plotlyOutput("eda_boxplot", height = 420)))
              )
      ),
      
      # ================= EDA: SCATTERPLOTS =================
      tabItem(tabName = "eda_scatter",
              fluidRow(
                box(title = "Scatterplot Controls", status = "primary", solidHeader = TRUE,
                    width = 3,
                    pickerInput("scatter_x", "X Variable:",
                                choices = numeric_features, selected = numeric_features[1]),
                    pickerInput("scatter_y", "Y Variable:",
                                choices = numeric_features, selected = numeric_features[2]),
                    switchInput("scatter_trend", "Show Trend Line", value = TRUE, onLabel = "Yes", offLabel = "No"),
                    uiOutput("scatter_corr_text")),
                box(title = "Scatterplot", status = "primary", solidHeader = TRUE, width = 9,
                    withSpinner(plotlyOutput("eda_scatterplot", height = 420)))
              )
      ),
      
      # ================= EDA: CORRELATION =================
      tabItem(tabName = "eda_corr",
              fluidRow(
                box(title = "Correlation Heatmap", status = "primary", solidHeader = TRUE,
                    width = 7,
                    withSpinner(plotlyOutput("eda_corr_heatmap", height = 480))),
                box(title = "Correlation Matrix", status = "primary", solidHeader = TRUE,
                    width = 5,
                    withSpinner(DTOutput("eda_corr_table")))
              )
      ),
      
      # ================= EDA: VULNERABILITY =================
      tabItem(tabName = "eda_vuln",
              fluidRow(
                box(title = "Top 10 Most Vulnerable Municipalities", status = "danger",
                    solidHeader = TRUE, width = 6,
                    withSpinner(plotlyOutput("vuln_top10", height = 380))),
                box(title = "Bottom 10 (Least Vulnerable) Municipalities", status = "success",
                    solidHeader = TRUE, width = 6,
                    withSpinner(plotlyOutput("vuln_bottom10", height = 380)))
              ),
              fluidRow(
                box(title = "Search / Filter Municipalities", status = "primary",
                    solidHeader = TRUE, width = 12,
                    withSpinner(DTOutput("vuln_table")))
              )
      ),
      
      # ================= CLASSIFICATION: MODEL EVALUATION =================
      tabItem(tabName = "class_eval",
              fluidRow(
                valueBoxOutput("nb_acc_box", width = 4),
                valueBoxOutput("dt_acc_box", width = 4),
                valueBoxOutput("knn_acc_box", width = 4)
              ),
              fluidRow(
                box(title = "Naive Bayes - Confusion Matrix", status = "primary",
                    solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("nb_cm_plot", height = 320))),
                box(title = "Decision Tree - Confusion Matrix", status = "primary",
                    solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("dt_cm_plot", height = 320))),
                box(title = "KNN - Confusion Matrix", status = "primary",
                    solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("knn_cm_plot", height = 320)))
              ),
              fluidRow(
                box(title = "Decision Tree Structure", status = "primary", solidHeader = TRUE,
                    width = 6,
                    withSpinner(plotOutput("dt_tree_plot", height = 380))),
                box(title = "Decision Tree - Variable Importance", status = "primary",
                    solidHeader = TRUE, width = 6,
                    withSpinner(plotlyOutput("dt_varimp_plot", height = 380)))
              ),
              fluidRow(
                box(title = "KNN Accuracy vs Number of Neighbors (k)", status = "primary",
                    solidHeader = TRUE, width = 12,
                    withSpinner(plotlyOutput("knn_k_plot", height = 360)))
              ),
              fluidRow(
                box(title = "Classification Metrics Summary", status = "primary",
                    solidHeader = TRUE, width = 12,
                    withSpinner(DTOutput("class_metrics_table")))
              )
      ),
      
      # ================= CLASSIFICATION: PREDICTION =================
      tabItem(tabName = "class_predict",
              fluidRow(
                box(title = "Input Predictor Values", status = "primary", solidHeader = TRUE,
                    width = 5,
                    pickerInput("class_model_choice", "Choose Model:",
                                choices = c("Naive Bayes", "Decision Tree", "KNN"),
                                selected = "Decision Tree"),
                    numericInput("in_strong_ratio", "Strong Ratio:", value = round(mean(raw_data$strong_ratio, na.rm = TRUE), 3), min = 0, max = 1, step = 0.01),
                    numericInput("in_light_ratio", "Light Ratio:", value = round(mean(raw_data$light_ratio, na.rm = TRUE), 3), min = 0, max = 1, step = 0.01),
                    numericInput("in_salvage_ratio", "Salvage Ratio:", value = round(mean(raw_data$salvage_ratio, na.rm = TRUE), 3), min = 0, max = 1, step = 0.01),
                    numericInput("in_population_pressure", "Population Pressure:", value = round(mean(raw_data$population_pressure, na.rm = TRUE), 2), min = 0, step = 0.1),
                    numericInput("in_health_access_index", "Health Access Index:", value = round(mean(raw_data$health_access_index, na.rm = TRUE), 1), min = 0, step = 1),
                    numericInput("in_evacuation_capacity", "Evacuation Capacity:", value = round(mean(raw_data$evacuation_capacity, na.rm = TRUE), 3), min = 0, step = 0.01),
                    numericInput("in_disaster_intensity_municipal", "Disaster Intensity (Municipal):", value = round(mean(raw_data$disaster_intensity_municipal, na.rm = TRUE), 2), min = 0, step = 0.1),
                    numericInput("in_drr_adoption_pct", "DRR Adoption (%):", value = round(mean(raw_data$drr_adoption_pct, na.rm = TRUE), 1), min = 0, max = 100, step = 1),
                    actionButton("predict_risk_btn", "Predict Risk Class",
                                 class = "btn-primary btn-block", icon = icon("bolt"))
                ),
                box(title = "Prediction Result", status = "warning", solidHeader = TRUE,
                    width = 7,
                    uiOutput("risk_prediction_card"),
                    withSpinner(plotlyOutput("risk_prediction_prob_plot", height = 300))
                )
              )
      ),
      
      # ================= REGRESSION: EVALUATION =================
      tabItem(tabName = "reg_eval",
              fluidRow(
                valueBoxOutput("reg_r2_box", width = 4),
                valueBoxOutput("reg_adjr2_box", width = 4),
                valueBoxOutput("reg_rmse_box", width = 4)
              ),
              fluidRow(
                box(title = "Simple Linear Regression: Impact Score ~ Population Pressure",
                    status = "primary", solidHeader = TRUE, width = 6,
                    withSpinner(plotlyOutput("simple_lm_plot", height = 360))),
                box(title = "Multiple Regression: Actual vs Predicted", status = "primary",
                    solidHeader = TRUE, width = 6,
                    withSpinner(plotlyOutput("mlr_avp_plot", height = 360)))
              ),
              fluidRow(
                box(title = "Residual Plot (Multiple Regression)", status = "primary",
                    solidHeader = TRUE, width = 6,
                    withSpinner(plotlyOutput("mlr_resid_plot", height = 360))),
                box(title = "Regression Coefficients (Multiple Regression)", status = "primary",
                    solidHeader = TRUE, width = 6,
                    withSpinner(DTOutput("mlr_coef_table")))
              )
      ),
      
      # ================= REGRESSION: PREDICTION =================
      tabItem(tabName = "reg_predict",
              fluidRow(
                box(title = "Input Predictor Values", status = "primary", solidHeader = TRUE,
                    width = 5,
                    numericInput("reg_strong_ratio", "Strong Ratio:", value = round(mean(raw_data$strong_ratio, na.rm = TRUE), 3), min = 0, max = 1, step = 0.01),
                    numericInput("reg_light_ratio", "Light Ratio:", value = round(mean(raw_data$light_ratio, na.rm = TRUE), 3), min = 0, max = 1, step = 0.01),
                    numericInput("reg_salvage_ratio", "Salvage Ratio:", value = round(mean(raw_data$salvage_ratio, na.rm = TRUE), 3), min = 0, max = 1, step = 0.01),
                    numericInput("reg_population_pressure", "Population Pressure:", value = round(mean(raw_data$population_pressure, na.rm = TRUE), 2), min = 0, step = 0.1),
                    numericInput("reg_health_access_index", "Health Access Index:", value = round(mean(raw_data$health_access_index, na.rm = TRUE), 1), min = 0, step = 1),
                    numericInput("reg_evacuation_capacity", "Evacuation Capacity:", value = round(mean(raw_data$evacuation_capacity, na.rm = TRUE), 3), min = 0, step = 0.01),
                    numericInput("reg_disaster_intensity_municipal", "Disaster Intensity (Municipal):", value = round(mean(raw_data$disaster_intensity_municipal, na.rm = TRUE), 2), min = 0, step = 0.1),
                    numericInput("reg_drr_adoption_pct", "DRR Adoption (%):", value = round(mean(raw_data$drr_adoption_pct, na.rm = TRUE), 1), min = 0, max = 100, step = 1),
                    actionButton("predict_impact_btn", "Predict Impact Score",
                                 class = "btn-primary btn-block", icon = icon("bolt"))
                ),
                box(title = "Prediction Result", status = "warning", solidHeader = TRUE,
                    width = 7,
                    uiOutput("impact_prediction_card"),
                    withSpinner(plotlyOutput("impact_gauge_plot", height = 300))
                )
              )
      ),
      
      # ================= CLUSTERING: K-MEANS =================
      tabItem(tabName = "kmeans_tab",
              fluidRow(
                box(title = "Controls", status = "primary", solidHeader = TRUE, width = 3,
                    sliderInput("kmeans_k", "Number of Clusters (K):", min = 2, max = 8, value = 3),
                    actionButton("run_kmeans_btn", "Run K-Means", class = "btn-primary btn-block", icon = icon("rotate"))),
                box(title = "Elbow Method", status = "primary", solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("elbow_plot", height = 340))),
                box(title = "Cluster Visualization", status = "primary", solidHeader = TRUE, width = 5,
                    withSpinner(plotlyOutput("kmeans_cluster_plot", height = 340)))
              ),
              fluidRow(
                box(title = "Cluster Sizes", status = "primary", solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("kmeans_sizes_plot", height = 320))),
                box(title = "Cluster Profiles", status = "primary", solidHeader = TRUE, width = 8,
                    withSpinner(DTOutput("kmeans_profile_table")))
              )
      ),
      
      # ================= CLUSTERING: HIERARCHICAL =================
      tabItem(tabName = "hclust_tab",
              fluidRow(
                box(title = "Controls", status = "primary", solidHeader = TRUE, width = 3,
                    sliderInput("hclust_k", "Number of Clusters:", min = 2, max = 8, value = 3),
                    actionButton("run_hclust_btn", "Update Clustering", class = "btn-primary btn-block", icon = icon("rotate"))),
                box(title = "Dendrogram", status = "primary", solidHeader = TRUE, width = 9,
                    withSpinner(plotOutput("hclust_dendro_plot", height = 400)))
              ),
              fluidRow(
                box(title = "Cluster Distribution", status = "primary", solidHeader = TRUE, width = 4,
                    withSpinner(plotlyOutput("hclust_dist_plot", height = 320))),
                box(title = "Cluster Membership Table", status = "primary", solidHeader = TRUE, width = 8,
                    withSpinner(DTOutput("hclust_membership_table")))
              )
      ),
      
      # ================= UPLOAD / BATCH PREDICTION =================
      tabItem(tabName = "upload",
              fluidRow(
                box(title = "Upload CSV Dataset", status = "primary", solidHeader = TRUE,
                    width = 12,
                    fileInput("upload_file", "Choose CSV File", accept = ".csv"),
                    helpText("Uploaded data should contain at least the following columns: ",
                             paste(knn_features, collapse = ", "), " (and 'impact_score' / 'risk_class' if available).")
                )
              ),
              fluidRow(
                box(title = "Data Preview", status = "primary", solidHeader = TRUE, width = 12,
                    withSpinner(DTOutput("upload_preview_table")))
              ),
              fluidRow(
                box(title = "Missing Value Summary", status = "primary", solidHeader = TRUE, width = 6,
                    withSpinner(DTOutput("upload_missing_table"))),
                box(title = "Data Structure Summary", status = "primary", solidHeader = TRUE, width = 6,
                    withSpinner(verbatimTextOutput("upload_structure_summary")))
              ),
              fluidRow(
                box(title = "Batch Classification", status = "primary", solidHeader = TRUE, width = 6,
                    pickerInput("batch_class_model", "Model:",
                                choices = c("Naive Bayes", "Decision Tree", "KNN"),
                                selected = "Decision Tree"),
                    actionButton("run_batch_class_btn", "Run Classification on Uploaded Data",
                                 class = "btn-primary btn-block", icon = icon("play")),
                    br(), br(),
                    withSpinner(DTOutput("batch_class_table")),
                    downloadButton("download_batch_class", "Download Classification Results")
                ),
                box(title = "Batch Regression (Impact Score)", status = "primary", solidHeader = TRUE, width = 6,
                    actionButton("run_batch_reg_btn", "Run Regression on Uploaded Data",
                                 class = "btn-primary btn-block", icon = icon("play")),
                    br(), br(),
                    withSpinner(DTOutput("batch_reg_table")),
                    downloadButton("download_batch_reg", "Download Regression Results")
                )
              )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================
server <- function(input, output, session) {
  
  # ----------------------------------------------------------
  # Helper: descriptive statistics for a numeric vector
  # ----------------------------------------------------------
  compute_desc_stats <- function(x) {
    x <- x[!is.na(x)]
    list(
      mean = mean(x), median = median(x), mode = get_mode(x),
      min = min(x), max = max(x), range = diff(range(x)),
      q1 = quantile(x, 0.25), q2 = quantile(x, 0.50), q3 = quantile(x, 0.75),
      iqr = IQR(x), sd = sd(x), variance = var(x)
    )
  }
  
  # ============================================================
  # A. OVERVIEW
  # ============================================================
  output$vbox_municipalities <- renderValueBox({
    valueBox(format(nrow(raw_data), big.mark = ","), "Total Municipalities",
             icon = icon("city"), color = "blue")
  })
  
  output$vbox_vuln <- renderValueBox({
    valueBox(round(mean(raw_data$vulnerability_index, na.rm = TRUE), 3),
             "Avg Vulnerability Index", icon = icon("triangle-exclamation"), color = "red")
  })
  
  output$vbox_impact <- renderValueBox({
    valueBox(round(mean(raw_data$impact_score, na.rm = TRUE), 3),
             "Avg Impact Score", icon = icon("burst"), color = "orange")
  })
  
  output$vbox_population <- renderValueBox({
    valueBox(format(round(sum(raw_data$total_population, na.rm = TRUE)), big.mark = ","),
             "Total Population", icon = icon("users"), color = "purple")
  })
  
  output$vbox_resilience <- renderValueBox({
    valueBox(round(mean(raw_data$housing_resilience_score, na.rm = TRUE), 3),
             "Avg Housing Resilience", icon = icon("house-chimney"), color = "green")
  })
  
  output$vbox_health <- renderValueBox({
    valueBox(round(mean(raw_data$health_access_index, na.rm = TRUE), 1),
             "Avg Health Access Index", icon = icon("hospital"), color = "teal")
  })
  
  output$vbox_evac <- renderValueBox({
    valueBox(round(mean(raw_data$evacuation_capacity, na.rm = TRUE), 3),
             "Avg Evacuation Capacity", icon = icon("person-shelter"), color = "yellow")
  })
  
  output$vbox_riskhigh <- renderValueBox({
    pct_high <- round(mean(raw_data$risk_class == "High", na.rm = TRUE) * 100, 1)
    valueBox(paste0(pct_high, "%"), "High Risk Municipalities", icon = icon("skull-crossbones"), color = "maroon")
  })
  
  output$overview_vuln_hist <- renderPlotly({
    p <- ggplot(raw_data, aes(x = vulnerability_index)) +
      geom_histogram(bins = 30, fill = "#3c8dbc", color = "white") +
      labs(x = "Vulnerability Index", y = "Frequency") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$overview_risk_pie <- renderPlotly({
    risk_counts <- raw_data |> count(risk_class)
    plot_ly(risk_counts, labels = ~risk_class, values = ~n, type = "pie",
            marker = list(colors = c("#00a65a", "#dd4b39"))) |>
      layout(title = "")
  })
  
  output$overview_table <- renderDT({
    df <- raw_data |>
      select(region, province, municipality_city, risk_class,
             vulnerability_index, impact_score, everything())
    datatable(df, options = list(scrollX = TRUE, pageLength = 10))
  })
  
  # ============================================================
  # B. DESCRIPTIVE STATISTICS
  # ============================================================
  desc_stats <- reactive({
    compute_desc_stats(raw_data[[input$desc_feature]])
  })
  
  output$desc_mean <- renderValueBox({
    valueBox(round(desc_stats()$mean, 4), "Mean", icon = icon("calculator"), color = "blue")
  })
  output$desc_median <- renderValueBox({
    valueBox(round(desc_stats()$median, 4), "Median", icon = icon("sort"), color = "light-blue")
  })
  output$desc_mode <- renderValueBox({
    valueBox(round(desc_stats()$mode, 4), "Mode", icon = icon("star"), color = "navy")
  })
  output$desc_sd <- renderValueBox({
    valueBox(round(desc_stats()$sd, 4), "Std. Deviation", icon = icon("arrows-left-right"), color = "purple")
  })
  output$desc_min <- renderValueBox({
    valueBox(round(desc_stats()$min, 4), "Minimum", icon = icon("arrow-down"), color = "green")
  })
  output$desc_max <- renderValueBox({
    valueBox(round(desc_stats()$max, 4), "Maximum", icon = icon("arrow-up"), color = "red")
  })
  output$desc_range <- renderValueBox({
    valueBox(round(desc_stats()$range, 4), "Range", icon = icon("ruler-horizontal"), color = "orange")
  })
  output$desc_variance <- renderValueBox({
    valueBox(round(desc_stats()$variance, 4), "Variance", icon = icon("chart-area"), color = "maroon")
  })
  output$desc_q1 <- renderValueBox({
    valueBox(round(desc_stats()$q1, 4), "Q1 (25th Percentile)", icon = icon("percent"), color = "teal")
  })
  output$desc_q2 <- renderValueBox({
    valueBox(round(desc_stats()$q2, 4), "Q2 / Median (50th Percentile)", icon = icon("percent"), color = "teal")
  })
  output$desc_q3 <- renderValueBox({
    valueBox(round(desc_stats()$q3, 4), "Q3 (75th Percentile)", icon = icon("percent"), color = "teal")
  })
  
  output$desc_hist <- renderPlotly({
    feat <- input$desc_feature
    p <- ggplot(raw_data, aes(x = .data[[feat]])) +
      geom_histogram(bins = 30, fill = "#3c8dbc", color = "white") +
      labs(x = feat, y = "Frequency") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$desc_density <- renderPlotly({
    feat <- input$desc_feature
    p <- ggplot(raw_data, aes(x = .data[[feat]])) +
      geom_density(fill = "#00a65a", alpha = 0.5) +
      labs(x = feat, y = "Density") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$desc_box <- renderPlotly({
    feat <- input$desc_feature
    p <- ggplot(raw_data, aes(y = .data[[feat]])) +
      geom_boxplot(fill = "#f39c12") +
      labs(y = feat) +
      theme_minimal()
    ggplotly(p)
  })
  
  output$desc_summary_table <- renderDT({
    s <- desc_stats()
    df <- data.frame(
      Statistic = c("Mean", "Median", "Mode", "Minimum", "Maximum", "Range",
                    "Q1", "Q2 (Median)", "Q3", "IQR", "Standard Deviation", "Variance"),
      Value = round(c(s$mean, s$median, s$mode, s$min, s$max, s$range,
                      s$q1, s$q2, s$q3, s$iqr, s$sd, s$variance), 4)
    )
    datatable(df, options = list(dom = 't', paging = FALSE))
  })
  
  # ============================================================
  # C. EDA - HISTOGRAMS
  # ============================================================
  output$eda_histogram <- renderPlotly({
    var <- input$hist_var
    p <- ggplot(raw_data, aes(x = .data[[var]])) +
      geom_histogram(bins = input$hist_bins, fill = "#3c8dbc", color = "white") +
      labs(title = paste("Histogram of", var), x = var, y = "Frequency") +
      theme_minimal()
    ggplotly(p)
  })
  
  # ============================================================
  # D. EDA - BOXPLOTS
  # ============================================================
  output$eda_boxplot <- renderPlotly({
    var <- input$box_var
    grp <- input$box_group
    p <- ggplot(raw_data, aes(x = .data[[grp]], y = .data[[var]], fill = .data[[grp]])) +
      geom_boxplot() +
      labs(title = paste("Boxplot of", var, "by", grp), x = grp, y = var) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    if (grp == "risk_class") {
      p <- p + scale_fill_manual(values = c("Low" = "#00a65a", "High" = "#dd4b39"))
    } else {
      p <- p + theme(legend.position = "none")
    }
    ggplotly(p)
  })
  
  # ============================================================
  # E. EDA - SCATTERPLOTS
  # ============================================================
  scatter_correlation <- reactive({
    x <- raw_data[[input$scatter_x]]
    y <- raw_data[[input$scatter_y]]
    cor(x, y, use = "complete.obs")
  })
  
  output$scatter_corr_text <- renderUI({
    corr_val <- round(scatter_correlation(), 4)
    tags$div(
      style = "margin-top: 15px; padding: 10px; background-color: #ecf0f5; border-radius: 5px;",
      tags$b("Correlation Coefficient: "), tags$span(corr_val)
    )
  })
  
  output$eda_scatterplot <- renderPlotly({
    xv <- input$scatter_x
    yv <- input$scatter_y
    p <- ggplot(raw_data, aes(x = .data[[xv]], y = .data[[yv]])) +
      geom_point(color = "#3c8dbc", alpha = 0.5) +
      labs(title = paste(yv, "vs", xv), x = xv, y = yv) +
      theme_minimal()
    if (isTRUE(input$scatter_trend)) {
      p <- p + geom_smooth(method = "lm", color = "red", se = FALSE)
    }
    ggplotly(p)
  })
  
  # ============================================================
  # F. EDA - CORRELATION
  # ============================================================
  corr_matrix_reactive <- reactive({
    cols <- numeric_features
    cor(raw_data[cols], use = "complete.obs")
  })
  
  output$eda_corr_heatmap <- renderPlotly({
    cm <- corr_matrix_reactive()
    cm_melt <- melt(cm)
    p <- ggplot(cm_melt, aes(x = Var1, y = Var2, fill = value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = round(value, 2)), size = 3) +
      scale_fill_gradient2(low = "#3c8dbc", mid = "white", high = "#dd4b39",
                           midpoint = 0, limit = c(-1, 1)) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "", y = "", fill = "Correlation")
    ggplotly(p)
  })
  
  output$eda_corr_table <- renderDT({
    cm <- round(corr_matrix_reactive(), 4)
    datatable(as.data.frame(cm), options = list(scrollX = TRUE, dom = 't', paging = FALSE))
  })
  
  # ============================================================
  # G. EDA - VULNERABILITY
  # ============================================================
  output$vuln_top10 <- renderPlotly({
    top10 <- raw_data |> arrange(desc(vulnerability_index)) |> head(10)
    p <- ggplot(top10, aes(x = reorder(municipality_city, vulnerability_index),
                           y = vulnerability_index)) +
      geom_col(fill = "#dd4b39") +
      coord_flip() +
      labs(x = "Municipality", y = "Vulnerability Index") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$vuln_bottom10 <- renderPlotly({
    bottom10 <- raw_data |> arrange(vulnerability_index) |> head(10)
    p <- ggplot(bottom10, aes(x = reorder(municipality_city, -vulnerability_index),
                              y = vulnerability_index)) +
      geom_col(fill = "#00a65a") +
      coord_flip() +
      labs(x = "Municipality", y = "Vulnerability Index") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$vuln_table <- renderDT({
    df <- raw_data |>
      select(region, province, municipality_city, vulnerability_index, impact_score,
             housing_resilience_score, health_access_index,
             evacuation_capacity, disaster_intensity_municipal,
             population_pressure, risk_class) |>
      arrange(desc(vulnerability_index))
    datatable(df, filter = "top", options = list(scrollX = TRUE, pageLength = 10))
  })
  
  # ============================================================
  # H. CLASSIFICATION - MODEL EVALUATION
  # ============================================================
  output$nb_acc_box <- renderValueBox({
    valueBox(paste0(round(nb_cm$overall["Accuracy"] * 100, 2), "%"),
             "Naive Bayes Accuracy", icon = icon("brain"), color = "blue")
  })
  output$dt_acc_box <- renderValueBox({
    valueBox(paste0(round(dt_cm$overall["Accuracy"] * 100, 2), "%"),
             "Decision Tree Accuracy", icon = icon("sitemap"), color = "green")
  })
  output$knn_acc_box <- renderValueBox({
    valueBox(paste0(round(knn_cm$overall["Accuracy"] * 100, 2), "%"),
             paste0("KNN Accuracy (k=", knn_k_default, ")"), icon = icon("circle-nodes"), color = "purple")
  })
  
  plot_confusion_matrix <- function(cm, title) {
    cm_table <- as.data.frame(cm$table)
    p <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), size = 5, color = "black") +
      scale_fill_gradient(low = "#ecf0f5", high = "#3c8dbc") +
      labs(title = title, x = "Actual", y = "Predicted") +
      theme_minimal()
    ggplotly(p)
  }
  
  output$nb_cm_plot <- renderPlotly({
    plot_confusion_matrix(nb_cm, "Naive Bayes")
  })
  output$dt_cm_plot <- renderPlotly({
    plot_confusion_matrix(dt_cm, "Decision Tree")
  })
  output$knn_cm_plot <- renderPlotly({
    plot_confusion_matrix(knn_cm, "KNN")
  })
  
  output$dt_tree_plot <- renderPlot({
    rpart.plot(dt_model, type = 3, extra = 104, fallen.leaves = TRUE,
               main = "Decision Tree for Risk Classification")
  })
  
  output$dt_varimp_plot <- renderPlotly({
    vi <- dt_model$variable.importance
    vi_df <- data.frame(Variable = names(vi), Importance = as.numeric(vi)) |>
      arrange(Importance)
    p <- ggplot(vi_df, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "#00a65a") +
      coord_flip() +
      labs(x = "Variable", y = "Importance") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$knn_k_plot <- renderPlotly({
    p <- ggplot(k_results, aes(x = k, y = accuracy)) +
      geom_line(color = "#3c8dbc") +
      geom_point(color = "#001f3f", size = 2) +
      geom_vline(xintercept = best_k, linetype = "dashed", color = "red") +
      labs(title = paste("Best k =", best_k), x = "k (Number of Neighbors)", y = "Accuracy") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$class_metrics_table <- renderDT({
    extract_metrics <- function(cm, model_name) {
      data.frame(
        Model = model_name,
        Accuracy = round(cm$overall["Accuracy"], 4),
        Kappa = round(cm$overall["Kappa"], 4),
        Sensitivity = round(cm$byClass["Sensitivity"], 4),
        Specificity = round(cm$byClass["Specificity"], 4),
        Precision = round(cm$byClass["Pos Pred Value"], 4),
        Recall = round(cm$byClass["Sensitivity"], 4),
        F1 = round(2 * (cm$byClass["Pos Pred Value"] * cm$byClass["Sensitivity"]) /
                     (cm$byClass["Pos Pred Value"] + cm$byClass["Sensitivity"]), 4)
      )
    }
    df <- rbind(
      extract_metrics(nb_cm, "Naive Bayes"),
      extract_metrics(dt_cm, "Decision Tree"),
      extract_metrics(knn_cm, paste0("KNN (k=", knn_k_default, ")"))
    )
    rownames(df) <- NULL
    datatable(df, options = list(dom = 't', paging = FALSE))
  })
  
  # ============================================================
  # I. CLASSIFICATION - PREDICTION
  # ============================================================
  risk_prediction <- eventReactive(input$predict_risk_btn, {
    new_obs <- data.frame(
      strong_ratio = input$in_strong_ratio,
      light_ratio = input$in_light_ratio,
      salvage_ratio = input$in_salvage_ratio,
      population_pressure = input$in_population_pressure,
      health_access_index = input$in_health_access_index,
      evacuation_capacity = input$in_evacuation_capacity,
      disaster_intensity_municipal = input$in_disaster_intensity_municipal,
      drr_adoption_pct = input$in_drr_adoption_pct
    )
    
    model_choice <- input$class_model_choice
    
    if (model_choice == "Naive Bayes") {
      pred_class <- predict(nb_model, new_obs)
      pred_prob  <- predict(nb_model, new_obs, type = "raw")
      probs <- as.numeric(pred_prob[1, ])
      names(probs) <- colnames(pred_prob)
    } else if (model_choice == "Decision Tree") {
      pred_class <- predict(dt_model, new_obs, type = "class")
      pred_prob  <- predict(dt_model, new_obs, type = "prob")
      probs <- as.numeric(pred_prob[1, ])
      names(probs) <- colnames(pred_prob)
    } else { # KNN
      new_matrix <- as.matrix(new_obs[, knn_features])
      new_scaled <- sweep(sweep(new_matrix, 2, train_center, "-"), 2, train_scale_attr, "/")
      pred_class <- knn(train = train_scaled, test = new_scaled,
                        cl = train_data$risk_class, k = knn_k_default, prob = TRUE)
      p_win <- attr(pred_class, "prob")
      # Construct two-class probability vector
      if (as.character(pred_class) == "Low") {
        probs <- c(Low = p_win, High = 1 - p_win)
      } else {
        probs <- c(Low = 1 - p_win, High = p_win)
      }
    }
    
    list(class = as.character(pred_class), probs = probs, model = model_choice)
  })
  
  output$risk_prediction_card <- renderUI({
    if (is.null(input$predict_risk_btn) || input$predict_risk_btn == 0) {
      return(tags$div(style = "padding: 20px; text-align:center; color:#999;",
                      "Enter values and click 'Predict Risk Class' to see results."))
    }
    res <- risk_prediction()
    color <- if (res$class == "High") "#dd4b39" else "#00a65a"
    label <- if (res$class == "High") "HIGH RISK" else "LOW RISK"
    conf <- round(max(res$probs) * 100, 1)
    
    tags$div(
      style = paste0("padding: 20px; background-color: ", color,
                     "; color: white; border-radius: 8px; text-align: center; margin-bottom: 15px;"),
      tags$h2(label, style = "margin: 0;"),
      tags$p(paste("Model used:", res$model)),
      tags$h3(paste0("Confidence: ", conf, "%"))
    )
  })
  
  output$risk_prediction_prob_plot <- renderPlotly({
    if (is.null(input$predict_risk_btn) || input$predict_risk_btn == 0) {
      return(plotly_empty(type = "bar"))
    }
    res <- risk_prediction()
    df <- data.frame(Class = names(res$probs), Probability = as.numeric(res$probs))
    p <- ggplot(df, aes(x = Class, y = Probability, fill = Class)) +
      geom_col() +
      scale_fill_manual(values = c("Low" = "#00a65a", "High" = "#dd4b39")) +
      labs(title = "Predicted Class Probabilities", y = "Probability") +
      ylim(0, 1) +
      theme_minimal()
    ggplotly(p)
  })
  
  # ============================================================
  # J. REGRESSION - MODEL EVALUATION
  # ============================================================
  output$reg_r2_box <- renderValueBox({
    valueBox(round(mlr_r2, 4), "R-squared (Multiple LR)", icon = icon("square-root-variable"), color = "blue")
  })
  output$reg_adjr2_box <- renderValueBox({
    valueBox(round(mlr_adj_r2, 4), "Adjusted R-squared", icon = icon("chart-line"), color = "green")
  })
  output$reg_rmse_box <- renderValueBox({
    valueBox(round(mlr_rmse, 4), "RMSE", icon = icon("ruler"), color = "orange")
  })
  
  output$simple_lm_plot <- renderPlotly({
    p <- ggplot(regression_data, aes(x = population_pressure, y = impact_score)) +
      geom_point(color = "#3c8dbc", alpha = 0.5) +
      geom_smooth(method = "lm", color = "red", se = TRUE) +
      labs(x = "Population Pressure", y = "Impact Score") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$mlr_avp_plot <- renderPlotly({
    p <- ggplot(regression_data, aes(x = impact_score, y = predicted_impact)) +
      geom_point(color = "#00a65a", alpha = 0.5) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      labs(x = "Actual Impact Score", y = "Predicted Impact Score") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$mlr_resid_plot <- renderPlotly({
    resid_df <- data.frame(
      fitted = fitted(multiple_lm),
      residuals = resid(multiple_lm)
    )
    p <- ggplot(resid_df, aes(x = fitted, y = residuals)) +
      geom_point(color = "#f39c12", alpha = 0.5) +
      geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
      labs(x = "Fitted Values", y = "Residuals") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$mlr_coef_table <- renderDT({
    coef_df <- as.data.frame(summary(multiple_lm)$coefficients)
    coef_df <- round(coef_df, 4)
    coef_df$Term <- rownames(coef_df)
    coef_df <- coef_df[, c("Term", "Estimate", "Std. Error", "t value", "Pr(>|t|)")]
    rownames(coef_df) <- NULL
    datatable(coef_df, options = list(dom = 't', paging = FALSE))
  })
  
  # ============================================================
  # K. REGRESSION - PREDICTION
  # ============================================================
  impact_prediction <- eventReactive(input$predict_impact_btn, {
    new_obs <- data.frame(
      strong_ratio = input$reg_strong_ratio,
      light_ratio = input$reg_light_ratio,
      salvage_ratio = input$reg_salvage_ratio,
      population_pressure = input$reg_population_pressure,
      health_access_index = input$reg_health_access_index,
      evacuation_capacity = input$reg_evacuation_capacity,
      disaster_intensity_municipal = input$reg_disaster_intensity_municipal,
      drr_adoption_pct = input$reg_drr_adoption_pct
    )
    pred <- predict(multiple_lm, new_obs, interval = "prediction", level = 0.95)
    list(fit = pred[1, "fit"], lwr = pred[1, "lwr"], upr = pred[1, "upr"])
  })
  
  output$impact_prediction_card <- renderUI({
    if (is.null(input$predict_impact_btn) || input$predict_impact_btn == 0) {
      return(tags$div(style = "padding: 20px; text-align:center; color:#999;",
                      "Enter values and click 'Predict Impact Score' to see results."))
    }
    res <- impact_prediction()
    tags$div(
      style = "padding: 20px; background-color: #f39c12; color: white; border-radius: 8px; text-align: center; margin-bottom: 15px;",
      tags$h2(paste("Predicted Impact Score:", round(res$fit, 3)), style = "margin: 0;"),
      tags$p(paste0("95% Prediction Interval: [", round(res$lwr, 3), ", ", round(res$upr, 3), "]"))
    )
  })
  
  output$impact_gauge_plot <- renderPlotly({
    if (is.null(input$predict_impact_btn) || input$predict_impact_btn == 0) {
      return(plotly_empty())
    }
    res <- impact_prediction()
    max_scale <- max(regression_data$impact_score, na.rm = TRUE) * 1.1
    
    plot_ly(
      type = "indicator",
      mode = "gauge+number",
      value = round(res$fit, 3),
      gauge = list(
        axis = list(range = list(0, max_scale)),
        bar = list(color = "#f39c12"),
        steps = list(
          list(range = c(0, max_scale * 0.33), color = "#dff0d8"),
          list(range = c(max_scale * 0.33, max_scale * 0.66), color = "#fcf8e3"),
          list(range = c(max_scale * 0.66, max_scale), color = "#f2dede")
        )
      ),
      title = list(text = "Predicted Impact Score")
    )
  })
  
  # ============================================================
  # L. CLUSTERING - K-MEANS
  # ============================================================
  output$elbow_plot <- renderPlotly({
    wss <- sapply(1:8, function(k) {
      kmeans(kmeans_scaled_global, centers = k, nstart = 10)$tot.withinss
    })
    df <- data.frame(k = 1:8, wss = wss)
    p <- ggplot(df, aes(x = k, y = wss)) +
      geom_line(color = "#3c8dbc") +
      geom_point(color = "#001f3f", size = 2) +
      labs(title = "Elbow Method", x = "Number of Clusters (K)", y = "Total Within-Cluster Sum of Squares") +
      theme_minimal()
    ggplotly(p)
  })
  
  kmeans_result <- eventReactive(input$run_kmeans_btn, {
    set.seed(123)
    kmeans(kmeans_scaled_global, centers = input$kmeans_k, nstart = 25)
  }, ignoreNULL = FALSE)
  
  output$kmeans_cluster_plot <- renderPlotly({
    km <- kmeans_result()
    pca <- prcomp(kmeans_scaled_global)
    pc_df <- as.data.frame(pca$x[, 1:2])
    pc_df$cluster <- factor(km$cluster)
    
    p <- ggplot(pc_df, aes(x = PC1, y = PC2, color = cluster)) +
      geom_point(alpha = 0.6) +
      labs(title = paste("K-Means Clustering (k =", input$kmeans_k, ")"),
           x = "PC1", y = "PC2") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$kmeans_sizes_plot <- renderPlotly({
    km <- kmeans_result()
    df <- as.data.frame(table(km$cluster))
    names(df) <- c("Cluster", "Count")
    p <- ggplot(df, aes(x = Cluster, y = Count, fill = Cluster)) +
      geom_col() +
      labs(title = "Cluster Sizes", x = "Cluster", y = "Number of Municipalities") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$kmeans_profile_table <- renderDT({
    km <- kmeans_result()
    merged_clustered <- raw_data |>
      select(all_of(kmeans_features_cols), vulnerability_index) |>
      na.omit()
    merged_clustered$cluster <- km$cluster
    
    profile <- merged_clustered |>
      group_by(cluster) |>
      summarise(
        count = n(),
        across(all_of(c(kmeans_features_cols, "vulnerability_index")), ~round(mean(.x), 4)),
        .groups = "drop"
      )
    datatable(profile, options = list(dom = 't', paging = FALSE, scrollX = TRUE))
  })
  
  # ============================================================
  # M. CLUSTERING - HIERARCHICAL
  # ============================================================
  hclust_result <- reactive({
    dist_matrix <- dist(kmeans_scaled_global, method = "euclidean")
    hclust(dist_matrix, method = "ward.D2")
  })
  
  hclust_cut <- eventReactive(input$run_hclust_btn, {
    cutree(hclust_result(), k = input$hclust_k)
  }, ignoreNULL = FALSE)
  
  output$hclust_dendro_plot <- renderPlot({
    hc <- hclust_result()
    k <- if (is.null(input$hclust_k)) 3 else input$hclust_k
    plot(hc, main = "Hierarchical Clustering Dendrogram",
         xlab = "Municipalities", ylab = "Height", labels = FALSE)
    rect.hclust(hc, k = k, border = "red")
  })
  
  output$hclust_dist_plot <- renderPlotly({
    clusters <- hclust_cut()
    df <- as.data.frame(table(clusters))
    names(df) <- c("Cluster", "Count")
    p <- ggplot(df, aes(x = Cluster, y = Count, fill = Cluster)) +
      geom_col() +
      labs(title = "Hierarchical Cluster Sizes", x = "Cluster", y = "Number of Municipalities") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$hclust_membership_table <- renderDT({
    clusters <- hclust_cut()
    municip_data <- raw_data |>
      select(region, province, municipality_city, all_of(kmeans_features_cols), vulnerability_index) |>
      na.omit()
    municip_data$cluster <- clusters
    municip_data <- municip_data |>
      select(region, province, municipality_city, cluster, everything())
    datatable(municip_data, filter = "top", options = list(scrollX = TRUE, pageLength = 10))
  })
  
  # ============================================================
  # N. DATA UPLOAD / BATCH PREDICTION
  # ============================================================
  uploaded_data <- reactive({
    req(input$upload_file)
    read_csv(input$upload_file$datapath, show_col_types = FALSE)
  })
  
  output$upload_preview_table <- renderDT({
    req(uploaded_data())
    datatable(uploaded_data(), options = list(scrollX = TRUE, pageLength = 10))
  })
  
  output$upload_missing_table <- renderDT({
    req(uploaded_data())
    df <- uploaded_data()
    missing_df <- data.frame(
      Column = names(df),
      Missing_Count = colSums(is.na(df)),
      Missing_Pct = round(colSums(is.na(df)) / nrow(df) * 100, 2)
    )
    rownames(missing_df) <- NULL
    datatable(missing_df, options = list(dom = 't', paging = FALSE))
  })
  
  output$upload_structure_summary <- renderPrint({
    req(uploaded_data())
    str(uploaded_data())
  })
  
  # --- Batch Classification ---
  batch_class_results <- eventReactive(input$run_batch_class_btn, {
    df <- uploaded_data()
    
    missing_cols <- setdiff(knn_features, names(df))
    validate(need(length(missing_cols) == 0,
                  paste("Uploaded data is missing required columns:",
                        paste(missing_cols, collapse = ", "))))
    
    model_choice <- input$batch_class_model
    pred_df <- df[, knn_features]
    
    if (model_choice == "Naive Bayes") {
      preds <- predict(nb_model, pred_df)
    } else if (model_choice == "Decision Tree") {
      preds <- predict(dt_model, pred_df, type = "class")
    } else {
      new_matrix <- as.matrix(pred_df)
      new_scaled <- sweep(sweep(new_matrix, 2, train_center, "-"), 2, train_scale_attr, "/")
      preds <- knn(train = train_scaled, test = new_scaled,
                   cl = train_data$risk_class, k = knn_k_default)
    }
    
    df$predicted_risk_class <- as.character(preds)
    df
  })
  
  output$batch_class_table <- renderDT({
    req(batch_class_results())
    datatable(batch_class_results(), options = list(scrollX = TRUE, pageLength = 10))
  })
  
  output$download_batch_class <- downloadHandler(
    filename = function() paste0("batch_classification_results_", Sys.Date(), ".csv"),
    content = function(file) {
      write_csv(batch_class_results(), file)
    }
  )
  
  # --- Batch Regression ---
  batch_reg_results <- eventReactive(input$run_batch_reg_btn, {
    df <- uploaded_data()
    
    missing_cols <- setdiff(knn_features, names(df))
    validate(need(length(missing_cols) == 0,
                  paste("Uploaded data is missing required columns:",
                        paste(missing_cols, collapse = ", "))))
    
    pred_df <- df[, knn_features]
    df$predicted_impact_score <- predict(multiple_lm, pred_df)
    df
  })
  
  output$batch_reg_table <- renderDT({
    req(batch_reg_results())
    datatable(batch_reg_results(), options = list(scrollX = TRUE, pageLength = 10))
  })
  
  output$download_batch_reg <- downloadHandler(
    filename = function() paste0("batch_regression_results_", Sys.Date(), ".csv"),
    content = function(file) {
      write_csv(batch_reg_results(), file)
    }
  )
  
}

# ============================================================
# RUN APP
# ============================================================
shinyApp(ui = ui, server = server)