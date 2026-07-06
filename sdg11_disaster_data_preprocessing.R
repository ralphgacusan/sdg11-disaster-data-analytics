# ============================================================
# INSTALL PACKAGES (run once)
# ============================================================
library(dplyr)
library(tidyr)
library(stringr)
library(stringi)
library(readr)
library(ggplot2)
library(corrplot)

# ============================================================
# INSTALL ADDITIONAL PACKAGES (run once)
# ============================================================
install.packages(c("e1071", "rpart", "rpart.plot", "class", "caret", "cluster", "factoextra"))

library(e1071)        # Naive Bayes
library(rpart)        # Decision Trees
library(rpart.plot)   # Decision Tree visualization
library(class)        # KNN
library(caret)        # Train/test split, confusion matrix
library(cluster)      # K-Means / clustering utilities
library(factoextra)   # Cluster visualization

# ============================================================
# SET WORKING DIRECTORY
# ============================================================
setwd("C:/Users/AZEY/Downloads/GACUSAN/Data Analytics")

# ============================================================
# LOAD DATASETS
# ============================================================
housing <- read_csv("housing.csv")
population <- read_csv("municipality_population.csv")
disaster <- read_csv("disaster.csv")
evacuation <- read_csv("evacuation.csv")
health_facility <- read_csv("health_facility.csv")

cat("Housing shape:", dim(housing), "\n")
cat("Population shape:", dim(population), "\n")
cat("Disaster shape:", dim(disaster), "\n")
cat("Evacuation shape:", dim(evacuation), "\n")
cat("Health Facility shape:", dim(health_facility), "\n")

head(housing, 2)
head(population, 2)
head(disaster, 2)
head(evacuation, 2)
head(health_facility, 2)

# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================
standardize_cols <- function(df) {
  names(df) <- names(df) |> str_trim() |> str_to_lower() |> str_replace_all(" ", "_")
  df
}

housing <- standardize_cols(housing)
disaster <- standardize_cols(disaster)
evacuation <- standardize_cols(evacuation)
health_facility <- standardize_cols(health_facility)
population <- standardize_cols(population)

print(names(housing))
print(names(disaster))
print(names(evacuation))
print(names(health_facility))
print(names(population))

# ============================================================
# MISSING VALUES CHECK
# ============================================================
colSums(is.na(housing))
colSums(is.na(disaster))
colSums(is.na(evacuation))
colSums(is.na(health_facility))
colSums(is.na(population))

# ============================================================
# REGION NAME NORMALIZATION
# ============================================================
normalize_region <- function(x) {
  x <- toupper(str_trim(as.character(x)))
  
  mapping <- c(
    "REGION I (ILOCOS REGION)" = "REGION I",
    "REGION II (CAGAYAN VALLEY)" = "REGION II",
    "REGION III (CENTRAL LUZON)" = "REGION III",
    "REGION IV-A (CALABARZON)" = "REGION IV-A",
    "REGION V (BICOL REGION)" = "REGION V",
    "REGION VI (WESTERN VISAYAS)" = "REGION VI",
    "REGION VII (CENTRAL VISAYAS)" = "REGION VII",
    "REGION VIII (EASTERN VISAYAS)" = "REGION VIII",
    "REGION IX (ZAMBOANGA PENINSULA)" = "REGION IX",
    "REGION X (NORTHERN MINDANAO)" = "REGION X",
    "REGION XI (DAVAO REGION)" = "REGION XI",
    "REGION XII (SOCCSKSARGEN)" = "REGION XII",
    "REGION XIII (CARAGA)" = "REGION XIII",
    "NATIONAL CAPITAL REGION (NCR)" = "NCR",
    "CORDILLERA ADMINISTRATIVE REGION (CAR)" = "CAR",
    "MIMAROPA REGION" = "MIMAROPA",
    "BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM)" = "BARMM",
    "AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)" = "BARMM"
  )
  
  ifelse(x %in% names(mapping), mapping[x], x)
}

housing$region <- normalize_region(housing$region)
evacuation$region <- normalize_region(evacuation$region)
health_facility$region <- normalize_region(health_facility$region)
population$region <- normalize_region(population$region)
disaster$region <- normalize_region(disaster$region)

# ============================================================
# REGION CONSISTENCY CHECK
# ============================================================
setdiff(unique(housing$region), unique(evacuation$region))
setdiff(unique(housing$region), unique(health_facility$region))
setdiff(unique(housing$region), unique(population$region))
setdiff(unique(housing$region), unique(disaster$region))

# ============================================================
# POPULATION CLEANING (REMOVE COMMAS, CONVERT TO NUMERIC)
# ============================================================
population <- population |>
  mutate(
    female = as.numeric(str_replace_all(as.character(female), ",", "")),
    male   = as.numeric(str_replace_all(as.character(male), ",", ""))
  )

colSums(is.na(population[c("female", "male")]))

# ============================================================
# POPULATION AGGREGATION BY MUNICIPALITY
# ============================================================
population_agg <- population |>
  group_by(region, region_code, province, province_code,
           municipality_city, municipality_city_code) |>
  summarise(
    female = sum(female, na.rm = TRUE),
    male = sum(male, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(total_population = female + male)

cat("Aggregated Shape:", dim(population_agg), "\n")
head(population_agg)

# ============================================================
# HEALTH FACILITY COLUMN RENAMING
# ============================================================
health_facility <- health_facility |>
  rename(
    municipality_city = city_municipality,
    municipality_city_code = city_mun_code,
    region_code = reg_code,
    province_code = prov_code
  )

housing$region <- str_trim(housing$region)
disaster$region <- str_trim(disaster$region)

print(unique(housing$region))
print(unique(disaster$region))

# ============================================================
# FILTER DISASTER DATA BY YEAR
# ============================================================
year_to_use <- 2018

disaster_year <- disaster |>
  filter(year == year_to_use) |>
  mutate(region = normalize_region(region))

print(head(disaster_year[c("region", "year", "affected_per_100k")]))

print(unique(housing$region))
print(unique(disaster$region))

# ============================================================
# CLEANING FUNCTION (REGION, PROVINCE, MUNICIPALITY)
# ============================================================
clean_df <- function(df) {
  df <- df |>
    mutate(
      region = str_trim(toupper(as.character(region))),
      province = str_trim(toupper(as.character(province))),
      municipality_city = as.character(municipality_city) |>
        toupper() |>
        str_trim() |>
        str_replace_all("\\(.*?\\)", "") |>
        str_replace_all("-", " ") |>
        str_replace_all("CITY OF ", "") |>
        str_replace_all("Ã‘", "Ñ") |>
        str_replace_all("\\?", "Ñ")
    )
  
  # Unicode normalization
  df$municipality_city <- stri_trans_nfkd(df$municipality_city)
  
  # Normalize spacing
  df <- df |>
    mutate(
      municipality_city = str_replace_all(municipality_city, "\\s+", " ") |> str_trim()
    )
  
  # Remove Cotabato City conflict row
  df <- df |>
    filter(!(region == "REGION XII" & str_detect(municipality_city, "COTABATO")))
  
  # NCR fix
  ncr_mask <- str_detect(df$region, "NATIONAL CAPITAL REGION|NCR")
  df$municipality_city[ncr_mask] <- "MANILA CITY"
  
  df
}

# ============================================================
# MERGE ALL DATASETS
# ============================================================
housing <- normalize_region_col <- housing |> mutate(region = normalize_region(region))
evacuation <- evacuation |> mutate(region = normalize_region(region))
health_facility <- health_facility |> mutate(region = normalize_region(region))
population_agg <- population_agg |> mutate(region = normalize_region(region))
disaster <- disaster |> mutate(region = normalize_region(region))

merged <- clean_df(housing)
evacuation <- clean_df(evacuation)
health_facility <- clean_df(health_facility)
population_agg <- clean_df(population_agg)

join_cols <- c("region", "region_code", "province", "province_code",
               "municipality_city", "municipality_city_code")

merged <- merged |> left_join(evacuation, by = join_cols)
cat("After merging evacuation data:", dim(merged), "\n")

merged <- merged |> left_join(health_facility, by = join_cols)
cat("After merging health facility data:", dim(merged), "\n")

merged <- merged |> left_join(population_agg, by = join_cols)
cat("After merging population data:", dim(merged), "\n")

merged <- merged |> left_join(disaster_year |> select(-year), by = "region")
cat("Final merged dataset shape:", dim(merged), "\n")

# Check for NaNs
nan_cols <- names(merged)[colSums(is.na(merged)) > 0]
print(nan_cols)
print(colSums(is.na(merged[nan_cols])))

print(head(merged, 2))

setdiff(unique(merged$region), unique(disaster_year$region))
setdiff(unique(disaster_year$region), unique(merged$region))

for (nm in list(
  list("housing", housing),
  list("evacuation", evacuation),
  list("health", health_facility),
  list("population", population),
  list("disaster", disaster)
)) {
  cat(nm[[1]], ":", paste(unique(nm[[2]]$region), collapse = ", "), "\n")
}

# ============================================================
# HOUSING STRUCTURAL CATEGORIES
# ============================================================
merged <- merged |>
  mutate(
    strong = `strong_roof/strong_wall`,
    light = `strong_roof/light_wall` + `light_roof/strong_wall` + `light_roof/light_wall`,
    salvage = `strong_roof/salvage_wall` + `light_roof/salvage_wall` +
      `salvaged_roof/strong_wall` + `salvaged_roof/light_wall` + `salvaged_roof/salvage_wall`
  )

print(head(merged[c("strong", "light", "salvage")]))

# ============================================================
# STRUCTURAL RATIOS
# ============================================================
merged <- merged |>
  mutate(
    strong_ratio = strong / housing_units,
    light_ratio = light / housing_units,
    salvage_ratio = salvage / housing_units
  )

print(head(merged[c("strong_ratio", "light_ratio", "salvage_ratio")]))

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Housing Resilience Score
merged <- merged |>
  mutate(
    housing_resilience_score = strong_ratio * 1.0 + light_ratio * 0.5 + salvage_ratio * 0.0
  )

# Health Access Index
health_cols <- c("hospital", "rural_health_unit", "barangay_health_station",
                 "municipal_health_office", "birthing_home")

merged$health_access_index <- rowSums(merged[health_cols], na.rm = TRUE)

# Evacuation Capacity
merged <- merged |>
  mutate(
    evacuation_capacity = (number_of_evacuation_center / total_population) * 1000
  )

# Disaster Intensity
merged <- merged |>
  mutate(
    disaster_intensity = deaths_per_100k + missing_per_100k + affected_per_100k,
    disaster_intensity_municipal = log1p(disaster_intensity * total_population / 100000)
  )

# Population Pressure
merged <- merged |>
  mutate(
    population_pressure = total_population / housing_units
  )

# Vulnerability Index
merged <- merged |>
  mutate(
    vulnerability_index = (1 - housing_resilience_score) +
      (disaster_intensity_municipal / 10000) +
      population_pressure -
      (health_access_index / (max(health_access_index, na.rm = TRUE) + 1)) -
      (evacuation_capacity / (max(evacuation_capacity, na.rm = TRUE) + 1))
  )

# Risk Classification (quantile-based 2-bin)
merged <- merged |>
  mutate(
    risk_class = cut(
      vulnerability_index,
      breaks = quantile(vulnerability_index, probs = c(0, 0.5, 1), na.rm = TRUE),
      labels = c("Low", "High"),
      include.lowest = TRUE
    )
  )

# Impact Score
merged <- merged |>
  mutate(
    affected_log = log1p(affected_per_100k),
    deaths_log = log1p(deaths_per_100k),
    impact_score = affected_log * 1 + missing_per_100k * 5 + deaths_log * 10,
    impact_score = impact_score * (population_pressure / max(population_pressure, na.rm = TRUE))
  )

# ============================================================
# FEATURE ENGINEERING SUMMARY
# ============================================================
engineered_features <- c(
  "housing_resilience_score", "health_access_index", "evacuation_capacity",
  "disaster_intensity_municipal", "population_pressure", "vulnerability_index",
  "risk_class", "impact_score"
)

print(head(merged[engineered_features]))
print(summary(merged[setdiff(engineered_features, "risk_class")]))
print(table(merged$risk_class))

# ============================================================
# BINARY / CATEGORICAL FEATURE ENGINEERING
# ============================================================

merged <- merged |>
  mutate(
    housing_resilience_bin = cut(
      housing_resilience_score,
      breaks = c(0, 0.6, 0.8, 1.0),
      labels = c("LOW RESILIENCE", "MEDIUM RESILIENCE", "HIGH RESILIENCE"),
      include.lowest = TRUE
    ),
    health_access_bin = cut(
      health_access_index,
      breaks = quantile(health_access_index, probs = seq(0, 1, length.out = 4), na.rm = TRUE),
      labels = c("POOR ACCESS", "FAIR ACCESS", "GOOD ACCESS"),
      include.lowest = TRUE
    ),
    evacuation_capacity_bin = cut(
      evacuation_capacity,
      breaks = c(-0.01, 0.05, 0.5, Inf),
      labels = c("LOW", "ADEQUATE", "HIGH")
    ),
    disaster_intensity_bin = cut(
      disaster_intensity_municipal,
      breaks = quantile(disaster_intensity_municipal, probs = seq(0, 1, length.out = 4), na.rm = TRUE),
      labels = c("LOW DISASTER", "MODERATE DISASTER", "HIGH DISASTER"),
      include.lowest = TRUE
    ),
    population_pressure_bin = cut(
      population_pressure,
      breaks = quantile(population_pressure, probs = seq(0, 1, length.out = 4), na.rm = TRUE),
      labels = c("LOW PRESSURE", "MEDIUM PRESSURE", "HIGH PRESSURE"),
      include.lowest = TRUE
    ),
    vulnerability_bin = cut(
      vulnerability_index,
      breaks = quantile(vulnerability_index, probs = seq(0, 1, length.out = 4), na.rm = TRUE),
      labels = c("LOW RISK", "MEDIUM RISK", "HIGH RISK"),
      include.lowest = TRUE
    ),
    impact_score_bin = cut(
      impact_score,
      breaks = quantile(impact_score, probs = seq(0, 1, length.out = 4), na.rm = TRUE),
      labels = c("LOW IMPACT", "MODERATE IMPACT", "HIGH IMPACT"),
      include.lowest = TRUE
    ),
    risk_class_bin = paste0(toupper(as.character(risk_class)), " RISK")
  )

cat("All binning features successfully created.\n")

# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# ============================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# ------------------------------------------------------------
# 4.1 DESCRIPTIVE STATISTICS: MEAN, MEDIAN, MODE, RANGE,
#     QUARTILES, STANDARD DEVIATION, VARIANCE
# ------------------------------------------------------------

# Helper function to compute mode (R has no built-in mode function)
get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

desc_features <- c(
  "housing_resilience_score", "health_access_index", "evacuation_capacity",
  "disaster_intensity_municipal", "population_pressure",
  "vulnerability_index", "impact_score"
)

cat("\n============================================================\n")
cat("4.1 DESCRIPTIVE STATISTICS SUMMARY\n")
cat("============================================================\n")

desc_summary <- data.frame(
  feature  = desc_features,
  mean     = sapply(desc_features, function(c) mean(merged[[c]], na.rm = TRUE)),
  median   = sapply(desc_features, function(c) median(merged[[c]], na.rm = TRUE)),
  mode     = sapply(desc_features, function(c) get_mode(merged[[c]])),
  min      = sapply(desc_features, function(c) min(merged[[c]], na.rm = TRUE)),
  max      = sapply(desc_features, function(c) max(merged[[c]], na.rm = TRUE)),
  range    = sapply(desc_features, function(c) diff(range(merged[[c]], na.rm = TRUE))),
  q1       = sapply(desc_features, function(c) quantile(merged[[c]], 0.25, na.rm = TRUE)),
  q2_median= sapply(desc_features, function(c) quantile(merged[[c]], 0.50, na.rm = TRUE)),
  q3       = sapply(desc_features, function(c) quantile(merged[[c]], 0.75, na.rm = TRUE)),
  iqr      = sapply(desc_features, function(c) IQR(merged[[c]], na.rm = TRUE)),
  sd       = sapply(desc_features, function(c) sd(merged[[c]], na.rm = TRUE)),
  variance = sapply(desc_features, function(c) var(merged[[c]], na.rm = TRUE))
)

rownames(desc_summary) <- NULL
print(desc_summary, digits = 4)

write_csv(desc_summary, "descriptive_statistics_summary.csv")
cat("\nSaved descriptive_statistics_summary.csv\n")


# ------------------------------------------------------------
# 4.2 HISTOGRAMS — DISTRIBUTION OF KEY FEATURES
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("4.2 HISTOGRAMS\n")
cat("============================================================\n")

# Histogram: Vulnerability Index
hist_vuln <- ggplot(merged, aes(x = vulnerability_index)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Histogram: Distribution of Vulnerability Index",
       x = "Vulnerability Index", y = "Frequency (Number of Municipalities)")
print(hist_vuln)

# Histogram: Housing Resilience Score
hist_resil <- ggplot(merged, aes(x = housing_resilience_score)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "black") +
  labs(title = "Histogram: Distribution of Housing Resilience Score",
       x = "Housing Resilience Score", y = "Frequency (Number of Municipalities)")
print(hist_resil)

# Histogram: Population Pressure
hist_pop <- ggplot(merged, aes(x = population_pressure)) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  labs(title = "Histogram: Distribution of Population Pressure",
       x = "Population Pressure (People per Housing Unit)", y = "Frequency")
print(hist_pop)

# Histogram: Impact Score
hist_impact <- ggplot(merged, aes(x = impact_score)) +
  geom_histogram(bins = 30, fill = "purple", color = "black") +
  labs(title = "Histogram: Distribution of Impact Score",
       x = "Impact Score", y = "Frequency (Number of Municipalities)")
print(hist_impact)


# ------------------------------------------------------------
# 4.3 BOXPLOTS — SPREAD AND OUTLIERS ACROSS RISK CLASSES
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("4.3 BOXPLOTS\n")
cat("============================================================\n")

# Boxplot: Vulnerability Index by Risk Class
box_vuln <- ggplot(merged, aes(x = risk_class, y = vulnerability_index, fill = risk_class)) +
  geom_boxplot() +
  labs(title = "Boxplot: Vulnerability Index by Risk Class",
       x = "Risk Class", y = "Vulnerability Index")
print(box_vuln)

# Boxplot: Population Pressure by Risk Class
box_pop <- ggplot(merged, aes(x = risk_class, y = population_pressure, fill = risk_class)) +
  geom_boxplot() +
  labs(title = "Boxplot: Population Pressure by Risk Class",
       x = "Risk Class", y = "Population Pressure")
print(box_pop)

# Boxplot: Housing Resilience Score by Risk Class
box_resil <- ggplot(merged, aes(x = risk_class, y = housing_resilience_score, fill = risk_class)) +
  geom_boxplot() +
  labs(title = "Boxplot: Housing Resilience Score by Risk Class",
       x = "Risk Class", y = "Housing Resilience Score")
print(box_resil)

# Boxplot: Impact Score by Risk Class
box_impact <- ggplot(merged, aes(x = risk_class, y = impact_score, fill = risk_class)) +
  geom_boxplot() +
  labs(title = "Boxplot: Impact Score by Risk Class",
       x = "Risk Class", y = "Impact Score")
print(box_impact)


# ------------------------------------------------------------
# 4.4 SCATTERPLOTS — RELATIONSHIPS BETWEEN VARIABLES
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("4.4 SCATTERPLOTS\n")
cat("============================================================\n")

# Scatterplot: Housing Resilience vs Disaster Intensity
correlation_hr_di <- cor(merged$housing_resilience_score, merged$disaster_intensity_municipal, use = "complete.obs")
cat("\nCorrelation (Housing Resilience vs Disaster Intensity):", round(correlation_hr_di, 4), "\n")

scatter_hr_di <- ggplot(merged, aes(x = housing_resilience_score, y = disaster_intensity_municipal)) +
  geom_point(color = "steelblue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Scatterplot: Housing Resilience vs Disaster Intensity",
       x = "Housing Resilience Score", y = "Disaster Intensity (Municipal)")
print(scatter_hr_di)

# Scatterplot: Population Pressure vs Vulnerability Index
correlation_pp_vi <- cor(merged$population_pressure, merged$vulnerability_index, use = "complete.obs")
cat("Correlation (Population Pressure vs Vulnerability Index):", round(correlation_pp_vi, 4), "\n")

scatter_pp_vi <- ggplot(merged, aes(x = population_pressure, y = vulnerability_index)) +
  geom_point(color = "darkorange", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Scatterplot: Population Pressure vs Vulnerability Index",
       x = "Population Pressure", y = "Vulnerability Index")
print(scatter_pp_vi)

# Scatterplot: Population Pressure vs Impact Score
correlation_pp_is <- cor(merged$population_pressure, merged$impact_score, use = "complete.obs")
cat("Correlation (Population Pressure vs Impact Score):", round(correlation_pp_is, 4), "\n")

scatter_pp_is <- ggplot(merged, aes(x = population_pressure, y = impact_score)) +
  geom_point(color = "darkgreen", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Scatterplot: Population Pressure vs Impact Score",
       x = "Population Pressure", y = "Impact Score")
print(scatter_pp_is)

# Correlation matrix (retained)
corr_cols <- c("housing_resilience_score", "health_access_index", "evacuation_capacity",
               "disaster_intensity_municipal", "population_pressure", "vulnerability_index", "impact_score")

corr_matrix <- cor(merged[corr_cols], use = "complete.obs")
cat("\n=== CORRELATION MATRIX ===\n")
print(round(corr_matrix, 4))

corrplot(corr_matrix, method = "color", addCoef.col = "black",
         title = "Feature Correlation Heatmap", mar = c(0, 0, 2, 0))


# Top 10 most vulnerable municipalities (retained)
top_vulnerable <- merged |>
  arrange(desc(vulnerability_index)) |>
  head(10)

cat("\n=== TOP 10 MOST VULNERABLE MUNICIPALITIES ===\n")
print(top_vulnerable[c("municipality_city", "vulnerability_index")])

bar_top10 <- ggplot(top_vulnerable, aes(x = reorder(municipality_city, vulnerability_index), y = vulnerability_index)) +
  geom_col(fill = "firebrick") +
  coord_flip() +
  labs(title = "Top 10 Most Vulnerable Municipalities", x = "Municipality", y = "Vulnerability Index")
print(bar_top10)


# ============================================================
# SECTION 5: CLASSIFICATION
# ============================================================

cat("\n============================================================\n")
cat("SECTION 5: CLASSIFICATION\n")
cat("============================================================\n")

# ------------------------------------------------------------
# 5.1 CLASSIFICATION DATA PREPARATION
# ------------------------------------------------------------

cat("\n--- 5.1 Classification Data Preparation ---\n")

classification_data <- merged |>
  select(
    strong_ratio, light_ratio, salvage_ratio,
    population_pressure, health_access_index,
    evacuation_capacity, disaster_intensity_municipal,
    drr_adoption_pct, risk_class
  ) |>
  na.omit()

classification_data$risk_class <- as.factor(classification_data$risk_class)

# Train/test split (70/30)
set.seed(123)
train_index <- createDataPartition(classification_data$risk_class, p = 0.7, list = FALSE)
train_data <- classification_data[train_index, ]
test_data  <- classification_data[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")
cat("Predictors used:", paste(setdiff(names(classification_data), "risk_class"), collapse = ", "), "\n")


# ------------------------------------------------------------
# 5.2 NAIVE BAYES CLASSIFICATION
# ------------------------------------------------------------

cat("\n--- 5.2 Naive Bayes Classification ---\n")

nb_model <- naiveBayes(risk_class ~ ., data = train_data)

nb_pred <- predict(nb_model, test_data)

cat("\n=== NAIVE BAYES CONFUSION MATRIX ===\n")
nb_cm <- confusionMatrix(nb_pred, test_data$risk_class)
print(nb_cm)

cat("\nNaive Bayes Accuracy:", round(nb_cm$overall["Accuracy"], 4), "\n")


# ------------------------------------------------------------
# 5.3 DECISION TREES
# ------------------------------------------------------------

cat("\n--- 5.3 Decision Tree Classification ---\n")

dt_model <- rpart(risk_class ~ ., data = train_data, method = "class")

# Visualize the decision tree
rpart.plot(dt_model, type = 3, extra = 104, fallen.leaves = TRUE,
           main = "Decision Tree for Risk Classification")

dt_pred <- predict(dt_model, test_data, type = "class")

cat("\n=== DECISION TREE CONFUSION MATRIX ===\n")
dt_cm <- confusionMatrix(dt_pred, test_data$risk_class)
print(dt_cm)

cat("\nDecision Tree Accuracy:", round(dt_cm$overall["Accuracy"], 4), "\n")

cat("\n=== DECISION TREE VARIABLE IMPORTANCE ===\n")
print(dt_model$variable.importance)


# ------------------------------------------------------------
# 5.4 K NEAREST NEIGHBOR (KNN)
# ------------------------------------------------------------

cat("\n--- 5.4 K-Nearest Neighbor Classification ---\n")

knn_features <- c("strong_ratio", "light_ratio", "salvage_ratio",
                  "population_pressure", "health_access_index",
                  "evacuation_capacity", "disaster_intensity_municipal",
                  "drr_adoption_pct")

train_df <- as.data.frame(train_data)[, knn_features]
test_df  <- as.data.frame(test_data)[, knn_features]

train_matrix <- as.matrix(train_df)
test_matrix  <- as.matrix(test_df)

# Use base::scale explicitly to avoid masking from other packages
train_scaled <- base::scale(train_matrix)

train_center <- attr(train_scaled, "center")
train_scale_attr <- attr(train_scaled, "scale")

cat("length of center:", length(train_center), "\n")
cat("length of scale:", length(train_scale_attr), "\n")

# Fallback: if attributes are still missing, compute manually
if (length(train_center) == 0 || length(train_scale_attr) == 0) {
  cat("\nNOTE: scale() attributes missing — computing manually.\n")
  train_center <- colMeans(train_matrix, na.rm = TRUE)
  train_scale_attr <- apply(train_matrix, 2, sd, na.rm = TRUE)
  train_scaled <- sweep(train_matrix, 2, train_center, "-")
  train_scaled <- sweep(train_scaled, 2, train_scale_attr, "/")
}

test_scaled <- sweep(test_matrix, 2, train_center, "-")
test_scaled <- sweep(test_scaled, 2, train_scale_attr, "/")

cat("train_scaled dimensions:", dim(train_scaled), "\n")
cat("test_scaled dimensions:", dim(test_scaled), "\n")

knn_pred <- knn(train = train_scaled, test = test_scaled,
                cl = train_data$risk_class, k = 5)

cat("\n=== KNN (k = 5) CONFUSION MATRIX ===\n")
knn_cm <- confusionMatrix(knn_pred, test_data$risk_class)
print(knn_cm)

cat("\nKNN Accuracy (k = 5):", round(knn_cm$overall["Accuracy"], 4), "\n")

# Test different k values to find optimal k
k_values <- 1:15
k_accuracy <- sapply(k_values, function(k) {
  pred <- knn(train = train_scaled, test = test_scaled,
              cl = train_data$risk_class, k = k)
  mean(pred == test_data$risk_class)
})

k_results <- data.frame(k = k_values, accuracy = round(k_accuracy, 4))
cat("\n=== KNN ACCURACY FOR k = 1 TO 15 ===\n")
print(k_results)

best_k <- k_results$k[which.max(k_results$accuracy)]
cat("\nBest k value:", best_k, "with accuracy:", max(k_results$accuracy), "\n")

knn_tune_plot <- ggplot(k_results, aes(x = k, y = accuracy)) +
  geom_line(color = "steelblue") +
  geom_point(color = "darkblue", size = 2) +
  labs(title = "KNN Accuracy vs Number of Neighbors (k)",
       x = "k (Number of Neighbors)", y = "Accuracy")
print(knn_tune_plot)

# ------------------------------------------------------------
# K-MEANS CLUSTERING
# ------------------------------------------------------------

cat("\n--- K-Means Clustering ---\n")

kmeans_features <- merged |>
  select(housing_resilience_score, health_access_index,
         evacuation_capacity, disaster_intensity_municipal,
         population_pressure) |>
  na.omit()

kmeans_scaled <- scale(kmeans_features)

# Determine optimal number of clusters using the elbow method
elbow_plot <- fviz_nbclust(kmeans_scaled, kmeans, method = "wss") +
  labs(title = "Elbow Method for Optimal Number of Clusters")
print(elbow_plot)

# Run K-means with k = 3
set.seed(123)
km_model <- kmeans(kmeans_scaled, centers = 3, nstart = 25)

cat("\n=== K-MEANS CLUSTER CENTERS (k = 3) ===\n")
print(round(km_model$centers, 4))

cat("\n=== K-MEANS CLUSTER SIZES ===\n")
print(table(km_model$cluster))

# Visualize clusters
kmeans_plot <- fviz_cluster(km_model, data = kmeans_scaled,
                            geom = "point", ellipse.type = "convex",
                            main = "K-Means Clustering of Municipalities (k = 3)")
print(kmeans_plot)

# Attach cluster assignment back to merged dataset for interpretation
merged_clustered <- merged |> na.omit()
merged_clustered$cluster <- km_model$cluster

cat("\n=== K-MEANS CLUSTER PROFILE SUMMARY ===\n")
cluster_profile <- merged_clustered |>
  group_by(cluster) |>
  summarise(
    count = n(),
    avg_housing_resilience = round(mean(housing_resilience_score), 4),
    avg_health_access = round(mean(health_access_index), 4),
    avg_evacuation_capacity = round(mean(evacuation_capacity), 4),
    avg_disaster_intensity = round(mean(disaster_intensity_municipal), 4),
    avg_population_pressure = round(mean(population_pressure), 4),
    avg_vulnerability_index = round(mean(vulnerability_index), 4)
  )

print(cluster_profile)


# ============================================================
# SECTION 5.5: REGRESSION
# ============================================================

cat("\n============================================================\n")
cat("SECTION 5.5: REGRESSION\n")
cat("============================================================\n")

regression_data <- merged |>
  select(
    impact_score, strong_ratio, light_ratio, salvage_ratio,
    population_pressure, health_access_index,
    evacuation_capacity, disaster_intensity_municipal,
    drr_adoption_pct
  ) |>
  na.omit()

# ------------------------------------------------------------
# SIMPLE LINEAR REGRESSION
# ------------------------------------------------------------

cat("\n--- Simple Linear Regression: Impact Score ~ Population Pressure ---\n")

simple_lm <- lm(impact_score ~ population_pressure, data = regression_data)

cat("\n=== SIMPLE LINEAR REGRESSION SUMMARY ===\n")
print(summary(simple_lm))

simple_lm_plot <- ggplot(regression_data, aes(x = population_pressure, y = impact_score)) +
  geom_point(color = "steelblue", alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Simple Linear Regression: Impact Score vs Population Pressure",
       x = "Population Pressure", y = "Impact Score")
print(simple_lm_plot)


# ------------------------------------------------------------
# MULTIPLE LINEAR REGRESSION
# ------------------------------------------------------------

cat("\n--- Multiple Linear Regression: Impact Score ~ All Predictors ---\n")

multiple_lm <- lm(impact_score ~ strong_ratio + light_ratio + salvage_ratio +
                    population_pressure + health_access_index +
                    evacuation_capacity + disaster_intensity_municipal +
                    drr_adoption_pct,
                  data = regression_data)

cat("\n=== MULTIPLE LINEAR REGRESSION SUMMARY ===\n")
print(summary(multiple_lm))

# Compare actual vs predicted values
regression_data$predicted_impact <- predict(multiple_lm, regression_data)

mlr_plot <- ggplot(regression_data, aes(x = impact_score, y = predicted_impact)) +
  geom_point(color = "darkgreen", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Multiple Linear Regression: Actual vs Predicted Impact Score",
       x = "Actual Impact Score", y = "Predicted Impact Score")
print(mlr_plot)

# Model performance metrics
mlr_rmse <- sqrt(mean((regression_data$impact_score - regression_data$predicted_impact)^2))
mlr_r2 <- summary(multiple_lm)$r.squared
mlr_adj_r2 <- summary(multiple_lm)$adj.r.squared

cat("\n=== MULTIPLE LINEAR REGRESSION PERFORMANCE METRICS ===\n")
cat("RMSE:", round(mlr_rmse, 4), "\n")
cat("R-squared:", round(mlr_r2, 4), "\n")
cat("Adjusted R-squared:", round(mlr_adj_r2, 4), "\n")


# ============================================================
# SECTION 6.1: CLUSTERING (HIERARCHICAL)
# ============================================================

cat("\n============================================================\n")
cat("SECTION 6.1: HIERARCHICAL CLUSTERING\n")
cat("============================================================\n")

# Re-use kmeans_scaled from above for hierarchical clustering comparison

# Hierarchical clustering
dist_matrix <- dist(kmeans_scaled, method = "euclidean")
hc_model <- hclust(dist_matrix, method = "ward.D2")

# Dendrogram
plot(hc_model, main = "Hierarchical Clustering Dendrogram",
     xlab = "Municipalities", ylab = "Height", labels = FALSE)
rect.hclust(hc_model, k = 3, border = "red")

# Cut tree into 3 clusters and compare with K-means
hc_clusters <- cutree(hc_model, k = 3)

cat("\n=== HIERARCHICAL CLUSTERING CLUSTER SIZES (k = 3) ===\n")
print(table(hc_clusters))

# Cross-tabulation: K-means vs Hierarchical clustering agreement
cat("\n=== K-MEANS vs HIERARCHICAL CLUSTERING CROSS-TAB ===\n")
print(table(KMeans = km_model$cluster, Hierarchical = hc_clusters))

# Final cluster visualization
hc_plot <- fviz_cluster(list(data = kmeans_scaled, cluster = hc_clusters),
                        geom = "point", ellipse.type = "convex",
                        main = "Hierarchical Clustering of Municipalities (k = 3)")
print(hc_plot)

# ============================================================
# FINAL MODELING DATASET
# ============================================================
# ============================================================
# FINAL MODELING DATASET
# ============================================================
model_df <- merged |>
  select(
    region, province, municipality_city,
    housing_units, strong_ratio, light_ratio, salvage_ratio,
    total_population,
    hospital, rural_health_unit, barangay_health_station,
    municipal_health_office, birthing_home,
    number_of_evacuation_center, drr_adoption_pct,
    deaths_per_100k, missing_per_100k, affected_per_100k,
    housing_resilience_score, health_access_index,
    evacuation_capacity, population_pressure,
    disaster_intensity_municipal, vulnerability_index,
    impact_score, risk_class
  )

cat("Final modeling shape:", dim(model_df), "\n")
print(head(model_df))

model_path <- "sdg11_disaster_dataset.csv"
write_csv(model_df, model_path)

cat("======================================\n")
cat("DATASET EXPORT COMPLETED SUCCESSFULLY\n")
cat("======================================\n\n")
cat("MODEL DATASET saved to:", normalizePath(model_path), "\n")
cat("\n=== DATASET SHAPES ===\n")
cat("Model DF shape:", dim(model_df), "\n")