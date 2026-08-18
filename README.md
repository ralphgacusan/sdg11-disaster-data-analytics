# Disaster Risk Analytics and Predictive Modeling of Municipal Vulnerability, Preparedness, and Disaster Impacts in the Philippines Using R

An end-to-end R data analytics project that integrates housing, demographic, health, evacuation, and disaster-related datasets across all **1,632 municipalities** of the Philippines to identify drivers of disaster vulnerability and impact, and presents the results through an interactive **R Shiny dashboard**.

## Overview

The Philippines is ranked the most disaster-prone country in the world (WorldRiskIndex 2025). While Republic Act 10121 mandates disaster risk reduction and management at the local level, disaster impact still varies significantly across municipalities with similar hazard exposure. This project combines public datasets from the **Philippine Statistics Authority (PSA) OpenSTAT** and the **UN OCHA Humanitarian Data Exchange (HDX)** into a single municipal-level dataset, then applies classification, regression, and clustering techniques to uncover the structural and institutional predictors of disaster severity.

### Research Questions

1. What are the distributions, patterns, and relationships among housing resilience, health access, evacuation capacity, disaster intensity, population pressure, vulnerability index, and impact score across Philippine municipalities?
2. Which factors most strongly influence municipal disaster vulnerability and impact?
3. How accurately can classification models (Naive Bayes, Decision Tree, K-Nearest Neighbor) classify municipalities into High-Risk and Low-Risk categories?
4. How effectively can regression models predict municipal disaster impact scores?
5. What distinct municipal risk and resilience profiles emerge through clustering (K-Means, hierarchical)?
6. How can these results be integrated into an interactive dashboard to support evidence-based DRR planning?

## Repository Structure

```
├── housing.csv                          # Raw housing structure data (roof/wall material by municipality)
├── municipality_population.csv          # Raw population data by age group, sex, and municipality
├── disaster.csv                         # Raw regional disaster impact indicators (deaths, missing, affected)
├── evacuation.csv                       # Raw evacuation center counts by municipality
├── health_facility.csv                  # Raw health facility counts by municipality
├── sdg11_disaster_data_preprocessing.R  # Data cleaning, transformation, and feature engineering script
├── sdg11_disaster_dataset.csv           # Final cleaned & merged modeling dataset (1,632 rows x 14 cols)
├── descriptive_statistics_summary.csv   # Summary statistics for all engineered features
└── sdg11_disaster_data_dashboard.R      # R Shiny interactive dashboard application
```

## Data Sources

| Source                                                                              | Indicator                                                                 | Format          |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------- |
| [PSA OpenSTAT](https://openstat.psa.gov.ph)                                         | SDG 11.5.1 - Deaths, missing, and affected persons per 100,000 population | PXWeb / tabular |
| [PSA OpenSTAT](https://openstat.psa.gov.ph)                                         | SDG 11.b.2 - Proportion of LGUs adopting DRR strategies                   | PXWeb / tabular |
| [UN OCHA HDX](https://data.humdata.org/dataset/philippines-pre-disaster-indicators) | Philippines Pre-Disaster Indicators                                       | CSV             |

## Data Pipeline

`sdg11_disaster_data_preprocessing.R` performs the full cleaning and feature engineering workflow:

1. Standardizes column names and region naming conventions across all five raw datasets (including the historical ARMM → BARMM rename).
2. Cleans comma-formatted population figures and aggregates population data from age-group level to municipal level.
3. Resolves municipality-name inconsistencies (parenthetical text, hyphens, "City of" prefixes, encoding artifacts, the Cotabato City duplicate record, NCR sub-city naming).
4. Merges all five datasets into a unified municipal-level table via sequential `left_join()` operations.
5. Engineers seven composite indicators:
   - **Housing Resilience Score**
   - **Health Access Index**
   - **Evacuation Capacity**
   - **Disaster Intensity** (municipal, log-scaled)
   - **Population Pressure**
   - **Vulnerability Index**
   - **Impact Score**
6. Derives a binary **Risk Class** (Low/High) via a median split of the Vulnerability Index, and exports the final modeling dataset to `sdg11_disaster_dataset.csv` (1,632 rows x 14 columns, no missing values).

## Modeling & Key Findings

**Classification** (Risk Class: Low vs. High)

| Model             | Accuracy   | Kappa     |
| ----------------- | ---------- | --------- |
| Naive Bayes       | 80.12%     | 0.603     |
| **Decision Tree** | **93.65%** | **0.873** |
| KNN (k = 5)       | 83.20%     | 0.664     |

The Decision Tree is the recommended classifier, with population pressure as the dominant split variable, followed by housing construction ratios and DRR adoption.

**Regression** (continuous Impact Score)

| Model                                               | R²     | RMSE  |
| --------------------------------------------------- | ------ | ----- |
| Simple Linear Regression (Population Pressure only) | 0.2475 | 1.808 |
| Multiple Linear Regression (all 8 predictors)       | 0.4055 | 1.606 |

**Clustering** (K-Means and Ward's D2 hierarchical, k = 3) converge on a consistent municipal typology: a large moderate-risk majority, a well-served resilient minority, and a small (~44–113 municipality) high-overcrowding, low-resilience group concentrated in BARMM requiring priority intervention.

Population pressure is consistently the strongest driver of both risk classification and disaster impact across every model.

## Dashboard

`sdg11_disaster_data_dashboard.R` is an R Shiny application (`shinydashboard`) with sidebar navigation across seven tabs:

- **Overview** — summary value boxes, vulnerability distribution, risk class breakdown
- **Descriptive Statistics** — interactive per-variable statistical summaries
- **Exploratory Data Analysis** — histograms, boxplots by risk class, scatterplots, correlation heatmap, vulnerability rankings
- **Classification** — model evaluation (accuracy, Kappa, confusion matrices) and a live risk-class prediction tool
- **Regression** — model evaluation and a live impact-score prediction tool
- **Clustering** — K-Means and hierarchical clustering visualizations, elbow plot, cluster profiles
- **Data Upload & Batch Prediction** — upload a CSV and apply trained models at scale

### Running the Dashboard

```r
# Install required packages
install.packages(c(
  "shiny", "shinydashboard", "shinyWidgets", "shinycssloaders",
  "dplyr", "tidyr", "readr", "reshape2",
  "ggplot2", "plotly", "corrplot", "DT",
  "e1071", "rpart", "rpart.plot", "class", "caret",
  "cluster", "factoextra"
))

# Run the app (make sure sdg11_disaster_dataset.csv is in the working directory)
shiny::runApp("sdg11_disaster_data_dashboard.R")
```

## Tech Stack

- **Language:** R
- **Data wrangling:** dplyr, tidyr, readr, reshape2, stringr, stringi
- **Visualization:** ggplot2, plotly, corrplot
- **Machine learning:** e1071 (Naive Bayes), rpart / rpart.plot (Decision Tree), class (KNN), caret (evaluation), cluster / factoextra (K-Means & hierarchical clustering)
- **Dashboard:** Shiny, shinydashboard, shinyWidgets, shinycssloaders, DT

## References

Key references include the WorldRiskIndex 2025 (Bündnis Entwicklung Hilft), the INFORM Risk Index, Republic Act No. 10121 (Philippine DRRM Act of 2010), the UN Sendai Framework for DRR 2015–2030, and SDG Target 11.5. Full citations are available in the project report.......
