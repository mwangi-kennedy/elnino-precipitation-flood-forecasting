# ENSO-Driven Precipitation Anomaly & Flood-Risk Forecasting Pipeline

## Overview

This repository contains an end-to-end, leakage-safe, and reproducible Google Colab workflow designed to forecast regional precipitation anomalies and flood-risk indicators driven by the El Niño Southern Oscillation (ENSO).

The pipeline integrates ENSO climate indicators, regional meteorological observations, causal data cleaning, domain-specific feature engineering, time-aware machine-learning validation, and flood-risk event evaluation into a single reproducible forecasting workflow.

---

## Pipeline Architecture

The pipeline executes a sequential 10-stage workflow:

| **#** | **Stage**                     | **Description**                                                                                                              |
| ----: | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
|     0 | **Setup & Config**            | Defines global seeds, geographic coordinates, forecast horizons, and output paths.                                           |
|     1 | **Data Ingestion**            | Pulls NOAA Niño 3.4 SST anomalies, Southern Oscillation Index (SOI), and NASA POWER regional data with a synthetic fallback. |
|     2 | **Causal Cleaning**           | Performs gap-filling utilizing **strictly past data** to prevent data leakage.                                               |
|     3 | **Feature Engineering**       | Generates ENSO lags ($0\text{--}6$ months), rolling statistics, causal anomalies, and flood-potential interaction features.  |
|     4 | **Exploratory Data Analysis** | Analyzes lag-correlations, seasonality, and ENSO phase composites.                                                           |
|     5 | **Model Training**            | Evaluates multiple regressors including Ridge baseline, Random Forest, and Gradient Boosting.                                |
|     6 | **Validation**                | Implements `TimeSeriesSplit` with a temporal gap and a final chronological hold-out set.                                     |
|     7 | **Evaluation**                | Computes metrics including $R^2$, RMSE, MAE, precision, and recall on extreme-rain and flood-risk events.                    |
|     8 | **Artifacts**                 | Generates and saves visualization charts, trained models, metrics, and documentation.                                        |
|     9 | **GitHub Push**               | Executes token-authenticated commits and pushes directly from the Colab environment.                                         |

---

## Configuration & Region of Interest

* **Default Region:** Central Kenya Highlands (`LAT: -0.4`, `LON: 36.9`), capturing strong teleconnections during the October–December "short rains". Coordinates can be modified for any location globally.
* **Temporal Scope:** 1981–2023 monthly telemetry data.
* **Forecast Horizon:** 1-month lead time predicting precipitation anomalies at $t + \text{HORIZON}$.
* **Maximum ENSO Lag:** 6 months.
* **Extreme Wet Event Threshold:** Top 20% of precipitation events (`EXTREME_Q = 0.80`).
* **Final Chronological Hold-out:** 20% of the available data.

### Default Configuration

```python
LAT = -0.4
LON = 36.9

START_YEAR = 1981
END_YEAR = 2023

HORIZON = 1
MAX_LAG = 6

EXTREME_Q = 0.80

TEST_SIZE = 0.20
```

The geographic coordinates can be modified to apply the pipeline to another region globally.

---

## Data Telemetry & Sources

The pipeline integrates open scientific datasets with zero required API keys.

### NOAA PSL

The pipeline retrieves monthly climate indices from NOAA Physical Sciences Laboratory (PSL), including:

* Niño 3.4 sea-surface temperature anomaly.
* Southern Oscillation Index (SOI).

These variables provide the primary ENSO-related predictors used by the forecasting models.

### NASA POWER API

The pipeline retrieves regional meteorological and land-surface information using the NASA POWER API.

The primary parameters include:

* Corrected precipitation (`PRECTOTCORR`).
* 2-meter temperature (`T2M`).
* Root-zone soil wetness (`GWETROOT`).

The daily telemetry is aggregated into monthly observations for integration with the ENSO indices.

### Fallback Mechanism

The pipeline automatically switches to a physically motivated synthetic generator if:

* The environment is offline.
* An external API request fails.
* The required telemetry cannot be retrieved.

This ensures that the complete workflow can still be executed for testing, development, and pipeline validation.

> **Important:** Synthetic data is intended as a fallback for pipeline execution and testing. It should not be interpreted as observed climate data.

---

## Causal Data Cleaning & Imputation

To maintain operational realism and eliminate look-ahead bias, data cleaning adheres to strict causal constraints.

The pipeline never uses future observations to fill missing historical values.

### Imputation Strategy

1. Short gaps ($\le 3$ months) are forward-filled using the last known values.

2. Longer gaps are imputed using the mean of the **previous 6 months only**:

```python
shift(1).rolling(6).mean()
```

3. Boolean tracking flags (`*_was_missing`) are retained for auditing.

### Why Causal Imputation Matters

Traditional interpolation methods can accidentally introduce future information into historical observations.

For example, interpolating a missing value at time $t$ using observations from both $t-1$ and $t+1$ would allow information from the future to influence the historical feature.

The pipeline therefore uses only information available before or at the prediction time.

---

## Feature Engineering & Domain Features

The feature matrix extracts **27 domain-specific variables across 432 usable monthly samples**.

The features are organized into several groups.

### Primary ENSO Indicators

The pipeline incorporates:

* Niño 3.4 lags ($0\text{--}6$).
* SOI lags.
* 3-month rolling means.
* 6-month rolling means.
* 3-month ENSO tendency.
* Binary El Niño phase flags.
* Binary La Niña phase flags.

These features allow the models to capture both current ENSO conditions and delayed teleconnection effects.

---

### Local Compounding Factors

Local environmental conditions are incorporated through:

* Detrended temperature anomalies.
* Soil wetness anomalies.
* Recent precipitation anomaly persistence lags.

These variables provide information about local environmental conditions that can amplify or reduce flood potential.

---

### Baselines & Temporal Encodings

The feature matrix also includes:

* Historical month climatological means.
* Cyclical month encodings.
* Month sine transformation.
* Month cosine transformation.

The cyclical representation is used because calendar months are periodic rather than linearly ordered.

For example:

```text
December → January
```

should be treated as adjacent seasons rather than distant points in a numerical sequence.

---

### Interaction Indices

The pipeline includes domain-informed interaction metrics such as:

* `nino3_x_soil`
* `flood_potential_idx`

The flood potential index combines ENSO conditions and soil wetness:

$$
\text{flood\_potential\_idx}
=
\text{nino34\_roll3}
\times
\text{soil\_wetness}
$$

These interactions are intended to capture situations where climate forcing and local antecedent conditions jointly increase precipitation or flood potential.

---

## Exploratory Data Analysis

The EDA stage investigates relationships between ENSO conditions, precipitation anomalies, seasonality, and flood-related indicators.

The analysis includes:

### Lag Correlations

The pipeline evaluates the relationship between precipitation anomalies and lagged ENSO conditions.

This helps identify whether ENSO signals at earlier months contain useful information for forecasting future precipitation.

### Seasonality

Monthly precipitation behavior is analyzed to identify recurring seasonal patterns.

This is particularly relevant to the Central Kenya Highlands, where the October–December period is associated with the **short rains**.

### ENSO Phase Composites

Precipitation behavior is compared across:

* El Niño periods.
* La Niña periods.
* Neutral ENSO periods.

These composites help identify differences in precipitation characteristics associated with ENSO phases.

---

## Model Training & Evaluation

The pipeline evaluates multiple regression algorithms.

### Algorithms

#### Ridge Regression

Ridge regression provides a regularized linear baseline.

It is useful for determining how much predictive power can be obtained from approximately linear relationships between ENSO variables and precipitation anomalies.

#### Random Forest Regressor

Random Forest captures nonlinear relationships and feature interactions without requiring an explicitly specified functional form.

#### Histogram Gradient Boosting Regressor

Histogram Gradient Boosting provides another nonlinear modeling approach optimized for efficient gradient-boosted tree learning.

---

## Machine-Learning Pipeline

Models are trained using scikit-learn pipelines incorporating feature preprocessing and scaling where appropriate.

The workflow uses:

* `StandardScaler`
* Ridge regression
* Random Forest regression
* Histogram Gradient Boosting regression

Using consistent pipelines helps ensure that preprocessing is applied correctly within each training fold.

---

## Validation Strategy

The pipeline uses time-aware validation rather than random train/test splitting.

### TimeSeriesSplit

Models are evaluated across **5 chronological cross-validation splits** using `TimeSeriesSplit`.

This respects temporal ordering and helps prevent future observations from being used to train models evaluated on earlier observations.

### Temporal Gap

A temporal gap is incorporated between training and validation data to further reduce the risk of temporal leakage.

### Final Hold-out Set

In addition to cross-validation, the workflow maintains a final chronological hold-out set consisting of approximately **20% of the available observations**.

This provides an additional evaluation of how the selected model performs on unseen future data.

---

## Evaluation Metrics

The pipeline evaluates both continuous precipitation predictions and extreme-event predictions.

### Regression Metrics

The primary regression metrics include:

* $R^2$
* RMSE
* MAE

### Coefficient of Determination

The coefficient of determination is represented as:

$$
R^2
$$

It measures the proportion of variance in the observed target that is explained by the model.

### Root Mean Squared Error

RMSE is calculated as:

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

### Mean Absolute Error

MAE is calculated as:

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

---

## Extreme Rainfall & Flood-Risk Evaluation

In addition to continuous precipitation forecasting, the pipeline evaluates extreme-rain and flood-risk events.

The extreme wet-event threshold is defined using the upper 20% of precipitation observations:

```python
EXTREME_Q = 0.80
```

This allows the workflow to evaluate whether the model can distinguish unusually wet periods from normal conditions.

Classification-oriented metrics include:

* Precision
* Recall

These metrics provide additional information beyond regression performance when the primary concern is identifying potentially hazardous precipitation events.

---

## Forecasting Objective

The primary forecasting objective is to predict the regional precipitation anomaly at a future time step:

$$
\hat{Y}_{t+h} = f(X_t)
$$

where:

* $Y_{t+h}$ is the precipitation anomaly at the forecast horizon.
* $h$ is the forecast horizon, set to **1 month** by default.
* $X_t$ represents information available at time $t$.
* $f$ represents the trained forecasting model.

The pipeline additionally evaluates whether predicted precipitation conditions correspond to **extreme-rainfall and elevated flood-risk events**.

---

## Repository Structure

```text
outputs/
├── data/
│   ├── raw_telemetry.csv
│   └── model_table.csv
├── figures/
│   └── [Generated EDA and validation charts]
└── models/
    └── [Saved model artifacts (.joblib)]
```

### Data

The `data/` directory contains:

```text
raw_telemetry.csv
```

The original or ingested telemetry used by the workflow.

```text
model_table.csv
```

The cleaned and feature-engineered dataset used for model training and evaluation.

### Figures

The `figures/` directory contains generated visualizations, including EDA and model-validation charts.

### Models

The `models/` directory contains serialized trained model artifacts, typically using the `.joblib` format.

---

## Expected Outputs

After execution, the pipeline produces:

* Raw telemetry datasets.
* Cleaned and feature-engineered modeling tables.
* Exploratory data analysis figures.
* ENSO phase and precipitation composites.
* Lag-correlation visualizations.
* Model validation charts.
* Regression performance metrics.
* Extreme-rain event metrics.
* Flood-risk indicators.
* Serialized trained models.
* Documentation and experiment artifacts.

These outputs are stored under the `outputs/` directory and can be committed directly to a GitHub repository from Google Colab.

---

## Artifacts

The artifact-generation stage automatically saves the main outputs required for analysis and reproducibility.

These include:

### Data Artifacts

```text
outputs/data/raw_telemetry.csv
outputs/data/model_table.csv
```

### Visualization Artifacts

```text
outputs/figures/
```

This directory contains generated EDA, validation, correlation, and forecasting charts.

### Model Artifacts

```text
outputs/models/
```

This directory contains serialized machine-learning models.

Model artifacts can be loaded later without retraining the entire pipeline.

---

## GitHub Integration

The final pipeline stage provides automated GitHub integration.

The workflow can execute token-authenticated:

1. Git configuration.
2. Repository synchronization.
3. File staging.
4. Commit creation.
5. Push operations.

This allows the generated data, figures, models, and documentation to be committed directly from the Google Colab environment.

> **Security Note:** GitHub authentication tokens should never be hard-coded into notebooks or committed to the repository. Use secure environment variables, Colab secrets, or another appropriate credential-management mechanism.

---

## Reproducibility

The complete workflow is intended to run from Google Colab with minimal manual configuration.

The pipeline uses reproducibility-oriented practices including:

* Global random seeds.
* Deterministic feature-generation procedures where possible.
* Explicit configuration parameters.
* Saved datasets.
* Saved model artifacts.
* Chronological validation.
* Documented output paths.
* Causal preprocessing.

The default configuration can be modified to support different geographical regions, forecast horizons, ENSO definitions, and model configurations.

---

## Changing the Region

The default region is the Central Kenya Highlands:

```python
LAT = -0.4
LON = 36.9
```

To apply the workflow to another region, these coordinates can be changed.

For example:

```python
LAT = <target_latitude>
LON = <target_longitude>
```

The same modeling architecture can therefore be applied to different locations globally, provided that the required telemetry is available.

---

## Key Design Principles

The pipeline is built around several principles that are particularly important for climate and hydrological forecasting.

### 1. Leakage Prevention

All preprocessing, imputation, feature generation, and validation procedures are designed to prevent information from the future from influencing predictions about the past.

### 2. Temporal Validation

Random train/test splitting is avoided.

Instead, chronological validation techniques are used to better simulate real-world forecasting conditions.

### 3. Reproducibility

Global random seeds, deterministic processing where possible, standardized outputs, and saved model artifacts make experiments easier to reproduce.

### 4. Causal Imputation

Missing observations are filled using only information that would have been available at the time of prediction.

### 5. Domain-Aware Features

The feature engineering process incorporates:

* ENSO dynamics.
* Precipitation persistence.
* Soil wetness.
* Temperature anomalies.
* Seasonality.
* ENSO/local-condition interaction effects.

This provides a climate-domain-aware feature space rather than relying exclusively on generic machine-learning variables.

### 6. Operational Fallback

When external data APIs are unavailable, the pipeline can fall back to a physically motivated synthetic data generator, allowing the complete workflow to be executed and tested offline.

---

## Why ENSO Matters

The El Niño Southern Oscillation is a major source of interannual climate variability.

ENSO conditions can influence atmospheric circulation and precipitation patterns across distant regions through large-scale atmospheric and oceanic teleconnections.

For East Africa, ENSO-related variability can be particularly relevant to seasonal rainfall patterns, including the October–December short-rains period.

However, the relationship between ENSO and local precipitation is not deterministic.

Consequently, this pipeline treats ENSO as a predictive climate signal rather than assuming that ENSO alone determines local rainfall.

---

## Flood-Risk Concept

The pipeline does not attempt to directly simulate river hydraulics, drainage networks, or inundation depth.

Instead, it generates **flood-risk indicators** based on combinations of:

* Predicted precipitation anomalies.
* ENSO conditions.
* Recent precipitation persistence.
* Soil wetness.
* Local environmental conditions.
* Interaction features.

The `flood_potential_idx` is an example of such a composite indicator.

This approach is useful for identifying periods where atmospheric forcing and antecedent land-surface conditions may jointly increase the potential for flooding.

---

## Scientific Scope

This project is intended as a **research and forecasting pipeline**, not as a replacement for official meteorological or flood early-warning systems.

ENSO is only one driver of regional precipitation variability.

Other important factors may include:

* Local topography.
* Indian Ocean variability.
* Land-surface conditions.
* Atmospheric circulation.
* Tropical cyclones.
* Monsoon dynamics.
* Other climate modes.
* Regional convective processes.

Consequently, flood-risk indicators generated by this pipeline should be interpreted as **model-derived risk signals**, rather than deterministic flood predictions.

---

## Limitations

Several limitations should be considered when interpreting the results.

### Limited Predictive Drivers

The model focuses heavily on ENSO and selected local environmental variables. Additional atmospheric and oceanic predictors could potentially improve forecasting performance.

### Point-Based Regional Data

NASA POWER point data represents conditions at a specified geographic coordinate and may not capture all spatial variability across a larger region.

### Historical Period

The default dataset covers 1981–2023. Relationships learned from this period may not remain stationary under future climate conditions.

### Synthetic Fallback

When the fallback generator is used, resulting model performance should not be interpreted as evidence of real-world predictive skill.

### Flood Risk Is Not Inundation Modeling

The flood-risk component is an indicator system rather than a physical flood inundation model.

It does not directly predict:

* Flood depth.
* Flood extent.
* River discharge.
* Drainage capacity.
* Number of affected households.
* Infrastructure damage.

---

## Potential Extensions

The pipeline can be extended in several directions.

### Additional Climate Predictors

Potential additional predictors include:

* Indian Ocean Dipole (IOD).
* Sea-surface temperature fields.
* Atmospheric pressure.
* Wind fields.
* Humidity.
* Geopotential height.
* Satellite precipitation products.

### Spatial Forecasting

Instead of forecasting precipitation at a single point, the system could be extended to:

* Gridded precipitation prediction.
* County-level forecasts.
* Watershed-level forecasting.
* Spatial flood-risk maps.

### Advanced Machine Learning

Potential models include:

* XGBoost.
* LightGBM.
* CatBoost.
* Support Vector Regression.
* Temporal convolutional networks.
* LSTM networks.
* Transformer-based time-series models.

### Probabilistic Forecasting

The current workflow focuses primarily on point predictions.

A future implementation could generate:

* Prediction intervals.
* Probabilistic rainfall forecasts.
* Exceedance probabilities.
* Calibrated flood-risk probabilities.

### Operational Early-Warning System

The pipeline could eventually be integrated with:

* Automated monthly data ingestion.
* Scheduled model retraining.
* Real-time monitoring dashboards.
* SMS/email alerts.
* Web-based flood-risk maps.
* Automated forecast reports.

---

## Example End-to-End Workflow

The complete pipeline follows the sequence:

```text
┌─────────────────────────────┐
│       Setup & Config        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Data Ingestion         │
│ NOAA + NASA POWER           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Causal Cleaning        │
│ Past-only imputation        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Feature Engineering      │
│ ENSO + Soil + Rainfall      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│          EDA                │
│ Lag + Seasonality + ENSO    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Model Training         │
│ Ridge + RF + HistGB         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        Validation           │
│ TimeSeriesSplit + Gap       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        Evaluation           │
│ R² + RMSE + MAE + Events    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│          Artifacts          │
│ Data + Figures + Models     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        GitHub Push          │
└─────────────────────────────┘
```

---

## Pipeline Stage Summary

| Stage                            | Main Function                                         | Key Output           |
| -------------------------------- | ----------------------------------------------------- | -------------------- |
| **0. Setup & Config**            | Initialize configuration and reproducibility settings | Global configuration |
| **1. Data Ingestion**            | Retrieve ENSO and regional telemetry                  | Raw telemetry        |
| **2. Causal Cleaning**           | Handle missing observations without leakage           | Clean telemetry      |
| **3. Feature Engineering**       | Generate domain-specific predictors                   | Model table          |
| **4. Exploratory Data Analysis** | Investigate relationships and seasonality             | EDA figures          |
| **5. Model Training**            | Train candidate forecasting models                    | Trained models       |
| **6. Validation**                | Perform chronological cross-validation                | Validation results   |
| **7. Evaluation**                | Measure forecasting and event-detection performance   | Metrics              |
| **8. Artifacts**                 | Save datasets, figures, and models                    | Output artifacts     |
| **9. GitHub Push**               | Version-control generated artifacts                   | Updated repository   |

---

## Technology Stack

The workflow is designed primarily for the Python ecosystem and Google Colab.

### Core Technologies

* **Python**
* **Google Colab**
* **pandas**
* **NumPy**
* **scikit-learn**
* **Matplotlib**
* **Joblib**

### Data Sources

* **NOAA Physical Sciences Laboratory**
* **NASA POWER**

### Version Control

* **Git**
* **GitHub**

---

## Reproducible Research Workflow

The intended workflow for researchers and developers is:

```text
Configure Region
      ↓
Retrieve Data
      ↓
Validate Data
      ↓
Apply Causal Cleaning
      ↓
Generate Features
      ↓
Explore Climate Relationships
      ↓
Train Candidate Models
      ↓
Perform Time-Series Cross-Validation
      ↓
Evaluate on Chronological Hold-out
      ↓
Generate Figures & Artifacts
      ↓
Save Models
      ↓
Commit to GitHub
```

This structure makes it possible to rerun the experiment after changing the region, forecast horizon, model parameters, or climate predictors.

---

## Reproducibility Checklist

Before interpreting model results, verify that:

* [ ] The correct geographic coordinates were configured.
* [ ] The intended temporal range was downloaded.
* [ ] NOAA ENSO data was successfully retrieved.
* [ ] NASA POWER data was successfully retrieved.
* [ ] The synthetic fallback was not unintentionally used.
* [ ] Missing-data flags were generated.
* [ ] Imputation used only historical information.
* [ ] Features were generated without future information.
* [ ] Chronological cross-validation was used.
* [ ] The final hold-out set remained untouched during model selection.
* [ ] Evaluation metrics were calculated on unseen observations.
* [ ] Model artifacts were saved successfully.
* [ ] Generated figures were saved.
* [ ] GitHub credentials were not exposed in source code.

---

## Interpretation of Results

Model performance should be interpreted using multiple metrics rather than relying on a single score.

For precipitation regression:

* **Higher $R^2$** generally indicates that the model explains more of the observed variance.
* **Lower RMSE** indicates smaller squared prediction errors.
* **Lower MAE** indicates smaller average absolute errors.

For extreme-event detection:

* **Higher precision** means fewer predicted extreme events are false alarms.
* **Higher recall** means the model identifies a larger proportion of actual extreme events.

In flood-risk applications, recall may be particularly important because failing to identify a genuinely hazardous event can have substantially greater consequences than issuing an unnecessary warning.

However, precision and recall should be evaluated together because optimizing exclusively for one can negatively affect the other.

---

## Intended Use

This repository can serve as a foundation for:

* Climate-data research.
* ENSO teleconnection analysis.
* Seasonal precipitation forecasting.
* Flood-risk research.
* Machine-learning experimentation.
* Climate-risk dashboards.
* Early-warning system prototypes.
* Academic research projects.
* Reproducible climate modeling workflows.

---

## Disclaimer

This project is a research and machine-learning forecasting workflow.

It should **not** be used as the sole basis for emergency flood warnings, evacuation decisions, infrastructure design, or other safety-critical decisions.

Official meteorological agencies, hydrological observations, local flood-monitoring systems, and expert assessments should be consulted when making operational decisions.

---

## Summary

The **ENSO-Driven Precipitation Anomaly & Flood-Risk Forecasting Pipeline** combines:

* ENSO climate indicators.
* NOAA Niño 3.4 SST anomalies.
* Southern Oscillation Index data.
* NASA POWER regional meteorological data.
* Causal missing-data imputation.
* ENSO lag features.
* Rolling climate statistics.
* Soil wetness and temperature anomalies.
* Precipitation persistence.
* Seasonal and cyclical temporal features.
* Climate-domain interaction features.
* Ridge regression.
* Random Forest regression.
* Histogram Gradient Boosting regression.
* Chronological `TimeSeriesSplit` validation.
* Final chronological hold-out testing.
* Regression performance metrics.
* Extreme precipitation detection.
* Flood-potential indicators.
* Reproducible model artifacts.
* Automated GitHub integration.

The resulting workflow provides a reproducible foundation for investigating how ENSO-related climate variability can be translated into regional precipitation anomaly forecasts and potential flood-risk signals.
