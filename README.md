## House Price Prediction - Project Overview
This repository contains a comprehensive end-to-end machine learning pipeline for predicting house prices. The project covers the entire data science lifecycle, from initial exploratory data analysis to advanced modeling using XGBoost.

### 🧠 The Engineering Philosophy: Time Series Integrity

The most critical technical hurdle in predictive modeling for real estate is **Temporal Leakage**. In this project, the data is split chronologically rather than randomly.

* **The Problem**: In housing, prices are influenced by seasonal trends and economic cycles. A random split would allow the model to "peek" at future prices to predict past ones, leading to over-optimistic results that fail in production.
* **The Solution**: I implemented a strict **Time Series Split** in `Data Split.ipynb`. By training on earlier months and testing on the most recent months, I prove that the model's predictive power holds true for "unseen future" data.

---

### 🛠️ Advanced Feature Engineering & Data Enrichment

Beyond simple cleaning, this repo showcases the ability to merge disparate data sources to find hidden signals:

* **Macro-Economic Integration**: I didn't just look at house specs; we merged the core dataset with `usmetros.csv` to include **Labor Force statistics, Poverty rates, and Median Rent** data. This demonstrates a "big picture" approach to feature selection.
* **Target Encoding**: To handle high-cardinality categorical data (like Cities and Zipcodes) without creating thousands of sparse columns, I utilized `category_encoders`. This keeps the model efficient and prevents the "curse of dimensionality."
* **Cyclical Temporal Features**: Dates were transformed into meaningful signals (Quarterly/Monthly) to capture the inherent seasonality of the real estate market.

---

### 📈 Multi-Stage Modeling & Evaluation

This repo follows a rigorous hierarchy:

1. **Baseline**: I established a "Naive" baseline using a median-strategy `DummyRegressor`. If a complex model can't beat the median, it’s not worth the compute.
2. **Linear Benchmarking**: I tested **Ridge, Lasso, and ElasticNet** to understand linear relationships and feature coefficients.
3. **Non-Linear Optimization**: The final **XGBoost** implementation captures complex interactions between features. I used **Feature Importance (Gain)** analysis to ensure the model was making decisions based on logical drivers (like location and square footage) rather than noise.

---

### 💎 Why  

* **Modular Codebase**: Files are separated by concern (Cleaning, Encoding, Modeling), making the project readable and easy to peer-review.
* **Business Impact Focused**: The use of **Mean Absolute Error (MAE)** ensures the results are interpretable in dollar amounts, making it easy to explain the model's accuracy to non-technical stakeholders.
* **Scalable Architecture**: The pipeline is designed so that new data can be dropped in, processed, and used for inference with minimal refactoring.

---
### 📂 Project Structure
The project is organized into several Jupyter Notebooks, each representing a specific stage of the pipeline:

1. **[EDA & Cleaning]**: Initial data exploration, handling missing values, and merging external datasets like usmetros.csv.

2. **[Data Split]**: Implementation of the data splitting strategy, ensuring the temporal integrity of the model training.

3. **[Feature Engineering and Encoding]**: Extraction of date-based features, reordering columns, and applying target encoding for categorical variables.

4. **[Baseline]**: Establishment of a baseline performance using a DummyRegressor (median strategy) to evaluate the relative success of future models.

5. **[Linear Regression]**: Implementation of standard regression models including Linear Regression, Ridge, Lasso, and ElasticNet.

6. **[XGB Model]**: Final high-performance model using XGBoost with hyperparameter tuning and feature importance analysis.
   
### 🚀 How to Run

1. **Data Prep**: Run `Data Split.ipynb` followed by `EDA & Cleaning.ipynb`.
2. **Transformation**: Run `Feature Engineering and Encoding.ipynb` to generate the final training vectors.
3. **Modeling**: Open `XGB Model.ipynb` to see the final training loop and evaluation metrics.
