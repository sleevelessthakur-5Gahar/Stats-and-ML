📘 Practical Exercise (2) – Data Preprocessing
Objective:
In this exercise, you will apply essential data preprocessing techniques to prepare a real-world dataset for machine learning. The focus is on data quality, not model training.



Step 1: Handling Missing Values
Identify columns that contain missing values.

Choose an appropriate strategy for each case:

Numerical features: mean or median imputation

Categorical features: mode imputation

Explain briefly why you chose each method.

📌 Hint: Use isnull(), sum(), and simple imputation methods.

Step 2: Noise Detection and Handling
Select one numerical feature.

Add artificial noise (small random variations) to this feature.

Apply a simple noise-handling technique, such as:

Moving average

Smoothing by aggregation

Compare the feature before and after noise handling.

📌 Goal: Understand the difference between raw and cleaned signals.

Step 3: Outlier Detection and Handling
Detect outliers in at least one numerical feature using:

Z-score method or

Visualization (boxplot)

Handle outliers by:

Removing them or

Transforming the feature

Justify your decision in 1–2 sentences.

Step 4: Data Transformation
Apply at least one transformation:

Encode a categorical feature (e.g., one-hot encoding), or

Create a new feature from existing ones (simple feature engineering), or

Apply binning to a numerical feature.

Explain what problem this transformation solves.

Step 5: Feature Scaling
Select 2–3 numerical features.

Apply:

Standardization (Z-score)

Normalization (Min–Max scaling)

Compare the scaled values and explain:

When Z-score is preferred

When Min–Max scaling is preferred

Deliverables
Clean and well-documented notebook

Short explanations (1–2 sentences) for each step

Clear separation between:

Raw data

Processed data

