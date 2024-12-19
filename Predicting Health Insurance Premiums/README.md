# Predicting Health Insurance Premiums with Machine Learning
## Problem Statement:
- In the health insurance industry, accurately pricing premiums is essential for balancing profitability and customer retention. Traditional methods rely heavily on broad actuarial tables and historical averages, which often fail to capture individual health profiles and risk factors. This can lead to either underestimated premiums, causing financial risks for insurers, or overestimated premiums, deterring potential customers.
  
- This project leverages machine learning to predict health insurance premiums based on individual characteristics, including age, medical history, and lifestyle factors. By modeling individual risk profiles, insurers can determine fair and precise premiums that reflect each person’s specific health risks. This approach aims to improve pricing accuracy, enhance competitiveness, and boost customer satisfaction, fostering a more personalized, data-driven approach to health insurance.
  
## Dataset Overview:
The dataset used for this project consists of the following attributes:
- Age: Age in years (18-66).
- Diabetes: Binary indicator (0 or 1) for diabetes.
- BloodPressureProblems: Binary indicator (0 or 1) for blood pressure issues.
- AnyTransplants: Binary indicator (0 or 1) for transplants.
- AnyChronicDiseases: Binary indicator (0 or 1) for chronic diseases.
- Height: Height in centimeters.
- Weight: Weight in kilograms.
- KnownAllergies: Binary indicator (0 or 1) for known allergies.
- HistoryOfCancerInFamily: Binary indicator (0 or 1) for family history of cancer.
- NumberOfMajorSurgeries: Count of major surgeries (0-3).
- PremiumPrice: Premium price in currency units (target variable).
  
## Approach
### Exploratory Data Analysis (EDA) and Feature Profiling:
- Histogram and Pair Plot Analysis: Visualized the distribution of each attribute and correlations with the target variable, PremiumPrice.
**Key Insights:**
- Certain factors, like age, surgeries, and chronic diseases, showed moderate correlations with premium costs.
- Other features, such as family history of cancer and allergies, displayed weak or negligible influence on premium prices.
  
### Data Preprocessing:
- Handling Missing Values: Checked and imputed missing values where necessary.
- Feature Transformation: Applied standardization on numerical features (e.g., Age, Height, Weight) to optimize model performance.

### Hypothesis Testing:
- Conducted hypothesis testing to validate the impact of various health conditions on PremiumPrice.
- Person Having Diabetes should have higher premium price in comparision to someone who is not having Diabetes.
- Person Having BloodPressureProblems should have higher premium price in comparision to someone who is not having BloodPressureProblems.
- Person Have gone through any kind of transplant should have higher premium price in comparision to someone who has not.
- Person Having AnyChronicDiseases should have higher premium price in comparision to someone who is not having AnyChronicDiseases.
- Person Having any KnownAllergies do not have affect on PremiumPrice.
- Person Having any HistoryOfCancerInFamily should have higher premium price in comparision to someone who is not having any History.
- Person Have gone through NumberOfMajorSurgeries should have higher premium price in comparision to someone who has not gone through any major surgeries.
- Person having high Age should have higher premium price.
- Person Having hight BMI should have higher premium price.

## Model Selection and Training:
Evaluated four machine learning models:
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### Model Evaluation:
- Compared model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
- Random Forest Regressor had the best performance, followed closely by Gradient Boosting.

### Comparison of Feature Importances for Various Models:
![new_image](https://github.com/SachinChauhan0911/Predicting-Health-Insurance-Premiums-with-Machine-Learning/blob/main/images/Feature%20importances.png)

## Results
### Hyperparameter Tuning Results of Random Forest Regressor:
The optimal parameters selected by GridSearchCV are:
- max_depth: 10
- max_leaf_nodes: 20
- min_samples_leaf: 2
- min_samples_split: 10
- n_estimators: 50
- These parameters were selected to minimize the mean squared error, indicating they provided the best balance between model complexity and predictive power.
### Performance on Training Data:
- R²: 0.8963
- Adjusted R²: 0.8951
- MSE: 3,997,109
- RMSE: 1999.27
- The high R² and Adjusted R² values suggest that the model captures the variance in the training data well, while the RMSE provides an error estimate in the same units as the target variable.
### Performance on Testing Data:
- R²: 0.9305
- Adjusted R²: 0.9272
- MSE: 2,454,897
- RMSE: 1,566.81
- The model generalizes well to the test data, with a higher R² and a lower RMSE compared to the training data. This suggests that the model isn’t overfitting and effectively captures important relationships in the data.
## Final Takeaways
- Model Accuracy: The tuned RandomForestRegressor model performs strongly on both training and testing datasets, with high R² values and relatively low errors, making it a good fit for predicting insurance premiums.
- While the model performs well, experimenting with additional models or advanced ensemble techniques (like Gradient Boosting or XGBoost) might yield further improvements.
- Checked with Boosting models — their performance is not as good as Random Forest regressor model.
- Therefore, adjusted_r2 score we get on the test data is around 93% (from bagging method).

## Conclusion
- This machine learning solution enables a more personalized and accurate approach to predicting health insurance premiums, accounting for individual health characteristics rather than generalized actuarial assumptions. By implementing this model, insurers can enhance pricing accuracy, improve customer satisfaction, and manage risk more effectively.
  
## Future Work
- To further improve the model, future steps may include:
- Feature Engineering: Incorporate additional health metrics or lifestyle factors to refine predictions.
- Model Optimization: Experiment with other ensemble methods like XGBoost or Neural Networks.
- Real-World Testing: Validate the model's performance with live data to ensure robustness and accuracy in practical scenarios.
