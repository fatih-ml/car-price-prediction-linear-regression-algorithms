# Car Price Prediction: Advanced Feature Engineering Techniques for Linear Regression Algorithms

The automotive industry's dynamic landscape poses unique challenges in predicting car prices accurately. In this project, we delve into the intricacies of car price prediction using advanced feature engineering techniques coupled with various linear regression algorithms. The dataset, sourced from a prominent online car trading company in 2019, comprises information on nine distinct car models. Although relatively clean, the dataset demands meticulous preprocessing and feature engineering to enhance its predictive capabilities. The ensuing stages involve model building, thorough evaluation, hyperparameter tuning and model selection.

## Project Overview:

The primary objective of this undertaking is to explore and apply diverse machine learning algorithms -specifically linear models- providing a comprehensive implementation with detailed explanations where necessary. The central focus is on minimizing errors in predicting car prices. Our approach encompasses the application of an array of linear regression techniques and regularization methods, fostering an in-depth comparison to determine the most effective model.

## Data Exploration and Feature Engineering:

Initiating with the importation of dependencies and a preliminary data examination, we progress to exploratory data analysis (EDA) and feature engineering. Categorically and numerically distinct, the variables undergo meticulous scrutiny, including detailed visualizations and statistical analyses. Special attention is given to problematic categorical variables where observations contain multiple stacked values in a single cell. This challenge is addressed by processing the columns, assigning weighted grades to each car specification, and consolidating them into an overall grade. This transformation made a significant contribution to our model. Also some other features transformed in a more meaningful representation with feature engineering, again improving scores importantly.

## Handling Outliers and Transformation:

Correlation analysis reveals that engine power holds the highest correlation with price (0.7). Outliers, a common challenge, are addressed by grouping cars based on make-models and further regrouping them by kilometer ranges. Median prices are then calculated to identify and remove relative outliers. The target feature's right skewness is mitigated through log transformation.

## Linear Regression and Model Comparison:

The project meticulously applies various linear regression and regularization techniques, comparing their respective scores. Key feature engineering steps, such as log transformation, grading stacked specifications, and outlier removal, significantly enhance predictive accuracy. The comparison underscores the effectiveness of Lasso Regression, with its interpretability and emphasis on fewer features.

## Key Insights:

The analysis unveils the pivotal factors influencing car prices, with the make_model (brand) standing out as the most crucial determinant. Age, engine power, and kilometers follow as significant contributors. While the focus remains on linear regression in this notebook, the door is open for exploration into ensemble techniques for potentially improved scores.

## Future Considerations:

Opportunities for further improvement include exploring ensemble techniques such as Gradient Boosting Regressor or Random Forest Regressor etc. Additionally, there is room for nuanced feature engineering, particularly in refining stacked values with domain knowledge. Further regrouping cars based on nuanced criteria and addressing potential inconsistencies in data representation offer avenues for enhancement. Moreover, you can have a look to residuals part where we already prepared a depth analysis dataframe to dive into details to increase model performance

## Feature Selection
After deciding the best model and best performing variables, altough not neccessarily done each time, we decided to run the best model with an icreasing number of selected features from 1 to all possible features to make a comparison of model performance with different number of features. thanks to speed of linear regression algorithms and not too big dataset, we had a chance to showcase how feature selection works with a trade-off. Eventually we continued with only 6 features to model deployment, while we decreased features from 22 to 6 the model score only increased from 0.104278 (RMSE) to 0.108222 (%3.8) and R2 score from 0.93 to 0.925 (%0.5).

## Connect with Me:

For questions, collaborations, or further discussions, feel free to reach out on [Linkedin](https://www.linkedin.com/in/fatih-calik-469961237/), [Github](https://github.com/fatih-ml) or [Kaggle](https://www.kaggle.com/fatihkgg)

__This notebook serves as a comprehensive guide to linear regression modeling, laying the groundwork for future explorations into more advanced algorithms and nuanced feature engineering strategies.__