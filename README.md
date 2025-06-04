# Credit Risk Modeling with Logistic Regression, Random Forest and WOE Transformation

This project demonstrates the development of a credit risk scorecard using logistic regression with Weight of Evidence (WOE) transformation and Information Value (IV) feature selection. The goal is to predict the likelihood of default for consumer loans and provide insights for risk-based pricing and strategy.

## Dataset

The project uses a dataset from LendingClub, containing information on consumer loans, including loan characteristics, borrower attributes, and loan performance. The dataset has been preprocessed to handle missing values, remove duplicates, and engineer meaningful features.

## Methodology

1. **Data Exploration and Preprocessing:**
   - Explored the dataset to understand its structure and characteristics.
   - Handled missing values and removed duplicate records.
   - Defined the modeling target by mapping loan status to binary labels (good/bad).
   - Removed features with high missingness or potential data leakage.

2. **Feature Engineering:**
   - Created new features like "loan_burden" by combining existing variables.
   - Converted timestamp features into meaningful numerical representations.
   - Handled high-cardinality categorical variables through binning or encoding.

3. **Feature Binning and WOE Transformation:**
   - Applied decision tree-based optimal binning to numerical features.
   - Calculated Weight of Evidence (WOE) and Information Value (IV) for each feature bin.
   - Selected features based on their predictive power using IV scores.

4. **Logistic Regression Modeling:**
   - Trained a logistic regression model using WOE-transformed features.
   - Evaluated model performance using metrics like AUC, KS statistic, and Gini coefficient.
   - Interpreted model coefficients and their relationship with credit risk.

5. **Advanced Machine Learning Models**
 - Trained Gradient Boost, XGBoost and Random Forest models.
 - Performed hyper-parameter tuning to figure out best models.
 - Comapred model performance to the results from logistic regression.
 
6. **Scorecard Development:**
   - Scaled model coefficients to create a credit scorecard.
   - Generated scores for each feature bin based on their WOE values and model weights.
   - Visualized the score distribution for good and bad borrowers.

7. **Business Impact Analysis:**
   - Quantified the expected loss reduction and financial benefits of the model.
   - Segmented the portfolio based on risk scores for targeted pricing and strategy.
   - Provided insights on how the model can be used for risk-based decision-making.

8. **Implementation and Monitoring Framework:**
   - Outlined a pre-deployment checklist and model documentation summary.
   - Defined a monitoring framework to track model performance and data stability.
   - Proposed a phased deployment strategy for seamless integration into production.

## Results

The logistic regression model achieved an AUC of 0.702, a KS statistic of 0.346, and a Gini coefficient of 0.404 on the holdout test set. The model's performance demonstrates its ability to effectively discriminate between good and bad borrowers. The scorecard provides a transparent and interpretable way to assess credit risk and make data-driven decisions.

The business impact analysis shows that the model has the potential to reduce portfolio losses by 15-20% through improved risk segmentation and pricing. The project also presents a comprehensive monitoring framework and deployment strategy to ensure the model's long-term stability and performance.

## Future Enhancements

- Experiment with alternative modeling techniques (e.g., random forest, gradient boosting) and compare their performance with logistic regression.
- Conduct a more in-depth analysis of the model's fairness and bias across different demographic groups to ensure regulatory compliance.
- Develop a prototype of the monitoring dashboard to showcase the ability to translate model insights into actionable business tools.

## Requirements

- Python 3.x
- Jupyter Notebook
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Usage

1. Clone the repository.
2. Install the required libraries.
3. Download the dataset from Kaggle - https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
4. Run the Jupyter Notebook `credit_risk_modeling.ipynb` to reproduce the analysis and results.

Feel free to explore the code, experiment with different techniques, and adapt the methodology to your specific use case.
