# Model Card

For more details on model cards, see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a logistic regression classifier trained to predict whether an individual's income is `<=50K` or `>50K` based on the UCI Census Adult dataset. The model uses categorical and numeric demographic features. It was trained in Python using scikit-learn.

## Intended Use
The model is intended for educational purposes only. It demonstrates how to train, evaluate, and deploy a machine learning model following MLOps practices. It is not designed for real-world applications or impactful financial/employment decisions.

## Training Data
The model was trained using a cleaned version of the `census.csv` dataset from the UCI Census Adult dataset. It includes demographic and work-related features such as:

- Age  
- Workclass  
- Education  
- Occupation  
- Relationship  
- Race  
- Sex  
- Hours per week  
- Native country  

80% of the dataset was used for training.

## Evaluation Data
The remaining 20% of the data was held out as a test set. The same preprocessing steps were applied to the test dataset as to the training dataset.

## Metrics
The model was evaluated using **precision**, **recall**, and **F1 score**, focusing on the `>50K` class as the positive class.

### Performance on Test Data

| Metric     | Score  |
|------------|--------|
| Precision  | 0.7262 |
| Recall     | 0.6142 |
| F1 Score   | 0.6655 |

These results show that the model performs reasonably well, although there is a trade-off between precision and recall.

## Data Slices
Performance was evaluated across slices of key categorical features (e.g., education, sex, race). The results are provided in `slice_output.txt`. Different demographic slices showed variation in performance, which may reflect bias in the data.

## Ethical Considerations
- The dataset contains sensitive demographic attributes such as gender and race, which can introduce bias.
- Using this model in real-world scenarios (e.g., hiring, credit decisions) could lead to unfair or discriminatory outcomes.
- This model should not be used in any decision-making systems that affect individuals' lives.

## Caveats and Recommendations
- The Census Adult dataset is older and may not represent current populations.
- Logistic regression is a simple baseline model; more advanced models could perform better but require careful fairness evaluation.
- Additional hyperparameter tuning and feature engineering could improve performance.
- This project is for learning purposes only and is not production-ready.
