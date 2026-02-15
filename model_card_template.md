# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
## Model Details

This model is a supervised binary classification model. It's used to predict whether an individual's income exceeds $50,000 per year based on demographic and employment-relation data. The model used scikit-learn and was trained on the U.S. Census Income dataset. Categorical features were encoded using OneHotEncoder and the target variable was turned into a binary variable using LabelBinarizer. The model was trained and deployed as a part of a scalable machine learning pipeline using FastAPI. The final trained model used is a Random Forest classifier.

## Intended Use

This model is meant to display the deployment of a scalable machine learning pipeline. The model predicts whether an individuals income will exceed 50K annually based on features in the dataset. This project is for educational purposes only and shouldn't be used for real-world decision making. It demonstrates classification workflows, API development, model monitoring, and slice-based performance evaluation.

## Training Data

The model was trained using U.S. Census Income data. The dataset contains demogrpahic and employment information, including age, workclass, education, marital status, occupation, relationship, race, sex, native country, hours per week, and capitcal gain and loss. These variables contain a mix of both numerical and categorical values. Categorical variables were one-hot encoded before training the model. The label variable is "salary", which indicated whether income is over 50K or is equal/below 50K. The data was split into training and testing sets with an 80/20 division.

## Evaluation Data

The evaluation dataset used 20% of the original dataset, which was separated out using a train-test split. Both sets used the same encoder and label binarizer to fit the data. Performance was also evaluated on categorical feature slices to determine whether model performance varies across demographic subgroups.

## Metrics

The model was evaluated on it's precision, recall and F1 score.The test dataset model achieved a precision of 0.7346, a recall of 0.6261, and a F1 score of 0.6760. Additionally, slice-based evaluation was used across all categorical features. Performance varied across different demographic groups, which indicates some potential disparities in predictive performance depending on subgroup features.

## Ethical Considerations

This model used demographic attributes such as race, sex, and native country, which are sensitive and personal features. Including these in the model can allow for societal biases. Additionally, income can tie into historical inequalities, which can reflect across some demographic groups. This model should not be used for real-world decision making without bias analysis and ethical review. Wihtout fairness testing or mitigation strategies, the model might produce unfair outcomes across demographic groups.

## Caveats and Recommendations

The model is limited by many factors such as historical bias, class imbalance in income distribution, use of train-test split over cross-validation, and a lack of fairness mitigation techniques. Possible future improvements would include applying cross-validation, testing additional models, implementing fairness mitigation techniques, and removing sensitive features. This model is suitable for demonstrating ML deployment workflows but not for real-world decision making.