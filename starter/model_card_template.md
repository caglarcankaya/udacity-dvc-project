# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
## Model Details
The trained model is a Random Forest Classifier without any hyperparameter tuning from Sklearn library.
## Intended Use
Model is intended to use in salary prediction using features such as sex, age, capital-gain, capital-loss, hours per week, native-country, education, relationship, race etc. 
## Training and Evaluation Data
All the data from census dataset is split into 2 pieces using train test split ratio 0.2 and random seed value 42
## Metrics
Precision recall and F1 score is used as performance metrics.
Results on test data is as following:
- Precision on test data is 0.7335766423357665
- Recall on test data is 0.638906547997457
- F1 score on test data is 0.6829765545361876
## Ethical Considerations
According to slices test, model has biased predictions listed cases:
- precision = 1.00, recall = 0.50, f1 = 0.67 on category native-country and feature Greece
- precision = 1.00, recall = 0.68, f1 = 0.81 on category marital-status and feature Widowed

## Caveats and Recommendations
The training did with a data where there are missing fields marked as ?. Some tests may be done to remove these rows to increase performance. 
Furthermore, with hyperparameter tuning a better model can be developed. 
Lastly, to make a better model considering ethical decisions, a model with less features can be better for general predictions and not contain discriminative patterns.