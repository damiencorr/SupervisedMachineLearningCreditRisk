# Cryptocurrencies
### CHALLENGE

The goals of this challenge are to implement machine learning models, use resampling to attempt to address class imbalances and then evaluate the performance of machine learning models to assess credit risk for a sample data set. For details of the How To of this challenge please refer to the Appendix below.

### Machine learning models & Resampling approaches where relevant:
- sklearn.linear_model's LogisticRegression with solver='lbfgs' (the default solver in sklearn LogisticRegression), Oversample using RandomOverSampler and SMOTE algorithms
- LogisticRegression as above, this time with Undersample using the cluster centroids algorithm
- Again LinearRegression, using a combination over- and undersampling approach with SMOTEENN algorithm
- Balanced Random Forest Classifier
- Easy Ensemble AdaBoost Classifier

### How to analyze model performance:
How do I measure and compare the performance of multiple classification machine learning models ?

We use several scoring mechanisms to assess performance of classification models, and we can we use the scores to compare models for their respective strenghts and weaknesses.
- Confusion Matrix - depicts model predictions against actual observations, get a sense of the number of predicted versus actual trues, and the number of predicted vs actual falses
- Precision - (How many selected items are relevant) the ratio of True Positives (TP) vs True Positives + False Positives (FP) (predeicted true but actually false)
- Recall - (How many relevant items are selected) the ratio of TP vs TP + False Negative (predicted as false but actually true)
- F1 - harmonic mean between precision and recall, score between 0 - 1, better is higher
- Balanced Accuracy - for imbalanced data sets (like this example where the number of high risk candidates is very small compared to low risk) this score normalizes true positive and true negative predictions

This analysis used the following to produce model performance scores for comparison:
- sklearn.metrics's confusion_matrix
- sklearn.metrics's balanced_accuracy_score
- imblearn.metrics's classification_report_imbalanced, providing (among others) precision (labelled pre), recall (labelled rec) and F1 (labelled f1)

### Summary and analysis of the models’ performance:
-
-
-
-
-
-
-
-
-
-
-
-
- I will provide summary tomorrow.











**NOTE:**
The following is I think an easy to understand qualitative statement on how to consider the performance of one model, rather than attempting to compare multiple models tackling the same challenge.

How do I measure the performance of my model?

A good fitting model is one where the difference between the actual or observed values and predicted values for the selected model is small and unbiased for train, validation and test data sets. (https://medium.com/datadriveninvestor/how-to-evaluate-the-performance-of-a-machine-learning-model-45063a7a38a7)

In this challenge we employed only training and validation data, testing data was not employed, as the first step to chossing a model to test is to find a mdeol that looks strongest out of the typically available options.



# APPENDIX - Module 17 Challenge - APPENDIX

Jill commends you for all your hard work. Piece by piece, you have been building up your skills in data preparation, statistical reasoning, and machine learning.

You are now ready to apply machine learning to solve real-world challenges.
In this challenge, you’ll build and evaluate several machine learning models to assess credit risk, using data from LendingClub; a peer-to-peer lending services company.

**Background** 
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, you’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Your final task is to evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

**Objectives**
The goals of this challenge are for you to:

- Implement machine learning models.
- Use resampling to attempt to address class imbalance.
- Evaluate the performance of machine learning models.

**Instructions**

You’ll use the **imbalanced-learn** library to resample the data and build and evaluate logistic regression classifiers using the resampled data. Download the files you’ll need, which include starter code and the dataset:

Download Module -17-Challenge-Resources.zip

You will:

- Oversample the data using the RandomOverSampler and SMOTE algorithms.
- Undersample the data using the cluster centroids algorithm.
- Use a combination approach with the SMOTEENN algorithm.


For each of the above, you’ll:

- Train a logistic regression classifier (from Scikit-learn) using the resampled data.
- Calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics.
- Generate a confusion_matrix.
- Print the classification report (classification_report_imbalanced from imblearn.metrics).
- Lastly, you’ll write a brief summary and analysis of the models’ performance. Describe the precision and recall scores, as well as the balanced accuracy score. Additionally, include a final recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

### Extension
For the extension, you’ll train and compare two different ensemble classifiers to predict loan risk and evaluate each model. Note that you’ll use the following modules, which you have not used before. They are very similar to ones you’ve seen: 
- BalancedRandomForestClassifier and EasyEnsembleClassifier, both from imblearn.ensemble. 
These modules combine resampling and model training into a single step. Consult the following documentation for more details:

- Section 5.1.2. Forest of randomized trees (Links to an external site.)
- imblearn.ensemble.EasyEnsembleClassifier (Links to an external site.)

Use 100 estimators for both classifiers, and complete the following steps for each model:

- Train the model and generate predictions.
- Calculate the balanced accuracy score.
- Generate a confusion matrix.
- Print the classification report (classification_report_imbalanced from imblearn.metrics).
- For the BalancedRandomForestClassifier, print the feature importance, sorted in descending order (from most to least important feature), along with the feature score.
- Lastly, you’ll write a brief summary and analysis of the models’ performance. Describe the precision and recall scores, as well as the balanced accuracy score. Additionally, include a final recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

