# ML-Olympiad---Autism-Prediction-Challenge

Description
Welcome to ML Olympiad hosted by TFUG Chennai and TFUG Mysuru .

What is ML Olympiad?
An associated Kaggle Community Competitions hosted by ML GDEs or TFUGs, sponsored by Google Developers.

Source

Abstract
Improve Autism Screening by creating predicting the likelihood of having this condition.

About this dataset
What is Autism
Autism, or autism spectrum disorder (ASD), refers to a broad range of conditions characterized by challenges with social skills, repetitive behaviors, speech and nonverbal communication.

Causes and Challenges
It is mostly influenced by a combination of genetic and environmental factors. Because autism is a spectrum disorder, each person with autism has a distinct set of strengths and challenges. The ways in which people with autism learn, think and problem-solve can range from highly skilled to severely challenged.
Research has made clear that high quality early intervention can improve learning, communication and social skills, as well as underlying brain development. Yet the diagnostic process can take several years.

The Role of Machine Learning
This dataset is composed of survey results for more than 700 people who filled an app form. There are labels portraying whether the person received a diagnosis of autism, allowing machine learning models to predict the likelihood of having autism, therefore allowing healthcare professionals prioritize their resources.
 Sure, I can help you create a README.md file for your project based on the provided content. Here's a template for a README.md file that includes the content you've shared:

```markdown
# Autism Prediction using Machine Learning

## Introduction

Autism is a neurological disorder that affects a person’s ability to interact with others, make eye contact, learn, and exhibit various behavioral and social capabilities. This project explores the application of machine learning to predict whether a person has Autism or not.

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Importing Libraries

In this project, we utilize several Python libraries to handle data, perform analysis, and build machine learning models. The key libraries include:
- Pandas
- Numpy
- Matplotlib/Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn

## Importing Dataset

The first step in the project is to import the dataset into a pandas DataFrame and explore its characteristics. The dataset used in this project contains information about individuals, including whether they have Autism or not.

## Exploratory Data Analysis (EDA)

EDA is a crucial step to understand the dataset and its characteristics. In this project, we perform EDA to:
- Visualize data imbalances
- Explore the distribution of numerical and categorical features
- Analyze correlations between variables

## Feature Engineering

Feature engineering involves creating new features from existing data to improve model performance and gain deeper insights. In this project, we:
- Group ages into categories
- Sum up clinical scores
- Apply log transformations to remove skewness
- Encode categorical labels

## Model Development and Evaluation

We build and evaluate several machine learning models to predict Autism. The models considered in this project include:
- Logistic Regression
- XGBoost
- Support Vector Classifier (SVC)

The models are trained on the preprocessed data and evaluated based on training and validation accuracy. We also address the issue of data imbalance using oversampling techniques.

## Conclusion

This project demonstrates the application of machine learning in predicting Autism, despite the lack of traditional diagnostic methods. The models achieve an accuracy of around 80% to 85%, showcasing the potential of machine learning in solving real-world problems.

For more details, please refer to the project code and Jupyter Notebook.

Feel free to explore the code, dataset, and analysis to gain a better understanding of the work presented here.

```
# Autism Prediction using Machine Learning

## Introduction

Autism is a neurological disorder that affects a person’s ability to interact with others, make eye contact, learn, and exhibit various behavioral and social capabilities. This project explores the application of machine learning to predict whether a person has Autism or not.

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Importing Libraries

In this project, we utilize several Python libraries to handle data, perform analysis, and build machine learning models. The key libraries include:
- Pandas
- Numpy
- Matplotlib/Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn

## Importing Dataset

The first step in the project is to import the dataset into a pandas DataFrame and explore its characteristics. The dataset used in this project contains information about individuals, including whether they have Autism or not.

## Exploratory Data Analysis (EDA)

EDA is a crucial step to understand the dataset and its characteristics. In this project, we perform EDA to:
- Visualize data imbalances
- Explore the distribution of numerical and categorical features
- Analyze correlations between variables

## Feature Engineering

Feature engineering involves creating new features from existing data to improve model performance and gain deeper insights. In this project, we:
- Group ages into categories
- Sum up clinical scores
- Apply log transformations to remove skewness
- Encode categorical labels

## Model Development and Evaluation

We build and evaluate several machine learning models to predict Autism. The models considered in this project include:
- Logistic Regression
- XGBoost
- Support Vector Classifier (SVC)

The models are trained on the preprocessed data and evaluated based on training and validation accuracy. We also address the issue of data imbalance using oversampling techniques.

## Conclusion

This project demonstrates the application of machine learning in predicting Autism, despite the lack of traditional diagnostic methods. The models achieve an accuracy of around 80% to 85%, showcasing the potential of machine learning in solving real-world problems.

For more details, please refer to the project code and Jupyter Notebook.

Feel free to explore the code, dataset, and analysis to gain a better understanding of the work presented here.

LogisticRegression() : 
Training Accuracy :  0.845766129032258
Validation Accuracy :  0.8348022135683542

XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=None, ...) : 
Training Accuracy :  1.0
Validation Accuracy :  0.836441893830703

SVC() : 
Training Accuracy :  0.9203629032258065
Validation Accuracy :  0.8398237343717975

Model Evaluation
From the above accuracies, we can say that Logistic Regression and SVC() classifier perform better on the validation data with less difference between the validation and training data. Let’s plot the confusion matrix as well for the validation data using the Logistic Regression model.

metrics.plot_confusion_matrix(models[0], X_val, Y_val)
plt.show()
 

 
