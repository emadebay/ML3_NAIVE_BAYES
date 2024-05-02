STATISTICAL MACHINE LEARNINING HOMEWORK TAUGHT AT PURDUE UNIVERSITY INDIANAPOLIS FALL 2023

The requirements of this homework is in the homeowrk_3.pdf and the written solution is in the solution.pdf

The code for the naive bayes implemenation is in the naives_bayes.py

Naive Bayes Classifier
This repository contains Python code to implement a Naive Bayes classifier. The classifier is trained using Maximum Likelihood Estimation (MLE) to estimate the conditional probabilities of feature values given the class labels, as well as the prior probability distribution of the class labels.

Problem Description
The goal is to build a Naive Bayes classifier using the provided training data, where the features are represented by the attributes W1 and W2, and the class labels are 'Y' and 'N'.

File Description
naive_bayes.py: Python script containing the implementation of the Naive Bayes classifier.
README.md: Readme file providing an overview of the project and instructions for running the code.
Usage
Run the naive_bayes.py script to perform the following tasks:
Read the training data.
Estimate the conditional probabilities of feature values given the class labels.
Estimate the prior probability distribution of the class labels.
Classify new data samples using the trained model.
Code Implementation
The script includes functions to read the data, estimate conditional probabilities, estimate the prior probability distribution, and classify new samples.
The classifier follows the Naive Bayes principle, where the class prediction is made by maximizing the product of the conditional probabilities of feature values given the class labels and the prior probability distribution of the class labels.
Results
The results of estimating the conditional probabilities for feature values 'blue' and 'cat' for attributes W1 and W2, respectively, are provided in the output of the script.
The estimated log-probability of the prior probability distribution P(y = Y) is also displayed in the output.
Requirements
Python 3.x
How to Run
Clone the repository: git clone <repository-url>
Navigate to the repository directory: cd naive-bayes-classifier
Run the Python script: python naive_bayes.py
