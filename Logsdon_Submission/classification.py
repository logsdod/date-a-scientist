'''
This program will use machine learning classification techniques to predict whether
or not an OkCupid user is male or female based on the data in his/her profile
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

#load .csv containing the profile data into a pandas dataframe
df = pd.read_csv('profiles.csv')

'''
The first part of this program will use a Naive Bayes Classifier to predict
whether or not a user is male or female based on his/her essays
'''

#define the essay columns to be combined
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

#Remove the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)

#combine the essays into one string
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

#split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(all_essays, df.sex, test_size=0.2, random_state=1)

#initialize the counter, fit it with the training data, and create the count vectors for the data
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

#initialize the classifier, fit it with the data, and predict the labels of the test data
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

#print the accuracy, recall, precision, and F1 score of the model
#female was chosen for the positive value for recall, precision, and F1 score
print(accuracy_score(test_labels, predictions))
print(recall_score(test_labels, predictions, pos_label='f', average='binary'))
print(precision_score(test_labels, predictions, pos_label='f', average='binary'))
print(f1_score(test_labels, predictions, pos_label='f', average='binary'))

#print the confusion matrix to evaluate the model's performance
print(confusion_matrix(test_labels, predictions, labels=['m', 'f']))

'''
The following section of the program will use a K-Nearest Neighbors classifier to predict
whether or not a user is male or female based on his/her education level and income
'''

#create a dataframe only containing users who have graduated from a program
graduated = df[
    (df.education == 'graduated from college/university') |
    (df.education == 'graduated from masters program') |
    (df.education == 'graduated from two-year college') |
    (df.education == 'graduated from high school') |
    (df.education == 'graduated from ph.d program') |
    (df.education == 'graduated from law school') |
    (df.education == 'graduated from med school')
]

#assign a numberical value to the education level
#ph.d, law school, and med school were gruoped together
graduated_mapping = {
    'graduated from high school': 0,
    'graduated from two-year college': 1,
    'graduated from college/university': 2,
    'graduated from masters program': 3,
    'graduated from ph.d program': 4,
    'graduated from law school': 4,
    'graduated from med school': 4
}

graduated['education_code'] = graduated.education.map(graduated_mapping)

#create a dataframe which does not contain the -1 value for income
#it is uncertain what this value represents
graduated_with_income = graduated[(graduated['income'] != -1)]

#create a dataframe wich contains only the columns that will be used for the classification
model_data = graduated_with_income[['income', 'education_code', 'sex']]

#remove any rows which do not have values
model_data.dropna(inplace=True)

#normalize the features, z-score was chosen because there are outliers in the income feature
features = model_data[['income', 'education_code']]
scaled_features = stats.zscore(features)

#split the data into training and test sets
train_data_KNN, test_data_KNN, train_labels_KNN, test_labels_KNN = train_test_split(scaled_features, model_data.sex, test_size=0.2, random_state=1)

#evaluate the model's performance as a function of k
k_values = list(range(1, 201))
accuracy = []
recall = []
precision = []
f1 = []

for k in k_values:
    #create the classifier for each k in the list, fit the data, and predict the labels of the test data
    classifier_KNN = KNeighborsClassifier(n_neighbors=k)
    classifier_KNN.fit(train_data_KNN, train_labels_KNN)
    predictions_KNN = classifier_KNN.predict(test_data_KNN)

    #store the accuracy, recall, precision, and F1 score for each k value
    accuracy.append(accuracy_score(test_labels_KNN, predictions_KNN))
    recall.append(recall_score(test_labels_KNN, predictions_KNN, pos_label='f', average='binary'))
    precision.append(precision_score(test_labels_KNN, predictions_KNN, pos_label='f', average='binary'))
    f1.append(f1_score(test_labels_KNN, predictions_KNN, pos_label='f', average='binary'))

#plot the accuracy, recall, precision, and F1 score as a function of k
f, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(8,5))

ax[0,0].plot(k_values, accuracy)
ax[0,0].set_ylabel('Accuracy', fontweight='bold')

ax[0,1].plot(k_values, recall)
ax[0,1].set_ylabel('Recall', fontweight='bold')

ax[1,0].plot(k_values, precision)
ax[1,0].set_xlabel('k')
ax[1,0].set_ylabel('Precision', fontweight='bold')

ax[1,1].plot(k_values, f1)
ax[1,1].set_xlabel('k')
ax[1,1].set_ylabel('F1 Score', fontweight='bold')

plt.subplots_adjust(wspace=0.15)

plt.suptitle('Model Performace as a function of k', fontweight='bold')

plt.savefig('model_performance.png')

#find the k value which results in the highest accuracy
k = k_values[accuracy.index(max(accuracy))]
print(k)

#use the best k to create the classifier, fit the data, and predict the labels of the test data
classifier_KNN = KNeighborsClassifier(n_neighbors=k)
classifier_KNN.fit(train_data_KNN, train_labels_KNN)
predictions_KNN = classifier_KNN.predict(test_data_KNN)

#print the accuracy, recall, precision, and F1 score of the model
#female was chosen for the positive value for recall, precision, and F1 score
print(accuracy_score(test_labels_KNN, predictions_KNN))
print(recall_score(test_labels_KNN, predictions_KNN, pos_label='f', average='binary'))
print(precision_score(test_labels_KNN, predictions_KNN, pos_label='f', average='binary'))
print(f1_score(test_labels_KNN, predictions_KNN, pos_label='f', average='binary'))

#print the confusion matrix to evaluate the model's performance
print(confusion_matrix(test_labels_KNN, predictions_KNN, labels=['m', 'f']))