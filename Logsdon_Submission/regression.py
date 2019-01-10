'''
This program will use machine learning regression techniques to predict the income of
an OkCupid user based on that user's education level, drinking habits, and drug use
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

#load .csv containing data into a pandas dataframe
df = pd.read_csv('profiles.csv')

'''
The first part of this program will use Multible Linear Regression to predict the income
of an OkCupid user based on that user's education level, drinking habits, and drug use
'''

#assign numerical values to the drinks column
drinks_mapping = {
    'not at all': 0,
    'rarely': 1,
    'socially': 2,
    'often': 3,
    'very often': 4,
    'desperately': 5
}

df['drinks_code'] = df.drinks.map(drinks_mapping)

#assign numerical values to the drugs column
drugs_mapping = {
    'never': 0,
    'sometimes': 1,
    'often': 2
}

df['drugs_code'] = df.drugs.map(drugs_mapping)

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

#create a dataframe wich contains only the columns that will be used for the regression
model_data = graduated_with_income[['income', 'education_code', 'drinks_code', 'drugs_code']]

#remove any rows which do not have values
model_data.dropna(inplace=True)

#examine the correlations between the various features and the dependent variable
#education has the strongest correlation with income followed by drinking and then drug use
print(model_data.corr())

#plot each feature vs income to observe any linear trends
f1, ax1 = plt.subplots(figsize=(8, 5))
plt.scatter(model_data['education_code'], model_data['income'], alpha=0.1)
plt.title('Education Level vs Income', fontsize=16, fontweight='bold')
plt.xlabel('Education Level', fontsize=14, fontweight='bold')
plt.ylabel('Income', fontsize=14, fontweight='bold')
ax1.set_xticks(range(5))
ax1.set_xticklabels(['High School', 'Two-Year College', 'College', 'Masters', 'PhD/Law/Med' ])
plt.savefig('education_level_vs_income.png')

f2, ax2 = plt.subplots(figsize=(8, 5))
plt.scatter(model_data['drinks_code'], model_data['income'], alpha=0.1)
plt.title('Drinking Habits vs Income', fontsize=16, fontweight='bold')
plt.xlabel('Drinking Habits', fontsize=14, fontweight='bold')
plt.ylabel('Income', fontsize=14, fontweight='bold')
ax2.set_xticks(range(6))
ax2.set_xticklabels(['Never', 'Rarely', 'Socially', 'Often', 'Very Often', 'Desperately'])
plt.savefig('drinking_habits_vs_income.png')

f3, ax3 = plt.subplots(figsize=(8, 5))
plt.scatter(model_data['drugs_code'], model_data['income'], alpha=0.1)
plt.title('Drug Use vs Income', fontsize=16, fontweight='bold')
plt.xlabel('Drug Use', fontsize=14, fontweight='bold')
plt.ylabel('Income', fontsize=14, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['Never', 'Sometimes', 'Often'])
plt.savefig('drug_use_vs_income.png')

#separate the features from the dependent variable
features = model_data[['education_code', 'drinks_code', 'drugs_code']]

income = model_data['income']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=1)

#create the linear regression model and fit the data
model = LinearRegression()
model.fit(X_train, y_train)

#print the r^2 value to evaluate the model's performance
#these r^2 values are very low indicating that this is not a good linear regression model
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

#input the x test set values data into the model to generate predicted y values
y_predicted = model.predict(X_test)

#calculate the percent error for each predicted y value for the linear regression model
error_LR = []
y_test_values = y_test.values

for i in range(len(y_predicted)):
    error_LR.append(
        ((y_predicted[i] - y_test_values[i]) / y_test_values[i])*100
    )

#plot a histogram of the percent error values to view the distribution
f4, ax4 = plt.subplots(figsize=(8, 5))
plt.hist(error_LR, bins=100)
plt.title('Error Distribution of Predicted Income after MLR', fontsize=16, fontweight='bold')
plt.xlabel('Percent Error', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.savefig('error_distribution_MLR.png')

#print the mean and median of the percent error values to evaluate the accuracy of the model
#both the mean and median errors are very high indicating that this regression model is not very accurate
print(np.mean(error_LR))
print(np.median(error_LR))

#plot the predicted y values vs the actual y values to compare them
#in a perfect model, all values would be on the line y=x
#the majority of the values in this plot do not lie on the line y=x indicating once again that this model is not very accurate
f5, ax5 = plt.subplots(figsize=(8, 5))
plt.scatter(y_test, y_predicted, alpha=0.1)
plt.title('Predicted Income vs Actual Income after MLR', fontsize=16, fontweight='bold')
plt.xlabel('Actual Income', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Income', fontsize=14, fontweight='bold')
plt.savefig('predicted_v_actual_MLR.png')

'''
The second part of this program will use K-Nearest Neighbors Regression to predict the income
of an OkCupid user based on that user's education level, drinking habits, and drug use
'''

#normalize the features for use in the K-Nearest Neighbor Regressor
scaler = MinMaxScaler()
x = features.values
x_scaled = scaler.fit_transform(x)
scaled_features = pd.DataFrame(x_scaled, columns=features.columns)

#split the scaled data into training and testing sets
X_train_KNR, X_test_KNR, y_train_KNR, y_test_KNR = train_test_split(scaled_features, income, test_size=0.2, random_state=1)

#create a function that will find the best value of k for a given dataset
def best_k(X_train, X_test, y_train, y_test):
    k_values = list(range(1, 201))
    median_error = []

    for k in k_values:
        #create the K-Nearest Neighbor regressor and fit the data
        regressor = KNeighborsRegressor(n_neighbors=k)
        regressor.fit(X_train, y_train)

        #input the x test set values data into the model to generate predicted y values
        y_predicted = regressor.predict(X_test)

        #calculate the percent error for each predicted y value for the K-Nearest Neighbor Regressor
        error = []
        y_test_values = y_test.values

        for i in range(len(y_predicted)):
            error.append(
                ((y_predicted[i] - y_test_values[i]) / y_test_values[i])*100
            )

        #store the median of the percent error values to evaluate the accuracy of the model
        median_error.append(np.median(error))

    #find the absolute values of the errors
    median_error_abs = []

    for error in median_error:
        median_error_abs.append(abs(error))

    #return the value of k that results in an median error cloesest to zero
    return k_values[median_error_abs.index(min(median_error_abs))]

k = best_k(X_train_KNR, X_test_KNR, y_train_KNR, y_test_KNR)

#create the K-Nearest Neighbor regressor with the best k and fit the data
regressor = KNeighborsRegressor(n_neighbors=k)
regressor.fit(X_train_KNR, y_train_KNR)

#input the x test set values data into the model to generate predicted y values
y_predicted_KNR = regressor.predict(X_test_KNR)

#calculate the percent error for each predicted y value for the K-Nearest Neighbor Regressor
error_KNR = []
y_test_KNR_values = y_test_KNR.values

for i in range(len(y_predicted_KNR)):
    error_KNR.append(
        ((y_predicted_KNR[i] - y_test_KNR_values[i]) / y_test_KNR_values[i])*100
    )

#plot a histogram of the percent error values to view the distribution
#the histogram reveals that the median error is a better estimate of the error than the mean error
f6, ax6 = plt.subplots(figsize=(8, 5))
plt.hist(error_KNR, bins=100)
plt.title('Error Distribution of Predicted Income after KNR', fontsize=16, fontweight='bold')
plt.xlabel('Percent Error', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.savefig('error_distribution_KNR.png')

#print the mean and median of the percent error values to evaluate the accuracy of the model
#the K-Nearest Neighbor Regressor is much more accurate than the linear regression model
print(np.mean(error_KNR))
print(np.median(error_KNR))

#plot the predicted y values vs the actual y values to compare them
#in a perfect model, all values would be on the line y=x
#many outliers can be seen in the plot which lead to the high mean error value
f7, ax7 = plt.subplots(figsize=(8, 5))
plt.scatter(y_test_KNR, y_predicted_KNR, alpha=0.1)
plt.title('Predicted Income vs Actual Income after KNR', fontsize=16, fontweight='bold')
plt.xlabel('Actual Income', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Income', fontsize=14, fontweight='bold')
plt.savefig('predicted_v_actual_KNR')