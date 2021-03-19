import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap


CSV_FILE = './cv-valid-train.csv'

# Read CSV file from Sonneta and convert categoricals to integers
all_data = pd.read_csv(CSV_FILE, index_col=0)
all_data = pd.get_dummies(all_data, drop_first=True, columns=['age', 'gender'])

# Filter
all_data = all_data[all_data.Status == 'Normal']

# Move features to X and labels to y
label_names = ['Age', 'Group_Young', 'Gender_Male', 'Status']
X = all_data.drop(label_names, axis=1)
y = all_data[label_names]
del all_data

# Drop unnecessary features and labels
y.drop(['Age', 'Status'], axis=1, inplace=True)
X.drop(['SFR_dB', 'SD_SFR_dB'], axis=1, inplace=True)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

group_label = ['Older', 'Young']
gender_label = [' Female', ' Male']
labels = []
for i in range(2):
    for j in range(2):
        labels.append(group_label[j] + gender_label[i])

def evaluate(model, y_actual, y_predicted):
    class_predicted = y_predicted[:, 0] + 2*y_predicted[:, 1]
    class_actual = y_actual[:, 0] + 2*y_actual[:, 1]
    
    report = classification_report(class_actual, class_predicted, target_names=labels)
    print(report)
    
    print('Confusion Matrix')
    cm = confusion_matrix(class_actual, class_predicted)
    cm_norm = 1 / np.sum(cm, axis=1, keepdims=True)
    print(np.round(100 * cm * cm_norm) / 100)

from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

model = RandomForestClassifier(class_weight='balanced')

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("parameters = {0}".format(results['params'][candidate]))
            print("")

# Specify parameters and distributions to sample from
param_dist = {"max_depth": np.append(np.arange(3, X.shape[1] + 1), None),
              "max_features": sp_randint(1, X.shape[1]),
              "min_samples_split": sp_randint(2, 6),
              "min_samples_leaf": sp_randint(1, 5),
              "n_estimators": sp_randint(5, 50),
              "criterion": ["gini", "entropy"]}

# Run randomized search
n_iter_search = 1000
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search, cv=5)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

TRIALS = 10
parameters = {'criterion': 'entropy', 'max_depth': 6, 'max_features': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 39}

y_jack_test = None
for X_test, y_test in zip(X.iterrows(), y.iterrows()):
    X_train = X.drop(X_test[0], axis=0)
    y_train = y.drop(y_test[0], axis=0)
    X_test = pd.DataFrame([X_test[1]], [X_test[0]])
    y_test = pd.DataFrame([y_test[1]], [y_test[0]])
    
    # Run several trials for each holdout set to average over the randomness in the model training
    for k in range(TRIALS):
        model = RandomForestClassifier(**parameters)
        model.set_params(class_weight='balanced')
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)

        if y_jack_test is None:
            y_jack_test = y_test.values
            y_jack_pred = y_predicted
        else:
            y_jack_test = np.concatenate((y_jack_test, y_test.values), axis=0)
            y_jack_pred = np.concatenate((y_jack_pred, y_predicted), axis=0)

print('Trials:', TRIALS, 'Parameters:', model.get_params())

evaluate(model, y_jack_test, y_jack_pred)


# Train a model using all the data
model = RandomForestClassifier(**parameters)
model.set_params(class_weight='balanced')
model.fit(X, y)

# Plot SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, class_names=['Age Group', 'Gender'], plot_type='bar')


print('Impact [Old - Young]')
shap.summary_plot(shap_values[0], X, plot_type='layered_violin')


print('Impact [Female - Male]')
shap.summary_plot(shap_values[1], X, plot_type='layered_violin')