#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = ['poi'] + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print "Original data set has {0} items".format(len(data_dict))

### Task 2: Remove outliers
obvious_outlier_keys = ['TOTAL', # Spreadsheet quirk
                        'THE TRAVEL AGENCY IN THE PARK', # Doesn't sound like a person's name 
                       ]
[data_dict.pop(key) for key in obvious_outlier_keys]
print "Removing {0} obvious outliers".format(len(obvious_outlier_keys))
import numpy as np
features_to_check = ['salary', 'bonus'] # check most obvious features for outliers
data_to_check = featureFormat(data_dict, features_to_check, remove_all_zeroes=False)
# Plot salary vs. bonus
from matplotlib import pyplot as plt
for point in data_to_check:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
plt.xlabel("salary")
plt.ylabel("bonus")
# Commenting out the plot to prevent the program stalling if run unattended
# plt.show()

# The plot showed some points with very high salaries (>1MM) and/or bonuses (>6MM).  Let's see who they are.
def text_to_num(txt):
    return 0 if txt == 'NaN' else int(txt)
additional_outlier_keys = [i for i in data_dict if text_to_num(data_dict[i]['salary']) > 1000000 or text_to_num(data_dict[i]['bonus']) > 6000000]
# print additional_outlier_keys

# Let's remove those folks
big_folks = []
# big_folks = ['LAVORATO JOHN J', 'LAY KENNETH L', 'SKILLING JEFFREY K', 'FREVERT MARK A']
[data_dict.pop(key) for key in big_folks]
print "Removing {0} folks with very high salaries/bonuses".format(len(big_folks))

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Attempt 0
from sklearn.naive_bayes import GaussianNB
clf0 = GaussianNB()

# Attempt 1
from sklearn import neighbors
clf1 = neighbors.KNeighborsClassifier(5)

# Attempt 2
from sklearn import neighbors
clf2 = neighbors.KNeighborsClassifier(4)

# Attempt 3
clf3 = neighbors.KNeighborsClassifier(6)

# Attempt 4
from sklearn import svm
clf4 = svm.SVC()

# Attempt 5
from sklearn import svm
clf5 = svm.SVC(kernel='rbf', C=0.1)

# Attempt 6
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('pca', PCA()), ('svm', SVC(kernel='rbf'))]
clf6 = Pipeline(estimators)

# Attempt 7
from sklearn.pipeline import Pipeline
from sklearn import neighbors
from sklearn.decomposition import PCA
estimators = [('pca', PCA()), ('kneighbours', neighbors.KNeighborsClassifier(3))]
clf7 = Pipeline(estimators)

# Make a custom scorer
from sklearn.metrics import make_scorer
def my_custom_score_function(predictions, labels_test):
    false_positives, true_positives, false_negatives, true_negatives = 0.0, 0.0, 0.0, 0.0
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
    try:
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
    except:
        return 0.0
    return min(precision, recall)

my_custom_scorer = make_scorer(my_custom_score_function, greater_is_better=True)


# Attempt 8
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
parameters = {'pca__n_components': [5, ], 'svm__kernel':['rbf', ], 'svm__C':[0.1, 1, 10], 'svm__gamma':[10**i for i in range(-2, 3)]}
estimators = [('pca', PCA()), ('svm', SVC(kernel='rbf'))]
clf8 = GridSearchCV(Pipeline(estimators), parameters, verbose=5, n_jobs=5)


"""
# Attempt 9
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
scaler = preprocessing.MinMaxScaler()
parameters = {'pca__n_components': range(3,15), 'svm__kernel':['rbf', ], 'svm__C':[100*(1.05**i) for i in range(-10, 11)], 'svm__gamma':[1.05**i for i in range(-10, 11)]}
estimators = [('scaler', scaler), ('pca', PCA(n_components=5)), ('svm', SVC(kernel='rbf'))]
cv = StratifiedShuffleSplit(
    labels,
    n_iter=10,
    random_state=42)
gs = GridSearchCV(Pipeline(estimators), parameters, cv=cv, verbose=5, scoring='f1', n_jobs=5)
gs.fit(features, labels)
clf9 = gs.best_estimator_
"""

"""
# Attempt 10 (without PCA)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
scaler = preprocessing.MinMaxScaler()
parameters = {'svm__kernel':['rbf', ], 'svm__C':[100*(1.05**i) for i in range(-10, 11)], 'svm__gamma':[1.05**i for i in range(-10, 11)]}
estimators = [('scaler', scaler), ('svm', SVC(kernel='rbf'))]
cv = StratifiedShuffleSplit(
    labels,
    n_iter=10,
    test_size=0.3,
    random_state=42)
gs = GridSearchCV(Pipeline(estimators),
                  parameters,
                  cv=cv,
                  verbose=5,
                  scoring=my_custom_scorer,
                  n_jobs=5)
gs.fit(features, labels)
clf10 = gs.best_estimator_
"""

"""
# Attempt 11 (same as #10, but with PCA)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
scaler = preprocessing.MinMaxScaler()
parameters = {'pca__n_components': range(10,20), 'svm__kernel':['rbf', ], 'svm__C':[100*(1.05**i) for i in range(-10, 11)], 'svm__gamma':[1.05**i for i in range(-10, 11)]}
estimators = [('scaler', scaler), ('pca', PCA()), ('svm', SVC(kernel='rbf'))]
cv = StratifiedShuffleSplit(
    labels,
    n_iter=10,
    test_size=0.3,
    random_state=42)
gs = GridSearchCV(Pipeline(estimators),
                  parameters,
                  cv=cv,
                  verbose=5,
                  scoring=my_custom_scorer,
                  n_jobs=-1)
gs.fit(features, labels)
clf11 = gs.best_estimator_
"""

# Attempt 12 (same as #11, but with SelectKBest, and smaller range for pca__n_components)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
scaler = preprocessing.MinMaxScaler()
parameters = {'selectkbest__k': range(11,17), 'pca__n_components': range(6,11), 'svm__kernel':['rbf', 'poly'], 'svm__C':[(3**i) for i in range(-10, 11)], 'svm__gamma':[(3**i) for i in range(-10, 11)]}
estimators = [('selectkbest', SelectKBest()), ('scaler', scaler), ('pca', PCA()), ('svm', SVC(kernel='rbf'))]
cv = StratifiedShuffleSplit(
    labels,
    n_iter=10,
    test_size=0.3,
    random_state=42)
gs = GridSearchCV(Pipeline(estimators),
                  parameters,
                  cv=cv,
                  verbose=5,
                  scoring=my_custom_scorer,
                  n_jobs=-1)
gs.fit(features, labels)
clf12 = gs.best_estimator_



# Pick which attempt to use
clf=clf12

print len(features)
print len(labels)

params = clf.get_params()
for p in params:
    print '{0}: {1}'.format(p, params[p])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
