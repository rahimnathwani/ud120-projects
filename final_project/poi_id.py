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

### Task 2: Remove outliers
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
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# Attempt 1
# 	Accuracy: 0.87360	Precision: 0.57602	Recall: 0.19700	F1: 0.29359	F2: 0.22685
#	Total predictions: 15000	True positives:  394	False positives:  290	False negatives: 1606	True negatives: 12710
# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier(5)

# Attempt 2
#	Accuracy: 0.86960	Precision: 0.54089	Recall: 0.14550	F1: 0.22931	F2: 0.17041
#	Total predictions: 15000	True positives:  291	False positives:  247	False negatives: 1709	True negatives: 12753
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(4)

# Attempt 3
# Precision or recall may be undefined due to a lack of true positive predicitons.
# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier(6)

# Attempt 4
# Precision or recall may be undefined due to a lack of true positive predicitons.
# from sklearn import svm
# clf = svm.SVC()

# Attempt 5
from sklearn import svm
clf = svm.SVC(kernel='rbf')

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
