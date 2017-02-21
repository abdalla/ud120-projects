#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.svm import SVC

clf = SVC(C=10000, kernel="rbf")

print "kernel:", clf.kernel
print "C:", clf.C

t0 = time()
#features_train = features_train[:len(features_train) / 100]  # keeping only 1% of the data
#labels_train = labels_train[:len(labels_train) / 100]   # keeping only 1% of the data
clf.fit( features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

print "featuresTreain: ", len(features_train)
print "labelsTrain: ", len(labels_train)

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time() - t1, 3), "s"

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "Accuracy found: {:.4f}".format(acc)

"""
	What class does your SVM (0 or 1, corresponding to Sara and Chris respectively)
	predict for element 10 of the test set? The 26th? The 50th?
	(Use the RBF kernel, C=10000, and 1% of the training set. Normally you'd get
	the best results using the full training set, but we found that using 1% sped up
	the computation considerably and did not change our results--so feel free to use
	that shortcut here.)
"""
print "10th: %r, 26th: %r, 50th: %r" % (pred[10], pred[26], pred[50])

# There are over 1700 test events, how many are predicted to be in the
# "Chris" (1) class?
print "No. of predicted to be in the 'Chris'(1): %r" % sum(pred)


#########################################################


