import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import argparse
from sklearn.utils.multiclass import unique_labels

STANCES = ['AGAINST', 'FAVOR', 'NONE']

def plot_confusion_matrix(y_true, y_pred, figname, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('{}.eps'.format(figname), format='eps', dpi=100)
    return ax

def compute_accuracy(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    return np.equal(y_pred, y_test).sum() / len(y_test)

def RandomForest(X_train, X_test, y_train, y_test, figname="confusion_matrix"):
    
    # GRID SEARCH
    parameters = {'n_estimators':range(100,600,100), 'max_depth':range(1,20,5)}
    rlf = RandomForestClassifier(random_state=0)
    rlf = GridSearchCV(rlf, parameters, cv=5)
    rlf.fit(X_train, y_train)
    
    y_pred = rlf.predict(X_test)

    rf_cm = confusion_matrix(y_test, y_pred)
    acc = compute_accuracy(y_test, y_pred)
    
    plot_confusion_matrix(y_test, y_pred, figname, classes=STANCES,
                      title='Confusion matrix, without normalization')

    rf_cv = cross_val_score(rlf, X_train, y_train, cv=5, scoring='f1_macro')
    report = {}
    report["train_accuracy"] = compute_accuracy(y_train, rlf.predict(X_train))
    report["test_accuracy"] = acc
    report["RF_cross_val_score"] = rf_cv.tolist()
    report["RF_mean_acc"] = rf_cv.mean()
    report["RF_std_acc"] = rf_cv.std()*2
    report["RF_best_estimator"] = rlf.best_estimator_
    
    report["RF_CM"] = rf_cm.tolist()
    
    f1_macro = f1_score(y_test, y_pred, average='macro') 
    report["RF_F1_SCORE"] = f1_macro

    
    return y_pred, report

def SVMClassifier(X_train, X_test, y_train, y_test, figname="confusion_matrix"):

    parameters = {'kernel':['linear','rbf','poly'], 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    clf = svm.SVC(gamma='auto')
    clf = GridSearchCV(clf, parameters, cv=5)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    clf_cm = confusion_matrix(y_test, y_pred)
    acc = compute_accuracy(y_test, y_pred)
    
    plot_confusion_matrix(y_test, y_pred, figname, classes=STANCES,
                      title='Confusion matrix, without normalization')
    
    clf_cv = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
    report = {}
    report["train_accuracy"] = compute_accuracy(y_train, clf.predict(X_train))
    report["test_ccuracy"] = acc
    report["SVM_cross_val_score"] = clf_cv.tolist()
    report["SVM_mean_acc"] = clf_cv.mean()
    report["SVM_std_acc"] = clf_cv.std()*2
    report["SVM_best_estimator"] = clf.best_estimator_

    report["SVM_CM"] = clf_cm.tolist()
    
    f1_macro = f1_score(y_test, y_pred, average='macro') 
    report["SVM_F1_SCORE"] = f1_macro

    
    return y_pred, report
    