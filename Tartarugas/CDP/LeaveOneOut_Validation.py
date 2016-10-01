from CGD.GerenciaDados import base_recover,base_update
from sklearn import cross_validation, svm
from time import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from CIH.Screen import plot_confusion_matrix
import numpy as np
import copy

# Reads the list from the base, and organizes it in a random state #
def lists_random():
    turtle = base_recover()
    rand = list(range(0, len(turtle)))
    shuffle(rand)
    rgb = []
    ycbcr = []
    target = []
    for i in rand:
        rgb.append(turtle[i].rgb)
        ycbcr.append(turtle[i].ycbcr)
        target.append(turtle[i].race[0:7])
    return rgb, ycbcr, target

# Reads the list from the base, and organizes it in the way it was read #
def lists_normal():
    turtle = base_recover()
    rgb = []
    ycbcr = []
    target = []
    for t in turtle:
        rgb.append(t.rgb)
        ycbcr.append(t.ycbcr)
        target.append(t.race[0:7])
    return rgb, ycbcr, target

# Sum the the matrix's [i][j]
def sum_matrix(m1, m2):
    matrix = []
    l1 = len(m1)
    l2 = len(m2)
    c1 = len(m1[0])
    c2 = len(m2[0])
    fim = 0
    if(c2 == 0):
        return m1
    if(c1 == 0):
        return m2
    if(l1>l2):
        matrix = copy.copy(m1)
        for i in range(l2):
            for j in range(l2):
                sum = matrix[i][j] + m2[i][j]
                matrix[i][j] = sum
    if(l2>l1):
        matrix = copy.copy(m2)
        for i in range(l1):
            for j in range(l1):
                sum = matrix[i][j] + m1[i][j]
                matrix[i][j] = sum
    if(l1==l2):
        for i in range(l1):
            matrix.append([])
            for j in range(l1):
                sum = m1[i][j] + m2[i][j]
                matrix[i].append(sum)
    return matrix


# takes the _indices values and create the X and y matrix
def LOO_list(list, target, test_indices, train_indices):
    X_test = []
    y_test = []
    X_train = []
    y_train = []
    for i in test_indices:
        X_test.append(list[i])
        y_test.append(target[i])
    for i in train_indices:
        X_train.append(list[i])
        y_train.append(target[i])
    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

# Calculates the LOO Values and Confusion Matrix
def LOO_result(clf, list, target, test_indices, train_indices):
    X_train, X_test, y_train, y_test = LOO_list(list, target, test_indices, train_indices)
    #t0 = time()
    clf.fit(X_train, y_train)
    #print("Train: done in %0.3fs" % (time() - t0))
    #t0 = time()
    y_pred = clf.predict(X_test)
    #print("Pred: done in %0.3fs" % (time() - t0))
    #print("Classification Report")
    #print(classification_report(y_test, y_pred, target_names=target))
    #print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    r = accuracy_score(y_test, y_pred)
    #print("Expected Value: ")
    #print(y_test)
    #print("Predicted Value: ")
    #print(y_pred)
    #print("     Accuracy: %0.6f percent (+/- %0.2f percent)" % (r.mean() * 100, r.std() * 2 * 100))
    #print("Score: %0.2f percent" %(r*100))
    return r, cm

# Calculates the total LOO Values
def LOO(clf, list, target, nome):
    LOO = cross_validation.LeaveOneOut(len(target))
    result = []
    round = 1
    cm = [[]]
    for train_indices, test_indices in LOO:
        #print("Round: %d" %round)
        r, m = (LOO_result(clf, list, target, test_indices, train_indices))
        cm = sum_matrix(m, cm)
        result.append(r)
        round += 1
    scores = np.array(result)

    print((u"Score mínimo: {0:.2f} percent \nScore máximo: {1:.2f} percent \nScore médio: {2:.2f} percent").format(
        scores.min()*100, scores.max()*100, scores.mean()*100))
    set_target = set(target)
    plot_confusion_matrix(set_target, m, title=nome)

# KNN accuracy Value
def accuracy_KNN(list, target, nome):
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='ball_tree')
    print("KNN with " + nome + " :")
    nome = "LOO - KNN with " + nome
    LOO(clf, list, target, nome)

# SVM accuracy Value
def accuracy_SVM(list, target, nome):
    clf = svm.SVC(probability=True, class_weight='balanced', C=1, kernel='linear')
    print("SVM with " + nome + " :")
    nome = "LOO - SVM with " + nome
    LOO(clf, list, target, nome)


def accuracy_LOO():
    # base_update()
    # print("Base Update in: done in %0.3fs" % (time() - t0))
    rgb, ycbcr, target = lists_random()
    accuracy_SVM(rgb, target, "rgb")
    accuracy_KNN(rgb, target, "rgb")
    accuracy_SVM(ycbcr, target, "ycbcr")
    accuracy_KNN(ycbcr, target, "ycbcr")


