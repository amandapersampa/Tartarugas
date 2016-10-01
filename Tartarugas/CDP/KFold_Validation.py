from CGD.GerenciaDados import base_recover,base_update
from CGD.Gest_KFold import saveKfold,upload_Kfold
from sklearn import cross_validation, svm
from time import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from CIH.Screen import plot_confusion_matrix, saveImages_erro
import numpy as np
import copy
import os
# Reads the lista from the base, and organizes it in a random state #
def listas_random():
    turtle = base_recover()
    rand = list(range(0, len(turtle)))
    shuffle(rand)
    t = []
    for i in rand:
        t.append(turtle[i])
    rgb = []
    ycbcr= []
    target = []
    for i in t:
        rgb.append(i.rgb)
        ycbcr.append(i.ycbcr)
        target.append(i.race[0:7])
    return rgb, ycbcr, target

# Reads the lista from the base, and organizes it in the way it was read #
def listas_normal():
    turtle = base_recover()
    rgb = []
    ycbcr = []
    target = []
    file = []
    for t in turtle:
        rgb.append(t.rgb)
        ycbcr.append(t.ycbcr)
        target.append(t.race[0:7])
        file.append(t.file)
    return rgb, ycbcr, target,file

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
def KFold_lista(lista, target,file, test_indices, train_indices):
    X_test = []
    y_test = []
    X_train = []
    y_train = []
    X_test_file = []
    for i in test_indices:
        X_test.append(lista[i])
        y_test.append(target[i])
        X_test_file.append(file[i])
    for i in train_indices:
        X_train.append(lista[i])
        y_train.append(target[i])
    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test), np.array(X_test_file)

# Calculates the KFold Values and Confusion Matrix
def KFold_result(clf, lista, target, file, it, test_indices, train_indices):
    X_train, X_test, y_train, y_test, X_test_file = KFold_lista(lista, target, file, test_indices, train_indices)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    set_labels = set(target)
    labels = list(set_labels)
    labels.sort()
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    r = accuracy_score(y_test, y_pred)

    return r, cm, labels, y_pred,y_test,X_test_file

def error_list(y_test, y_pred, X_test_file):
    error = []
    name_test = []
    name_pred = []
    size = len(y_test)
    for count in range(size):
        if (y_pred[count] != y_test[count]):
            error.append(X_test_file[count])
            name_test.append(y_test[count])
            name_pred.append(y_pred[count])
    return error, name_test, name_pred

# Calculates the total Kfold Values Saving the error Images
def kfold(clf, lista, target, file, clf_name, mode_name, random = True, save = False, kfold_update=False):
    if (kfold_update):
        kfold = cross_validation.KFold(len(target), n_folds=10, shuffle=random)
        saveKfold(kfold)
    else:
        kfold = upload_Kfold()
    result = []
    cm = [[]]
    it = 0
    error_file=[]
    name_test = []
    name_pred = []
    for train_indices, test_indices in kfold:
        r, m, labels, y_pred,y_test,X_test_file = (KFold_result(clf, lista, target, file, 'IT'+str(it), test_indices, train_indices))
        cm = sum_matrix(m, cm)
        result.append(r)
        error_aux, name_test_aux, name_pred_aux = error_list(y_test, y_pred, X_test_file)
        error_file.extend(error_aux)
        name_test.extend(name_test_aux)
        name_pred.extend(name_pred_aux)

    if (save):
        saveImages_erro(error_file, name_test,name_pred, clf_name, mode_name)


    scores = np.array(result)
    print((u"Score mínimo: {0:.2f} percent \nScore máximo: {1:.2f} percent \nScore médio: {2:.2f} percent").format(
        scores.min()*100, scores.max()*100, scores.mean()*100))
    plot_confusion_matrix(labels, cm, title=clf_name+" "+mode_name)
    return error_file, name_test, name_pred

# KNN accuracy Value
def accuracy_KNN(lista, target, file, nome,save = False, random = True, kfold_update=False):
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='ball_tree')
    print("KNN with " + nome + " :")
    return kfold(clf, lista, target, file, 'KNN', nome, random, save=save, kfold_update=kfold_update)

# SVM accuracy Value
def accuracy_SVM(lista, target, file, nome, save=False, random = True, kfold_update=False):
    clf = svm.SVC(probability=True, class_weight='balanced', C=1, kernel='linear')
    print("SVM with " + nome + " :")
    return kfold(clf, lista, target, file, 'SVM', nome, random, save = save, kfold_update=kfold_update)

# The total accurancy of the KFold test
def accuracyK_Fold(mode, clf, save=False, random = True, rotation_inv=True, b_update = False, kfold_update=False):
    if(b_update):
        t0 = time()
        base_update(rotation_inv)
        print("Base Update in: done in %0.3fs" % (time() - t0))
    rgb, ycbcr, target, file = listas_normal()
    if(clf == 'SVM'):
        if(mode == 'RGB'):
            return accuracy_SVM(rgb, target, file, mode, save=save, random = random,kfold_update=kfold_update)
        else:
            return accuracy_SVM(ycbcr, target, file, mode, save=save, random=random,kfold_update=kfold_update)
    else:
        if (mode == 'RGB'):
            return accuracy_KNN(rgb, target, file, mode, save=save, random = random,kfold_update=kfold_update)
        else:
            return accuracy_KNN(ycbcr, target, file, mode, save=save, random=random,kfold_update=kfold_update)

# The accurancy of the Kfold test using SVM and RGB
def accuracyK_Fold_SVM_RGB(save=False, random = True, rotation_inv=True, b_update = False, kfold_update=False):
    return accuracyK_Fold('RGB', 'SVM', save=save, random = random,
                          rotation_inv=rotation_inv, b_update = b_update, kfold_update=kfold_update)

# The accurancy of the Kfold test using SVM and YCbCr
def accuracyK_Fold_SVM_YCBCR(save=False, random = True, rotation_inv=True, b_update = False, kfold_update=False):
    return accuracyK_Fold('YCBCR', 'SVM', save=save, random = random,
                          rotation_inv=rotation_inv, b_update = b_update, kfold_update=kfold_update)

# The accurancy of the Kfold test using KNN and RGB
def accuracyK_Fold_KNN_RGB(save=False, random = True, rotation_inv=True, b_update = False, kfold_update=False):
    return accuracyK_Fold('RGB', 'KNN', save=save, random = random,
                          rotation_inv=rotation_inv, b_update = b_update, kfold_update=kfold_update)

# The accurancy of the Kfold test using KNN and YCbCr
def accuracyK_Fold_KNN_YCBCR(save=False, random = True, rotation_inv=True, b_update = False, kfold_update=False):
    return accuracyK_Fold('YCBCR', 'KNN', save=save, random = random,
                          rotation_inv=rotation_inv, b_update = b_update, kfold_update=kfold_update)

# Cleans the results in the folders
def clean_Result():
    for root, dir, files in os.walk('C:\\Users\\amand\\Documents\\Tartarugas\\CGT'):
        for f in files:
            if f.endswith('.png'):
                os.remove(os.path.join(root,f))
#base_update()