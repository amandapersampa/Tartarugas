import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import skimage.io as io
import PIL
from time import time

def plot_confusion_matrix(target, cm, title='Confusion matrix', cmap=plt.cm.Blues):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    width = len(cm)
    for x in range(width):
        for y in range(width):
            ax.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(target))
    plt.xticks(tick_marks, target, rotation=45)
    plt.yticks(tick_marks, target)
    plt.ylabel(u'Classe Verdadeira', fontsize=16)
    plt.xlabel(u'Classe Estimada', fontsize=16)

    plt.tight_layout()

    title = title +'.png'
    plt.savefig(title)
    plt.close(fig)


def saveImages_erro(error_file, name_test, name_pred, clf_name, mode_name):
    iterate = 0
    for i in range(len(error_file)):
        fig = plt.figure()
        plt.imshow(error_file[i])
        folder = 'pred-' + name_pred[i] + '-esp-' + name_test[i]
        base = 'C:\\Users\\amand\\Documents\\Tartarugas\\CGT\\Result\\' + clf_name + '\\' + mode_name + '\\' + folder
        plt.title('Especie: ' + name_test[i] + ' - Previsto: ' + name_pred[i])
        file = 'Imagem-original'
        title = base + '\\' + str(i) + file + '.png'
        plt.savefig(title)
        plt.close(fig)
        print("File: " + title + " Salvo")
        iterate+=1
