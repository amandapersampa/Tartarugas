from CGD.Extracao_caracteristicas import lbp_rotation_invariant, histograma
from scipy.misc import imsave
import matplotlib.pyplot as plt

def save_lbp_RGB(file_erro, name_test, name_pred, clf, mode):
    for count in range(len(file_erro)):
        folder = 'pred-'+name_pred[count]+'-esp-'+name_test[count]
        base = 'C:\\Users\\amand\\Documents\\Tartarugas\\CGT\\Result\\' + clf + '\\'+ mode+'\\'+folder
        imsave(base + '\\' + str(count) + 'imagem-erro' + '.png', file_erro[count])
        save(file_erro[count], 'imagem-erro',0, clf, mode,count,name_test, name_pred, 'R')
        save(file_erro[count], 'imagem-erro', 1, clf, mode, count,name_test, name_pred, 'G')
        save(file_erro[count], 'imagem-erro', 2, clf, mode, count,name_test, name_pred, 'B')

def save_lbp_YCBCR(file_erro, name_test, name_pred, clf, mode):
    for count in range(len(file_erro)):
        folder = 'pred-' + name_pred[count] + '-esp-' + name_test[count]
        base = 'C:\\Users\\amand\\Documents\\Tartarugas\\CGT\\Result\\' + clf + '\\' + mode + '\\' + folder
        file = 'imagem-erro'
        imsave(base + '\\' + str(count) + file + '.png', file_erro[count])
        save(file_erro[count], file,0, clf, mode,count,name_test, name_pred, 'Y')
        save(file_erro[count], file, 1, clf, mode, count,name_test, name_pred, 'Cb')
        save(file_erro[count], file, 2, clf, mode, count,name_test, name_pred, 'Cr')

def save_hist(img, name_test, name_pred, count, path, c):
    hist = histograma(img)
    fig = plt.figure()
    plt.hist(hist, bins=10,normed=True)
    plt.title('Histograma - Especie: ' + name_test[count] + ' - Previsto: ' + name_pred[count])
    plt.tight_layout()

    name = 'pred-' + name_pred[count] + '-esp-' + name_test[count]
    plt.savefig(path + '\\' + str(count) + c + '-hist-' + name + '.png')
    print('File: ' + path + '\\' + str(count) + c+'-hist-'+ name + '.png Salvo')
    plt.close(fig)

def save(image, file, channel, clf, mode,count, name_test, name_pred, c):
    #pred-caretta-esp-cheloni
    folder = 'pred-'+name_pred[count]+'-esp-'+name_test[count]
    base = 'C:\\Users\\amand\\Documents\\Tartarugas\\CGT\\Result\\' + clf + '\\'+ mode+'\\'+folder
    img = lbp_rotation_invariant(image[:, :, channel])
    imsave(base + '\\' + str(count) + c+'-lbp.png', img)
    imsave(base + '\\' + str(count) + c+'-image.png', image[:, :, channel])

    save_hist(img, name_test, name_pred, count, base, c)

    print('File: '+ base + '\\' + str(count) + c+'-' + file + '.png Salvo')
    print('File: ' + base + '\\' + str(count) + c+'-im-'+ file + '.png Salvo')
